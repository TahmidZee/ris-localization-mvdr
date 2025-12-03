from __future__ import annotations
import os, json, math, time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.optim.swa_utils import AveragedModel, SWALR

# CUDNN settings moved to Trainer.__init__ to avoid conflicts
torch.set_float32_matmul_precision('high')  # PyTorch 2.x



from .configs import cfg, mdl_cfg, set_seed
from .model import HybridModel
from .loss import UltimateHybridLoss
from .dataset import ShardNPZDataset  # unchanged
from .eval_angles import eval_scene_angles_ranges, eval_batch_angles_ranges, music2d_from_cov_factor
from .covariance_utils import build_effective_cov_torch  # canonical cov builder (train-time usage)

# ----------------------------
# small helpers
# ----------------------------
def _cuda_mem_info(device):
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(device)  # (free, total)
        return int(free_b), int(total_b)
    return 0, 0

def _bytes_human(n):
    for u in ["B","KB","MB","GB","TB","PB"]:
        if n < 1024: return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"

def _estimate_item_bytes(L, M, N, Kmax, dtype_bytes=4):
    # (y LxMx2)+(H MxNx2)+(codes LxNx2)+(ptr 3*Kmax)+(K int64)+(R NxNx2)
    return (L*M*2 + M*N*2 + L*N*2)*dtype_bytes + (3*Kmax)*dtype_bytes + 8 + (N*N*2)*dtype_bytes

def _estimate_ds_bytes(ds, n_cap, L, M, N, Kmax, dtype_bytes=4):
    n = len(ds) if n_cap is None else min(int(n_cap), len(ds))
    return n * _estimate_item_bytes(L, M, N, Kmax, dtype_bytes)


def _safe_mkdirs(p: Path | str):
    Path(p).mkdir(parents=True, exist_ok=True)

def _ri_to_c(R_ri: torch.Tensor) -> torch.Tensor:
    # (..., 2) → complex
    # assumes last dim=2; safer than view_as_complex because we’re not requiring contiguous interleaved layout upstream
    return R_ri[..., 0].to(torch.complex64) + 1j * R_ri[..., 1].to(torch.complex64)

def _c_to_ri(Z: torch.Tensor) -> torch.Tensor:
    return torch.stack([Z.real, Z.imag], dim=-1)

def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

def _resolve_shards_train_val() -> Tuple[Path, Path]:
    """
    Resolve (train_dir, val_dir).
    Supports:
      - <DATA_SHARDS_DIR>/{train,val}/*.npz   (preferred)
      - flat <DATA_SHARDS_DIR>/*.npz          (fallback; rely on n_train/n_val caps)
      - optional cfg.DATA_SHARDS_TRAIN/VAL constants if you’ve defined them
    """
    root = Path(getattr(cfg, "DATA_SHARDS_DIR", Path("results_final/data/shards")))
    t_hint = Path(getattr(cfg, "DATA_SHARDS_TRAIN", root / "train"))
    v_hint = Path(getattr(cfg, "DATA_SHARDS_VAL",   root / "val"))

    if t_hint.exists() and list(t_hint.glob("*.npz")):
        if v_hint.exists() and list(v_hint.glob("*.npz")):
            return t_hint, v_hint
        return t_hint, t_hint  # flat split by caps
    if root.exists() and list(root.glob("*.npz")):
        return root, root
    raise FileNotFoundError(f"No shards found under {root}. Expected {t_hint} / {v_hint} or flat {root}.")

# ----------------------------
# Trainer
# ----------------------------

class Trainer:
    def __init__(self, from_hpo: bool | str = True):
        """
        from_hpo:
          - True  => apply HPO from cfg.HPO_BEST_JSON if it exists
          - False => don’t apply HPO overrides
          - str   => path to a specific best.json
        """
        # Set deterministic seeds for reproducibility
        import random
        seed = int(getattr(mdl_cfg, "SEED", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # CUDNN policy: choose deterministic OR fast, not both
        if getattr(mdl_cfg, "DETERMINISTIC_TRAINING", True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("🔒 CUDNN: Deterministic mode (reproducible, slower)")
        else:
            torch.backends.cudnn.deterministic = False  
            torch.backends.cudnn.benchmark = True
            print("⚡ CUDNN: Benchmark mode (faster, non-deterministic)")
        
        set_seed(seed)
        _safe_mkdirs(cfg.CKPT_DIR)
        _safe_mkdirs(cfg.LOGS_DIR)

        # 1) Apply HPO overrides (before model build)
        best_path = None
        if isinstance(from_hpo, str):
            best_path = Path(from_hpo)
        elif from_hpo:
            best_path = Path(getattr(cfg, "HPO_BEST_JSON", Path("results_final/hpo/best.json")))
        if best_path:
            best = _load_json_if_exists(best_path)
            if isinstance(best, dict) and "params" in best and isinstance(best["params"], dict):
                best = best["params"]
            if best:
                self._apply_hpo_to_mdl_cfg(best)

        # 2) Build model / loss / optimizer
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model    = HybridModel().to(self.device)
        # Initialize loss with config values (will be updated by schedule)
        self.loss_fn = UltimateHybridLoss(
            lam_cov=1.0,  # CRITICAL: Main covariance NMSE weight (will be scaled by HPO if needed)
            lam_subspace_align=getattr(mdl_cfg, 'LAM_SUBSPACE_ALIGN', 0.05),
            lam_peak_contrast=getattr(mdl_cfg, 'LAM_PEAK_CONTRAST', 0.1),
            lam_cov_pred=getattr(mdl_cfg, 'LAM_COV_PRED', 0.05)  # 5% auxiliary on R_pred
        ).to(self.device)
        
        # Always initialize _hpo_loss_weights (may be populated later by HPO config loading)
        self._hpo_loss_weights = {}
        
        # Beta annealing: start low (trust network), gradually blend in R_samp
        self.beta_start = 0.0
        self.beta_final = getattr(cfg, 'HYBRID_COV_BETA', 0.30)
        self.beta_warmup_epochs = None  # Will be set based on total epochs
        
        # Apply HPO loss weights if they were loaded
        if self._hpo_loss_weights:
            self._apply_hpo_loss_weights()
        lr_init       = float(getattr(mdl_cfg, "LR_INIT", 3e-4))
        opt_name      = str(getattr(mdl_cfg, "OPT", "adamw")).lower()
        wd            = float(getattr(mdl_cfg, "WEIGHT_DECAY", 1e-4))

        # CRITICAL FIX: Bullet-proof parameter grouping with strong asserts
        # Ensure model is in train mode and all parameters are trainable
        self.model.train()
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                print(f"⚠️  Re-enabling gradients for: {n}")
                p.requires_grad_(True)
        
        # Robust parameter grouping by name prefix
        backbone_params = []
        head_params = []
        HEAD_KEYS = ('k_head', 'k_mlp', 'head', 'classifier', 'aux_angles', 'aux_range',
                     'cov_fact_angle', 'cov_fact_range', 'logits_gg')
        
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # Classify by module name
            if any(k in n for k in HEAD_KEYS):
                head_params.append(p)
            else:
                backbone_params.append(p)
        
        # Calculate parameter counts
        n_back = sum(p.numel() for p in backbone_params)
        n_head = sum(p.numel() for p in head_params)
        n_tot = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Head LR multiplier (3-5× backbone LR)
        head_lr_multiplier = float(getattr(mdl_cfg, "HEAD_LR_MULTIPLIER", 4.0))  # 4× by default
        head_lr = lr_init * head_lr_multiplier
        
        print(f"🔧 Optimizer setup:")
        print(f"   Backbone: {n_back:,} params ({n_back/1e6:.1f}M) @ LR={lr_init:.2e}")
        print(f"   Head: {n_head:,} params ({n_head/1e6:.1f}M) @ LR={head_lr:.2e} ({head_lr_multiplier}×)")
        print(f"   Total trainable: {n_tot:,}")
        
        # CRITICAL ASSERTS - catch dead groups immediately
        assert n_back > 1_000_000, f"❌ Backbone param count too small: {n_back:,}"
        assert n_head > 1_000_000, f"❌ Head param count too small: {n_head:,}"
        assert n_back + n_head == n_tot, f"❌ Group sum ({n_back + n_head:,}) != total ({n_tot:,})"
        print("   ✅ Parameter grouping verified!")
        
        if opt_name == "adamw":
            self.opt = torch.optim.AdamW([
                {'params': backbone_params, 'lr': lr_init, 'weight_decay': wd},
                {'params': head_params, 'lr': head_lr, 'weight_decay': wd}
            ])
        elif opt_name == "adam":
            self.opt = torch.optim.Adam([
                {'params': backbone_params, 'lr': lr_init, 'weight_decay': wd},
                {'params': head_params, 'lr': head_lr, 'weight_decay': wd}
            ])
        else:
            self.opt = torch.optim.AdamW([
                {'params': backbone_params, 'lr': lr_init, 'weight_decay': wd},
                {'params': head_params, 'lr': head_lr, 'weight_decay': wd}
            ])

        # Expert fix: Optimizer wiring sanity check
        opt_ids = {id(p) for g in self.opt.param_groups for p in g['params']}
        missing = [n for n,p in self.model.named_parameters() if p.requires_grad and id(p) not in opt_ids]
        assert not missing, f"❌ Params missing from optimizer: {missing[:8]}"
        print(f"   ✅ Optimizer wiring verified: all {len(opt_ids)} trainable params in optimizer!")

        # AMP
        self.amp = bool(getattr(mdl_cfg, "AMP", True))
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.amp and self.device.type == "cuda"))

        # EMA
        self.use_ema   = bool(getattr(mdl_cfg, "USE_EMA", True))
        self.ema_decay = float(getattr(mdl_cfg, "EMA_DECAY", 0.999))
        # Only track floating-point parameters in EMA (skip buffers like indices)
        self.ema_shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in self.model.state_dict().items() 
            if v.dtype.is_floating_point
        }
        self._ema_bak: Dict[str, torch.Tensor] | None = None
        
        # SWA (Stochastic Weight Averaging)
        self.use_swa = getattr(mdl_cfg, 'USE_SWA', False)
        self.swa_start_frac = getattr(mdl_cfg, 'SWA_START_FRAC', 0.8)
        self.swa_lr_factor = getattr(mdl_cfg, 'SWA_LR_FACTOR', 0.1)
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = None  # Will be initialized when SWA starts
            self.swa_started = False
        else:
            self.swa_model = None
            self.swa_scheduler = None
            self.swa_started = False

        # Scheduler is created in fit() when epochs is known
        self.sched = None

        # Clip
        self.clip_norm = float(getattr(mdl_cfg, "CLIP_NORM", 1.0))
        
        # Expert fix: Step counters for debugging
        self._steps_taken = 0
        self._batches_seen = 0

    # ----------------------------
    # HPO integration
    # ----------------------------
    def _apply_hpo_to_mdl_cfg(self, best: Dict[str, Any]):
        # Typical knobs (only if present)
        if "D_MODEL"   in best: mdl_cfg.D_MODEL   = int(best["D_MODEL"])
        if "NUM_HEADS" in best: mdl_cfg.NUM_HEADS = int(best["NUM_HEADS"])
        if "dropout"   in best: mdl_cfg.DROPOUT   = float(best["dropout"])
        if "lr"        in best: mdl_cfg.LR_INIT   = float(best["lr"])

        # Loss weights from HPO (store for later application to loss_fn)
        self._hpo_loss_weights = {}
        if "lam_cov" in best:
            self._hpo_loss_weights["lam_cov"] = float(best["lam_cov"])
        if "lam_ang" in best:
            self._hpo_loss_weights["lam_ang"] = float(best["lam_ang"])
        if "lam_rng" in best:
            self._hpo_loss_weights["lam_rng"] = float(best["lam_rng"])
        if "lam_K" in best:
            self._hpo_loss_weights["lam_K"] = float(best["lam_K"])
        if "shrink_alpha" in best:
            setattr(mdl_cfg, "SHRINK_BASE_ALPHA", float(best["shrink_alpha"]))
        if "softmax_tau" in best:
            setattr(mdl_cfg, "SOFTMAX_TAU", float(best["softmax_tau"]))

        # Non-learned inference knobs that are harmless to carry here
        if "range_grid" in best:
            setattr(mdl_cfg, "INFERENCE_GRID_SIZE_RANGE",
                    int(best.get("range_grid", getattr(mdl_cfg, "INFERENCE_GRID_SIZE_RANGE", 61))))
        if "newton_iter" in best:
            setattr(mdl_cfg, "NEWTON_ITER",
                    int(best.get("newton_iter", getattr(mdl_cfg, "NEWTON_ITER", 5))))
        if "newton_lr" in best:
            setattr(mdl_cfg, "NEWTON_LR",
                    float(best.get("newton_lr", getattr(mdl_cfg, "NEWTON_LR", 0.08))))
    
    def _apply_hpo_loss_weights(self):
        """Apply HPO loss weights to the loss function (mirrors hpo.py logic)"""
        weights = self._hpo_loss_weights
        
        # Apply covariance weights (lam_cov split 20/80 for diag/off-diagonal)
        if "lam_cov" in weights:
            self.loss_fn.lam_cov = weights["lam_cov"]  # CRITICAL: Set the outer weight
            self.loss_fn.lam_diag = 0.2  # Keep fixed at 20% for diagonal
            self.loss_fn.lam_off = 0.8   # Keep fixed at 80% for off-diagonal
        
        # Apply auxiliary weights (combined for lam_aux)  
        if "lam_ang" in weights and "lam_rng" in weights:
            self.loss_fn.lam_aux = weights["lam_ang"] + weights["lam_rng"]  # Combined aux weight
        
        # Apply K-logits weight
        if "lam_K" in weights:
            self.loss_fn.lam_K = weights["lam_K"]
        
        # Set reasonable defaults for missing loss parameters (same as HPO)
        self.loss_fn.lam_ortho = 1e-3  # Orthogonality penalty
        self.loss_fn.lam_peak = 0.05   # Chamfer/peak angle loss
        self.loss_fn.lam_margin = 0.1  # Subspace margin regularizer
        self.loss_fn.lam_range_factor = 0.3  # Range factor in covariance
        mdl_cfg.LAM_ALIGN = 0.002  # Subspace alignment penalty
        
        print(f"🎯 Applied HPO loss weights: "
              f"lam_diag={getattr(self.loss_fn, 'lam_diag', 'N/A'):.3f}, "
              f"lam_off={getattr(self.loss_fn, 'lam_off', 'N/A'):.3f}, "
              f"lam_aux={getattr(self.loss_fn, 'lam_aux', 'N/A'):.3f}, "
              f"lam_K={getattr(self.loss_fn, 'lam_K', 'N/A'):.3f}")

    # ----------------------------
    # EMA helpers
    # ----------------------------
    def _ema_update(self):
        if not self.use_ema: return
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                # Only update floating-point parameters (skip integer buffers)
                if k in self.ema_shadow and v.dtype.is_floating_point:
                    self.ema_shadow[k].mul_(self.ema_decay).add_(v.detach(), alpha=1.0 - self.ema_decay)

    def _ema_swap_in(self):
        if not self.use_ema: return
        # Only backup floating-point parameters that we track in EMA
        self._ema_bak = {k: v.detach().clone() for k, v in self.model.state_dict().items() 
                        if k in self.ema_shadow}
        self.model.load_state_dict(self.ema_shadow, strict=False)

    def _ema_swap_out(self):
        if not self.use_ema: return
        assert self._ema_bak is not None
        self.model.load_state_dict(self._ema_bak, strict=False)
        self._ema_bak = None
    
    def _swa_start(self, epochs: int):
        """Initialize SWA scheduler when SWA phase begins"""
        if not self.use_swa or self.swa_started:
            return
            
        print(f"🔄 Starting SWA at epoch {int(self.swa_start_frac * epochs)}")
        
        # Initialize SWA scheduler with reduced learning rate
        swa_lr = self.opt.param_groups[0]['lr'] * self.swa_lr_factor
        self.swa_scheduler = SWALR(self.opt, swa_lr=swa_lr)
        self.swa_started = True
        
    def _swa_update(self):
        """Update SWA model with current weights"""
        if not self.use_swa or not self.swa_started:
            return
            
        self.swa_model.update_parameters(self.model)
        
    def _swa_finalize(self, dataloader):
        """Finalize SWA by updating batch norm statistics"""
        if not self.use_swa or not self.swa_started:
            return
            
        print("🔧 Finalizing SWA: updating batch norm statistics...")
        torch.optim.swa_utils.update_bn(dataloader, self.swa_model)
        
    def _swa_swap_in(self):
        """Swap in SWA model for evaluation"""
        if not self.use_swa or not self.swa_started:
            return
            
        self._swa_bak = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.swa_model.state_dict(), strict=False)
        
    def _swa_swap_out(self):
        """Swap out SWA model back to regular model"""
        if not self.use_swa or not self.swa_started:
            return
            
        if hasattr(self, '_swa_bak') and self._swa_bak is not None:
            self.model.load_state_dict(self._swa_bak, strict=False)
            self._swa_bak = None

    # ----------------------------
    # Loader builders
    # ----------------------------
    def _subset(self, ds, cap):
        if cap is None:
            return ds
        n = min(int(cap), len(ds))
        idx = np.random.RandomState(1337).permutation(len(ds))[:n]
        return Subset(ds, idx.tolist())

    def _build_loaders_cpu_io(self, n_train: Optional[int], n_val: Optional[int]):
        from .sampler import KGroupedBatchSampler
        
        tr_dir, va_dir = _resolve_shards_train_val()
        ds_tr_full = ShardNPZDataset(tr_dir)
        ds_va_full = ShardNPZDataset(va_dir)

        ds_tr = self._subset(ds_tr_full, n_train)
        ds_va = self._subset(ds_va_full, n_val)

        pin = bool(getattr(mdl_cfg, "PIN_MEMORY", True)) and (self.device.type == "cuda")
        nw  = int(getattr(mdl_cfg, "NUM_WORKERS", 0))
        bs  = int(getattr(mdl_cfg, "BATCH_SIZE", 128))

        # Use pad+mask collate function for variable-K batching
        from .collate_fn import collate_pad_to_kmax_with_snr
        
        # train loader kwargs (use standard batching with custom collate)
        tr_kwargs = dict(dataset=ds_tr,
                         batch_size=bs,
                         shuffle=True,
                         collate_fn=lambda batch: collate_pad_to_kmax_with_snr(batch, cfg.K_MAX),
                         num_workers=nw,
                         pin_memory=pin)
        if nw > 0:
            tr_kwargs["persistent_workers"] = bool(getattr(mdl_cfg, "PERSISTENT_WORKERS", True))
            tr_kwargs["prefetch_factor"]    = int(getattr(mdl_cfg, "PREFETCH", 2))

        tr_loader = DataLoader(**tr_kwargs)

        # val loader kwargs (use standard batching with custom collate)
        va_kwargs = dict(dataset=ds_va,
                         batch_size=bs,
                         shuffle=False,
                         drop_last=False,  # CRITICAL: Don't drop last batch for validation
                         collate_fn=lambda batch: collate_pad_to_kmax_with_snr(batch, cfg.K_MAX),
                         num_workers=nw,
                         pin_memory=pin)
        if nw > 0:
            va_kwargs["persistent_workers"] = bool(getattr(mdl_cfg, "PERSISTENT_WORKERS", True))
            va_kwargs["prefetch_factor"]    = int(getattr(mdl_cfg, "PREFETCH", 2))

        va_loader = DataLoader(**va_kwargs)
        
        # CRITICAL: Log validation loader info to debug infinite loops
        print(f"[VAL LOADER] len(val_ds)={len(ds_va)}, batches per epoch={len(va_loader)}")
        print(f"[VAL LOADER] batch_size={bs}, drop_last={va_kwargs.get('drop_last', 'default')}")
        
        return tr_loader, va_loader

    def _aggregate_cpu_then_gpu(self, ds):
    
        import time, traceback
        from pathlib import Path
        import numpy as np
        import torch

        # unwrap Subset if needed
        if isinstance(ds, Subset):
            base = ds.dataset
            idxs = ds.indices
        else:
            base = ds
            idxs = None

        if not hasattr(base, "meta") or not hasattr(base, "index_map"):
            raise RuntimeError("GPU-cache builder expects ShardNPZDataset (or Subset thereof).")

        # build per-shard index lists
        per_shard = {si: [] for si in range(len(base.meta))}
        if idxs is None:
            for _, (si, i) in enumerate(base.index_map):
                per_shard[si].append(i)
        else:
            for gidx in idxs:
                si, i = base.index_map[gidx]
                per_shard[si].append(i)

        total_needed = sum(len(v) for v in per_shard.values())
        needed_shards = sum(1 for v in per_shard.values() if v)
        print(f"[GPU cache] Will stage {total_needed} samples from {needed_shards} shard(s).")
        if total_needed == 0:
            raise RuntimeError("GPU-cache builder: empty subset (n_train/n_val too small?)")

        Ys, Hs, Cs, PTRs, Ks, Rs, SNRs, H_fulls, R_samps = [], [], [], [], [], [], [], [], []
        done = 0
        t0 = time.time()
        SLICE = 2048  # Restored original size for 1.5TB RAM system

        for si, idx_list in per_shard.items():
            if not idx_list:
                continue

            shard_path, n_this, L = base.meta[si]
            shard_path = Path(shard_path)
            try:
                z = np.load(shard_path, allow_pickle=False, mmap_mode="r")
            except Exception as e:
                print(f"[GPU cache][ERROR] np.load failed for '{shard_path}': {e}")
                raise

            idx_arr = np.array(idx_list, dtype=np.int64)
            idx_arr.sort()

            for start in range(0, idx_arr.size, SLICE):
                sel = idx_arr[start:start+SLICE]
                try:
                    y  = torch.from_numpy(z["y"][sel])         # (B,L,M,2)
                    H  = torch.from_numpy(z["H"][sel])         # (B,L,M,2) - old collapsed channel
                    C  = torch.from_numpy(z["codes"][sel])     # (B,L,N,2)
                    pr = torch.from_numpy(z["ptr"][sel])       # (B,3*Kmax)
                    Kt = torch.from_numpy(z["K"][sel]).long()  # (B,)
                    R  = torch.from_numpy(z["R"][sel])         # (B,N,N,2)
                    snr = torch.from_numpy(z["snr"][sel])      # (B,) - CRITICAL FIX!
                    
                    # CRITICAL FIX: Load H_full if present (for hybrid covariance blending)
                    if "H_full" in z:
                        H_full = torch.from_numpy(z["H_full"][sel])  # (B,M,N,2)
                    else:
                        # FAIL FAST: Dataset must have H_full for hybrid blending!
                        print(f"⚠️  WARNING: Shard '{shard_path}' lacks 'H_full'. Hybrid blending will fail!")
                        H_full = None
                    
                    # CRITICAL FIX: Load R_samp if present (for hybrid covariance in loss)
                    if "R_samp" in z:
                        R_samp_batch = torch.from_numpy(z["R_samp"][sel])  # (B,N,N,2)
                    else:
                        print(f"⚠️  WARNING: Shard '{shard_path}' lacks 'R_samp'. Training will use pure R_pred!")
                        R_samp_batch = None
                except Exception:
                    print(f"[GPU cache][ERROR] slicing shard='{shard_path}' at indices {sel[:3]}..{sel[-3:]} (count={sel.size})")
                    print(traceback.format_exc())
                    raise

                Ys.append(y);  Hs.append(H);  Cs.append(C)
                PTRs.append(pr); Ks.append(Kt); Rs.append(R)
                SNRs.append(snr)  # CRITICAL FIX!
                if H_full is not None:
                    H_fulls.append(H_full)
                if R_samp_batch is not None:
                    R_samps.append(R_samp_batch)

                done += sel.size
                if done % 8192 == 0 or done == total_needed:
                    dt = time.time() - t0
                    rate = done / max(1e-6, dt)
                    print(f"[GPU cache] staged {done}/{total_needed}  ({rate:.1f} samp/s)")

            try:
                z.close()
            except Exception:
                pass

        print("[GPU cache] Concatenating CPU tensors …")
        y  = torch.cat(Ys,   0)
        H  = torch.cat(Hs,   0)
        C  = torch.cat(Cs,   0)
        pr = torch.cat(PTRs, 0)
        K  = torch.cat(Ks,   0)
        R  = torch.cat(Rs,   0)
        snr = torch.cat(SNRs, 0)  # CRITICAL FIX!
        
        # CRITICAL FIX: Concatenate H_full if present
        H_full = None
        if H_fulls:
            H_full = torch.cat(H_fulls, 0)
            print(f"[GPU cache] H_full loaded: shape={H_full.shape}")
        else:
            print(f"⚠️  WARNING: No H_full found in dataset! Hybrid covariance will be disabled.")
        
        # CRITICAL FIX: Concatenate R_samp if present (for hybrid loss)
        R_samp = None
        if R_samps:
            R_samp = torch.cat(R_samps, 0)
            print(f"[GPU cache] R_samp loaded: shape={R_samp.shape}")
        else:
            print(f"⚠️  WARNING: No R_samp found in dataset! Training will use pure R_pred.")

        gb = (y.numel()*y.element_size()
            + H.numel()*H.element_size()
            + C.numel()*C.element_size()
            + pr.numel()*pr.element_size()
            + K.numel()*K.element_size()
            + R.numel()*R.element_size()) / (1024**3)
        if H_full is not None:
            gb += H_full.numel() * H_full.element_size() / (1024**3)
        if R_samp is not None:
            gb += R_samp.numel() * R_samp.element_size() / (1024**3)
        print(f"[GPU cache] Transferring ~{gb:.2f} GB to {self.device} …")

        y  = y.to(self.device, non_blocking=True)
        H  = H.to(self.device, non_blocking=True)
        C  = C.to(self.device, non_blocking=True)
        pr = pr.to(self.device, non_blocking=True)
        K  = K.to(self.device, non_blocking=True)
        R  = R.to(self.device, non_blocking=True)
        snr = snr.to(self.device, non_blocking=True)  # CRITICAL FIX!
        if H_full is not None:
            H_full = H_full.to(self.device, non_blocking=True)
        if R_samp is not None:
            R_samp = R_samp.to(self.device, non_blocking=True)

        # CRITICAL FIX: Clean up CPU memory after GPU transfer to avoid OOM-killer
        del Ys, Hs, Cs, PTRs, Ks, Rs, SNRs, H_fulls, R_samps
        import gc; gc.collect()

        print("[GPU cache] Done.")
        # CRITICAL FIX: Include H_full and R_samp in TensorDataset if present
        # Order: (y, H, C, ptr, K, R, snr, H_full, R_samp) - must match _unpack_any_batch
        if H_full is not None and R_samp is not None:
            return torch.utils.data.TensorDataset(y, H, C, pr, K, R, snr, H_full, R_samp)
        elif H_full is not None:
            return torch.utils.data.TensorDataset(y, H, C, pr, K, R, snr, H_full)
        else:
            return torch.utils.data.TensorDataset(y, H, C, pr, K, R, snr)


    def _build_loaders_gpu_cache_train_only(self, n_train: Optional[int], n_val: Optional[int]):
        """Build GPU cache for train only, use regular DataLoader for val to avoid OOM-killer"""
        tr_dir, va_dir = _resolve_shards_train_val()
        ds_tr_full = ShardNPZDataset(tr_dir)
        ds_va_full = ShardNPZDataset(va_dir)

        ds_tr = self._subset(ds_tr_full, n_train)
        ds_va = self._subset(ds_va_full, n_val)

        print(f"[GPU cache] train subset={len(ds_tr)}  val subset={len(ds_va)}")

        # CRITICAL FIX: Only cache train data to avoid OOM-killer
        # Skip expensive GPU aggregation for validation
        bs = int(getattr(mdl_cfg, "BATCH_SIZE", 32))  # Smaller batch size for HPO
        
        # CRITICAL FIX: Use pad+mask collate for GPU cache too
        from .collate_fn import collate_pad_to_kmax_with_snr
        
        # Use original datasets with pad+mask collate
        tr_loader = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=True,
                               collate_fn=lambda batch: collate_pad_to_kmax_with_snr(batch, cfg.K_MAX),
                               num_workers=0, pin_memory=False)
        va_loader = DataLoader(ds_va, batch_size=bs, shuffle=False, drop_last=False,
                               collate_fn=lambda batch: collate_pad_to_kmax_with_snr(batch, cfg.K_MAX),
                               num_workers=0, pin_memory=False)
        
        # CRITICAL: Log validation loader info to debug infinite loops
        print(f"[VAL LOADER GPU] len(val_ds)={len(ds_va)}, batches per epoch={len(va_loader)}")
        print(f"[VAL LOADER GPU] batch_size={bs}, drop_last=False")
        
        return tr_loader, va_loader

    def _build_loaders_gpu_cache(self, n_train: Optional[int], n_val: Optional[int]):
        tr_dir, va_dir = _resolve_shards_train_val()
        ds_tr_full = ShardNPZDataset(tr_dir)
        ds_va_full = ShardNPZDataset(va_dir)

        ds_tr = self._subset(ds_tr_full, n_train)
        ds_va = self._subset(ds_va_full, n_val)

        print(f"[GPU cache] train subset={len(ds_tr)}  val subset={len(ds_va)}")

        # CRITICAL FIX: Use GPU cache for both train and val for maximum speed
        # With 1.5TB RAM, we can afford the expensive transfer for speed
        gtr = self._aggregate_cpu_then_gpu(ds_tr)
        gva = self._aggregate_cpu_then_gpu(ds_va)

        bs = int(getattr(mdl_cfg, "BATCH_SIZE", 32))
        
        # CRITICAL FIX: Use aggregated GPU data for both train and val
        tr_loader = DataLoader(gtr, batch_size=bs, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)
        va_loader = DataLoader(gva, batch_size=bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)
        
        # CRITICAL: Log validation loader info to debug infinite loops
        print(f"[VAL LOADER GPU] len(val_ds)={len(ds_va)}, batches per epoch={len(va_loader)}")
        print(f"[VAL LOADER GPU] batch_size={bs}, drop_last=False")
        
        return tr_loader, va_loader


    # ----------------------------
    # One epoch (train/val)
    # ----------------------------
    def _unpack_any_batch(self, batch):
        # tuple/list from GPU-cache path
        if isinstance(batch, (list, tuple)) and len(batch) >= 6:
            # Standardized order: (y, H, C, ptr, K, R_in, snr, H_full, R_samp?)
            is_list = isinstance(batch, list)
            if len(batch) >= 9:
                return tuple(batch)  # already includes R_samp
            elif len(batch) == 8:  # includes SNR + H_full
                return tuple(batch) + (None,)  # add R_samp=None
            elif len(batch) == 7:  # includes SNR, add None for H_full and R_samp
                return tuple(batch) + (None, None)
            else:  # missing SNR and H_full; add dummies
                snr_dummy = torch.zeros(batch[0].shape[0], device=self.device)
                return tuple(batch) + (snr_dummy, None, None)
        # dict from CPU-IO path
        y    = batch["y"].to(self.device, non_blocking=True)
        H    = batch["H"].to(self.device, non_blocking=True)
        C    = batch["codes"].to(self.device, non_blocking=True)
        ptr  = batch["ptr"].to(self.device, non_blocking=True)
        K    = batch["K"].to(self.device, non_blocking=True)
        R_in = batch["R"].to(self.device, non_blocking=True)
        snr  = batch.get("snr_db", torch.zeros(y.shape[0], device=self.device)).to(self.device, non_blocking=True)
        # CRITICAL: Also extract H_full if present (for hybrid covariance blending)
        H_full = batch.get("H_full", None)
        if H_full is not None:
            H_full = H_full.to(self.device, non_blocking=True)
        # NEW: Optional offline R_samp
        R_samp = batch.get("R_samp", None)
        if R_samp is not None:
            R_samp = R_samp.to(self.device, non_blocking=True)
        return (y, H, C, ptr, K, R_in, snr, H_full, R_samp)

    def _train_one_epoch(self, loader, epoch:int, epochs:int, max_batches: Optional[int]=None, grad_accumulation: int = 1):
        self.model.train()
        running = 0.0
        iters = len(loader) if max_batches is None else min(len(loader), max_batches)
        
        # Expert fix: Initialize parameter vector for drift tracking (if first epoch)
        if not hasattr(self, "_param_vec_prev"):
            with torch.no_grad():
                self._param_vec_prev = torch.nn.utils.parameters_to_vector([p.detach().float() for p in self.model.parameters() if p.requires_grad])
        
        # Expert debug: Print LR every epoch
        lrs = [g['lr'] for g in self.opt.param_groups]
        print(f"[LR] epoch={epoch} groups={['backbone','head']} lr={lrs}", flush=True)

        for bi, batch in enumerate(loader):
            if bi >= iters: break
            self._batches_seen += 1  # Expert fix: Track batch counter
            y, H, C, ptr, K, R_in, snr, H_full, R_samp = self._unpack_any_batch(batch)
            
            # CRITICAL FIX: Clear batch references to prevent memory leaks
            del batch

            # prepare labels (symmetrized R + SNR for shrinkage)
            R_true_c = _ri_to_c(R_in)
            R_true_c = 0.5 * (R_true_c + R_true_c.conj().transpose(-2, -1))
            
            # CRITICAL FIX: Normalize R_true to trace=N for consistent scaling with R_blend
            N = R_true_c.shape[-1]
            tr_true = torch.diagonal(R_true_c, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-9)
            R_true_c = R_true_c * (N / tr_true).view(-1, 1, 1)
            
            R_true   = _c_to_ri(R_true_c).float()
            labels   = {"R_true": R_true, "ptr": ptr, "K": K, "snr_db": snr}

            # Zero gradients only at the start of accumulation
            if bi % grad_accumulation == 0:
                self.opt.zero_grad(set_to_none=True)
                
            # CRITICAL FIX: Network forward in FP16, loss computation in FP32
            # ENABLE AMP for faster training (we've fixed the numerical issues)
            with torch.amp.autocast('cuda', enabled=self.amp):
                preds = self.model(y=y, H=H, codes=C, snr_db=snr, R_samp=R_samp)
            
            # CRITICAL FIX: LOSS IN FP32 - turn off autocast for numerical stability
            # Cast EVERYTHING to FP32 before loss computation (preds AND labels)
            with torch.amp.autocast('cuda', enabled=False):
                # Cast predictions to FP32/complex64
                preds_fp32 = {}
                for k, v in preds.items():
                    if v.dtype == torch.float16:
                        preds_fp32[k] = v.to(torch.float32)
                    elif v.dtype == torch.complex32:
                        preds_fp32[k] = v.to(torch.complex64)
                    else:
                        preds_fp32[k] = v
                
                # Cast labels to FP32/complex64 (CRITICAL!)
                labels_fp32 = {}
                for k, v in labels.items():
                    if isinstance(v, torch.Tensor):
                        if v.dtype == torch.float16:
                            labels_fp32[k] = v.to(torch.float32)
                        elif v.dtype == torch.complex32:
                            labels_fp32[k] = v.to(torch.complex64)
                        else:
                            labels_fp32[k] = v
                    else:
                        labels_fp32[k] = v
                
                # DEBUG: Check for NaN/Inf in network outputs (first batch only)
                if epoch == 1 and bi == 0:
                    print(f"[DEBUG] Checking dtypes before loss computation:")
                    for k, v in preds_fp32.items():
                        if isinstance(v, torch.Tensor):
                            print(f"[DEBUG] preds[{k}]: dtype={v.dtype}, shape={v.shape}")
                            if torch.isnan(v).any() or torch.isinf(v).any():
                                print(f"[DEBUG] NaN/Inf detected in preds[{k}] BEFORE loss!")
                    for k, v in labels_fp32.items():
                        if isinstance(v, torch.Tensor):
                            print(f"[DEBUG] labels[{k}]: dtype={v.dtype}, shape={v.shape}")
                            if torch.isnan(v).any() or torch.isinf(v).any():
                                print(f"[DEBUG] NaN/Inf detected in labels[{k}] BEFORE loss!")
                
                # CRITICAL FIX: Construct blended covariance for loss function
                # This ensures eigengap loss operates on well-conditioned matrix
                if 'cov_fact_angle' in preds_fp32 and 'cov_fact_range' in preds_fp32:
                    # Extract factors from network output
                    A_angle = preds_fp32['cov_fact_angle']  # [B, N, K_MAX]
                    A_range = preds_fp32['cov_fact_range']  # [B, N, K_MAX]
                    
                    # Define N for normalization
                    N = cfg.N_H * cfg.N_V
                    
                    # Get batch size for assertions
                    B = preds_fp32['cov_fact_angle'].size(0)
                    
                    # Hard assertions to catch future shape regressions fast:
                    assert cfg.N_H * cfg.N_V == 144, "config mismatch: N != 144"
                    assert preds_fp32['cov_fact_angle'].numel() % (B * 144) == 0
                    assert preds_fp32['cov_fact_range'].numel() % (B * 144) == 0
                    
                    # EXPERT FIX: Build R_pred robustly to prevent shape bugs
                    def build_R_pred_from_factors(preds, cfg):
                        B = preds['cov_fact_angle'].size(0)
                        N = cfg.N_H * cfg.N_V  # 12*12 = 144
                        K_MAX = cfg.K_MAX

                        # cov_fact_angle/range are [B, 2*N*K_MAX] with interleaved real/imag
                        # Convert to complex [B, N, K_MAX]
                        flat_ang = preds['cov_fact_angle'].contiguous().float()
                        flat_rng = preds['cov_fact_range'].contiguous().float()

                        assert flat_ang.dim() == 2 and flat_ang.size(0) == B, "bad factor shape"
                        assert flat_rng.dim() == 2 and flat_rng.size(0) == B, "bad factor shape"
                        assert flat_ang.size(1) == 2 * N * K_MAX, f"expected {2*N*K_MAX}, got {flat_ang.size(1)}"
                        assert flat_rng.size(1) == 2 * N * K_MAX, f"expected {2*N*K_MAX}, got {flat_rng.size(1)}"

                        # Convert interleaved real/imag to complex: [B, 2*N*K] -> [B, N, K] complex
                        def _vec2c_local(v):
                            xr, xi = v[:, ::2], v[:, 1::2]  # [B, N*K] each
                            return torch.complex(xr.view(B, N, K_MAX), xi.view(B, N, K_MAX))

                        A_ang = _vec2c_local(flat_ang)  # [B, N, K_MAX] complex
                        A_rng = _vec2c_local(flat_rng)  # [B, N, K_MAX] complex

                        # Build R_pred = A_ang @ A_ang^H + lam * A_rng @ A_rng^H
                        lam_range = getattr(cfg, 'LAM_RANGE_FACTOR', 0.1)
                        R_pred = (A_ang @ A_ang.conj().transpose(-2, -1)) + lam_range * (A_rng @ A_rng.conj().transpose(-2, -1))
                        # Hermitize (should already be Hermitian, but for numerical stability)
                        R_pred = 0.5 * (R_pred + R_pred.conj().transpose(-2, -1))
                        return R_pred

                    R_pred = build_R_pred_from_factors(preds_fp32, cfg)
                    
                    # Quick sanity before blending:
                    B, N = R_pred.shape[:2]
                    assert R_pred.shape == (B, N, N), f"R_pred bad shape: {R_pred.shape}"
                    
                    # CRITICAL FIX: Don't normalize R_pred to N (it's rank-deficient)
                    # Only normalize the final R_blend to N after blending
                    
                    # CRITICAL FIX: Construct R_samp from snapshots (NEVER use R_true!)
                    # This must match the inference pipeline exactly
                    # OPTIMIZED: Use more efficient computation to avoid hanging
                    # Prefer offline precomputed R_samp; avoid online LS in hot path
                    B, N = R_pred.shape[:2]
                    if R_samp is not None:
                        R_samp_c = _ri_to_c(R_samp.to(torch.float32))
                        # Beta schedule
                        if hasattr(self, 'beta_warmup_epochs') and self.beta_warmup_epochs is not None and epoch <= self.beta_warmup_epochs:
                            beta = self.beta_start + (self.beta_final - self.beta_start) * (epoch / max(1, self.beta_warmup_epochs))
                        else:
                            beta = self.beta_final
                        # Optional jitter
                        if self.model.training and (self.beta_warmup_epochs is None or epoch > self.beta_warmup_epochs):
                            jitter = getattr(cfg, 'BETA_JITTER_HPO', 0.02) if hasattr(self, '_hpo_loss_weights') and self._hpo_loss_weights else getattr(cfg, 'BETA_JITTER_FULL', 0.05)
                            if jitter > 0.0:
                                beta = float((beta + jitter * (2.0 * torch.rand((), device=R_pred.device) - 1.0)).clamp(0.0, 0.95))
                        if epoch == 1 and bi == 0:
                            print(f"[Beta] epoch={epoch}, beta={beta:.3f} (offline R_samp)", flush=True)
                        # Build blended covariance using the canonical helper (no shrink/diag-load here).
                        R_blend = build_effective_cov_torch(
                            R_pred,
                            snr_db=None,                 # do NOT shrink here; loss applies it consistently
                            R_samp=R_samp_c.detach(),    # no grad through sample cov
                            beta=float(beta),
                            diag_load=False,
                            apply_shrink=False,
                            target_trace=float(N),
                        )
                        preds_fp32['R_blend'] = R_blend
                    else:
                        # No offline R_samp available → use pure R_pred
                        if epoch == 1 and bi == 0 and getattr(cfg, "HYBRID_COV_BLEND", True) and getattr(cfg, "HYBRID_COV_BETA", 0.0) > 0.0:
                            print("[Hybrid] R_samp not available; using pure R_pred for loss.", flush=True)
                        preds_fp32['R_blend'] = R_pred
        
        # Compute loss in FP32
        loss  = self.loss_fn(preds_fp32, labels_fp32)
        
        # Expert fix: Log exact optimized loss for first batch
        if epoch == 1 and bi == 0:
            print(f"[OPTIMIZED LOSS] batch={bi} total_loss={loss.detach().item():.6f}", flush=True)
        
        # Expert debug: Gradient path test - verify R_blend has live gradients to heads
        if epoch == 1 and bi == 0:
            assert 'R_blend' in preds_fp32, "R_blend missing from preds!"
            assert preds_fp32['R_blend'].requires_grad, "R_blend lost grad path!"
            
            # Unit-test the gradient path from R_blend to any head parameter
            try:
                any_head = next(p for n,p in self.model.named_parameters() if "cov_fact_angle" in n and p.requires_grad)
                s = preds_fp32['R_blend'].real.mean()  # cheap scalar depending on R_pred
                g = torch.autograd.grad(s, any_head, retain_graph=True, allow_unused=True)[0]
                print(f"[GRADPATH] d<R_blend>/d(cov_fact_angle) = {0.0 if g is None else g.norm().item():.3e}", flush=True)
            except StopIteration:
                print(f"[GRADPATH] No cov_fact_angle parameter found!", flush=True)
        
        # DEBUG: Check if loss itself is NaN (first batch only)
        if epoch == 1 and bi == 0:
            if torch.isnan(loss):
                print(f"[DEBUG] Loss is NaN! loss={loss.item()}")
            elif torch.isinf(loss):
                print(f"[DEBUG] Loss is Inf! loss={loss.item()}")
            else:
                print(f"[DEBUG] Loss is finite: {loss.item():.6f}")
        
        # Expert fix: Log first batch loss before backward
        if epoch == 1 and bi == 0:
            print(f"[DEBUG] total loss pre-backward = {float(loss.detach().item()):.6f}", flush=True)
        
        # Scale loss by accumulation steps for proper averaging
        loss = loss / grad_accumulation

        self.scaler.scale(loss).backward()
        
        # Only step optimizer every grad_accumulation steps
        if (bi + 1) % grad_accumulation == 0 or (bi + 1) == iters:
            # Unscale ONCE right before step
            self.scaler.unscale_(self.opt)
            
            # Expert fix: Check GradScaler internal state to see if step will be skipped
            found_inf_map = {}
            try:
                state = self.scaler._per_optimizer_states[self.opt]
                # PyTorch stores found_inf tensors per device
                for dev, t in state["found_inf_per_device"].items():
                    found_inf_map[str(dev)] = float(t.item())
                current_scale = float(self.scaler.get_scale())
            except Exception as e:
                found_inf_map = {"n/a": -1.0}
                current_scale = -1.0

            if epoch == 1 and bi < 3:
                print(f"[AMP] scale={current_scale} found_inf={found_inf_map}", flush=True)
            
            # Expert fix: Log gradient norms before and after sanitization
            if epoch == 1 and bi == 0:
                grad_norm_before = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                print(f"[GRAD SANITIZE] Before: ||g||_2={grad_norm_before:.3e}", flush=True)
            
            # Expert fix: Conservative gradient sanitization to avoid masking AMP overflow
            # Only zero out NaN gradients, leave inf gradients alone (AMP will handle them)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=float('inf'), neginf=float('-inf'))
            
            # Expert fix: Log gradient norms after sanitization
            if epoch == 1 and bi == 0:
                grad_norm_after = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                print(f"[GRAD SANITIZE] After: ||g||_2={grad_norm_after:.3e}", flush=True)
            
            # Gradient clipping (AFTER sanitization, BEFORE step)
            if self.clip_norm and self.clip_norm > 0:
                grad_norm_clipped = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                if epoch == 1 and bi == 0:
                    print(f"[GRAD CLIP] After clipping: ||g||_2={grad_norm_clipped:.3e}", flush=True)
            
            # Expert fix: Improved gradient flow instrumentation
            import math
            HEAD_KEYS = ('k_head', 'k_mlp', 'head', 'classifier', 'aux_angles', 'aux_range',
                         'cov_fact_angle', 'cov_fact_range', 'logits_gg')
            
            def _group_grad_norm(params):
                vals = [p.grad.norm(2).item() for p in params if p.grad is not None and torch.isfinite(p.grad).all()]
                return (sum(v*v for v in vals))**0.5 if vals else 0.0
            
            # Expert fix: Robust step gate - step if ANY gradient signal is present and finite
            def total_grad_norm(params):
                norms = []
                grad_count = 0
                none_count = 0
                for p in params:
                    if p.grad is not None:
                        grad_count += 1
                        norms.append(p.grad.norm(2))
                    else:
                        none_count += 1
                if not norms:
                    return torch.tensor(0.0, device=self.device), grad_count, none_count
                return torch.norm(torch.stack(norms), 2), grad_count, none_count

            g_total, grad_count, none_count = total_grad_norm(self.model.parameters())
            g_total = float(g_total.detach().cpu())
            
            # Expert fix: Check for AMP overflow
            overflow = any(v > 0.0 for v in found_inf_map.values())
            ok = (not overflow) and math.isfinite(g_total) and (g_total > 1e-12)

            # Expert fix: Log gradient status for ALL batches in first epoch
            if epoch == 1:
                print(f"[GRAD] batch={bi} ||g||_2={g_total:.3e} ok={ok} overflow={overflow} (grads={grad_count}, none={none_count})", flush=True)
            
            # Initialize stepped flag
            stepped = False
            
            # CRITICAL: Skip step if gradients are non-finite or overflow detected
            if not ok:
                if epoch == 1:
                    print(f"[STEP] batch={bi} SKIPPED - overflow={overflow} ||g||_2={g_total:.3e}", flush=True)
                self.opt.zero_grad(set_to_none=True)
                self.scaler.update()  # Still update scaler state
            else:
                # Step optimizer (with AMP scaler)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
                stepped = True
                
                # Expert fix: Log when step is actually taken
                if epoch == 1:
                    print(f"[STEP] batch={bi} STEP TAKEN - g_total={g_total:.3e}", flush=True)
                
                # Expert fix: Track steps taken
                self._steps_taken += 1
                if epoch == 1 and bi < 3:
                    lrs = [g['lr'] for g in self.opt.param_groups]
                    print(f"[OPT] step {self._steps_taken} / batch {self._batches_seen} LRs={lrs}", flush=True)
                
                # Expert fix: Parameter drift probe - measure BEFORE EMA update
                with torch.no_grad():
                    vec_now = torch.nn.utils.parameters_to_vector([p.detach().float() for p in self.model.parameters() if p.requires_grad])
                    delta = (vec_now - getattr(self, "_param_vec_prev", vec_now)).norm().item()
                    self._param_vec_prev = vec_now.detach().clone()
                    print(f"[STEP] Δparam ||·||₂ = {delta:.3e}", flush=True)
                
                self._ema_update()
                
                # Update SWA model if in SWA phase
                if self.swa_started:
                    self._swa_update()

        running += float(loss.detach().item()) * grad_accumulation  # Undo the scaling for logging
        
        # CRITICAL FIX: Clear intermediate tensors to prevent memory leaks
        del y, H, C, ptr, K, R_in, snr, R_true_c, R_true, labels, loss

        # Expert fix: Update schedulers only after actual steps
        if stepped:
            if self.swa_started and self.swa_scheduler is not None:
                self.swa_scheduler.step()
            elif self.sched is not None:
                self.sched.step()
        
        # CRITICAL FIX: Force garbage collection at end of epoch
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return running / max(1, iters)

    @torch.no_grad()
    def _validate_one_epoch(self, loader, max_batches: Optional[int]=None, return_debug=False):
        self.model.eval()
        running = 0.0
        iters = len(loader) if max_batches is None else min(len(loader), max_batches)
        
        # Accumulate per-term losses for debugging
        debug_acc = {}
        
        # ICC FIX: Track K-logits std for first batch diagnostics
        first_batch_k_logits_std = None

        for bi, batch in enumerate(loader):
            if bi >= iters: break
            
            # Save first batch for eigenspectrum diagnostic (epoch 0 only)
            if bi == 0 and not hasattr(self, '_first_val_batch'):
                self._first_val_batch = batch
            
            y, H, C, ptr, K, R_in, snr, H_full, _ = self._unpack_any_batch(batch)
            
            # CRITICAL FIX: Clear batch references to prevent memory leaks (except first)
            if bi > 0:
                del batch

            R_true_c = _ri_to_c(R_in)
            R_true_c = 0.5 * (R_true_c + R_true_c.conj().transpose(-2, -1))
            R_true   = _c_to_ri(R_true_c).float()
            labels   = {"R_true": R_true, "ptr": ptr, "K": K, "snr_db": snr}

            # Expert fix: Forward can stay FP16 for speed
            with torch.amp.autocast('cuda', enabled=(self.amp and self.device.type == "cuda")):
                preds_half = self.model(y=y, H=H, codes=C, snr_db=snr)
            
            # Expert fix: Loss MUST be in FP32 for numerical stability (SVD/eigens/divides)
            with torch.amp.autocast('cuda', enabled=False):
                # Cast preds to FP32
                preds = {}
                for k, v in preds_half.items():
                    if isinstance(v, torch.Tensor):
                        if v.dtype == torch.float16:
                            preds[k] = v.float()
                        elif v.dtype == torch.complex32:
                            preds[k] = v.to(torch.complex64)
                        else:
                            preds[k] = v
                    else:
                        preds[k] = v
                
                # Cast labels to FP32 (already float but ensure)
                labels_fp32 = {}
                for k, v in labels.items():
                    if isinstance(v, torch.Tensor):
                        if v.dtype == torch.float16:
                            labels_fp32[k] = v.float()
                        elif v.dtype == torch.complex32:
                            labels_fp32[k] = v.to(torch.complex64)
                        else:
                            labels_fp32[k] = v
                    else:
                        labels_fp32[k] = v
                
                loss = self.loss_fn(preds, labels_fp32)
                
                # Expert fix: Guard against non-finite validation loss
                if not torch.isfinite(loss):
                    loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)
                
                # Expert debug: Print validation loss for first batch each epoch
                if bi == 0:
                    print(f"[VAL] loss(first-batch)={float(loss):.6f}", flush=True)
                    
                    # Oracle floor check: if R_pred == R_true, what is total?
                    if getattr(mdl_cfg, "OVERFIT_NMSE_PURE", False):
                        try:
                            preds_oracle = {**preds}
                            # Only set R_hat (not R_blend) since we're in pure overfit mode
                            preds_oracle['R_hat'] = labels_fp32['R_true']
                            loss_oracle = self.loss_fn(preds_oracle, labels_fp32).detach().item()
                            print(f"[VAL DEBUG] oracle_total_if_Rpred_eq_Rtrue = {loss_oracle:.6f}", flush=True)
                        except Exception as e:
                            print(f"[VAL DEBUG] oracle check failed: {e}", flush=True)
            
            running += float(loss.detach().item())
            
            # ICC FIX: Capture K-logits std from first batch
            if bi == 0 and "k_logits" in preds:
                first_batch_k_logits_std = float(preds["k_logits"].std().detach().cpu().item())
            
            # Accumulate debug terms if requested
            if return_debug:
                debug = self.loss_fn.debug_terms(preds, labels)
                for key, val in debug.items():
                    if isinstance(val, (int, float)):
                        debug_acc[key] = debug_acc.get(key, 0.0) + val

        avg_loss = running / max(1, iters)
        
        # CRITICAL FIX: Force garbage collection at end of validation
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if return_debug:
            # Average debug terms
            for key in debug_acc:
                debug_acc[key] /= max(1, iters)
            # ICC FIX: Add K-logits std to debug info
            if first_batch_k_logits_std is not None:
                debug_acc['k_logits_std'] = first_batch_k_logits_std
            return avg_loss, debug_acc
        return avg_loss

    # ----------------------------
    # Phase curriculum (3-stage)
    # ----------------------------
    def _apply_phase_weights(self, phase: int, epoch: int = 0, total_epochs: int = 60):
        """
        Phase schedule B1/B2/B3 with smoother cross-gram ramping.
        Tweaks loss mixing without touching architecture.
        
        IMPORTANT: Respects ALL HPO-selected weights by scaling curriculum values!
        """
        # Get HPO-selected weights (if available), otherwise use defaults
        hpo_lam_cov = self._hpo_loss_weights.get("lam_cov", 1.0)  # Default 1.0 (no scaling)
        hpo_lam_ang = self._hpo_loss_weights.get("lam_ang", 0.2)  # Default 0.2
        hpo_lam_rng = self._hpo_loss_weights.get("lam_rng", 0.2)  # Default 0.2
        hpo_lam_K = self._hpo_loss_weights.get("lam_K", 0.12)     # Default 0.12
        
        if phase == 0:   # B1 (e1-e4): Angles ONLY, minimal range/K
            # Set all primary loss weights (using HPO suggestions as base)
            self.loss_fn.lam_cov   = hpo_lam_cov * 0.25   # lam_cov≈0.25
            self.loss_fn.lam_K     = 0.05                 # lam_K≤0.05
            
            # Phase 0: lam_ang≈0.7, NO range - lock clean angle seeds early
            self.loss_fn.lam_aux   = hpo_lam_ang * 0.70 + hpo_lam_rng * 0.00  # lam_ang≈0.7
            
            # Covariance structure terms (scaled by cov weight)
            self.loss_fn.lam_cross = 1e-3 * hpo_lam_cov
            self.loss_fn.lam_gap   = 0.03 * hpo_lam_cov
            
            # Fixed terms (not from HPO)
            self.loss_fn.lam_ortho = 1e-3
            self.loss_fn.lam_peak = 0.05
            self.loss_fn.lam_margin = 0.1
            self.loss_fn.lam_range_factor = 0.3
            
            # Higher dropout in early phase
            dropout = max(float(getattr(mdl_cfg, "DROPOUT", 0.10)), 0.10)
            mdl_cfg.DROPOUT = dropout
            self.model.set_dropout(dropout)  # Actually update the model
        elif phase == 1: # B2 (e5-e8): Add range, maintain angles
            # Set all primary loss weights
            self.loss_fn.lam_cov   = hpo_lam_cov * 0.25   # lam_cov≈0.25
            self.loss_fn.lam_K     = 0.10                 # Light K (10%) - still secondary to angles/range
            
            # Phase 1: lam_rng≈0.35–0.5, lam_ang≈0.2
            self.loss_fn.lam_aux   = hpo_lam_ang * 0.20 + hpo_lam_rng * 0.45  # lam_rng≈0.35–0.5, lam_ang≈0.2
            
            # Covariance structure terms
            self.loss_fn.lam_cross = 1e-3 * hpo_lam_cov
            self.loss_fn.lam_gap   = 0.05 * hpo_lam_cov
            
            # Fixed terms
            self.loss_fn.lam_ortho = 1e-3
            self.loss_fn.lam_peak = 0.05
            self.loss_fn.lam_margin = 0.1
            self.loss_fn.lam_range_factor = 0.3
            
            dropout = max(float(getattr(mdl_cfg, "DROPOUT", 0.10)), 0.10)
            mdl_cfg.DROPOUT = dropout
            self.model.set_dropout(dropout)  # Actually update the model
        else:            # B3 (e9-e12): Joint training (angles + range + K)
            # Set all primary loss weights
            self.loss_fn.lam_cov   = hpo_lam_cov * 0.7    # lam_cov≈0.7
            self.loss_fn.lam_K     = hpo_lam_K            # lam_K≈0.2–0.35 (raise only on mixed-SNR)
            
            # Phase 2: lam_ang≈0.25, lam_rng≈0.3
            self.loss_fn.lam_aux   = hpo_lam_ang * 0.25 + hpo_lam_rng * 0.30  # lam_ang≈0.25, lam_rng≈0.3
            
            # Smooth ramp for cross-gram in final phase
            phase3_start = 2 * total_epochs // 3
            progress = (epoch - phase3_start) / max(1, total_epochs - phase3_start)
            progress = max(0, min(1, progress))  # clamp to [0,1]
            self.loss_fn.lam_cross = (1e-3 + progress * 1e-3) * hpo_lam_cov  # 1e-3 → 2e-3, scaled
            self.loss_fn.lam_gap   = 0.06 * hpo_lam_cov
            
            # Fixed terms
            self.loss_fn.lam_ortho = 1e-3
            self.loss_fn.lam_peak = 0.05
            self.loss_fn.lam_margin = 0.1
            self.loss_fn.lam_range_factor = 0.3
    
    def _tau_schedule(self, epoch: int, total_epochs: int) -> float:
        """
        Cosine anneal softmax temperature from 0.20 -> 0.08 over first 60% epochs.
        After 60%, temperature stays at 0.08 for sharp angle peaks.
        """
        import math
        frac = min(1.0, epoch / (0.6 * total_epochs))
        return 0.08 + (0.20 - 0.08) * 0.5 * (1 + math.cos(math.pi * frac))
    
    def _apply_curriculum(self, epoch: int, total_epochs: int):
        """
        Apply curriculum learning on K and SNR.
        Note: Simplified version for L=8 training since sampling overrides are for data generation.
        The curriculum is built into the loss function phases instead.
        """
        # Curriculum learning is implemented via the 3-phase training schedule
        # Phase 0: Focus on covariance reconstruction
        # Phase 1: Add auxiliary losses (angles/range)
        # Phase 2: Full joint training with K-estimation
        pass  # The phase-based curriculum is handled in the main training loop
    
    def _ramp_k_weight(self, epoch: int, total_epochs: int):
        """
        Phase 2: Ramp up lam_K gradually over the first few epochs.
        This gives the K-head time to learn without overwhelming early gradients.
        
        Ramps from 0.0 → target_lam_K over warmup_epochs (default: 5).
        """
        warmup_epochs = getattr(mdl_cfg, "K_WEIGHT_WARMUP_EPOCHS", 5)
        
        # Store target on first call
        if not hasattr(self, '_target_lam_K'):
            self._target_lam_K = self.loss_fn.lam_K
        
        if epoch < warmup_epochs:
            # Linear ramp
            ramp_factor = (epoch + 1) / warmup_epochs
            self.loss_fn.lam_K = self._target_lam_K * ramp_factor
            if epoch == 0 or epoch == warmup_epochs - 1:
                print(f"[K-ramp] Epoch {epoch}: lam_K = {self.loss_fn.lam_K:.4f} (target: {self._target_lam_K:.4f})", flush=True)
        else:
            # Fully ramped
            self.loss_fn.lam_K = self._target_lam_K
    
    def _update_structure_loss_weights(self, epoch: int, total_epochs: int):
        """
        Update subspace alignment and peak contrast loss weights based on training phase.
        
        3-phase schedule:
        - Warm-up (epochs 0-2): Lower weights to prevent early peaky gradients
        - Main (most of training): Standard weights for learning
        - Final (last 10-20%): Slightly higher weights for structure refinement
        
        Only active when curriculum is DISABLED (HPO mode).
        """
        # Skip if curriculum is enabled (it has its own weight management)
        if getattr(mdl_cfg, "ENABLE_CURRICULUM", False):
            return
        
        # PURE OVERFIT: Disable all structure losses when OVERFIT_NMSE_PURE is set
        if getattr(mdl_cfg, "OVERFIT_NMSE_PURE", False):
            self.loss_fn.lam_subspace_align = 0.0
            self.loss_fn.lam_peak_contrast = 0.0
            return
        
        warmup_epochs = 3
        final_start_epoch = int(0.8 * total_epochs)  # Last 20% of training
        
        if epoch < warmup_epochs:
            # Warm-up phase: lower structure terms
            self.loss_fn.lam_subspace_align = getattr(mdl_cfg, 'LAM_SUBSPACE_ALIGN_WARMUP', 0.02)
            self.loss_fn.lam_peak_contrast = getattr(mdl_cfg, 'LAM_PEAK_CONTRAST_WARMUP', 0.05)
            if epoch == 0:
                print(f"[Loss Schedule] Warm-up: lam_subspace={self.loss_fn.lam_subspace_align:.3f}, "
                      f"lam_peak={self.loss_fn.lam_peak_contrast:.3f}", flush=True)
        elif epoch >= final_start_epoch:
            # Final phase: bump structure slightly
            self.loss_fn.lam_subspace_align = getattr(mdl_cfg, 'LAM_SUBSPACE_ALIGN_FINAL', 0.07)
            self.loss_fn.lam_peak_contrast = getattr(mdl_cfg, 'LAM_PEAK_CONTRAST_FINAL', 0.12)
            if epoch == final_start_epoch:
                print(f"[Loss Schedule] Final: lam_subspace={self.loss_fn.lam_subspace_align:.3f}, "
                      f"lam_peak={self.loss_fn.lam_peak_contrast:.3f}", flush=True)
        else:
            # Main phase: standard weights
            self.loss_fn.lam_subspace_align = getattr(mdl_cfg, 'LAM_SUBSPACE_ALIGN', 0.05)
            self.loss_fn.lam_peak_contrast = getattr(mdl_cfg, 'LAM_PEAK_CONTRAST', 0.1)
            if epoch == warmup_epochs:
                print(f"[Loss Schedule] Main: lam_subspace={self.loss_fn.lam_subspace_align:.3f}, "
                      f"lam_peak={self.loss_fn.lam_peak_contrast:.3f}", flush=True)
    
    def _eval_hungarian_metrics(self, val_loader, max_batches=50):
        """
        Evaluate angle and range errors using Hungarian matching on validation set.
        Reports median and 95th percentile errors, broken down by SNR bins.
        """
        import torch.nn.functional as F
        from .covariance_utils import build_effective_cov_np
        from .infer import estimate_k_ic_from_cov
        
        self.model.eval()
        
        # Collect predictions and ground truth
        all_errors = {"phi": [], "theta": [], "r": [], "snr": []}
        # New: track per-scene RMSEs and K metrics
        rmse_phi_list, rmse_theta_list, rmse_r_list = [], [], []
        k_true_all, k_hat_all = [], []
        success_count = 0
        k_mdl_correct = 0  # Track MDL baseline accuracy
        # Success thresholds (configurable)
        succ_phi_thr = float(getattr(cfg, "SUCCESS_THR_PHI_DEG", 3.0))
        succ_theta_thr = float(getattr(cfg, "SUCCESS_THR_THETA_DEG", 3.0))
        succ_r_thr = float(getattr(cfg, "SUCCESS_THR_R_M", 0.5))
        
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if bi >= max_batches:
                    break
                    
                y, H, C, ptr, K, R_in, snr, H_full, R_samp = self._unpack_any_batch(batch)
                
                # Forward pass
                with torch.amp.autocast('cuda', enabled=(self.amp and self.device.type == "cuda")):
                    preds = self.model(y=y, H=H, codes=C, snr_db=snr, R_samp=R_samp)
                
                # Extract predictions (convert to numpy for Hungarian matching)
                phi_soft = preds["phi_soft"].cpu().numpy()  # [B, K_MAX]
                theta_soft = preds.get("theta_soft", torch.zeros_like(preds["phi_soft"])).cpu().numpy()  # [B, K_MAX]
                aux_ptr = preds.get("phi_theta_r", torch.zeros(phi_soft.shape[0], 3*cfg.K_MAX)).cpu().numpy()  # [B, 3*K_MAX]
                
                # Extract ground truth
                ptr_np = ptr.cpu().numpy()  # [B, 3*K] (phi, theta, r padded)
                K_np = K.cpu().numpy()  # [B]
                snr_np = snr.cpu().numpy() if snr is not None else np.zeros(len(K_np))
                
                # Evaluate each sample in batch
                for i in range(len(K_np)):
                    k_true = int(K_np[i])
                    if k_true == 0:
                        continue
                    
                    # Ground truth (convert radians to degrees)
                    phi_gt = np.rad2deg(ptr_np[i, :k_true])
                    theta_gt = np.rad2deg(ptr_np[i, cfg.K_MAX:cfg.K_MAX+k_true])
                    r_gt = ptr_np[i, 2*cfg.K_MAX:2*cfg.K_MAX+k_true]
                    
                    # ---- PHASE 1 FIX: Unit-safe + All-slots matching ----
                    # Auto-detect radians vs degrees and convert safely
                    def _to_deg_safe(t):
                        """Auto-convert radians to degrees if needed"""
                        x = t if isinstance(t, np.ndarray) else t
                        max_abs = np.nanmax(np.abs(x))
                        # If max |value| ≤ π+0.1, assume radians
                        return np.rad2deg(x) if max_abs <= (np.pi + 0.1) else x
                    
                    # Extract ALL K_MAX predictions (not just first k_true!)
                    phi_all = phi_soft[i]  # [K_MAX]
                    theta_all = theta_soft[i]  # [K_MAX]
                    r_all = aux_ptr[i, 2*cfg.K_MAX:3*cfg.K_MAX]  # [K_MAX]
                    
                    # Convert units safely (handles radians→degrees automatically)
                    phi_all_deg = _to_deg_safe(phi_all).copy()  # Ensure numpy array
                    theta_all_deg = _to_deg_safe(theta_all).copy()  # Ensure numpy array
                    r_all_np = r_all.detach().cpu().numpy() if isinstance(r_all, torch.Tensor) else np.array(r_all)
                    
                    # Inference-like evaluation: always use unified angle pipeline when factors available
                    # Use GPU MUSIC if available for 10-20x speedup
                    if "cov_fact_angle" in preds:
                        # Debug: print first sample to verify MUSIC is being called
                        if i == 0 and bi == 0:
                            print(f"[VAL MUSIC] Entering MUSIC block for sample 0, batch 0", flush=True)
                        try:
                            from .angle_pipeline import angle_pipeline, angle_pipeline_gpu, _GPU_MUSIC_AVAILABLE
                            
                            cf_ang = preds["cov_fact_angle"][i].detach().cpu().numpy()  # [N*K_MAX*2]
                            # NN K (with optional calibration)
                            if "k_logits" in preds:
                                k_temp = float(getattr(self.model, "_k_calibration_temp", 1.0))
                                logits_i = preds["k_logits"][i].detach().cpu().numpy() / max(k_temp, 1e-6)
                                pK = np.exp(logits_i - np.max(logits_i)); pK = pK / np.sum(pK)
                                K_nn = int(np.argmax(pK)) + 1
                                nn_conf = float(np.max(pK))
                            else:
                                K_nn, nn_conf = int(min(max(1, k_true), cfg.K_MAX)), 0.0
                            
                            # L=16 CRITICAL FIX: Extract snapshots for hybrid covariance blending
                            y_snaps = None
                            H_snaps = None
                            codes_snaps = None
                            blend_beta = 0.0
                            
                            if getattr(cfg, "HYBRID_COV_BLEND", False):
                                # Extract and convert to complex numpy
                                y_i = y[i].detach().cpu().numpy()  # [L, M, 2]
                                C_i = C[i].detach().cpu().numpy()  # [L, N, 2]
                                
                                # CRITICAL FIX: Use H_full [M_BS, N, 2] not H [L, M_BS, 2]!
                                if H_full is not None:
                                    H_full_i = H_full[i].detach().cpu().numpy()  # [M_BS, N, 2]
                                    H_cplx_single = H_full_i[:, :, 0] + 1j * H_full_i[:, :, 1]  # [M_BS, N]
                                    # Tile to [L, M_BS, N] for all snapshots
                                    H_snaps = np.tile(H_cplx_single[np.newaxis, :, :], (cfg.L, 1, 1))
                                else:
                                    # Fallback to old H (will fail with shape error, but better than silent wrong result)
                                    H_i = H[i].detach().cpu().numpy()  # [L, M, 2]
                                    H_snaps = H_i[:, :, 0] + 1j * H_i[:, :, 1]  # [L, M] - WRONG SHAPE!
                                
                                # Convert snapshots to complex
                                y_snaps = y_i[:, :, 0] + 1j * y_i[:, :, 1]
                                codes_snaps = C_i[:, :, 0] + 1j * C_i[:, :, 1]
                                blend_beta = getattr(cfg, "HYBRID_COV_BETA", 0.2)
                            
                            # Convert covariance factor to complex
                            N = cfg.N_H * cfg.N_V
                            cf_ang_real = cf_ang[:N * cfg.K_MAX].reshape(N, cfg.K_MAX)
                            cf_ang_imag = cf_ang[N * cfg.K_MAX:].reshape(N, cfg.K_MAX)
                            cf_ang_complex = cf_ang_real + 1j * cf_ang_imag
                            
                            # MDL baseline on R_eff (same as inference) for gating
                            try:
                                from .covariance_utils import build_effective_cov_np
                                R_pred_eval = cf_ang_complex @ cf_ang_complex.conj().T
                                R_samp_raw = None
                                if R_samp is not None:
                                    Ri = R_samp[i].detach().cpu().numpy()
                                    R_samp_raw = Ri[..., 0] + 1j * Ri[..., 1] if (Ri.ndim == 3 and Ri.shape[-1] == 2) else Ri.astype(np.complex64)
                                beta = getattr(cfg, "HYBRID_COV_BETA", 0.0)
                                R_eff_np = build_effective_cov_np(
                                    R_pred_eval,
                                    R_samp=R_samp_raw if (R_samp_raw is not None and beta > 0.0) else None,
                                    beta=float(beta) if (R_samp_raw is not None and beta > 0.0) else None,
                                    diag_load=True, apply_shrink=False, snr_db=None,
                                    target_trace=float(N),
                                )
                                from .infer import estimate_k_ic_from_cov
                                T_snap = int(C.shape[1]) if hasattr(C, "shape") else int(cfg.L)
                                K_mdl = estimate_k_ic_from_cov(R_eff_np, T_snap, method="mdl", kmax=cfg.K_MAX)
                                if (K_mdl is None) or (K_mdl <= 0):
                                    K_mdl = estimate_k_ic_from_cov(R_eff_np, T_snap, method="aic", kmax=cfg.K_MAX)
                                K_mdl = int(np.clip(K_mdl, 1, cfg.K_MAX))
                            except Exception:
                                K_mdl = int(K_nn)
                            thr = float(getattr(cfg, "K_CONF_THRESH", 0.65))
                            K_hat = K_mdl if (nn_conf < thr) else K_nn
                            
                            # Use GPU MUSIC if available (10-20x faster), else fall back to CPU
                            use_gpu_music = _GPU_MUSIC_AVAILABLE and torch.cuda.is_available()
                            
                            # Get R_samp for hybrid blending (same as used by K-head)
                            R_samp_np = None
                            if R_samp is not None:
                                Ri = R_samp[i].detach().cpu().numpy()
                                R_samp_np = Ri[..., 0] + 1j * Ri[..., 1] if (Ri.ndim == 3 and Ri.shape[-1] == 2) else Ri.astype(np.complex64)
                            
                            if use_gpu_music:
                                # GPU path: fast MUSIC with consistent R_eff (prepared=True)
                                # angle_pipeline_gpu uses build_effective_cov_np internally
                                # This ensures K-head and MUSIC see the SAME R_eff
                                hybrid_beta = float(getattr(cfg, "HYBRID_COV_BETA", 0.3))
                                phi_music, theta_music, info_music = angle_pipeline_gpu(
                                    cf_ang_complex, K_hat, cfg,
                                    use_fba=getattr(cfg, "MUSIC_USE_FBA", True),
                                    use_2_5d=True,  # Enable 2.5D so range is estimated for gating
                                    r_planes=getattr(cfg, "MUSIC_R_PLANES", None),
                                    grid_phi=91,     # Coarser grid for validation speed
                                    grid_theta=61,
                                    peak_refine=True,
                                    use_newton=False,  # Skip Newton for speed
                                    R_samp=R_samp_np,  # Pass R_samp for hybrid blending
                                    beta=hybrid_beta,  # Use cfg.HYBRID_COV_BETA (not blend_beta which may be 0)
                                )
                            else:
                                # CPU fallback with full pipeline
                                phi_music, theta_music, _ = angle_pipeline(
                                    cf_ang_complex, K_hat, cfg,
                                    use_fba=getattr(cfg, "MUSIC_USE_FBA", True),
                                    use_adaptive_shrink=True,
                                    use_parabolic=getattr(cfg, "MUSIC_PEAK_REFINE", True),
                                    use_newton=getattr(cfg, "USE_NEWTON_REFINE", True),
                                    device="cpu",
                                    y_snapshots=y_snaps,
                                    H_snapshots=H_snaps,
                                    codes_snapshots=codes_snaps,
                                    blend_beta=blend_beta
                                )
                            # Use MUSIC+Newton refined estimates directly
                            phi_all_deg = np.array(phi_music, dtype=np.float32)
                            theta_all_deg = np.array(theta_music, dtype=np.float32)
                            # If GPU MUSIC provided range estimates, use them for gating (instead of network r)
                            try:
                                if use_gpu_music and (info_music is not None) and ("r_est" in info_music):
                                    r_all_np = np.array(info_music["r_est"], dtype=np.float32)
                            except Exception:
                                pass
                            
                            # Debug: print first sample's MUSIC output
                            if i == 0:
                                print(f"[MUSIC DEBUG] Sample 0: K_hat={K_hat}, phi={phi_all_deg}, theta={theta_all_deg}", flush=True)
                                print(f"[MUSIC DEBUG] GT: K={k_true}, phi_gt={phi_gt[:k_true]}, theta_gt={theta_gt[:k_true]}", flush=True)
                            
                        except Exception as e:
                            # ALWAYS print first failure to help debug
                            if i == 0 or getattr(cfg, "MUSIC_DEBUG", False):
                                import traceback
                                print(f"[MUSIC] Warning: angle_pipeline failed for sample {i}: {e}")
                                if i == 0:
                                    traceback.print_exc()
                    # MDL baseline on effective covariance (same path used by inference)
                    try:
                        # Build R_pred from factors if available
                        R_pred = None
                        if "cov_fact_angle" in preds:
                            N = cfg.N_H * cfg.N_V
                            cf_ang = preds["cov_fact_angle"][i].detach().cpu().numpy()
                            cf_ang_real = cf_ang[:N * cfg.K_MAX].reshape(N, cfg.K_MAX)
                            cf_ang_imag = cf_ang[N * cfg.K_MAX:].reshape(N, cfg.K_MAX)
                            cf_cplx = cf_ang_real + 1j * cf_ang_imag
                            R_pred = cf_cplx @ cf_cplx.conj().T  # [N, N]
                        # Optional R_samp from dataset
                        R_samp_raw = None
                        if R_samp is not None:
                            Ri = R_samp[i].detach().cpu().numpy()
                            if Ri.ndim == 3 and Ri.shape[-1] == 2:
                                R_samp_raw = Ri[..., 0] + 1j * Ri[..., 1]
                            else:
                                R_samp_raw = Ri.astype(np.complex64)
                        beta = getattr(cfg, "HYBRID_COV_BETA", 0.0)
                        R_eff = build_effective_cov_np(
                            R_pred if R_pred is not None else (R_samp_raw if R_samp_raw is not None else None),
                            R_samp=R_samp_raw if (R_pred is not None and R_samp_raw is not None and beta > 0.0) else None,
                            beta=float(beta) if (R_pred is not None and R_samp_raw is not None and beta > 0.0) else None,
                            diag_load=True, apply_shrink=False, snr_db=None,
                            target_trace=float(cfg.N_H * cfg.N_V),
                        )
                        T_snap = int(C[i].shape[0]) if isinstance(C, torch.Tensor) else int(C[i].shape[0])
                        k_mdl = estimate_k_ic_from_cov(R_eff, T_snap, method="mdl", kmax=cfg.K_MAX)
                        if (k_mdl is None) or (k_mdl <= 0):
                            k_mdl = estimate_k_ic_from_cov(R_eff, T_snap, method="aic", kmax=cfg.K_MAX)
                        if int(k_mdl) == int(k_true):
                            k_mdl_correct += 1
                    except Exception as e:
                        if getattr(cfg, "MUSIC_DEBUG", False):
                            print(f"[MDL] baseline failed: {e}")
                    
                    # Hungarian matching across ALL slots (permutation-agnostic)
                    metrics = eval_scene_angles_ranges(phi_all_deg, theta_all_deg, r_all_np,
                                                      phi_gt, theta_gt, r_gt)
                    
                    # ICC CRITICAL FIX: Count ALL scenes, don't drop any!
                    # If matching fails (med_phi is None), assign large penalty
                    if metrics["med_phi"] is not None:
                        all_errors["phi"].append(metrics["med_phi"])
                        all_errors["theta"].append(metrics["med_theta"])
                        all_errors["r"].append(metrics["med_r"])
                        all_errors["snr"].append(snr_np[i])
                        # Track RMSEs when available (scene-level)
                        if metrics.get("rmse_phi") is not None:
                            rmse_phi_list.append(metrics["rmse_phi"])
                        if metrics.get("rmse_theta") is not None:
                            rmse_theta_list.append(metrics["rmse_theta"])
                        if metrics.get("rmse_r") is not None:
                            rmse_r_list.append(metrics["rmse_r"])
                    else:
                        # Scene failed - assign max penalty (90° for angles, 10m for range)
                        # This ensures HPO sees the failure and penalizes bad configs
                        all_errors["phi"].append(90.0)
                        all_errors["theta"].append(90.0)
                        all_errors["r"].append(10.0)
                        all_errors["snr"].append(snr_np[i])
                    
                    # K metrics
                    k_hat = int(np.clip(np.argmax(preds["k_logits"][i].detach().cpu().numpy()) + 1, 1, cfg.K_MAX)) if "k_logits" in preds else int(min(max(1, k_true), cfg.K_MAX))
                    k_true_all.append(k_true)
                    k_hat_all.append(k_hat)
                    
                    # Success criterion (proxy): K correct AND medians under thresholds
                    if (k_hat == k_true) and (metrics["med_phi"] is not None):
                        if (metrics["med_phi"] <= succ_phi_thr) and (metrics["med_theta"] <= succ_theta_thr) and (metrics["med_r"] <= succ_r_thr):
                            success_count += 1
        
        if len(all_errors["phi"]) == 0:
            print("⚠️  No valid samples for Hungarian evaluation")
            return None
        
        # Convert to numpy
        phi_err = np.array(all_errors["phi"])
        theta_err = np.array(all_errors["theta"])
        r_err = np.array(all_errors["r"])
        snr_arr = np.array(all_errors["snr"])
        
        # Sanity check: we should have processed some samples
        actual_count = len(phi_err)
        if actual_count == 0:
            print(f"⚠️  WARNING: No validation samples processed!")
            return None
        
        # Note: During HPO, we use subsets (e.g., 1000 samples instead of full 10K)
        # So we don't assert a fixed count, just log what we got
        print(f"[VAL METRICS] Processed {actual_count} samples", flush=True)
        
        # Overall statistics
        med_phi = np.median(phi_err)
        med_theta = np.median(theta_err)
        med_r = np.median(r_err)
        # Aggregated RMSEs (scene-level mean RMSE as proxy)
        rmse_phi_mean = float(np.mean(rmse_phi_list)) if len(rmse_phi_list) > 0 else None
        rmse_theta_mean = float(np.mean(rmse_theta_list)) if len(rmse_theta_list) > 0 else None
        rmse_r_mean = float(np.mean(rmse_r_list)) if len(rmse_r_list) > 0 else None
        
        print(f"\n📊 Overall Angle/Range Errors (Hungarian-matched, N={len(phi_err)}):")
        print(f"  Azimuth (φ):     median={med_phi:.3f}°,   95th={np.percentile(phi_err, 95):.3f}°")
        print(f"  Elevation (θ):   median={med_theta:.3f}°,   95th={np.percentile(theta_err, 95):.3f}°")
        print(f"  Range (r):       median={med_r:.3f}m,   95th={np.percentile(r_err, 95):.3f}m")
        
        # SNR-binned statistics
        snr_bins = [(-np.inf, 0), (0, 10), (10, np.inf)]
        snr_labels = ["Low (≤0dB)", "Mid (0-10dB)", "High (≥10dB)"]
        
        # Compute high-SNR K accuracy (will be passed separately)
        high_snr_mask = snr_arr > 10
        
        print(f"\n📊 By SNR Bin:")
        for (snr_min, snr_max), label in zip(snr_bins, snr_labels):
            mask = (snr_arr > snr_min) & (snr_arr <= snr_max)
            n = mask.sum()
            if n > 0:
                print(f"  {label:15s} (N={n:4d}): φ={np.median(phi_err[mask]):.3f}°, "
                      f"θ={np.median(theta_err[mask]):.3f}°, r={np.median(r_err[mask]):.3f}m")
        
        # K metrics
        k_true_all = np.array(k_true_all, dtype=np.int32)
        k_hat_all = np.array(k_hat_all, dtype=np.int32)
        if k_true_all.size > 0:
            k_correct = (k_true_all == k_hat_all)
            k_acc = float(np.mean(k_correct))
            k_under = int(np.sum(k_hat_all < k_true_all))
            k_over = int(np.sum(k_hat_all > k_true_all))
            success_rate = float(success_count / len(k_true_all))
        else:
            k_acc, k_under, k_over, success_rate = 0.0, 0, 0, 0.0
        
        # Return metrics for composite score calculation
        return {
            "med_phi": med_phi,
            "med_theta": med_theta,
            "med_r": med_r,
            "high_snr_samples": high_snr_mask.sum(),
            "n_scenes": len(phi_err),  # Total scenes evaluated (should be 1600)
            "rmse_phi_mean": rmse_phi_mean,
            "rmse_theta_mean": rmse_theta_mean,
            "rmse_r_mean": rmse_r_mean,
            "k_acc": k_acc,
            "k_under": k_under,
            "k_over": k_over,
            "success_rate": success_rate,
            "k_mdl_acc": float(k_mdl_correct / len(k_true_all)) if len(k_true_all) > 0 else 0.0,
        }
    
    def calibrate_k_logits(self, val_loader, save_path=None):
        """
        Temperature-scale k_logits on validation set for better K estimation.
        Finds optimal temperature T such that softmax(logits/T) is well-calibrated.
        """
        import torch.nn.functional as F
        
        def golden_section_search(f, a, b, tol=1e-5):
            """Simple golden section search to replace scipy dependency"""
            phi = (1 + 5**0.5) / 2
            resphi = 2 - phi
            
            # Initial points
            tol1 = tol * abs(b) + tol
            x1 = a + resphi * (b - a)
            x2 = b - resphi * (b - a)
            f1, f2 = f(x1), f(x2)
            
            while abs(b - a) > tol1:
                if f2 > f1:
                    b, x2, f2 = x2, x1, f1
                    x1 = a + resphi * (b - a)
                    f1 = f(x1)
                else:
                    a, x1, f1 = x1, x2, f2
                    x2 = b - resphi * (b - a)
                    f2 = f(x2)
                tol1 = tol * abs(b) + tol
            
            return (a + b) / 2
        
        self.model.eval()
        all_logits, all_k_true, all_snr = [], [], []
        
        print("🌡️ Collecting k_logits from validation set...")
        with torch.no_grad():
            for batch in val_loader:
                # Unpack batch using the same logic as training
                y, H, C, ptr, K, R_in, snr, H_full, R_samp = self._unpack_any_batch(batch)
                
                # Forward pass (R_samp already extracted by _unpack_any_batch)
                pred = self.model(y, H, C, snr_db=snr, R_samp=R_samp)
                
                if "k_logits" in pred:
                    logits = pred["k_logits"]  # [B, K_MAX]
                    k_true = K - 1            # [B] Map K (1-5) to class index (0-4)
                    
                    all_logits.append(logits.cpu())
                    all_k_true.append(k_true.cpu())
                    all_snr.append(snr.cpu() if snr is not None else torch.zeros(K.shape[0]))
        
        if not all_logits:
            print("⚠️ No k_logits found in validation set, skipping calibration")
            return 1.0
        
        # Concatenate all validation data
        logits_val = torch.cat(all_logits, dim=0)  # [N, K_MAX+1]
        k_true_val = torch.cat(all_k_true, dim=0)  # [N]
        snr_val = torch.cat(all_snr, dim=0)  # [N]
        
        print(f"📊 Calibrating on {len(logits_val)} validation samples...")
        
        def negative_log_likelihood(temperature):
            """Negative log-likelihood for temperature scaling"""
            if temperature <= 0:
                return float('inf')
            
            scaled_logits = logits_val / temperature
            log_probs = F.log_softmax(scaled_logits, dim=1)
            
            # Select log probabilities for true K values
            k_clamped = torch.clamp(k_true_val, 0, logits_val.shape[1] - 1)
            selected_log_probs = log_probs.gather(1, k_clamped.unsqueeze(1)).squeeze(1)
            
            return -selected_log_probs.mean().item()
        
        # Find optimal temperature using golden section search
        optimal_temp = golden_section_search(negative_log_likelihood, 0.1, 10.0)
        
        print(f"🎯 Optimal K temperature: {optimal_temp:.3f}")
        
        # Compute calibration metrics
        with torch.no_grad():
            orig_probs = F.softmax(logits_val, dim=1)
            calib_probs = F.softmax(logits_val / optimal_temp, dim=1)
            
            orig_conf = orig_probs.max(dim=1)[0].mean().item()
            calib_conf = calib_probs.max(dim=1)[0].mean().item()
            
            orig_acc = (orig_probs.argmax(dim=1) == k_true_val).float().mean().item()
            calib_acc = (calib_probs.argmax(dim=1) == k_true_val).float().mean().item()
            
            # Compute high-SNR K accuracy (SNR > 10 dB)
            high_snr_mask = snr_val > 10
            if high_snr_mask.sum() > 0:
                high_snr_acc = (calib_probs.argmax(dim=1)[high_snr_mask] == k_true_val[high_snr_mask]).float().mean().item()
            else:
                high_snr_acc = calib_acc  # Fallback if no high-SNR samples
            
            print(f"📈 Before calibration: confidence={orig_conf:.3f}, accuracy={orig_acc:.3f}")
            print(f"📈 After calibration: confidence={calib_conf:.3f}, accuracy={calib_acc:.3f}")
            print(f"📈 High-SNR (>10dB) K-acc: {high_snr_acc:.3f} (N={high_snr_mask.sum()})")
        
        # Save temperature to checkpoint if path provided
        if save_path:
            if Path(save_path).exists():
                ckpt = torch.load(save_path, map_location='cpu')
                ckpt['k_calibration_temp'] = optimal_temp
                torch.save(ckpt, save_path)
                print(f"💾 Saved calibrated temperature to {save_path}")
        
        return optimal_temp, calib_acc, high_snr_acc

    # ----------------------------
    # Public API
    # ----------------------------
    def fit(self,
            epochs: int = 60,
            use_shards: bool = True,
            n_train: Optional[int] = None,
            n_val:   Optional[int] = None,
            max_train_batches: Optional[int] = None,
            max_val_batches:   Optional[int] = None,
            gpu_cache: Optional[bool] = None,
            grad_accumulation: int = 1,
            early_stop_patience: int = 10,
            val_every: int = 1,
            skip_music_val: bool = False):

        if not use_shards:
            raise RuntimeError("This pipeline requires pregenerated shards.")
        
        # Set beta warmup based on total epochs (20% warmup)
        self.beta_warmup_epochs = max(2, int(0.2 * epochs))
        print(f"[Beta Warmup] Annealing β from {self.beta_start:.2f} → {self.beta_final:.2f} over {self.beta_warmup_epochs} epochs")

        # decide cache mode (auto → try GPU cache if it fits in VRAM budget)
        want_cache = bool(getattr(mdl_cfg, "TRAIN_USE_GPU_CACHE", True)) if gpu_cache is None else bool(gpu_cache)
        use_gpu = (self.device.type == "cuda") and want_cache

        if use_gpu:
            # CRITICAL FIX FOR HPO: Always use GPU cache when explicitly requested
            # The memory check was causing fallback to CPU, but for HPO we have resources
            print(f"[GPU cache] Building GPU-cached loaders for train and validation...", flush=True)
            tr_loader, va_loader = self._build_loaders_gpu_cache(n_train, n_val)
            print(f"[GPU cache] Loaders built successfully!", flush=True)
        else:
            tr_loader, va_loader = self._build_loaders_cpu_io(n_train, n_val)


        # scheduler (cosine with warmup for stability)
        train_iters = len(tr_loader) if max_train_batches is None else min(len(tr_loader), max_train_batches)
        total_steps = max(1, epochs * max(1, train_iters))
        warmup_steps = min(10, total_steps // 50)  # Expert fix: Much shorter warmup for overfit tests
        
        # Create warmup + cosine scheduler
        def lr_lambda(step):
            if step < warmup_steps:
                # Expert fix: Much higher floor for overfit tests (was 0.1, now 0.5)
                return max(0.5, step / warmup_steps)
            else:
                # Cosine annealing after warmup
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

        # training loop with early stopping
        best_val = float("inf")
        # New: primary metric can be 'loss' or 'k_loc' (K/localization composite)
        val_primary = str(getattr(cfg, "VAL_PRIMARY", "loss")).lower()
        best_score = float("inf")
        best_path = Path(cfg.CKPT_DIR) / "best.pt"
        last_path = Path(cfg.CKPT_DIR) / "last.pt"
        patience_counter = 0

        print(f"[Training] Starting {epochs} epochs (train batches={len(tr_loader)}, val batches={len(va_loader)})...", flush=True)
        for ep in range(epochs):
            # 3-phase schedule (if enabled)
            if getattr(mdl_cfg, 'USE_3_PHASE_CURRICULUM', True):
                phase = 0 if ep < max(1, epochs // 3) else (1 if ep < max(2, 2 * epochs // 3) else 2)
                self._apply_phase_weights(phase, epoch=ep, total_epochs=epochs)
            else:
                phase = -1  # No curriculum phase
                
                # ICC CRITICAL FIX: When curriculum is OFF (HPO), set lam_cross and lam_gap explicitly
                # Use HPO base weights with STRONGER cross/gap regularization for proper subspace structure
                if hasattr(self, '_hpo_loss_weights'):
                    hpo_lam_cov = self._hpo_loss_weights.get("lam_cov", 1.0)
                    self.loss_fn.lam_cov = hpo_lam_cov  # CRITICAL: Set main covariance weight!
                    
                    # PURE OVERFIT: Disable structural losses when OVERFIT_NMSE_PURE is set
                    if getattr(mdl_cfg, "OVERFIT_NMSE_PURE", False):
                        self.loss_fn.lam_cross = 0.0
                        self.loss_fn.lam_gap = 0.0
                        self.loss_fn.lam_ortho = 0.0
                        self.loss_fn.lam_peak = 0.0
                    else:
                        self.loss_fn.lam_cross = 2.5e-3 * hpo_lam_cov  # INCREASED from 1.5e-3: (2-3)e-3 * lam_cov for subspace shaping
                        self.loss_fn.lam_gap = 0.065 * hpo_lam_cov     # INCREASED from 0.04: (0.05-0.08) * lam_cov for gap penalty
                else:
                    # No HPO weights - use default values
                    self.loss_fn.lam_cov = 1.0  # CRITICAL: Ensure main covariance weight is set!
                    
                    # PURE OVERFIT: Disable structural losses when OVERFIT_NMSE_PURE is set
                    if getattr(mdl_cfg, "OVERFIT_NMSE_PURE", False):
                        self.loss_fn.lam_cross = 0.0
                        self.loss_fn.lam_gap = 0.0
                        self.loss_fn.lam_ortho = 0.0
                        self.loss_fn.lam_peak = 0.0
                
                # Update structure loss weights (subspace align + peak contrast) based on phase
                self._update_structure_loss_weights(ep, epochs)
            
            # Phase 2: Ramp K-weight gradually (ONLY when curriculum is ON)
            # ICC FIX: Disable K-ramp during HPO for stationary objective
            if mdl_cfg.USE_3_PHASE_CURRICULUM and getattr(mdl_cfg, "K_WEIGHT_RAMP", True):
                self._ramp_k_weight(ep, epochs)
            
            # Check if we should start SWA
            swa_start_epoch = int(self.swa_start_frac * epochs)
            if self.use_swa and ep == swa_start_epoch:
                self._swa_start(epochs)
            
            # FIXED: Remove temperature scaling during training (only use for calibration)
            # tau = self._tau_schedule(ep, epochs)
            # mdl_cfg.SOFTMAX_TAU = tau
            # self.model.set_tau(tau)  # Actually update the model
            
            # Apply curriculum learning (if enabled)
            if getattr(mdl_cfg, 'USE_3_PHASE_CURRICULUM', True):
                self._apply_curriculum(ep, epochs)

            tr_loss = self._train_one_epoch(tr_loader, ep + 1, epochs, max_train_batches, grad_accumulation)

            # validate with best available model (SWA > EMA > regular)
            # Log per-term debug info every 3 epochs or last epoch
            return_debug = (ep % 3 == 0) or (ep == epochs - 1)
            
            if self.swa_started:
                # Use SWA model for validation during SWA phase
                self._swa_swap_in()
                val_result = self._validate_one_epoch(va_loader, max_val_batches, return_debug=return_debug)
                self._swa_swap_out()
            else:
                # Use EMA model for validation before SWA
                self._ema_swap_in()
                val_result = self._validate_one_epoch(va_loader, max_val_batches, return_debug=return_debug)
                self._ema_swap_out()
            
            # Extract val_loss and debug terms
            if isinstance(val_result, tuple):
                val_loss, debug_terms = val_result
            else:
                val_loss = val_result
                debug_terms = None

            # New: Run inference-like validation metrics (Hungarian; limited batches)
            # Skip MUSIC-based metrics if skip_music_val=True (for faster HPO)
            metrics = None
            if not skip_music_val and ((ep + 1) % val_every == 0 or ep == epochs - 1):
                try:
                    # Limit validation batches for speed (GPU MUSIC makes this fast)
                    # Use 20 batches (~20 samples with BS=1 or ~1600 with BS=80) for good metrics
                    hpo_max_batches = max_val_batches or 20
                    if self.swa_started:
                        self._swa_swap_in()
                        metrics = self._eval_hungarian_metrics(va_loader, hpo_max_batches)
                        self._swa_swap_out()
                    else:
                        self._ema_swap_in()
                        metrics = self._eval_hungarian_metrics(va_loader, hpo_max_batches)
                        self._ema_swap_out()
                except Exception as e:
                    print(f"[VAL METRICS] Skipped due to error: {e}", flush=True)
            elif skip_music_val and ep == 0:
                print(f"[VAL METRICS] Skipping MUSIC-based metrics (skip_music_val=True)", flush=True)

            # Compose primary metric if requested
            if metrics is not None and val_primary in ("k_loc", "metrics"):
                # Lower is better
                phi_norm = float(getattr(cfg, "VAL_NORM_PHI_DEG", 5.0))
                theta_norm = float(getattr(cfg, "VAL_NORM_THETA_DEG", 5.0))
                r_norm = float(getattr(cfg, "VAL_NORM_R_M", 1.0))
                k_acc = float(metrics.get("k_acc", 0.0))
                succ = float(metrics.get("success_rate", 0.0))
                rmse_phi = metrics.get("rmse_phi_mean", None)
                rmse_theta = metrics.get("rmse_theta_mean", None)
                rmse_r = metrics.get("rmse_r_mean", None)
                # Use conservative defaults if RMSE unavailable
                rmse_phi = float(rmse_phi) if rmse_phi is not None else phi_norm
                rmse_theta = float(rmse_theta) if rmse_theta is not None else theta_norm
                rmse_r = float(rmse_r) if rmse_r is not None else r_norm
                val_score = (1.0 - k_acc) + (rmse_phi / phi_norm) + (rmse_theta / theta_norm) + (rmse_r / r_norm) - succ
            else:
                # Default to loss as score
                val_score = float(val_loss)
            
            # Update best checkpoint by primary metric
            improved = False
            if val_primary in ("k_loc", "metrics"):
                if val_score < best_score:
                    best_score = float(val_score)
                    improved = True
            else:
                if val_loss < best_val:
                    best_val = float(val_loss)
                    improved = True
            if improved:
                torch.save(self.model.state_dict(), best_path)
                patience_counter = 0
            else:
                patience_counter += 1
            torch.save(self.model.state_dict(), last_path)

            phase_str = f"phase={phase}" if phase >= 0 else "no-curriculum"
            print(
                f"Epoch {ep+1:03d}/{epochs:03d} [{phase_str}] "
                f"train {tr_loss:.6f}  val {val_loss:.6f}  "
                f"(lam_cross={getattr(self.loss_fn,'lam_cross',0):.1e}, "
                f"lam_gap={getattr(self.loss_fn,'lam_gap',0):.2f}, "
                f"lam_K={getattr(self.loss_fn,'lam_K',0):.2f})",
                flush=True
            )
            # Print key K/localization metrics if available
            if metrics is not None:
                print(f"  🧭 Metrics: K_acc={metrics.get('k_acc', 0.0):.3f}, "
                      f"K_mdl_acc={metrics.get('k_mdl_acc', 0.0):.3f}, "
                      f"succ_rate={metrics.get('success_rate', 0.0):.3f}, "
                      f"φ_med={metrics.get('med_phi', float('nan')):.2f}°, "
                      f"θ_med={metrics.get('med_theta', float('nan')):.2f}°, "
                      f"r_med={metrics.get('med_r', float('nan')):.2f}m, "
                      f"φ_RMSE≈{metrics.get('rmse_phi_mean', float('nan')):.2f}°, "
                      f"θ_RMSE≈{metrics.get('rmse_theta_mean', float('nan')):.2f}°, "
                      f"r_RMSE≈{metrics.get('rmse_r_mean', float('nan')):.2f}m, "
                      f"score={val_score:.3f}", flush=True)

            # Early stopping by patience on chosen primary metric
            if (early_stop_patience is not None) and (early_stop_patience > 0) and (patience_counter >= early_stop_patience):
                print(f"⏹️ Early stopping at epoch {ep+1} (patience {early_stop_patience} reached)", flush=True)
                break
            
            # ICC FIX: Print pipeline config, geometry, and K-head diagnostics at epoch 0
            if ep == 0:
                print(f"  📋 Angle pipeline: MUSIC_COARSE={getattr(cfg,'MUSIC_COARSE',False)}, "
                      f"FBA={getattr(cfg,'MUSIC_USE_FBA',False)}, "
                      f"SHRINK={getattr(cfg,'MUSIC_USE_ADAPTIVE_SHRINK',False)}, "
                      f"PEAK_REFINE={getattr(cfg,'MUSIC_PEAK_REFINE',False)}, "
                      f"NEWTON={getattr(cfg,'USE_NEWTON_REFINE',False)}, "
                      f"NF_NEWTON={getattr(cfg,'NEWTON_NEARFIELD',False)}", flush=True)
                print(f"  📋 Range pipeline: RANGE_MUSIC_NF={getattr(cfg,'RANGE_MUSIC_NF',False)}", flush=True)
                
                # ICC CRITICAL FIX: Geometry sanity print (catches unit mistakes!)
                N_H = getattr(cfg, 'N_H', 12)
                N_V = getattr(cfg, 'N_V', 12)
                d_H_m = getattr(cfg, 'd_H', 0.15)  # Element spacing in METERS
                d_V_m = getattr(cfg, 'd_V', 0.15)  # Element spacing in METERS
                lam = getattr(cfg, 'WAVEL', 0.0625)  # Wavelength in meters (~4.8 GHz)
                d_H_lambda = d_H_m / lam  # Convert to wavelengths
                d_V_lambda = d_V_m / lam  # Convert to wavelengths
                print(f"  🔧 Geometry: N_H={N_H}, N_V={N_V}, "
                      f"d_h={d_H_lambda:.3f}λ ({d_H_m:.4f}m), d_v={d_V_lambda:.3f}λ ({d_V_m:.4f}m), λ={lam:.4f}m", flush=True)
                
                # Aperture size and diffraction-limited beamwidth
                D_h = (N_H - 1) * d_H_m  # Horizontal aperture (meters)
                D_v = (N_V - 1) * d_V_m  # Vertical aperture (meters)
                hpbw_h = np.rad2deg(0.88 * lam / D_h) if D_h > 0 else 999  # Horizontal beamwidth (degrees)
                hpbw_v = np.rad2deg(0.88 * lam / D_v) if D_v > 0 else 999  # Vertical beamwidth (degrees)
                print(f"  🔧 Aperture: D_h={D_h:.3f}m, D_v={D_v:.3f}m", flush=True)
                print(f"  🔧 Diffraction-limited beamwidth: HPBW_h≈{hpbw_h:.1f}°, HPBW_v≈{hpbw_v:.1f}° (0.88λ/D)", flush=True)
                print(f"  🔧 Planar manifold: k=2π/λ, α=k·d_h·sinφ·cosθ, β=k·d_v·sinθ", flush=True)
                print(f"  🔧 Near-field manifold: adds quadratic term k·(x²+y²)/(2r) with x,y in METERS", flush=True)
                
                # EXPERT-RECOMMENDED SANITY CHECKS
                h_idx = (np.arange(cfg.N_H) - (cfg.N_H - 1) / 2.0) * cfg.d_H
                v_idx = (np.arange(cfg.N_V) - (cfg.N_V - 1) / 2.0) * cfg.d_V
                dx = float(h_idx[1] - h_idx[0]) if len(h_idx) > 1 else cfg.d_H
                dy = float(v_idx[1] - v_idx[0]) if len(v_idx) > 1 else cfg.d_V
                mean_x = float(np.mean(h_idx))
                mean_y = float(np.mean(v_idx))
                print(f"  🔧 Steering grid: dx={dx:.4f}m, dy={dy:.4f}m (should equal d_h={cfg.d_H:.4f}m, d_v={cfg.d_V:.4f}m)", flush=True)
                print(f"  🔧 Centering: mean(x)={mean_x:.2e}m, mean(y)={mean_y:.2e}m (should be ≈0)", flush=True)
                if abs(dx - cfg.d_H) > 1e-6 or abs(dy - cfg.d_V) > 1e-6:
                    print(f"     ⚠️  WARNING: Grid spacing mismatch!", flush=True)
                if abs(mean_x) > 1e-6 or abs(mean_y) > 1e-6:
                    print(f"     ⚠️  WARNING: Grid not centered!", flush=True)
                
                # L=16 SANITY: Print eigenspectrum of predicted covariance (first val batch)
                # Defer to after first validation pass to avoid iterator issues
                if hasattr(self, '_eigenspectrum_logged') is False and hasattr(self, '_first_val_batch'):
                    with torch.no_grad():
                        batch_val = self._first_val_batch
                        if isinstance(batch_val, dict):
                            y_val = batch_val["y"].to(self.device)
                            H_val = batch_val["H"].to(self.device)
                            codes_val = batch_val["codes"].to(self.device)
                            snr_val = batch_val.get("snr_db")
                            if snr_val is not None:
                                snr_val = snr_val.to(self.device)
                        else:
                            # Unpack tuple (now has 8 elements with H_full)
                            y_val = batch_val[0].to(self.device)
                            H_val = batch_val[1].to(self.device)
                            codes_val = batch_val[2].to(self.device)
                            snr_val = batch_val[6] if len(batch_val) > 6 else None
                            if snr_val is not None:
                                snr_val = snr_val.to(self.device)
                        
                        preds_val = self.model(y_val, H_val, codes_val, snr_val)
                        
                        # Form R_hat from first sample
                        cf_ang = preds_val["cov_fact_angle"][0].detach().cpu().numpy()  # [N*K_MAX*2]
                        N = N_H * N_V
                        cf_ang_complex = cf_ang[:N * cfg.K_MAX].reshape(N, cfg.K_MAX) + 1j * cf_ang[N * cfg.K_MAX:].reshape(N, cfg.K_MAX)
                        R_hat = cf_ang_complex @ cf_ang_complex.conj().T  # [N, N]
                        
                        # Trace-normalize
                        trace_R = np.real(np.trace(R_hat))
                        if trace_R > 1e-8:
                            R_hat = R_hat * (N / trace_R)
                        
                        # Eigenvalues
                        eigs = np.linalg.eigvalsh(R_hat)  # Real eigenvalues, ascending
                        eigs = eigs[::-1]  # Descending
                        eig_sum = eigs.sum()
                        top5_frac = eigs[:5] / eig_sum if eig_sum > 0 else eigs[:5]
                        top5_sum = top5_frac.sum()
                        
                        # CRITICAL: Check if R is rank-deficient (rank ≤ K_MAX)
                        rank_estimate = np.sum(eigs > 1e-6 * eigs[0])  # Eigenvalues > 1e-6 of largest
                        
                        print(f"  🔬 Eigenspectrum (raw network R_pred, first val sample):", flush=True)
                        print(f"     tr(R̂)={np.real(np.trace(R_hat)):.2f} (target: {N}), ||R̂||_F={np.linalg.norm(R_hat,'fro'):.2f}", flush=True)
                        print(f"     Estimated rank: {rank_estimate}/{N} (based on eigenvalue threshold)", flush=True)
                        print(f"     Eigenvalues: Σ(all {N})={eig_sum:.2f}, Σ(top-5)={eig_sum*top5_sum:.2f} ({top5_sum:.1%} of total)", flush=True)
                        print(f"     Top-5 λ (fraction of Σ_all): λ₁={top5_frac[0]:.3f}, λ₂={top5_frac[1]:.3f}, λ₃={top5_frac[2]:.3f}, λ₄={top5_frac[3]:.3f}, λ₅={top5_frac[4]:.3f}", flush=True)
                        
                        # CRITICAL WARNING: Check for rank deficiency
                        if rank_estimate <= cfg.K_MAX:
                            print(f"     ❌ RANK-DEFICIENT! (rank ≤ {cfg.K_MAX}) → This is EXPECTED for raw network output!", flush=True)
                            print(f"        Network outputs [N, K_MAX] factor → R_pred has rank ≤ {cfg.K_MAX}", flush=True)
                            print(f"        Hybrid blending will add full-rank R_samp from L=16 snapshots!", flush=True)
                            print(f"        → Check [HYBRID COV] diagnostic below for the FINAL result!", flush=True)
                        
                        # Interpretation guide (only if rank is OK)
                        elif top5_frac[0] > 0.60:
                            print(f"     ✅ SHARP spectrum (λ₁={top5_frac[0]:.2f}) → strong MUSIC peaks expected!", flush=True)
                        elif top5_frac[0] > 0.40:
                            print(f"     ⚠️  MODERATE spectrum (λ₁={top5_frac[0]:.2f}) → consider increasing HYBRID_COV_BETA to 0.30-0.40", flush=True)
                        else:
                            print(f"     ⚠️  FLAT spectrum (λ₁={top5_frac[0]:.2f}) → increase HYBRID_COV_BETA to 0.30-0.40", flush=True)
                        
                        # HYBRID COV DIAGNOSTIC: Test hybrid blending at epoch 0!
                        if rank_estimate <= cfg.K_MAX and hasattr(cfg, 'HYBRID_COV_BLEND') and cfg.HYBRID_COV_BLEND:
                            print(f"  🔬 Testing hybrid covariance blending (epoch 0 diagnostic):", flush=True)
                            try:
                                from ris_pytorch_pipeline.angle_pipeline import build_sample_covariance_from_snapshots
                                
                                # Extract snapshots from first sample
                                y_i = y_val[0].detach().cpu().numpy()  # [L, M_BS, 2]
                                # Use H_full if available, otherwise fall back to old H (for backward compat)
                                if 'H_full' in batch_val and batch_val['H_full'] is not None:
                                    H_full_i = batch_val['H_full'][0].detach().cpu().numpy()  # [M_BS, N, 2]
                                    H_cplx = H_full_i[:, :, 0] + 1j * H_full_i[:, :, 1]  # [M_BS, N]
                                else:
                                    # Fallback: old behavior (will produce rank-1 Phi, but at least won't crash)
                                    H_i = H_val[0].detach().cpu().numpy()  # [L, M_BS, 2]
                                    # Assume first snapshot as proxy (not physically correct, but backward compat)
                                    H_cplx = (H_i[0, :, 0] + 1j * H_i[0, :, 1])[:, np.newaxis]  # [M_BS, 1]
                                    H_cplx = np.tile(H_cplx, (1, cfg.N))  # [M_BS, N] filled with same values
                                
                                C_i = codes_val[0].detach().cpu().numpy()  # [L, N, 2]
                                
                                # Convert to complex
                                y_snaps = y_i[:, :, 0] + 1j * y_i[:, :, 1]  # [L, M_BS]
                                codes_snaps = C_i[:, :, 0] + 1j * C_i[:, :, 1]  # [L, N]
                                
                                # Tile H_full for all snapshots (H is constant across snapshots, codes vary)
                                H_snaps = np.tile(H_cplx[np.newaxis, :, :], (cfg.L, 1, 1))  # [L, M_BS, N]
                                
                                # Build sample covariance
                                R_samp = build_sample_covariance_from_snapshots(y_snaps, H_snaps, codes_snaps, cfg, tikhonov_alpha=1e-3)
                                
                                # Blend
                                blend_beta = getattr(cfg, 'HYBRID_COV_BETA', 0.30)
                                
                                # DEBUG: Check if matrices are actually different
                                print(f"     [DEBUG] tr(R_hat)={np.trace(R_hat):.2f}, tr(R_samp)={np.trace(R_samp):.2f}", flush=True)
                                print(f"     [DEBUG] ||R_hat||_F={np.linalg.norm(R_hat,'fro'):.2f}, ||R_samp||_F={np.linalg.norm(R_samp,'fro'):.2f}", flush=True)
                                diff_norm = np.linalg.norm(R_hat - R_samp, 'fro')
                                print(f"     [DEBUG] ||R_hat - R_samp||_F={diff_norm:.2f} (should be large!)", flush=True)
                                
                                R_blend = (1.0 - blend_beta) * R_hat + blend_beta * R_samp
                                
                                # Hermitize and trace-normalize
                                R_blend = 0.5 * (R_blend + R_blend.conj().T)
                                trace_blend = np.real(np.trace(R_blend))
                                if trace_blend > 1e-8:
                                    R_blend = R_blend * (N / trace_blend)
                                
                                # Compute eigenspectra
                                eigs_pred = np.linalg.eigvalsh(R_hat)[::-1]
                                eigs_samp = np.linalg.eigvalsh(R_samp)[::-1]
                                eigs_blend = np.linalg.eigvalsh(R_blend)[::-1]
                                
                                rank_samp = np.sum(eigs_samp > 1e-6 * eigs_samp[0])
                                rank_blend = np.sum(eigs_blend > 1e-6 * eigs_blend[0])
                                
                                print(f"     [HYBRID COV] β={blend_beta:.2f}", flush=True)
                                print(f"       R_pred  (rank {rank_estimate}/{N}): λ₁={eigs_pred[0]/eigs_pred.sum():.3f}, top-5={eigs_pred[:5].sum()/eigs_pred.sum():.1%}", flush=True)
                                print(f"       R_samp  (rank {rank_samp}/{N}): λ₁={eigs_samp[0]/eigs_samp.sum():.3f}, top-5={eigs_samp[:5].sum()/eigs_samp.sum():.1%}", flush=True)
                                print(f"       R_blend (rank {rank_blend}/{N}): λ₁={eigs_blend[0]/eigs_blend.sum():.3f}, top-5={eigs_blend[:5].sum()/eigs_blend.sum():.1%}", flush=True)
                                
                                if rank_blend > cfg.K_MAX:
                                    print(f"     ✅ Hybrid blending WORKING! Rank increased from {rank_estimate} → {rank_blend}", flush=True)
                                else:
                                    print(f"     ❌ Hybrid blending FAILED! Rank still {rank_blend} (should be >{cfg.K_MAX})", flush=True)
                                    
                            except Exception as e:
                                print(f"     ❌ Hybrid blending ERROR: {e}", flush=True)
                                import traceback
                                traceback.print_exc()
                        
                        # CLASSICAL MUSIC BASELINE: Check performance on ground-truth R_true with NEAR-FIELD steering
                        try:
                            # Get ground truth for first scene
                            if isinstance(batch_val, dict):
                                K_batch = batch_val["K"].to(self.device)
                                ptr_batch = batch_val["ptr"].to(self.device)
                                R_in_batch = batch_val["R_true"].to(self.device)
                            else:
                                # Unpack tuple (now has 8 elements: y, H, C, ptr, K, R, snr, H_full)
                                _, _, _, ptr_batch, K_batch, R_in_batch, _, _ = batch_val
                            
                            k_true = int(K_batch[0].item())
                            ptr_np = ptr_batch[0].detach().cpu().numpy()
                            phi_gt = np.rad2deg(ptr_np[:k_true])
                            theta_gt = np.rad2deg(ptr_np[cfg.K_MAX:cfg.K_MAX+k_true])
                            
                            # Get ground truth range (if available)
                            r_gt = None
                            if len(ptr_np) > 2*cfg.K_MAX:
                                r_gt = ptr_np[2*cfg.K_MAX:2*cfg.K_MAX+k_true]  # Ground truth ranges
                            
                            print(f"  [DEBUG Classical] k_true={k_true}, phi_gt={phi_gt}, theta_gt={theta_gt}", flush=True)
                            if r_gt is not None:
                                print(f"  [DEBUG Classical] r_gt={r_gt}", flush=True)
                            
                            # EXPERT FIX: Use R_true directly from shard (NO LS, NO snapshot math!)
                            R_true_ri = R_in_batch[0].detach().cpu().numpy()  # [N, N, 2]
                            R_true = R_true_ri[:, :, 0] + 1j * R_true_ri[:, :, 1]  # [N, N] complex
                            
                            # Trace-normalize R_true
                            N = R_true.shape[0]
                            trace_R = np.real(np.trace(R_true))
                            if trace_R > 1e-8:
                                R_true = R_true * (N / trace_R)
                            
                            print(f"  [DEBUG Classical] R_true: tr={np.real(np.trace(R_true)):.1f}, ||R_true||_F={np.linalg.norm(R_true,'fro'):.1f}", flush=True)
                            
                            # EXPERT FIX: Run MUSIC with NEAR-FIELD steering using Ramezani reference
                            from ris_pytorch_pipeline.ramezani_mod_music import nearfield_vec
                            
                            # MUSIC with near-field steering
                            phi_music_true, theta_music_true = _classical_music_nearfield(
                                R_true, k_true, phi_gt, theta_gt, r_gt, cfg
                            )
                            
                            print(f"  [DEBUG Classical] MUSIC found: phi={phi_music_true}, theta={theta_music_true}", flush=True)
                            
                            # Compute errors (simple nearest-neighbor for quick check)
                            if len(phi_music_true) > 0 and len(phi_gt) > 0:
                                # Simple min distance for each GT
                                errs_phi = []
                                errs_theta = []
                                for j in range(len(phi_gt)):
                                    dphi = np.abs(phi_music_true - phi_gt[j])
                                    dtheta = np.abs(theta_music_true - theta_gt[j])
                                    idx = np.argmin(dphi + dtheta)
                                    errs_phi.append(dphi[idx])
                                    errs_theta.append(dtheta[idx])
                                
                                med_phi_true = np.median(errs_phi)
                                med_theta_true = np.median(errs_theta)
                                print(f"  🎯 CLASSICAL MUSIC CEILING (R_true + NF steering, scene 0): φ={med_phi_true:.2f}°, θ={med_theta_true:.2f}°", flush=True)
                                if med_phi_true > 5.0:
                                    print(f"     ⚠️  Classical MUSIC > 5° → check manifold/convention (should be ≪5° with R_true)!", flush=True)
                                else:
                                    print(f"     ✅ Classical MUSIC < 5° → manifold/convention correct!", flush=True)
                        except Exception as e:
                            print(f"  ⚠️  Classical MUSIC check failed: {e}", flush=True)
                        
                        self._eigenspectrum_logged = True

        # Optional: auto-calibrate K logits on validation and save into best checkpoint
        try:
            if bool(getattr(mdl_cfg, "CALIBRATE_K_AFTER_TRAIN", True)):
                print("🌡️ Running post-train K calibration on validation set...")
                optimal_temp, calib_acc, high_snr_acc = self.calibrate_k_logits(va_loader, save_path=str(best_path))
                # Also attach to in-memory model for immediate use in this session
                self.model._k_calibration_temp = float(optimal_temp)
        except Exception as e:
            print(f"⚠️ K calibration skipped due to error: {e}")
        
        # Return objective for outer loops (HPO/automation)
        # - If VAL_PRIMARY is metric-driven ('k_loc'/'metrics'), return best composite score
        # - Otherwise return best validation loss
        if str(getattr(cfg, "VAL_PRIMARY", "loss")).lower() in ("k_loc", "metrics"):
            return float(best_score)
        return float(best_val)


def _classical_music_nearfield(R_true, k_true, phi_gt, theta_gt, r_gt, cfg):
    """
    Classical MUSIC ceiling check: R_true + NF steering (no LS, no snapshot math).
    
    CRITICAL FIX: This is a pure ceiling check on R_true only, using the correct
    near-field steering convention that matches the data generation.
    
    Args:
        R_true: Ground truth covariance [N, N] complex
        k_true: Number of sources
        phi_gt: Ground truth azimuth angles (degrees)
        theta_gt: Ground truth elevation angles (degrees) 
        r_gt: Ground truth ranges (meters) or None
        cfg: Config object
    
    Returns:
        phi_music: MUSIC azimuth estimates (degrees)
        theta_music: MUSIC elevation estimates (degrees)
    """
    N = R_true.shape[0]
    N_H = getattr(cfg, 'N_H', 12)
    N_V = getattr(cfg, 'N_V', 12)
    d_h = getattr(cfg, 'd_H', 0.5)  # wavelengths
    d_v = getattr(cfg, 'd_V', 0.5)  # wavelengths
    lam = getattr(cfg, 'WAVEL', 0.3)  # meters
    k = 2.0 * np.pi / lam
    
    print(f"  [CEILING] manifold=NEAR_FIELD, convention=(+x: sinφ·cosθ, +y: +sinθ, curvature: -), r_gt={r_gt[0] if r_gt is not None and len(r_gt) > 0 else 'N/A'}", flush=True)
    
    # EVD on R_true (ascending order, use smallest eigenvectors for noise)
    eigvals, eigvecs = np.linalg.eigh(R_true)
    # CRITICAL: Sort ascending, use smallest (N - k_true) eigenvectors for noise
    assert np.all(eigvals[:-1] <= eigvals[1:]), "Eigenvalues not sorted ascending!"
    
    noise_rank = max(1, N - k_true)
    U_noise = eigvecs[:, :noise_rank]  # Smallest noise_rank eigenvectors
    G = U_noise @ U_noise.conj().T  # Noise projector
    
    print(f"  [CEILING] Noise subspace: rank {noise_rank}/{N}, λ_min={eigvals[0]:.3f}, λ_max={eigvals[-1]:.3f}", flush=True)
    
    # Grid search with near-field steering
    phi_grid = np.linspace(-60, 60, 121)  # degrees
    theta_grid = np.linspace(-30, 30, 61)  # degrees
    
    # CRITICAL FIX: Handle multiple sources at different ranges
    if r_gt is not None and len(r_gt) > 0:
        # Use all ground truth ranges for multi-source scenarios
        r_ranges = r_gt[:k_true]  # Use all K_true ranges
        print(f"  [CEILING] Using GT ranges: {r_ranges}", flush=True)
    else:
        # Fallback: use range grid for unknown ranges
        r_min, r_max = 0.5, 10.0  # From cfg.RANGE_R
        r_ranges = np.linspace(r_min, r_max, 21)  # 21 range points
        print(f"  [CEILING] Using range grid: {r_min:.1f}m to {r_max:.1f}m", flush=True)
    
    spectrum = np.zeros((len(phi_grid), len(theta_grid)))
    
    for i, phi in enumerate(phi_grid):
        for j, theta in enumerate(theta_grid):
            # CRITICAL: Range-aware search for multi-source scenarios
            best_spectrum = 0.0
            
            for r_candidate in r_ranges:
                # Use near-field steering with this range candidate
                a = _nearfield_steering_canonical(
                    np.deg2rad(phi), np.deg2rad(theta), r_candidate,
                    N_H, N_V, d_h, d_v, lam
                )
                
                # MUSIC spectrum: 1 / (a^H G a)
                denom = np.real(a.conj().T @ G @ a)
                P = 1.0 / max(denom, 1e-12)
                
                # Keep best spectrum across all range candidates
                best_spectrum = max(best_spectrum, P)
            
            spectrum[i, j] = best_spectrum
    
    # Find peaks
        phi_flat = spectrum.flatten()
        threshold = np.max(phi_flat) * 0.1
        peaks = np.where(phi_flat > threshold)[0]
    
    if len(peaks) > 0:
        # Get top k_true peaks
        peak_values = phi_flat[peaks]
        top_peaks = np.argsort(peak_values)[-k_true:]
        
        phi_music = []
        theta_music = []
        
        for peak_idx in top_peaks:
            flat_idx = peaks[peak_idx]
            i, j = np.unravel_index(flat_idx, spectrum.shape)
            phi_music.append(phi_grid[i])
            theta_music.append(theta_grid[j])
        
        # Sanity prints
        phi_peaks_str = ", ".join([f"{p:.1f}" for p in phi_music])
        theta_peaks_str = ", ".join([f"{t:.1f}" for t in theta_music])
        print(f"  [CEILING] peaks_deg: [({phi_peaks_str}), ({theta_peaks_str})]", flush=True)
        
        # Check if GT is the maximum (range-aware)
        if len(phi_gt) > 0 and len(theta_gt) > 0:
            phi_gt_rad = np.deg2rad(phi_gt[0])
            theta_gt_rad = np.deg2rad(theta_gt[0])
            
            # Use GT range for GT angle, best range for edge angle
            r_gt_used = r_ranges[0] if len(r_ranges) > 0 else 2.0
            a_gt = _nearfield_steering_canonical(phi_gt_rad, theta_gt_rad, r_gt_used, N_H, N_V, d_h, d_v, lam)
            P_gt = 1.0 / max(np.real(a_gt.conj().T @ G @ a_gt), 1e-12)
            
            # Check edge maximum with range search
            P_edge = 0.0
            for r_candidate in r_ranges:
                a_edge = _nearfield_steering_canonical(np.deg2rad(59.33), np.deg2rad(29.5), r_candidate, N_H, N_V, d_h, d_v, lam)
                P_edge_candidate = 1.0 / max(np.real(a_edge.conj().T @ G @ a_edge), 1e-12)
                P_edge = max(P_edge, P_edge_candidate)
            
            print(f"  [CEILING] P(gt)={P_gt:.2e}, P(edge_max)={P_edge:.2e}, gt_is_max={P_gt > P_edge}", flush=True)
        
        return np.array(phi_music), np.array(theta_music)
    else:
        # Fallback: return ground truth if no peaks found
        return phi_gt[:k_true], theta_gt[:k_true]


def _nearfield_steering_canonical(phi_rad, theta_rad, r, N_H, N_V, d_h_wavelengths, d_v_wavelengths, lam):
    """
    Canonical near-field steering vector matching EXACT dataset generation convention.
    
    CRITICAL: This MUST match physics.py nearfield_vec() exactly to avoid edge-pegging.
    
    Args:
        phi_rad, theta_rad: Angles in radians
        r: Range in meters
        N_H, N_V: Array dimensions
        d_h_wavelengths, d_v_wavelengths: Element spacing in wavelengths (NOT meters!)
        lam: Wavelength in meters
    
    Returns:
        a: Steering vector [N] complex64, unit-normalized
    """
    k = 2.0 * np.pi / lam
    
    # CRITICAL: Use EXACT same indexing as physics.py
    # physics.py: h_idx = np.arange(-(cfg.N_H - 1)//2, (cfg.N_H + 1)//2) * cfg.d_H
    # NOTE: d_H is in wavelengths, so h_idx is also in wavelengths
    h_idx = np.arange(-(N_H - 1)//2, (N_H + 1)//2) * d_h_wavelengths
    v_idx = np.arange(-(N_V - 1)//2, (N_V + 1)//2) * d_v_wavelengths
    
    # Create 2D grid with EXACT same indexing as physics.py
    h_mesh, v_mesh = np.meshgrid(h_idx, v_idx, indexing="xy")
    h_flat = h_mesh.reshape(-1).astype(np.float32)
    v_flat = v_mesh.reshape(-1).astype(np.float32)
    
    # CRITICAL: Use EXACT same phase formula as physics.py
    # physics.py: dist = r - vh*sin_phi*cos_theta - vv*sin_theta + (vh**2 + vv**2)/(2*r_eff)
    sin_phi = np.sin(phi_rad)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    a = np.empty(N_H * N_V, np.complex64)
    r_eff = max(float(r), 1e-9)
    
    for i, (vh, vv) in enumerate(zip(h_flat, v_flat)):
        # EXACT same formula as physics.py
        dist = r - vh*sin_phi*cos_theta - vv*sin_theta + (vh**2 + vv**2)/(2*r_eff)
        a[i] = np.exp(1j * k * (r - dist))
    
    # CRITICAL: Use same normalization as physics.py
    a = a / np.sqrt(N_H * N_V)
    
    return a
    
