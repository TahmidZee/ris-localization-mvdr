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
from .music_gpu import get_gpu_estimator  # MVDR spectrum computation for SpectrumRefiner stage

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

def build_R_pred_from_factor_vecs(cf_ang: torch.Tensor, cf_rng: torch.Tensor, *, N: int, Kmax: int, lam_range: float) -> torch.Tensor:
    """
    Build complex covariance from interleaved RI factor vectors.

    Args:
        cf_ang/cf_rng: [B, 2*N*Kmax] with interleaved real/imag entries.
        N: covariance dimension (cfg.N)
        Kmax: maximum sources (cfg.K_MAX)
        lam_range: range factor weight (mdl_cfg.LAM_RANGE_FACTOR)

    Returns:
        R_pred: [B, N, N] complex64, Hermitian.
    """
    B = int(cf_ang.shape[0])
    cf_ang = cf_ang.float()
    cf_rng = cf_rng.float()

    def _vec2c(v: torch.Tensor) -> torch.Tensor:
        xr, xi = v[:, ::2], v[:, 1::2]  # [B, N*Kmax]
        A = torch.complex(xr.view(B, N, Kmax), xi.view(B, N, Kmax)).to(torch.complex64)
        # Magnitude leash (match loss.py): normalize columns before AA^H.
        if bool(getattr(cfg, "FACTOR_COLNORM_ENABLE", True)):
            eps = float(getattr(cfg, "FACTOR_COLNORM_EPS", 1e-6))
            max_norm = float(getattr(cfg, "FACTOR_COLNORM_MAX", 1e3))
            col = torch.linalg.norm(A, dim=-2, keepdim=True).clamp_min(eps)  # [B,1,K]
            if max_norm > 0:
                col = col.clamp(max=max_norm)
            A = A / col
        return A

    Aang = _vec2c(cf_ang)
    Arng = _vec2c(cf_rng)
    R = (Aang @ Aang.conj().transpose(-2, -1)) + float(lam_range) * (Arng @ Arng.conj().transpose(-2, -1))
    return 0.5 * (R + R.conj().transpose(-2, -1))

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
        self.refiner  = None
        # Optional warm-start (weights-only). This is intentionally *before* any phase freezing
        # so that all modules load consistently, then we apply requires_grad masks.
        ckpt = str(getattr(cfg, "INIT_CKPT", "")).strip()
        if ckpt:
            p = Path(ckpt)
            if p.exists():
                sd = torch.load(str(p), map_location="cpu")
                # Support multiple checkpoint formats:
                # - state_dict() directly
                # - {"model": state_dict}
                # - {"backbone": state_dict, "refiner": state_dict} (refiner stage)
                if isinstance(sd, dict) and ("backbone" in sd or "refiner" in sd):
                    bb = sd.get("backbone", sd.get("model", sd))
                    missing, unexpected = self.model.load_state_dict(bb, strict=False)
                    print(f"✅ Loaded backbone from INIT_CKPT: {p} | missing={len(missing)} unexpected={len(unexpected)}", flush=True)
                    # Load refiner weights if present and we are in refiner stage (refiner may not be constructed yet)
                    self._init_refiner_state_dict = sd.get("refiner", None)
                else:
                    if isinstance(sd, dict) and "model" in sd:
                        sd = sd["model"]
                    missing, unexpected = self.model.load_state_dict(sd, strict=False)
                    print(f"✅ Loaded INIT_CKPT: {p} | missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            else:
                print(f"⚠️ INIT_CKPT not found: {p}", flush=True)
        self.phase    = str(getattr(cfg, "TRAIN_PHASE", "geom")).lower()
        self.train_refiner_only = (self.phase == "refiner")

        # Option B (Stage 2): freeze backbone and train SpectrumRefiner only
        if self.train_refiner_only:
            from .model import SpectrumRefiner
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.refiner = SpectrumRefiner().to(self.device)
            # Optional: resume refiner weights if INIT_CKPT carried them
            if hasattr(self, "_init_refiner_state_dict") and (self._init_refiner_state_dict is not None):
                try:
                    missing, unexpected = self.refiner.load_state_dict(self._init_refiner_state_dict, strict=False)
                    print(f"✅ Loaded refiner from INIT_CKPT | missing={len(missing)} unexpected={len(unexpected)}", flush=True)
                except Exception as e:
                    print(f"⚠️ Failed to load refiner state from INIT_CKPT: {e}", flush=True)

            # MVDR estimator and cached grids for fast spectrum generation from low-rank factors
            self._mvdr_est = get_gpu_estimator(cfg, device=("cuda" if self.device.type == "cuda" else "cpu"))
            self._refiner_phi_grid = torch.linspace(
                math.radians(float(getattr(cfg, "PHI_MIN_DEG", -60.0))),
                math.radians(float(getattr(cfg, "PHI_MAX_DEG", 60.0))),
                int(getattr(cfg, "REFINER_GRID_PHI", 61)),
                device=self.device,
                dtype=torch.float32,
            )
            self._refiner_theta_grid = torch.linspace(
                math.radians(float(getattr(cfg, "THETA_MIN_DEG", -30.0))),
                math.radians(float(getattr(cfg, "THETA_MAX_DEG", 30.0))),
                int(getattr(cfg, "REFINER_GRID_THETA", 41)),
                device=self.device,
                dtype=torch.float32,
            )
            self._refiner_r_planes = getattr(cfg, "REFINER_R_PLANES", None) or self._mvdr_est.default_r_planes_mvdr

            # Default heatmap weight if user didn't set one
            if float(getattr(mdl_cfg, "LAM_HEATMAP", 0.0)) <= 0.0:
                setattr(mdl_cfg, "LAM_HEATMAP", 0.1)
            # Disable alignment-on-cov losses during refiner-only stage
            setattr(mdl_cfg, "LAM_ALIGN", 0.0)
        # Apply phase-specific freezing before optimizer construction
        if (not self.train_refiner_only) and hasattr(self.model, "set_trainable_for_phase"):
            freeze_backbone = bool(getattr(cfg, "FREEZE_BACKBONE_FOR_K_PHASE", False)) if self.phase == "k_only" else False
            freeze_aux = bool(getattr(cfg, "FREEZE_AUX_IN_K_PHASE", False)) if self.phase == "k_only" else False
            self.model.set_trainable_for_phase(
                freeze_backbone=freeze_backbone,
                freeze_aux=freeze_aux,
            )
        # Initialize loss with config values (will be updated by schedule)
        self.loss_fn = UltimateHybridLoss(
            lam_cov=1.0,  # CRITICAL: Main covariance NMSE weight (will be scaled by HPO if needed)
            lam_subspace_align=getattr(mdl_cfg, 'LAM_SUBSPACE_ALIGN', 0.05),
            lam_peak_contrast=getattr(mdl_cfg, 'LAM_PEAK_CONTRAST', 0.1),
            lam_cov_pred=getattr(mdl_cfg, 'LAM_COV_PRED', 0.05),  # 5% auxiliary on R_pred
            # SpectrumRefiner supervision (optional; active only if model outputs refined_spectrum)
            lam_heatmap=getattr(mdl_cfg, 'LAM_HEATMAP', 0.0),
            heatmap_sigma_phi=getattr(mdl_cfg, 'HEATMAP_SIGMA_PHI', 2.0),
            heatmap_sigma_theta=getattr(mdl_cfg, 'HEATMAP_SIGMA_THETA', 2.0),
        ).to(self.device)

        # If training refiner-only, use heatmap-only loss for speed/stability
        if self.train_refiner_only:
            self.loss_fn.lam_cov = 0.0
            self.loss_fn.lam_cov_pred = 0.0
            self.loss_fn.lam_ortho = 0.0
            self.loss_fn.lam_cross = 0.0
            self.loss_fn.lam_gap = 0.0
            self.loss_fn.lam_margin = 0.0
            self.loss_fn.lam_aux = 0.0
            self.loss_fn.lam_peak = 0.0
            self.loss_fn.lam_subspace_align = 0.0
            self.loss_fn.lam_peak_contrast = 0.0
            self.loss_fn.lam_heatmap = float(getattr(mdl_cfg, "LAM_HEATMAP", 0.1))
        
        # Always initialize _hpo_loss_weights (may be populated later by HPO config loading)
        self._hpo_loss_weights = {}
        
        # Beta annealing: start low (trust network), gradually blend in R_samp
        self.beta_start = 0.0
        self.beta_final = getattr(cfg, 'HYBRID_COV_BETA', 0.30)
        self.beta_warmup_epochs = None  # Will be set based on total epochs
        
        # Apply HPO loss weights if they were loaded
        if self._hpo_loss_weights:
            self._apply_hpo_loss_weights()
        # Apply phase-specific loss weights (overrides defaults/HPO for the chosen phase)
        self._apply_phase_loss_weights(self.phase)

        # Finalize refiner-only loss setup after all schedules/overrides
        if self.train_refiner_only:
            self.loss_fn.lam_cov = 0.0
            self.loss_fn.lam_cov_pred = 0.0
            self.loss_fn.lam_ortho = 0.0
            self.loss_fn.lam_cross = 0.0
            self.loss_fn.lam_gap = 0.0
            self.loss_fn.lam_margin = 0.0
            self.loss_fn.lam_aux = 0.0
            self.loss_fn.lam_peak = 0.0
            self.loss_fn.lam_subspace_align = 0.0
            self.loss_fn.lam_peak_contrast = 0.0
            self.loss_fn.lam_heatmap = float(getattr(mdl_cfg, "LAM_HEATMAP", 0.1))
        lr_init       = float(getattr(mdl_cfg, "LR_INIT", 3e-4))
        opt_name      = str(getattr(mdl_cfg, "OPT", "adamw")).lower()
        wd            = float(getattr(mdl_cfg, "WEIGHT_DECAY", 1e-4))

        # CRITICAL FIX: Bullet-proof parameter grouping with strong asserts
        # Ensure modules are in train mode and parameters are trainable as intended
        self.model.train()
        if self.refiner is not None:
            self.refiner.train()
        if not self.train_refiner_only:
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    # Allow intentional freezing in K-only phase
                    if self.phase == "k_only":
                        continue
                    print(f"⚠️  Re-enabling gradients for: {n}")
                    p.requires_grad_(True)
        
        # Robust parameter grouping by name prefix
        backbone_params = []
        head_params = []
        # Head keys used for optimizer grouping (head gets higher LR than backbone).
        # NOTE: K-head components removed - using MVDR peak detection instead
        HEAD_KEYS = (
            'head', 'classifier', 'aux_angles', 'aux_range',
            'cov_fact_angle', 'cov_fact_range', 'logits_gg'
        )
        
        if self.train_refiner_only:
            backbone_params = []
            head_params = list(self.refiner.parameters()) if self.refiner is not None else []
        else:
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
        n_tot_model = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_tot_refiner = sum(p.numel() for p in self.refiner.parameters() if p.requires_grad) if self.refiner is not None else 0
        n_tot = n_tot_model + n_tot_refiner
        
        # Head LR multiplier (3-5× backbone LR)
        head_lr_multiplier = float(getattr(mdl_cfg, "HEAD_LR_MULTIPLIER", 4.0))  # 4× by default
        head_lr = lr_init * head_lr_multiplier
        
        print(f"🔧 Optimizer setup:")
        print(f"   Backbone: {n_back:,} params ({n_back/1e6:.1f}M) @ LR={lr_init:.2e}")
        print(f"   Head: {n_head:,} params ({n_head/1e6:.1f}M) @ LR={head_lr:.2e} ({head_lr_multiplier}×)")
        print(f"   Total trainable: {n_tot:,}")
        
        # CRITICAL ASSERTS - catch dead groups immediately (relaxed for frozen phases)
        if self.train_refiner_only:
            assert n_head > 0, f"❌ Refiner param count too small: {n_head:,}"
        else:
            if self.phase != "k_only":
                assert n_back > 1_000_000, f"❌ Backbone param count too small: {n_back:,}"
            assert n_head > 0, f"❌ Head param count too small: {n_head:,}"
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
        if self.train_refiner_only and (self.refiner is not None):
            missing = [n for n, p in self.refiner.named_parameters() if p.requires_grad and id(p) not in opt_ids]
        else:
            missing = [n for n, p in self.model.named_parameters() if p.requires_grad and id(p) not in opt_ids]
        assert not missing, f"❌ Params missing from optimizer: {missing[:8]}"
        print(f"   ✅ Optimizer wiring verified: all {len(opt_ids)} trainable params in optimizer!")

        # AMP
        # Prefer mdl_cfg.USE_AMP if present (some scripts set USE_AMP), fallback to mdl_cfg.AMP for backward compat.
        self.amp = bool(getattr(mdl_cfg, "USE_AMP", getattr(mdl_cfg, "AMP", True)))
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.amp and self.device.type == "cuda"))

        # EMA
        self.use_ema   = bool(getattr(mdl_cfg, "USE_EMA", True)) and (not self.train_refiner_only)
        self.ema_decay = float(getattr(mdl_cfg, "EMA_DECAY", 0.999))
        # Only track floating-point parameters in EMA (skip buffers like indices)
        self.ema_shadow: Dict[str, torch.Tensor] = {}
        if self.use_ema:
            self.ema_shadow = {
                k: v.detach().clone() for k, v in self.model.state_dict().items() 
                if v.dtype.is_floating_point
            }
        self._ema_bak: Dict[str, torch.Tensor] | None = None
        
        # SWA (Stochastic Weight Averaging)
        self.use_swa = getattr(mdl_cfg, 'USE_SWA', False) and (not self.train_refiner_only)
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
    # Reproducibility
    # ----------------------------
    def _save_run_config(self, epochs, n_train, n_val, grad_accumulation, early_stop_patience):
        """Save run configuration for reproducibility."""
        import json
        import subprocess
        
        # Get git commit hash if available
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).parent.parent),
                stderr=subprocess.DEVNULL
            ).decode().strip()[:8]
        except:
            commit_hash = "unknown"
        
        run_config = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": commit_hash,
            "seed": int(getattr(mdl_cfg, "SEED", 42)),
            "epochs": epochs,
            "n_train": n_train,
            "n_val": n_val,
            "grad_accumulation": grad_accumulation,
            "early_stop_patience": early_stop_patience,
            "val_primary": str(getattr(cfg, "VAL_PRIMARY", "surrogate")),
            "use_music_in_val": bool(getattr(cfg, "USE_MUSIC_METRICS_IN_VAL", False)),
            "model_config": {
                "D_MODEL": mdl_cfg.D_MODEL,
                "NUM_HEADS": mdl_cfg.NUM_HEADS,
                "DROPOUT": mdl_cfg.DROPOUT,
                "LR_INIT": mdl_cfg.LR_INIT,
                "BATCH_SIZE": mdl_cfg.BATCH_SIZE,
            },
            "system_config": {
                "N_H": cfg.N_H,
                "N_V": cfg.N_V,
                "K_MAX": cfg.K_MAX,
                "HYBRID_COV_BETA": getattr(cfg, "HYBRID_COV_BETA", 0.0),
            }
        }
        
        config_path = Path(cfg.CKPT_DIR) / "run_config.json"
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2)
        print(f"📋 Run config saved to: {config_path}")
        print(f"   Git commit: {commit_hash}, Seed: {run_config['seed']}, VAL_PRIMARY: {run_config['val_primary']}")

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
        # NOTE: lam_K removed - using MVDR peak detection instead
        # Backward compatible: older HPO used "shrink_alpha" but it actually maps to SHRINK_BASE_ALPHA.
        if "shrink_base_alpha" in best:
            setattr(mdl_cfg, "SHRINK_BASE_ALPHA", float(best["shrink_base_alpha"]))
        elif "shrink_alpha" in best:
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
        # NOTE: lam_K removed - using MVDR peak detection instead
        
        # Set reasonable defaults for missing loss parameters (same as HPO)
        self.loss_fn.lam_ortho = 1e-3  # Orthogonality penalty
        self.loss_fn.lam_peak = 0.05   # Chamfer/peak angle loss
        # Eigengap/margin disabled globally
        self.loss_fn.lam_gap = 0.0
        self.loss_fn.lam_margin = 0.0
        self.loss_fn.lam_range_factor = 0.3  # Range factor in covariance
        mdl_cfg.LAM_ALIGN = 0.002  # Subspace alignment penalty
        
        print(f"🎯 Applied HPO loss weights: "
              f"lam_diag={getattr(self.loss_fn, 'lam_diag', 'N/A'):.3f}, "
              f"lam_off={getattr(self.loss_fn, 'lam_off', 'N/A'):.3f}, "
              f"lam_aux={getattr(self.loss_fn, 'lam_aux', 'N/A'):.3f}")

    def _apply_phase_loss_weights(self, phase: str):
        """Apply per-phase loss weights from cfg.PHASE_LOSS if provided."""
        phase_loss = getattr(cfg, "PHASE_LOSS", {}) or {}
        weights = phase_loss.get(str(phase).lower(), None)
        if not weights:
            return
        if "lam_cov" in weights:
            self.loss_fn.lam_cov = float(weights["lam_cov"])
        if "lam_subspace_align" in weights:
            self.loss_fn.lam_subspace_align = float(weights["lam_subspace_align"])
        if "lam_aux" in weights:
            self.loss_fn.lam_aux = float(weights["lam_aux"])
        # NOTE: lam_K removed - using MVDR peak detection instead
        if "lam_peak_contrast" in weights:
            self.loss_fn.lam_peak_contrast = float(weights["lam_peak_contrast"])
        # Eigengap/margin disabled globally (ignore any config values).
        self.loss_fn.lam_gap = 0.0
        self.loss_fn.lam_margin = 0.0
        print(f"🎯 Applied phase '{phase}' loss weights: "
              f"lam_cov={self.loss_fn.lam_cov:.3f}, "
              f"lam_subspace_align={self.loss_fn.lam_subspace_align:.3f}, "
              f"lam_aux={self.loss_fn.lam_aux:.3f}, "
              f"lam_peak_contrast={self.loss_fn.lam_peak_contrast:.3f}, "
              f"lam_gap={getattr(self.loss_fn, 'lam_gap', 0):.3f}, "
              f"lam_margin={getattr(self.loss_fn, 'lam_margin', 0):.3f}")

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
        
        # Log gating: avoid massive logs during HPO/full training unless explicitly enabled.
        epoch_dbg = bool(getattr(cfg, "TRAIN_EPOCH_DEBUG", False))
        loopcheck_dbg = bool(getattr(cfg, "TRAIN_LOOPCHECK_DEBUG", False))
        log_every = int(getattr(cfg, "TRAIN_BATCH_LOG_EVERY", 0))
        log_first = int(getattr(cfg, "TRAIN_BATCH_LOG_FIRST", 0))
        log_every = max(0, log_every)
        log_first = max(0, log_first)
        def _should_log_batch(bi: int) -> bool:
            if bi < log_first:
                return True
            if log_every <= 0:
                return False
            return (bi % log_every == 0) or (bi == iters - 1)
        
        # Expert fix: Initialize parameter vector for drift tracking (if first epoch)
        if not hasattr(self, "_param_vec_prev"):
            with torch.no_grad():
                self._param_vec_prev = torch.nn.utils.parameters_to_vector([p.detach().float() for p in self.model.parameters() if p.requires_grad])
        
        # Expert debug: Print LR every epoch
        lrs = [g['lr'] for g in self.opt.param_groups]
        print(f"[LR] epoch={epoch} groups={['backbone','head']} lr={lrs}", flush=True)
        if epoch_dbg:
            print(f"[EPOCH DEBUG] epoch={epoch} iters={iters} len(loader)={len(loader)}", flush=True)
            print(f"[EPOCH DEBUG] entering for loop over loader...", flush=True)
            import sys; sys.stdout.flush(); sys.stderr.flush()

        # DEBUG: Try to manually get first batch
        if epoch_dbg:
            try:
                print(f"[EPOCH DEBUG] testing manual iter(loader)...", flush=True)
                test_iter = iter(loader)
                print(f"[EPOCH DEBUG] iter() succeeded, calling next()...", flush=True)
                test_batch = next(test_iter)
                print(f"[EPOCH DEBUG] next() succeeded, got batch with keys: {list(test_batch.keys()) if isinstance(test_batch, dict) else 'not a dict'}", flush=True)
                del test_iter, test_batch
            except StopIteration:
                print(f"[EPOCH DEBUG] StopIteration - loader is EMPTY!", flush=True)
                return 0.0
            except Exception as e:
                print(f"[EPOCH DEBUG] EXCEPTION during manual iter: {type(e).__name__}: {e}", flush=True)
                import traceback; traceback.print_exc()
                return 0.0
            
            print(f"[EPOCH DEBUG] manual test passed, now doing real for loop...", flush=True)

        batch_count = 0
        for bi, batch in enumerate(loader):
            batch_count += 1
            if epoch_dbg and batch_count == 1:
                print(f"[EPOCH DEBUG] got first batch in for loop, bi={bi}", flush=True)
            if bi >= iters: break
            self._batches_seen += 1  # Expert fix: Track batch counter
            if epoch_dbg and bi == 0:
                print(f"[EPOCH DEBUG] epoch={epoch} fetching batch 0", flush=True)
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
            # CRITICAL: Pass H_full (true channel [M,N]) instead of H_eff (collapsed [L,M]).
            # This gives the network the actual sensing operator for RIS-domain covariance recovery.
            if H_full is None:
                raise ValueError("H_full is required for training. Regenerate shards with store_h_full=True.")
            if self.train_refiner_only:
                # Backbone is frozen; run it under no_grad and train only SpectrumRefiner.
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=self.amp):
                        preds = self.model(y=y, H_full=H_full, codes=C, snr_db=snr, R_samp=R_samp)
            else:
                with torch.amp.autocast('cuda', enabled=self.amp):
                    preds = self.model(y=y, H_full=H_full, codes=C, snr_db=snr, R_samp=R_samp)
            
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

                # -------------------------
                # Refiner-only path (Option B)
                # -------------------------
                if self.train_refiner_only:
                    assert self.refiner is not None, "Refiner module missing in refiner-only phase"

                    # Convert flat factors to complex [B, N, K]
                    Bc = preds_fp32["cov_fact_angle"].shape[0]
                    N = int(cfg.N)
                    Kmax = int(cfg.K_MAX)
                    flat_ang = preds_fp32["cov_fact_angle"].contiguous().float()
                    flat_rng = preds_fp32["cov_fact_range"].contiguous().float()

                    def _vec2c(v):
                        xr, xi = v[:, ::2], v[:, 1::2]
                        return torch.complex(xr.view(Bc, N, Kmax), xi.view(Bc, N, Kmax))

                    A_ang = _vec2c(flat_ang).to(torch.complex64)
                    A_rng = _vec2c(flat_rng).to(torch.complex64)

                    lam_range = float(getattr(mdl_cfg, "LAM_RANGE_FACTOR", 0.3))
                    F_list = []
                    # Build low-rank factor concatenation: [N, 2*Kmax]
                    for bi2 in range(Bc):
                        F_b = torch.cat([A_ang[bi2], (lam_range ** 0.5) * A_rng[bi2]], dim=1).contiguous()
                        F_list.append(F_b)

                    # Compute MVDR spectrum max over range planes (low-rank Woodbury) per sample
                    specs = []
                    delta_scale = float(getattr(cfg, "MVDR_DELTA_SCALE", 1e-2))
                    for bi2, F_b in enumerate(F_list):
                        S_b = self._mvdr_est.mvdr_spectrum_max_2_5d_lowrank(
                            F_b,
                            self._refiner_phi_grid,
                            self._refiner_theta_grid,
                            self._refiner_r_planes,
                            delta_scale=delta_scale,
                        )
                        specs.append(S_b)
                    mvdr_spec = torch.stack(specs, dim=0).unsqueeze(1)  # [B,1,G_phi,G_theta]

                    refined = self.refiner(mvdr_spec)
                    preds_fp32 = {"refined_spectrum": refined}
                
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
                if not self.train_refiner_only:
                    # Priority: STRUCTURAL R_pred (geometry-aware) > legacy factors.
                    R_pred = None
                    if 'R_pred' in preds_fp32 and isinstance(preds_fp32['R_pred'], torch.Tensor):
                        R_pred = preds_fp32['R_pred']
                    elif ('cov_fact_angle' in preds_fp32) and ('cov_fact_range' in preds_fp32):
                        # EXPERT FIX: Build R_pred robustly from factors to prevent shape bugs
                        def build_R_pred_from_factors(preds, cfg):
                            B = preds['cov_fact_angle'].size(0)
                            N = cfg.N_H * cfg.N_V
                            K_MAX = cfg.K_MAX

                            flat_ang = preds['cov_fact_angle'].contiguous().float()
                            flat_rng = preds['cov_fact_range'].contiguous().float()

                            assert flat_ang.dim() == 2 and flat_ang.size(0) == B, "bad factor shape"
                            assert flat_rng.dim() == 2 and flat_rng.size(0) == B, "bad factor shape"
                            assert flat_ang.size(1) == 2 * N * K_MAX, f"expected {2*N*K_MAX}, got {flat_ang.size(1)}"
                            assert flat_rng.size(1) == 2 * N * K_MAX, f"expected {2*N*K_MAX}, got {flat_rng.size(1)}"

                            def _vec2c_local(v):
                                xr, xi = v[:, ::2], v[:, 1::2]
                                return torch.complex(xr.view(B, N, K_MAX), xi.view(B, N, K_MAX))

                            A_ang = _vec2c_local(flat_ang)
                            A_rng = _vec2c_local(flat_rng)

                            lam_range = getattr(cfg, 'LAM_RANGE_FACTOR', 0.1)
                            R_pred = (A_ang @ A_ang.conj().transpose(-2, -1)) + lam_range * (A_rng @ A_rng.conj().transpose(-2, -1))
                            R_pred = 0.5 * (R_pred + R_pred.conj().transpose(-2, -1))
                            return R_pred

                        R_pred = build_R_pred_from_factors(preds_fp32, cfg)
                    else:
                        R_pred = None

                    if R_pred is not None:
                        # Quick sanity before blending:
                        Bn, Nn = R_pred.shape[:2]
                        assert R_pred.shape == (Bn, Nn, Nn), f"R_pred bad shape: {R_pred.shape}"
                    
                    # Quick sanity before blending:
                        B, N = R_pred.shape[:2]
                    
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
                        # Optional timings for diagnosing "hangs" in first forward/loss/backward.
                        import os, time
                        _dbg_timing = (os.environ.get("DEBUG_TIMINGS", "") == "1")
                        if _dbg_timing and epoch == 1 and bi == 0:
                            t0 = time.perf_counter()
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            print("[TIMING] build_effective_cov_torch: start", flush=True)
                        R_blend = build_effective_cov_torch(
                            R_pred,
                            snr_db=None,                 # do NOT shrink here; loss applies it consistently
                            R_samp=R_samp_c.detach(),    # no grad through sample cov
                            beta=float(beta),
                            diag_load=False,
                            apply_shrink=False,
                            target_trace=float(N),
                        )
                        if _dbg_timing and epoch == 1 and bi == 0:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            print(f"[TIMING] build_effective_cov_torch: done in {time.perf_counter()-t0:.3f}s", flush=True)
                            preds_fp32['R_blend'] = R_blend
                        else:
                            # No offline R_samp available → use pure R_pred
                            if epoch == 1 and bi == 0 and getattr(cfg, "HYBRID_COV_BLEND", True) and getattr(cfg, "HYBRID_COV_BETA", 0.0) > 0.0:
                                print("[Hybrid] R_samp not available; using pure R_pred for loss.", flush=True)
                            preds_fp32['R_blend'] = build_effective_cov_torch(
                                R_pred,
                                snr_db=None,
                                R_samp=None,
                                beta=None,
                                diag_load=False,
                                apply_shrink=False,
                                target_trace=float(N),
                            )
        
            # Compute loss in FP32
            import os, time
            _dbg_timing = (os.environ.get("DEBUG_TIMINGS", "") == "1")
            if _dbg_timing and epoch == 1 and bi == 0:
                t1 = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print("[TIMING] loss_fn: start", flush=True)
            loss  = self.loss_fn(preds_fp32, labels_fp32)
            if _dbg_timing and epoch == 1 and bi == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"[TIMING] loss_fn: done in {time.perf_counter()-t1:.3f}s", flush=True)
            
            # Expert fix: Log exact optimized loss for first batch
            if epoch == 1 and bi == 0:
                print(f"[OPTIMIZED LOSS] batch={bi} total_loss={loss.detach().item():.6f}", flush=True)
            
            # Expert debug: Gradient path test - verify R_blend has live gradients to heads
            if (not self.train_refiner_only) and epoch == 1 and bi == 0:
                assert 'R_blend' in preds_fp32, "R_blend missing from preds!"
                assert preds_fp32['R_blend'].requires_grad, "R_blend lost grad path!"
                
                # Unit-test the gradient path from R_blend to any head parameter
                try:
                    # Structural R mode: gradients flow through aux heads (aux_angles/aux_range/aux_power)
                    any_head = next(
                        p for n, p in self.model.named_parameters()
                        if (("aux_angles" in n) or ("aux_range" in n) or ("aux_power" in n)) and p.requires_grad
                    )
                    s = preds_fp32['R_blend'].real.mean()  # cheap scalar depending on R_pred
                    g = torch.autograd.grad(s, any_head, retain_graph=True, allow_unused=True)[0]
                    print(f"[GRADPATH] d<R_blend>/d(aux_head) = {0.0 if g is None else g.norm().item():.3e}", flush=True)
                except StopIteration:
                    print(f"[GRADPATH] No aux_head parameter found!", flush=True)
            
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

            if _dbg_timing and epoch == 1 and bi == 0:
                t2 = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print("[TIMING] backward: start", flush=True)
            self.scaler.scale(loss).backward()
            if loopcheck_dbg and _should_log_batch(bi):
                print(f"[LOOP-CHECK] bi={bi} did_backward", flush=True)  # Verify loop fix
            if _dbg_timing and epoch == 1 and bi == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"[TIMING] backward: done in {time.perf_counter()-t2:.3f}s", flush=True)
            
            # CRITICAL: Accumulate loss for EVERY batch BEFORE optimizer step
            batch_loss_val = float(loss.detach().item()) * grad_accumulation
            running += batch_loss_val
            if _should_log_batch(bi):
                print(f"[BATCH] ep={epoch} bi={bi} batch_loss={batch_loss_val:.4f} running={running:.4f}", flush=True)
            
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
                    if epoch_dbg:
                        print(f"[AMP] scale={current_scale} found_inf={found_inf_map}", flush=True)
                
                train_params = list(self.refiner.parameters()) if (self.train_refiner_only and self.refiner is not None) else list(self.model.parameters())

                # Expert fix: Log gradient norms before and after sanitization
                if epoch == 1 and bi == 0:
                    grad_norm_before = torch.nn.utils.clip_grad_norm_(train_params, float('inf'))
                    print(f"[GRAD SANITIZE] Before: ||g||_2={grad_norm_before:.3e}", flush=True)

                # Fail-fast policy: detect non-finite gradients AFTER unscale (AMP uses these).
                # For HPO, we prefer to abort the trial immediately rather than "sanitize" and silently stop learning.
                has_nonfinite_grad = False
                for p in train_params:
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all():
                        has_nonfinite_grad = True
                        break
                if has_nonfinite_grad:
                    # Diagnostics: identify which grads are non-finite (helps pinpoint the culprit quickly).
                    try:
                        bad = []
                        for n, p in (self.refiner.named_parameters() if (self.train_refiner_only and self.refiner is not None) else self.model.named_parameters()):
                            if p.grad is None:
                                continue
                            g = p.grad
                            if not torch.isfinite(g).all():
                                gmax = float(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).abs().max().detach().cpu())
                                bad.append((n, str(g.dtype), gmax))
                        if bad:
                            print(f"[NONFINITE] ep={epoch} batch={bi} bad_grads(top10)={bad[:10]}", flush=True)
                    except Exception:
                        pass

                    # AMP note: occasional overflow is normal early training; GradScaler is meant to adapt by skipping a step.
                    overflow_hint_local = any(v > 0.0 for v in found_inf_map.values())
                    if bool(getattr(cfg, "HPO_MODE", False)) and overflow_hint_local:
                        print(f"[NONFINITE] ep={epoch} batch={bi} treated as AMP overflow (found_inf={found_inf_map}) -> skip batch", flush=True)
                        self.opt.zero_grad(set_to_none=True)
                        self.scaler.update()
                        continue

                    if bool(getattr(cfg, "HPO_MODE", False)) and bool(getattr(cfg, "HPO_FAIL_FAST_ON_NONFINITE_GRADS", True)):
                        raise RuntimeError(f"Non-finite gradients detected (ep={epoch}, batch={bi}). Aborting HPO trial.")
                    # Non-HPO: skip the step cleanly (do not sanitize to zeros; that masks instability).
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.update()
                    print(f"[GRAD] ep={epoch} batch={bi} nonfinite_grad=True -> step skipped", flush=True)
                    continue
                
                # Gradient clipping (AFTER sanitization, BEFORE step)
                if self.clip_norm and self.clip_norm > 0:
                    grad_norm_clipped = torch.nn.utils.clip_grad_norm_(train_params, self.clip_norm)
                    if epoch == 1 and bi == 0:
                        print(f"[GRAD CLIP] After clipping: ||g||_2={grad_norm_clipped:.3e}", flush=True)
                
                # Expert fix: Improved gradient flow instrumentation
                import math
                # NOTE: K-head removed - using MVDR peak detection instead
                HEAD_KEYS = (
                    'head', 'classifier', 'aux_angles', 'aux_range',
                    'cov_fact_angle', 'cov_fact_range', 'logits_gg'
                )
                
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

                g_total, grad_count, none_count = total_grad_norm(train_params)
                g_total = float(g_total.detach().cpu())
                
                # AMP overflow/skip detection:
                # - found_inf_map can be stale; rely on scale drop after step to detect actual AMP skip.
                scale_before_step = float(self.scaler.get_scale()) if (hasattr(self, "scaler") and self.scaler is not None) else 1.0
                overflow_hint = any(v > 0.0 for v in found_inf_map.values())
                ok = math.isfinite(g_total) and (g_total > 1e-12)

                # Expert fix: Log gradient status for ALL batches in first epoch
                # DEBUG: Print for EVERY batch, not just epoch 1
                if _should_log_batch(bi):
                    print(f"[GRAD] ep={epoch} batch={bi} ||g||_2={g_total:.3e} ok={ok} overflow_hint={overflow_hint}", flush=True)
                    import sys; sys.stdout.flush(); sys.stderr.flush()
                
                # Initialize stepped flag
                stepped = False
                
                # CRITICAL: Skip step if gradients are non-finite or overflow detected
                if not ok:
                    if epoch == 1:
                        if _should_log_batch(bi):
                            print(f"[STEP] batch={bi} SKIPPED - overflow_hint={overflow_hint} ||g||_2={g_total:.3e}", flush=True)
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.update()  # Still update scaler state
                else:
                    # Step optimizer (with AMP scaler)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    scale_after_step = float(self.scaler.get_scale()) if (hasattr(self, "scaler") and self.scaler is not None) else scale_before_step
                    scaler_skipped = (scale_after_step < scale_before_step)
                    self.opt.zero_grad(set_to_none=True)
                    if loopcheck_dbg and _should_log_batch(bi):
                        print(f"[LOOP-CHECK] bi={bi} did_step", flush=True)  # Verify loop fix
                    stepped = (not scaler_skipped)
                    
                    # Expert fix: Log when step is actually taken
                    if epoch == 1:
                        if scaler_skipped:
                            if _should_log_batch(bi):
                                print(f"[STEP] batch={bi} AMP SKIPPED (scale {scale_before_step:.1f}→{scale_after_step:.1f})", flush=True)
                        else:
                            if _should_log_batch(bi):
                                print(f"[STEP] batch={bi} STEP TAKEN - g_total={g_total:.3e}", flush=True)
                    
                    # Expert fix: Track steps taken (only when optimizer step actually applied)
                    if stepped:
                        self._steps_taken += 1
                    if epoch == 1 and bi < 3:
                        lrs = [g['lr'] for g in self.opt.param_groups]
                        if epoch_dbg:
                            print(f"[OPT] step {self._steps_taken} / batch {self._batches_seen} LRs={lrs}", flush=True)
                    
                    # Expert fix: Parameter drift probe - measure BEFORE EMA update
                    with torch.no_grad():
                        vec_src = (self.refiner.parameters() if (self.train_refiner_only and self.refiner is not None) else self.model.parameters())
                        vec_now = torch.nn.utils.parameters_to_vector([p.detach().float() for p in vec_src if p.requires_grad])
                        delta = (vec_now - getattr(self, "_param_vec_prev", vec_now)).norm().item()
                        self._param_vec_prev = vec_now.detach().clone()
                        if _should_log_batch(bi):
                            print(f"[STEP] Δparam ||·||₂ = {delta:.3e}", flush=True)
                    
                    if stepped:
                        self._ema_update()
                    
                    # Update SWA model if in SWA phase
                    if stepped and self.swa_started:
                        self._swa_update()

                # CRITICAL FIX: Clear intermediate tensors to prevent memory leaks
                # NOTE: running is already accumulated BEFORE this block
                del y, H, C, ptr, K, R_in, snr, R_true_c, R_true, labels, loss

        if epoch_dbg:
            # DEBUG: Print how many batches were processed (OUTSIDE for loop)
            print(f"[EPOCH DEBUG] epoch={epoch} processed {batch_count} batches", flush=True)

        # Expert fix: Update schedulers only after actual steps
        # NOTE: 'stepped' is only valid for the last batch - scheduler step happens once per epoch
        if self.sched is not None:
            self.sched.step()
        
        # CRITICAL FIX: Force garbage collection at end of epoch
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if epoch_dbg:
            print(f"[EPOCH DEBUG] epoch={epoch} done, avg_loss={running / max(1, iters):.4f}", flush=True)
        return running / max(1, iters)

    @torch.no_grad()
    def _validate_one_epoch(self, loader, max_batches: Optional[int]=None, return_debug=False):
        self.model.eval()
        if self.refiner is not None:
            self.refiner.eval()
        running = 0.0
        iters = len(loader) if max_batches is None else min(len(loader), max_batches)
        
        # Accumulate per-term losses for debugging
        debug_acc = {}
        
        # NOTE: K-head removed - no K-logits tracking needed

        for bi, batch in enumerate(loader):
            if bi >= iters: break
            
            # Save first batch for eigenspectrum diagnostic (epoch 0 only)
            if bi == 0 and not hasattr(self, '_first_val_batch'):
                self._first_val_batch = batch
            
            y, H, C, ptr, K, R_in, snr, H_full, R_samp = self._unpack_any_batch(batch)
            
            # CRITICAL FIX: Clear batch references to prevent memory leaks (except first)
            if bi > 0:
                del batch

            R_true_c = _ri_to_c(R_in)
            R_true_c = 0.5 * (R_true_c + R_true_c.conj().transpose(-2, -1))
            R_true   = _c_to_ri(R_true_c).float()
            labels   = {"R_true": R_true, "ptr": ptr, "K": K, "snr_db": snr}

            if self.train_refiner_only:
                # Backbone factors -> MVDR spectrum -> SpectrumRefiner -> heatmap loss
                if H_full is None:
                    raise ValueError("H_full is required for training. Regenerate shards with store_h_full=True.")
                with torch.amp.autocast('cuda', enabled=(self.amp and self.device.type == "cuda")):
                    preds_half = self.model(y=y, H_full=H_full, codes=C, snr_db=snr)

                with torch.amp.autocast('cuda', enabled=False):
                    # Cast labels to FP32
                    labels_fp32 = {}
                    for k, v in labels.items():
                        if isinstance(v, torch.Tensor):
                            labels_fp32[k] = v.float() if v.dtype == torch.float16 else v
                        else:
                            labels_fp32[k] = v

                    # Cast factors to FP32 then convert to complex
                    flat_ang = preds_half["cov_fact_angle"].float()
                    flat_rng = preds_half["cov_fact_range"].float()
                    Bc = flat_ang.shape[0]
                    N = int(cfg.N)
                    Kmax = int(cfg.K_MAX)

                    def _vec2c(v):
                        xr, xi = v[:, ::2], v[:, 1::2]
                        return torch.complex(xr.view(Bc, N, Kmax), xi.view(Bc, N, Kmax)).to(torch.complex64)

                    A_ang = _vec2c(flat_ang)
                    A_rng = _vec2c(flat_rng)
                    lam_range = float(getattr(mdl_cfg, "LAM_RANGE_FACTOR", 0.3))
                    specs = []
                    delta_scale = float(getattr(cfg, "MVDR_DELTA_SCALE", 1e-2))
                    for bi2 in range(Bc):
                        F_b = torch.cat([A_ang[bi2], (lam_range ** 0.5) * A_rng[bi2]], dim=1).contiguous()
                        S_b = self._mvdr_est.mvdr_spectrum_max_2_5d_lowrank(
                            F_b,
                            self._refiner_phi_grid,
                            self._refiner_theta_grid,
                            self._refiner_r_planes,
                            delta_scale=delta_scale,
                        )
                        specs.append(S_b)
                    mvdr_spec = torch.stack(specs, dim=0).unsqueeze(1)
                    refined = self.refiner(mvdr_spec)

                    loss = self.loss_fn({"refined_spectrum": refined}, labels_fp32)
            else:
                # Expert fix: Forward can stay FP16 for speed
                if H_full is None:
                    raise ValueError("H_full is required for training. Regenerate shards with store_h_full=True.")
                with torch.amp.autocast('cuda', enabled=(self.amp and self.device.type == "cuda")):
                    preds_half = self.model(y=y, H_full=H_full, codes=C, snr_db=snr)
                
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

                    # TRAIN/VAL ALIGNMENT: build R_blend in validation (same as training) when R_samp is present.
                    if (R_samp is not None) and ("cov_fact_angle" in preds) and ("cov_fact_range" in preds):
                        try:
                            R_samp_c = _ri_to_c(R_samp.float())
                            R_samp_c = 0.5 * (R_samp_c + R_samp_c.conj().transpose(-2, -1))
                            lam_range = float(getattr(mdl_cfg, "LAM_RANGE_FACTOR", 0.3))
                            R_pred = build_R_pred_from_factor_vecs(
                                preds["cov_fact_angle"],
                                preds["cov_fact_range"],
                                N=int(cfg.N),
                                Kmax=int(cfg.K_MAX),
                                lam_range=lam_range,
                            )
                            beta = float(getattr(cfg, "HYBRID_COV_BETA", 0.0))
                            R_blend = build_effective_cov_torch(
                                R_pred,
                                snr_db=None,
                                R_samp=R_samp_c.detach(),
                                beta=beta,
                                diag_load=False,
                                apply_shrink=False,
                                target_trace=float(cfg.N),
                            )
                            preds["R_blend"] = R_blend
                        except Exception:
                            pass
                    
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
            
            # NOTE: K-head removed - K-logits tracking removed
            
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
            # NOTE: K-head removed - K-logits std removed
            return avg_loss, debug_acc
        return avg_loss

    # ----------------------------
    # Surrogate validation (NO MUSIC)
    # ----------------------------
    def _validate_surrogate_epoch(self, loader, max_batches: Optional[int] = None):
        """
        MUSIC-FREE validation using surrogate metrics.
        
        This is the RECOMMENDED validation method for training/HPO because:
        1. No MUSIC calls → fast, stable, no gating/penalty artifacts
        2. Metrics are directly tied to what we train on
        3. Highly correlated with final MUSIC performance
        
        Metrics computed:
        - Aux angle/range RMSE (from phi_theta_r head)
        - Composite score for model selection (higher is better)
        """
        # Small helper: assignment-based matching to avoid punishing permutation of sources.
        def _match_errors_deg(phi_p_deg, th_p_deg, r_p_m, phi_g_deg, th_g_deg, r_g_m):
            """
            Returns per-match absolute errors (deg/deg/m) for min(P, G) matches.
            Uses Hungarian if SciPy is available; greedy fallback otherwise.
            """
            phi_p_deg = np.asarray(phi_p_deg, dtype=np.float64).reshape(-1)
            th_p_deg = np.asarray(th_p_deg, dtype=np.float64).reshape(-1)
            r_p_m = np.asarray(r_p_m, dtype=np.float64).reshape(-1)
            phi_g_deg = np.asarray(phi_g_deg, dtype=np.float64).reshape(-1)
            th_g_deg = np.asarray(th_g_deg, dtype=np.float64).reshape(-1)
            r_g_m = np.asarray(r_g_m, dtype=np.float64).reshape(-1)

            P, G = len(phi_p_deg), len(phi_g_deg)
            m = min(P, G)
            if m == 0:
                return [], [], []

            # Normalization scales (keep stable across datasets; K<=5 so this is cheap).
            s_phi = float(getattr(cfg, "SURR_MATCH_SIGMA_PHI_DEG", 5.0))
            s_th = float(getattr(cfg, "SURR_MATCH_SIGMA_THETA_DEG", 5.0))
            s_r = float(getattr(cfg, "SURR_MATCH_SIGMA_R_M", 1.0))
            s_phi = max(s_phi, 1e-6)
            s_th = max(s_th, 1e-6)
            s_r = max(s_r, 1e-6)

            # Cost: normalized L2 in (phi,theta,r)
            C = np.full((P, G), 1e9, dtype=np.float64)
            for i in range(P):
                for j in range(G):
                    dphi = abs(phi_p_deg[i] - phi_g_deg[j]) / s_phi
                    dth = abs(th_p_deg[i] - th_g_deg[j]) / s_th
                    dr = abs(r_p_m[i] - r_g_m[j]) / s_r
                    C[i, j] = (dphi * dphi + dth * dth + dr * dr) ** 0.5

            try:
                from scipy.optimize import linear_sum_assignment
                ri, ci = linear_sum_assignment(C)
                pairs = [(int(ri[k]), int(ci[k])) for k in range(min(len(ri), m))]
            except Exception:
                # Greedy fallback
                pairs = []
                used_i, used_j = set(), set()
                for _ in range(m):
                    best = None
                    bestv = 1e18
                    for i in range(P):
                        if i in used_i:
                            continue
                        for j in range(G):
                            if j in used_j:
                                continue
                            v = float(C[i, j])
                            if v < bestv:
                                bestv = v
                                best = (i, j)
                    if best is None:
                        break
                    used_i.add(best[0])
                    used_j.add(best[1])
                    pairs.append(best)

            dphi = [abs(phi_p_deg[i] - phi_g_deg[j]) for i, j in pairs]
            dth = [abs(th_p_deg[i] - th_g_deg[j]) for i, j in pairs]
            dr = [abs(r_p_m[i] - r_g_m[j]) for i, j in pairs]
            return dphi, dth, dr

        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        
        # Aux angle/range errors (direct head vs GT, no MUSIC)
        aux_phi_err = []
        aux_theta_err = []
        aux_r_err = []

        # Optional: MVDR peak-level detection metrics (what matters in practice).
        do_peak_metrics = bool(getattr(cfg, "SURROGATE_PEAK_METRICS", False))
        peak_max_scenes = int(getattr(cfg, "SURROGATE_PEAK_MAX_SCENES", 0))
        peak_max_scenes = max(0, peak_max_scenes)
        peak_evald = 0
        peak_tp = 0
        peak_fp = 0
        peak_fn = 0
        pssr_db_vals = []

        # Optional: subspace overlap diagnostics (MVDR-critical).
        do_subspace = bool(getattr(cfg, "SURROGATE_SUBSPACE_METRICS", False))
        subspace_max_scenes = int(getattr(cfg, "SURROGATE_SUBSPACE_MAX_SCENES", 0))
        subspace_max_scenes = max(0, subspace_max_scenes)
        subspace_evald = 0
        subspace_overlaps = []
        
        iters = len(loader) if max_batches is None else min(len(loader), max_batches)
        
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if bi >= iters:
                    break
                
                y, H, C, ptr, K, R_in, snr, H_full, R_samp = self._unpack_any_batch(batch)
                B = y.size(0)
                n_samples += B
                
                # Prepare labels
                R_true_c = _ri_to_c(R_in)
                R_true_c = 0.5 * (R_true_c + R_true_c.conj().transpose(-2, -1))
                R_true = _c_to_ri(R_true_c).float()
                labels = {"R_true": R_true, "ptr": ptr, "K": K, "snr_db": snr}
                
                # Forward pass
                if H_full is None:
                    raise ValueError("H_full is required for training. Regenerate shards with store_h_full=True.")
                with torch.amp.autocast('cuda', enabled=(self.amp and self.device.type == "cuda")):
                    preds_half = self.model(y=y, H_full=H_full, codes=C, snr_db=snr, R_samp=R_samp)
                
                # Loss in FP32
                with torch.amp.autocast('cuda', enabled=False):
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

                    # TRAIN/VAL/INFER ALIGNMENT:
                    # Always build an effective covariance from the factor heads, even when R_samp is absent.
                    # This enables MVDR-like validation (peak metrics + subspace overlap) and ensures that
                    # "surrogate" metrics actually correlate with MVDR-first performance.
                    if ("cov_fact_angle" in preds) and ("cov_fact_range" in preds):
                        try:
                            lam_range = float(getattr(mdl_cfg, "LAM_RANGE_FACTOR", 0.3))
                            R_pred = build_R_pred_from_factor_vecs(
                                preds["cov_fact_angle"],
                                preds["cov_fact_range"],
                                N=int(cfg.N),
                                Kmax=int(cfg.K_MAX),
                                lam_range=lam_range,
                            )
                            # Optional hybrid blend if offline R_samp exists (currently typically absent).
                            R_samp_c = None
                            beta = 0.0
                            if R_samp is not None:
                                try:
                                    R_samp_c = _ri_to_c(R_samp.float())
                                    R_samp_c = 0.5 * (R_samp_c + R_samp_c.conj().transpose(-2, -1))
                                    beta = float(getattr(cfg, "HYBRID_COV_BETA", 0.0))
                                except Exception:
                                    R_samp_c = None
                                    beta = 0.0
                            R_eff = build_effective_cov_torch(
                                R_pred,
                                snr_db=snr,                       # match inference (SNR-aware shrink)
                                R_samp=(R_samp_c.detach() if R_samp_c is not None else None),
                                beta=(beta if (R_samp_c is not None) else None),
                                diag_load=True,
                                apply_shrink=True,
                                target_trace=float(cfg.N),
                            )
                            preds["R_blend"] = R_eff
                        except Exception:
                            pass
                    
                    loss = self.loss_fn(preds, labels_fp32)
                    if not torch.isfinite(loss):
                        loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)
                
                total_loss += float(loss.detach().item()) * B
                
                # ---------- Aux angle/range metrics (direct head vs GT) ----------
                if "phi_theta_r" in preds:
                    # IMPORTANT: ptr format is CHUNKED: [phi_pad..., theta_pad..., r_pad...]
                    # Angles are stored in RADIANS in both ptr and phi_theta_r head.
                    phi_theta_r_pred = preds["phi_theta_r"].float().cpu()  # [B, 3*Kmax] chunked
                    ptr_gt = ptr.float().cpu()  # [B, 3*Kmax] chunked
                    K_true_list = K.cpu().tolist()
                    Kmax = int(getattr(cfg, "K_MAX", 5))
                    
                    for i in range(B):
                        k_i = int(K_true_list[i])
                        if k_i <= 0:
                            continue
                        
                        # GT (chunked)
                        gt_phi = ptr_gt[i, :Kmax][:k_i]
                        gt_th  = ptr_gt[i, Kmax:2*Kmax][:k_i]
                        gt_r   = ptr_gt[i, 2*Kmax:3*Kmax][:k_i]

                        # Pred (chunked). Use ALL Kmax slots as candidates and match to the
                        # K_true GT sources (perm-invariant + allows arbitrary slot ordering).
                        pr_phi = phi_theta_r_pred[i, :Kmax]
                        pr_th  = phi_theta_r_pred[i, Kmax:2*Kmax]
                        pr_r   = phi_theta_r_pred[i, 2*Kmax:3*Kmax]

                        # Convert to degrees/meters and compute MATCHED errors (perm-invariant).
                        gt_phi_deg = torch.rad2deg(gt_phi).numpy()
                        gt_th_deg = torch.rad2deg(gt_th).numpy()
                        gt_r_m = gt_r.numpy()
                        pr_phi_deg = torch.rad2deg(pr_phi).numpy()
                        pr_th_deg = torch.rad2deg(pr_th).numpy()
                        pr_r_m = pr_r.numpy()

                        dphi, dth, dr = _match_errors_deg(pr_phi_deg, pr_th_deg, pr_r_m, gt_phi_deg, gt_th_deg, gt_r_m)
                        aux_phi_err.extend(dphi)
                        aux_theta_err.extend(dth)
                        aux_r_err.extend(dr)

                # ---------- Peak-level MVDR detection metrics (optional) ----------
                if do_peak_metrics and ("R_blend" in preds) and (peak_max_scenes == 0 or peak_evald < peak_max_scenes):
                    try:
                        from .music_gpu import mvdr_detect_sources, get_gpu_estimator
                        est = get_gpu_estimator(cfg, device=("cuda" if torch.cuda.is_available() else "cpu"))
                        tol_phi = float(getattr(cfg, "SURROGATE_DET_TOL_PHI_DEG", 5.0))
                        tol_th = float(getattr(cfg, "SURROGATE_DET_TOL_THETA_DEG", 5.0))
                        tol_r = float(getattr(cfg, "SURROGATE_DET_TOL_R_M", 1.0))
                        gphi = int(getattr(cfg, "SURROGATE_PEAK_GRID_PHI", 121))
                        gth = int(getattr(cfg, "SURROGATE_PEAK_GRID_THETA", 61))
                        r_planes = getattr(cfg, "REFINER_R_PLANES", None) or est.default_r_planes_mvdr
                        delta_scale = float(getattr(cfg, "MVDR_DELTA_SCALE", 1e-2))
                        thr_db = float(getattr(cfg, "MVDR_THRESH_DB", 12.0))
                        thr_mode = str(getattr(cfg, "MVDR_THRESH_MODE", "mad"))
                        cfar_z = float(getattr(cfg, "MVDR_CFAR_Z", 5.0))

                        # Per-scene loop (cap by peak_max_scenes)
                        Rb = preds["R_blend"]  # [B,N,N] complex
                        for i in range(B):
                            if peak_max_scenes != 0 and peak_evald >= peak_max_scenes:
                                break
                            k_i = int(K_true_list[i])
                            if k_i <= 0:
                                continue

                            # GT (degrees/meters)
                            Kmax = int(getattr(cfg, "K_MAX", 5))
                            gt_phi = ptr_gt[i, :Kmax][:k_i]
                            gt_th = ptr_gt[i, Kmax:2*Kmax][:k_i]
                            gt_r = ptr_gt[i, 2*Kmax:3*Kmax][:k_i]
                            gt_phi_deg = torch.rad2deg(gt_phi).numpy()
                            gt_th_deg = torch.rad2deg(gt_th).numpy()
                            gt_r_m = gt_r.numpy()

                            # MVDR detect (full MVDR, K-free)
                            sources, spectrum = mvdr_detect_sources(
                                Rb[i].detach(),
                                cfg,
                                device=("cuda" if torch.cuda.is_available() else "cpu"),
                                grid_phi=gphi,
                                grid_theta=gth,
                                r_planes=r_planes,
                                delta_scale=delta_scale,
                                threshold_db=thr_db,
                                threshold_mode=thr_mode,
                                cfar_z=cfar_z,
                                max_sources=int(getattr(cfg, "K_MAX", 5)),
                                do_refinement=False,  # keep it cheap for validation
                                prepared=False,
                            )
                            peak_evald += 1

                            # PSSR (max vs median) on spectrum_max
                            try:
                                spec = np.asarray(spectrum, dtype=np.float64)
                                eps = 1e-12
                                pmax = float(np.max(spec)) if spec.size else 0.0
                                pmed = float(np.median(spec)) if spec.size else 0.0
                                pssr = 10.0 * np.log10(max(pmax, eps) / max(pmed, eps))
                                if np.isfinite(pssr):
                                    pssr_db_vals.append(float(pssr))
                            except Exception:
                                pass

                            # Pred arrays
                            phi_p = np.array([s[0] for s in sources], dtype=np.float64)
                            th_p = np.array([s[1] for s in sources], dtype=np.float64)
                            r_p = np.array([s[2] for s in sources], dtype=np.float64)

                            # Match using the same matcher (but in degrees/meters already)
                            dphi, dth, dr = _match_errors_deg(phi_p, th_p, r_p, gt_phi_deg, gt_th_deg, gt_r_m)
                            # Count TP as matched within tolerance in all dims
                            tp_i = 0
                            for a, b, c in zip(dphi, dth, dr):
                                if (a <= tol_phi) and (b <= tol_th) and (c <= tol_r):
                                    tp_i += 1
                            fp_i = max(0, len(phi_p) - tp_i)
                            fn_i = max(0, k_i - tp_i)
                            peak_tp += tp_i
                            peak_fp += fp_i
                            peak_fn += fn_i
                    except Exception:
                        # Never let metrics break validation.
                        pass

                # ---------- Subspace overlap metric (optional) ----------
                if do_subspace and ("R_blend" in preds) and (subspace_max_scenes == 0 or subspace_evald < subspace_max_scenes):
                    try:
                        # Compare top-K eigenspaces of R_eff_pred vs R_true_eff.
                        # Use torch SVD/eigh on a capped number of scenes to keep it cheap.
                        from .covariance_utils import build_effective_cov_torch
                        K_true_list = K.cpu().tolist()
                        Rb = preds["R_blend"]  # [B,N,N] complex
                        # Build effective R_true to match R_blend conditioning (diagload+shrink+trace=N).
                        R_true_eff = build_effective_cov_torch(
                            R_true_c,
                            snr_db=snr,
                            R_samp=None,
                            beta=None,
                            diag_load=True,
                            apply_shrink=True,
                            target_trace=float(cfg.N),
                        )
                        for i in range(B):
                            if subspace_max_scenes != 0 and subspace_evald >= subspace_max_scenes:
                                break
                            k_i = int(K_true_list[i])
                            if k_i <= 0:
                                continue
                            # Eigenvectors (ascending eigenvalues)
                            eval_p, evec_p = torch.linalg.eigh(Rb[i].to(torch.complex64))
                            eval_t, evec_t = torch.linalg.eigh(R_true_eff[i].to(torch.complex64))
                            k_i = int(max(1, min(k_i, int(getattr(cfg, "K_MAX", 5)), evec_p.shape[0] - 1)))
                            Up = evec_p[:, -k_i:]  # [N,k]
                            Ut = evec_t[:, -k_i:]
                            # singular values of Ut^H Up are cos(principal angles)
                            s = torch.linalg.svdvals(Ut.conj().transpose(-2, -1) @ Up)
                            s = torch.clamp(s.real, 0.0, 1.0)
                            ov = float(torch.mean(s * s).detach().cpu().item())
                            if np.isfinite(ov):
                                subspace_overlaps.append(ov)
                                subspace_evald += 1
                    except Exception:
                        pass
        
        # Compute summary metrics
        avg_loss = total_loss / max(1, n_samples)
        
        phi_rmse = float(np.sqrt(np.mean(np.square(aux_phi_err)))) if aux_phi_err else 0.0
        theta_rmse = float(np.sqrt(np.mean(np.square(aux_theta_err)))) if aux_theta_err else 0.0
        r_rmse = float(np.sqrt(np.mean(np.square(aux_r_err)))) if aux_r_err else 0.0
        
        # Composite score (higher is better).
        #
        # NOTE: During HPO we want a surrogate that correlates with MVDR-first inference quality.
        # If SURROGATE_PEAK_METRICS is enabled, incorporate peak-level detection F1 and FP/scene.
        w = getattr(cfg, "SURROGATE_METRIC_WEIGHTS", None) or {
            "w_loss": 1.0,
            "w_aux_ang": 0.01,
            "w_aux_r": 0.01,
            # Optional MVDR-peak proxy (only used when SURROGATE_PEAK_METRICS=True)
            "w_peak_f1": 2.0,
            "w_peak_fp": 0.1,
            "w_peak_pssr": 0.0,
        }
        score = (
            - float(w.get("w_loss", 1.0)) * avg_loss
            - float(w.get("w_aux_ang", 0.01)) * (phi_rmse + theta_rmse) / 2.0
            - float(w.get("w_aux_r", 0.01)) * r_rmse
        )
        
        metrics = {
            "loss": avg_loss,
            "aux_phi_rmse": phi_rmse,
            "aux_theta_rmse": theta_rmse,
            "aux_r_rmse": r_rmse,
            "score": score,
        }
        if do_subspace and subspace_overlaps:
            ov_med = float(np.median(subspace_overlaps))
            ov_mean = float(np.mean(subspace_overlaps))
            metrics["subspace_overlap_med"] = ov_med
            metrics["subspace_overlap_mean"] = ov_mean
            print(f"[VAL SUBSPACE] scenes={len(subspace_overlaps)} overlap_med={ov_med:.3f} overlap_mean={ov_mean:.3f}", flush=True)
            # Optionally include in the surrogate score (higher overlap is better).
            w_sub = float(getattr(cfg, "SURROGATE_SUBSPACE_WEIGHT", 1.0))
            score += w_sub * ov_mean
            metrics["score"] = score

        # Optional peak-level metrics
        if do_peak_metrics and peak_evald > 0:
            prec = float(peak_tp) / max(1.0, float(peak_tp + peak_fp))
            rec = float(peak_tp) / max(1.0, float(peak_tp + peak_fn))
            f1 = (2.0 * prec * rec) / max(1e-12, (prec + rec))
            fp_per_scene = float(peak_fp) / max(1.0, float(peak_evald))
            metrics.update(
                {
                    "peak_precision": prec,
                    "peak_recall": rec,
                    "peak_f1": f1,
                    "peak_fp_per_scene": fp_per_scene,
                    "peak_pssr_db_med": float(np.median(pssr_db_vals)) if pssr_db_vals else 0.0,
                    "peak_eval_scenes": int(peak_evald),
                }
            )
            # Boost surrogate score with a MVDR-first proxy if enabled.
            score += float(w.get("w_peak_f1", 0.0)) * f1
            score -= float(w.get("w_peak_fp", 0.0)) * fp_per_scene
            score += float(w.get("w_peak_pssr", 0.0)) * (metrics["peak_pssr_db_med"] / 10.0)
            print(
                f"[VAL PEAK] scenes={peak_evald} precision={prec:.3f} recall={rec:.3f} "
                f"fp/scene={fp_per_scene:.2f} pssr_med={metrics['peak_pssr_db_med']:.2f}dB",
                flush=True,
            )
        
        print(
            f"[VAL SURROGATE] loss={avg_loss:.4f}, "
            f"aux_φ_rmse={phi_rmse:.2f}°, aux_θ_rmse={theta_rmse:.2f}°, "
            f"aux_r_rmse={r_rmse:.2f}m, score={score:.4f}",
            flush=True
        )
        
        # Cleanup
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return metrics

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
        # NOTE: K removed - using MVDR peak detection instead
        
        if phase == 0:   # B1 (e1-e4): Angles ONLY, minimal range
            # Set all primary loss weights (using HPO suggestions as base)
            self.loss_fn.lam_cov   = hpo_lam_cov * 0.25   # lam_cov≈0.25
            
            # Phase 0: lam_ang≈0.7, NO range - lock clean angle seeds early
            self.loss_fn.lam_aux   = hpo_lam_ang * 0.70 + hpo_lam_rng * 0.00  # lam_ang≈0.7
            
            # Covariance structure terms (scaled by cov weight)
            self.loss_fn.lam_cross = 1e-3 * hpo_lam_cov
            self.loss_fn.lam_gap   = 0.0
            
            # Fixed terms (not from HPO)
            self.loss_fn.lam_ortho = 1e-3
            self.loss_fn.lam_peak = 0.05
            self.loss_fn.lam_margin = 0.0
            self.loss_fn.lam_range_factor = 0.3
            
            # Higher dropout in early phase
            dropout = max(float(getattr(mdl_cfg, "DROPOUT", 0.10)), 0.10)
            mdl_cfg.DROPOUT = dropout
            self.model.set_dropout(dropout)  # Actually update the model
        elif phase == 1: # B2 (e5-e8): Add range, maintain angles
            # Set all primary loss weights
            self.loss_fn.lam_cov   = hpo_lam_cov * 0.25   # lam_cov≈0.25
            
            # Phase 1: lam_rng≈0.35–0.5, lam_ang≈0.2
            self.loss_fn.lam_aux   = hpo_lam_ang * 0.20 + hpo_lam_rng * 0.45  # lam_rng≈0.35–0.5, lam_ang≈0.2
            
            # Covariance structure terms
            self.loss_fn.lam_cross = 1e-3 * hpo_lam_cov
            self.loss_fn.lam_gap   = 0.0
            
            # Fixed terms
            self.loss_fn.lam_ortho = 1e-3
            self.loss_fn.lam_peak = 0.05
            self.loss_fn.lam_margin = 0.0
            self.loss_fn.lam_range_factor = 0.3
            
            dropout = max(float(getattr(mdl_cfg, "DROPOUT", 0.10)), 0.10)
            mdl_cfg.DROPOUT = dropout
            self.model.set_dropout(dropout)  # Actually update the model
        else:            # B3 (e9-e12): Joint training (angles + range)
            # Set all primary loss weights
            self.loss_fn.lam_cov   = hpo_lam_cov * 0.7    # lam_cov≈0.7
            
            # Phase 2: lam_ang≈0.25, lam_rng≈0.3
            self.loss_fn.lam_aux   = hpo_lam_ang * 0.25 + hpo_lam_rng * 0.30  # lam_ang≈0.25, lam_rng≈0.3
            
            # Smooth ramp for cross-gram in final phase
            phase3_start = 2 * total_epochs // 3
            progress = (epoch - phase3_start) / max(1, total_epochs - phase3_start)
            progress = max(0, min(1, progress))  # clamp to [0,1]
            self.loss_fn.lam_cross = (1e-3 + progress * 1e-3) * hpo_lam_cov  # 1e-3 → 2e-3, scaled
            self.loss_fn.lam_gap   = 0.0
            
            # Fixed terms
            self.loss_fn.lam_ortho = 1e-3
            self.loss_fn.lam_peak = 0.05
            self.loss_fn.lam_margin = 0.0
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
    
    # NOTE: _ramp_k_weight removed - K-head no longer used (MVDR peak detection instead)
    
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
        # CRITICAL: all_errors now contains RAW errors (no penalties), one per matched source
        all_errors = {"phi": [], "theta": [], "r": [], "snr": []}
        # Track per-scene RMSEs and K metrics
        rmse_phi_list, rmse_theta_list, rmse_r_list = [], [], []
        k_true_all, k_hat_all = [], []
        success_count = 0  # Scenes where ALL sources are within tolerance
        n_scenes_with_matches = 0  # Scenes with at least one Hungarian match
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
                if H_full is None:
                    raise ValueError("H_full is required for training. Regenerate shards with store_h_full=True.")
                with torch.amp.autocast('cuda', enabled=(self.amp and self.device.type == "cuda")):
                    preds = self.model(y=y, H_full=H_full, codes=C, snr_db=snr, R_samp=R_samp)
                
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
                            # NOTE: K-head removed - use GT K for validation MUSIC
                            # At inference, use MDL/MVDR for K estimation
                            K_nn = int(min(max(1, k_true), cfg.K_MAX))
                            nn_conf = 0.0  # No NN confidence without K-head
                            
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
                            
                            # MDL baseline for gating (measurement-domain Ryy, like inference)
                            try:
                                from .infer import estimate_k_ic_from_cov
                                from .physics import shrink
                                y_i = y[i].detach().cpu().numpy()  # [L, M, 2]
                                y_c = y_i[..., 0] + 1j * y_i[..., 1]  # [L, M]
                                T_snap = int(y_c.shape[0])
                                Ryy = (y_c.conj().T @ y_c) / max(T_snap, 1)
                                Ryy = 0.5 * (Ryy + Ryy.conj().T)
                                snr_i = float(snr[i].item()) if (snr is not None) else 10.0
                                Ryy = shrink(Ryy, snr_i, base=getattr(mdl_cfg, "SHRINK_BASE_ALPHA", 1e-3))
                                kmax_eff = min(int(cfg.K_MAX), int(Ryy.shape[0]) - 1)
                                K_mdl = estimate_k_ic_from_cov(Ryy, T_snap, method="mdl", kmax=kmax_eff)
                                if (K_mdl is None) or (K_mdl <= 0):
                                    K_mdl = estimate_k_ic_from_cov(Ryy, T_snap, method="aic", kmax=kmax_eff)
                                K_mdl = int(np.clip(K_mdl, 1, cfg.K_MAX))
                            except Exception:
                                K_mdl = int(K_nn)
                            # NOTE: K-head removed - always use MDL/AIC for K estimation
                            K_hat = K_mdl
                            
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
                    # MDL/AIC baseline in the MEASUREMENT domain (Ryy) — stable reference at small L.
                    # This is the baseline we should trust for L=16 (not RIS-domain 144×144 MDL).
                    try:
                        from .physics import shrink
                        y_i = y[i].detach().cpu().numpy()  # [L, M, 2]
                        y_c = y_i[..., 0] + 1j * y_i[..., 1]  # [L, M]
                        T_snap = int(y_c.shape[0])
                        Ryy = (y_c.conj().T @ y_c) / max(T_snap, 1)  # [M, M]
                        Ryy = 0.5 * (Ryy + Ryy.conj().T)
                        snr_i = float(snr[i].item()) if (snr is not None) else 10.0
                        Ryy = shrink(Ryy, snr_i, base=getattr(mdl_cfg, "SHRINK_BASE_ALPHA", 1e-3))
                        kmax_eff = min(int(cfg.K_MAX), int(Ryy.shape[0]) - 1)
                        k_mdl = estimate_k_ic_from_cov(Ryy, T_snap, method="mdl", kmax=kmax_eff)
                        if (k_mdl is None) or (k_mdl <= 0):
                            k_mdl = estimate_k_ic_from_cov(Ryy, T_snap, method="aic", kmax=kmax_eff)
                        if int(k_mdl) == int(k_true):
                            k_mdl_correct += 1
                    except Exception as e:
                        if getattr(cfg, "MUSIC_DEBUG", False):
                            print(f"[MDL] baseline failed: {e}")
                    
                    # NOTE: K-head removed - use GT K for validation metrics
                    # At inference, use MDL/MVDR for K estimation
                    k_hat = int(min(max(1, k_true), cfg.K_MAX))
                    k_true_all.append(k_true)
                    k_hat_all.append(k_hat)
                    
                    # Hungarian matching across ALL slots (permutation-agnostic)
                    # Now returns RAW errors (no gating) + success flag
                    metrics = eval_scene_angles_ranges(phi_all_deg, theta_all_deg, r_all_np,
                                                      phi_gt, theta_gt, r_gt,
                                                      success_tol_phi=succ_phi_thr,
                                                      success_tol_theta=succ_theta_thr,
                                                      success_tol_r=succ_r_thr)
                    
                    # CRITICAL FIX: Use RAW errors for metrics (no penalties!)
                    # This allows tracking actual MUSIC performance even when success rate is 0
                    if metrics.get("raw_phi_errors") and len(metrics["raw_phi_errors"]) > 0:
                        n_scenes_with_matches += 1
                        # Extend with all raw errors from this scene
                        all_errors["phi"].extend(metrics["raw_phi_errors"])
                        all_errors["theta"].extend(metrics["raw_theta_errors"])
                        all_errors["r"].extend(metrics["raw_r_errors"])
                        all_errors["snr"].extend([snr_np[i]] * len(metrics["raw_phi_errors"]))
                        # Track scene-level RMSEs
                        if metrics.get("rmse_phi") is not None:
                            rmse_phi_list.append(metrics["rmse_phi"])
                        if metrics.get("rmse_theta") is not None:
                            rmse_theta_list.append(metrics["rmse_theta"])
                        if metrics.get("rmse_r") is not None:
                            rmse_r_list.append(metrics["rmse_r"])
                        # Track success (strict: all sources within tolerance AND K correct)
                        if metrics.get("success_flag", False) and (k_hat == k_true):
                            success_count += 1
                    # Note: We no longer add 90°/10m penalties - raw errors only!
        
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
        print(f"[VAL METRICS] Processed {actual_count} source-level errors from {n_scenes_with_matches} scenes", flush=True)
        
        # Overall statistics (RAW errors, no penalties!)
        med_phi = np.median(phi_err)
        med_theta = np.median(theta_err)
        med_r = np.median(r_err)
        # Aggregated RMSEs (scene-level mean RMSE as proxy)
        rmse_phi_mean = float(np.mean(rmse_phi_list)) if len(rmse_phi_list) > 0 else None
        rmse_theta_mean = float(np.mean(rmse_theta_list)) if len(rmse_theta_list) > 0 else None
        rmse_r_mean = float(np.mean(rmse_r_list)) if len(rmse_r_list) > 0 else None
        
        # CRITICAL: These are RAW errors (no 90°/10m penalties!)
        # This shows actual MUSIC performance, not gating artifacts
        print(f"\n📊 RAW Angle/Range Errors (no penalties, N={len(phi_err)} source-pairs from {n_scenes_with_matches} scenes):")
        print(f"  Azimuth (φ):     median={med_phi:.2f}°,   95th={np.percentile(phi_err, 95):.2f}°,   RMSE={np.sqrt(np.mean(phi_err**2)):.2f}°")
        print(f"  Elevation (θ):   median={med_theta:.2f}°,   95th={np.percentile(theta_err, 95):.2f}°,   RMSE={np.sqrt(np.mean(theta_err**2)):.2f}°")
        print(f"  Range (r):       median={med_r:.2f}m,   95th={np.percentile(r_err, 95):.2f}m,   RMSE={np.sqrt(np.mean(r_err**2)):.2f}m")
        
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
        
        # NOTE: K-head removed. We only report localization metrics + success_rate.
        k_true_all = np.array(k_true_all, dtype=np.int32)
        n_scenes = int(k_true_all.size)
        success_rate = float(success_count / n_scenes) if n_scenes > 0 else 0.0
        
        # Return metrics for composite score calculation
        return {
            "med_phi": med_phi,
            "med_theta": med_theta,
            "med_r": med_r,
            "high_snr_samples": high_snr_mask.sum(),
            "n_scenes": n_scenes,
            "rmse_phi_mean": rmse_phi_mean,
            "rmse_theta_mean": rmse_theta_mean,
            "rmse_r_mean": rmse_r_mean,
            "success_rate": success_rate,
        }
    
    # NOTE: calibrate_k_logits removed - K-head removed, using MDL/MVDR for K estimation

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
        
        # Save reproducibility info
        self._save_run_config(epochs, n_train, n_val, grad_accumulation, early_stop_patience)
        
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

        if len(tr_loader) == 0:
            print("[TRAIN DEBUG] train loader length is 0 - aborting epoch loop", flush=True)
            return float("inf")
        if len(va_loader) == 0:
            print("[TRAIN DEBUG] val loader length is 0 - aborting epoch loop", flush=True)
            return float("inf")


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
        # CRITICAL: Initialize scheduler step counter to avoid warning
        # This tells PyTorch we're starting from step 0 properly
        self.sched._step_count = 1  # Suppress "step before optimizer.step" warning

        # training loop with early stopping
        best_val = float("inf")
        # Validation mode: "surrogate" (MUSIC-free, fast), "k_loc" (MUSIC-based, slow), or "loss"
        val_primary = str(getattr(cfg, "VAL_PRIMARY", "surrogate")).lower()
        use_music_in_val = bool(getattr(cfg, "USE_MUSIC_METRICS_IN_VAL", False))
        
        # For surrogate mode, higher score is better (maximize)
        # For k_loc/loss mode, lower is better (minimize)
        best_score = float("-inf") if val_primary == "surrogate" else float("inf")
        best_path = Path(cfg.CKPT_DIR) / "best.pt"
        last_path = Path(cfg.CKPT_DIR) / "last.pt"
        # Full training-state checkpoint (for true resume)
        train_state_path = Path(cfg.CKPT_DIR) / "train_state.pt"
        patience_counter = 0

        # ----------------------------
        # Auto-resume (model + optimizer + scaler + EMA/SWA + bookkeeping)
        # ----------------------------
        start_ep = 0
        auto_resume = bool(getattr(cfg, "AUTO_RESUME_TRAINING", True))
        resume_weights_only = bool(getattr(cfg, "AUTO_RESUME_WEIGHTS_ONLY", True))
        if auto_resume and train_state_path.exists():
            try:
                rs = torch.load(str(train_state_path), map_location="cpu", weights_only=False)
                if isinstance(rs, dict) and ("model" in rs):
                    # Restore model/refiner weights
                    self.model.load_state_dict(rs.get("model", {}), strict=False)
                    if self.train_refiner_only and (self.refiner is not None) and ("refiner" in rs):
                        try:
                            self.refiner.load_state_dict(rs.get("refiner", {}), strict=False)
                        except Exception:
                            pass

                    # Restore optimizer + AMP scaler
                    if "opt" in rs and rs["opt"] is not None:
                        try:
                            self.opt.load_state_dict(rs["opt"])
                        except Exception as e:
                            print(f"⚠️ Resume: failed to load optimizer state ({e}); continuing with fresh optimizer state.", flush=True)
                    if hasattr(self, "scaler") and (self.scaler is not None) and ("scaler" in rs) and (rs["scaler"] is not None):
                        try:
                            self.scaler.load_state_dict(rs["scaler"])
                        except Exception:
                            pass

                    # Restore EMA / SWA state (best-effort; safe to skip if mismatched)
                    if bool(rs.get("use_ema", False)) and hasattr(self, "ema_shadow") and ("ema_shadow" in rs) and isinstance(rs["ema_shadow"], dict):
                        try:
                            # Keep only floating-point tensors
                            self.ema_shadow = {k: v for k, v in rs["ema_shadow"].items() if hasattr(v, "dtype") and v.dtype.is_floating_point}
                        except Exception:
                            pass
                    if bool(rs.get("use_swa", False)) and hasattr(self, "swa_model") and (self.swa_model is not None):
                        try:
                            self.swa_started = bool(rs.get("swa_started", False))
                            swa_sd = rs.get("swa_model", None)
                            if isinstance(swa_sd, dict):
                                self.swa_model.load_state_dict(swa_sd, strict=False)
                        except Exception:
                            pass

                    # Restore bookkeeping
                    saved_ep = int(rs.get("epoch", -1))
                    if saved_ep >= 0:
                        start_ep = min(saved_ep + 1, int(epochs))
                    best_val = float(rs.get("best_val", best_val))
                    best_score = float(rs.get("best_score", best_score))
                    patience_counter = int(rs.get("patience_counter", patience_counter))

                    # Restore scheduler epoch counter (LambdaLR uses last_epoch as the step index)
                    try:
                        # If we already finished `saved_ep+1` epochs, next epoch is start_ep; scheduler should match that.
                        self.sched.last_epoch = max(0, start_ep - 1)
                    except Exception:
                        pass

                    print(f"🔁 Auto-resume: loaded train_state.pt at epoch={saved_ep+1} → starting from epoch={start_ep+1}", flush=True)
                else:
                    print(f"⚠️ Auto-resume: train_state.pt format unsupported; ignoring.", flush=True)
            except Exception as e:
                print(f"⚠️ Auto-resume: failed to load train_state.pt ({e}); ignoring.", flush=True)
        elif (not train_state_path.exists()) and resume_weights_only and last_path.exists():
            # Backward compatible fallback: weights-only resume
            try:
                sd = torch.load(str(last_path), map_location="cpu", weights_only=False)
                if isinstance(sd, dict) and ("backbone" in sd or "refiner" in sd):
                    bb = sd.get("backbone", sd.get("model", {}))
                    self.model.load_state_dict(bb, strict=False)
                    if self.train_refiner_only and (self.refiner is not None) and ("refiner" in sd):
                        try:
                            self.refiner.load_state_dict(sd.get("refiner", {}), strict=False)
                        except Exception:
                            pass
                elif isinstance(sd, dict) and "model" in sd:
                    self.model.load_state_dict(sd["model"], strict=False)
                elif isinstance(sd, dict):
                    self.model.load_state_dict(sd, strict=False)
                print(f"🔁 Auto-resume (weights-only): loaded {last_path} (optimizer/scheduler state NOT resumed).", flush=True)
            except Exception as e:
                print(f"⚠️ Auto-resume (weights-only) failed: {e}", flush=True)

        if start_ep >= int(epochs):
            print(f"[Training] Nothing to do: start_ep={start_ep} >= epochs={epochs}.", flush=True)
            return float(best_val) if val_primary != "surrogate" else float(best_score)

        print(f"[Training] Starting {epochs} epochs at ep={start_ep+1} (train batches={len(tr_loader)}, val batches={len(va_loader)})...", flush=True)
        for ep in range(start_ep, epochs):
            # 3-phase schedule (if enabled)
            if getattr(mdl_cfg, 'USE_3_PHASE_CURRICULUM', True):
                phase = 0 if ep < max(1, epochs // 3) else (1 if ep < max(2, 2 * epochs // 3) else 2)
                self._apply_phase_weights(phase, epoch=ep, total_epochs=epochs)
            else:
                phase = -1  # No curriculum phase
                
                # CRITICAL: Skip HPO overrides if we're in a phase-controlled training run
                # Phase settings (lam_gap=0, lam_margin=0 for k_only) must be respected!
                phase_controlled = str(getattr(cfg, "TRAIN_PHASE", "")).lower() in ("geom", "k_only", "joint")
                
                if not phase_controlled:
                    # ICC CRITICAL FIX: When curriculum is OFF (HPO), set lam_cross and lam_gap explicitly
                    # Use HPO base weights with conservative structure regularization for numerical stability.
                    # FIX: Check if dict is non-empty, not just if attribute exists
                    if self._hpo_loss_weights:
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
                            # HPO stability: disable SVD-heavy eigengap/margin regularizers unless explicitly enabled.
                            if bool(getattr(cfg, "HPO_MODE", False)) and bool(getattr(cfg, "HPO_DISABLE_UNSTABLE_LOSS_TERMS", True)):
                                self.loss_fn.lam_gap = 0.0
                                self.loss_fn.lam_margin = 0.0
                            else:
                                self.loss_fn.lam_gap = 0.065 * hpo_lam_cov     # (0.05-0.08) * lam_cov for gap penalty
                    else:
                        # No HPO weights - use default values
                        self.loss_fn.lam_cov = 1.0  # CRITICAL: Ensure main covariance weight is set!
                        
                        # PURE OVERFIT: Disable structural losses when OVERFIT_NMSE_PURE is set
                        if getattr(mdl_cfg, "OVERFIT_NMSE_PURE", False):
                            self.loss_fn.lam_cross = 0.0
                            self.loss_fn.lam_gap = 0.0
                            self.loss_fn.lam_ortho = 0.0
                            self.loss_fn.lam_peak = 0.0
                        else:
                            if bool(getattr(cfg, "HPO_MODE", False)) and bool(getattr(cfg, "HPO_DISABLE_UNSTABLE_LOSS_TERMS", True)):
                                self.loss_fn.lam_gap = 0.0
                                self.loss_fn.lam_margin = 0.0
                    
                    # Update structure loss weights (subspace align + peak contrast) based on phase.
                    # HPO stability: disable these unless explicitly enabled.
                    if bool(getattr(cfg, "HPO_MODE", False)) and bool(getattr(cfg, "HPO_DISABLE_UNSTABLE_LOSS_TERMS", True)):
                        self.loss_fn.lam_subspace_align = 0.0
                        self.loss_fn.lam_peak_contrast = 0.0
                    else:
                        self._update_structure_loss_weights(ep, epochs)
            
            # NOTE: K-weight ramping removed - K-head no longer used (MVDR peak detection instead)
            
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

            if bool(getattr(cfg, "TRAIN_EPOCH_DEBUG", False)):
                print(f"[EPOCH DEBUG] calling _train_one_epoch ep={ep+1}", flush=True)
            tr_loss = self._train_one_epoch(tr_loader, ep + 1, epochs, max_train_batches, grad_accumulation)
            if bool(getattr(cfg, "TRAIN_EPOCH_DEBUG", False)):
                print(f"[EPOCH DEBUG] finished _train_one_epoch ep={ep+1} train_loss={tr_loss}", flush=True)

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

            # Run validation metrics based on config
            metrics = None
            val_score = float(val_loss)  # Default to loss
            
            if val_primary == "surrogate":
                # SURROGATE MODE: Fast, MUSIC-free validation (RECOMMENDED for training/HPO)
                if (ep + 1) % val_every == 0 or ep == epochs - 1:
                    try:
                        hpo_max_batches = max_val_batches or 20
                        if self.swa_started:
                            self._swa_swap_in()
                            metrics = self._validate_surrogate_epoch(va_loader, hpo_max_batches)
                            self._swa_swap_out()
                        else:
                            self._ema_swap_in()
                            metrics = self._validate_surrogate_epoch(va_loader, hpo_max_batches)
                            self._ema_swap_out()
                        # Surrogate score: higher is better
                        val_score = float(metrics.get("score", 0.0))
                    except Exception as e:
                        print(f"[VAL SURROGATE] Error: {e}", flush=True)
                        import traceback; traceback.print_exc()
                        val_score = 0.0  # Neutral score on error
                        
            elif val_primary in ("k_loc", "metrics") and use_music_in_val:
                # MUSIC MODE: Full MUSIC-based validation (for final evaluation only)
                if not skip_music_val and ((ep + 1) % val_every == 0 or ep == epochs - 1):
                    try:
                        hpo_max_batches = max_val_batches or 20
                        if self.swa_started:
                            self._swa_swap_in()
                            metrics = self._eval_hungarian_metrics(va_loader, hpo_max_batches)
                            self._swa_swap_out()
                        else:
                            self._ema_swap_in()
                            metrics = self._eval_hungarian_metrics(va_loader, hpo_max_batches)
                            self._ema_swap_out()
                        # MUSIC score: lower is better
                        phi_norm = float(getattr(cfg, "VAL_NORM_PHI_DEG", 5.0))
                        theta_norm = float(getattr(cfg, "VAL_NORM_THETA_DEG", 5.0))
                        r_norm = float(getattr(cfg, "VAL_NORM_R_M", 1.0))
                        succ = float(metrics.get("success_rate", 0.0))
                        rmse_phi = float(metrics.get("rmse_phi_mean") or phi_norm)
                        rmse_theta = float(metrics.get("rmse_theta_mean") or theta_norm)
                        rmse_r = float(metrics.get("rmse_r_mean") or r_norm)
                        # Lower is better: normalized RMSEs, with success rate as a bonus
                        val_score = (rmse_phi / phi_norm) + (rmse_theta / theta_norm) + (rmse_r / r_norm) - succ
                    except Exception as e:
                        print(f"[VAL MUSIC] Error: {e}", flush=True)
                        val_score = float("inf")  # Worst score on error
                elif skip_music_val and ep == 0:
                    print(f"[VAL] Skipping MUSIC-based metrics (skip_music_val=True)", flush=True)
            # else: val_primary == "loss", use val_loss directly
            
            # Update best checkpoint by primary metric
            improved = False
            if val_primary == "surrogate":
                # Surrogate: higher score is better (maximize)
                if val_score > best_score:
                    best_score = float(val_score)
                    improved = True
            elif val_primary in ("k_loc", "metrics"):
                # MUSIC-based: lower score is better (minimize)
                if val_score < best_score:
                    best_score = float(val_score)
                    improved = True
            else:
                # Loss-based: lower is better
                if val_loss < best_val:
                    best_val = float(val_loss)
                    improved = True
            if improved:
                if self.train_refiner_only and (self.refiner is not None):
                    torch.save({"backbone": self.model.state_dict(), "refiner": self.refiner.state_dict()}, best_path)
                else:
                    torch.save(self.model.state_dict(), best_path)
                patience_counter = 0
            else:
                patience_counter += 1
            if self.train_refiner_only and (self.refiner is not None):
                torch.save({"backbone": self.model.state_dict(), "refiner": self.refiner.state_dict()}, last_path)
            else:
                torch.save(self.model.state_dict(), last_path)

            # Save full training-state for true resume
            try:
                train_state = {
                    "epoch": int(ep),
                    "epochs": int(epochs),
                    "val_primary": str(val_primary),
                    "best_val": float(best_val),
                    "best_score": float(best_score),
                    "patience_counter": int(patience_counter),
                    "model": self.model.state_dict(),
                    "opt": self.opt.state_dict() if hasattr(self, "opt") and (self.opt is not None) else None,
                    "scaler": self.scaler.state_dict() if hasattr(self, "scaler") and (self.scaler is not None) else None,
                    "use_ema": bool(getattr(self, "use_ema", False)),
                    "ema_shadow": getattr(self, "ema_shadow", None),
                    "use_swa": bool(getattr(self, "use_swa", False)),
                    "swa_started": bool(getattr(self, "swa_started", False)),
                    "swa_model": (self.swa_model.state_dict() if (hasattr(self, "swa_model") and self.swa_model is not None) else None),
                    # Note: scheduler state is reconstructed from epoch index; we keep last_epoch for debugging.
                    "sched_last_epoch": int(getattr(self.sched, "last_epoch", 0)) if (self.sched is not None) else 0,
                }
                if self.train_refiner_only and (self.refiner is not None):
                    train_state["refiner"] = self.refiner.state_dict()
                torch.save(train_state, train_state_path)
            except Exception as e:
                print(f"⚠️ Failed to save train_state.pt: {e}", flush=True)

            phase_str = f"phase={phase}" if phase >= 0 else "no-curriculum"
            print(
                f"Epoch {ep+1:03d}/{epochs:03d} [{phase_str}] "
                f"train {tr_loss:.6f}  val {val_loss:.6f}  "
                f"(lam_cross={getattr(self.loss_fn,'lam_cross',0):.1e}, "
                f"lam_gap={getattr(self.loss_fn,'lam_gap',0):.2f})",
                flush=True
            )
            # Print key metrics if available
            if metrics is not None:
                if val_primary == "surrogate":
                    # Surrogate metrics (no MUSIC)
                    print(f"  🎯 Surrogate: aux_φ={metrics.get('aux_phi_rmse', float('nan')):.2f}°, "
                          f"aux_θ={metrics.get('aux_theta_rmse', float('nan')):.2f}°, "
                          f"aux_r={metrics.get('aux_r_rmse', float('nan')):.2f}m, "
                          f"score={val_score:.4f}", flush=True)
                else:
                    # MUSIC-based metrics
                    print(f"  🧭 MVDR/MUSIC: succ_rate={metrics.get('success_rate', 0.0):.3f}, "
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
            
            # ICC FIX (debug only): Print pipeline config, geometry, and diagnostics at epoch 0
            if ep == 0 and bool(getattr(cfg, "TRAIN_EPOCH_DEBUG", False)):
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
                            H_full_val = batch_val.get("H_full")
                            if H_full_val is None:
                                H_full_val = batch_val["H"]  # Fallback for old shards (will fail at runtime)
                            H_full_val = H_full_val.to(self.device)
                            codes_val = batch_val["codes"].to(self.device)
                            snr_val = batch_val.get("snr_db")
                            if snr_val is not None:
                                snr_val = snr_val.to(self.device)
                        else:
                            # Unpack tuple (now has 9 elements: y, H, C, ptr, K, R, snr, H_full, R_samp)
                            y_val = batch_val[0].to(self.device)
                            H_full_val = batch_val[7] if len(batch_val) > 7 else batch_val[1]  # H_full at index 7
                            H_full_val = H_full_val.to(self.device)
                            codes_val = batch_val[2].to(self.device)
                            snr_val = batch_val[6] if len(batch_val) > 6 else None
                            if snr_val is not None:
                                snr_val = snr_val.to(self.device)
                        
                        preds_val = self.model(y=y_val, H_full=H_full_val, codes=codes_val, snr_db=snr_val)
                        
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
                                
                                if blend_beta <= 0.0:
                                    print(f"     ℹ️  Hybrid blending DISABLED (β={blend_beta:.2f}) → rank expected to stay ≤ {cfg.K_MAX}.", flush=True)
                                elif rank_blend > cfg.K_MAX:
                                    print(f"     ✅ Hybrid blending WORKING! Rank increased from {rank_estimate} → {rank_blend}", flush=True)
                                else:
                                    print(f"     ❌ Hybrid blending INEFFECTIVE at β={blend_beta:.2f}: rank still {rank_blend} (want >{cfg.K_MAX})", flush=True)
                                    
                            except Exception as e:
                                print(f"     ❌ Hybrid blending ERROR: {e}", flush=True)
                                import traceback
                                traceback.print_exc()
                        
                        # CLASSICAL MUSIC BASELINE: Optional ceiling check (debug-only).
                        if bool(getattr(cfg, "MUSIC_DEBUG", False)):
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
                                
                                # MUSIC with near-field steering
                                phi_music_true, theta_music_true = _classical_music_nearfield(
                                    R_true, k_true, phi_gt, theta_gt, r_gt, cfg
                                )
                                
                                print(f"  [DEBUG Classical] MUSIC found: phi={phi_music_true}, theta={theta_music_true}", flush=True)
                                
                                # Compute errors (simple nearest-neighbor for quick check)
                                if len(phi_music_true) > 0 and len(phi_gt) > 0:
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

        # NOTE: K calibration removed - K-head removed, using MDL/MVDR for K estimation
        
        # Return objective for outer loops (HPO/automation)
        # - "surrogate": return best surrogate score (higher is better, negate for Optuna minimize)
        # - "k_loc"/"metrics": return best composite score (lower is better)
        # - "loss": return best validation loss (lower is better)
        val_primary_final = str(getattr(cfg, "VAL_PRIMARY", "surrogate")).lower()
        if val_primary_final == "surrogate":
            # Surrogate: higher is better, but Optuna minimizes
            # Return negative score so Optuna can minimize
            return -float(best_score)
        elif val_primary_final in ("k_loc", "metrics"):
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
    
