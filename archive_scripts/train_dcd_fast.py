"""
High-performance DCD training with batched K processing and optimized data loading.
Expected speedup: 10-50x over per-sample processing.
"""
import math, argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import types
import warnings
import time

# Fast imports
from fast_feature_dataset import FastRxFeatureDataset
from batch_k_sampler import ByKBatchSampler

# Original imports
from ris_pytorch_pipeline.configs import cfg
from ai_subspace_methods.models_pack.dcd_music import DCDMUSIC
from ai_subspace_methods.system_model import SystemModel, SystemModelParams

def _k_eff(Ki, L):
    """Enforce identifiability K < L"""
    return max(0, min(int(Ki), int(L) - 1))

def _patch_safe_eigh(model, eps=1e-6, detach=False):
    """Patch model's eigen-decomposition to be stable and differentiable."""
    def _safe_eigh(A):
        Hh = 0.5*(A + A.conj().transpose(-1,-2))
        # Normalize and add jitter for stability
        diag_mean = Hh.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1, keepdim=True).unsqueeze(-1)
        Hh = Hh / (diag_mean + 1e-9)
        I  = torch.eye(Hh.shape[-1], device=A.device, dtype=Hh.dtype)
        Hh = (Hh + eps*I)
        w, V = torch.linalg.eigh(Hh)
        if detach: V = V.detach()
        return w, V
        
    for branch in ("angle_branch","range_branch"):
        obj = getattr(model, branch, None)
        if obj is not None and hasattr(obj, "diff_method"):
            dm = obj.diff_method
            if hasattr(dm, "eigh"):
                dm.eigh = types.MethodType(lambda self, A: _safe_eigh(A), dm)
            # Loosen peak finder thresholds
            if hasattr(dm, "min_prominence"): dm.min_prominence = 0.0
            if hasattr(dm, "smooth_sigma"):   dm.smooth_sigma = getattr(dm, "smooth_sigma", 1.0) * 1.5

def build_dcd_fast(cfg, tau_in):
    """Build DCD model with optimized preprocessing."""
    params = SystemModelParams()
    params.N = cfg.N_H * cfg.N_V
    params.wavelength = cfg.WAVEL
    params.field_type = "near"
    
    sm = SystemModel(params)
    model = DCDMUSIC(system_model=sm, tau=tau_in,
                     diff_method=("esprit", "music_1d"),
                     variant="small", norm_layer=True, batch_norm=False, psd_epsilon=1e-6)
    
    if hasattr(model, "update_train_mode"):
        model.update_train_mode("position")
    
    def _pp_cov(x):
        if x.dim() != 4:
            return x
        B, C, L, _ = x.shape
        if C < 2:
            raise ValueError(f"Need at least 2 channels (Re/Im R0); got C={C}")
        # Extract τ=0 and reshape for DCD
        x0 = x[:, 0:2, :, :]  # [B, 2, L, L]
        x_view = x0.reshape(B, 1, 2*L, L)  # [B, 1, 2L, L]
        return x_view.contiguous().float()

    # Monkey-patch all preprocessing
    model.pre_processing = _pp_cov
    if hasattr(model, 'angle_branch') and hasattr(model.angle_branch, 'pre_processing'):
        model.angle_branch.pre_processing = _pp_cov
    if hasattr(model, 'range_branch') and hasattr(model.range_branch, 'pre_processing'):
        model.range_branch.pre_processing = _pp_cov
    
    return model

def train_epoch_batched(model, loader, opt, device):
    """Batched training with uniform K per batch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in loader:
        Rx = batch["Rx"].to(device).float()     # [B, C, L, L]
        K = batch["K"].to(device).long()        # [B] (all same K)
        phi = batch["phi"].to(device).float()   # [B, KMAX]
        theta = batch["theta"].to(device).float()
        r = batch["r"].to(device).float()
        
        B, _, L, _ = Rx.shape
        k_batch = _k_eff(K[0].item(), L)  # All K are same in this batch
        
        # Verify K uniformity (safety check)
        if not torch.all(K == K[0]):
            print(f"Warning: Non-uniform K in batch: {K}")
            continue
        
        opt.zero_grad()
        
        try:
            # Single batched forward pass!
            out = model(Rx, number_of_sources=k_batch)
            
            if isinstance(out, (list, tuple)) and len(out) >= 2 and out[0] is not None:
                # Extract predictions
                phi_pred = out[0]  # [B, k_batch]
                
                # Compute loss (MSE on valid predictions)
                valid_k = min(k_batch, phi.shape[1])
                if valid_k > 0:
                    loss = torch.nn.functional.mse_loss(
                        phi_pred[:, :valid_k], 
                        phi[:, :valid_k]
                    )
                    loss.backward()
                    total_loss += loss.item()
                    
            opt.step()
            num_batches += 1
            
        except Exception as e:
            print(f"Batch failed: {e}")
            continue
    
    return total_loss / max(1, num_batches)

@torch.no_grad()
def eval_epoch_batched(model, loader, device):
    """Batched evaluation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in loader:
        Rx = batch["Rx"].to(device).float()
        K = batch["K"].to(device).long()
        phi = batch["phi"].to(device).float()
        
        B, _, L, _ = Rx.shape
        k_batch = _k_eff(K[0].item(), L)
        
        if not torch.all(K == K[0]):
            continue
        
        try:
            out = model(Rx, number_of_sources=k_batch)
            
            if isinstance(out, (list, tuple)) and len(out) >= 2 and out[0] is not None:
                phi_pred = out[0]
                valid_k = min(k_batch, phi.shape[1])
                if valid_k > 0:
                    loss = torch.nn.functional.mse_loss(
                        phi_pred[:, :valid_k], 
                        phi[:, :valid_k]
                    )
                    total_loss += loss.item()
                    
            num_batches += 1
            
        except Exception as e:
            continue
    
    return total_loss / max(1, num_batches)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_final/baselines/features_dcd_nf")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)  # Larger batches for efficiency
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_dir", default="checkpoints/dcd_fast")
    ap.add_argument("--workers", type=int, default=1)  # Cluster limitation
    ap.add_argument("--cache_size", type=int, default=20)  # Chunk cache size
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Fast dataset with caching
    tr = FastRxFeatureDataset("train", root=args.root, use_labels=True, cache_size=args.cache_size)
    va = FastRxFeatureDataset("val", root=args.root, use_labels=True, cache_size=args.cache_size//2)
    
    # Batched samplers (group by K)
    tr_sampler = ByKBatchSampler(tr, batch_size=args.bs, shuffle=True)
    va_sampler = ByKBatchSampler(va, batch_size=args.bs, shuffle=False)
    
    tl = DataLoader(tr, batch_sampler=tr_sampler, num_workers=args.workers, pin_memory=True)
    vl = DataLoader(va, batch_sampler=va_sampler, num_workers=args.workers, pin_memory=True)
    
    # Build model
    sample = tr[0]
    L = int(sample["Rx"].shape[1])
    print(f"[FastDCD] Sensor grid: {L}x{L}")
    
    model = build_dcd_fast(cfg, tau_in=1).to(device)
    _patch_safe_eigh(model, eps=1e-6, detach=False)
    
    warnings.filterwarnings("once", message="MUSIC._peak_finder_1d: No peaks were found")
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_loss = math.inf
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training: {len(tr)} samples, ~{len(tl)} batches/epoch")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        tr_loss = train_epoch_batched(model, tl, opt, device)
        va_loss = eval_epoch_batched(model, vl, device)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}: train={tr_loss:.6f}, val={va_loss:.6f}, time={epoch_time:.1f}s")
        
        if va_loss < best_loss:
            best_loss = va_loss
            torch.save(model.state_dict(), Path(args.save_dir) / "best.pt")
    
    print(f"Training complete! Best val loss: {best_loss:.6f}")

if __name__ == "__main__":
    main()
