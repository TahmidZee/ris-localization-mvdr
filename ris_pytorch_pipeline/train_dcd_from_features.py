import math, argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import types
import warnings
from .feature_dataset import RxFeatureDataset
from .configs import cfg

# Import from AI-Subspace-Methods (explicit path if needed)
from ai_subspace_methods.models_pack.dcd_music import DCDMUSIC
from ai_subspace_methods.system_model import SystemModel, SystemModelParams

def build_dcd(cfg, tau_in):
    # Create system model parameters
    params = SystemModelParams()
    params.N = cfg.N_H * cfg.N_V
    params.wavelength = cfg.WAVEL
    params.field_type = "near"
    
    sm = SystemModel(params)
    model = DCDMUSIC(system_model=sm, tau=tau_in,  # <- use C here
                     diff_method=("esprit", "music_1d"),
                     variant="small", norm_layer=True, batch_norm=False, psd_epsilon=1e-6)
    if hasattr(model, "update_train_mode"):
        model.update_train_mode("position")
    # DCD expects a [B, 2*N, N] surrogate view. Build it from our stack.
    model.expects_cov_stack = True
    
    def _pp_cov(x):
        # x: [B, C, L, L] with C = 2*(tau_max+1)  (our feature dump)
        if x.dim() != 4:
            # fall back to original (raw path) if someone passes [B,N,T]
            return x
        B, C, L, _ = x.shape
        if C < 2:
            raise ValueError(f"Need at least 2 channels (Re/Im R0); got C={C}")
        # take τ=0 only → channels [0,1] = [Re R0, Im R0]
        x0 = x[:, 0:2, :, :]                # [B, 2, L, L]
        # Model expects [B, tau, 2N, N] format, so we need to reshape appropriately
        # We have [B, 2, L, L] and need [B, 1, 2L, L] (tau=1, 2N=2L, N=L)
        x_view = x0.reshape(B, 1, 2*L, L)   # [B, 1, 2L, L]
        return x_view.contiguous().float()

    # Monkey-patch preprocessing for all branches that need it
    model.pre_processing = _pp_cov
    if hasattr(model, 'angle_branch') and hasattr(model.angle_branch, 'pre_processing'):
        model.angle_branch.pre_processing = _pp_cov
    if hasattr(model, 'range_branch') and hasattr(model.range_branch, 'pre_processing'):
        model.range_branch.pre_processing = _pp_cov
    
    return model

def _ensure_batched(x):
    """Make sure tensors are [B, ...]. If 1D, add batch dim."""
    if isinstance(x, torch.Tensor) and x.ndim == 1:
        return x.unsqueeze(0)
    return x

def mse_triplet_per_sample(pred_phi, pred_th, pred_r, gt_phi, gt_th, gt_r, K):
    """
    pred_*, gt_*: tensors with shape [B, >=K] (e.g., [B, K_MAX]) after padding or model output
    K: LongTensor [B] with true source counts
    Returns scalar loss averaged over valid samples (K>0).
    """
    B = K.shape[0]
    loss = pred_phi.new_tensor(0.0)
    valid = 0
    for b in range(B):
        k = int(K[b].item())
        if k <= 0:
            continue
        # allow model to output fewer than K (take min to be safe)
        k_eff = min(k, pred_phi.shape[-1], gt_phi.shape[-1])
        loss = loss + ((pred_phi[b, :k_eff] - gt_phi[b, :k_eff])**2).mean()
        loss = loss + ((pred_th [b, :k_eff] - gt_th [b, :k_eff])**2).mean()
        loss = loss + ((pred_r  [b, :k_eff] - gt_r  [b, :k_eff])**2).mean()
        valid += 1
    if valid == 0:
        return loss  # zero
    return loss / valid

def _forward_predict(model, Rx, K):
    """
    Calls model and returns (pred_phi, pred_th, pred_r) as float tensors with a batch dim.
    """
    out = model(Rx, number_of_sources=K)
    if not (isinstance(out, tuple) and len(out) >= 3):
        raise RuntimeError("DCDMUSIC forward did not return (phi, theta, r)")
    pphi, pth, pr = out[0], out[1], out[2]
    # Ensure batched shape [B, ...]
    pphi = _ensure_batched(pphi)
    pth  = _ensure_batched(pth)
    pr   = _ensure_batched(pr)
    # If model returns ragged last dim, we keep it; the per-sample slice by K handles it.
    return pphi.float(), pth.float(), pr.float()

def _k_eff(Ki, L):  # Enforce identifiability K < L
    return max(0, min(int(Ki), int(L) - 1))

def to_dcd_tau0_input(Rx):
    """Convert [B,C,L,L] to DCD's expected [B,1,2L,L] format using τ=0 only."""
    B, C, L, _ = Rx.shape
    if C < 2:
        raise ValueError(f"Need at least 2 channels (Re/Im R0); got C={C}")
    # take τ=0 only → channels [0,1] = [Re R0, Im R0]
    x0 = Rx[:, 0:2, :, :]                # [B, 2, L, L]
    # DCD's conv path expects [B, 1, 2L, L] 
    x_view = x0.reshape(B, 1, 2*L, L)    # [B, 1, 2L, L]
    return x_view.contiguous()

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
        if detach: V = V.detach()  # flip to True if grad still wobbles
        return w, V
    for branch in ("angle_branch","range_branch"):
        obj = getattr(model, branch, None)
        if obj is not None and hasattr(obj, "diff_method"):
            dm = obj.diff_method
            if hasattr(dm, "eigh"):
                dm.eigh = types.MethodType(lambda self, A: _safe_eigh(A), dm)
            # Loosen peak finder thresholds if available
            if hasattr(dm, "min_prominence"): dm.min_prominence = 0.0
            if hasattr(dm, "smooth_sigma"):   dm.smooth_sigma = getattr(dm, "smooth_sigma", 1.0) * 1.5

def _mse_triplet_k(pred, gt, k):
    pphi, pth, pr = pred  # 1D tensors (predictions for this sample)
    gphi, gth, gr = gt
    k = int(k)
    if k <= 0:
        return (pphi.sum()*0)  # 0 on the right device/dtype
    return ((pphi[:k]-gphi[:k])**2).mean() + ((pth[:k]-gth[:k])**2).mean() + ((pr[:k]-gr[:k])**2).mean()

def mse_triplet(pred, gt):
    """Simple MSE loss for (phi, theta, r) triplets."""
    pphi, pth, pr = pred
    gphi, gth, gr = gt
    return torch.nn.functional.mse_loss(pphi, gphi) + torch.nn.functional.mse_loss(pth, gth) + torch.nn.functional.mse_loss(pr, gr)

def train_epoch(model, loader, opt, device, use_float64: bool):
    model.train()
    tot = 0.0
    for b in loader:
        dtype = torch.float64 if use_float64 else torch.float32
        Rx = b["Rx"].to(device=device, dtype=dtype)                 # [B,C,L,L]
        K  = b["K"].to(device=device, dtype=torch.long)
        g = (b["phi"].to(device=device, dtype=dtype),
             b["theta"].to(device=device, dtype=dtype),
             b["r"].to(device=device, dtype=dtype))
        
        opt.zero_grad()
        # τ=0 mapping to [B,1,2L,L] keeps dtype
        Rx_tau0 = to_dcd_tau0_input(Rx)
        # per-sample K clamp if you have it; otherwise pass K
        out = model(Rx_tau0, number_of_sources=K)
        # normalize output
        if not (isinstance(out, tuple) and len(out) >= 3):
            raise RuntimeError("DCDMUSIC forward did not return (phi,theta,r)")
        pred = (out[0].to(dtype).reshape(-1),
                out[1].to(dtype).reshape(-1),
                out[2].to(dtype).reshape(-1))
        # your loss (masked/K-aware if you implemented it)
        loss = mse_triplet(pred, g)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # optional, safer
        opt.step()
        tot += float(loss.item())
    return tot / max(1, len(loader))

@torch.no_grad()
def eval_epoch(model, loader, device, use_float64: bool):
    model.eval()
    tot = 0.0
    for b in loader:
        dtype = torch.float64 if use_float64 else torch.float32
        Rx = b["Rx"].to(device=device, dtype=dtype)
        K  = b["K"].to(device=device, dtype=torch.long)
        g = (b["phi"].to(device=device, dtype=dtype),
             b["theta"].to(device=device, dtype=dtype),
             b["r"].to(device=device, dtype=dtype))
        Rx_tau0 = to_dcd_tau0_input(Rx)
        out = model(Rx_tau0, number_of_sources=K)
        pred = (out[0].to(dtype).reshape(-1),
                out[1].to(dtype).reshape(-1),
                out[2].to(dtype).reshape(-1))
        loss = mse_triplet(pred, g)
        tot += float(loss.item())
    return tot / max(1, len(loader))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_final/baselines/features_dcd_nf")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_dir", default="checkpoints/dcd")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--dtype", choices=["float32","float64"], default="float64")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = RxFeatureDataset("train", root=args.root, use_labels=True)
    va = RxFeatureDataset("val",   root=args.root, use_labels=True)
    
    # infer feature channels from one sample
    sample = tr[0]
    C = int(sample["Rx"].shape[0])   # channels = 2*(tau_max+1)
    L = int(sample["Rx"].shape[1])   # sensors per dimension (just FYI)
    print(f"[DCD] Feature channels: {C}, Sensor grid: {L}x{L}")
    
    # DCD will use only τ=0 (first 2 channels) and reshape to [B, 1, 2*L, L]
    tau_in_dcd = 1  # DCD expects tau=1 after preprocessing (single lag)
    print(f"[DCD] DCD tau parameter: {tau_in_dcd} (using τ=0 only, reshaped to [B,1,2L,L])")
    
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, pin_memory=True, persistent_workers=False, drop_last=False)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False,
                    num_workers=args.workers, pin_memory=True, persistent_workers=False, drop_last=False)

    model = build_dcd(cfg, tau_in=tau_in_dcd).to(device)
    if args.dtype == "float64":
        model = model.double()  # parameters & buffers to float64
    _patch_safe_eigh(model, eps=1e-6, detach=False)
    
    # Silence MUSIC peak finder warnings for cleaner training logs
    warnings.filterwarnings("once", message="MUSIC._peak_finder_1d: No peaks were found")
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = math.inf
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.save_dir) / "dcd_all.pt"

    use_float64 = (args.dtype == "float64")
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, tl, opt, device, use_float64)
        va_loss = eval_epoch(model, vl, device, use_float64)
        print(f"[DCD] epoch {ep:03d}  train {tr_loss:.4f}  val {va_loss:.4f}")
        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(),
                        "dtype": args.dtype,
                        "cfg": dict(
                            N_H=cfg.N_H, N_V=cfg.N_V, d_H=cfg.d_H, d_V=cfg.d_V,
                            WAVEL=cfg.WAVEL, K_MAX=cfg.K_MAX, TAU=getattr(cfg, 'TAU', 12)
                        )}, ckpt)
            print("  ✓ saved", ckpt)

if __name__ == "__main__":
    main()
