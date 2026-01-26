import numpy as np
import torch
from .configs import cfg, mdl_cfg
from .physics import alpha_from_snr_db


# ---------------------- Torch helpers ----------------------

def hermitize_torch(R: torch.Tensor) -> torch.Tensor:
    return 0.5 * (R + R.conj().transpose(-2, -1))

def trace_norm_torch(R: torch.Tensor, target_trace: float = None) -> torch.Tensor:
    """
    Hermitize and trace-normalize a batch of complex covariances.
    Default target trace = cfg.N (MUSIC convention in pipeline).
    """
    if target_trace is None:
        target_trace = float(getattr(cfg, "N", R.shape[-1]))
    R = hermitize_torch(R)
    tr = torch.diagonal(R, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-9)
    return R * (target_trace / tr).view(-1, 1, 1)

def shrink_torch(R: torch.Tensor, snr_db: torch.Tensor, base_alpha: float = None) -> torch.Tensor:
    """
    Torch-native shrinkage:
      R_shrunk = (1 - alpha) R + alpha * mu I,  where mu = trace(R)/N

    Consistency note:
    - We mirror `physics.shrink()` semantics: start from the SNR-aware piecewise mapping
      (scaled by sqrt(16/L)), then scale by (base_alpha / 1e-3).
    """
    if base_alpha is None:
        base_alpha = float(getattr(mdl_cfg, "SHRINK_BASE_ALPHA", 1e-3))
    B, N, _ = R.shape
    device = R.device
    dtype = R.dtype
    snr_db = snr_db.to(device=device, dtype=torch.float32).view(-1)

    # Build the base alpha via the same piecewise mapping used in NumPy.
    # Then apply L-aware scaling and the (base/1e-3) scale factor.
    # Piecewise thresholds: >=15, >=8, >=3, >=-2, else
    a = torch.empty_like(snr_db)
    a = torch.where(snr_db >= 15.0, torch.tensor(0.02, device=device), a)
    a = torch.where((snr_db < 15.0) & (snr_db >= 8.0), torch.tensor(0.04, device=device), a)
    a = torch.where((snr_db < 8.0) & (snr_db >= 3.0), torch.tensor(0.07, device=device), a)
    a = torch.where((snr_db < 3.0) & (snr_db >= -2.0), torch.tensor(0.10, device=device), a)
    a = torch.where(snr_db < -2.0, torch.tensor(0.14, device=device), a)

    L = float(max(1, int(getattr(cfg, "L", 16))))
    scale_L = float(np.sqrt(16.0 / L))
    scale_base = float(base_alpha) / 1e-3
    alpha = (a * float(scale_L) * float(scale_base)).clamp(min=1e-4, max=0.25).view(B, 1, 1)
    tr = torch.diagonal(R, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-9)  # [B]
    mu = (tr / float(N)).view(B, 1, 1)
    I = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, N, N)
    return (1.0 - alpha) * R + alpha * mu * I

def build_effective_cov_torch(
    R_pred: torch.Tensor,
    snr_db: torch.Tensor = None,
    R_samp: torch.Tensor = None,
    beta: float = None,
    diag_load: bool = True,
    apply_shrink: bool = True,
    target_trace: float = None,
) -> torch.Tensor:
    """
    Unified builder for the effective covariance that MUSIC will consume.
    Steps:
      - hermitize + trace-normalize R_pred
      - optional hybrid blend with R_samp (normalize both first, then blend)
      - re-normalize after blend (convex combo preserves trace, but hermitize may shift)
      - optional diagonal loading
      - optional SNR-aware shrinkage (torch-native, per-sample)
      - FINAL trace normalization to ensure tr(R) = target_trace exactly
    """
    if target_trace is None:
        target_trace = float(getattr(cfg, "N", R_pred.shape[-1]))
    B, N, _ = R_pred.shape
    R = trace_norm_torch(R_pred, target_trace=target_trace)

    if (R_samp is not None) and (beta is not None) and (beta > 0.0):
        # Ensure R_samp dtype/device matches and normalize before blending
        R_samp = R_samp.to(R.dtype).to(R.device)
        R_samp = trace_norm_torch(R_samp, target_trace=target_trace)
        R = (1.0 - float(beta)) * R + float(beta) * R_samp
        # Hermitize after blending (defensive, should already be close to hermitian)
        R = hermitize_torch(R)
        # Convex combination preserves trace in theory, but numerical noise + hermitize may shift it
        # Re-normalize to ensure exact trace
        R = trace_norm_torch(R, target_trace=target_trace)

    if diag_load:
        eps_cov = float(getattr(cfg, "C_EPS", 1e-3))
        R = R + eps_cov * torch.eye(N, device=R.device, dtype=R.dtype).unsqueeze(0).expand(B, N, N)
        # Diagonal loading adds eps*N to trace, re-normalize
        R = trace_norm_torch(R, target_trace=target_trace)

    if apply_shrink and (snr_db is not None):
        R = shrink_torch(R, snr_db)
        # Shrinkage formula: (1-α)R + α*μ*I preserves trace if μ = tr(R)/N, which it does
        # So no re-norm needed here (and we want to preserve the shrinkage effect)

    # Optional sanity checks (useful for catching silent numeric corruption early).
    if bool(getattr(cfg, "COV_SANITY_CHECK", False)):
        try:
            tol_trace = float(getattr(cfg, "COV_SANITY_TRACE_RTOL", 1e-2))
            tol_herm = float(getattr(cfg, "COV_SANITY_HERMITIAN_ATOL", 1e-3))
            # Finite
            if not torch.isfinite(R.real).all() or not torch.isfinite(R.imag).all():
                raise RuntimeError("non-finite entries")
            # Hermitian (relative to scale)
            anti = R - R.conj().transpose(-2, -1)
            anti_max = float(torch.max(torch.abs(anti)).detach().cpu().item())
            scale = float(torch.max(torch.abs(R)).detach().cpu().item())
            scale = max(scale, 1e-12)
            if anti_max > (tol_herm * scale + 1e-12):
                raise RuntimeError(f"not Hermitian enough (anti_max={anti_max:.3e}, scale={scale:.3e})")
            # Trace close to target
            tr = torch.diagonal(R, dim1=-2, dim2=-1).real.sum(-1)  # [B]
            target = float(target_trace)
            tr_rel = torch.max(torch.abs(tr - target) / max(target, 1e-9)).detach().cpu().item()
            if float(tr_rel) > tol_trace:
                raise RuntimeError(f"bad trace (max_rel_err={float(tr_rel):.3e}, target={target:.3g})")
        except Exception as e:
            # In HPO we want to fail-fast (prune), otherwise warn once and continue.
            if bool(getattr(cfg, "HPO_MODE", False)):
                raise
            if not hasattr(build_effective_cov_torch, "_warned_sanity"):
                print(f"[WARN] COV_SANITY_CHECK failed (continuing): {e}", flush=True)
                setattr(build_effective_cov_torch, "_warned_sanity", True)

    return R


# ---------------------- NumPy helpers ----------------------

def hermitize_np(R: np.ndarray) -> np.ndarray:
    return 0.5 * (R + R.conj().T)

def trace_norm_np(R: np.ndarray, target_trace: float = None) -> np.ndarray:
    if target_trace is None:
        # If batch dimension present, handle per-sample
        if R.ndim == 3:
            N = R.shape[-1]
            out = np.empty_like(R)
            for i in range(R.shape[0]):
                out[i] = trace_norm_np(R[i], target_trace=N)
            return out
        target_trace = R.shape[0]
    R = hermitize_np(R)
    tr = float(np.real(np.trace(R)))
    if tr > 1e-9:
        R = (target_trace / tr) * R
    return R

def shrink_np(R: np.ndarray, snr_db: float, base_alpha: float = None) -> np.ndarray:
    if base_alpha is None:
        base_alpha = float(getattr(mdl_cfg, "SHRINK_BASE_ALPHA", 1e-3))
    N = R.shape[0]
    alpha0 = alpha_from_snr_db(float(snr_db), L=int(getattr(cfg, "L", 16)))
    alpha = float(alpha0) * (float(base_alpha) / 1e-3)
    alpha = float(np.clip(alpha, 1e-4, 0.25))
    mu = float(np.real(np.trace(R))) / float(N)
    return (1.0 - alpha) * R + alpha * mu * np.eye(N, dtype=R.dtype)

def build_effective_cov_np(
    R_pred: np.ndarray,
    R_samp: np.ndarray = None,
    beta: float = None,
    diag_load: bool = True,
    apply_shrink: bool = False,
    snr_db: float = None,
    target_trace: float = None,
) -> np.ndarray:
    """
    NumPy version for inference/eval. By default, skip shrink here and let
    MUSIC code apply its own (set apply_shrink=True to apply once here).
    
    Steps match torch version:
      - hermitize + trace-normalize R_pred
      - optional hybrid blend with R_samp (normalize both first)
      - re-normalize after blend
      - optional diagonal loading + re-normalize
      - optional SNR-aware shrinkage (preserves trace by construction)
    """
    if target_trace is None:
        target_trace = float(R_pred.shape[-1])
    R = trace_norm_np(R_pred, target_trace=target_trace)
    if (R_samp is not None) and (beta is not None) and (beta > 0.0):
        R_samp_n = trace_norm_np(R_samp, target_trace=target_trace)
        R = (1.0 - float(beta)) * R + float(beta) * R_samp_n
        R = hermitize_np(R)
        # Re-normalize after blend to ensure exact trace
        R = trace_norm_np(R, target_trace=target_trace)
    if diag_load:
        eps_cov = float(getattr(cfg, "C_EPS", 1e-3))
        R = R + eps_cov * np.eye(R.shape[0], dtype=R.dtype)
        # Re-normalize after diagonal loading
        R = trace_norm_np(R, target_trace=target_trace)
    if apply_shrink and (snr_db is not None):
        R = shrink_np(R, snr_db)
        # Shrinkage preserves trace by construction (μ = tr(R)/N)
    return R



