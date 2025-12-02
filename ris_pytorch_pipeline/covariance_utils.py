import numpy as np
import torch
from .configs import cfg, mdl_cfg


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
    alpha = base * 10^(-snr_db/20), clamped to [1e-4, 5e-2]
    """
    if base_alpha is None:
        base_alpha = float(getattr(mdl_cfg, "SHRINK_BASE_ALPHA", 1e-3))
    B, N, _ = R.shape
    device = R.device
    dtype = R.dtype
    snr_db = snr_db.to(device=device, dtype=torch.float32).view(-1)
    alpha = base_alpha * torch.pow(10.0, -snr_db / 20.0)
    alpha = alpha.clamp(min=1e-4, max=5e-2).view(B, 1, 1)
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
    alpha = base_alpha * (10.0 ** (-float(snr_db) / 20.0))
    alpha = float(np.clip(alpha, 1e-4, 5e-2))
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



