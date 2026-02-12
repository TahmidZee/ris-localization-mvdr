"""
V2 inference wrapper.

Reuses v1 covariance post-processing + MVDR detector.
"""

from __future__ import annotations
import math
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..covariance_utils import build_effective_cov_np
from ..music_gpu import mvdr_detect_sources
from .config_v2 import v2_cfg


def _to_numpy_ri_or_complex(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def _ri_to_complex_np(x):
    x = _to_numpy_ri_or_complex(x)
    if x is None:
        return None
    if np.iscomplexobj(x):
        return x
    if x.shape[-1] == 2:
        return x[..., 0] + 1j * x[..., 1]
    return x.astype(np.complex64)


def _to_torch_ri_batch(x, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x.to(device=device)
    else:
        t = torch.from_numpy(np.asarray(x)).to(device=device)
    if t.dim() == 3:  # [L,M,2] or [L,N,2]
        t = t.unsqueeze(0)
    if t.dim() == 4 and t.shape[-1] == 2:
        return t.float()
    if t.dim() == 5 and t.shape[-1] == 2:
        return t.float()
    raise ValueError(f"Unsupported RI tensor shape for inference: {tuple(t.shape)}")


def infer_v2_sample(
    model,
    sample: Dict[str, Any],
    force_k: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run v2 model on one sample and detect sources with v1 MVDR.
    """
    model_device = next(model.parameters()).device
    dev = torch.device(device) if device else model_device

    y = sample.get("y")
    H = sample.get("H")
    codes = sample.get("codes")
    if y is None or H is None or codes is None:
        raise ValueError("sample must contain y, H, and codes fields")

    y_t = _to_torch_ri_batch(y, dev)
    H_t = _to_torch_ri_batch(H, dev)
    C_t = _to_torch_ri_batch(codes, dev)

    # Optional tap-domain dict.
    h_taps = sample.get("H_taps", None)
    h_taps_t = None
    if isinstance(h_taps, dict) and len(h_taps) > 0:
        h_taps_t = {}
        for key, value in h_taps.items():
            if value is None:
                continue
            if torch.is_tensor(value):
                t = value.to(dev)
            else:
                t = torch.from_numpy(np.asarray(value)).to(dev)
            if t.dim() >= 1 and t.shape[0] != y_t.shape[0]:
                t = t.unsqueeze(0)
            h_taps_t[key] = t.float()

    snr_db = float(sample.get("snr_db", sample.get("snr", 10.0)))
    snr_t = torch.tensor([snr_db], device=dev, dtype=torch.float32)

    with torch.no_grad():
        out = model(y_t, H_t, C_t, snr_db=snr_t, H_taps=h_taps_t)
    R_pred = _ri_to_complex_np(out["R_pred"][0])

    R_samp = _ri_to_complex_np(sample.get("R_samp", None))
    R_eff = build_effective_cov_np(
        R_pred,
        R_samp=R_samp,
        beta=float(getattr(v2_cfg, "HYBRID_COV_BETA", 0.0)) if R_samp is not None else 0.0,
        diag_load=True,
        apply_shrink=True,
        snr_db=snr_db,
        target_trace=float(v2_cfg.N),
    )

    max_sources = int(force_k) if force_k is not None else int(getattr(v2_cfg, "K_MAX", 5))
    mvdr_sources, mvdr_spectrum = mvdr_detect_sources(
        R_eff,
        v2_cfg,
        device=("cuda" if (dev.type == "cuda" and torch.cuda.is_available()) else "cpu"),
        max_sources=max_sources,
    )

    # Convert to rad outputs for consistency with existing evaluation code.
    phi_rad = [math.radians(float(s[0])) for s in mvdr_sources]
    theta_rad = [math.radians(float(s[1])) for s in mvdr_sources]
    range_m = [float(s[2]) for s in mvdr_sources]
    conf = [float(s[3]) for s in mvdr_sources]

    return {
        "R_pred": R_pred,
        "R_eff": R_eff,
        "sources_deg": mvdr_sources,
        "phi_rad": phi_rad,
        "theta_rad": theta_rad,
        "range_m": range_m,
        "confidence": conf,
        "spectrum": mvdr_spectrum,
    }
