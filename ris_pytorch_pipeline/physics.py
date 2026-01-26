
import numpy as np, math
from .configs import cfg, mdl_cfg

def nearfield_vec(cfg, phi, theta, r, h_flat=None, v_flat=None):
    if h_flat is None or v_flat is None:
        h_idx = np.arange(-(cfg.N_H - 1)//2, (cfg.N_H + 1)//2) * cfg.d_H
        v_idx = np.arange(-(cfg.N_V - 1)//2, (cfg.N_V + 1)//2) * cfg.d_V
        h_mesh, v_mesh = np.meshgrid(h_idx, v_idx, indexing="xy")
        h_flat = h_mesh.reshape(-1).astype(np.float32)
        v_flat = v_mesh.reshape(-1).astype(np.float32)
    a = np.empty(cfg.N, np.complex64)
    r_eff = max(float(r), 1e-9)
    sin_phi, cos_theta, sin_theta = math.sin(phi), math.cos(theta), math.sin(theta)
    for i,(vh, vv) in enumerate(zip(h_flat, v_flat)):
        dist = r - vh*sin_phi*cos_theta - vv*sin_theta + (vh**2 + vv**2)/(2*r_eff)
        a[i] = np.exp(1j*cfg.k0*(r - dist))
    return a/np.sqrt(cfg.N)

def _rician_bs2ris(M, N, k0, d_H, kfac, ang_range):
    phi_bs, theta_bs = np.random.uniform(-ang_range, ang_range, 2)
    a_bs = np.exp(-1j*k0*(np.arange(M)-(M-1)/2)*d_H*np.sin(phi_bs)*np.cos(theta_bs))/np.sqrt(M)
    # simple nearfield steering for RIS (square grid assumption in basis)
    side = int(np.sqrt(N)); side2 = int(np.ceil(N/side))
    h_idx = np.arange(-(side - 1)//2, (side + 1)//2) * d_H
    v_idx = np.arange(-(side2 - 1)//2, (side2 + 1)//2) * d_H
    h_mesh, v_mesh = np.meshgrid(h_idx, v_idx, indexing="xy")
    h_flat = h_mesh.reshape(-1).astype(np.float32)[:N]
    v_flat = v_mesh.reshape(-1).astype(np.float32)[:N]
    ris_obj = type("obj",(object,),dict(N=N,k0=k0,d_H=d_H,N_H=side,N_V=side2,d_V=d_H))()
    a_ris = nearfield_vec(ris_obj, phi_bs, theta_bs, 100.0, h_flat, v_flat)
    H_los = np.outer(a_bs, a_ris.conj())
    H_nlos = (np.random.randn(M,N)+1j*np.random.randn(M,N))/np.sqrt(2*M)
    alpha, beta = np.sqrt(kfac/(kfac+1)), np.sqrt(1/(kfac+1))
    return (alpha*H_los + beta*H_nlos).astype(np.complex64)

def quantise_phase(vec, bits=3):
    if bits is None: return vec
    levels = 2 ** bits
    step = 2 * np.pi / levels
    qphase = np.round(np.angle(vec) / step) * step
    return np.exp(1j * qphase).astype(vec.dtype)

def alpha_from_snr_db(snr_db: float, *, L: int | None = None) -> float:
    """
    SNR-aware shrinkage coefficient (piecewise mapping).

    Notes:
    - The base mapping below was tuned around an L≈16 snapshot regime.
    - We scale it by sqrt(16/L) to reduce shrinkage as snapshot diversity increases.
      (More snapshots => sample covariance is better conditioned => less shrink needed.)
    """
    d = float(snr_db)
    if d >= 15:     a = 0.02
    elif d >= 8:    a = 0.04
    elif d >= 3:    a = 0.07
    elif d >= -2:   a = 0.10
    else:           a = 0.14
    # L-aware scaling (reference L=16). Clamp to sane bounds.
    if L is None:
        L = int(getattr(cfg, "L", 16))
    L = max(1, int(L))
    scale = math.sqrt(16.0 / float(L))
    a = a * scale
    return float(np.clip(a, 1e-4, 0.25))

def shrink(R, snr_db: float = None, base: float = 1e-3, alpha: float = None):
    """
    C11: Convex shrinkage formula.
    Returns (1-α)*R + α*μ*I where μ = tr(R)/N

    IMPORTANT (consistency + HPO):
    - If `alpha` is provided: use it directly.
    - Else: compute a default SNR-aware alpha via `alpha_from_snr_db(snr_db)`, and then
      scale it by `(base / 1e-3)`.

    This makes `base` (typically `mdl_cfg.SHRINK_BASE_ALPHA`) a true knob that affects
    shrinkage everywhere, including call-sites that historically passed `base=...` but
    saw no effect because `base` was ignored.
    """
    if base is None:
        base = float(getattr(mdl_cfg, "SHRINK_BASE_ALPHA", 1e-3))
    if alpha is None:
        if snr_db is None: raise ValueError("need alpha or snr_db")
        # SNR-aware mapping (L-aware) scaled by base relative to the reference 1e-3.
        alpha0 = alpha_from_snr_db(snr_db, L=int(getattr(cfg, "L", 16)))
        alpha = float(alpha0) * (float(base) / 1e-3)

    alpha = float(np.clip(alpha, 1e-4, 0.25))
    
    mu = np.trace(R).real / R.shape[0]  # Mean eigenvalue
    return (1.0 - alpha) * R + alpha * mu * np.eye(R.shape[0], dtype=R.dtype)
