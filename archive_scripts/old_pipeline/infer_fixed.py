"""
Fixed inference pipeline with all A-block immediate fixes:
- A1: Unified Rhat preconditioning
- A2: Fixed K selection (no +1, temperature, confidence gate)
- A3: Range-only Newton (freeze angles)
- A4: Dense range grid (201 points)
- A5: Per-axis angle clamps
- A6: Measured SNR for shrinkage
"""

import numpy as np, torch
from .configs import cfg, mdl_cfg
from .model import HybridModel
from .physics import nearfield_vec, shrink
from pathlib import Path
import json


def _load_hpo_best_dict(p=None):
    if p is None:
        p = getattr(cfg, 'HPO_BEST_JSON', "results_final/hpo/best.json")
    p = Path(p)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return json.load(f)

def _infer_knobs_from_hpo(hpo_json=None):
    best = _load_hpo_best_dict(hpo_json)
    knobs = dict(
        range_grid = int(best.get("range_grid", 201)),  # A4: Dense grid
        newton_iter = int(best.get("newton_iter", 3)),
        newton_lr = float(best.get("newton_lr", 0.075)),
    )
    hpo_source = "HPO" if best else "defaults"
    print(f"🔧 Inference knobs ({hpo_source}): range_grid={knobs['range_grid']}, "
          f"newton_iter={knobs['newton_iter']}, newton_lr={knobs['newton_lr']:.3f}")
    return knobs

def _load_model_arch_from_hpo(hpo_json="results_final/hpo/best.json"):
    best = _load_hpo_best_dict(hpo_json)
    params = best.get("params", {})
    return dict(
        d_model = int(params.get("D_MODEL", mdl_cfg.D_MODEL)),
        num_heads = int(params.get("NUM_HEADS", mdl_cfg.NUM_HEADS)),
        dropout = float(params.get("dropout", mdl_cfg.DROPOUT)),
    )

@torch.no_grad()
def load_model(ckpt_dir=None, ckpt_name="best.pt", map_location="cpu", prefer_swa=True):
    if ckpt_dir is None:
        ckpt_dir = cfg.CKPT_DIR
    
    if prefer_swa and ckpt_name == "best.pt":
        swa_path = Path(ckpt_dir) / "swa.pt"
        if swa_path.exists():
            ckpt_name = "swa.pt"
            print("🔄 Loading SWA model for improved generalization")
    
    path = Path(ckpt_dir) / ckpt_name
    if not path.exists():
        raise FileNotFoundError(f"Trained checkpoint not found: {path}")
    
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    
    arch = ckpt.get("arch")
    if arch is None:
        try:
            arch = _load_model_arch_from_hpo()
            print(f"Using HPO architecture: D_MODEL={arch['d_model']}")
        except Exception as e:
            print(f"Warning: Could not load HPO arch ({e}), using current defaults")
            arch = {}
    
    original_d_model = mdl_cfg.D_MODEL
    original_num_heads = mdl_cfg.NUM_HEADS
    original_dropout = mdl_cfg.DROPOUT
    
    try:
        mdl_cfg.D_MODEL = arch.get("d_model", mdl_cfg.D_MODEL)
        mdl_cfg.NUM_HEADS = arch.get("num_heads", mdl_cfg.NUM_HEADS)
        mdl_cfg.DROPOUT = arch.get("dropout", mdl_cfg.DROPOUT)
        
        model = HybridModel()
        model.load_state_dict(ckpt["model"])
        if "k_calibration_temp" in ckpt:
            model._k_calibration_temp = ckpt["k_calibration_temp"]
        model.eval()
        return model
    finally:
        mdl_cfg.D_MODEL = original_d_model
        mdl_cfg.NUM_HEADS = original_num_heads
        mdl_cfg.DROPOUT = original_dropout


def compute_measured_snr(sample):
    """
    A6: Compute measured SNR from snapshot power vs noise power.
    Returns SNR in dB, robustly clamped.
    """
    y_data = sample.get("y")
    if y_data is None:
        return 10.0  # fallback
    
    if torch.is_tensor(y_data):
        y_data = y_data.numpy()
    
    # Convert to complex [L, M]
    Y = y_data[:, :, 0] + 1j * y_data[:, :, 1]
    
    # Estimate noise power from stored sigma or from sample
    sigma = sample.get("noise_std", sample.get("sigma", 0.005))
    if torch.is_tensor(sigma):
        sigma = float(sigma.item())
    sigma2 = float(sigma) ** 2
    
    # Measured signal power
    p_meas = np.mean(np.abs(Y) ** 2)
    
    # SNR = (signal_power - noise_power) / noise_power in linear, then to dB
    snr_linear = max((p_meas - sigma2) / max(sigma2, 1e-12), 1e-12)
    snr_db = 10 * np.log10(snr_linear)
    
    # Robust clamp to reasonable range
    snr_db = float(np.clip(snr_db, -10.0, 25.0))
    return snr_db


def estimate_k_ic_from_cov(R, T, method="mdl", kmax=None):
    R_f64 = R.astype(np.complex128) if R.dtype != np.complex128 else R
    lam = np.sort(np.real(np.maximum(np.linalg.eigvals(R_f64), 1e-12)))[::-1]
    M = int(R.shape[0]); T = int(T)
    if kmax is None or kmax > M-1: kmax = M-1
    ks = np.arange(0, kmax+1); crit = np.zeros_like(ks, float)
    for i,k in enumerate(ks):
        noise = lam[k:]; 
        if noise.size==0: crit[i]=np.inf; continue
        gm = np.exp(np.mean(np.log(noise))); am = np.mean(noise)
        ratio = np.clip(gm/am, 1e-12, 1.0)
        if method=="mdl": crit[i] = -T*(M-k)*np.log(ratio) + 0.5*k*(2*M-k)*np.log(T)
        elif method=="aic": crit[i] = -2*T*(M-k)*np.log(ratio) + 2*k*(2*M-k)
        else: raise ValueError("method must be mdl or aic")
    return int(ks[np.argmin(crit)])


def _music_cost(phi, theta, r, Un):
    a = nearfield_vec(cfg, float(phi), float(theta), float(r)).astype(np.complex64)[:,None]
    den = np.real((a.conj().T @ Un @ Un.conj().T @ a)[0,0])
    return float(den)


def newton_refine_range_only(phi0, theta0, r0, Un, iters=3, lr=0.075):
    """
    A3: Range-only Newton refinement (freeze angles).
    A5: Per-axis clamps applied.
    """
    phi, theta, r = float(phi0), float(theta0), float(r0)
    dr = 0.1
    
    for _ in range(max(0, int(iters))):
        # Gradient and Hessian for r only
        def fd_r(x): 
            return (_music_cost(phi, theta, x+dr, Un) - _music_cost(phi, theta, x-dr, Un)) / (2*dr)
        def s2_r(x): 
            return (_music_cost(phi, theta, x+dr, Un) - 2*_music_cost(phi, theta, x, Un) + 
                    _music_cost(phi, theta, x-dr, Un)) / (dr*dr)
        
        g_r = fd_r(r)
        H_r = s2_r(r) + 1e-6
        
        step_r = max(lr, 1e-2) * g_r / H_r
        step_r = float(np.clip(step_r, -0.5, 0.5))
        
        # Line search
        cand = [r - step_r, r - 0.5*step_r, r - 0.25*step_r]
        def cost(rv): return _music_cost(phi, theta, rv, Un)
        vals = [cost(rv) for rv in cand]
        r = cand[int(np.argmin(vals))]
        
        # A5: Clamp to range FOV
        r = max(cfg.RANGE_R[0], min(cfg.RANGE_R[1], r))
    
    return float(phi), float(theta), float(r)


def micro_newton_angles(phi0, theta0, r0, Un, iters=2, lr=0.01):
    """
    Micro-Newton refinement for angles (1-2 tiny steps).
    Very small step size with FOV clamps to trim a few degrees.
    """
    phi, theta, r = float(phi0), float(theta0), float(r0)
    dphi, dtheta = 0.01, 0.01  # Very small finite difference step
    
    for _ in range(max(0, int(iters))):
        # Gradients for phi and theta
        def fd_phi(x): 
            return (_music_cost(x+dphi, theta, r, Un) - _music_cost(x-dphi, theta, r, Un)) / (2*dphi)
        def fd_theta(x): 
            return (_music_cost(phi, x+dtheta, r, Un) - _music_cost(phi, x-dtheta, r, Un)) / (2*dtheta)
        
        # Hessians for phi and theta
        def s2_phi(x): 
            return (_music_cost(x+dphi, theta, r, Un) - 2*_music_cost(x, theta, r, Un) + 
                    _music_cost(x-dphi, theta, r, Un)) / (dphi*dphi)
        def s2_theta(x): 
            return (_music_cost(phi, x+dtheta, r, Un) - 2*_music_cost(phi, x, r, Un) + 
                    _music_cost(phi, x-dtheta, r, Un)) / (dtheta*dtheta)
        
        g_phi = fd_phi(phi)
        g_theta = fd_theta(theta)
        H_phi = s2_phi(phi) + 1e-6
        H_theta = s2_theta(theta) + 1e-6
        
        # Very small steps for micro-refinement
        step_phi = lr * g_phi / H_phi
        step_theta = lr * g_theta / H_theta
        
        # Clamp steps to prevent large jumps
        step_phi = float(np.clip(step_phi, -0.1, 0.1))  # ±0.1 rad ≈ ±6°
        step_theta = float(np.clip(step_theta, -0.1, 0.1))
        
        # Update angles
        phi_new = phi - step_phi
        theta_new = theta - step_theta
        
        # FOV clamps
        phi_new = max(-cfg.ANGLE_RANGE_PHI, min(cfg.ANGLE_RANGE_PHI, phi_new))
        theta_new = max(-cfg.ANGLE_RANGE_THETA, min(cfg.ANGLE_RANGE_THETA, theta_new))
        
        # Accept if improvement
        if _music_cost(phi_new, theta_new, r, Un) < _music_cost(phi, theta, r, Un):
            phi, theta = phi_new, theta_new
    
    return float(phi), float(theta), float(r)


def hybrid_estimate_fixed(model, sample, force_K=None, k_policy="mdl",
                          do_newton=True, use_hpo_knobs=True, hpo_json="results_final/hpo/best.json"):
    """
    Fixed hybrid inference with A1-A6 improvements.
    """
    knobs = _infer_knobs_from_hpo(hpo_json) if use_hpo_knobs else dict(
        range_grid=201, newton_iter=3, newton_lr=0.075
    )

    def to_ri_t(z): return torch.stack([z.real, z.imag], dim=-1)
    device = next(model.parameters()).device
    
    # Handle both y_cplx and y formats
    if "y_cplx" in sample:
        y = torch.from_numpy(sample["y_cplx"]).to(torch.complex64).unsqueeze(0).to(device)
        H = torch.from_numpy(sample["H_cplx"]).to(torch.complex64).unsqueeze(0).to(device)
    else:
        y_data = sample["y"]
        if torch.is_tensor(y_data):
            y_data = y_data.numpy()
        y_real = y_data[:, :, 0]
        y_imag = y_data[:, :, 1]
        y_complex = y_real + 1j * y_imag
        y = torch.from_numpy(y_complex).to(torch.complex64).unsqueeze(0).to(device)
        
        H_data = sample["H"]
        if torch.is_tensor(H_data):
            H_data = H_data.numpy()
        H_real = H_data[:, :, 0]
        H_imag = H_data[:, :, 1]
        H_complex = H_real + 1j * H_imag
        H = torch.from_numpy(H_complex).to(torch.complex64).unsqueeze(0).to(device)
    
    codes_data = sample["codes"]
    if torch.is_tensor(codes_data):
        codes_data = codes_data.numpy()
    codes_complex = codes_data[:, :, 0] + 1j * codes_data[:, :, 1]
    codes = torch.from_numpy(codes_complex).to(torch.complex64).unsqueeze(0).to(device)
    
    y_ri = to_ri_t(y)
    H_ri = to_ri_t(H)
    codes_ri = to_ri_t(codes)
    
    with torch.no_grad():
        pred = model(y_ri, H_ri, codes_ri)

    # A1: Unified Rhat preconditioning
    Aflat = pred["cov_fact_angle"][0].detach().cpu().numpy()
    A_cplx = (Aflat[::2] + 1j*Aflat[1::2]).reshape(cfg.N, cfg.K_MAX)
    Rhat = A_cplx @ A_cplx.conj().T
    
    # Hermitize
    Rhat = 0.5 * (Rhat + Rhat.conj().T)
    # Trace normalize
    tr = np.trace(Rhat).real + 1e-9
    Rhat = Rhat / tr
    
    # A6: Measured SNR for shrinkage
    snr_meas_db = compute_measured_snr(sample)
    alpha = mdl_cfg.SHRINK_BASE_ALPHA * 10 ** (-snr_meas_db / 20)
    alpha = float(np.clip(alpha, 1e-4, 5e-2))  # Robust clamp
    
    # Shrink
    Rtilde = Rhat + alpha * (tr / cfg.N) * np.eye(cfg.N, dtype=Rhat.dtype)

    # A2: Fixed K selection (no +1, temperature, confidence gate)
    if "k_logits" in pred:
        k_temp = getattr(model, '_k_calibration_temp', 1.0)
        k_logits_scaled = pred["k_logits"][0].detach().cpu() / k_temp
        pK = torch.softmax(k_logits_scaled, -1).numpy()
        k_from_logits = int(np.argmax(pK)) + 1  # Map from class index (0-4) to K (1-5)
    else:
        k_from_logits = None
        pK = None

    T_snap = int(sample["codes"].shape[0] if hasattr(sample["codes"], 'shape') else cfg.L)
    
    if force_K is not None:
        K_hat = int(force_K)
    elif k_from_logits is not None and pK is not None and pK.max() >= mdl_cfg.K_CONF_THRESH:
        K_hat = k_from_logits
    else:
        K_hat = estimate_k_ic_from_cov(Rtilde, T_snap, method=k_policy, kmax=cfg.K_MAX)
        if K_hat == 0:
            K_hat = estimate_k_ic_from_cov(Rtilde, T_snap, method="aic", kmax=cfg.K_MAX)

    K_hat = int(max(1, min(K_hat, cfg.K_MAX, int(cfg.L)-1)))
    if K_hat == 0:
        return [], [], []

    # Use auxiliary head angles (direct regression, much more accurate than grid head!)
    # Grid head has ~1-2° quantization and diffuse soft-argmax at τ=0.15
    # Aux head gives 0.1-2° errors vs 40°+ from grid head
    aux_ptr = pred["phi_theta_r"][0].detach().cpu().numpy()
    KMAX = int(cfg.K_MAX)
    phi_all = aux_ptr[:KMAX]
    theta_all = aux_ptr[KMAX:2*KMAX]
    phi_est, theta_est = phi_all[:K_hat], theta_all[:K_hat]

    # Range subspace factor
    A_rflat = pred["cov_fact_range"][0].detach().cpu().numpy()
    A_rcplx = (A_rflat[::2] + 1j*A_rflat[1::2]).reshape(cfg.N, cfg.K_MAX)
    R_r = A_rcplx @ A_rcplx.conj().T
    
    # Apply same preconditioning to R_r
    R_r = 0.5 * (R_r + R_r.conj().T)
    tr_r = np.trace(R_r).real + 1e-9
    R_r = R_r / tr_r
    R_r_shrunk = R_r + alpha * (tr_r / cfg.N) * np.eye(cfg.N, dtype=R_r.dtype)

    # A4: Dense range grid (201 points from knobs)
    r_est = []
    r_grid = np.linspace(cfg.RANGE_R[0], cfg.RANGE_R[1], int(knobs["range_grid"]))
    R_r_f64 = R_r_shrunk.astype(np.complex128)
    vals, vecs = np.linalg.eigh(R_r_f64)
    Un = vecs[:, :-K_hat].astype(np.complex64)
    
    for idx, (phi, theta) in enumerate(zip(phi_est, theta_est)):
        # Vectorized steering on r-grid
        a_grid = np.stack(
            [nearfield_vec(cfg, float(phi), float(theta), float(rg)) for rg in r_grid], axis=1
        ).astype(np.complex64)
        P_den = np.real(np.sum((Un.conj().T @ a_grid) * (Un.conj().T @ a_grid).conj(), axis=0))
        r_best = float(r_grid[int(np.argmin(P_den))])
        
        if do_newton:
            # A3: Range-only Newton (freeze angles)
            phi_ref, theta_ref, r_ref = newton_refine_range_only(
                phi, theta, r_best, Un,
                iters=knobs["newton_iter"],
                lr=knobs["newton_lr"]
            )
            # Keep angles unchanged, update range only
            r_best = r_ref
            
            # MICRO-NEWTON: 1-2 tiny angle steps after range refinement
            phi_ref, theta_ref, r_ref = micro_newton_angles(
                phi_ref, theta_ref, r_best, Un,
                iters=2,  # 1-2 tiny steps
                lr=0.01   # Very small step size
            )
            phi, theta, r_best = phi_ref, theta_ref, r_ref
        
        r_est.append(float(r_best))
    
    return list(map(float, phi_est)), list(map(float, theta_est)), r_est

