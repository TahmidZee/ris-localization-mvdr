
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
    # Fallbacks to mdl_cfg defaults if key absent
    knobs = dict(
        range_grid = int(best.get("range_grid", getattr(mdl_cfg, "INFERENCE_GRID_SIZE_RANGE", 61))),
        newton_iter = int(best.get("newton_iter", getattr(mdl_cfg, "NEWTON_ITER", 5))),
        newton_lr = float(best.get("newton_lr", getattr(mdl_cfg, "NEWTON_LR", 0.1))),
    )
    
    # Log active knobs for reproducibility
    hpo_source = "HPO" if best else "defaults"
    print(f"🔧 Inference knobs ({hpo_source}): range_grid={knobs['range_grid']}, "
          f"newton_iter={knobs['newton_iter']}, newton_lr={knobs['newton_lr']:.3f}")
    
    return knobs

def _load_model_arch_from_hpo(hpo_json="results_final/hpo/best.json"):
    """Load model architecture parameters from HPO results"""
    best = _load_hpo_best_dict(hpo_json)
    params = best.get("params", {})
    
    # Extract architecture parameters, fallback to current defaults
    return dict(
        d_model = int(params.get("D_MODEL", mdl_cfg.D_MODEL)),
        num_heads = int(params.get("NUM_HEADS", mdl_cfg.NUM_HEADS)),
        dropout = float(params.get("dropout", mdl_cfg.DROPOUT)),
        # Add other architecture parameters as needed
    )

@torch.no_grad()
def load_model(ckpt_dir=None, ckpt_name="best.pt", map_location="cpu", prefer_swa=True):
    if ckpt_dir is None:
        ckpt_dir = cfg.CKPT_DIR  # Use cfg.CKPT_DIR as default
    
    # Check for SWA model first if preferred
    if prefer_swa and ckpt_name == "best.pt":
        swa_path = Path(ckpt_dir) / "swa.pt"
        if swa_path.exists():
            ckpt_name = "swa.pt"
            print("🔄 Loading SWA model for improved generalization")
    
    path = Path(ckpt_dir) / ckpt_name
    if not path.exists():
        raise FileNotFoundError(f"Trained checkpoint not found: {path}")
    
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    
    # Try to get architecture from checkpoint first
    arch = ckpt.get("arch")
    if arch is None:
        # Fallback to HPO configuration
        try:
            arch = _load_model_arch_from_hpo()
            print(f"Using HPO architecture: D_MODEL={arch['d_model']}")
        except Exception as e:
            print(f"Warning: Could not load HPO arch ({e}), using current defaults")
            arch = {}
    
    # Build model with correct architecture
    # Temporarily override mdl_cfg with HPO values
    original_d_model = mdl_cfg.D_MODEL
    original_num_heads = mdl_cfg.NUM_HEADS
    original_dropout = mdl_cfg.DROPOUT
    
    try:
        mdl_cfg.D_MODEL = arch.get("d_model", mdl_cfg.D_MODEL)
        mdl_cfg.NUM_HEADS = arch.get("num_heads", mdl_cfg.NUM_HEADS)
        mdl_cfg.DROPOUT = arch.get("dropout", mdl_cfg.DROPOUT)
        
        model = HybridModel()
        missing_keys, unexpected_keys = model.load_state_dict(ckpt["model"], strict=False)
        
        # Report missing/unexpected keys for debugging
        if missing_keys:
            print(f"⚠️  Missing keys in checkpoint (using random init): {len(missing_keys)} keys")
            if len(missing_keys) <= 5:
                for key in missing_keys:
                    print(f"    - {key}")
            else:
                print(f"    - {missing_keys[0]} ... and {len(missing_keys)-1} more")
        
        if unexpected_keys:
            print(f"⚠️  Unexpected keys in checkpoint (ignored): {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 5:
                for key in unexpected_keys:
                    print(f"    - {key}")
            else:
                print(f"    - {unexpected_keys[0]} ... and {len(unexpected_keys)-1} more")
        
        # Store calibrated temperature if available
        if "k_calibration_temp" in ckpt:
            model._k_calibration_temp = ckpt["k_calibration_temp"]
        model.eval()
        return model
        
    finally:
        # Restore original values
        mdl_cfg.D_MODEL = original_d_model
        mdl_cfg.NUM_HEADS = original_num_heads
        mdl_cfg.DROPOUT = original_dropout


def estimate_k_ic_from_cov(R, T, method="mdl", kmax=None):
    # Cast to float64 for stable eigenvalue computation
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

def newton_refine(phi0, theta0, r0, Un, iters=None, lr=None):
    if iters is None: iters = mdl_cfg.NEWTON_ITER
    if lr    is None: lr    = mdl_cfg.NEWTON_LR
    phi, theta, r = float(phi0), float(theta0), float(r0)
    dphi, dtheta, dr = 1e-3, 1e-3, 0.1
    for _ in range(max(0,int(iters))):
        def fd_phi(x): return (_music_cost(x+dphi, theta, r, Un)-_music_cost(x-dphi, theta, r, Un))/(2*dphi)
        def fd_theta(x): return (_music_cost(phi, x+dtheta, r, Un)-_music_cost(phi, x-dtheta, r, Un))/(2*dtheta)
        def fd_r(x): return (_music_cost(phi, theta, x+dr, Un)-_music_cost(phi, theta, x-dr, Un))/(2*dr)
        g_phi, g_theta, g_r = fd_phi(phi), fd_theta(theta), fd_r(r)
        def s2_phi(x): return (_music_cost(x+dphi, theta, r, Un)-2*_music_cost(x, theta, r, Un)+_music_cost(x-dphi, theta, r, Un))/(dphi*dphi)
        def s2_theta(x): return (_music_cost(phi, x+dtheta, r, Un)-2*_music_cost(phi, x, r, Un)+_music_cost(phi, x-dtheta, r, Un))/(dtheta*dtheta)
        def s2_r(x): return (_music_cost(phi, theta, x+dr, Un)-2*_music_cost(phi, theta, x, Un)+_music_cost(phi, theta, x-dr, Un))/(dr*dr)
        H_phi, H_theta, H_r = s2_phi(phi)+1e-6, s2_theta(theta)+1e-6, s2_r(r)+1e-6
        phi   -= lr * g_phi   / H_phi
        theta -= lr * g_theta / H_theta
        step_r = max(lr, 1e-2) * g_r / H_r
        step_r = float(np.clip(step_r, -0.5, 0.5))
        cand = [r - step_r, r - 0.5*step_r, r - 0.25*step_r]
        def cost(rv): return _music_cost(phi, theta, rv, Un)
        vals = [cost(rv) for rv in cand]
        r = cand[int(np.argmin(vals))]
        # Clamp to respective FOV ranges
        phi = max(-cfg.ANGLE_RANGE_PHI, min(cfg.ANGLE_RANGE_PHI, phi))
        theta = max(-cfg.ANGLE_RANGE_THETA, min(cfg.ANGLE_RANGE_THETA, theta))
        r = max(cfg.RANGE_R[0], min(cfg.RANGE_R[1], r))
    return float(phi), float(theta), float(r)

def hybrid_estimate_final(model, sample, force_K=None, k_policy="mdl",
                         do_newton=True, use_hpo_knobs=True, hpo_json="results_final/hpo/best.json",
                         prefer_logits=True):
    # --- HPO knobs (safe for inference only) ---
    knobs = _infer_knobs_from_hpo(hpo_json) if use_hpo_knobs else dict(
        range_grid=getattr(mdl_cfg, "INFERENCE_GRID_SIZE_RANGE", 61),
        newton_iter=getattr(mdl_cfg, "NEWTON_ITER", 5),
        newton_lr=getattr(mdl_cfg, "NEWTON_LR", 0.1),
    )

    def to_ri_t(z): return torch.stack([z.real, z.imag], dim=-1)
    device = next(model.parameters()).device
    # Handle both y_cplx and y formats
    if "y_cplx" in sample:
        y = torch.from_numpy(sample["y_cplx"]).to(torch.complex64).unsqueeze(0).to(device)
        H = torch.from_numpy(sample["H_cplx"]).to(torch.complex64).unsqueeze(0).to(device)
    else:
        # Convert y from [L, M, 2] to complex (handle both numpy and torch)
        y_data = sample["y"]
        if torch.is_tensor(y_data):
            y_data = y_data.numpy()
        y_real = y_data[:, :, 0]
        y_imag = y_data[:, :, 1]
        y_complex = y_real + 1j * y_imag
        y = torch.from_numpy(y_complex).to(torch.complex64).unsqueeze(0).to(device)
        
        # Convert H from [L, N, 2] to complex (handle both numpy and torch)
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
    # Convert codes from [L, N, 2] to complex [L, N]
    codes_complex = codes_data[:, :, 0] + 1j * codes_data[:, :, 1]
    codes = torch.from_numpy(codes_complex).to(torch.complex64).unsqueeze(0).to(device)
    y_ri = to_ri_t(y)
    H_ri = to_ri_t(H)
    codes_ri = to_ri_t(codes)
    # Optional: build R_samp from snapshots for hybrid-aware K-head
    try:
        from .angle_pipeline import build_sample_covariance_from_snapshots
        L_snap = y_complex.shape[0]
        H_stack = np.repeat(H_complex[None, :, :], L_snap, axis=0)  # [L, M, N]
        R_samp_np = build_sample_covariance_from_snapshots(y_complex, H_stack, codes_complex, cfg)
        R_samp_t = torch.from_numpy(R_samp_np).to(torch.complex64).unsqueeze(0).to(device)  # [1,N,N]
    except Exception:
        R_samp_t = None
    with torch.no_grad():
        pred = model(y_ri, H_ri, codes_ri, R_samp=R_samp_t)

    # angle subspace factor -> Rhat
    Aflat = pred["cov_fact_angle"][0].detach().cpu().numpy()
    A_cplx = (Aflat[::2] + 1j*Aflat[1::2]).reshape(cfg.N, cfg.K_MAX)
    Rhat = A_cplx @ A_cplx.conj().T
    
    # --- make Rhat well-behaved for IC ---
    Rhat = 0.5*(Rhat + Rhat.conj().T)
    Rhat /= (np.trace(Rhat).real + 1e-9)
    Rhat = shrink(Rhat, sample.get("snr_db", 10.0), base=mdl_cfg.SHRINK_BASE_ALPHA)

    # --- prefer model's own K logits if present (with calibration) ---
    if "k_logits" in pred:
        k_temp = getattr(model, '_k_calibration_temp', 1.0)
        k_logits_scaled = pred["k_logits"][0].detach().cpu() / k_temp
        pK = torch.softmax(k_logits_scaled, -1).numpy()
        k_from_logits = int(np.argmax(pK)) + 1  # Map from class index (0-4) to K (1-5)
        nn_conf = float(np.max(pK))
    else:
        k_from_logits, nn_conf = None, 0.0

    T_snap = int(sample["codes"].shape[0])
    # Compute MDL/AIC baseline K on Rhat (same object used for IC)
    K_mdl = estimate_k_ic_from_cov(Rhat, T_snap, method="mdl", kmax=cfg.K_MAX)
    if K_mdl == 0:
        K_mdl = estimate_k_ic_from_cov(Rhat, T_snap, method="aic", kmax=cfg.K_MAX)
    K_mdl = int(max(1, min(K_mdl, cfg.K_MAX, int(cfg.L)-1)))

    if force_K is not None:
        K_hat = int(force_K)
    elif prefer_logits and (k_from_logits is not None):
        # Confidence-gated NN vs MDL
        thr = float(getattr(cfg, "K_CONF_THRESH", 0.65))
        K_nn = min(max(1, k_from_logits), cfg.K_MAX)
        K_hat = K_mdl if (nn_conf < thr) else K_nn
    else:
        # Fallback purely on information criteria
        K_hat = K_mdl

    # near-field subspace rank limit
    K_hat = int(max(1, min(K_hat, cfg.K_MAX, int(cfg.L)-1)))
    if K_hat == 0:
        return [], [], []

    # SOTA FIX: Use unified angle pipeline (MUSIC → Parabolic → Newton)
    # This is the SAME stack as training evaluation for consistency
    # Replaces old aux head which was causing train-test mismatch
    from .angle_pipeline import angle_pipeline
    
    # Get covariance factor for MUSIC
    cf_ang = pred["cov_fact_angle"][0].detach().cpu().numpy()  # [N*2, K_MAX] real
    cf_cplx = (cf_ang[::2] + 1j*cf_ang[1::2]).reshape(cfg.N, cfg.K_MAX)  # [N, K_MAX] complex
    
    # Run unified angle pipeline: MUSIC → Parabolic → Newton
    phi_est, theta_est, info = angle_pipeline(
        cf_cplx, K_hat, cfg,
        use_fba=getattr(cfg, "MUSIC_USE_FBA", True),
        use_adaptive_shrink=True,
        use_parabolic=getattr(cfg, "MUSIC_PEAK_REFINE", True),
        use_newton=getattr(cfg, "USE_NEWTON_REFINE", True),
        device="cpu",  # Inference is typically single-sample, CPU is fine
        # Align inference with train-eval: pass snapshots for hybrid blending
        y_snapshots=y[0].detach().cpu().numpy(),
        H_snapshots=H[0].detach().cpu().numpy(),
        codes_snapshots=codes[0].detach().cpu().numpy(),
        blend_beta=float(getattr(cfg, "HYBRID_COV_BETA", 0.0)),
    )

    # range subspace factor -> R_r
    A_rflat = pred["cov_fact_range"][0].detach().cpu().numpy()
    A_rcplx = (A_rflat[::2] + 1j*A_rflat[1::2]).reshape(cfg.N, cfg.K_MAX)
    R_r = A_rcplx @ A_rcplx.conj().T
    R_r = shrink(R_r, sample["snr_db"], base=mdl_cfg.SHRINK_BASE_ALPHA)

    # MUSIC over range grid (use HPO range_grid) - use float64 for better EVD accuracy
    r_est = []
    r_grid = np.linspace(cfg.RANGE_R[0], cfg.RANGE_R[1], int(knobs["range_grid"]))
    R_r_f64 = R_r.astype(np.complex128)  # Cast to float64 for EVD
    vals, vecs = np.linalg.eigh(R_r_f64)
    Un = vecs[:, :-K_hat].astype(np.complex64)  # Convert back to float32
    for idx,(phi, theta) in enumerate(zip(phi_est, theta_est)):
        # Vectorized steering on the r-grid for this (phi, theta)
        a_grid = np.stack(
            [nearfield_vec(cfg, float(phi), float(theta), float(rg)) for rg in r_grid], axis=1
        ).astype(np.complex64)  # [N, G]
        # Projected energy (denominator)
        P_den = np.real(np.sum((Un.conj().T @ a_grid) * (Un.conj().T @ a_grid).conj(), axis=0))  # [G]
        r_best = float(r_grid[int(np.argmin(P_den))])  # maximize 1/den  == minimize den
        
        if do_newton:
            # pass HPO-tuned iters/LR to your Newton (limit iterations for speed)
            phi_ref, theta_ref, r_ref = newton_refine(phi, theta, r_best, Un,
                                                      iters=min(knobs["newton_iter"], 3),
                                                      lr=knobs["newton_lr"])
            phi_est[idx], theta_est[idx] = phi_ref, theta_ref
            r_best = r_ref
        r_est.append(float(r_best))
    return list(map(float,phi_est)), list(map(float,theta_est)), r_est


def hybrid_estimate_raw(model, sample, force_K=None, prefer_logits=True, angle_source="aux"):
    """Fast, no-refinement estimate using the model's soft-argmax angles and auxiliary range.
    Returns (phi_list, theta_list, r_list) with K chosen from logits or force_K.
    """
    import numpy as np
    def to_ri_t(z): return torch.stack([z.real, z.imag], dim=-1)
    device = next(model.parameters()).device
    # y/H/codes conversion (same as final)
    if "y_cplx" in sample:
        y = torch.from_numpy(sample["y_cplx"]).to(torch.complex64).unsqueeze(0).to(device)
        H = torch.from_numpy(sample["H_cplx"]).to(torch.complex64).unsqueeze(0).to(device)
    else:
        y_data = sample["y"]; H_data = sample["H"]; codes_data = sample["codes"]
        if torch.is_tensor(y_data): y_data = y_data.numpy()
        if torch.is_tensor(H_data): H_data = H_data.numpy()
        if torch.is_tensor(codes_data): codes_data = codes_data.numpy()
        y = torch.from_numpy(y_data[:, :, 0] + 1j * y_data[:, :, 1]).to(torch.complex64).unsqueeze(0).to(device)
        H = torch.from_numpy(H_data[:, :, 0] + 1j * H_data[:, :, 1]).to(torch.complex64).unsqueeze(0).to(device)
    codes_data = sample["codes"]
    if torch.is_tensor(codes_data): codes_data = codes_data.numpy()
    codes = torch.from_numpy(codes_data[:, :, 0] + 1j * codes_data[:, :, 1]).to(torch.complex64).unsqueeze(0).to(device)

    # Optional R_samp for hybrid-aware K-head
    try:
        from .angle_pipeline import build_sample_covariance_from_snapshots
        L_snap = y.shape[1]
        y_np = (y[0].real.cpu().numpy() + 1j * y[0].imag.cpu().numpy())  # [L,M]
        H_np = (H[0].real.cpu().numpy() + 1j * H[0].imag.cpu().numpy())  # [M,N]
        codes_np = (codes[0].real.cpu().numpy() + 1j * codes[0].imag.cpu().numpy())  # [L,N]
        H_stack = np.repeat(H_np[None, :, :], L_snap, axis=0)
        R_samp_np = build_sample_covariance_from_snapshots(y_np, H_stack, codes_np, cfg)
        R_samp_t = torch.from_numpy(R_samp_np).to(torch.complex64).unsqueeze(0).to(device)
    except Exception:
        R_samp_t = None
    with torch.no_grad():
        pred = model(to_ri_t(y), to_ri_t(H), to_ri_t(codes), R_samp=R_samp_t)

    # K from logits
    if force_K is not None:
        K_hat = int(force_K)
    else:
        if "k_logits" in pred:
            k_temp = getattr(model, '_k_calibration_temp', 1.0)
            pK = torch.softmax(pred["k_logits"][0] / k_temp, -1)
            k_from_logits = int(torch.argmax(pK).item()) + 1  # Map from class index to K
            K_hat = min(max(1, k_from_logits), cfg.K_MAX) if prefer_logits else max(1, int(cfg.K_MAX/2))
        else:
            K_hat = 1

    KMAX = int(cfg.K_MAX)
    
    # SOTA FIX: Always use unified angle pipeline (ignore angle_source)
    # This ensures train==eval==infer consistency
    from .angle_pipeline import angle_pipeline
    
    # Get covariance factor
    cf_ang = pred["cov_fact_angle"][0].detach().cpu().numpy()  # [N*2, K_MAX] real
    cf_cplx = (cf_ang[::2] + 1j*cf_ang[1::2]).reshape(cfg.N, cfg.K_MAX)  # [N, K_MAX] complex
    
    # Run unified angle pipeline
    phi_est, theta_est, _ = angle_pipeline(
        cf_cplx, K_hat, cfg,
        use_fba=getattr(cfg, "MUSIC_USE_FBA", True),
        use_adaptive_shrink=True,
        use_parabolic=getattr(cfg, "MUSIC_PEAK_REFINE", True),
        use_newton=getattr(cfg, "USE_NEWTON_REFINE", True),
        device="cpu"
    )
    
    # Range from auxiliary head (still useful for range estimation)
    aux_ptr = pred["phi_theta_r"][0].detach().cpu().numpy()
    r_all = aux_ptr[2*KMAX:3*KMAX]
    r_est = r_all[:K_hat]

    return list(map(float, phi_est)), list(map(float, theta_est)), list(map(float, r_est))



def estimate_k_blind(R, T, kmax=None):
    try:
        k = estimate_k_ic_from_cov(R, T, method='mdl', kmax=kmax)
    except Exception:
        k = None
    if (k is None) or (k<=0):
        try:
            k = estimate_k_ic_from_cov(R, T, method='aic', kmax=kmax)
        except Exception:
            k = 1
    return max(1,int(k))
