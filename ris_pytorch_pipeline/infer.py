
import numpy as np, torch
from .configs import cfg, mdl_cfg
from .model import HybridModel, SpectrumRefiner
from .physics import nearfield_vec, shrink
from pathlib import Path
import json
import torch.nn.functional as F


_REFINER_FALLBACK_COUNT = 0

def _log_refiner_fallback_once(reason: str):
    """
    Avoid spamming logs when running benchmark suites (called per sample).
    Prints the first few occurrences and then stays quiet.
    """
    global _REFINER_FALLBACK_COUNT
    _REFINER_FALLBACK_COUNT += 1
    # Print first N, then every 200th as a heartbeat.
    if _REFINER_FALLBACK_COUNT <= 3 or (_REFINER_FALLBACK_COUNT % 200 == 0):
        print(f"[INFER] {reason} -> MVDR fallback", flush=True)

def _load_hpo_best_dict(p=None):
    if p is None:
        p = getattr(cfg, 'HPO_BEST_JSON', str(Path(getattr(cfg, "HPO_DIR", cfg.RESULTS_DIR)) / "best.json"))
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

def _load_model_arch_from_hpo(hpo_json=None):
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

def _infer_arch_from_state_dict(sd: dict) -> dict:
    """
    Infer minimal architecture (currently just D_MODEL) from a checkpoint state_dict.
    This prevents load-time shape mismatches when cfg/`best.json` drift from the actual
    trained checkpoint (common after multiple HPO/training runs).
    """
    arch = {}
    try:
        if isinstance(sd, dict):
            if "fusion.bias" in sd:
                arch["d_model"] = int(sd["fusion.bias"].numel())
            elif "transformer.layers.0.self_attn.in_proj_weight" in sd:
                arch["d_model"] = int(sd["transformer.layers.0.self_attn.in_proj_weight"].shape[1])
            # Backward/forward compatibility for old/new tokenizers
            elif "y_conv_tok.weight" in sd:
                # [D/2, 2M, k]
                arch["d_model"] = int(sd["y_conv_tok.weight"].shape[0]) * 2
            elif "y_conv2.weight" in sd:
                arch["d_model"] = int(sd["y_conv2.weight"].shape[0])
    except Exception:
        pass
    return arch

@torch.no_grad()
def load_model(ckpt_dir=None, ckpt_name="best.pt", map_location="cpu", prefer_swa=True, *, device=None, require_refiner: bool = True):
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
    
    # Resolve checkpoint format early so we can infer architecture if needed.
    refiner_sd = None
    if isinstance(ckpt, dict) and ("backbone" in ckpt or "refiner" in ckpt):
        sd = ckpt.get("backbone", ckpt.get("model", {}))
        refiner_sd = ckpt.get("refiner")
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict):
        # Heuristic: treat as raw state_dict
        sd = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    # Try to get architecture from checkpoint first (if ckpt is a dict wrapper)
    arch = ckpt.get("arch") if isinstance(ckpt, dict) else None
    if arch is None:
        # Prefer inferring from the state_dict (always consistent with weights).
        arch = _infer_arch_from_state_dict(sd) or None
    if arch is None:
        # Fallback to HPO configuration (best.json)
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

        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        
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
        
        # Attach SpectrumRefiner (required for production inference by default)
        if refiner_sd is not None:
            refiner = SpectrumRefiner().to(next(model.parameters()).device)
            missing_r, unexpected_r = refiner.load_state_dict(refiner_sd, strict=False)
            if missing_r or unexpected_r:
                # Non-strict load tolerates small naming drift, but report for reproducibility.
                print(f"⚠️  SpectrumRefiner load: missing={len(missing_r)} unexpected={len(unexpected_r)}", flush=True)
            refiner.eval()
            # Attach to model for inference usage
            model._spectrum_refiner = refiner
            print("✅ Loaded SpectrumRefiner weights from checkpoint")
        elif require_refiner and (not bool(getattr(cfg, "ALLOW_INFER_WITHOUT_REFINER", False))):
            raise ValueError(
                "SpectrumRefiner weights not found in checkpoint. "
                "Inference now requires a Stage-2 refiner checkpoint (format: {'backbone':..., 'refiner':...}). "
                "Run `python ris_pytorch_pipeline/ris_pipeline.py train-refiner` (Option B) and use that checkpoint."
            )
        elif require_refiner:
            print("⚠️ SpectrumRefiner weights not found in checkpoint; proceeding with MVDR-only fallback (ALLOW_INFER_WITHOUT_REFINER=True).", flush=True)

        # Optionally move to a target device (useful for benchmarking/inference on GPU).
        if device is not None:
            try:
                dev = device if isinstance(device, torch.device) else torch.device(str(device))
            except Exception:
                dev = torch.device("cpu")
            if dev.type == "cuda" and (not torch.cuda.is_available()):
                dev = torch.device("cpu")
            model = model.to(dev)
            if hasattr(model, "_spectrum_refiner") and (getattr(model, "_spectrum_refiner", None) is not None):
                model._spectrum_refiner = model._spectrum_refiner.to(dev)

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
                         do_newton=True, use_hpo_knobs=True, hpo_json=None,
                         prefer_logits=True):
    """
    MVDR-first inference (K-free).

    This function keeps the legacy name for compatibility with benchmarking scripts,
    but internally it runs **MVDR peak detection** on the effective blended covariance.

    Returns:
        phi_list, theta_list in **radians**, r_list in meters.
    """
    from .covariance_utils import build_effective_cov_np
    from .music_gpu import mvdr_detect_sources, get_gpu_estimator

    # --- helpers ---
    def _to_numpy(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    def _ri_to_c_np(x_ri):
        x_ri = _to_numpy(x_ri)
        if x_ri is None:
            return None
        if np.iscomplexobj(x_ri):
            return x_ri
        if x_ri.shape[-1] == 2:
            return x_ri[..., 0] + 1j * x_ri[..., 1]
        return x_ri.astype(np.complex64)

    def _to_ri_t(z_cplx: torch.Tensor) -> torch.Tensor:
        return torch.stack([z_cplx.real, z_cplx.imag], dim=-1)

    # --- device ---
    dev = next(model.parameters()).device
    device_str = "cuda" if (dev.type == "cuda" and torch.cuda.is_available()) else "cpu"

    # --- unpack sample ---
    # Accept either complex inputs (*_cplx) or RI inputs (default dataset format).
    y_c = _ri_to_c_np(sample.get("y_cplx", sample.get("y")))
    C_c = _ri_to_c_np(sample.get("codes_cplx", sample.get("codes")))
    
    # CRITICAL: Use H_full (true BS→RIS channel [M,N]), NOT H_eff (collapsed [L,M]).
    # H_full is required for the network to see the actual sensing operator.
    H_full_c = _ri_to_c_np(sample.get("H_full_cplx", sample.get("H_full")))
    if H_full_c is None:
        raise ValueError("sample must contain H_full (shape [M,N]) - regenerate shards with store_h_full=True")
    
    if y_c is None or C_c is None:
        raise ValueError("sample must contain y and codes (either *_cplx complex or RI [...,2])")

    # y: [L,M], H_full: [M,N], codes: [L,N]
    y_t = torch.from_numpy(y_c).to(torch.complex64).unsqueeze(0).to(dev)
    H_full_t = torch.from_numpy(H_full_c).to(torch.complex64).unsqueeze(0).to(dev)
    C_t = torch.from_numpy(C_c).to(torch.complex64).unsqueeze(0).to(dev)

    snr_db = float(sample.get("snr_db", sample.get("snr", 10.0)))

    # --- model forward (no grad) ---
    # IMPORTANT: During training we pass per-sample SNR into the model; inference must match.
    # Otherwise any SNR-conditioned shrinkage/gating inside the model will be effectively disabled.
    with torch.no_grad():
        pred = model(y=_to_ri_t(y_t), H_full=_to_ri_t(H_full_t), codes=_to_ri_t(C_t), snr_db=snr_db, R_samp=None)

    # --- get R_pred (STRUCTURAL) or build from factors (legacy) ---
    def _vec2c_flat(v_flat: np.ndarray) -> np.ndarray:
        v_flat = v_flat.astype(np.float32)
        xr, xi = v_flat[::2], v_flat[1::2]
        A = (xr + 1j * xi).reshape(cfg.N, cfg.K_MAX).astype(np.complex64)  # [N,K]

        # Magnitude leash (same spirit as loss.py): prevent rare huge factor spikes from
        # overflowing before downstream trace-normalization/conditioning.
        if bool(getattr(cfg, "FACTOR_COLNORM_ENABLE", True)):
            eps = float(getattr(cfg, "FACTOR_COLNORM_EPS", 1e-6))
            max_norm = float(getattr(cfg, "FACTOR_COLNORM_MAX", 1e3))
            col = np.linalg.norm(A, axis=0, keepdims=True).astype(np.float32)
            col = np.maximum(col, eps)
            if max_norm > 0:
                col = np.minimum(col, max_norm)
            A = (A / col).astype(np.complex64)
        return A

    if "R_pred" in pred:
        # NEW: Structural covariance from geometry predictions
        R_pred = _to_numpy(pred["R_pred"][0])
        if torch.is_tensor(R_pred):
            R_pred = R_pred.detach().cpu().numpy()
        if not np.iscomplexobj(R_pred):
            # shouldn't happen; keep robust
            R_pred = R_pred.astype(np.complex64)
        A_ang, A_rng, lam_range_factor = None, None, None
    else:
        # Legacy: build from factors
        A_ang = _vec2c_flat(_to_numpy(pred["cov_fact_angle"][0]))
        A_rng = _vec2c_flat(_to_numpy(pred["cov_fact_range"][0]))
        lam_range_factor = float(getattr(mdl_cfg, "LAM_RANGE_FACTOR", 0.3))
        R_pred = (A_ang @ A_ang.conj().T) + lam_range_factor * (A_rng @ A_rng.conj().T)

    # --- optional R_samp (hybrid blending) ---
    R_samp = None
    if sample.get("R_samp") is not None:
        R_samp = _ri_to_c_np(sample.get("R_samp"))
    else:
        # Build R_samp from snapshots using H_full (already loaded above)
        try:
            from .angle_pipeline import build_sample_covariance_from_snapshots
            # build_sample_covariance_from_snapshots accepts [M,N] directly (no need to repeat)
            R_samp = build_sample_covariance_from_snapshots(y_c, H_full_c, C_c, cfg, tikhonov_alpha=1e-3)
        except Exception:
            R_samp = None

    beta = float(getattr(cfg, "HYBRID_COV_BETA", 0.0))
    R_eff = build_effective_cov_np(
        R_pred,
        R_samp=R_samp,
        beta=beta if (R_samp is not None) else 0.0,
        diag_load=True,
        apply_shrink=True,
        snr_db=snr_db,
        target_trace=float(cfg.N),
    )

    # --- MVDR detection (K-free) ---
    max_sources = int(force_K) if (force_K is not None) else int(cfg.K_MAX)

    use_refiner_cfg = bool(getattr(cfg, "USE_SPECTRUM_REFINER_IN_INFER", True))
    has_refiner = hasattr(model, "_spectrum_refiner") and (getattr(model, "_spectrum_refiner", None) is not None)
    allow_fallback = bool(getattr(cfg, "REFINER_GUARD_FALLBACK_TO_MVDR", True))
    # Structural mode: refiner path currently relies on low-rank factors; force MVDR fallback.
    if (A_ang is None) or (A_rng is None):
        use_refiner_cfg = False

    # Helper: raw MVDR fallback path (K-free, uses robust thresholding).
    def _mvdr_fallback():
        try:
            est = get_gpu_estimator(cfg, device=device_str)
            sources, _spec = mvdr_detect_sources(
                R_eff,
                cfg,
                device=device_str,
                grid_phi=int(getattr(cfg, "MVDR_GRID_PHI", 361)),
                grid_theta=int(getattr(cfg, "MVDR_GRID_THETA", 181)),
                r_planes=getattr(cfg, "REFINER_R_PLANES", None) or est.default_r_planes_mvdr,
                delta_scale=float(getattr(cfg, "MVDR_DELTA_SCALE", 1e-2)),
                threshold_db=float(getattr(cfg, "MVDR_THRESH_DB", 12.0)),
                threshold_mode=str(getattr(cfg, "MVDR_THRESH_MODE", "mad")),
                cfar_z=float(getattr(cfg, "MVDR_CFAR_Z", 5.0)),
                max_sources=max_sources,
                force_k=(int(force_K) if (force_K is not None) else None),
                do_refinement=bool(getattr(cfg, "MVDR_DO_REFINEMENT", True)),
            )
            if len(sources) == 0:
                return [], [], []
            phi_deg = np.array([s[0] for s in sources], dtype=np.float32)
            theta_deg = np.array([s[1] for s in sources], dtype=np.float32)
            r_m = np.array([s[2] for s in sources], dtype=np.float32)
            return np.deg2rad(phi_deg).tolist(), np.deg2rad(theta_deg).tolist(), r_m.tolist()
        except Exception:
            return [], [], []

    # If refiner is disabled/missing, fall back if allowed.
    if (not use_refiner_cfg) or (not has_refiner):
        if allow_fallback:
            global _REFINER_FALLBACK_COUNT
            _REFINER_FALLBACK_COUNT += 1
            # Avoid per-scene spam during HPO end-to-end evaluation.
            # Default: keep existing behavior; in HPO mode default to silence unless overridden.
            if bool(getattr(cfg, "HPO_MODE", False)):
                nlog = int(getattr(cfg, "HPO_REFINER_REJECT_LOG_EVERY", 0))
            else:
                nlog = int(getattr(cfg, "REFINER_REJECT_LOG_EVERY", 1))
            if nlog != 0 and ((_REFINER_FALLBACK_COUNT - 1) % max(1, nlog) == 0):
                _log_refiner_fallback_once("Refiner unavailable/disabled")
            return _mvdr_fallback()
        raise ValueError(
            "SpectrumRefiner is required for inference but was not attached to the model and fallback is disabled. "
            "Load a Stage-2 refiner checkpoint or set REFINER_GUARD_FALLBACK_TO_MVDR=True."
        )

    if True:
        # Refiner-assisted inference:
        # 1) low-rank MVDR spectrum max over range planes (from factors)
        # 2) SpectrumRefiner -> probability heatmap
        # 3) 2D NMS peak picking
        # 4) per-peak range selection over r_planes

        est = get_gpu_estimator(cfg, device=device_str)
        grid_phi = torch.linspace(
            np.deg2rad(float(getattr(cfg, "PHI_MIN_DEG", -60.0))),
            np.deg2rad(float(getattr(cfg, "PHI_MAX_DEG", 60.0))),
            int(getattr(cfg, "MVDR_GRID_PHI", 361)),
            device=est.device,
            dtype=torch.float32,
        )
        grid_theta = torch.linspace(
            np.deg2rad(float(getattr(cfg, "THETA_MIN_DEG", -30.0))),
            np.deg2rad(float(getattr(cfg, "THETA_MAX_DEG", 30.0))),
            int(getattr(cfg, "MVDR_GRID_THETA", 181)),
            device=est.device,
            dtype=torch.float32,
        )
        r_planes = getattr(cfg, "REFINER_R_PLANES", None) or est.default_r_planes_mvdr

        # Build low-rank factors F = [A_ang, sqrt(lam)*A_rng]
        if (A_ang is None) or (A_rng is None):
            # Should not reach here because we force use_refiner_cfg=False above.
            return _mvdr_fallback()
        F_b = np.concatenate([A_ang, np.sqrt(lam_range_factor) * A_rng], axis=1).astype(np.complex64)
        F_t = torch.as_tensor(F_b, device=est.device, dtype=torch.complex64)

        delta_scale = float(getattr(cfg, "MVDR_DELTA_SCALE", 1e-2))
        spec = est.mvdr_spectrum_max_2_5d_lowrank(F_t, grid_phi, grid_theta, r_planes, delta_scale=delta_scale)  # [G_phi,G_theta]
        spec = spec.unsqueeze(0).unsqueeze(0)  # [1,1,G_phi,G_theta]

        refiner = getattr(model, "_spectrum_refiner")
        refiner = refiner.to(est.device)
        refiner.eval()
        prob = refiner(spec)  # [1,1,G_phi,G_theta]

        # 2D NMS peak detection
        min_sep = int(getattr(cfg, "REFINER_NMS_MIN_SEP", 2))
        prob2d = prob[0, 0]
        ksize = 2 * min_sep + 1
        pooled = F.max_pool2d(prob2d[None, None], kernel_size=ksize, stride=1, padding=min_sep)[0, 0]
        is_max = (prob2d >= pooled)

        # Guardrail: reject pathological refiner outputs (flat/saturated/non-finite/too-many-peaks).
        if bool(getattr(cfg, "REFINER_GUARD_ENABLE", True)):
            try:
                if (not torch.isfinite(prob2d).all()):
                    raise RuntimeError("non-finite prob map")
                pstd = float(prob2d.std().item())
                if pstd < float(getattr(cfg, "REFINER_GUARD_MIN_STD", 1e-4)):
                    raise RuntimeError(f"prob map too flat (std={pstd:.3e})")
                sat = float((prob2d > 0.99).float().mean().item())
                if sat > float(getattr(cfg, "REFINER_GUARD_MAX_SAT_FRAC", 0.10)):
                    raise RuntimeError(f"prob map too saturated (sat_frac={sat:.3f})")
            except Exception as e:
                if allow_fallback:
                    if (not bool(getattr(cfg, "HPO_MODE", False))) and (int(getattr(cfg, "REFINER_REJECT_LOG_EVERY", 1)) != 0):
                        _log_refiner_fallback_once(f"Refiner rejected ({e})")
                    return _mvdr_fallback()
                raise

        # Apply absolute or relative thresholding (relative is default; avoids brittle calibration).
        peak_thresh = getattr(cfg, "REFINER_PEAK_THRESH", None)
        # Oracle-K mode: do NOT apply thresholds; just take the top-K local maxima.
        if force_K is None:
            if peak_thresh is not None:
                is_max = is_max & (prob2d >= float(peak_thresh))
            else:
                rel = float(getattr(cfg, "REFINER_REL_THRESH", 0.20))
                rel = float(np.clip(rel, 0.0, 1.0))
                pmax = float(prob2d.max().item()) if prob2d.numel() > 0 else 0.0
                is_max = is_max & (prob2d >= (rel * pmax))

        idx = torch.nonzero(is_max, as_tuple=False)
        if bool(getattr(cfg, "REFINER_GUARD_ENABLE", True)):
            max_raw = int(getattr(cfg, "REFINER_GUARD_MAX_RAW_PEAKS", 2000))
            if idx.numel() > 0 and idx.shape[0] > max_raw:
                if allow_fallback:
                    if (not bool(getattr(cfg, "HPO_MODE", False))) and (int(getattr(cfg, "REFINER_REJECT_LOG_EVERY", 1)) != 0):
                        _log_refiner_fallback_once(f"Refiner rejected (too many peaks: {idx.shape[0]})")
                    return _mvdr_fallback()
                # else: keep going and rely on top-K truncation
        if idx.numel() == 0:
            # Fallback: take top-K from the full grid (no NMS).
            flat = prob2d.flatten()
            if flat.numel() == 0:
                return [], [], []
            k_take = int(force_K) if (force_K is not None) else 1
            k_take = max(1, min(k_take, int(flat.numel())))
            inds = torch.topk(flat, k_take).indices
            idx = torch.stack([inds // prob2d.shape[1], inds % prob2d.shape[1]], dim=1)

        # Sort peaks by probability desc
        vals = prob2d[idx[:, 0], idx[:, 1]]
        order = torch.argsort(vals, descending=True)
        # In oracle-K mode, return exactly K peaks (up to available).
        k_keep = int(force_K) if (force_K is not None) else int(max_sources)
        k_keep = max(1, min(k_keep, int(idx.shape[0])))
        idx = idx[order][:k_keep]

        # Convert to angles and pick best range plane per peak
        sources = []
        for (pi, ti) in idx.tolist():
            phi = float(torch.rad2deg(grid_phi[pi]).item())
            theta = float(torch.rad2deg(grid_theta[ti]).item())

            # Find best r by evaluating MVDR at this (phi,theta) over r_planes
            phi_g = grid_phi[pi:pi+1]
            theta_g = grid_theta[ti:ti+1]
            best_r = float(r_planes[0])
            best_v = -1.0
            for r in r_planes:
                A1 = est._steering_nearfield_grid(phi_g, theta_g, float(r)).reshape(1, -1)  # [1,N]
                v = est._compute_spectrum_mvdr_lowrank(A1.to(torch.complex64), F_t, delta_scale=delta_scale)[0].item()
                if v > best_v:
                    best_v = float(v)
                    best_r = float(r)
            conf = float(prob2d[pi, ti].item())
            sources.append((phi, theta, best_r, conf))
    else:
        # Legacy non-refiner inference path removed (refiner is mandatory).
        sources = []

    if len(sources) == 0:
        return [], [], []

    phi_deg = np.array([s[0] for s in sources], dtype=np.float32)
    theta_deg = np.array([s[1] for s in sources], dtype=np.float32)
    r_m = np.array([s[2] for s in sources], dtype=np.float32)

    # Return radians for compatibility with existing evaluation code
    return np.deg2rad(phi_deg).tolist(), np.deg2rad(theta_deg).tolist(), r_m.tolist()


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
        # Prefer H_full if provided; H_cplx is legacy H_eff and is no longer accepted by the backbone.
        if sample.get("H_full_cplx") is not None:
            H_full = torch.from_numpy(sample["H_full_cplx"]).to(torch.complex64).unsqueeze(0).to(device)
        elif sample.get("H_full") is not None:
            H_full_np = sample["H_full"]
            if torch.is_tensor(H_full_np):
                H_full_np = H_full_np.numpy()
            H_full = torch.from_numpy(H_full_np[..., 0] + 1j * H_full_np[..., 1]).to(torch.complex64).unsqueeze(0).to(device)
        else:
            raise ValueError("sample must contain H_full for inference (H_cplx/H_eff is not supported)")
    else:
        y_data = sample["y"]; codes_data = sample["codes"]
        H_full_data = sample.get("H_full", None)
        if H_full_data is None:
            raise ValueError("sample must contain H_full for inference (H/H_eff is not supported)")
        if torch.is_tensor(y_data): y_data = y_data.numpy()
        if torch.is_tensor(H_full_data): H_full_data = H_full_data.numpy()
        if torch.is_tensor(codes_data): codes_data = codes_data.numpy()
        y = torch.from_numpy(y_data[:, :, 0] + 1j * y_data[:, :, 1]).to(torch.complex64).unsqueeze(0).to(device)
        # H_full: [M,N,2] -> complex [M,N]
        H_full = torch.from_numpy(H_full_data[:, :, 0] + 1j * H_full_data[:, :, 1]).to(torch.complex64).unsqueeze(0).to(device)
    codes_data = sample["codes"]
    if torch.is_tensor(codes_data): codes_data = codes_data.numpy()
    codes = torch.from_numpy(codes_data[:, :, 0] + 1j * codes_data[:, :, 1]).to(torch.complex64).unsqueeze(0).to(device)

    # Optional R_samp for hybrid-aware K-head
    R_samp_t = None
    try:
        # Prefer precomputed R_samp if available
        if sample.get("R_samp") is not None:
            R_samp_raw = sample["R_samp"]
            if torch.is_tensor(R_samp_raw):
                R_samp_raw = R_samp_raw.numpy()
            R_samp_raw = np.asarray(R_samp_raw)
            if R_samp_raw.shape[-1] == 2:
                R_samp_np = (R_samp_raw[..., 0] + 1j * R_samp_raw[..., 1]).astype(np.complex64)
            else:
                R_samp_np = R_samp_raw.astype(np.complex64)
            R_samp_t = torch.from_numpy(R_samp_np).to(torch.complex64).unsqueeze(0).to(device)
        # Otherwise try building from H_full (not H, which is per-snapshot effective channel)
        elif sample.get("H_full") is not None:
            from .angle_pipeline import build_sample_covariance_from_snapshots
            H_full_raw = sample["H_full"]
            if torch.is_tensor(H_full_raw):
                H_full_raw = H_full_raw.numpy()
            H_full_raw = np.asarray(H_full_raw)
            if H_full_raw.shape[-1] == 2:
                H_full_np = (H_full_raw[..., 0] + 1j * H_full_raw[..., 1]).astype(np.complex64)  # [M,N]
            else:
                H_full_np = H_full_raw.astype(np.complex64)
            y_np = (y[0].cpu().numpy().real + 1j * y[0].cpu().numpy().imag).astype(np.complex64)  # [L,M]
            codes_np = (codes[0].cpu().numpy().real + 1j * codes[0].cpu().numpy().imag).astype(np.complex64)  # [L,N]
            R_samp_np = build_sample_covariance_from_snapshots(y_np, H_full_np, codes_np, cfg)
            R_samp_t = torch.from_numpy(R_samp_np).to(torch.complex64).unsqueeze(0).to(device)
    except Exception:
        R_samp_t = None
    with torch.no_grad():
        pred = model(y=to_ri_t(y), H_full=to_ri_t(H_full), codes=to_ri_t(codes), R_samp=R_samp_t)

    # K estimation: use MDL (K-head removed)
    if force_K is not None:
        K_hat = int(force_K)
    else:
        # Fallback to MDL on R_blend/sample covariance
        try:
            if 'R_pred' in pred:
                R_est = pred['R_pred'][0].detach().cpu().numpy()
            elif 'R_blend' in pred:
                R_est = pred['R_blend'][0].detach().cpu().numpy()
            else:
                # Legacy: Build from factors
                cf_ang = pred["cov_fact_angle"][0].detach().cpu().numpy()
                cf_cplx = (cf_ang[::2] + 1j*cf_ang[1::2]).reshape(cfg.N, cfg.K_MAX)
                R_est = cf_cplx @ cf_cplx.conj().T
            L_snap = y.shape[1] if hasattr(y, 'shape') else cfg.L
            K_hat = estimate_k_ic_from_cov(R_est, L_snap, method="mdl", kmax=cfg.K_MAX)
            K_hat = max(1, min(K_hat, cfg.K_MAX))
        except Exception:
            K_hat = 1  # Fallback

    KMAX = int(cfg.K_MAX)
    
    # Structural mode: use MVDR detection directly from R_pred.
    if 'R_pred' in pred:
        from .covariance_utils import build_effective_cov_np
        from .music_gpu import mvdr_detect_sources, get_gpu_estimator
        R_pred_np = pred['R_pred'][0].detach().cpu().numpy()
        snr_db = float(sample.get("snr_db", sample.get("snr", 10.0)))
        R_eff = build_effective_cov_np(
            R_pred_np,
            R_samp=None,
            beta=0.0,
            diag_load=True,
            apply_shrink=True,
            snr_db=snr_db,
            target_trace=float(cfg.N),
        )
        est = get_gpu_estimator(cfg, device=("cuda" if torch.cuda.is_available() else "cpu"))
        sources, _ = mvdr_detect_sources(
            R_eff, cfg,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            grid_phi=int(getattr(cfg, "MVDR_GRID_PHI", 361)),
            grid_theta=int(getattr(cfg, "MVDR_GRID_THETA", 181)),
            r_planes=getattr(cfg, "REFINER_R_PLANES", None) or est.default_r_planes_mvdr,
            delta_scale=float(getattr(cfg, "MVDR_DELTA_SCALE", 1e-2)),
            threshold_db=float(getattr(cfg, "MVDR_THRESH_DB", 12.0)),
            threshold_mode=str(getattr(cfg, "MVDR_THRESH_MODE", "mad")),
            cfar_z=float(getattr(cfg, "MVDR_CFAR_Z", 5.0)),
            max_sources=int(force_K) if force_K is not None else int(cfg.K_MAX),
            force_k=(int(force_K) if force_K is not None else None),
            do_refinement=bool(getattr(cfg, "MVDR_DO_REFINEMENT", True)),
        )
        if len(sources) == 0:
            return [], [], []
        phi = [float(np.deg2rad(s[0])) for s in sources[:K_hat]]
        th = [float(np.deg2rad(s[1])) for s in sources[:K_hat]]
        rr = [float(s[2]) for s in sources[:K_hat]]
        return phi, th, rr

    # Legacy mode: use factors + angle pipeline (kept for backward compatibility)
    from .angle_pipeline import angle_pipeline
    cf_ang = pred["cov_fact_angle"][0].detach().cpu().numpy()
    cf_cplx = (cf_ang[::2] + 1j * cf_ang[1::2]).reshape(cfg.N, cfg.K_MAX)
    phi_est, theta_est, _ = angle_pipeline(
        cf_cplx, K_hat, cfg,
        use_fba=getattr(cfg, "MUSIC_USE_FBA", True),
        use_adaptive_shrink=True,
        use_parabolic=getattr(cfg, "MUSIC_PEAK_REFINE", True),
        use_newton=getattr(cfg, "USE_NEWTON_REFINE", True),
        device="cpu",
    )
    aux_ptr = pred["phi_theta_r"][0].detach().cpu().numpy()
    r_all = aux_ptr[2 * KMAX:3 * KMAX]
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
