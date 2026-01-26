"""
Unified angle estimation pipeline for train/eval/infer consistency.

This module ensures the SAME angle processing stack is used across:
  - Training evaluation (Hungarian metrics)
  - Inference (production)
  
Pipeline: MUSIC coarse → Parabolic refine → Newton refine → angles

L=16 CRITICAL FIX: Added hybrid covariance blending to exploit multi-snapshot diversity.
"""

import numpy as np
import torch
from typing import Tuple, Optional

from .eval_angles import music2d_from_cov_factor
from .nf_mle_refine import nf_mle_polish_after_newton
from .covariance_utils import build_effective_cov_np, trace_norm_np

# GPU MUSIC import (optional, falls back to CPU if unavailable)
try:
    from .music_gpu import GPUMusicEstimator, get_gpu_estimator
    _GPU_MUSIC_AVAILABLE = True
except ImportError:
    _GPU_MUSIC_AVAILABLE = False


def norm_trace(R, N):
    """
    Trace-normalize covariance matrix to tr(R) = N.
    
    This ensures both R_pred and R_samp have the same scale before blending,
    making the blend_beta parameter meaningful.
    
    Args:
        R: Covariance matrix [N, N] complex
        N: Target trace value (typically number of array elements)
    
    Returns:
        R_norm: Trace-normalized covariance [N, N] complex
    """
    tr = np.real(np.trace(R))
    if tr > 1e-12:
        return (N / tr) * R
    else:
        return R


def build_sample_covariance_from_snapshots(y, H, codes, cfg, tikhonov_alpha=1e-3):
    """
    Build sample-based covariance from L=16 coded snapshots.
    
    This exploits the multi-snapshot diversity that the network prediction doesn't directly expose.
    
    Args:
        y: Received signals [L, M_BS] complex (L snapshots, M_BS BS antennas)
        H: BS-RIS channel matrix [L, M_BS, N] complex (constant H repeated L times for convenience)
        codes: RIS phase codes [L, N] complex
        cfg: Config object
        tikhonov_alpha: Regularization for least squares (default: 1e-3)
    
    Returns:
        R_samp: Sample covariance [N, N] complex, trace-normalized and full-rank
    
    Algorithm (RIS-domain covariance estimate):
        Forward model: y[l] = (H @ diag(codes[l])) @ x[l] + noise
        where x[l] is the RIS-element domain signal at snapshot l.

        We estimate x_hat[l] from each snapshot, then form:
            R_samp = (1/L) Σ_l x_hat[l] x_hat[l]^H

        IMPORTANT:
          - Do NOT mean-center x_hat across snapshots. Mean-centering removes the signal
            component when x[l] has nonzero mean (common), leaving mostly noise covariance.
          - To make R_samp meaningful for multi-source covariance, the simulator should
            use time-varying source symbols across snapshots (see dataset.py).
    """
    debug = bool(getattr(cfg, "DEBUG_BUILD_R_SAMP", False))
    if debug:
        # DEBUG: Print input shapes
        print(f"     [DEBUG build_R_samp] y.shape={y.shape}, H.shape={H.shape}, codes.shape={codes.shape}", flush=True)
        print(f"     [DEBUG build_R_samp] y norm={np.linalg.norm(y):.2e}, H norm={np.linalg.norm(H):.2e}, codes norm={np.linalg.norm(codes):.2e}", flush=True)
    
    L = int(y.shape[0])
    M = int(y.shape[1])
    N = int(getattr(cfg, "N_H", 12) * getattr(cfg, "N_V", 12))

    # Accept either H=[M,N] or H=[L,M,N] (some call-sites repeat H across L).
    H = np.asarray(H)
    if H.ndim == 2:
        H_stack = np.repeat(H[None, :, :], L, axis=0)
    elif H.ndim == 3 and H.shape[0] == L:
        H_stack = H
    else:
        raise ValueError(f"H must be [M,N] or [L,M,N], got {H.shape}")

    # Solver choice:
    # - matched_filter: fast heuristic (often too weak for MVDR peak picking)
    # - ridge_ls: per-snapshot ridge solve (still limited because M < N)
    # - als_lowrank: joint low-rank alternating least squares across snapshots (recommended for offline shards)
    solver = str(getattr(cfg, "RSAMP_SOLVER", "als_lowrank")).lower().strip()

    if solver == "als_lowrank":
        # Joint low-rank ALS across snapshots:
        #   y_l ≈ (H @ diag(c_l)) @ (F s_l) ,  F:[N,K], s_l:[K]
        # Solve for {s_l} given F, then update F via stacked ridge LS; repeat.
        K = int(getattr(cfg, "RSAMP_ALS_K", getattr(cfg, "K_MAX", 5)))
        K = max(1, min(K, int(getattr(cfg, "K_MAX", K))))
        n_iter = int(getattr(cfg, "RSAMP_ALS_ITERS", 4))
        ridge = float(getattr(cfg, "RSAMP_ALS_RIDGE", 1e-2))

        # Stack per-snapshot sensing matrices B_l = H * c_l[None,:]
        B_stack = np.zeros((L * M, N), dtype=np.complex128)
        y_stack = np.zeros((L * M,), dtype=np.complex128)
        for ell in range(L):
            H_ell = H_stack[ell].astype(np.complex128)
            c_ell = codes[ell].astype(np.complex128)
            B_l = H_ell * c_ell[None, :]  # [M,N]
            B_stack[ell * M : (ell + 1) * M, :] = B_l
            y_stack[ell * M : (ell + 1) * M] = y[ell].astype(np.complex128)

        # Init F: random + small matched-filter hint
        rng = np.random.default_rng(int(getattr(cfg, "RSAMP_ALS_SEED", 0)))
        F = (rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K))).astype(np.complex128) / np.sqrt(2.0)
        # Normalize columns
        F = F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-12)

        # ALS loop
        s = np.zeros((L, K), dtype=np.complex128)
        eyeK = np.eye(K, dtype=np.complex128)
        eyeNK = np.eye(N * K, dtype=np.complex128)
        for _ in range(max(1, n_iter)):
            # 1) Solve s_l given F
            for ell in range(L):
                B_l = B_stack[ell * M : (ell + 1) * M, :]  # [M,N]
                A_l = B_l @ F  # [M,K]
                AH_A = A_l.conj().T @ A_l
                AH_y = A_l.conj().T @ y[ell].astype(np.complex128)
                tr = float(np.trace(AH_A).real)
                lam = ridge * tr / max(1.0, float(K))
                try:
                    s[ell] = np.linalg.solve(AH_A + lam * eyeK, AH_y)
                except np.linalg.LinAlgError:
                    s[ell] = np.linalg.lstsq(AH_A + lam * eyeK, AH_y, rcond=None)[0]

            # 2) Solve F given s (stacked ridge LS over vec(F))
            # Build A_big = [D_1 B_stack, ..., D_K B_stack] where D_k scales rows by s_lk per snapshot.
            A_big = np.zeros((L * M, N * K), dtype=np.complex128)
            for k in range(K):
                col0 = k * N
                # row scale factors for this component: s[ell,k] repeated M times
                scale = np.repeat(s[:, k], M).astype(np.complex128)  # [L*M]
                A_big[:, col0 : col0 + N] = (scale[:, None] * B_stack)

            # Ridge solve
            AH_A = A_big.conj().T @ A_big
            AH_y = A_big.conj().T @ y_stack
            tr = float(np.trace(AH_A).real)
            lam = ridge * tr / max(1.0, float(N * K))
            try:
                f_vec = np.linalg.solve(AH_A + lam * eyeNK, AH_y)
            except np.linalg.LinAlgError:
                f_vec = np.linalg.lstsq(AH_A + lam * eyeNK, AH_y, rcond=None)[0]
            F = f_vec.reshape(K, N).T  # [N,K]

            # Re-normalize columns to control scale ambiguity
            F = F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-12)

        R_samp = (F @ F.conj().T).astype(np.complex64)

    else:
        # Per-snapshot heuristic x_hat -> covariance (limited because M < N per snapshot)
        x_hat = np.zeros((L, N), dtype=np.complex128)
        for ell in range(L):
            H_ell = H_stack[ell].astype(np.complex128)     # [M,N]
            c_ell = codes[ell].astype(np.complex128)       # [N]
            y_ell = y[ell].astype(np.complex128)           # [M]
            # Effective sensing matrix: G = H @ diag(codes[ell]) == H * codes[ell][None,:]
            G = H_ell * c_ell[None, :]                     # [M,N]
            if solver == "ridge_ls":
                GH_G = G.conj().T @ G                      # [N,N]
                GH_y = G.conj().T @ y_ell                  # [N]
                tr = float(np.trace(GH_G).real)
                alpha_scaled = float(tikhonov_alpha) * tr / max(1, N)
                reg = GH_G + alpha_scaled * np.eye(N, dtype=np.complex128)
                try:
                    x_hat[ell] = np.linalg.solve(reg, GH_y)
                except np.linalg.LinAlgError:
                    x_hat[ell] = np.linalg.lstsq(reg, GH_y, rcond=None)[0]
            else:
                # Matched filter: cheap and stable (heuristic covariance proxy).
                x_hat[ell] = (G.conj().T @ y_ell) / max(1.0, float(M))

        # Sample covariance over snapshots (NO mean-centering).
        R_samp = (x_hat.conj().T @ x_hat).astype(np.complex64) / max(1.0, float(L))  # [N,N]

    # Ensure complex64
    R_samp = R_samp.astype(np.complex64, copy=False)
    
    # Hermitize and trace-normalize
    R_samp = 0.5 * (R_samp + R_samp.conj().T)
    trace_R = np.real(np.trace(R_samp))
    if trace_R > 1e-9:
        R_samp = R_samp * (N / trace_R)
    
    return R_samp


def newton_refine_angles(phi_init, theta_init, R, K, cfg, 
                         max_iters=10, step_size=0.5, min_sep_deg=0.5,
                         r_init=None, use_nearfield=False):
    """
    Newton/Gauss-Newton refinement of angles using MUSIC cost gradients.
    
    ICC FIX: Added near-field option for sub-degree accuracy unlock.
    
    Args:
        phi_init: Initial azimuth angles [K] (degrees)
        theta_init: Initial elevation angles [K] (degrees)
        R: Covariance matrix [N, N] complex
        K: Number of sources (int)
        cfg: Config object
        max_iters: Maximum Newton iterations (default: 10)
        step_size: Step size scaling (default: 0.5 for safety)
        min_sep_deg: Minimum separation between sources (degrees, default: 0.5)
        r_init: Initial range estimates [K] (meters, optional for near-field)
        use_nearfield: If True, use spherical near-field steering (default: False)
    
    Returns:
        phi_refined, theta_refined: Refined angles [K] (degrees)
    """
    # Extract geometry
    N_H = int(getattr(cfg, "N_H", 12))
    N_V = int(getattr(cfg, "N_V", 12))
    d_h = float(getattr(cfg, "d_H", 0.5))
    d_v = float(getattr(cfg, "d_V", 0.5))
    lam = float(getattr(cfg, "WAVEL", 0.0625))
    
    # Angle ranges (radians for internal computation)
    phi_min = float(getattr(cfg, "ANGLE_RANGE_PHI", np.pi/3))
    theta_min = float(getattr(cfg, "ANGLE_RANGE_THETA", np.pi/6))
    
    # Convert to radians
    phi = np.deg2rad(phi_init.copy())
    theta = np.deg2rad(theta_init.copy())
    
    # Move R to torch for gradient computation
    if isinstance(R, np.ndarray):
        R = torch.as_tensor(R, dtype=torch.complex64)
    
    # Noise projector from EVD
    evals, evecs = torch.linalg.eigh(R)
    U_noise = evecs[:, :( R.shape[0] - K)] if K > 0 else evecs[:, :1]
    G = U_noise @ U_noise.conj().T
    G_np = G.cpu().numpy()
    
    k_wave = 2.0 * np.pi / lam
    
    def planar_steer(phi_r, theta_r):
        """Planar steering vector (far-field)"""
        sin_phi = np.sin(phi_r)
        cos_theta = np.cos(theta_r)
        sin_theta = np.sin(theta_r)
        
        m = np.arange(N_H)
        n = np.arange(N_V)
        
        alpha = k_wave * d_h * sin_phi * cos_theta
        beta = k_wave * d_v * sin_theta
        
        a_h = np.exp(1j * np.outer(m, alpha))  # [N_H, K]
        a_v = np.exp(1j * np.outer(n, beta))   # [N_V, K]
        
        # Kronecker: [N_H*N_V, K]
        a = np.kron(a_h, a_v).astype(np.complex64)
        return a
    
    def nearfield_steer(phi_r, theta_r, r_r):
        """
        Near-field steering vector (Fresnel / quadratic phase), aligned with this repo’s
        canonical convention in `physics.nearfield_vec` and `music_gpu.py`.

        IMPORTANT:
        - `cfg.d_H` / `cfg.d_V` are **meters** in this repo (configs.py sets them as 0.5*WAVEL).
        - Dataset generation uses the same Fresnel-style model, so this must match for
          consistent Newton refinement / debug evaluations.
        """
        # Element positions (centered UPA), meters.
        # Match dataset.py / physics.py indexing exactly (note Python precedence of unary minus with //).
        h_idx = np.arange(-(N_H - 1)//2, (N_H + 1)//2, dtype=np.float32) * d_h
        v_idx = np.arange(-(N_V - 1)//2, (N_V + 1)//2, dtype=np.float32) * d_v
        h_mesh, v_mesh = np.meshgrid(h_idx, v_idx, indexing="xy")  # shapes [N_V, N_H]
        vh = h_mesh.reshape(-1).astype(np.float32)  # [N]
        vv = v_mesh.reshape(-1).astype(np.float32)  # [N]

        # Trig
        sin_phi = np.sin(phi_r).astype(np.float32)      # [K]
        cos_theta = np.cos(theta_r).astype(np.float32)  # [K]
        sin_theta = np.sin(theta_r).astype(np.float32)  # [K]
        r_eff = np.maximum(r_r.astype(np.float32), 1e-6)  # [K]

        # Fresnel phase: k0 * (planar - curvature)
        planar = (vh[:, None] * sin_phi[None, :] * cos_theta[None, :]) + (vv[:, None] * sin_theta[None, :])  # [N,K]
        curvature = (vh[:, None]**2 + vv[:, None]**2) / (2.0 * r_eff[None, :])  # [N,K]
        phase = k_wave * (planar - curvature)  # [N,K]
        a = np.exp(1j * phase).astype(np.complex64) / np.sqrt(N)
        return a  # [N,K]
    
    # ICC FIX: Select steering model based on use_nearfield flag
    if use_nearfield and r_init is not None:
        r = r_init.copy()  # Range estimates [K]
        
        def music_cost(phi_r, theta_r):
            """MUSIC cost with near-field steering"""
            A = nearfield_steer(phi_r, theta_r, r)  # [N, K]
            denom = np.real(np.diag(A.conj().T @ G_np @ A))  # [K]
            return denom.sum()
    else:
        def music_cost(phi_r, theta_r):
            """MUSIC cost with far-field planar steering"""
            A = planar_steer(phi_r, theta_r)  # [N, K]
            denom = np.real(np.diag(A.conj().T @ G_np @ A))  # [K]
            return denom.sum()
    
    def jacobian(phi_r, theta_r, eps=1e-4):
        """Numerical Jacobian of MUSIC cost w.r.t. (phi, theta)"""
        J = np.zeros((2*K,))
        
        for k in range(K):
            # Gradient w.r.t. phi[k]
            phi_p = phi_r.copy()
            phi_p[k] += eps
            phi_m = phi_r.copy()
            phi_m[k] -= eps
            J[k] = (music_cost(phi_p, theta_r) - music_cost(phi_m, theta_r)) / (2*eps)
            
            # Gradient w.r.t. theta[k]
            theta_p = theta_r.copy()
            theta_p[k] += eps
            theta_m = theta_r.copy()
            theta_m[k] -= eps
            J[K + k] = (music_cost(phi_r, theta_p) - music_cost(phi_r, theta_m)) / (2*eps)
        
        return J
    
    # Newton iterations
    for it in range(max_iters):
        grad = jacobian(phi, theta)
        
        # Gradient descent step (negative gradient to minimize)
        dphi = -step_size * grad[:K]
        dtheta = -step_size * grad[K:]
        
        phi_new = phi + dphi
        theta_new = theta + dtheta
        
        # Clamp to bounds
        phi_new = np.clip(phi_new, -phi_min, phi_min)
        theta_new = np.clip(theta_new, -theta_min, theta_min)
        
        # Enforce minimum separation (prevent source collapse)
        for i in range(K):
            for j in range(i+1, K):
                dist = np.sqrt((phi_new[i] - phi_new[j])**2 + (theta_new[i] - theta_new[j])**2)
                if dist < np.deg2rad(min_sep_deg):
                    # Push apart slightly
                    dir_phi = phi_new[i] - phi_new[j]
                    dir_theta = theta_new[i] - theta_new[j]
                    norm = np.sqrt(dir_phi**2 + dir_theta**2) + 1e-9
                    phi_new[i] += 0.5 * np.deg2rad(min_sep_deg) * dir_phi / norm
                    phi_new[j] -= 0.5 * np.deg2rad(min_sep_deg) * dir_phi / norm
                    theta_new[i] += 0.5 * np.deg2rad(min_sep_deg) * dir_theta / norm
                    theta_new[j] -= 0.5 * np.deg2rad(min_sep_deg) * dir_theta / norm
        
        # Check convergence
        delta = np.sqrt(np.sum((phi_new - phi)**2 + (theta_new - theta)**2))
        phi, theta = phi_new, theta_new
        
        if delta < 1e-5:  # Converged
            break
    
    # Convert back to degrees
    phi_refined = np.rad2deg(phi)
    theta_refined = np.rad2deg(theta)
    
    return phi_refined, theta_refined


def angle_pipeline(cov_factor_or_R, K_est, cfg, 
                   use_fba=True, use_adaptive_shrink=True, use_parabolic=True, use_newton=True,
                   r_init=None, device="cpu",
                   y_snapshots=None, H_snapshots=None, codes_snapshots=None, blend_beta=0.2):
    """
    Unified angle estimation pipeline: MUSIC → Parabolic → Newton.
    
    This is the SINGLE SOURCE OF TRUTH for angle estimation across train/eval/infer.
    
    ICC FIX: Added r_init parameter for near-field Newton option.
    L=16 CRITICAL FIX: Added hybrid covariance blending to exploit multi-snapshot diversity.
    
    Args:
        cov_factor_or_R: Either covariance factor [N, K_MAX] complex or full R [N, N] complex
        K_est: Estimated number of sources (int)
        cfg: Config object
        use_fba: Enable Forward-Backward Averaging (default: True)
        use_adaptive_shrink: Enable adaptive shrinkage (default: True)
        use_parabolic: Enable parabolic sub-grid refinement (default: True)
        use_newton: Enable Newton refinement after MUSIC (default: True)
        r_init: Initial range estimates [K_est] (meters, optional for near-field Newton)
        device: 'cpu' or 'cuda'
        y_snapshots: Optional [L, M] complex - received signals for hybrid blending
        H_snapshots: Optional [L, M] complex - BS-RIS channels for hybrid blending
        codes_snapshots: Optional [L, N] complex - RIS codes for hybrid blending
        blend_beta: Weight for sample covariance in hybrid (default: 0.2, 0=no blend)
    
    Returns:
        phi_deg [K_est]: Azimuth angles (degrees)
        theta_deg [K_est]: Elevation angles (degrees)
        info: Dict with intermediate results (spectrum, grid, etc.)
    """
    # Sanity check
    K_est = int(np.clip(K_est, 1, getattr(cfg, "K_MAX", 5)))
    
    # If we got a factor, reconstruct R for Newton refinement later
    if cov_factor_or_R.shape[0] != cov_factor_or_R.shape[1]:
        # It's a factor [N, K_MAX]
        cf = cov_factor_or_R
        if isinstance(cf, torch.Tensor):
            cf = cf.cpu().numpy()
        R_pred = cf @ cf.conj().T
    else:
        # It's already R [N, N]
        R_pred = cov_factor_or_R
    
    # Unified: build the effective covariance once (hybrid + hermitize + trace-norm + diag-load)
    _hybrid_logged = hasattr(cfg, '_hybrid_cov_logged') and cfg._hybrid_cov_logged
    try:
        if y_snapshots is not None and H_snapshots is not None and codes_snapshots is not None and blend_beta > 0:
            R_samp_raw = build_sample_covariance_from_snapshots(y_snapshots, H_snapshots, codes_snapshots, cfg)
        else:
            R_samp_raw = None
        R = build_effective_cov_np(R_pred, R_samp=R_samp_raw, beta=float(blend_beta) if R_samp_raw is not None else None,
                                   diag_load=True, apply_shrink=False, snr_db=None, target_trace=float(R_pred.shape[0]))
        if not _hybrid_logged:
            msg = "ENABLED" if R_samp_raw is not None else "DISABLED"
            print(f"[HYBRID COV] {msg} (β={blend_beta:.3f})", flush=True)
            cfg._hybrid_cov_logged = True
    except Exception as e:
        if not _hybrid_logged:
            print(f"[HYBRID COV] ERROR: {e}. Using R_pred only!", flush=True)
            import traceback
            traceback.print_exc()
            cfg._hybrid_cov_logged = True
        R = trace_norm_np(R_pred, target_trace=float(R_pred.shape[0]))
    
    # Step 1: MUSIC coarse scan with Phase 2 enhancements
    # Note: Pass the blended R (or R_pred if no blending) to MUSIC
    phi_coarse, theta_coarse, spectrum, phi_grid, theta_grid = music2d_from_cov_factor(
        R, K_est, cfg,
        shrink=None if use_adaptive_shrink else 0.10,  # Keep default behavior; we didn't apply shrink above
        grid_phi=getattr(cfg, "MUSIC_GRID_PHI", 181),
        grid_theta=getattr(cfg, "MUSIC_GRID_THETA", 121),
        topk=K_est,
        device=device,
        use_fba=use_fba,
        peak_refine=use_parabolic,
    )
    
    # Step 2: Newton refinement (optional, but recommended)
    # ICC FIX: Now supports near-field Newton for sub-degree unlock
    if use_newton and getattr(cfg, "USE_NEWTON_REFINE", True):
        use_nearfield = getattr(cfg, "NEWTON_NEARFIELD", False)
        phi_refined, theta_refined = newton_refine_angles(
            phi_coarse, theta_coarse, R, K_est, cfg,
            max_iters=getattr(cfg, "NEWTON_ITERS", 10),
            step_size=getattr(cfg, "NEWTON_STEP", 0.5),
            min_sep_deg=getattr(cfg, "NEWTON_MIN_SEP", 0.5),
            r_init=r_init,  # Pass range estimates for near-field option
            use_nearfield=use_nearfield,  # Enable near-field steering if set
        )
    else:
        phi_refined, theta_refined = phi_coarse, theta_coarse
    
    # Step 3: Short NF-MLE polish (CRITICAL: true low-SNR robustness!)
    if getattr(cfg, "USE_NF_MLE_POLISH", True) and len(phi_refined) > 0 and y_snapshots is not None:
        # Convert snapshots to the right format: [M, L]
        Y = y_snapshots.T  # [M, L] complex64
        
        # Extract geometry
        N_H = int(getattr(cfg, "N_H", 12))
        N_V = int(getattr(cfg, "N_V", 12))
        d_h = float(getattr(cfg, "d_H", 0.5)) * float(getattr(cfg, "WAVEL", 0.0625))
        d_v = float(getattr(cfg, "d_V", 0.5)) * float(getattr(cfg, "WAVEL", 0.0625))
        lam = float(getattr(cfg, "WAVEL", 0.0625))
        k = 2.0 * np.pi / lam
        
        # Create sensor coordinates
        m_idx = np.arange(N_H) - (N_H - 1) / 2.0
        n_idx = np.arange(N_V) - (N_V - 1) / 2.0
        x_coords = np.outer(m_idx, np.ones(N_V)) * d_h
        y_coords = np.outer(np.ones(N_H), n_idx) * d_v
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()
        
        # Estimate SNR for gating (simple estimate from R)
        try:
            evals = np.linalg.eigvalsh(R)
            noise_rank = max(1, R.shape[0] - K_est)
            signal_energy = np.sum(evals[-K_est:]) if K_est > 0 else np.sum(evals)
            noise_energy = np.sum(evals[:noise_rank])
            snr_linear = signal_energy / max(noise_energy, 1e-12)
            snr_est = 10.0 * np.log10(snr_linear)
        except:
            snr_est = 10.0  # Default to high SNR
        
        # Apply NF-MLE polish
        phi_final, theta_final, r_final = nf_mle_polish_after_newton(
            phi_refined, theta_refined, np.zeros(len(phi_refined)),  # No range estimates yet
            Y, x_coords, y_coords, k, cfg, snr_est
        )
        
        # Update refined angles
        phi_refined, theta_refined = phi_final, theta_final
        nf_mle_applied = True
    else:
        nf_mle_applied = False
    
    # Prepare sensor coordinates for NF-MLE polish
    N_H = int(getattr(cfg, "N_H", 12))
    N_V = int(getattr(cfg, "N_V", 12))
    d_h = float(getattr(cfg, "d_H", 0.5)) * float(getattr(cfg, "WAVEL", 0.0625))  # Convert to meters
    d_v = float(getattr(cfg, "d_V", 0.5)) * float(getattr(cfg, "WAVEL", 0.0625))
    
    # Create sensor coordinate arrays
    m_idx = np.arange(N_H) - (N_H - 1) / 2.0  # [-5.5, ..., 5.5] for 12x12
    n_idx = np.arange(N_V) - (N_V - 1) / 2.0
    x_coords = np.outer(m_idx, np.ones(N_V)) * d_h  # [N_H, N_V]
    y_coords = np.outer(np.ones(N_H), n_idx) * d_v  # [N_H, N_V]
    x_coords = x_coords.flatten()  # [N]
    y_coords = y_coords.flatten()  # [N]
    
    # Info dict for diagnostics
    info = {
        "phi_coarse": phi_coarse,
        "theta_coarse": theta_coarse,
        "phi_refined": phi_refined,
        "theta_refined": theta_refined,
        "spectrum": spectrum,
        "phi_grid": phi_grid,
        "theta_grid": theta_grid,
        "R": R,  # Store R for range MUSIC reuse
        "Y_snapshots": y_snapshots,  # Store snapshots for NF-MLE
        "sensor_coords": (x_coords, y_coords),  # Store sensor coordinates
        "snr_est": 10.0,  # Default SNR estimate
        "nf_mle_applied": nf_mle_applied,  # NF-MLE polish status
    }
    
    return phi_refined, theta_refined, info


def angle_pipeline_gpu(cov_factor_or_R, K_est, cfg,
                       use_fba=True, use_2_5d=False,
                       r_planes=None, grid_phi=121, grid_theta=81,
                       peak_refine=True, use_newton=False,
                       R_samp=None, beta=None):
    """
    GPU-accelerated angle estimation pipeline (10-20x faster than CPU).
    
    This is optimized for validation where speed matters. Uses the GPU MUSIC
    implementation for the coarse scan, with optional Newton refinement.
    
    SHRINKAGE CONSISTENCY (Option B):
    - This function uses build_effective_cov_np to prepare R_eff
    - R_eff is then passed to GPU MUSIC with prepared=True
    - This ensures K-head and MUSIC see the SAME R_eff (no double-shrink)
    
    Args:
        cov_factor_or_R: Covariance factor [N, K_MAX] or R [N, N] complex
        K_est: Number of sources (int)
        cfg: Config object
        use_fba: Forward-Backward Averaging (applied by builder, not MUSIC)
        use_2_5d: 2.5D range-aware steering (default: False for speed)
        r_planes: Range planes for 2.5D
        grid_phi, grid_theta: Grid resolution (coarser for speed)
        peak_refine: Parabolic refinement (default: True)
        use_newton: Newton refinement (default: False for speed)
        R_samp: Optional sample covariance for hybrid blending [N, N] complex
        beta: Hybrid blending weight (0 = pure R_pred, 1 = pure R_samp)
    
    Returns:
        phi_deg [K_est], theta_deg [K_est], info dict
    """
    if not _GPU_MUSIC_AVAILABLE:
        # Fall back to CPU
        return angle_pipeline(cov_factor_or_R, K_est, cfg,
                             use_fba=use_fba, use_parabolic=peak_refine,
                             use_newton=use_newton, device="cpu")
    
    import torch
    
    # Sanity check
    K_est = int(np.clip(K_est, 1, getattr(cfg, "K_MAX", 5)))
    N = cfg.N_H * cfg.N_V
    
    # Reconstruct R_pred if needed
    if cov_factor_or_R.shape[0] != cov_factor_or_R.shape[1]:
        cf = cov_factor_or_R
        if isinstance(cf, torch.Tensor):
            cf = cf.cpu().numpy()
        R_pred = cf @ cf.conj().T
    else:
        R_pred = cov_factor_or_R
        if isinstance(R_pred, torch.Tensor):
            R_pred = R_pred.cpu().numpy()
    
    # CRITICAL: Use the canonical builder to prepare R_eff
    # This is the SAME R_eff that K-head sees during training
    # Builder handles: hybrid blending, hermitization, trace-norm, shrink, diag-load
    use_hybrid = (R_samp is not None and beta is not None and beta > 0)
    R_eff = build_effective_cov_np(
        R_pred,
        R_samp=R_samp if use_hybrid else None,
        beta=beta if use_hybrid else None,
        diag_load=True,
        apply_shrink=True,  # Builder owns shrinkage
        snr_db=None,  # Use adaptive shrinkage
        target_trace=float(N),
    )
    
    # Debug: print hybrid status for first call
    if not hasattr(angle_pipeline_gpu, '_debug_printed'):
        angle_pipeline_gpu._debug_printed = True
        print(f"[GPU MUSIC] R_pred shape: {R_pred.shape}, R_samp: {'provided' if R_samp is not None else 'None'}, beta: {beta}, use_hybrid: {use_hybrid}", flush=True)
        if use_hybrid:
            print(f"[GPU MUSIC] R_samp shape: {R_samp.shape}, ||R_pred||_F: {np.linalg.norm(R_pred, 'fro'):.2f}, ||R_samp||_F: {np.linalg.norm(R_samp, 'fro'):.2f}", flush=True)
    
    # Get GPU estimator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    estimator = get_gpu_estimator(cfg, device=device)
    
    # Run GPU MUSIC with prepared=True (skip MUSIC's own shrink/diag-load)
    phi_deg, theta_deg, r_m, spectrum = estimator.estimate_single(
        R_eff, K_est,
        use_fba=False,  # FBA already applied by builder if needed
        use_2_5d=use_2_5d,
        r_planes=r_planes,
        grid_phi=grid_phi,
        grid_theta=grid_theta,
        peak_refine=peak_refine,
        prepared=True,  # CRITICAL: R_eff is already processed
    )
    
    # Optional Newton refinement (on GPU)
    if use_newton and len(phi_deg) > 0:
        phi_deg, theta_deg, r_m = estimator.newton_refine(
            phi_deg, theta_deg, r_m, R_eff, K_est, max_iters=5, lr=0.3
        )
    
    info = {
        "phi_coarse": phi_deg,
        "theta_coarse": theta_deg,
        "phi_refined": phi_deg,
        "theta_refined": theta_deg,
        "r_est": r_m,
        "spectrum": spectrum,
        "R": R_eff,
        "R_pred": R_pred,
        "gpu_accelerated": True,
        "prepared": True,
    }
    
    return phi_deg, theta_deg, info


def full_pipeline(cov_factor_or_R, K_est, cfg, r_init=None, device="cpu"):
    """
    PRODUCTION PIPELINE: Angles → Range (complete stack).
    
    Order: 2-D MUSIC (angles) → Parabolic → Newton (angles) → 1-D NF range MUSIC
    
    This is the recommended entry point for inference and evaluation.
    
    Args:
        cov_factor_or_R: Covariance factor [N, K_MAX] or full R [N, N] complex
        K_est: Estimated number of sources (int)
        cfg: Config object
        r_init: Prior range estimates [K_est] (meters, optional)
        device: 'cpu' or 'cuda'
    
    Returns:
        phi_deg [K_est]: Azimuth angles (degrees)
        theta_deg [K_est]: Elevation angles (degrees)
        r_refined [K_est]: Refined ranges (meters)
        info: Dict with intermediate results
    """
    # Step 1 + 2: Angle pipeline (MUSIC → Parabolic → Newton)
    phi_deg, theta_deg, info = angle_pipeline(
        cov_factor_or_R, K_est, cfg,
        use_fba=getattr(cfg, "MUSIC_USE_FBA", True),
        use_adaptive_shrink=True,
        use_parabolic=getattr(cfg, "MUSIC_PEAK_REFINE", True),
        use_newton=getattr(cfg, "USE_NEWTON_REFINE", True),
        r_init=r_init,
        device=device,
    )
    
    # Step 3: 1-D range MUSIC (PRODUCTION: 2× range improvement)
    if getattr(cfg, "RANGE_MUSIC_NF", True):
        # Need to compute G (noise projector) for range MUSIC
        R = info["R"]
        
        # EVD for noise subspace (reuse from angle MUSIC logic)
        from .eval_angles import adaptive_shrinkage
        
        # Apply shrinkage
        alpha = adaptive_shrinkage(R, K_est) if getattr(cfg, "MUSIC_SHRINK", None) is None else 0.10
        trace_n = np.real(np.trace(R)) / R.shape[0]
        R_shrunk = (1.0 - alpha) * R + alpha * trace_n * np.eye(R.shape[0], dtype=R.dtype)
        
        # EVD → noise projector G
        evals, evecs = np.linalg.eigh(R_shrunk)
        # Sort descending → noise subspace = smallest (N-K) eigenvectors
        idx_sort = np.argsort(evals)[::-1]  # Descending
        evecs_sorted = evecs[:, idx_sort]
        U_n = evecs_sorted[:, K_est:]  # Noise subspace [N, N-K]
        G = U_n @ U_n.conj().T  # Noise projector [N, N]
        
        # Run range MUSIC for each source
        r_refined = range_pipeline(phi_deg, theta_deg, R, G, cfg, r_prior_array=r_init)
        info["r_refined"] = r_refined
        info["range_method"] = "1D_NF_MUSIC"
    else:
        # Fall back to prior or zero
        r_refined = r_init if r_init is not None else np.zeros(K_est, dtype=np.float32)
        info["r_refined"] = r_refined
        info["range_method"] = "prior"
    
    # Step 4: Joint {φ,θ,r} Newton refinement (CRITICAL: sub-degree polish!)
    if getattr(cfg, "USE_JOINT_NEWTON", True):
        # 1-3 iterations of joint refinement (already close from MUSIC)
        phi_final, theta_final, r_final = joint_newton_refine(
            phi_deg, theta_deg, r_refined, info["R"], cfg,
            max_iters=getattr(cfg, "JOINT_NEWTON_ITERS", 2),
            step_size=getattr(cfg, "JOINT_NEWTON_STEP", 0.3),
        )
        info["joint_newton_applied"] = True
    else:
        phi_final, theta_final, r_final = phi_deg, theta_deg, r_refined
        info["joint_newton_applied"] = False
    
    # Step 5: Short NF-MLE polish (CRITICAL: true low-SNR robustness!)
    if getattr(cfg, "USE_NF_MLE_POLISH", True) and len(phi_final) > 0:
        # Check if we have snapshots for MLE polish
        if "Y_snapshots" in info and info["Y_snapshots"] is not None:
            Y = info["Y_snapshots"]  # [M, L] snapshots
            x_coords, y_coords = info.get("sensor_coords", (None, None))
            
            if x_coords is not None and y_coords is not None:
                # Extract geometry for MLE
                N_H = int(getattr(cfg, "N_H", 12))
                N_V = int(getattr(cfg, "N_V", 12))
                d_h = float(getattr(cfg, "d_H", 0.5)) * float(getattr(cfg, "WAVEL", 0.0625))
                d_v = float(getattr(cfg, "d_V", 0.5)) * float(getattr(cfg, "WAVEL", 0.0625))
                lam = float(getattr(cfg, "WAVEL", 0.0625))
                k = 2.0 * np.pi / lam
                
                # Estimate SNR for gating
                snr_est = info.get("snr_est", 10.0)  # Default to high SNR
                
                # Apply NF-MLE polish
                phi_final, theta_final, r_final = nf_mle_polish_after_newton(
                    phi_final, theta_final, r_final,
                    Y, x_coords, y_coords, k, cfg, snr_est
                )
                info["nf_mle_applied"] = True
            else:
                info["nf_mle_applied"] = False
        else:
            info["nf_mle_applied"] = False
    else:
        info["nf_mle_applied"] = False
    
    return phi_final, theta_final, r_final, info


# Convenience wrapper for batch processing
def angle_pipeline_batch(cov_factors, K_estimates, cfg, **kwargs):
    """
    Batch version of angle_pipeline.
    
    Args:
        cov_factors: [B, N, K_MAX] or [B, N, N]
        K_estimates: [B] estimated K values
        cfg: Config
        **kwargs: Passed to angle_pipeline
    
    Returns:
        phi_batch [B, K_MAX]: Azimuth (degrees), padded with NaN
        theta_batch [B, K_MAX]: Elevation (degrees), padded with NaN
    """
    B = cov_factors.shape[0]
    K_MAX = getattr(cfg, "K_MAX", 5)
    
    phi_batch = np.full((B, K_MAX), np.nan, dtype=np.float32)
    theta_batch = np.full((B, K_MAX), np.nan, dtype=np.float32)
    
    for b in range(B):
        K_est = int(K_estimates[b])
        phi, theta, _ = angle_pipeline(cov_factors[b], K_est, cfg, **kwargs)
        phi_batch[b, :len(phi)] = phi
        theta_batch[b, :len(theta)] = theta
    
    return phi_batch, theta_batch


def range_music_1d(phi_deg, theta_deg, R, G, cfg, r_prior=None, 
                   n_steps=121, span=0.25, refine=True):
    """
    1-D near-field range MUSIC for a single source at known angles.
    
    PRODUCTION FEATURE: Cuts range error ~2× vs. direct regression.
    
    Args:
        phi_deg: Azimuth angle (degrees)
        theta_deg: Elevation angle (degrees)
        R: Covariance matrix [N, N] complex (can be None if G provided)
        G: Noise projector U_n U_n^H [N, N] complex (reuse from angle MUSIC)
        cfg: Config object
        r_prior: Prior range estimate (meters, optional)
        n_steps: Grid steps (default: 121 for ~0.02-0.05m spacing)
        span: Prior window span (default: 0.25 = ±25%)
        refine: Apply 1-D parabolic refinement (default: True)
    
    Returns:
        r_refined: Refined range (meters)
        spectrum: MUSIC spectrum [n_steps] for diagnostics
    """
    # Extract geometry
    N_H = int(getattr(cfg, "N_H", 12))
    N_V = int(getattr(cfg, "N_V", 12))
    d_h = float(getattr(cfg, "d_H", 0.5))
    d_v = float(getattr(cfg, "d_V", 0.5))
    lam = float(getattr(cfg, "WAVEL", 0.0625))
    
    # Range limits
    r_min_global = float(getattr(cfg, "R_MIN", 0.5))
    r_max_global = float(getattr(cfg, "R_MAX", 10.0))
    
    # Build range grid
    if r_prior is not None and r_min_global < r_prior < r_max_global:
        # Focused window around prior
        r_min = max(r_min_global, r_prior * (1 - span))
        r_max = min(r_max_global, r_prior * (1 + span))
    else:
        # Global search
        r_min, r_max = r_min_global, r_max_global
    
    r_grid = np.linspace(r_min, r_max, n_steps, dtype=np.float32)
    
    # Convert angles to radians
    phi_rad = np.deg2rad(phi_deg)
    theta_rad = np.deg2rad(theta_deg)
    
    # Compute MUSIC spectrum P(r) = 1 / (a^H G a)
    # Use near-field steering: a(φ,θ,r)
    spectrum = np.zeros(n_steps, dtype=np.float32)
    
    for i, r in enumerate(r_grid):
        # Near-field steering vector
        a = _nearfield_steer_single(phi_rad, theta_rad, r, N_H, N_V, d_h, d_v, lam)
        
        # MUSIC cost
        denom = np.real(a.conj().T @ G @ a)
        denom = max(denom, 1e-8)  # Avoid divide-by-zero
        spectrum[i] = 1.0 / denom
    
    # Peak picking
    idx_peak = np.argmax(spectrum)
    r_coarse = r_grid[idx_peak]
    
    # Parabolic refinement
    if refine and 0 < idx_peak < n_steps - 1:
        # Fit parabola to peak and neighbors
        y0, y1, y2 = spectrum[idx_peak-1], spectrum[idx_peak], spectrum[idx_peak+1]
        x0, x1, x2 = r_grid[idx_peak-1], r_grid[idx_peak], r_grid[idx_peak+1]
        
        # Parabolic interpolation
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(denom) > 1e-9:
            A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
            B = (x2*x2 * (y0 - y1) + x1*x1 * (y2 - y0) + x0*x0 * (y1 - y2)) / denom
            
            if abs(A) > 1e-9:
                r_refined = -B / (2 * A)
                # Clamp to local window
                r_refined = np.clip(r_refined, x0, x2)
            else:
                r_refined = r_coarse
        else:
            r_refined = r_coarse
    else:
        r_refined = r_coarse
    
    # Global clamp
    r_refined = np.clip(r_refined, r_min_global, r_max_global)
    
    return float(r_refined), spectrum


def _nearfield_steer_single(phi_rad, theta_rad, r, N_H, N_V, d_h, d_v, lam):
    """
    Near-field steering vector for a single source at (φ, θ, r).
    
    Accounts for per-element distance ρ_mn = ||p_mn - s||
    where s = [r sinφ cosθ, r cosφ cosθ, r sinθ]
    
    Args:
        phi_rad, theta_rad: Angles (radians)
        r: Range (meters)
        N_H, N_V: Array dimensions
        d_h, d_v: Element spacing (meters)
        lam: Wavelength (meters)
    
    Returns:
        a: Steering vector [N=N_H*N_V] (complex64)
    """
    k = 2 * np.pi / lam
    
    # Source position (Cartesian)
    s_x = r * np.sin(phi_rad) * np.cos(theta_rad)
    s_y = r * np.cos(phi_rad) * np.cos(theta_rad)
    s_z = r * np.sin(theta_rad)
    
    # Array element positions (assume centered at origin)
    # Horizontal: m ∈ [0, N_H), spacing d_h
    # Vertical: n ∈ [0, N_V), spacing d_v
    m_idx = np.arange(N_H)
    n_idx = np.arange(N_V)
    
    # Center the array
    p_m = (m_idx - (N_H - 1) / 2.0) * d_h  # X positions
    p_n = (n_idx - (N_V - 1) / 2.0) * d_v  # Z positions
    
    # Mesh grid
    P_m, P_n = np.meshgrid(p_m, p_n, indexing='ij')  # [N_H, N_V]
    
    # Element positions
    elem_x = P_m.flatten()  # [N]
    elem_z = P_n.flatten()  # [N]
    elem_y = np.zeros_like(elem_x)  # Y = 0 (planar array in XZ plane)
    
    # Distance from source to each element
    dx = elem_x - s_x
    dy = elem_y - s_y
    dz = elem_z - s_z
    rho = np.sqrt(dx**2 + dy**2 + dz**2)  # [N]
    
    # Phase shift: exp(-j k ρ)
    a = np.exp(-1j * k * rho).astype(np.complex64)
    
    return a


def range_pipeline(phi_deg_array, theta_deg_array, R, G, cfg, r_prior_array=None):
    """
    Batch 1-D range MUSIC for multiple sources.
    
    Args:
        phi_deg_array: Azimuth angles [K] (degrees)
        theta_deg_array: Elevation angles [K] (degrees)
        R: Covariance matrix [N, N] complex
        G: Noise projector [N, N] complex (reuse from angle MUSIC)
        cfg: Config object
        r_prior_array: Prior range estimates [K] (meters, optional)
    
    Returns:
        r_refined_array: Refined ranges [K] (meters)
    """
    K = len(phi_deg_array)
    r_refined_array = np.zeros(K, dtype=np.float32)
    
    for k in range(K):
        phi = phi_deg_array[k]
        theta = theta_deg_array[k]
        r_prior = r_prior_array[k] if r_prior_array is not None else None
        
        r_refined, _ = range_music_1d(
            phi, theta, R, G, cfg,
            r_prior=r_prior,
            n_steps=getattr(cfg, "RANGE_GRID_STEPS", 121),
            span=getattr(cfg, "RANGE_PRIOR_SPAN", 0.25),
            refine=getattr(cfg, "RANGE_MUSIC_REFINE", True),
        )
        r_refined_array[k] = r_refined
    
    return r_refined_array


def joint_newton_refine(phi_init, theta_init, r_init, R, cfg, 
                        max_iters=3, step_size=0.3, min_sep_deg=0.5):
    """
    Joint Newton refinement of {φ, θ, r} simultaneously using near-field MUSIC cost.
    
    CRITICAL FIX: Refine all 3 parameters together after range MUSIC for sub-degree accuracy.
    This is a 1-3 step polish, not a full optimization (angles/ranges already good from MUSIC).
    
    Args:
        phi_init: Initial azimuth angles [K] (degrees)
        theta_init: Initial elevation angles [K] (degrees)
        r_init: Initial ranges [K] (meters)
        R: Covariance matrix [N, N] complex
        cfg: Config object
        max_iters: Maximum iterations (default: 3, typically 1-3 is enough)
        step_size: Step size scaling (default: 0.3, conservative for joint)
        min_sep_deg: Minimum separation between sources (degrees)
    
    Returns:
        phi_refined [K]: Refined azimuth (degrees)
        theta_refined [K]: Refined elevation (degrees)
        r_refined [K]: Refined ranges (meters)
    """
    # Extract geometry
    N_H = int(getattr(cfg, "N_H", 12))
    N_V = int(getattr(cfg, "N_V", 12))
    d_h = float(getattr(cfg, "d_H", 0.5)) * float(getattr(cfg, "WAVEL", 0.0625))  # Convert to meters
    d_v = float(getattr(cfg, "d_V", 0.5)) * float(getattr(cfg, "WAVEL", 0.0625))
    lam = float(getattr(cfg, "WAVEL", 0.0625))
    
    # Limits
    phi_max = float(getattr(cfg, "ANGLE_RANGE_PHI", np.pi/3))  # Assume radians in config
    theta_max = float(getattr(cfg, "ANGLE_RANGE_THETA", np.pi/6))
    r_min = float(getattr(cfg, "R_MIN", 0.5))
    r_max = float(getattr(cfg, "R_MAX", 10.0))
    
    K = len(phi_init)
    if K == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert to radians and copy
    phi = np.deg2rad(phi_init.copy())
    theta = np.deg2rad(theta_init.copy())
    r = r_init.copy()
    
    # CRITICAL: Diagonal loading for numerical stability
    eps_diag = 1e-3 * np.real(np.trace(R)) / R.shape[0]
    R_stable = R + eps_diag * np.eye(R.shape[0], dtype=R.dtype)
    
    # Eigendecomposition for noise projector
    evals, evecs = np.linalg.eigh(R_stable)
    U_noise = evecs[:, :max(1, R_stable.shape[0] - K)]
    G = U_noise @ U_noise.conj().T
    
    # Joint Newton iterations
    for it in range(max_iters):
        # Compute cost and gradients for each source
        for k in range(K):
            # Near-field steering vector and its derivatives
            a = _nearfield_steer_single(phi[k], theta[k], r[k], N_H, N_V, d_h, d_v, lam)
            
            # Finite-difference gradients (small perturbations)
            eps_phi = 0.001  # ~0.057°
            eps_theta = 0.001
            eps_r = 0.01  # 1cm
            
            a_phi_p = _nearfield_steer_single(phi[k] + eps_phi, theta[k], r[k], N_H, N_V, d_h, d_v, lam)
            a_phi_m = _nearfield_steer_single(phi[k] - eps_phi, theta[k], r[k], N_H, N_V, d_h, d_v, lam)
            da_dphi = (a_phi_p - a_phi_m) / (2 * eps_phi)
            
            a_theta_p = _nearfield_steer_single(phi[k], theta[k] + eps_theta, r[k], N_H, N_V, d_h, d_v, lam)
            a_theta_m = _nearfield_steer_single(phi[k], theta[k] - eps_theta, r[k], N_H, N_V, d_h, d_v, lam)
            da_dtheta = (a_theta_p - a_theta_m) / (2 * eps_theta)
            
            a_r_p = _nearfield_steer_single(phi[k], theta[k], r[k] + eps_r, N_H, N_V, d_h, d_v, lam)
            a_r_m = _nearfield_steer_single(phi[k], theta[k], r[k] - eps_r, N_H, N_V, d_h, d_v, lam)
            da_dr = (a_r_p - a_r_m) / (2 * eps_r)
            
            # MUSIC cost: J = a^H G a (want to minimize)
            Ga = G @ a
            cost = np.real(a.conj().T @ Ga)
            
            # Gradient: ∂J/∂x = 2·Re(da/dx^H G a)
            grad_phi = 2 * np.real(da_dphi.conj().T @ Ga)
            grad_theta = 2 * np.real(da_dtheta.conj().T @ Ga)
            grad_r = 2 * np.real(da_dr.conj().T @ Ga)
            
            # Hessian approximation (Gauss-Newton: use gradient outer product)
            # H ≈ 2·Re(da/dx^H G da/dx)
            Gda_phi = G @ da_dphi
            Gda_theta = G @ da_dtheta
            Gda_dr = G @ da_dr
            
            H_pp = 2 * np.real(da_dphi.conj().T @ Gda_phi)
            H_tt = 2 * np.real(da_dtheta.conj().T @ Gda_theta)
            H_rr = 2 * np.real(da_dr.conj().T @ Gda_dr)
            H_pt = 2 * np.real(da_dphi.conj().T @ Gda_theta)
            H_pr = 2 * np.real(da_dphi.conj().T @ Gda_dr)
            H_tr = 2 * np.real(da_dtheta.conj().T @ Gda_dr)
            
            # Build 3x3 Hessian
            H = np.array([
                [H_pp, H_pt, H_pr],
                [H_pt, H_tt, H_tr],
                [H_pr, H_tr, H_rr]
            ], dtype=np.float64)
            
            g = np.array([grad_phi, grad_theta, grad_r], dtype=np.float64)
            
            # Solve H·Δ = -g with regularization
            H_reg = H + 1e-6 * np.eye(3)
            try:
                delta = np.linalg.solve(H_reg, -g)
            except np.linalg.LinAlgError:
                delta = np.zeros(3)
            
            # CRITICAL: Guardrails - clamp step magnitude to prevent divergence
            delta_mag = np.linalg.norm(delta)
            max_delta = 0.1  # Max step: ~5.7° for angles, ~0.1m for range
            if delta_mag > max_delta:
                delta = delta * (max_delta / delta_mag)  # Clamp to max step
            
            # Early stop if delta explodes (sign of bad Hessian)
            if delta_mag > 1.0:  # > 57° or > 1m is clearly wrong
                break  # Stop this iteration, keep current estimates
            
            # Update with step size
            phi[k] += step_size * delta[0]
            theta[k] += step_size * delta[1]
            r[k] += step_size * delta[2]
            
            # Clamp
            phi[k] = np.clip(phi[k], -phi_max, phi_max)
            theta[k] = np.clip(theta[k], -theta_max, theta_max)
            r[k] = np.clip(r[k], r_min, r_max)
        
        # Enforce minimum separation (simple repulsion)
        for i in range(K):
            for j in range(i + 1, K):
                dphi = abs(phi[i] - phi[j])
                dtheta = abs(theta[i] - theta[j])
                sep_deg = np.rad2deg(np.sqrt(dphi**2 + dtheta**2))
                
                if sep_deg < min_sep_deg:
                    # Push apart slightly
                    phi[i] += 0.5 * np.sign(phi[i] - phi[j]) * np.deg2rad(0.1)
                    phi[j] -= 0.5 * np.sign(phi[i] - phi[j]) * np.deg2rad(0.1)
    
    # Convert back to degrees
    phi_refined = np.rad2deg(phi)
    theta_refined = np.rad2deg(theta)
    
    return phi_refined, theta_refined, r

