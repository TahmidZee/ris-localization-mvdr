"""
Hungarian-matching angle and range evaluation
Provides sub-degree evaluation metrics for DOA estimation
"""
import numpy as np


def _angle_cost_matrix(phi_pred, theta_pred, phi_gt, theta_gt, r_pred=None, r_gt=None, adaptive_scales=False):
    """
    Compute cost matrix for Hungarian assignment with normalized units.
    
    FOV-CRITICAL FIX: For bounded FOV (φ∈[-60°,+60°], θ∈[-30°,+30°]),
    use PLAIN absolute difference (no wrap-to-180).
    Wrap-to-180 creates artificial 90° floor when predictions escape FOV.
    
    Cost = sqrt((Δφ/σ_φ)² + (Δθ/σ_θ)² + (Δr/σ_r)²)
    
    Args:
        phi_pred: Predicted azimuth angles (degrees), shape [P]
        theta_pred: Predicted elevation angles (degrees), shape [P]
        phi_gt: Ground truth azimuth angles (degrees), shape [G]
        theta_gt: Ground truth elevation angles (degrees), shape [G]
        r_pred: Optional predicted ranges (meters), shape [P]
        r_gt: Optional ground truth ranges (meters), shape [G]
        adaptive_scales: If True, compute scales from data (default: False for stability)
    
    Returns:
        Cost matrix [max(P,G), max(P,G)] with normalized distances
    """
    P, G = len(phi_pred), len(phi_gt)
    C = np.zeros((max(P, G), max(P, G)), dtype=np.float64)
    C[:] = 999.0  # Large cost for unmatched pairs
    
    # Normalization scales (representative target errors)
    if adaptive_scales and P > 0 and G > 0:
        # Adaptive: use data quantiles (more robust across datasets)
        # Compute all pairwise errors (plain absolute difference, no wrap!)
        all_dphi = np.abs(phi_pred[:, None] - phi_gt[None, :]).flatten()
        all_dtheta = np.abs(theta_pred[:, None] - theta_gt[None, :]).flatten()
        
        sigma_phi = max(np.quantile(all_dphi, 0.75), 1.0)    # 75th percentile, min 1°
        sigma_theta = max(np.quantile(all_dtheta, 0.75), 1.0)
        
        if r_pred is not None and r_gt is not None:
            all_dr = np.abs(r_pred[:, None] - r_gt[None, :]).flatten()
            sigma_r = max(np.quantile(all_dr, 0.75), 0.5)  # min 0.5m
        else:
            sigma_r = 1.0
    else:
        # Fixed scales (conservative defaults, work across most datasets)
        sigma_phi = 5.0    # degrees (target: sub-5° → cost=1 at 5°)
        sigma_theta = 5.0  # degrees
        sigma_r = 1.0      # meters (target: sub-1m → cost=1 at 1m)
    
    for i in range(P):
        for j in range(G):
            # FOV-CRITICAL: Plain absolute difference (no wrap!) for bounded FOV
            dphi = np.abs(phi_pred[i] - phi_gt[j]) / sigma_phi
            dth = np.abs(theta_pred[i] - theta_gt[j]) / sigma_theta
            
            if r_pred is not None and r_gt is not None:
                dr = np.abs(r_pred[i] - r_gt[j]) / sigma_r
                C[i, j] = (dphi**2 + dth**2 + dr**2) ** 0.5
            else:
                C[i, j] = (dphi**2 + dth**2) ** 0.5
    
    return C


def _hungarian_assign(C):
    """
    Hungarian algorithm for optimal assignment
    
    Args:
        C: Cost matrix [N, N]
    
    Returns:
        row_indices, col_indices: Matched pairs
    """
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(C)
        return r, c
    except Exception:
        # Greedy fallback if scipy not available
        import numpy as np
        used_r, used_c, pairs = set(), set(), []
        H, W = C.shape
        for _ in range(min(H, W)):
            i, j = divmod(C.argmin(), W)
            while (i in used_r) or (j in used_c):
                C[i, j] = 1e9
                i, j = divmod(C.argmin(), W)
            used_r.add(i)
            used_c.add(j)
            pairs.append((i, j))
            C[i, :] = 1e9
            C[:, j] = 1e9
        return (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]))


def eval_scene_angles_ranges(phi_pred, theta_pred, r_pred, phi_gt, theta_gt, r_gt,
                              success_tol_phi=5.0, success_tol_theta=5.0, success_tol_r=1.0):
    """
    Evaluate angle and range estimation using Hungarian matching.
    
    CRITICAL FIX: Returns BOTH raw errors (no gating) and success-gated metrics.
    This allows tracking actual MUSIC performance even when success rate is 0.
    
    Args:
        phi_pred: Predicted azimuth angles (degrees), shape [K_pred]
        theta_pred: Predicted elevation angles (degrees), shape [K_pred]
        r_pred: Predicted ranges (meters), shape [K_pred]
        phi_gt: Ground truth azimuth angles (degrees), shape [K_gt]
        theta_gt: Ground truth elevation angles (degrees), shape [K_gt]
        r_gt: Ground truth ranges (meters), shape [K_gt]
        success_tol_phi: Tolerance for "success" (degrees)
        success_tol_theta: Tolerance for "success" (degrees)
        success_tol_r: Tolerance for "success" (meters)
    
    Returns:
        dict with:
        - RAW metrics (med_phi, rmse_phi, etc): computed on ALL matched pairs, no gating
        - SUCCESS metrics (success_flag): whether ALL sources are within tolerance
        - num_matched: number of Hungarian-matched pairs
    """
    Kp, Kg = len(phi_pred), len(phi_gt)
    
    if Kp == 0 or Kg == 0:
        return {
            "rmse_phi": None, "rmse_theta": None, "rmse_r": None,
            "med_phi": None, "med_theta": None, "med_r": None,
            "num_pred": Kp, "num_gt": Kg, "num_matched": 0,
            "success_flag": False,
            # Raw errors (empty)
            "raw_phi_errors": [], "raw_theta_errors": [], "raw_r_errors": [],
        }
    
    # Hungarian assignment based on normalized cost (includes range!)
    C = _angle_cost_matrix(phi_pred, theta_pred, phi_gt, theta_gt, r_pred, r_gt).astype(np.float64)
    r_idx, c_idx = _hungarian_assign(C.copy())
    
    m = min(Kp, Kg)
    if m == 0:
        return {
            "rmse_phi": None, "rmse_theta": None, "rmse_r": None,
            "med_phi": None, "med_theta": None, "med_r": None,
            "num_pred": Kp, "num_gt": Kg, "num_matched": 0,
            "success_flag": False,
            "raw_phi_errors": [], "raw_theta_errors": [], "raw_r_errors": [],
        }
    
    # Compute RAW errors for ALL matched pairs (no gating!)
    dphi_raw = []
    dth_raw = []
    dr_raw = []
    all_within_tol = True  # For success flag
    
    for i in range(m):
        if r_idx[i] < len(phi_pred) and c_idx[i] < len(phi_gt):
            # Plain absolute difference (no wrap!) for bounded FOV
            err_phi = abs(phi_pred[r_idx[i]] - phi_gt[c_idx[i]])
            err_theta = abs(theta_pred[r_idx[i]] - theta_gt[c_idx[i]])
            err_r = abs(r_pred[r_idx[i]] - r_gt[c_idx[i]])
            
            # Store RAW errors (no gating)
            dphi_raw.append(err_phi)
            dth_raw.append(err_theta)
            dr_raw.append(err_r)
            
            # Check success tolerance
            if err_phi > success_tol_phi or err_theta > success_tol_theta or err_r > success_tol_r:
                all_within_tol = False
    
    if len(dphi_raw) == 0:
        return {
            "rmse_phi": None, "rmse_theta": None, "rmse_r": None,
            "med_phi": None, "med_theta": None, "med_r": None,
            "num_pred": Kp, "num_gt": Kg, "num_matched": 0,
            "success_flag": False,
            "raw_phi_errors": [], "raw_theta_errors": [], "raw_r_errors": [],
        }
    
    dphi_raw = np.array(dphi_raw)
    dth_raw = np.array(dth_raw)
    dr_raw = np.array(dr_raw)
    
    return {
        # RAW metrics (no gating, no penalties) - use these for tracking progress
        "rmse_phi": float(np.sqrt(np.mean(dphi_raw**2))),
        "rmse_theta": float(np.sqrt(np.mean(dth_raw**2))),
        "rmse_r": float(np.sqrt(np.mean(dr_raw**2))),
        "med_phi": float(np.median(dphi_raw)),
        "med_theta": float(np.median(dth_raw)),
        "med_r": float(np.median(dr_raw)),
        "num_pred": Kp,
        "num_gt": Kg,
        "num_matched": len(dphi_raw),
        # Success flag (strict: ALL sources within tolerance)
        "success_flag": all_within_tol and len(dphi_raw) == Kg,
        # Raw error arrays for aggregation
        "raw_phi_errors": dphi_raw.tolist(),
        "raw_theta_errors": dth_raw.tolist(),
        "raw_r_errors": dr_raw.tolist(),
    }


def eval_batch_angles_ranges(pred_batch, gt_batch):
    """
    Evaluate a batch of predictions
    
    Args:
        pred_batch: List of (phi, theta, r) tuples, each with shape [K_pred]
        gt_batch: List of (phi, theta, r) tuples, each with shape [K_gt]
    
    Returns:
        Aggregated metrics across batch
    """
    all_errors = []
    
    for (phi_p, theta_p, r_p), (phi_g, theta_g, r_g) in zip(pred_batch, gt_batch):
        errors = eval_scene_angles_ranges(phi_p, theta_p, r_p, phi_g, theta_g, r_g)
        if errors["rmse_phi"] is not None:
            all_errors.append(errors)
    
    if not all_errors:
        return {
            "rmse_phi": None, "rmse_theta": None, "rmse_r": None,
            "med_phi": None, "med_theta": None, "med_r": None,
        }
    
    # Aggregate across batch
    rmse_phi_all = [e["rmse_phi"] for e in all_errors]
    rmse_theta_all = [e["rmse_theta"] for e in all_errors]
    rmse_r_all = [e["rmse_r"] for e in all_errors]
    med_phi_all = [e["med_phi"] for e in all_errors]
    med_theta_all = [e["med_theta"] for e in all_errors]
    med_r_all = [e["med_r"] for e in all_errors]
    
    return {
        "rmse_phi_mean": float(np.mean(rmse_phi_all)),
        "rmse_phi_med": float(np.median(rmse_phi_all)),
        "rmse_theta_mean": float(np.mean(rmse_theta_all)),
        "rmse_theta_med": float(np.median(rmse_theta_all)),
        "rmse_r_mean": float(np.mean(rmse_r_all)),
        "rmse_r_med": float(np.median(rmse_r_all)),
        "med_phi_mean": float(np.mean(med_phi_all)),
        "med_phi_med": float(np.median(med_phi_all)),
        "med_theta_mean": float(np.mean(med_theta_all)),
        "med_theta_med": float(np.median(med_theta_all)),
        "med_r_mean": float(np.mean(med_r_all)),
        "med_r_med": float(np.median(med_r_all)),
        "num_scenes": len(all_errors),
    }


if __name__ == "__main__":
    # Test
    phi_pred = np.array([10.0, 20.0, 30.0])
    theta_pred = np.array([5.0, 10.0, 15.0])
    r_pred = np.array([5.0, 7.0, 9.0])
    
    phi_gt = np.array([10.5, 19.5, 29.5])
    theta_gt = np.array([5.5, 9.5, 14.5])
    r_gt = np.array([5.1, 7.1, 8.9])
    
    errors = eval_scene_angles_ranges(phi_pred, theta_pred, r_pred, phi_gt, theta_gt, r_gt)
    
    print("Test Results:")
    print(f"  RMSE φ:     {errors['rmse_phi']:.4f}°")
    print(f"  RMSE θ:     {errors['rmse_theta']:.4f}°")
    print(f"  RMSE r:     {errors['rmse_r']:.4f} m")
    print(f"  Median φ:   {errors['med_phi']:.4f}°")
    print(f"  Median θ:   {errors['med_theta']:.4f}°")
    print(f"  Median r:   {errors['med_r']:.4f} m")


# ========================================================================
# 2D MUSIC Coarse Scan from Learned Covariance (Phase 1 + Phase 2)
# ========================================================================

import torch

def _exchange_matrix(n):
    """Create exchange (flip) matrix for FBA"""
    J = np.zeros((n, n), dtype=np.float32)
    J[:, ::-1] = np.eye(n, dtype=np.float32)
    return torch.as_tensor(J)


def forward_backward_average(R, N_H, N_V):
    """
    Forward-Backward Averaging for UPA (Uniform Planar Array).
    Exploits centro-symmetry: R_fba = 0.5 * (R + J R* J)
    
    Args:
        R: Covariance matrix [N, N] complex (torch tensor)
        N_H, N_V: Array dimensions
    
    Returns:
        R_fba: Averaged covariance [N, N] complex
    """
    device = R.device
    # Exchange matrices for horizontal and vertical
    J_H = _exchange_matrix(N_H).to(device)
    J_V = _exchange_matrix(N_V).to(device)
    # Kronecker product: J = J_H ⊗ J_V
    J = torch.kron(J_H, J_V).to(dtype=torch.complex64, device=device)
    # FBA: average forward and backward views
    return 0.5 * (R + J @ R.conj() @ J)


def adaptive_shrinkage(R):
    """
    Adaptive Ledoit-Wolf style shrinkage coefficient.
    Estimates shrinkage from eigenvalue spread and off-diagonal variance.
    
    Args:
        R: Covariance matrix [N, N] complex (torch tensor)
    
    Returns:
        R_shrunk: Shrunk covariance [N, N] complex
        alpha: Shrinkage coefficient used (for diagnostics)
    """
    N = R.shape[0]
    device = R.device
    
    # Target: identity scaled by average eigenvalue
    trace_N = torch.real(torch.trace(R)) / N
    
    # Estimate variance of off-diagonal elements (proxy for noise)
    off_diag = R - torch.diag(torch.diagonal(R))
    var_off = torch.mean(torch.abs(off_diag) ** 2)
    
    # Adaptive alpha: high when off-diagonals are noisy
    # Clamp to [2%, 10%] - conservative for hybrid covariance (β=0.30)
    # With full-rank R_blend from L=16 snapshots, we need less shrinkage
    alpha = torch.clamp(var_off / (var_off + trace_N**2 + 1e-12), 0.02, 0.10)
    
    # Shrink toward scaled identity
    I = torch.eye(N, dtype=torch.complex64, device=device)
    R_shrunk = (1.0 - alpha) * R + alpha * trace_N * I
    
    return R_shrunk, float(alpha)


def snr_adaptive_shrinkage_loading(R, cfg):
    """
    SNR-adaptive shrinkage + loading for low-SNR rescue.
    
    Args:
        R: Covariance matrix [N, N] complex
        cfg: Config object
    
    Returns:
        R_processed: Processed covariance matrix
        alpha_used: Shrinkage parameter used
        eps_used: Diagonal loading used
        snr_est: Estimated SNR in dB
    """
    N = R.shape[0]
    
    # Per-scene noise estimation from smallest eigenvalues
    evals = torch.linalg.eigvalsh(R)
    evals = torch.real(evals)  # Should be real for Hermitian matrix
    
    # Estimate noise floor from smallest eigenvalues
    # Use robust average of smallest (N-K) eigenvalues
    K_est = getattr(cfg, 'K_EST_DEFAULT', 2)  # Conservative estimate
    noise_rank = max(1, N - K_est)
    hat_sigma_sq = torch.mean(evals[:noise_rank])
    
    # Estimate SNR from eigenvalue distribution
    signal_energy = torch.sum(evals[-K_est:]) if K_est > 0 else torch.sum(evals)
    noise_energy = torch.sum(evals[:noise_rank])
    snr_linear = signal_energy / max(noise_energy, 1e-12)
    snr_est = 10.0 * torch.log10(snr_linear)
    
    # SNR gating: g = sigmoid((SNR_thr - snr_est) / w)
    SNR_thr = getattr(cfg, 'SNR_THRESHOLD', 3.0)  # dB
    w = getattr(cfg, 'SNR_GATE_WIDTH', 3.0)  # dB
    g = torch.sigmoid((SNR_thr - snr_est) / w)
    g = torch.clamp(g, 0.0, 1.0)
    
    # Adaptive shrinkage: α = α_min + g * k_α * (σ̂ / λ_med)
    alpha_min = getattr(cfg, 'ALPHA_MIN', 0.02)
    alpha_max = getattr(cfg, 'ALPHA_MAX', 0.15)
    k_alpha = getattr(cfg, 'K_ALPHA', 0.3)
    
    lam_med = torch.median(evals)
    alpha = alpha_min + g * k_alpha * (torch.sqrt(hat_sigma_sq) / lam_med)
    alpha = torch.clamp(alpha, alpha_min, alpha_max)
    
    # Adaptive diagonal loading: ε = g * c_ε * σ̂²
    c_eps = getattr(cfg, 'C_EPS', 1.0)
    eps = g * c_eps * hat_sigma_sq
    
    # Apply shrinkage
    trace_N = torch.real(torch.trace(R)) / N
    R_shrunk = (1.0 - alpha) * R + alpha * trace_N * torch.eye(N, dtype=torch.complex64, device=R.device)
    
    # Apply diagonal loading
    R_processed = R_shrunk + eps * torch.eye(N, dtype=torch.complex64, device=R.device)
    
    return R_processed, float(alpha), float(eps), float(snr_est)


def parabolic_refine_2d(S, i, j, phi_grid, theta_grid):
    """
    2D parabolic interpolation around peak (i, j) for sub-grid accuracy.
    Uses 3×3 neighborhood to fit quadratic and find vertex.
    
    Args:
        S: MUSIC spectrum [H, W] float
        i, j: Peak coordinates (integers)
        phi_grid, theta_grid: Grid arrays (degrees) [H], [W]
    
    Returns:
        phi_refined, theta_refined: Sub-grid peak location (degrees)
    """
    H, W = S.shape
    # Clamp to interior (need 3×3 neighborhood)
    i = int(np.clip(i, 1, H - 2))
    j = int(np.clip(j, 1, W - 2))
    
    # Extract 3×3 neighborhood
    Z = S[i-1:i+2, j-1:j+2].astype(np.float64)
    
    # 1D parabolic fit: a*x^2 + b*x + c, vertex at -b/(2a)
    def fit_1d(v, grid):
        """Fit parabola to v=[-1, 0, 1] and return offset in grid units"""
        a = 0.5 * (v[0] + v[2] - 2*v[1])
        b = 0.5 * (v[2] - v[0])
        delta = -b / (2*a + 1e-12)  # Vertex offset
        delta = np.clip(delta, -0.75, 0.75)  # Limit extrapolation
        # Convert to physical units
        return float(delta * (grid[1] - grid[0]))
    
    # Refine along phi (rows) and theta (cols)
    dphi = fit_1d(Z[:, 1], phi_grid)  # Middle column
    dtheta = fit_1d(Z[1, :], theta_grid)  # Middle row
    
    phi_refined = float(phi_grid[i]) + dphi
    theta_refined = float(theta_grid[j]) + dtheta
    
    return phi_refined, theta_refined


def _planar_steering(phi_rad, theta_rad, N_H, N_V, d_h, d_v, lam):
    """
    Planar array steering vector at RIS (angles in radians).
    Uses far-field approximation for fast 2D grid search.
    
    CRITICAL FIX: This now matches EXACT dataset generation convention from physics.py.
    
    Args:
        phi_rad: Azimuth angle in radians
        theta_rad: Elevation angle in radians
        N_H, N_V: Array dimensions
        d_h, d_v: Element spacing in METERS
        lam: Wavelength (meters)
    
    Returns:
        Steering vector [N_H*N_V] complex64, unit-normalized
    """
    k = 2.0 * np.pi / lam
    
    # CRITICAL: Use EXACT same indexing as physics.py
    # physics.py: h_idx = np.arange(-(cfg.N_H - 1)//2, (cfg.N_H + 1)//2) * cfg.d_H
    h_idx = np.arange(-(N_H - 1)//2, (N_H + 1)//2) * d_h
    v_idx = np.arange(-(N_V - 1)//2, (N_V + 1)//2) * d_v
    
    # Create 2D grid with EXACT same indexing as physics.py
    h_mesh, v_mesh = np.meshgrid(h_idx, v_idx, indexing="xy")
    h_flat = h_mesh.reshape(-1).astype(np.float32)
    v_flat = v_mesh.reshape(-1).astype(np.float32)
    
    # CRITICAL: Use EXACT same phase formula as physics.py (planar term only)
    # physics.py: dist = r - vh*sin_phi*cos_theta - vv*sin_theta + (vh**2 + vv**2)/(2*r_eff)
    # For planar: ignore the (vh**2 + vv**2)/(2*r_eff) term
    sin_phi = np.sin(phi_rad)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    a = np.empty(N_H * N_V, np.complex64)
    
    for i, (vh, vv) in enumerate(zip(h_flat, v_flat)):
        # Planar phase term only (no near-field curvature)
        phase = k * (vh * sin_phi * cos_theta + vv * sin_theta)
        a[i] = np.exp(1j * phase)
    
    # CRITICAL: Use same normalization as physics.py
    a = a / np.sqrt(N_H * N_V)
    
    return a


def _nearfield_steering(phi_rad, theta_rad, r, N_H, N_V, d_h, d_v, lam):
    """
    Near-field array steering vector at RIS (angles + range in radians/meters).
    Includes spherical wavefront curvature term.
    
    CRITICAL FIX: This now matches EXACT dataset generation convention from physics.py.
    
    Args:
        phi_rad: Azimuth angle in radians
        theta_rad: Elevation angle in radians
        r: Range in meters
        N_H, N_V: Array dimensions
        d_h, d_v: Element spacing in METERS
        lam: Wavelength (meters)
    
    Returns:
        Steering vector [N_H*N_V] complex64, unit-normalized
    """
    k = 2.0 * np.pi / lam
    
    # CRITICAL: Use EXACT same indexing as physics.py
    # physics.py: h_idx = np.arange(-(cfg.N_H - 1)//2, (cfg.N_H + 1)//2) * cfg.d_H
    # NOTE: In this repo cfg.d_H/cfg.d_V are in meters (see configs.py / dataset.py)
    h_idx = np.arange(-(N_H - 1)//2, (N_H + 1)//2) * d_h
    v_idx = np.arange(-(N_V - 1)//2, (N_V + 1)//2) * d_v
    h_idx = np.arange(-(N_H - 1)//2, (N_H + 1)//2) * d_h
    v_idx = np.arange(-(N_V - 1)//2, (N_V + 1)//2) * d_v
    
    # Create 2D grid with EXACT same indexing as physics.py
    h_mesh, v_mesh = np.meshgrid(h_idx, v_idx, indexing="xy")
    h_flat = h_mesh.reshape(-1).astype(np.float32)
    v_flat = v_mesh.reshape(-1).astype(np.float32)
    
    # CRITICAL: Use EXACT same phase formula as physics.py
    # physics.py: dist = r - vh*sin_phi*cos_theta - vv*sin_theta + (vh**2 + vv**2)/(2*r_eff)
    sin_phi = np.sin(phi_rad)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    a = np.empty(N_H * N_V, np.complex64)
    r_eff = max(float(r), 1e-9)
    
    for i, (vh, vv) in enumerate(zip(h_flat, v_flat)):
        # EXACT same formula as physics.py
        dist = r - vh*sin_phi*cos_theta - vv*sin_theta + (vh**2 + vv**2)/(2*r_eff)
        a[i] = np.exp(1j * k * (r - dist))
    
    # CRITICAL: Use same normalization as physics.py
    a = a / np.sqrt(N_H * N_V)
    
    return a


def _chunked_music_spectrum_2_5d(G, phi_grid, theta_grid, r_planes, geom, chunk=4096, device="cpu", soft_params=None):
    """
    2.5D MUSIC spectrum: P(φ,θ) = max_r P(φ,θ,r) with range-aware near-field steering
    
    Args:
        G: Noise projector U_noise @ U_noise^H, shape [N, N], complex64
        phi_grid, theta_grid: 1D arrays in radians
        r_planes: Range planes to test [r1, r2, r3] in meters
        geom: Tuple (N_H, N_V, d_h, d_v, lam)
        chunk: Batch size for steering vector computation
        device: 'cpu' or 'cuda'
    
    Returns:
        Tuple: (spectrum_max, winning_ranges)
        - spectrum_max: Max spectrum over ranges [len(phi_grid), len(theta_grid)]
        - winning_ranges: Best range per angle [len(phi_grid), len(theta_grid)]
    """
    N_H, N_V, d_h, d_v, lam = geom
    Nphi, Ntheta = len(phi_grid), len(theta_grid)
    N = N_H * N_V
    
    # Initialize outputs
    S_max = np.zeros((Nphi, Ntheta), dtype=np.float32)
    winning_ranges = np.zeros((Nphi, Ntheta), dtype=np.float32)
    
    # Move G to torch once for efficient GPU computation
    Gt = torch.as_tensor(G, dtype=torch.complex64, device=device)
    
    # Iterate over theta (outer loop), then chunk phi (inner loop)
    for t_idx in range(Ntheta):
        theta = theta_grid[t_idx]
        
        # Process phi in chunks for memory efficiency
        start = 0
        while start < Nphi:
            end = min(start + chunk, Nphi)
            phis = phi_grid[start:end]
            
            # CRITICAL: 2.5D range-aware search
            # For each (phi, theta), test all range planes and take max
            best_spectrum = np.zeros(len(phis), dtype=np.float32)
            best_ranges = np.zeros(len(phis), dtype=np.float32)
            
            for r in r_planes:
                # Build steering matrix A for this range: [batch, N]
                A = np.stack([
                    _nearfield_steering(phi, theta, r, N_H, N_V, d_h, d_v, lam)
                    for phi in phis
                ], axis=0)
                At = torch.as_tensor(A, dtype=torch.complex64, device=device)
                
                # Compute MUSIC denominator with soft projector support
                if soft_params and soft_params.get('use_soft', False):
                    # Soft projector: P^{-1} = Σ_i w_i |u_i^H a|^2
                    # where w_i = 1 / (λ_i + τ)^p
                    evals = soft_params['evals']
                    evecs = soft_params['evecs']
                    p = soft_params['p']
                    tau = soft_params['tau']
                    
                    # Compute soft weights
                    weights = 1.0 / ((evals + tau) ** p)
                    weights = weights.to(device)
                    
                    # Compute soft MUSIC spectrum
                    denom_soft = torch.zeros(len(phis), device=device, dtype=torch.float32)
                    for i in range(len(phis)):
                        a_i = At[i]  # [N]
                        # P^{-1} = Σ_j w_j |u_j^H a|^2
                        for j in range(len(evals)):
                            u_j = evecs[:, j]  # [N]
                            proj = torch.abs(torch.dot(u_j.conj(), a_i)) ** 2
                            denom_soft[i] += weights[j] * proj
                    
                    denom_soft = torch.clamp(denom_soft, min=1e-9)
                    P_r = (1.0 / denom_soft).float().cpu().numpy()
                else:
                    # Hard projector: standard MUSIC
                    GA_H = torch.matmul(Gt, At.conj().T)  # [N, batch]
                    denom = torch.real((At * GA_H.T.conj()).sum(dim=1))  # [batch]
                    denom = torch.clamp(denom, min=1e-9)
                    P_r = (1.0 / denom).float().cpu().numpy()
                
                # Keep best spectrum and corresponding range
                mask = P_r > best_spectrum
                best_spectrum[mask] = P_r[mask]
                best_ranges[mask] = r
            
            # Store results
            S_max[start:end, t_idx] = best_spectrum
            winning_ranges[start:end, t_idx] = best_ranges
            start = end
    
    return S_max, winning_ranges


def _chunked_music_spectrum(G, phi_grid, theta_grid, geom, chunk=4096, device="cpu"):
    """
    Vectorized 2D MUSIC spectrum: P(φ,θ) = 1 / (a^H G a)
    
    Args:
        G: Noise projector U_noise @ U_noise^H, shape [N, N], complex64
        phi_grid, theta_grid: 1D arrays in radians
        geom: Tuple (N_H, N_V, d_h, d_v, lam)
        chunk: Batch size for steering vector computation
        device: 'cpu' or 'cuda'
    
    Returns:
        Spectrum S with shape [len(phi_grid), len(theta_grid)] (float32)
    """
    N_H, N_V, d_h, d_v, lam = geom
    Nphi, Ntheta = len(phi_grid), len(theta_grid)
    N = N_H * N_V
    
    S = np.zeros((Nphi, Ntheta), dtype=np.float32)
    
    # Move G to torch once for efficient GPU computation
    Gt = torch.as_tensor(G, dtype=torch.complex64, device=device)
    
    # Iterate over theta (outer loop), then chunk phi (inner loop)
    for t_idx in range(Ntheta):
        theta = theta_grid[t_idx]
        
        # Process phi in chunks for memory efficiency
        start = 0
        while start < Nphi:
            end = min(start + chunk, Nphi)
            phis = phi_grid[start:end]
            
            # Build steering matrix A: [batch, N]
            # NOTE:
            # - This is the *2D* MUSIC path and is intentionally range-agnostic.
            # - For near-field multi-source localization, use the 2.5D path
            #   (_chunked_music_spectrum_2_5d), which searches over range planes.
            #
            # Therefore, in 2D mode we use a planar (far-field) steering model.
            A = np.stack([
                _planar_steering(phi, theta, N_H, N_V, d_h, d_v, lam)
                for phi in phis
            ], axis=0)
            At = torch.as_tensor(A, dtype=torch.complex64, device=device)
            
            # Compute MUSIC denominator: real(diag(A @ G @ A^H))
            # Efficiently: real(sum(A * (G @ A^H)^T, dim=1))
            GA_H = torch.matmul(Gt, At.conj().T)  # [N, batch]
            denom = torch.real((At * GA_H.T.conj()).sum(dim=1))  # [batch]
            denom = torch.clamp(denom, min=1e-9)
            
            S[start:end, t_idx] = (1.0 / denom).float().cpu().numpy()
            start = end
    
    return S


def music2d_from_cov_factor(cf_ang, K, cfg, *, shrink=None, grid_phi=181, grid_theta=121, 
                            topk=None, device="cpu", use_fba=True, peak_refine=True, 
                            use_2_5d=True, r_planes=None):
    """
    2D/2.5D MUSIC coarse scan from learned covariance factor (Phase 1 + Phase 2 enhanced).
    
    Phase 2 Enhancements:
      - Forward-Backward Averaging (FBA) for sharper peaks
      - Adaptive shrinkage (Ledoit-Wolf style) instead of fixed
      - Parabolic sub-grid refinement for sub-degree accuracy
      - 2.5D range-aware near-field steering for multi-source scenarios
    
    Args:
        cf_ang: Covariance factor [N, K_MAX] (such that R = cf_ang @ cf_ang^H)
        K: Number of sources (int)
        cfg: Config object with geometry (N_H, N_V, d_H, d_V, WAVEL, ANGLE_RANGE_*)
        shrink: Fixed shrinkage parameter (default: None = adaptive)
        grid_phi: Azimuth grid resolution (default: 181 for ~0.33° spacing)
        grid_theta: Elevation grid resolution (default: 121 for ~0.33° spacing)
        topk: Number of peaks to return (default: K)
        device: 'cpu' or 'cuda'
        use_fba: Enable Forward-Backward Averaging (default: True)
        peak_refine: Enable parabolic refinement (default: True)
        use_2_5d: Enable 2.5D range-aware near-field steering (default: True)
        r_planes: Range planes for 2.5D [r1, r2, r3] in meters (default: [0.7, 2.5, 7.5])
    
    Returns:
        Tuple: (peaks_phi_deg, peaks_theta_deg, [peaks_r_m], spectrum, phi_grid_deg, theta_grid_deg)
        - If use_2_5d=True: includes peaks_r_m (ranges in meters)
        - If use_2_5d=False: standard 2D return format
    """
    topk = topk or int(K)
    
    # Reconstruct covariance from factor
    if isinstance(cf_ang, torch.Tensor):
        cf_ang = cf_ang.cpu().numpy()
    R = cf_ang @ cf_ang.conj().T  # [N, N]
    N = R.shape[0]
    
    # ICC CRITICAL FIX: Trace-normalize R before any processing
    # Without this, MUSIC gets a sick covariance and φ stays stuck at ~28°
    trace_R = np.real(np.trace(R))
    if trace_R > 1e-8:  # Avoid division by zero
        R = R * (N / trace_R)
    
    # ICC FIX: One-time debug logging (first call only)
    if getattr(cfg, "MUSIC_DEBUG", False) and not hasattr(cfg, "_music_logged"):
        eigs = np.linalg.eigvalsh(R)
        eig_frac = eigs[-5:][::-1] / eigs.sum() if eigs.sum() > 0 else eigs[-5:][::-1]
        print(f"[MUSIC] After trace-norm: tr(R)={np.real(np.trace(R)):.2f}, "
              f"||R||_F={np.linalg.norm(R,'fro'):.2f}", flush=True)
        print(f"[MUSIC] Top-5 eigenvalues (fraction): {eig_frac}", flush=True)
        
        # HYBRID PROOF: Check if this is a hybrid covariance
        if hasattr(cfg, '_hybrid_R_pred') and hasattr(cfg, '_hybrid_R_samp'):
            R_pred = cfg._hybrid_R_pred
            R_samp = cfg._hybrid_R_samp
            R_blend_raw = (1 - cfg._hybrid_beta) * R_pred + cfg._hybrid_beta * R_samp
            blend_diff = np.linalg.norm(R_blend_raw - R_pred, 'fro')
            print(f"[MUSIC] HYBRID PROOF: ||R_blend_raw-R_pred||_F = {blend_diff:.2e} (should be >>0)", flush=True)
            print(f"[MUSIC] HYBRID PROOF: β = {cfg._hybrid_beta:.3f}, top-5 eigen fraction = {eig_frac[0]:.3f} (should be ~0.6-0.8)", flush=True)
        
        # CRITICAL: Steering validation prints
        print(f"[MUSIC] Steering validation:", flush=True)
        # Test steering vector on a small grid
        test_phis = np.linspace(-30, 30, 5)
        test_thetas = np.linspace(-15, 15, 5)
        norms = []
        x_means = []
        y_means = []
        
        for phi in test_phis:
            for theta in test_thetas:
                a = _planar_steering(np.deg2rad(phi), np.deg2rad(theta), N_H, N_V, d_h, d_v, lam)
                norms.append(np.linalg.norm(a))
                # Check centered positions
                x_pos = (np.arange(N_H) - (N_H - 1) / 2.0) * d_h
                y_pos = (np.arange(N_V) - (N_V - 1) / 2.0) * d_v
                x_means.append(np.mean(x_pos))
                y_means.append(np.mean(y_pos))
        
        print(f"  ||a|| variance: {np.var(norms):.2e} (should be ~0)", flush=True)
        print(f"  mean(x): {np.mean(x_means):.2e}, mean(y): {np.mean(y_means):.2e} (should be ~0)", flush=True)
        
        cfg._music_logged = True
    
    # Extract geometry from config
    N_H = int(getattr(cfg, "N_H", int(np.sqrt(N))))
    N_V = int(getattr(cfg, "N_V", int(np.sqrt(N))))
    d_h = float(getattr(cfg, "d_H", 0.5 * float(getattr(cfg, "WAVEL", 0.3))))  # meters
    d_v = float(getattr(cfg, "d_V", 0.5))
    lam = float(getattr(cfg, "WAVEL", 0.0625))  # meters
    
    # Move to torch for processing
    R_torch = torch.as_tensor(R, dtype=torch.complex64, device=device)
    
    # Phase 2: Forward-Backward Averaging (sharper peaks, less bias)
    if use_fba:
        R_torch = forward_backward_average(R_torch, N_H, N_V)
    
    # Phase 2: SNR-Adaptive shrinkage + loading (Low-SNR Rescue Kit)
    if shrink is None:
        # SNR-adaptive shrinkage with per-scene noise estimation
        R_torch, alpha_used, eps_used, snr_est = snr_adaptive_shrinkage_loading(R_torch, cfg)
        if getattr(cfg, "MUSIC_DEBUG", False):
            print(f"[MUSIC] SNR-adaptive: α={alpha_used:.3f}, ε={eps_used:.2e}, SNR={snr_est:.1f}dB", flush=True)
    else:
        trace_N = torch.real(torch.trace(R_torch)) / N
        R_torch = (1.0 - shrink) * R_torch + shrink * trace_N * torch.eye(N, dtype=torch.complex64, device=device)
    
    # CRITICAL: Diagonal loading for numerical stability before EVD
    # ε ≈ 1e-3 · tr(R)/N (proportional to signal level)
    eps_diag = 1e-3 * torch.real(torch.trace(R_torch)) / N
    R_torch = R_torch + eps_diag * torch.eye(N, dtype=torch.complex64, device=device)
    
    # Eigendecomposition → noise subspace
    evals, evecs = torch.linalg.eigh(R_torch)  # ascending order
    # CRITICAL FIX: Sort ascending, use smallest (N - K) eigenvectors for noise
    assert torch.all(evals[:-1] <= evals[1:]), "Eigenvalues not sorted ascending!"
    
    # Noise subspace: smallest (N - K) eigenvectors
    # CRITICAL FIX: Use correct noise rank = N - K̂ (clamp K̂ ≥ 1, noise rank ≥ 1)
    K_signal = max(1, min(K, N - 1))  # Signal subspace rank (clamped)
    noise_rank = max(1, N - K_signal)    # Noise subspace rank (at least 1)
    U_noise = evecs[:, :noise_rank]      # Smallest noise_rank eigenvectors
    G = U_noise @ U_noise.conj().T       # Noise projector
    
    # SOFT PROJECTOR: Eigen-weighted MUSIC at low SNR
    # Check if we should use soft projector based on SNR
    if getattr(cfg, 'USE_SOFT_PROJECTOR', True):
        # Estimate SNR for gating
        signal_energy = torch.sum(evals[-K_signal:]) if K_signal > 0 else torch.sum(evals)
        noise_energy = torch.sum(evals[:noise_rank])
        snr_linear = signal_energy / max(noise_energy, 1e-12)
        snr_est = 10.0 * torch.log10(snr_linear)
        
        # SNR gating for soft projector
        SNR_thr = getattr(cfg, 'SOFT_PROJECTOR_SNR_THR', 0.0)  # dB
        w = getattr(cfg, 'SOFT_PROJECTOR_GATE_WIDTH', 3.0)  # dB
        g_soft = torch.sigmoid((SNR_thr - snr_est) / w)
        g_soft = torch.clamp(g_soft, 0.0, 1.0)
        
        if g_soft > 0.1:  # Use soft projector when SNR is low
            # Soft projector parameters
            p = getattr(cfg, 'SOFT_PROJECTOR_P', 0.7)
            tau = getattr(cfg, 'SOFT_PROJECTOR_TAU', 1e-6)  # Will be set per-scene
            
            # Store for use in spectrum computation
            G_soft_params = {
                'use_soft': True,
                'g_soft': float(g_soft),
                'p': p,
                'tau': tau,
                'evals': evals,
                'evecs': evecs,
                'snr_est': float(snr_est)
            }
            
            if getattr(cfg, "MUSIC_DEBUG", False):
                print(f"[MUSIC] SOFT PROJECTOR: g={g_soft:.3f}, SNR={snr_est:.1f}dB, p={p}, τ={tau:.2e}", flush=True)
        else:
            G_soft_params = {'use_soft': False}
            if getattr(cfg, "MUSIC_DEBUG", False):
                print(f"[MUSIC] HARD PROJECTOR: SNR={snr_est:.1f}dB (high SNR)", flush=True)
    else:
        G_soft_params = {'use_soft': False}
    
    # Angle grids (RADIANS internally, respect FOV clamps)
    phi_min_deg = getattr(cfg, "PHI_MIN_DEG", -60.0)
    phi_max_deg = getattr(cfg, "PHI_MAX_DEG", 60.0)
    theta_min_deg = getattr(cfg, "THETA_MIN_DEG", -30.0)
    theta_max_deg = getattr(cfg, "THETA_MAX_DEG", 30.0)
    
    phi_grid_rad = np.linspace(np.deg2rad(phi_min_deg), np.deg2rad(phi_max_deg), grid_phi, dtype=np.float32)
    theta_grid_rad = np.linspace(np.deg2rad(theta_min_deg), np.deg2rad(theta_max_deg), grid_theta, dtype=np.float32)
    
    # Compute MUSIC spectrum (2.5D or 2D) with soft projector support
    G_np = G.cpu().numpy()
    
    if use_2_5d:
        # 2.5D: Range-aware near-field steering with multiple range planes
        if r_planes is None:
            r_planes = [0.6, 1.2, 2.5, 5.0, 8.5]  # 5 planes: better coverage of 0.5-10m range
        S, winning_ranges = _chunked_music_spectrum_2_5d(
            G_np, phi_grid_rad, theta_grid_rad, r_planes, 
            (N_H, N_V, d_h, d_v, lam), chunk=4096, device=device,
            soft_params=G_soft_params
        )
        if getattr(cfg, "MUSIC_DEBUG", False):
            print(f"[MUSIC] 2.5D coarse scan: {len(r_planes)} range planes {r_planes}m", flush=True)
            print(f"[MUSIC] 2.5D: True, range planes=[{','.join([f'{r:.1f}' for r in r_planes])}]", flush=True)
            print(f"[MUSIC] BASELINE LOCKDOWN: 2.5D ON, joint NF-Newton per peak", flush=True)
    else:
        # 2D: Range-agnostic planar steering (far-field approximation).
        # For near-field, keep use_2_5d=True.
        S = _chunked_music_spectrum(G_np, phi_grid_rad, theta_grid_rad, (N_H, N_V, d_h, d_v, lam), 
                                     chunk=4096, device=device)
        winning_ranges = None
    
    # Peak picking: global top-K peaks (coarse)
    flat = S.reshape(-1)
    if len(flat) < topk:
        topk = len(flat)
    
    # Find top-k peaks (unsorted, then sort descending)
    idxs = np.argpartition(flat, -topk)[-topk:]  # indices of top-k values
    idxs = idxs[np.argsort(-flat[idxs])]  # sort by value descending
    
    # Map flat indices to (i, j) grid coordinates
    i_coords = idxs // len(theta_grid_rad)
    j_coords = idxs % len(theta_grid_rad)
    
    # Convert grids to degrees for refinement
    phi_grid_deg = np.rad2deg(phi_grid_rad)
    theta_grid_deg = np.rad2deg(theta_grid_rad)
    
    # Phase 2: Parabolic sub-grid refinement
    phi_peaks = []
    theta_peaks = []
    r_peaks = []  # Store winning ranges for each peak
    
    for i, j in zip(i_coords, j_coords):
        if peak_refine:
            phi_refined, theta_refined = parabolic_refine_2d(S, i, j, phi_grid_deg, theta_grid_deg)
            phi_peaks.append(phi_refined)
            theta_peaks.append(theta_refined)
        else:
            # Coarse grid peaks only
            phi_peaks.append(float(phi_grid_deg[i]))
            theta_peaks.append(float(theta_grid_deg[j]))
        
        # Store winning range for this peak (if 2.5D was used)
        if winning_ranges is not None:
            r_peaks.append(float(winning_ranges[i, j]))
        else:
            r_peaks.append(2.0)  # Default range for 2D
    
    phi_deg = np.array(phi_peaks, dtype=np.float32)
    theta_deg = np.array(theta_peaks, dtype=np.float32)
    
    # FOV-CRITICAL FIX: Clamp to dataset FOV to prevent wrap-to-180 saturation
    phi_deg = np.clip(phi_deg, phi_min_deg, phi_max_deg)
    theta_deg = np.clip(theta_deg, theta_min_deg, theta_max_deg)
    
    # Return ranges if 2.5D was used
    if use_2_5d:
        r_deg = np.array(r_peaks, dtype=np.float32)
        return phi_deg, theta_deg, r_deg, S, phi_grid_deg, theta_grid_deg
    else:
        return phi_deg, theta_deg, S, phi_grid_deg, theta_grid_deg


# =============================================================================
# MVDR-based evaluation (K-free alternative to MUSIC)
# =============================================================================

def mvdr_localize(R, cfg, *, grid_phi=361, grid_theta=181, threshold_db=12.0,
                  max_sources=5, do_refinement=True, delta_scale=1e-2, device="cuda"):
    """
    K-free multi-source localization using MVDR spectrum.
    
    This is the recommended entry point for localization without K estimation.
    Sources are detected via peak detection on the MVDR spectrum.
    
    Args:
        R: Covariance matrix [N, N] complex (R_blend recommended)
        cfg: Configuration object
        grid_phi: Azimuth grid resolution
        grid_theta: Elevation grid resolution
        threshold_db: Detection threshold (dB below max)
        max_sources: Maximum sources to detect
        do_refinement: Whether to do local 3D refinement
        delta_scale: MVDR diagonal loading scale
        device: 'cuda' or 'cpu'
        
    Returns:
        phi_deg: Detected azimuth angles (degrees) [n_sources]
        theta_deg: Detected elevation angles (degrees) [n_sources]
        r_m: Detected ranges (meters) [n_sources]
        spectrum: 2D MVDR spectrum [grid_phi, grid_theta]
    """
    from .music_gpu import mvdr_detect_sources
    
    # Run MVDR detection
    sources, spectrum = mvdr_detect_sources(
        R, cfg, device=device,
        grid_phi=grid_phi, grid_theta=grid_theta,
        delta_scale=delta_scale,
        threshold_db=threshold_db,
        max_sources=max_sources,
        do_refinement=do_refinement
    )
    
    # Extract arrays
    if len(sources) == 0:
        return np.array([]), np.array([]), np.array([]), spectrum
    
    phi_deg = np.array([s[0] for s in sources], dtype=np.float32)
    theta_deg = np.array([s[1] for s in sources], dtype=np.float32)
    r_m = np.array([s[2] for s in sources], dtype=np.float32)
    
    return phi_deg, theta_deg, r_m, spectrum


def eval_scene_mvdr(R, phi_gt, theta_gt, r_gt, cfg, *,
                    threshold_db=12.0, max_sources=5, device="cuda",
                    success_tol_phi=5.0, success_tol_theta=5.0, success_tol_r=1.0):
    """
    Evaluate MVDR localization on a single scene.
    
    Args:
        R: Covariance matrix [N, N] complex (R_blend recommended)
        phi_gt: Ground truth azimuth (degrees) [K]
        theta_gt: Ground truth elevation (degrees) [K]
        r_gt: Ground truth range (meters) [K]
        cfg: Configuration object
        threshold_db: Detection threshold
        max_sources: Maximum sources to detect
        device: 'cuda' or 'cpu'
        success_tol_*: Success tolerance thresholds
        
    Returns:
        metrics: dict with RMSE, success flag, K_detected, etc.
    """
    # Run MVDR localization
    phi_pred, theta_pred, r_pred, _ = mvdr_localize(
        R, cfg, device=device,
        threshold_db=threshold_db,
        max_sources=max_sources
    )
    
    K_gt = len(phi_gt)
    K_det = len(phi_pred)
    
    # Empty prediction case
    if K_det == 0:
        return {
            "rmse_phi": np.nan,
            "rmse_theta": np.nan,
            "rmse_r": np.nan,
            "K_detected": 0,
            "K_gt": K_gt,
            "success_flag": False,
        }
    
    # Use Hungarian matching to evaluate
    metrics = eval_scene_angles_ranges(
        phi_pred, theta_pred, r_pred,
        phi_gt, theta_gt, r_gt,
        success_tol_phi=success_tol_phi,
        success_tol_theta=success_tol_theta,
        success_tol_r=success_tol_r
    )
    
    metrics["K_detected"] = K_det
    metrics["K_gt"] = K_gt
    
    return metrics
