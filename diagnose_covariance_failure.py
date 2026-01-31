#!/usr/bin/env python3
"""
CRITICAL DIAGNOSTIC: Why is R_pred not producing MVDR-usable covariances?

This script will:
1. Load the trained model
2. Compare R_pred vs R_true eigenstructure
3. Check what MDL sees on each
4. Check MVDR spectrum quality
5. Identify the exact failure mode
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from ris_pytorch_pipeline.configs import cfg, mdl_cfg
from ris_pytorch_pipeline.dataset import ShardNPZDataset
from ris_pytorch_pipeline.covariance_utils import build_effective_cov_torch
from ris_pytorch_pipeline.loss import _vec2c

# IMPORTANT: Use OLD architecture to analyze existing checkpoint
# Disable slimming options so we can load the checkpoint
mdl_cfg.USE_FACTORED_SOFTARGMAX = False
mdl_cfg.USE_CONV_HPROJ = False

from ris_pytorch_pipeline.infer import load_model, estimate_k_ic_from_cov

def to_complex(ri_tensor):
    """Convert [B, N, N, 2] real/imag to [B, N, N] complex"""
    if torch.is_complex(ri_tensor):
        return ri_tensor
    return ri_tensor[..., 0] + 1j * ri_tensor[..., 1]

def to_ri(x):
    """Convert numpy/tensor to real-imag format for model input"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if torch.is_complex(x):
        return torch.stack([x.real, x.imag], dim=-1).float()
    if x.shape[-1] != 2:
        raise ValueError(f"Expected last dim=2, got {x.shape}")
    return x.float()

def compute_eigenspectrum(R):
    """Compute eigenvalues of covariance matrix (descending order)"""
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()
    if np.iscomplexobj(R):
        eigvals = np.linalg.eigvalsh(R)
    else:
        eigvals = np.linalg.eigvalsh(R[..., 0] + 1j * R[..., 1])
    return np.sort(eigvals)[::-1]  # Descending

def compute_subspace_overlap(R_pred, R_true, K):
    """
    Compute how well the top-K subspace of R_pred matches R_true.
    Returns: overlap in [0, 1] where 1 = perfect alignment
    """
    # Get eigenvectors
    eigvals_pred, eigvecs_pred = np.linalg.eigh(R_pred)
    eigvals_true, eigvecs_true = np.linalg.eigh(R_true)
    
    # Reverse to get descending order
    eigvecs_pred = eigvecs_pred[:, ::-1]
    eigvecs_true = eigvecs_true[:, ::-1]
    
    # Top-K subspaces
    U_pred = eigvecs_pred[:, :K]
    U_true = eigvecs_true[:, :K]
    
    # Projection overlap: ||P_true @ U_pred||_F^2 / K
    P_true = U_true @ U_true.conj().T
    overlap = np.linalg.norm(P_true @ U_pred, 'fro')**2 / K
    return overlap

def mvdr_spectrum_at_gt(R, phi_gt, theta_gt, r_gt):
    """
    Compute MVDR spectrum value at GT location.
    Higher = better peak.
    """
    N = R.shape[0]
    
    # Build steering vector at GT location
    N_H, N_V = cfg.N_H, cfg.N_V
    d_H, d_V = cfg.d_H, cfg.d_V
    k0 = cfg.k0
    
    h = np.arange(-(N_H-1)//2, (N_H+1)//2) * d_H
    v = np.arange(-(N_V-1)//2, (N_V+1)//2) * d_V
    H, V = np.meshgrid(h, v, indexing='xy')
    x = H.flatten()[:N]
    y = V.flatten()[:N]
    
    sin_phi = np.sin(phi_gt)
    cos_theta = np.cos(theta_gt)
    sin_theta = np.sin(theta_gt)
    
    planar = x * sin_phi * cos_theta + y * sin_theta
    curvature = (x**2 + y**2) / (2.0 * max(r_gt, 0.1))
    phase = k0 * (planar - curvature)
    a = np.exp(1j * phase) / np.sqrt(N)
    
    # MVDR spectrum: 1 / (a^H R^{-1} a)
    try:
        R_reg = R + 1e-4 * np.eye(N)
        R_inv = np.linalg.inv(R_reg)
        denom = np.real(a.conj() @ R_inv @ a)
        return 1.0 / max(denom, 1e-12)
    except:
        return 0.0

def main():
    print("=" * 60)
    print("COVARIANCE FAILURE DIAGNOSTIC")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: M={cfg.M}, N={cfg.N}, L={cfg.L}, K_MAX={cfg.K_MAX}")
    
    # Load model
    print("\n[1] Loading model...")
    cfg.ALLOW_INFER_WITHOUT_REFINER = True
    try:
        model = load_model(require_refiner=False, device=device)
        model.eval()
        print("   ✅ Model loaded")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return
    
    # Load test data
    print("\n[2] Loading test data...")
    test_dir = getattr(cfg, "DATA_SHARDS_TEST", getattr(cfg, "DATA_SHARDS_VAL", None))
    if test_dir is None:
        test_dir = Path(cfg.DATA_SHARDS_DIR) / "test"
    if not Path(test_dir).exists():
        test_dir = Path(cfg.DATA_SHARDS_DIR) / "val"
    
    try:
        ds = ShardNPZDataset(str(test_dir))
        print(f"   ✅ Loaded {len(ds)} samples from {test_dir}")
    except Exception as e:
        print(f"   ❌ Failed to load data: {e}")
        return
    
    # Analysis
    print("\n[3] Running analysis on 100 samples...")
    
    nmse_list = []
    subspace_overlap_list = []
    eigval_ratio_list = []  # ratio of top-K to trace
    k_mdl_pred_list = []
    k_mdl_true_list = []
    k_true_list = []
    mvdr_peak_pred_list = []
    mvdr_peak_true_list = []
    
    n_samples = min(100, len(ds))
    
    # Check what keys are available
    sample0 = ds[0]
    print(f"   Available keys: {list(sample0.keys())}")
    has_R_true = "R_true" in sample0
    print(f"   Has R_true: {has_R_true}")
    
    for i in range(n_samples):
        sample = ds[i]
        
        # Prepare inputs
        y = to_ri(sample["y"]).unsqueeze(0).to(device)
        H_full = to_ri(sample["H_full"]).unsqueeze(0).to(device)
        codes = to_ri(sample["codes"]).unsqueeze(0).to(device)
        snr_db = torch.tensor([sample.get("snr_db", sample.get("snr", 10.0))], device=device)
        
        # Get or compute R_true
        if has_R_true:
            R_true_ri = sample["R_true"]
            if isinstance(R_true_ri, np.ndarray):
                R_true_ri = torch.from_numpy(R_true_ri)
            R_true = to_complex(R_true_ri.to(device))
        else:
            # Compute R_true from steering vectors and GT positions
            # R_true = sum_k p_k * a(phi_k, theta_k, r_k) @ a^H
            ptr_raw = sample["ptr"]
            if isinstance(ptr_raw, torch.Tensor):
                ptr_np = ptr_raw.cpu().numpy()
            else:
                ptr_np = np.array(ptr_raw)
            K_true_val = int(sample["K"])
            phi_gt_np = ptr_np[:cfg.K_MAX].astype(np.float64)
            theta_gt_np = ptr_np[cfg.K_MAX:2*cfg.K_MAX].astype(np.float64)
            r_gt_np = ptr_np[2*cfg.K_MAX:3*cfg.K_MAX].astype(np.float64)
            
            # Build steering vectors
            N_H, N_V = cfg.N_H, cfg.N_V
            d_H, d_V = cfg.d_H, cfg.d_V
            k0 = cfg.k0
            N = cfg.N
            
            h = np.arange(-(N_H-1)//2, (N_H+1)//2) * d_H
            v = np.arange(-(N_V-1)//2, (N_V+1)//2) * d_V
            H_grid, V_grid = np.meshgrid(h, v, indexing='xy')
            x = H_grid.flatten()[:N]
            y_coord = V_grid.flatten()[:N]
            
            R_true_np = np.zeros((N, N), dtype=np.complex128)
            for k in range(K_true_val):
                sin_phi = np.sin(phi_gt_np[k])
                cos_theta = np.cos(theta_gt_np[k])
                sin_theta = np.sin(theta_gt_np[k])
                r_k = max(float(r_gt_np[k]), 0.1)
                
                planar = x * sin_phi * cos_theta + y_coord * sin_theta
                curvature = (x**2 + y_coord**2) / (2.0 * r_k)
                phase = k0 * (planar - curvature)
                a = np.exp(1j * phase) / np.sqrt(N)
                
                # Equal power assumption (or use powers if available)
                R_true_np += np.outer(a, a.conj())
            
            # Normalize
            R_true_np = R_true_np / (np.trace(R_true_np) + 1e-12) * N
            R_true = torch.from_numpy(R_true_np).to(device).to(torch.complex64)
        
        K_true = int(sample["K"])
        ptr = sample["ptr"]  # [3*K_MAX]: phi, theta, r
        phi_gt = ptr[:cfg.K_MAX]
        theta_gt = ptr[cfg.K_MAX:2*cfg.K_MAX]
        r_gt = ptr[2*cfg.K_MAX:3*cfg.K_MAX]
        
        # Forward pass
        with torch.no_grad():
            out = model(y, H_full, codes, snr_db=snr_db)
        
        # Build R_pred from factors
        A_angle = _vec2c(out["cov_fact_angle"]).to(device)
        A_range = _vec2c(out["cov_fact_range"]).to(device)
        lam_range = float(getattr(mdl_cfg, "LAM_RANGE_FACTOR", 0.3))
        
        R_pred = A_angle @ A_angle.conj().transpose(-2, -1) + lam_range * (A_range @ A_range.conj().transpose(-2, -1))
        R_pred = 0.5 * (R_pred + R_pred.conj().transpose(-2, -1))
        
        # Normalize
        tr_pred = torch.diagonal(R_pred, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-9)
        R_pred = R_pred / tr_pred.view(-1, 1, 1)
        tr_true = torch.diagonal(R_true, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-9)
        R_true = R_true / tr_true.view(-1, 1, 1)
        
        # Build effective covariance (as inference does)
        R_eff_pred = build_effective_cov_torch(
            R_pred,
            snr_db=snr_db,
            R_samp=None,
            beta=None,
            diag_load=True,
            apply_shrink=True,
            target_trace=float(cfg.N),
        )
        
        R_pred_np = R_pred[0].cpu().numpy()
        R_true_np = R_true[0].cpu().numpy() if R_true.dim() == 3 else R_true.cpu().numpy()
        R_eff_pred_np = R_eff_pred[0].cpu().numpy()
        
        # NMSE
        diff = R_pred_np - R_true_np
        nmse = np.linalg.norm(diff, 'fro')**2 / (np.linalg.norm(R_true_np, 'fro')**2 + 1e-12)
        nmse_list.append(nmse)
        
        # Eigenspectrum analysis
        eigvals_pred = compute_eigenspectrum(R_pred_np)
        eigvals_true = compute_eigenspectrum(R_true_np)
        
        # Top-K energy ratio
        topk_pred = eigvals_pred[:K_true].sum() / (eigvals_pred.sum() + 1e-12)
        topk_true = eigvals_true[:K_true].sum() / (eigvals_true.sum() + 1e-12)
        eigval_ratio_list.append((topk_pred, topk_true))
        
        # Subspace overlap
        if K_true > 0 and K_true < cfg.N:
            overlap = compute_subspace_overlap(R_pred_np, R_true_np, K_true)
            subspace_overlap_list.append(overlap)
        
        # MDL K estimation
        eigvals_eff_pred = compute_eigenspectrum(R_eff_pred_np)
        k_mdl_pred = estimate_k_ic_from_cov(R_eff_pred_np, T=cfg.L, method="mdl", kmax=cfg.K_MAX)
        k_mdl_true = estimate_k_ic_from_cov(R_true_np, T=cfg.L, method="mdl", kmax=cfg.K_MAX)
        k_mdl_pred_list.append(k_mdl_pred)
        k_mdl_true_list.append(k_mdl_true)
        k_true_list.append(K_true)
        
        # MVDR peak quality at first GT source
        if K_true > 0:
            mvdr_pred = mvdr_spectrum_at_gt(R_eff_pred_np, phi_gt[0], theta_gt[0], r_gt[0])
            mvdr_true = mvdr_spectrum_at_gt(R_true_np, phi_gt[0], theta_gt[0], r_gt[0])
            mvdr_peak_pred_list.append(mvdr_pred)
            mvdr_peak_true_list.append(mvdr_true)
        
        if (i + 1) % 20 == 0:
            print(f"   Processed {i+1}/{n_samples}...")
    
    # Report
    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)
    
    print("\n[A] NMSE (R_pred vs R_true):")
    print(f"    Mean:   {np.mean(nmse_list):.4f}")
    print(f"    Median: {np.median(nmse_list):.4f}")
    print(f"    Std:    {np.std(nmse_list):.4f}")
    print(f"    NMSE < 0.5: {100*np.mean(np.array(nmse_list) < 0.5):.1f}%")
    print(f"    NMSE < 0.3: {100*np.mean(np.array(nmse_list) < 0.3):.1f}%")
    
    print("\n[B] Eigenvalue concentration (top-K / total):")
    topk_pred = [x[0] for x in eigval_ratio_list]
    topk_true = [x[1] for x in eigval_ratio_list]
    print(f"    R_pred: mean={np.mean(topk_pred):.4f}, median={np.median(topk_pred):.4f}")
    print(f"    R_true: mean={np.mean(topk_true):.4f}, median={np.median(topk_true):.4f}")
    print(f"    Gap:    {np.mean(topk_true) - np.mean(topk_pred):.4f}")
    
    print("\n[C] Subspace overlap (top-K eigenvectors):")
    if subspace_overlap_list:
        print(f"    Mean:   {np.mean(subspace_overlap_list):.4f}")
        print(f"    Median: {np.median(subspace_overlap_list):.4f}")
        print(f"    Overlap > 0.8: {100*np.mean(np.array(subspace_overlap_list) > 0.8):.1f}%")
        print(f"    Overlap > 0.5: {100*np.mean(np.array(subspace_overlap_list) > 0.5):.1f}%")
    
    print("\n[D] MDL K estimation:")
    k_true_arr = np.array(k_true_list)
    k_mdl_pred_arr = np.array(k_mdl_pred_list)
    k_mdl_true_arr = np.array(k_mdl_true_list)
    print(f"    K_true distribution: {np.bincount(k_true_arr, minlength=6)}")
    print(f"    MDL on R_pred:       {np.bincount(k_mdl_pred_arr, minlength=6)}")
    print(f"    MDL on R_true:       {np.bincount(k_mdl_true_arr, minlength=6)}")
    print(f"    MDL(R_pred) == K_true: {100*np.mean(k_mdl_pred_arr == k_true_arr):.1f}%")
    print(f"    MDL(R_true) == K_true: {100*np.mean(k_mdl_true_arr == k_true_arr):.1f}%")
    print(f"    MDL(R_pred) == 5 (max): {100*np.mean(k_mdl_pred_arr == 5):.1f}%")
    
    print("\n[E] MVDR peak quality at GT location:")
    if mvdr_peak_pred_list:
        mvdr_pred_arr = np.array(mvdr_peak_pred_list)
        mvdr_true_arr = np.array(mvdr_peak_true_list)
        print(f"    R_pred: median={np.median(mvdr_pred_arr):.4f}, mean={np.mean(mvdr_pred_arr):.4f}")
        print(f"    R_true: median={np.median(mvdr_true_arr):.4f}, mean={np.mean(mvdr_true_arr):.4f}")
        print(f"    Ratio (pred/true): {np.median(mvdr_pred_arr) / (np.median(mvdr_true_arr) + 1e-12):.4f}")
    
    # Diagnosis
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    nmse_mean = np.mean(nmse_list)
    overlap_mean = np.mean(subspace_overlap_list) if subspace_overlap_list else 0
    mdl_correct = np.mean(k_mdl_pred_arr == k_true_arr)
    mdl_max = np.mean(k_mdl_pred_arr == 5)
    
    if nmse_mean > 0.8:
        print("❌ NMSE is very high (>0.8): R_pred is not matching R_true at all.")
        print("   → Check if covariance loss gradient is flowing (requires_grad)")
        print("   → Check if loss weight lam_cov is non-zero")
    elif nmse_mean > 0.5:
        print("⚠️  NMSE is moderate (0.5-0.8): R_pred is partially learning R_true.")
        print("   → Eigenstructure might be off even if Frobenius norm is closer")
    else:
        print("✅ NMSE is reasonable (<0.5)")
    
    if overlap_mean < 0.5:
        print("❌ Subspace overlap is low (<0.5): Eigenvectors are misaligned.")
        print("   → MVDR needs correct eigenvector directions, not just eigenvalues")
        print("   → Subspace alignment loss might be too weak")
    elif overlap_mean < 0.8:
        print("⚠️  Subspace overlap is moderate (0.5-0.8)")
    else:
        print("✅ Subspace overlap is good (>0.8)")
    
    if mdl_max > 0.5:
        print("❌ MDL on R_pred predicts K=5 (max) >50% of the time!")
        print("   → R_pred eigenvalues are too uniform (no clear signal/noise gap)")
        print("   → Need stronger eigengap regularization or peak contrast loss")
    
    topk_gap = np.mean(topk_true) - np.mean(topk_pred)
    if topk_gap > 0.2:
        print(f"❌ Top-K energy gap is large ({topk_gap:.2f}): R_pred is too diffuse.")
        print("   → R_true concentrates energy in top-K, R_pred spreads it out")
        print("   → This is the core problem: low-rank structure is not learned")

if __name__ == "__main__":
    main()
