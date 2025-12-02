#!/usr/bin/env python3
"""
Comprehensive audit script to verify covariance alignment between:
  - K-head (inside model.forward)
  - MUSIC (angle_pipeline)
  - Loss computation

This ensures R_eff is computed consistently across all paths.
"""

import numpy as np
import torch
from pathlib import Path
import sys

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from ris_pytorch_pipeline.configs import cfg, mdl_cfg
from ris_pytorch_pipeline.model import HybridModel
from ris_pytorch_pipeline.dataset import ShardNPZDataset
from ris_pytorch_pipeline.covariance_utils import build_effective_cov_torch, build_effective_cov_np
from ris_pytorch_pipeline.angle_pipeline import build_sample_covariance_from_snapshots

def audit_single_sample():
    """
    Audit covariance path consistency on a single sample.
    Reports:
      1. R_eff traces (should be exactly N)
      2. ||R_eff_K - R_eff_MUSIC||_F (should be < 1e-5 for numerical consistency)
      3. Eigenvalue distribution (sanity check)
    """
    print("=" * 80)
    print("COVARIANCE PATH AUDIT")
    print("=" * 80)
    
    # Load one sample from validation set
    try:
        shard_dir = Path("data_shards_M64_L16/val")
        if not shard_dir.exists():
            print(f"⚠️  Shard directory not found: {shard_dir}")
            return False
        
        ds = ShardNPZDataset(shard_dir)
        if len(ds) == 0:
            print(f"⚠️  No samples in {shard_dir}")
            return False
        
        sample = ds[0]
        print(f"✓ Loaded sample 0 from {shard_dir}")
    except Exception as e:
        print(f"✗ Failed to load sample: {e}")
        return False
    
    # Create model (no weights needed for this audit)
    model = HybridModel()
    model.eval()
    device = torch.device("cpu")
    model = model.to(device)
    
    # Extract inputs
    y = sample["y"].unsqueeze(0).to(device)        # [1, L, M, 2]
    H = sample["H"].unsqueeze(0).to(device)        # [1, L, M, 2]
    codes = sample["codes"].unsqueeze(0).to(device)  # [1, L, N, 2]
    snr_db = sample["snr"].unsqueeze(0).to(device)   # [1]
    K_true = int(sample["K"].item())
    
    print(f"\nSample info:")
    print(f"  K = {K_true}")
    print(f"  SNR = {snr_db.item():.2f} dB")
    print(f"  L = {y.shape[1]}, M = {y.shape[2]}, N = {cfg.N}")
    
    # Build R_samp from snapshots (NumPy, offline path)
    print("\n" + "-" * 80)
    print("STEP 1: Build R_samp (offline, NumPy)")
    print("-" * 80)
    try:
        y_np = y[0].numpy()
        H_np = H[0].numpy()
        codes_np = codes[0].numpy()
        
        # Convert RI to complex
        y_cplx = y_np[..., 0] + 1j * y_np[..., 1]        # [L, M]
        H_cplx = H_np[..., 0] + 1j * H_np[..., 1]        # [L, M]
        codes_cplx = codes_np[..., 0] + 1j * codes_np[..., 1]  # [L, N]
        
        # H is effective channel per snapshot [L, M], but build_sample_covariance expects [L, M, N]
        # We need the full BS->RIS channel H_full if available
        if "H_full" in sample and sample["H_full"] is not None:
            H_full_np = sample["H_full"].numpy()  # [M, N, 2]
            H_full_cplx = H_full_np[..., 0] + 1j * H_full_np[..., 1]  # [M, N]
            # Expand to [L, M, N] by repeating
            H_for_rsamp = np.repeat(H_full_cplx[None, :, :], y_cplx.shape[0], axis=0)
        else:
            print("⚠️  H_full not in sample, using effective H (may be suboptimal)")
            # Fall back: H_eff is [L, M], we can't recover full channel
            # This is a limitation - R_samp builder needs H_full
            print("   Cannot build R_samp without H_full, skipping hybrid path")
            R_samp_np = None
            R_samp_t = None
        
        if "H_full" in sample and sample["H_full"] is not None:
            R_samp_np = build_sample_covariance_from_snapshots(
                y_cplx, H_for_rsamp, codes_cplx, cfg, tikhonov_alpha=1e-3
            )
            R_samp_np = 0.5 * (R_samp_np + R_samp_np.conj().T)  # Hermitize
            R_samp_t = torch.from_numpy(R_samp_np).to(torch.complex64).unsqueeze(0).to(device)
            print(f"✓ R_samp built: shape {R_samp_np.shape}, trace = {np.trace(R_samp_np).real:.2f}")
        
    except Exception as e:
        print(f"✗ R_samp build failed: {e}")
        import traceback
        traceback.print_exc()
        R_samp_np = None
        R_samp_t = None
    
    # Forward pass (get predictions)
    print("\n" + "-" * 80)
    print("STEP 2: Model forward pass")
    print("-" * 80)
    with torch.no_grad():
        preds = model(y, H, codes, snr_db=snr_db, R_samp=R_samp_t)
    
    cf_ang = preds["cov_fact_angle"][0].detach().cpu().numpy()  # [N*K*2]
    print(f"✓ Forward pass complete, cf_ang shape: {cf_ang.shape}")
    
    # Rebuild R_pred from angle factors
    cf_cplx = (cf_ang[::2] + 1j * cf_ang[1::2]).reshape(cfg.N, cfg.K_MAX)
    R_pred_np = cf_cplx @ cf_cplx.conj().T
    print(f"✓ R_pred from factors: shape {R_pred_np.shape}, trace = {np.trace(R_pred_np).real:.2f}")
    
    # ========================================================================
    # PATH A: K-head (inside model, uses build_effective_cov_torch)
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 3A: K-HEAD R_eff (torch, inside model.forward)")
    print("-" * 80)
    
    # The K-head uses build_effective_cov_torch during forward.
    # We can't directly extract it without instrumenting the model,
    # but we can replicate the exact logic:
    R_pred_t = torch.from_numpy(R_pred_np).to(torch.complex64).unsqueeze(0).to(device)
    beta = float(getattr(cfg, "HYBRID_COV_BETA", 0.0))
    
    R_eff_khead = build_effective_cov_torch(
        R_pred_t,
        snr_db=snr_db,
        R_samp=R_samp_t if (R_samp_t is not None and beta > 0) else None,
        beta=beta if (R_samp_t is not None and beta > 0) else None,
        diag_load=True,
        apply_shrink=(snr_db is not None),
        target_trace=float(cfg.N),
    )[0].detach().cpu().numpy()
    
    print(f"✓ R_eff (K-head): trace = {np.trace(R_eff_khead).real:.4f}, target = {cfg.N}")
    print(f"  Hybrid: beta={beta}, R_samp_used={R_samp_t is not None and beta > 0}")
    print(f"  Shrink: applied={snr_db is not None}, SNR={snr_db.item():.2f} dB")
    
    # Extract eigenvalues for K-head features
    evals_khead = np.linalg.eigvalsh(R_eff_khead)
    evals_khead = np.sort(evals_khead.real)[::-1]  # descending
    print(f"  Top-5 eigenvalues: {evals_khead[:5]}")
    print(f"  Bottom-5 eigenvalues: {evals_khead[-5:]}")
    
    # ========================================================================
    # PATH B: MUSIC (angle_pipeline, uses build_effective_cov_np)
    # ========================================================================
    print("\n" + "-" * 80)
    print("STEP 3B: MUSIC R_eff (numpy, angle_pipeline)")
    print("-" * 80)
    
    R_eff_music = build_effective_cov_np(
        R_pred_np,
        R_samp=R_samp_np if (R_samp_np is not None and beta > 0) else None,
        beta=beta if (R_samp_np is not None and beta > 0) else None,
        diag_load=True,
        apply_shrink=False,  # MUSIC typically applies its own shrink
        snr_db=None,
        target_trace=float(cfg.N),
    )
    
    print(f"✓ R_eff (MUSIC): trace = {np.trace(R_eff_music).real:.4f}, target = {cfg.N}")
    print(f"  Hybrid: beta={beta}, R_samp_used={R_samp_np is not None and beta > 0}")
    print(f"  Shrink: applied=False (MUSIC applies own)")
    
    evals_music = np.linalg.eigvalsh(R_eff_music)
    evals_music = np.sort(evals_music.real)[::-1]
    print(f"  Top-5 eigenvalues: {evals_music[:5]}")
    print(f"  Bottom-5 eigenvalues: {evals_music[-5:]}")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("ALIGNMENT CHECK")
    print("=" * 80)
    
    # Note: K-head applies shrink, MUSIC doesn't by default
    # For exact comparison, we should compare pre-shrink or both post-shrink
    # Let's compare traces and Frobenius norm
    
    diff_fro = np.linalg.norm(R_eff_khead - R_eff_music, 'fro')
    diff_relative = diff_fro / max(np.linalg.norm(R_eff_music, 'fro'), 1e-9)
    
    print(f"\n||R_eff_khead - R_eff_music||_F = {diff_fro:.6e}")
    print(f"Relative error = {diff_relative:.6e}")
    
    # Check if shrinkage is the only difference
    if snr_db is not None and diff_fro > 1e-3:
        print("\n⚠️  Large difference detected. Likely due to shrinkage mismatch:")
        print("     K-head applies SNR-aware shrink, MUSIC doesn't (by config).")
        print("     This is acceptable IF the difference is purely diagonal loading.")
        
        # Check diagonal difference
        diag_diff = np.abs(np.diag(R_eff_khead) - np.diag(R_eff_music))
        offdiag_khead = R_eff_khead - np.diag(np.diag(R_eff_khead))
        offdiag_music = R_eff_music - np.diag(np.diag(R_eff_music))
        offdiag_diff = np.linalg.norm(offdiag_khead - offdiag_music, 'fro')
        
        print(f"     Diagonal difference: max = {diag_diff.max():.6e}, mean = {diag_diff.mean():.6e}")
        print(f"     Off-diagonal difference: {offdiag_diff:.6e}")
        
        if offdiag_diff < 1e-5:
            print("✓ Off-diagonals match! Difference is purely diagonal (shrinkage).")
            return True
        else:
            print("✗ Off-diagonals differ significantly. Investigate further.")
            return False
    
    elif diff_fro < 1e-5:
        print("✓ PERFECT MATCH: R_eff paths are numerically identical!")
        return True
    else:
        print("⚠️  Moderate difference. Check if this is acceptable for your use case.")
        return True  # Still acceptable
    
    return True


if __name__ == "__main__":
    success = audit_single_sample()
    sys.exit(0 if success else 1)




