#!/usr/bin/env python3
"""
MVDR Spectrum Visualization and Testing

This script tests the K-free MVDR localization pipeline:
1. Loads test samples with ground truth
2. Computes MVDR spectrum from R_blend
3. Visualizes spectrum slices at each range plane
4. Compares detected sources vs ground truth

Usage:
    python eval_mvdr.py --n_samples 5 --save_plots
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ris_pytorch_pipeline.configs import cfg, mdl_cfg
from ris_pytorch_pipeline.dataset import ShardNPZDataset
from ris_pytorch_pipeline.music_gpu import (
    get_gpu_estimator, mvdr_detect_sources, mvdr_spectrum_2_5d
)


def ri_to_complex(ri_tensor):
    """Convert [N, N, 2] real/imag to [N, N] complex."""
    if ri_tensor is None:
        return None
    if isinstance(ri_tensor, torch.Tensor):
        ri_tensor = ri_tensor.numpy()
    return ri_tensor[..., 0] + 1j * ri_tensor[..., 1]


def compute_r_blend(R_true, R_samp, beta=0.3):
    """Compute hybrid blended covariance."""
    if R_samp is None or np.allclose(R_samp, 0):
        return R_true
    
    # Simple blend: beta * R_pred + (1-beta) * R_samp
    # For this test, use R_true as R_pred proxy
    R_blend = beta * R_true + (1 - beta) * R_samp
    
    # Hermitize
    R_blend = 0.5 * (R_blend + R_blend.conj().T)
    
    # Normalize
    trace_R = np.trace(R_blend).real
    if trace_R > 1e-8:
        R_blend = R_blend * (cfg.N / trace_R)
    
    return R_blend


def visualize_mvdr_spectrum(spectrum_3d, spectrum_max, candidates, gt_phi, gt_theta, gt_r,
                            r_planes, phi_range, theta_range, sample_idx, save_dir=None):
    """
    Visualize MVDR spectrum with ground truth overlay.
    """
    n_planes = spectrum_3d.shape[2]
    n_cols = min(4, n_planes)
    n_rows = (n_planes + n_cols - 1) // n_cols + 1  # +1 for max spectrum
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes)
    
    # Plot each range plane
    for ri in range(n_planes):
        row, col = ri // n_cols, ri % n_cols
        ax = axes[row, col]
        
        spectrum_slice = spectrum_3d[:, :, ri].T  # Transpose for imshow (theta on y-axis)
        im = ax.imshow(spectrum_slice, extent=[phi_range[0], phi_range[1], 
                                                theta_range[0], theta_range[1]],
                       aspect='auto', origin='lower', cmap='hot')
        
        # Overlay ground truth
        for k in range(len(gt_phi)):
            # Check if this source is close to this range plane
            r_dist = abs(gt_r[k] - r_planes[ri])
            if r_dist < 1.0:  # Within 1m of plane
                ax.scatter(np.rad2deg(gt_phi[k]), np.rad2deg(gt_theta[k]), 
                          c='cyan', s=100, marker='x', linewidths=2, label='GT' if k == 0 else '')
        
        ax.set_title(f'r = {r_planes[ri]:.2f} m')
        ax.set_xlabel('φ (deg)')
        ax.set_ylabel('θ (deg)')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Plot max spectrum in last row
    ax_max = axes[-1, 0]
    im = ax_max.imshow(spectrum_max.T, extent=[phi_range[0], phi_range[1],
                                                theta_range[0], theta_range[1]],
                       aspect='auto', origin='lower', cmap='hot')
    
    # Overlay ground truth
    for k in range(len(gt_phi)):
        ax_max.scatter(np.rad2deg(gt_phi[k]), np.rad2deg(gt_theta[k]),
                      c='cyan', s=100, marker='x', linewidths=2)
    
    # Overlay detected candidates
    for phi, theta, r, conf in candidates[:5]:
        ax_max.scatter(phi, theta, c='lime', s=80, marker='o', 
                      facecolors='none', linewidths=2)
    
    ax_max.set_title('Max over range (with detections)')
    ax_max.set_xlabel('φ (deg)')
    ax_max.set_ylabel('θ (deg)')
    plt.colorbar(im, ax=ax_max, shrink=0.8)
    
    # Hide unused subplots
    for col in range(1, n_cols):
        axes[-1, col].axis('off')
    
    plt.suptitle(f'MVDR Spectrum - Sample {sample_idx} (K={len(gt_phi)})', fontsize=14)
    plt.tight_layout()
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f'mvdr_spectrum_sample_{sample_idx:03d}.png', dpi=150)
        print(f"  Saved: {save_dir / f'mvdr_spectrum_sample_{sample_idx:03d}.png'}")
    else:
        plt.show()
    
    plt.close()


def compute_errors(detected, gt_phi, gt_theta, gt_r):
    """
    Compute localization errors using Hungarian matching.
    """
    from scipy.optimize import linear_sum_assignment
    
    K_det = len(detected)
    K_gt = len(gt_phi)
    
    if K_det == 0 or K_gt == 0:
        return {
            'K_det': K_det, 'K_gt': K_gt,
            'phi_rmse': np.nan, 'theta_rmse': np.nan, 'r_rmse': np.nan
        }
    
    # Build cost matrix
    cost = np.zeros((K_det, K_gt))
    for i, (phi_d, theta_d, r_d, _) in enumerate(detected):
        for j in range(K_gt):
            dphi = abs(phi_d - np.rad2deg(gt_phi[j]))
            dtheta = abs(theta_d - np.rad2deg(gt_theta[j]))
            dr = abs(r_d - gt_r[j])
            cost[i, j] = np.sqrt(dphi**2 + dtheta**2) + dr  # Combined metric
    
    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Compute per-dimension errors
    phi_errors, theta_errors, r_errors = [], [], []
    for i, j in zip(row_ind, col_ind):
        phi_d, theta_d, r_d, _ = detected[i]
        phi_errors.append(abs(phi_d - np.rad2deg(gt_phi[j])))
        theta_errors.append(abs(theta_d - np.rad2deg(gt_theta[j])))
        r_errors.append(abs(r_d - gt_r[j]))
    
    return {
        'K_det': K_det,
        'K_gt': K_gt,
        'phi_rmse': np.sqrt(np.mean(np.array(phi_errors)**2)),
        'theta_rmse': np.sqrt(np.mean(np.array(theta_errors)**2)),
        'r_rmse': np.sqrt(np.mean(np.array(r_errors)**2)),
        'n_matched': len(row_ind)
    }


def main():
    parser = argparse.ArgumentParser(description='MVDR Spectrum Visualization')
    parser.add_argument('--data_dir', type=str, default=cfg.DATA_SHARDS_TEST,
                        help='Path to test shards')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save plots instead of displaying')
    parser.add_argument('--output_dir', type=str, default='mvdr_visualizations',
                        help='Output directory for plots')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Blend weight for R_blend')
    parser.add_argument('--delta_scale', type=float, default=1e-2,
                        help='MVDR diagonal loading scale')
    parser.add_argument('--threshold_db', type=float, default=12.0,
                        help='Detection threshold (dB below max)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading data from: {args.data_dir}")
    try:
        dataset = ShardNPZDataset(args.data_dir)
        print(f"Loaded {len(dataset)} samples")
    except FileNotFoundError:
        print(f"ERROR: No data found at {args.data_dir}")
        print("Please run data generation first or specify --data_dir")
        return
    
    # Get estimator
    estimator = get_gpu_estimator(cfg, device)
    r_planes = estimator.default_r_planes_mvdr
    
    print(f"\nMVDR Configuration:")
    print(f"  Range planes: {r_planes}")
    print(f"  Delta scale: {args.delta_scale}")
    print(f"  Threshold: {args.threshold_db} dB")
    print(f"  Beta (blend): {args.beta}")
    
    # Process samples
    all_errors = []
    n_samples = min(args.n_samples, len(dataset))
    
    print(f"\nProcessing {n_samples} samples...")
    print("-" * 60)
    
    for idx in range(n_samples):
        sample = dataset[idx]
        
        # Extract ground truth
        K = sample['K'].item()
        gt_phi = sample['phi'].numpy()  # radians
        gt_theta = sample['theta'].numpy()  # radians
        gt_r = sample['r'].numpy()  # meters
        snr = sample['snr'].item()
        
        print(f"\nSample {idx}: K={K}, SNR={snr:.1f} dB")
        print(f"  GT φ: {[f'{np.rad2deg(p):.1f}°' for p in gt_phi]}")
        print(f"  GT θ: {[f'{np.rad2deg(t):.1f}°' for t in gt_theta]}")
        print(f"  GT r: {[f'{r:.2f}m' for r in gt_r]}")
        
        # Get covariances
        R_true = ri_to_complex(sample['R'])
        R_samp = ri_to_complex(sample.get('R_samp'))
        
        # Compute R_blend
        R_blend = compute_r_blend(R_true, R_samp, beta=args.beta)
        
        # Run MVDR detection
        sources, spectrum_max = mvdr_detect_sources(
            R_blend, cfg, device=device,
            delta_scale=args.delta_scale,
            threshold_db=args.threshold_db,
            max_sources=cfg.K_MAX,
            do_refinement=True
        )
        
        print(f"  Detected {len(sources)} sources:")
        for phi, theta, r, conf in sources:
            print(f"    φ={phi:.1f}°, θ={theta:.1f}°, r={r:.2f}m, conf={conf:.2e}")
        
        # Compute errors
        errors = compute_errors(sources, gt_phi, gt_theta, gt_r)
        all_errors.append(errors)
        print(f"  Errors: φ={errors['phi_rmse']:.2f}°, θ={errors['theta_rmse']:.2f}°, r={errors['r_rmse']:.3f}m")
        
        # Visualize
        if args.save_plots or args.n_samples <= 5:
            candidates, _, spectrum_3d = mvdr_spectrum_2_5d(
                R_blend, cfg, device=device, delta_scale=args.delta_scale
            )
            
            visualize_mvdr_spectrum(
                spectrum_3d, spectrum_max, sources,
                gt_phi, gt_theta, gt_r, r_planes,
                phi_range=(cfg.PHI_MIN_DEG, cfg.PHI_MAX_DEG),
                theta_range=(cfg.THETA_MIN_DEG, cfg.THETA_MAX_DEG),
                sample_idx=idx,
                save_dir=args.output_dir if args.save_plots else None
            )
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    valid_errors = [e for e in all_errors if not np.isnan(e['phi_rmse'])]
    if valid_errors:
        mean_phi = np.mean([e['phi_rmse'] for e in valid_errors])
        mean_theta = np.mean([e['theta_rmse'] for e in valid_errors])
        mean_r = np.mean([e['r_rmse'] for e in valid_errors])
        
        k_det_total = sum(e['K_det'] for e in all_errors)
        k_gt_total = sum(e['K_gt'] for e in all_errors)
        
        print(f"Mean φ RMSE: {mean_phi:.2f}°")
        print(f"Mean θ RMSE: {mean_theta:.2f}°")
        print(f"Mean r RMSE: {mean_r:.3f}m")
        print(f"Detection: {k_det_total}/{k_gt_total} sources ({100*k_det_total/max(1,k_gt_total):.1f}%)")
    else:
        print("No valid samples processed.")
    
    print("=" * 60)


if __name__ == '__main__':
    main()

