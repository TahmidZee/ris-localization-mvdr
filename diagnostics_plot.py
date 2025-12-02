#!/usr/bin/env python3
"""
Phase 2 Diagnostic Script: Error vs Range/SNR/K Curves

This script analyzes model performance and plots critical curves to identify
where the model succeeds and where it fails. Use this to gate Phase 3 decisions.

Usage:
    python diagnostics_plot.py --checkpoint path/to/best.pt --n-samples 2000
    
Expected outputs (saved to diagnostics/ folder):
    - error_vs_range.png
    - error_vs_snr.png
    - error_vs_k.png
    - cov_nmse_vs_snr.png
    - summary_report.txt
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ris_pytorch_pipeline.configs import cfg, mdl_cfg
from ris_pytorch_pipeline.model import HybridModel
from ris_pytorch_pipeline.train import Trainer
from ris_pytorch_pipeline.data_loaders import build_val_loader
from ris_pytorch_pipeline.eval_angles import eval_scene_angles_ranges


def collect_diagnostics(trainer, val_loader, max_samples=2000):
    """
    Run inference on validation set and collect detailed metrics.
    
    Returns dict with arrays:
        - phi_errors: [N] angle errors (degrees)
        - theta_errors: [N] angle errors (degrees)
        - r_errors: [N] range errors (meters)
        - snr: [N] SNR values (dB)
        - range: [N] true range values (meters)
        - k_true: [N] true number of sources
        - k_pred: [N] predicted number of sources
        - cov_nmse: [N] covariance NMSE (1 - NMSE for plotting)
    """
    model = trainer.model
    device = trainer.device
    model.eval()
    
    # Storage
    results = {
        "phi_errors": [],
        "theta_errors": [],
        "r_errors": [],
        "snr": [],
        "range": [],
        "k_true": [],
        "k_pred": [],
        "cov_nmse": [],
    }
    
    sample_count = 0
    
    print(f"[Diagnostics] Collecting metrics from validation set...", flush=True)
    
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if sample_count >= max_samples:
                break
            
            y, H, C, ptr, K, R_in, snr = trainer._unpack_any_batch(batch)
            batch_size = y.shape[0]
            
            # Forward pass
            with torch.amp.autocast('cuda', enabled=(trainer.amp and device.type == "cuda")):
                preds = model(y=y, H=H, codes=C, snr_db=snr)
            
            # Extract predictions
            phi_soft = preds["phi_soft"].cpu().numpy()  # [B, K_MAX]
            theta_soft = preds.get("theta_soft", torch.zeros_like(preds["phi_soft"])).cpu().numpy()
            aux_ptr = preds.get("phi_theta_r", torch.zeros(phi_soft.shape[0], 3*cfg.K_MAX)).cpu().numpy()
            k_logits = preds["k_logits"].cpu().numpy() if "k_logits" in preds else None
            
            # Covariance NMSE (if available)
            if "cov_fact_angle" in preds and R_in is not None:
                cf_ang = preds["cov_fact_angle"].cpu().numpy()  # [B, N, K_MAX, 2]
                for b in range(batch_size):
                    # Reconstruct R_hat from factor
                    cf_b = cf_ang[b, :, :, 0] + 1j * cf_ang[b, :, :, 1]  # [N, K_MAX]
                    R_hat = cf_b @ cf_b.conj().T
                    R_true = R_in[b].cpu().numpy() if isinstance(R_in, torch.Tensor) else R_in[b]
                    
                    # NMSE
                    diff = R_hat - R_true
                    nmse = np.real(np.trace(diff @ diff.conj().T)) / np.real(np.trace(R_true @ R_true.conj().T))
                    results["cov_nmse"].append(1.0 - nmse)  # Higher is better
            
            # Per-sample processing
            for i in range(batch_size):
                if sample_count >= max_samples:
                    break
                
                # Ground truth
                k_true = int(K[i].item())
                phi_gt = ptr[i, :k_true, 0].cpu().numpy()  # degrees
                theta_gt = ptr[i, :k_true, 1].cpu().numpy()
                r_gt = ptr[i, :k_true, 2].cpu().numpy()
                
                # Predictions (all K_MAX slots, Hungarian will match)
                def _to_deg_safe(t):
                    x = t if isinstance(t, np.ndarray) else t
                    max_abs = np.nanmax(np.abs(x))
                    return np.rad2deg(x) if max_abs <= (np.pi + 0.1) else x
                
                phi_all_deg = _to_deg_safe(phi_soft[i])
                theta_all_deg = _to_deg_safe(theta_soft[i])
                r_all = aux_ptr[i, 2*cfg.K_MAX:3*cfg.K_MAX]
                
                # K prediction
                k_pred = int(np.argmax(k_logits[i]) + 1) if k_logits is not None else k_true
                
                # Hungarian matching
                metrics = eval_scene_angles_ranges(phi_all_deg, theta_all_deg, r_all,
                                                    phi_gt, theta_gt, r_gt)
                
                # Store per-source errors
                for err_phi, err_theta, err_r in zip(metrics["phi_errors"], 
                                                      metrics["theta_errors"],
                                                      metrics["r_errors"]):
                    results["phi_errors"].append(err_phi)
                    results["theta_errors"].append(err_theta)
                    results["r_errors"].append(err_r)
                    results["snr"].append(float(snr[i].item()))
                    results["range"].append(float(np.mean(r_gt)))  # Average range for this sample
                    results["k_true"].append(k_true)
                    results["k_pred"].append(k_pred)
                
                sample_count += 1
            
            if (bi + 1) % 10 == 0:
                print(f"  Processed {sample_count} samples...", flush=True)
    
    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])
    
    print(f"[Diagnostics] Collected {sample_count} samples with {len(results['phi_errors'])} source measurements", flush=True)
    return results


def plot_error_vs_range(results, out_dir):
    """Plot angle and range errors vs true range (bins)"""
    ranges = results["range"]
    phi_err = results["phi_errors"]
    theta_err = results["theta_errors"]
    r_err = results["r_errors"]
    
    # Bin by range
    range_bins = np.linspace(ranges.min(), ranges.max(), 10)
    range_centers = 0.5 * (range_bins[:-1] + range_bins[1:])
    
    phi_med, theta_med, r_med = [], [], []
    phi_95, theta_95, r_95 = [], [], []
    
    for i in range(len(range_bins) - 1):
        mask = (ranges >= range_bins[i]) & (ranges < range_bins[i+1])
        if mask.sum() == 0:
            phi_med.append(np.nan)
            theta_med.append(np.nan)
            r_med.append(np.nan)
            phi_95.append(np.nan)
            theta_95.append(np.nan)
            r_95.append(np.nan)
            continue
        
        phi_med.append(np.median(phi_err[mask]))
        theta_med.append(np.median(theta_err[mask]))
        r_med.append(np.median(r_err[mask]))
        phi_95.append(np.percentile(phi_err[mask], 95))
        theta_95.append(np.percentile(theta_err[mask], 95))
        r_95.append(np.percentile(r_err[mask], 95))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(range_centers, phi_med, 'o-', label='φ median', linewidth=2)
    ax1.plot(range_centers, theta_med, 's-', label='θ median', linewidth=2)
    ax1.fill_between(range_centers, phi_med, phi_95, alpha=0.3)
    ax1.fill_between(range_centers, theta_med, theta_95, alpha=0.3)
    ax1.set_xlabel('True Range (m)', fontsize=12)
    ax1.set_ylabel('Angle Error (°)', fontsize=12)
    ax1.set_title('Angle Error vs Range', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(2.0, color='r', linestyle='--', alpha=0.5, label='2° target')
    
    ax2.plot(range_centers, r_med, 'o-', color='green', linewidth=2, label='Range median')
    ax2.fill_between(range_centers, r_med, r_95, alpha=0.3, color='green')
    ax2.set_xlabel('True Range (m)', fontsize=12)
    ax2.set_ylabel('Range Error (m)', fontsize=12)
    ax2.set_title('Range Error vs Range', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_range.png", dpi=150)
    print(f"  Saved: error_vs_range.png", flush=True)
    plt.close()


def plot_error_vs_snr(results, out_dir):
    """Plot angle and range errors vs SNR (bins)"""
    snr = results["snr"]
    phi_err = results["phi_errors"]
    theta_err = results["theta_errors"]
    r_err = results["r_errors"]
    
    # Bin by SNR
    snr_bins = np.linspace(snr.min(), snr.max(), 10)
    snr_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])
    
    phi_med, theta_med, r_med = [], [], []
    phi_95, theta_95, r_95 = [], [], []
    
    for i in range(len(snr_bins) - 1):
        mask = (snr >= snr_bins[i]) & (snr < snr_bins[i+1])
        if mask.sum() == 0:
            phi_med.append(np.nan)
            theta_med.append(np.nan)
            r_med.append(np.nan)
            phi_95.append(np.nan)
            theta_95.append(np.nan)
            r_95.append(np.nan)
            continue
        
        phi_med.append(np.median(phi_err[mask]))
        theta_med.append(np.median(theta_err[mask]))
        r_med.append(np.median(r_err[mask]))
        phi_95.append(np.percentile(phi_err[mask], 95))
        theta_95.append(np.percentile(theta_err[mask], 95))
        r_95.append(np.percentile(r_err[mask], 95))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(snr_centers, phi_med, 'o-', label='φ median', linewidth=2)
    ax1.plot(snr_centers, theta_med, 's-', label='θ median', linewidth=2)
    ax1.fill_between(snr_centers, phi_med, phi_95, alpha=0.3)
    ax1.fill_between(snr_centers, theta_med, theta_95, alpha=0.3)
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('Angle Error (°)', fontsize=12)
    ax1.set_title('Angle Error vs SNR', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(2.0, color='r', linestyle='--', alpha=0.5, label='2° target')
    
    ax2.plot(snr_centers, r_med, 'o-', color='green', linewidth=2, label='Range median')
    ax2.fill_between(snr_centers, r_med, r_95, alpha=0.3, color='green')
    ax2.set_xlabel('SNR (dB)', fontsize=12)
    ax2.set_ylabel('Range Error (m)', fontsize=12)
    ax2.set_title('Range Error vs SNR', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_snr.png", dpi=150)
    print(f"  Saved: error_vs_snr.png", flush=True)
    plt.close()


def plot_error_vs_k(results, out_dir):
    """Plot angle and range errors vs K (number of sources)"""
    k_true = results["k_true"]
    phi_err = results["phi_errors"]
    theta_err = results["theta_errors"]
    r_err = results["r_errors"]
    
    k_values = np.unique(k_true)
    phi_med, theta_med, r_med = [], [], []
    phi_95, theta_95, r_95 = [], [], []
    k_acc = []
    
    for k in k_values:
        mask = (k_true == k)
        if mask.sum() == 0:
            continue
        
        phi_med.append(np.median(phi_err[mask]))
        theta_med.append(np.median(theta_err[mask]))
        r_med.append(np.median(r_err[mask]))
        phi_95.append(np.percentile(phi_err[mask], 95))
        theta_95.append(np.percentile(theta_err[mask], 95))
        r_95.append(np.percentile(r_err[mask], 95))
        
        # K accuracy for this K value
        k_pred_for_k = results["k_pred"][mask]
        k_acc.append(np.mean(k_pred_for_k == k))
    
    # Plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1.plot(k_values, phi_med, 'o-', label='φ median', linewidth=2)
    ax1.fill_between(k_values, phi_med, phi_95, alpha=0.3)
    ax1.set_xlabel('True K', fontsize=12)
    ax1.set_ylabel('φ Error (°)', fontsize=12)
    ax1.set_title('Azimuth Error vs K', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(2.0, color='r', linestyle='--', alpha=0.5)
    
    ax2.plot(k_values, theta_med, 's-', color='orange', linewidth=2, label='θ median')
    ax2.fill_between(k_values, theta_med, theta_95, alpha=0.3, color='orange')
    ax2.set_xlabel('True K', fontsize=12)
    ax2.set_ylabel('θ Error (°)', fontsize=12)
    ax2.set_title('Elevation Error vs K', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(2.0, color='r', linestyle='--', alpha=0.5)
    
    ax3.plot(k_values, r_med, 'o-', color='green', linewidth=2, label='Range median')
    ax3.fill_between(k_values, r_med, r_95, alpha=0.3, color='green')
    ax3.set_xlabel('True K', fontsize=12)
    ax3.set_ylabel('Range Error (m)', fontsize=12)
    ax3.set_title('Range Error vs K', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4.bar(k_values, k_acc, color='purple', alpha=0.7, width=0.6)
    ax4.set_xlabel('True K', fontsize=12)
    ax4.set_ylabel('K Accuracy', fontsize=12)
    ax4.set_title('K Estimation Accuracy', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(0.7, color='r', linestyle='--', alpha=0.5, label='70% target')
    ax4.legend()
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_k.png", dpi=150)
    print(f"  Saved: error_vs_k.png", flush=True)
    plt.close()


def plot_cov_nmse_vs_snr(results, out_dir):
    """Plot covariance quality (1 - NMSE) vs SNR"""
    if len(results["cov_nmse"]) == 0:
        print("  Skipping cov_nmse plot (no data)", flush=True)
        return
    
    snr = results["snr"][:len(results["cov_nmse"])]  # Match lengths
    cov_quality = results["cov_nmse"]
    
    # Bin by SNR
    snr_bins = np.linspace(snr.min(), snr.max(), 10)
    snr_centers = 0.5 * (snr_bins[:-1] + snr_bins[1:])
    
    cov_med = []
    
    for i in range(len(snr_bins) - 1):
        mask = (snr >= snr_bins[i]) & (snr < snr_bins[i+1])
        if mask.sum() == 0:
            cov_med.append(np.nan)
        else:
            cov_med.append(np.median(cov_quality[mask]))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(snr_centers, cov_med, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Covariance Quality (1 - NMSE)', fontsize=12)
    ax.set_title('Covariance Learning vs SNR', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0.99, color='g', linestyle='--', alpha=0.5, label='0.99 target (excellent)')
    ax.axhline(0.95, color='orange', linestyle='--', alpha=0.5, label='0.95 target (good)')
    ax.legend()
    ax.set_ylim([0.85, 1.0])
    
    plt.tight_layout()
    plt.savefig(out_dir / "cov_nmse_vs_snr.png", dpi=150)
    print(f"  Saved: cov_nmse_vs_snr.png", flush=True)
    plt.close()


def generate_summary_report(results, out_dir):
    """Generate text summary with key statistics"""
    report = []
    report.append("=" * 80)
    report.append("PHASE 2 DIAGNOSTIC REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"  Samples analyzed: {len(np.unique(results['snr']))}")
    report.append(f"  Source measurements: {len(results['phi_errors'])}")
    report.append(f"  SNR range: {results['snr'].min():.1f} to {results['snr'].max():.1f} dB")
    report.append(f"  Range span: {results['range'].min():.2f} to {results['range'].max():.2f} m")
    report.append("")
    
    # Angle errors
    report.append("ANGLE ERRORS (all sources)")
    report.append("-" * 80)
    report.append(f"  φ median:  {np.median(results['phi_errors']):.3f}°")
    report.append(f"  φ 95th:    {np.percentile(results['phi_errors'], 95):.3f}°")
    report.append(f"  θ median:  {np.median(results['theta_errors']):.3f}°")
    report.append(f"  θ 95th:    {np.percentile(results['theta_errors'], 95):.3f}°")
    report.append("")
    
    # Range errors
    report.append("RANGE ERRORS (all sources)")
    report.append("-" * 80)
    report.append(f"  r median:  {np.median(results['r_errors']):.4f} m")
    report.append(f"  r 95th:    {np.percentile(results['r_errors'], 95):.4f} m")
    report.append("")
    
    # K accuracy
    k_acc = np.mean(results["k_true"] == results["k_pred"])
    report.append("K ESTIMATION")
    report.append("-" * 80)
    report.append(f"  Overall K accuracy: {k_acc:.3f} ({k_acc*100:.1f}%)")
    report.append("")
    
    # Covariance quality
    if len(results["cov_nmse"]) > 0:
        report.append("COVARIANCE QUALITY (1 - NMSE)")
        report.append("-" * 80)
        report.append(f"  Median: {np.median(results['cov_nmse']):.4f}")
        report.append(f"  Mean:   {np.mean(results['cov_nmse']):.4f}")
        report.append("")
    
    # Phase 3 decision
    report.append("=" * 80)
    report.append("PHASE 3 GATE DECISION")
    report.append("=" * 80)
    
    phi_med = np.median(results['phi_errors'])
    theta_med = np.median(results['theta_errors'])
    
    # Check if error increases at short range
    short_range_mask = results['range'] < 3.0
    long_range_mask = results['range'] >= 3.0
    
    if short_range_mask.sum() > 10 and long_range_mask.sum() > 10:
        phi_short = np.median(results['phi_errors'][short_range_mask])
        phi_long = np.median(results['phi_errors'][long_range_mask])
        short_range_spike = (phi_short > 1.5 * phi_long) and (phi_short > 2.0)
        
        report.append(f"  Short-range (<3m) φ error: {phi_short:.2f}°")
        report.append(f"  Long-range (≥3m) φ error:  {phi_long:.2f}°")
        report.append("")
    else:
        short_range_spike = False
    
    if phi_med < 2.0 and theta_med < 2.0:
        report.append("✅ STATUS: EXCELLENT! Sub-2° accuracy achieved.")
        report.append("   → No need for Phase 3 (near-field focusing).")
        report.append("   → Consider declaring victory and moving to deployment!")
    elif phi_med < 5.0 and theta_med < 3.0:
        if short_range_spike:
            report.append("⚠️  STATUS: GOOD, but short-range spike detected.")
            report.append("   → RECOMMEND Phase 3: Near-field focusing with 5 range candidates.")
            report.append("   → Expected improvement: 1.5-2x at short range.")
        else:
            report.append("✅ STATUS: GOOD! Angle errors in target range.")
            report.append("   → Phase 3 optional; consider other optimizations first.")
    else:
        report.append("❌ STATUS: Angles still high (>5°).")
        report.append("   → CHECK: Is covariance quality good (>0.95)?")
        if len(results["cov_nmse"]) > 0:
            cov_q = np.median(results["cov_nmse"])
            if cov_q < 0.95:
                report.append(f"   → Covariance quality: {cov_q:.3f} (LOW!) - focus on training first!")
            else:
                report.append(f"   → Covariance quality: {cov_q:.3f} (OK)")
                report.append("   → RECOMMEND Phase 3 or revisit model architecture.")
    
    report.append("")
    report.append("=" * 80)
    
    # Write to file
    report_text = "\n".join(report)
    with open(out_dir / "summary_report.txt", "w") as f:
        f.write(report_text)
    
    # Also print to console
    print("\n" + report_text)
    print(f"\n[Diagnostics] Report saved to: {out_dir / 'summary_report.txt'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Diagnostic Plots")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of validation samples to analyze")
    parser.add_argument("--out-dir", type=str, default="diagnostics", help="Output directory for plots")
    args = parser.parse_args()
    
    # Setup
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print(f"PHASE 2 DIAGNOSTICS")
    print(f"{'='*80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples:    {args.n_samples}")
    print(f"Output:     {out_dir}")
    print(f"{'='*80}\n", flush=True)
    
    # Load model
    print("[Diagnostics] Loading model...", flush=True)
    model = HybridModel(cfg, mdl_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    print("[Diagnostics] Checkpoint loaded!", flush=True)
    
    # Create trainer for inference utilities
    trainer = Trainer(model, device=device)
    
    # Load validation data
    print("[Diagnostics] Loading validation data...", flush=True)
    val_loader = build_val_loader(n_samples=args.n_samples)
    
    # Collect diagnostics
    results = collect_diagnostics(trainer, val_loader, max_samples=args.n_samples)
    
    # Generate plots
    print(f"\n[Diagnostics] Generating plots...", flush=True)
    plot_error_vs_range(results, out_dir)
    plot_error_vs_snr(results, out_dir)
    plot_error_vs_k(results, out_dir)
    plot_cov_nmse_vs_snr(results, out_dir)
    
    # Generate summary report
    generate_summary_report(results, out_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ DIAGNOSTICS COMPLETE!")
    print(f"{'='*80}")
    print(f"All outputs saved to: {out_dir}")
    print(f"{'='*80}\n", flush=True)


if __name__ == "__main__":
    main()


