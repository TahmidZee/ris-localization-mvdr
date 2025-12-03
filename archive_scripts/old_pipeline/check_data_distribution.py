#!/usr/bin/env python3
"""
Data Distribution Diagnostic Tool

Checks if training and validation datasets have matching distributions
for key variables: K, SNR, angles (phi, theta), ranges.

Usage:
    python check_data_distribution.py
    python check_data_distribution.py --train_dir path/to/train --val_dir path/to/val
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import Counter
import sys


class SimpleShardLoader:
    """Simple loader that directly reads NPZ shards without complex dependencies"""
    def __init__(self, shard_dir):
        self.shard_dir = Path(shard_dir)
        self.shard_files = sorted(list(self.shard_dir.glob("shard_*.npz")))
        
        if not self.shard_files:
            raise FileNotFoundError(f"No shard_*.npz files found in {shard_dir}")
        
        # Count total samples
        self.shard_lengths = []
        for sf in self.shard_files:
            with np.load(sf) as data:
                self.shard_lengths.append(len(data["K"]))
        
        self.cumulative_lengths = np.cumsum([0] + self.shard_lengths)
        self.total_samples = self.cumulative_lengths[-1]
        
        print(f"   Loaded {len(self.shard_files)} shards, {self.total_samples} samples")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Find which shard
        shard_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[shard_idx]
        
        # Load from shard
        with np.load(self.shard_files[shard_idx]) as data:
            sample = {
                "K": int(data["K"][local_idx]),
                "ptr": data["ptr"][local_idx],  # [3*K_MAX]
            }
            
            # Optional fields
            if "snr_db" in data or "snr" in data:
                sample["snr_db"] = float(data.get("snr_db", data.get("snr", [0]))[local_idx])
            
            return sample


# Default config values (fallback if cfg not available)
class DefaultConfig:
    K_MAX = 5
    ANGLE_RANGE_PHI = np.pi / 3.0    # ±60°
    ANGLE_RANGE_THETA = np.pi / 6.0  # ±30°
    RANGE_R = (0.5, 10.0)

try:
    from configs import cfg
except:
    print("⚠️  Could not import configs, using defaults")
    cfg = DefaultConfig()

def analyze_dataset_distribution(dataset, name="Dataset", max_samples=None):
    """
    Analyze distribution of key variables in a dataset.
    
    Args:
        dataset: SimpleShardLoader or compatible dataset instance
        name: Name for display
        max_samples: Maximum number of samples to analyze (None = all)
    
    Returns:
        dict with distribution statistics
    """
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    print(f"\n{'='*60}")
    print(f"Analyzing {name}: {n_samples} samples")
    print(f"{'='*60}")
    
    # Collect statistics
    K_values = []
    snr_values = []
    phi_values = []
    theta_values = []
    r_values = []
    
    print(f"Loading samples... ", end='', flush=True)
    for i in range(n_samples):
        if i % 1000 == 0 and i > 0:
            print(f"{i}...", end='', flush=True)
        
        sample = dataset[i]
        K = int(sample["K"])
        K_values.append(K)
        
        # SNR
        if "snr_db" in sample:
            snr_values.append(float(sample["snr_db"]))
        
        # Angles and ranges
        ptr = sample["ptr"]  # [3*K_MAX] format: [phi1..phiK, theta1..thetaK, r1..rK]
        K_MAX = len(ptr) // 3
        
        phi = ptr[:K_MAX][:K]
        theta = ptr[K_MAX:2*K_MAX][:K]
        r = ptr[2*K_MAX:][:K]
        
        phi_values.extend(phi.tolist() if hasattr(phi, 'tolist') else list(phi))
        theta_values.extend(theta.tolist() if hasattr(theta, 'tolist') else list(theta))
        r_values.extend(r.tolist() if hasattr(r, 'tolist') else list(r))
    
    print(" Done!\n")
    
    # Convert to numpy
    K_values = np.array(K_values)
    snr_values = np.array(snr_values) if snr_values else None
    phi_values = np.array(phi_values)
    theta_values = np.array(theta_values)
    r_values = np.array(r_values)
    
    # Print statistics
    print(f"📊 K Distribution:")
    k_counts = Counter(K_values)
    for k in sorted(k_counts.keys()):
        pct = 100 * k_counts[k] / len(K_values)
        print(f"   K={k}: {k_counts[k]:6d} samples ({pct:5.2f}%)")
    
    if snr_values is not None and len(snr_values) > 0:
        print(f"\n📡 SNR Distribution (dB):")
        print(f"   Mean:   {np.mean(snr_values):7.2f}")
        print(f"   Std:    {np.std(snr_values):7.2f}")
        print(f"   Min:    {np.min(snr_values):7.2f}")
        print(f"   Max:    {np.max(snr_values):7.2f}")
        print(f"   Median: {np.median(snr_values):7.2f}")
    
    print(f"\n📐 Azimuth Angle φ Distribution (rad):")
    print(f"   Mean:   {np.mean(phi_values):7.4f}")
    print(f"   Std:    {np.std(phi_values):7.4f}")
    print(f"   Min:    {np.min(phi_values):7.4f}")
    print(f"   Max:    {np.max(phi_values):7.4f}")
    print(f"   Range:  [{np.min(phi_values):.4f}, {np.max(phi_values):.4f}]")
    print(f"   Expected: ±{cfg.ANGLE_RANGE_PHI:.4f} rad (±{np.rad2deg(cfg.ANGLE_RANGE_PHI):.1f}°)")
    
    print(f"\n📐 Elevation Angle θ Distribution (rad):")
    print(f"   Mean:   {np.mean(theta_values):7.4f}")
    print(f"   Std:    {np.std(theta_values):7.4f}")
    print(f"   Min:    {np.min(theta_values):7.4f}")
    print(f"   Max:    {np.max(theta_values):7.4f}")
    print(f"   Range:  [{np.min(theta_values):.4f}, {np.max(theta_values):.4f}]")
    print(f"   Expected: ±{cfg.ANGLE_RANGE_THETA:.4f} rad (±{np.rad2deg(cfg.ANGLE_RANGE_THETA):.1f}°)")
    
    print(f"\n📏 Range Distribution (m):")
    print(f"   Mean:   {np.mean(r_values):7.2f}")
    print(f"   Std:    {np.std(r_values):7.2f}")
    print(f"   Min:    {np.min(r_values):7.2f}")
    print(f"   Max:    {np.max(r_values):7.2f}")
    print(f"   Expected: [{cfg.RANGE_R[0]:.1f}, {cfg.RANGE_R[1]:.1f}] m")
    
    return {
        'K': K_values,
        'snr': snr_values,
        'phi': phi_values,
        'theta': theta_values,
        'r': r_values,
        'name': name
    }


def compare_distributions(train_stats, val_stats):
    """
    Compare two distributions and flag significant differences.
    
    Args:
        train_stats: dict from analyze_dataset_distribution
        val_stats: dict from analyze_dataset_distribution
    """
    print(f"\n{'='*60}")
    print("🔍 DISTRIBUTION COMPARISON")
    print(f"{'='*60}\n")
    
    issues = []
    
    # Compare K distributions
    print("📊 K Distribution Comparison:")
    train_K = train_stats['K']
    val_K = val_stats['K']
    
    train_k_counts = Counter(train_K)
    val_k_counts = Counter(val_K)
    
    all_k = sorted(set(train_k_counts.keys()) | set(val_k_counts.keys()))
    
    print(f"   {'K':<5} {'Train %':>10} {'Val %':>10} {'Diff':>10}")
    print(f"   {'-'*40}")
    for k in all_k:
        train_pct = 100 * train_k_counts.get(k, 0) / len(train_K)
        val_pct = 100 * val_k_counts.get(k, 0) / len(val_K)
        diff = abs(train_pct - val_pct)
        
        flag = "⚠️" if diff > 5.0 else "✓"
        print(f"   {k:<5} {train_pct:9.2f}% {val_pct:9.2f}% {diff:9.2f}% {flag}")
        
        if diff > 5.0:
            issues.append(f"K={k}: distribution differs by {diff:.1f}%")
    
    # Compare SNR
    if train_stats['snr'] is not None and val_stats['snr'] is not None:
        print(f"\n📡 SNR Distribution Comparison:")
        train_snr = train_stats['snr']
        val_snr = val_stats['snr']
        
        train_mean, train_std = np.mean(train_snr), np.std(train_snr)
        val_mean, val_std = np.mean(val_snr), np.std(val_snr)
        
        mean_diff = abs(train_mean - val_mean)
        std_diff = abs(train_std - val_std)
        
        print(f"   {'Metric':<15} {'Train':>10} {'Val':>10} {'Diff':>10}")
        print(f"   {'-'*50}")
        print(f"   {'Mean (dB)':<15} {train_mean:10.2f} {val_mean:10.2f} {mean_diff:10.2f} {'⚠️' if mean_diff > 2.0 else '✓'}")
        print(f"   {'Std (dB)':<15} {train_std:10.2f} {val_std:10.2f} {std_diff:10.2f} {'⚠️' if std_diff > 1.0 else '✓'}")
        
        if mean_diff > 2.0:
            issues.append(f"SNR mean differs by {mean_diff:.1f} dB")
        if std_diff > 1.0:
            issues.append(f"SNR std differs by {std_diff:.1f} dB")
    
    # Compare angles
    print(f"\n📐 Angle Distribution Comparison:")
    
    for angle_name, train_angle, val_angle in [
        ('Azimuth φ', train_stats['phi'], val_stats['phi']),
        ('Elevation θ', train_stats['theta'], val_stats['theta'])
    ]:
        train_mean, train_std = np.mean(train_angle), np.std(train_angle)
        val_mean, val_std = np.mean(val_angle), np.std(val_angle)
        
        mean_diff = abs(train_mean - val_mean)
        std_diff = abs(train_std - val_std)
        
        print(f"\n   {angle_name}:")
        print(f"   {'Metric':<15} {'Train':>10} {'Val':>10} {'Diff':>10}")
        print(f"   {'-'*50}")
        print(f"   {'Mean (rad)':<15} {train_mean:10.4f} {val_mean:10.4f} {mean_diff:10.4f} {'⚠️' if mean_diff > 0.1 else '✓'}")
        print(f"   {'Std (rad)':<15} {train_std:10.4f} {val_std:10.4f} {std_diff:10.4f} {'⚠️' if std_diff > 0.1 else '✓'}")
        
        if mean_diff > 0.1:
            issues.append(f"{angle_name} mean differs by {mean_diff:.3f} rad ({np.rad2deg(mean_diff):.1f}°)")
        if std_diff > 0.1:
            issues.append(f"{angle_name} std differs by {std_diff:.3f} rad ({np.rad2deg(std_diff):.1f}°)")
    
    # Compare ranges
    print(f"\n📏 Range Distribution Comparison:")
    train_r = train_stats['r']
    val_r = val_stats['r']
    
    train_mean, train_std = np.mean(train_r), np.std(train_r)
    val_mean, val_std = np.mean(val_r), np.std(val_r)
    
    mean_diff = abs(train_mean - val_mean)
    std_diff = abs(train_std - val_std)
    
    print(f"   {'Metric':<15} {'Train':>10} {'Val':>10} {'Diff':>10}")
    print(f"   {'-'*50}")
    print(f"   {'Mean (m)':<15} {train_mean:10.2f} {val_mean:10.2f} {mean_diff:10.2f} {'⚠️' if mean_diff > 0.5 else '✓'}")
    print(f"   {'Std (m)':<15} {train_std:10.2f} {val_std:10.2f} {std_diff:10.2f} {'⚠️' if std_diff > 0.5 else '✓'}")
    
    if mean_diff > 0.5:
        issues.append(f"Range mean differs by {mean_diff:.2f} m")
    if std_diff > 0.5:
        issues.append(f"Range std differs by {std_diff:.2f} m")
    
    # Summary
    print(f"\n{'='*60}")
    if issues:
        print("⚠️  DISTRIBUTION MISMATCH DETECTED!")
        print(f"{'='*60}")
        print("\nIssues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n⚠️  Train/Val distributions differ significantly!")
        print("   This may cause overfitting and high validation loss.")
        print("   Consider:")
        print("   - Reshuffling and splitting data evenly")
        print("   - Using stratified sampling by K and SNR")
        print("   - Checking if shards have temporal/spatial biases")
    else:
        print("✅ Distributions match well!")
        print(f"{'='*60}")
        print("\nNo significant distribution mismatches detected.")
        print("Train and validation sets appear to be from the same distribution.")
    
    return issues


def plot_distributions(train_stats, val_stats, save_path=None):
    """
    Create visualization plots comparing distributions.
    
    Args:
        train_stats: dict from analyze_dataset_distribution
        val_stats: dict from analyze_dataset_distribution
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training vs Validation Distribution Comparison', fontsize=16)
    
    # K distribution
    ax = axes[0, 0]
    train_K = train_stats['K']
    val_K = val_stats['K']
    k_bins = np.arange(0.5, cfg.K_MAX + 1.5, 1)
    ax.hist([train_K, val_K], bins=k_bins, label=['Train', 'Val'], alpha=0.7)
    ax.set_xlabel('K (Number of Sources)')
    ax.set_ylabel('Count')
    ax.set_title('K Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # SNR distribution
    ax = axes[0, 1]
    if train_stats['snr'] is not None and val_stats['snr'] is not None:
        ax.hist([train_stats['snr'], val_stats['snr']], bins=30, label=['Train', 'Val'], alpha=0.7)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Count')
        ax.set_title('SNR Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No SNR data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('SNR Distribution (N/A)')
    
    # Azimuth angle
    ax = axes[0, 2]
    ax.hist([train_stats['phi'], val_stats['phi']], bins=50, label=['Train', 'Val'], alpha=0.7)
    ax.set_xlabel('Azimuth φ (rad)')
    ax.set_ylabel('Count')
    ax.set_title('Azimuth Distribution')
    ax.axvline(-cfg.ANGLE_RANGE_PHI, color='r', linestyle='--', alpha=0.5, label='Expected range')
    ax.axvline(cfg.ANGLE_RANGE_PHI, color='r', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Elevation angle
    ax = axes[1, 0]
    ax.hist([train_stats['theta'], val_stats['theta']], bins=50, label=['Train', 'Val'], alpha=0.7)
    ax.set_xlabel('Elevation θ (rad)')
    ax.set_ylabel('Count')
    ax.set_title('Elevation Distribution')
    ax.axvline(-cfg.ANGLE_RANGE_THETA, color='r', linestyle='--', alpha=0.5, label='Expected range')
    ax.axvline(cfg.ANGLE_RANGE_THETA, color='r', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Range
    ax = axes[1, 1]
    ax.hist([train_stats['r'], val_stats['r']], bins=50, label=['Train', 'Val'], alpha=0.7)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Count')
    ax.set_title('Range Distribution')
    ax.axvline(cfg.RANGE_R[0], color='r', linestyle='--', alpha=0.5, label='Expected range')
    ax.axvline(cfg.RANGE_R[1], color='r', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Summary statistics table
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Summary Statistics\n\n"
    summary_text += f"Train samples: {len(train_stats['K'])}\n"
    summary_text += f"Val samples: {len(val_stats['K'])}\n\n"
    
    summary_text += "K mean:\n"
    summary_text += f"  Train: {np.mean(train_stats['K']):.2f}\n"
    summary_text += f"  Val:   {np.mean(val_stats['K']):.2f}\n\n"
    
    if train_stats['snr'] is not None:
        summary_text += "SNR mean (dB):\n"
        summary_text += f"  Train: {np.mean(train_stats['snr']):.2f}\n"
        summary_text += f"  Val:   {np.mean(val_stats['snr']):.2f}\n\n"
    
    summary_text += f"φ range (°):\n"
    summary_text += f"  Train: [{np.rad2deg(np.min(train_stats['phi'])):.1f}, {np.rad2deg(np.max(train_stats['phi'])):.1f}]\n"
    summary_text += f"  Val:   [{np.rad2deg(np.min(val_stats['phi'])):.1f}, {np.rad2deg(np.max(val_stats['phi'])):.1f}]\n\n"
    
    summary_text += f"Range (m):\n"
    summary_text += f"  Train: [{np.min(train_stats['r']):.2f}, {np.max(train_stats['r']):.2f}]\n"
    summary_text += f"  Val:   [{np.min(val_stats['r']):.2f}, {np.max(val_stats['r']):.2f}]"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Check train/val data distribution match")
    parser.add_argument('--train_dir', type=str, default=None,
                       help='Training data directory (default: from cfg.DATA_SHARDS_DIR/train)')
    parser.add_argument('--val_dir', type=str, default=None,
                       help='Validation data directory (default: from cfg.DATA_SHARDS_DIR/val)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to analyze per dataset (default: all)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate distribution plots')
    parser.add_argument('--save_plot', type=str, default='distribution_comparison.png',
                       help='Path to save plot (default: distribution_comparison.png)')
    
    args = parser.parse_args()
    
    # Determine data directories
    if args.train_dir is None or args.val_dir is None:
        # Try common locations
        possible_roots = [
            Path(getattr(cfg, "DATA_SHARDS_DIR", "")),
            Path("results_final/data/shards"),
            Path("results_final_L16_12x12/data/shards"),
            Path("data/shards"),
        ]
        
        data_root = None
        for root in possible_roots:
            if root.exists() and (root / "train").exists():
                data_root = root
                break
        
        if data_root is None:
            print("\n❌ Error: Could not find data shards directory.")
            print("Please specify with --train_dir and --val_dir")
            return
        
        train_dir = Path(args.train_dir) if args.train_dir else data_root / "train"
        val_dir = Path(args.val_dir) if args.val_dir else data_root / "val"
    else:
        train_dir = Path(args.train_dir)
        val_dir = Path(args.val_dir)
    
    print("="*60)
    print("DATA DISTRIBUTION DIAGNOSTIC TOOL")
    print("="*60)
    print(f"\nTrain directory: {train_dir}")
    print(f"Val directory:   {val_dir}")
    
    # Check directories exist
    if not train_dir.exists():
        print(f"\n❌ Error: Train directory not found: {train_dir}")
        return
    if not val_dir.exists():
        print(f"\n❌ Error: Val directory not found: {val_dir}")
        return
    
    # Load datasets
    print("\n📂 Loading datasets...")
    try:
        train_dataset = SimpleShardLoader(train_dir)
        val_dataset = SimpleShardLoader(val_dir)
    except Exception as e:
        print(f"\n❌ Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Val dataset:   {len(val_dataset)} samples")
    
    # Analyze distributions
    train_stats = analyze_dataset_distribution(
        train_dataset, 
        name="Training Set", 
        max_samples=args.max_samples
    )
    
    val_stats = analyze_dataset_distribution(
        val_dataset, 
        name="Validation Set", 
        max_samples=args.max_samples
    )
    
    # Compare
    issues = compare_distributions(train_stats, val_stats)
    
    # Plot if requested
    if args.plot:
        print("\n📊 Generating distribution plots...")
        plot_distributions(train_stats, val_stats, save_path=args.save_plot if args.save_plot else None)
    
    # Exit code
    if issues:
        print("\n⚠️  Exiting with warning (distribution mismatch detected)")
        sys.exit(1)
    else:
        print("\n✅ Exiting with success (distributions match)")
        sys.exit(0)


if __name__ == "__main__":
    main()

