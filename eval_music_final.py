#!/usr/bin/env python3
"""
OFFLINE MUSIC EVALUATION SCRIPT

This script runs the full MUSIC-based evaluation pipeline on trained checkpoints.
Use this ONLY for final evaluation, NOT during training/HPO.

Usage:
    python eval_music_final.py --checkpoint results_final_L16_12x12/checkpoints/best.pt
    python eval_music_final.py --checkpoint results_final_L16_12x12/checkpoints/best.pt --n_samples 2000

The script will:
1. Load the checkpoint
2. Run full 2.5D GPU MUSIC with hybrid covariance
3. Compute Hungarian-matched angle/range errors
4. Compare to MDL baseline
5. Report comprehensive metrics

This is where the full MUSIC pipeline belongs - NOT in training/HPO loops.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ris_pytorch_pipeline.configs import cfg, mdl_cfg, set_seed
from ris_pytorch_pipeline.train import Trainer


def main():
    parser = argparse.ArgumentParser(description="Offline MUSIC evaluation for trained checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file (e.g., best.pt)")
    parser.add_argument("--n_samples", type=int, default=None,
                       help="Number of validation samples to evaluate (default: all)")
    parser.add_argument("--max_batches", type=int, default=None,
                       help="Maximum validation batches (default: all)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results (default: auto-generate)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Verify checkpoint exists
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("OFFLINE MUSIC EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print(f"N samples: {args.n_samples or 'all'}")
    print(f"Max batches: {args.max_batches or 'all'}")
    print()
    
    # CRITICAL: Enable MUSIC metrics for this evaluation
    cfg.VAL_PRIMARY = "k_loc"
    cfg.USE_MUSIC_METRICS_IN_VAL = True
    
    # Create trainer
    print("[1/4] Creating trainer...")
    t = Trainer(from_hpo=False)
    
    # Load checkpoint
    print(f"[2/4] Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=t.device)
    t.model.load_state_dict(state_dict)
    t.model.eval()
    print(f"      ✓ Loaded model with {sum(p.numel() for p in t.model.parameters()):,} parameters")
    
    # Build validation loader
    print("[3/4] Building validation loader...")
    n_val = args.n_samples
    _, va_loader = t._build_loaders_gpu_cache(n_train=1000, n_val=n_val)  # Minimal train, full val
    print(f"      ✓ Validation loader: {len(va_loader)} batches")
    
    # Run MUSIC-based evaluation
    print("[4/4] Running MUSIC-based evaluation...")
    print("      (This may take a while - full MUSIC pipeline is running)")
    start_time = time.time()
    
    max_batches = args.max_batches or len(va_loader)
    metrics = t._eval_hungarian_metrics(va_loader, max_batches=max_batches)
    
    elapsed = time.time() - start_time
    print(f"      ✓ Evaluation completed in {elapsed:.1f}s")
    print()
    
    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if metrics is None:
        print("❌ Evaluation failed - no metrics returned")
        sys.exit(1)
    
    print("\n📊 K Estimation:")
    print(f"   K accuracy (NN):   {metrics.get('k_acc', 0.0):.3f}")
    print(f"   K accuracy (MDL):  {metrics.get('k_mdl_acc', 0.0):.3f}")
    print(f"   K underestimation: {metrics.get('k_under', 0)}")
    print(f"   K overestimation:  {metrics.get('k_over', 0)}")
    
    print("\n📊 Localization (MUSIC + Hungarian):")
    print(f"   Azimuth (φ):     median={metrics.get('med_phi', float('nan')):.2f}°, "
          f"RMSE={metrics.get('rmse_phi_mean', float('nan')):.2f}°")
    print(f"   Elevation (θ):   median={metrics.get('med_theta', float('nan')):.2f}°, "
          f"RMSE={metrics.get('rmse_theta_mean', float('nan')):.2f}°")
    print(f"   Range (r):       median={metrics.get('med_r', float('nan')):.2f}m, "
          f"RMSE={metrics.get('rmse_r_mean', float('nan')):.2f}m")
    
    print("\n📊 Success Rate:")
    print(f"   Success rate: {metrics.get('success_rate', 0.0):.3f}")
    print(f"   (K correct AND all sources within tolerance)")
    
    print("\n📊 Summary:")
    print(f"   Samples evaluated: {metrics.get('n_scenes', 'N/A')}")
    print(f"   High-SNR samples:  {metrics.get('high_snr_samples', 'N/A')}")
    
    # Save results to JSON
    output_path = args.output
    if output_path is None:
        output_path = ckpt_path.parent / f"eval_music_{ckpt_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    results = {
        "checkpoint": str(ckpt_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": args.n_samples,
        "max_batches": max_batches,
        "elapsed_seconds": elapsed,
        "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                   for k, v in metrics.items() if v is not None}
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

