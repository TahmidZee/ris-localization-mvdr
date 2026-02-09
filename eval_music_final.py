#!/usr/bin/env python3
"""
OFFLINE MUSIC EVALUATION SCRIPT

This script runs the full MUSIC-based evaluation pipeline on trained checkpoints.
Use this ONLY for final evaluation, NOT during training/HPO.

REVIEWER-PROOF FIX (2026-02-09): Defaults to TEST split, not validation.
Final reported numbers MUST come from test data that was never used for
model selection, early stopping, or HPO. Use --split val only for debugging.

Usage:
    python eval_music_final.py --checkpoint results_M64_N256_L64/checkpoints/best.pt
    python eval_music_final.py --checkpoint results_M64_N256_L64/checkpoints/best.pt --n_samples 2000
    python eval_music_final.py --checkpoint results_M64_N256_L64/checkpoints/best.pt --split val  # debug only

The script will:
1. Load the checkpoint
2. Run full 2.5D GPU MUSIC with hybrid covariance
3. Compute Hungarian-matched angle/range errors
4. Compare to MDL baseline
5. Report comprehensive metrics
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
from ris_pytorch_pipeline.dataset import ShardNPZDataset


def _resolve_eval_dir(split: str) -> Path:
    """Resolve the data directory for the requested split."""
    split = split.lower().strip()
    if split == "test":
        d = Path(getattr(cfg, "DATA_SHARDS_TEST", Path(cfg.DATA_SHARDS_DIR) / "test"))
    elif split == "val":
        d = Path(getattr(cfg, "DATA_SHARDS_VAL", Path(cfg.DATA_SHARDS_DIR) / "val"))
    elif split == "train":
        d = Path(getattr(cfg, "DATA_SHARDS_TRAIN", Path(cfg.DATA_SHARDS_DIR) / "train"))
    else:
        raise ValueError(f"Unknown split '{split}'. Use test, val, or train.")
    if not d.exists() or not list(d.glob("*.npz")):
        raise FileNotFoundError(
            f"No shards found for split '{split}' at {d}. "
            f"Run `pregen-split` to create train/val/test splits."
        )
    return d


def main():
    parser = argparse.ArgumentParser(description="Offline MUSIC evaluation for trained checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file (e.g., best.pt)")
    parser.add_argument("--split", type=str, default="test",
                       choices=["test", "val", "train"],
                       help="Data split to evaluate on (default: test). "
                            "Use 'test' for final paper numbers, 'val' for debugging only.")
    parser.add_argument("--n_samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all)")
    parser.add_argument("--max_batches", type=int, default=None,
                       help="Maximum batches (default: all)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results (default: auto-generate)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Verify checkpoint exists
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    # Resolve data directory
    eval_dir = _resolve_eval_dir(args.split)
    
    print("=" * 60)
    print("OFFLINE MUSIC EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Data split: {args.split.upper()} ({eval_dir})")
    if args.split != "test":
        print(f"  WARNING: Using '{args.split}' split. For paper numbers, use --split test")
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
    state_dict = torch.load(ckpt_path, map_location=t.device, weights_only=False)
    # Handle both formats: raw state_dict or {"model": state_dict}
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    t.model.load_state_dict(state_dict, strict=False)
    t.model.eval()
    print(f"      Loaded model with {sum(p.numel() for p in t.model.parameters()):,} parameters")
    
    # Build evaluation loader from the requested split
    print(f"[3/4] Building {args.split} loader...")
    ds_full = ShardNPZDataset(eval_dir)
    n_cap = args.n_samples
    if n_cap is not None:
        n_cap = min(n_cap, len(ds_full))
        idx = np.random.RandomState(args.seed).permutation(len(ds_full))[:n_cap]
        from torch.utils.data import Subset
        ds_eval = Subset(ds_full, idx.tolist())
    else:
        ds_eval = ds_full
    
    # Build GPU-cached loader for the eval split
    gds = t._aggregate_cpu_then_gpu(ds_eval)
    bs = int(getattr(mdl_cfg, "BATCH_SIZE", 64))
    from torch.utils.data import DataLoader
    eval_loader = DataLoader(gds, batch_size=bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)
    print(f"      {args.split} loader: {len(eval_loader)} batches ({len(ds_eval)} samples)")
    
    # Run MUSIC-based evaluation
    print("[4/4] Running MUSIC-based evaluation...")
    print("      (This may take a while - full MUSIC pipeline is running)")
    start_time = time.time()
    
    max_batches = args.max_batches or len(eval_loader)
    metrics = t._eval_hungarian_metrics(eval_loader, max_batches=max_batches)
    
    elapsed = time.time() - start_time
    print(f"      Evaluation completed in {elapsed:.1f}s")
    print()
    
    # Print results
    print("=" * 60)
    print(f"RESULTS (split={args.split.upper()})")
    print("=" * 60)
    
    if metrics is None:
        print("Evaluation failed - no metrics returned")
        sys.exit(1)
    
    print(f"\nK Estimation (blind MDL):")
    print(f"   K accuracy (MDL):  {metrics.get('k_mdl_acc', 0.0):.3f}")
    print(f"   K accuracy (blind):{metrics.get('k_acc_blind', 0.0):.3f}")
    
    print(f"\nLocalization (MUSIC + Hungarian):")
    print(f"   Azimuth  : median={metrics.get('med_phi', float('nan')):.2f}, "
          f"RMSE={metrics.get('rmse_phi_mean', float('nan')):.2f}")
    print(f"   Elevation: median={metrics.get('med_theta', float('nan')):.2f}, "
          f"RMSE={metrics.get('rmse_theta_mean', float('nan')):.2f}")
    print(f"   Range    : median={metrics.get('med_r', float('nan')):.2f}m, "
          f"RMSE={metrics.get('rmse_r_mean', float('nan')):.2f}m")
    
    print(f"\nSuccess Rate:")
    print(f"   Success rate: {metrics.get('success_rate', 0.0):.3f}")
    print(f"   (K correct via blind MDL AND all sources within tolerance)")
    
    print(f"\nSummary:")
    print(f"   Samples evaluated: {metrics.get('n_scenes', 'N/A')}")
    print(f"   High-SNR samples:  {metrics.get('high_snr_samples', 'N/A')}")
    
    # Save results to JSON
    output_path = args.output
    if output_path is None:
        output_path = ckpt_path.parent / f"eval_music_{args.split}_{ckpt_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    results = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": len(ds_eval),
        "max_batches": max_batches,
        "elapsed_seconds": elapsed,
        "metrics": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v 
                   for k, v in metrics.items() if v is not None}
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
