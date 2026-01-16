#!/usr/bin/env python3
"""
STAGE 1: HPO EXPLORATION (10% Data)
===================================
Optimizes K̂ + AoA/Range accuracy (not raw loss)
Date: 2025-11-26

TWO-STAGE HPO STRATEGY:
  Stage 1 (THIS SCRIPT): Explore hyperparameter space on 10% data
    - 50 trials, 20 epochs max, aggressive pruning
    - Goal: Find good regions, not final hyperparameters
    - Time: ~6-10 hours
  Stage 2 (manual): Train top configs on full data using `python -m ris_pytorch_pipeline.ris_pipeline train`
  Optional Stage 2b (MVDR): Train SpectrumRefiner using `python -m ris_pytorch_pipeline.ris_pipeline train-refiner`
"""

import sys
from pathlib import Path
import numpy as np

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from ris_pytorch_pipeline.hpo import run_hpo


def main():
    print("=" * 80)
    print("STAGE 1: HPO EXPLORATION (10% Data)")
    print("=" * 80)
    print()
    print("Strategy:")
    print("  - 50 trials on 10% subset (10K train / 1K val)")
    print("  - Max 20 epochs per trial (most pruned at 6-10)")
    print("  - Goal: Find promising hyperparameter regions")
    print("  - After this: Run Stage 2 with top 5 configs on full data")
    print()
    
    # Stage 1 Configuration (Exploration)
    N_TRIALS = 50          # 40-60 trials for good coverage
    EPOCHS_PER_TRIAL = 20  # Max 20 epochs (most trials pruned early)
    SPACE = "wide"         # Full search space
    EARLY_STOP = 6         # Aggressive early stopping (5-6 epochs)
    EXPORT_CSV = True      # Export results to CSV
    
    # Paths
    PROJECT_DIR = Path(__file__).parent
    HPO_DIR = PROJECT_DIR / "results_final" / "hpo"
    LOG_DIR = PROJECT_DIR / "results_final" / "logs"
    
    print("Configuration:")
    print(f"  Project directory: {PROJECT_DIR}")
    print(f"  HPO directory: {HPO_DIR}")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Epochs per trial: {EPOCHS_PER_TRIAL} (max, most pruned earlier)")
    print(f"  Search space: {SPACE}")
    print(f"  Early stopping patience: {EARLY_STOP} epochs")
    print()
    
    # Verify shards exist
    train_shard = PROJECT_DIR / "data_shards_M64_L16" / "train" / "shard_000.npz"
    if not train_shard.exists():
        print(f"✗ ERROR: Training shards not found at {train_shard.parent}")
        print("  Please run ./regenerate_shards.sh first")
        sys.exit(1)
    print("✓ Training shards verified")
    
    # Verify R_samp in shards
    try:
        z = np.load(train_shard)
        has_rsamp = 'R_samp' in z.keys()
        z.close()
        
        if not has_rsamp:
            print("⚠️  WARNING: R_samp not found in shards!")
            print("   HPO will run but hybrid blending (β>0) won't work")
            print("   Recommendation: Regenerate shards with R_samp")
        else:
            print("✓ R_samp present in shards")
    except Exception as e:
        print(f"⚠️  Could not verify R_samp: {e}")
    
    print()
    
    # Create output directories
    HPO_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Starting Stage 1 HPO...")
    print(f"  Study database: {HPO_DIR / 'hpo.db'}")
    print()
    print("Expected runtime: ~6-10 hours (with pruning)")
    print("Press Ctrl+C within 5 seconds to abort...")
    
    import time
    for i in range(5, 0, -1):
        print(f"  Starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    print()
    print()
    
    # Run HPO
    try:
        run_hpo(
            n_trials=N_TRIALS,
            epochs_per_trial=EPOCHS_PER_TRIAL,
            space=SPACE,
            export_csv=EXPORT_CSV,
            early_stop_patience=EARLY_STOP
        )
        
        print()
        print("=" * 80)
        print("STAGE 1 HPO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        
        # Display results
        best_json = HPO_DIR / "best.json"
        print(f"Best hyperparameters saved to: {best_json}")
        
        if best_json.exists():
            import json
            with open(best_json, 'r') as f:
                best = json.load(f)
            print()
            print("Best trial from Stage 1:")
            print(json.dumps(best, indent=2))
            print()
        
        print("=" * 80)
        print("NEXT STEP: Run Stage 2 Refinement")
        print("=" * 80)
        print()
        print("Stage 2 will train top configs on FULL dataset (100K samples).")
        print("The legacy run_stage2_refinement scripts were removed during MVDR cleanup.")
        print()
        print("Manual full-data training:")
        print("  python -m ris_pytorch_pipeline.ris_pipeline train --epochs 50 --use_shards --n_train 100000 --n_val 10000")
        print()
        print("Optional Stage 2b (MVDR SpectrumRefiner):")
        print("  python -m ris_pytorch_pipeline.ris_pipeline train-refiner --backbone_ckpt <path_to_best.pt> --epochs 10 --use_shards")
        print()
        print("  python train_ris.py --epochs 50 --batch-size 64")
        print()
        
    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print("STAGE 1 HPO INTERRUPTED BY USER")
        print("=" * 80)
        print()
        print("Partial results saved to database. You can resume by running again.")
        sys.exit(1)
        
    except Exception as e:
        print()
        print("=" * 80)
        print("STAGE 1 HPO FAILED")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
