#!/usr/bin/env python3
"""
Regenerate shards with R_samp for hybrid blending
Alternative Python script (if you prefer Python over bash)
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from ris_pytorch_pipeline.dataset import prepare_split_shards, set_sampling_overrides_from_cfg
from ris_pytorch_pipeline.configs import mdl_cfg
import numpy as np


def main():
    print("=" * 80)
    print("SHARD REGENERATION SCRIPT (Python)")
    print("=" * 80)
    print()
    
    # Configuration
    SHARD_DIR = Path("data_shards_M64_L16")
    N_TRAIN = 100000
    N_VAL = 10000
    N_TEST = 10000
    SHARD_SIZE = 25000
    SEED = 42
    ETA = 0.0  # No element perturbation
    L = 16     # L=16 snapshots
    
    # Robust training ranges
    PHI_FOV = 60.0
    THETA_FOV = 30.0
    R_MIN = 0.5
    R_MAX = 10.0
    SNR_MIN = -5.0
    SNR_MAX = 20.0
    
    print("Configuration:")
    print(f"  Shard directory: {SHARD_DIR}")
    print(f"  Training samples: {N_TRAIN}")
    print(f"  Validation samples: {N_VAL}")
    print(f"  Test samples: {N_TEST}")
    print(f"  Shard size: {SHARD_SIZE}")
    print(f"  L (snapshots): {L}")
    print(f"  PHI FOV: ±{PHI_FOV}°, THETA FOV: ±{THETA_FOV}°")
    print(f"  Range: {R_MIN}m to {R_MAX}m")
    print(f"  SNR: {SNR_MIN} dB to {SNR_MAX} dB")
    print()
    
    # Step 1: Backup old shards
    if SHARD_DIR.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"{SHARD_DIR}_backup_{timestamp}")
        print("Step 1: Backing up old shards...")
        print(f"  Old: {SHARD_DIR}")
        print(f"  Backup: {backup_dir}")
        shutil.move(str(SHARD_DIR), str(backup_dir))
        print("  ✓ Backup complete")
        print()
    else:
        print("Step 1: No existing shards to backup")
        print()
    
    # Step 2: Create fresh directory structure
    print("Step 2: Creating fresh directory structure...")
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    (SHARD_DIR / "train").mkdir(exist_ok=True)
    (SHARD_DIR / "val").mkdir(exist_ok=True)
    (SHARD_DIR / "test").mkdir(exist_ok=True)
    print(f"  ✓ Created {SHARD_DIR}/{{train,val,test}}")
    print()
    
    # Step 3: Set sampling overrides
    print("Step 3: Configuring sampling parameters...")
    mdl_cfg.TRAIN_PHI_FOV_DEG = PHI_FOV
    mdl_cfg.TRAIN_THETA_FOV_DEG = THETA_FOV
    mdl_cfg.TRAIN_R_MIN_MAX = (R_MIN, R_MAX)
    mdl_cfg.SNR_DB_RANGE = (SNR_MIN, SNR_MAX)
    mdl_cfg.SNR_TARGETED = True
    set_sampling_overrides_from_cfg(mdl_cfg)
    print("  ✓ Sampling overrides configured")
    print()
    
    # Step 4: Generate shards
    print("Step 4: Generating shards (this will take ~30-60 minutes)...")
    print("  [INFO] R_samp will be pre-computed for each sample (offline path)")
    print("  [INFO] Press Ctrl+C within 5 seconds to abort...")
    print()
    
    import time
    for i in range(5, 0, -1):
        print(f"  Starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    print()
    
    try:
        prepare_split_shards(
            root_dir=SHARD_DIR,
            n_train=N_TRAIN,
            n_val=N_VAL,
            n_test=N_TEST,
            shard_size=SHARD_SIZE,
            seed=SEED,
            eta_perturb=ETA,
            override_L=L
        )
        print()
        print("  ✓ Shard generation complete")
        print()
    except KeyboardInterrupt:
        print()
        print("  ✗ Aborted by user")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"  ✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Verify shards
    print("Step 5: Verifying shards...")
    print()
    
    # Count shards
    train_shards = list((SHARD_DIR / "train").glob("*.npz"))
    val_shards = list((SHARD_DIR / "val").glob("*.npz"))
    test_shards = list((SHARD_DIR / "test").glob("*.npz"))
    
    print("  Shard counts:")
    print(f"    Train: {len(train_shards)} shards")
    print(f"    Val:   {len(val_shards)} shards")
    print(f"    Test:  {len(test_shards)} shards")
    print()
    
    # Check R_samp in first training shard
    if not train_shards:
        print("  ✗ No training shards found!")
        sys.exit(1)
    
    print("  Checking R_samp presence...")
    first_shard = train_shards[0]
    z = np.load(first_shard)
    keys = list(z.keys())
    has_rsamp = 'R_samp' in keys
    
    print(f"  Keys in shard: {keys}")
    print(f"  R_samp present: {has_rsamp}")
    
    if has_rsamp:
        R_samp = z['R_samp']
        print(f"  R_samp shape: {R_samp.shape}")
        print(f"  R_samp dtype: {R_samp.dtype}")
        # Check if R_samp is non-zero
        if np.any(R_samp != 0):
            print("  ✓ R_samp is present and non-zero")
        else:
            print("  ⚠️  R_samp is all zeros (check computation)")
            z.close()
            sys.exit(1)
    else:
        print("  ✗ R_samp NOT found in shard!")
        z.close()
        sys.exit(1)
    
    z.close()
    print()
    print("  ✓ Verification passed")
    print()
    
    # Success summary
    print("=" * 80)
    print("SHARD REGENERATION COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✓ Old shards backed up (if existed)")
    print("  ✓ New shards generated with R_samp")
    print("  ✓ Verification passed")
    print()
    print("Next steps:")
    print("  1. Run audit: python audit_covariance_paths.py")
    print("  2. Run overfit test: python test_overfit.py")
    print("  3. Full training: python train_ris.py --epochs 50 --batch-size 64")
    print()
    print("Note: Hybrid blending is enabled (cfg.HYBRID_COV_BETA = 0.3)")
    print("      K-head and MUSIC will now use the same hybrid R_eff")
    print()


if __name__ == "__main__":
    main()




