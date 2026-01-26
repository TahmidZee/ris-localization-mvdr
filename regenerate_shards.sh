#!/bin/bash
# Regenerate shards with R_samp for hybrid blending
# Date: 2025-11-26
# Purpose: Clean old shards and generate new ones with offline R_samp computation

set -e  # Exit on error

echo "=============================================================================="
echo "SHARD REGENERATION SCRIPT"
echo "=============================================================================="
echo ""

# Configuration (derive shard dir + L from cfg by default)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
SHARD_DIR="$(python -c "from ris_pytorch_pipeline.configs import cfg; print(cfg.DATA_SHARDS_DIR)")"
N_TRAIN=100000
N_VAL=10000
N_TEST=10000
SHARD_SIZE="$(python -c "from ris_pytorch_pipeline.configs import cfg; print(int(getattr(cfg,'SHARD_SIZE_DEFAULT', 25000)))")"
SEED=42
ETA=0.0  # No element perturbation
L="$(python -c "from ris_pytorch_pipeline.configs import cfg; print(int(cfg.L))")"

# Robust training ranges (match paper setup)
PHI_FOV=60.0     # ±60° azimuth
THETA_FOV=30.0   # ±30° elevation
R_MIN=0.5        # 0.5m minimum range
R_MAX=10.0       # 10m maximum range
SNR_MIN=-5.0     # -5 dB minimum SNR
SNR_MAX=20.0     # 20 dB maximum SNR

echo "Configuration:"
echo "  Shard directory: $SHARD_DIR"
echo "  Training samples: $N_TRAIN"
echo "  Validation samples: $N_VAL"
echo "  Test samples: $N_TEST"
echo "  Shard size: $SHARD_SIZE"
echo "  L (snapshots): $L"
echo "  Store R_samp in shards: $(python -c \"from ris_pytorch_pipeline.configs import cfg; print(bool(getattr(cfg,'STORE_RSAMP_IN_SHARDS', False)))\")"
echo "  PHI FOV: ±${PHI_FOV}°, THETA FOV: ±${THETA_FOV}°"
echo "  Range: ${R_MIN}m to ${R_MAX}m"
echo "  SNR: ${SNR_MIN} dB to ${SNR_MAX} dB"
echo ""

# Step 1: Backup old shards (if they exist)
if [ -d "$SHARD_DIR" ]; then
    BACKUP_DIR="${SHARD_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    echo "Step 1: Backing up old shards..."
    echo "  Old: $SHARD_DIR"
    echo "  Backup: $BACKUP_DIR"
    mv "$SHARD_DIR" "$BACKUP_DIR"
    echo "  ✓ Backup complete"
    echo ""
else
    echo "Step 1: No existing shards to backup"
    echo ""
fi

# Step 2: Create fresh shard directory structure
echo "Step 2: Creating fresh directory structure..."
mkdir -p "$SHARD_DIR"/{train,val,test}
echo "  ✓ Created $SHARD_DIR/{train,val,test}"
echo ""

# Step 3: Generate shards with R_samp
echo "Step 3: Generating shards (this will take ~30-60 minutes)..."
echo "  [INFO] R_samp will be pre-computed for each sample (offline path)"
echo "  [INFO] Press Ctrl+C within 5 seconds to abort..."
sleep 5
echo ""

# Run the pregen-split command
python -m ris_pytorch_pipeline.ris_pipeline pregen-split \
    --out_dir "$SHARD_DIR" \
    --n-train $N_TRAIN \
    --n-val $N_VAL \
    --n-test $N_TEST \
    --shard $SHARD_SIZE \
    --eta $ETA \
    --L $L \
    --seed $SEED \
    --phi-fov-deg $PHI_FOV \
    --theta-fov-deg $THETA_FOV \
    --r-min $R_MIN \
    --r-max $R_MAX \
    --snr-min $SNR_MIN \
    --snr-max $SNR_MAX

echo ""
echo "  ✓ Shard generation complete"
echo ""

# Step 4: Verify shards
echo "Step 4: Verifying shards..."
echo ""

# Count shards
TRAIN_COUNT=$(ls -1 "$SHARD_DIR/train"/*.npz 2>/dev/null | wc -l)
VAL_COUNT=$(ls -1 "$SHARD_DIR/val"/*.npz 2>/dev/null | wc -l)
TEST_COUNT=$(ls -1 "$SHARD_DIR/test"/*.npz 2>/dev/null | wc -l)

echo "  Shard counts:"
echo "    Train: $TRAIN_COUNT shards"
echo "    Val:   $VAL_COUNT shards"
echo "    Test:  $TEST_COUNT shards"
echo ""

# Check if R_samp is present in the first training shard
echo "  Checking R_samp presence..."
python -c "
import numpy as np
from pathlib import Path

shard_path = Path('$SHARD_DIR/train/shard_000.npz')
if not shard_path.exists():
    print('  ✗ First training shard not found!')
    exit(1)

z = np.load(shard_path)
keys = list(z.keys())
has_rsamp = 'R_samp' in keys

print(f'  Keys in shard: {keys}')
print(f'  R_samp present: {has_rsamp}')

if has_rsamp:
    R_samp = z['R_samp']
    print(f'  R_samp shape: {R_samp.shape}')
    print(f'  R_samp dtype: {R_samp.dtype}')
    # Check if R_samp is non-zero
    if np.any(R_samp != 0):
        print('  ✓ R_samp is present and non-zero')
    else:
        print('  ⚠️  R_samp is all zeros (check computation)')
else:
    print('  ✗ R_samp NOT found in shard!')
    exit(1)

z.close()
"

if [ $? -eq 0 ]; then
    echo ""
    echo "  ✓ Verification passed"
else
    echo ""
    echo "  ✗ Verification FAILED"
    exit 1
fi

echo ""
echo "=============================================================================="
echo "SHARD REGENERATION COMPLETE"
echo "=============================================================================="
echo ""
echo "Summary:"
echo "  ✓ Old shards backed up (if existed)"
echo "  ✓ New shards generated with R_samp"
echo "  ✓ Verification passed"
echo ""
echo "Next steps:"
echo "  1. Run audit: python audit_covariance_paths.py"
echo "  2. Run overfit test: python test_overfit.py"
echo "  3. Full training: python train_ris.py --epochs 50 --batch-size 64"
echo ""
echo "Note: Hybrid blending is enabled (cfg.HYBRID_COV_BETA = 0.3)"
echo "      K-head and MUSIC will now use the same hybrid R_eff"
echo ""




