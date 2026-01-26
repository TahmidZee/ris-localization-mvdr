#!/bin/bash
# =============================================================================
# STAGE 1: HPO EXPLORATION (10% Data)
# =============================================================================
# Optimizes K̂ + AoA/Range accuracy (not raw loss)
# Date: 2025-11-26
#
# TWO-STAGE HPO STRATEGY:
#   Stage 1 (THIS SCRIPT): Explore hyperparameter space on 10% data
#     - 50 trials, 20 epochs max, aggressive pruning
#     - Goal: Find good regions, not final hyperparameters
#     - Time: ~6-10 hours
#   Stage 2 (manual): Train top configs on full data using `ris_pipeline train`
#   Optional Stage 2b (MVDR): Train SpectrumRefiner using `ris_pipeline train-refiner`

set -e  # Exit on error

echo "=============================================================================="
echo "STAGE 1: HPO EXPLORATION (10% Data)"
echo "=============================================================================="
echo ""
echo "Strategy:"
echo "  - 50 trials on 10% subset (10K train / 1K val)"
echo "  - Max 20 epochs per trial (most pruned at 6-10)"
echo "  - Goal: Find promising hyperparameter regions"
echo "  - After this: Run Stage 2 with top 5 configs on full data"
echo ""

# Stage 1 Configuration (Exploration)
N_TRIALS=50          # 40-60 trials for good coverage
EPOCHS_PER_TRIAL=20  # Max 20 epochs (most trials pruned early)
SPACE="wide"         # Full search space
EARLY_STOP=6         # Aggressive early stopping (5-6 epochs)
# Note: CSV export is handled automatically by hpo.py

# Paths (derive from cfg so L/M changes don't break scripts)
PROJECT_DIR="/home/tahit/ris/MainMusic"
cd "$PROJECT_DIR"
HPO_DIR="$PROJECT_DIR/$(python -c "from ris_pytorch_pipeline.configs import cfg; print(cfg.HPO_DIR)")"
LOG_DIR="$PROJECT_DIR/$(python -c "from ris_pytorch_pipeline.configs import cfg; print(cfg.LOGS_DIR)")"
SHARDS_DIR="$PROJECT_DIR/$(python -c "from ris_pytorch_pipeline.configs import cfg; print(cfg.DATA_SHARDS_DIR)")"

echo "Configuration:"
echo "  Project directory: $PROJECT_DIR"
echo "  HPO directory: $HPO_DIR"
echo "  Trials: $N_TRIALS"
echo "  Epochs per trial: $EPOCHS_PER_TRIAL (max, most pruned earlier)"
echo "  Search space: $SPACE"
echo "  Early stopping patience: $EARLY_STOP epochs"
echo ""

# Verify shards exist
TRAIN_SHARD="$SHARDS_DIR/train/shard_000.npz"
if [ ! -f "$TRAIN_SHARD" ]; then
    echo "✗ ERROR: Training shards not found at $SHARDS_DIR/train/"
    echo "  Please regenerate shards first (see regenerate_shards.sh or ris_pipeline pregen-split)."
    exit 1
fi
echo "✓ Training shards verified"

# Verify R_samp in shards
HAS_RSAMP=$(python -c "import numpy as np; z = np.load('$TRAIN_SHARD'); print('R_samp' in z.keys()); z.close()")
if [ "$HAS_RSAMP" != "True" ]; then
    echo "⚠️  WARNING: R_samp not found in shards!"
    echo "   HPO will run but hybrid blending (β>0) won't work"
    echo "   Recommendation: Regenerate shards with R_samp"
else
    echo "✓ R_samp present in shards"
fi
echo ""

# Create output directories
mkdir -p "$HPO_DIR"
mkdir -p "$LOG_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/hpo_stage1_${TIMESTAMP}.log"

echo "Starting Stage 1 HPO..."
echo "  Log file: $LOG_FILE"
echo "  Study database: $HPO_DIR/hpo.db"
echo ""
echo "Expected runtime: ~6-10 hours (with pruning)"
echo "Press Ctrl+C within 5 seconds to abort..."
sleep 5
echo ""

# Run HPO via Python module
cd "$PROJECT_DIR"

# Note: export_csv is handled internally by hpo.py, not via CLI flag
python -m ris_pytorch_pipeline.ris_pipeline hpo \
    --trials $N_TRIALS \
    --hpo-epochs $EPOCHS_PER_TRIAL \
    --space $SPACE \
    --early-stop-patience $EARLY_STOP \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "=============================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "STAGE 1 HPO COMPLETED SUCCESSFULLY"
else
    echo "STAGE 1 HPO FAILED (exit code: $EXIT_CODE)"
fi
echo "=============================================================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    # Display results
    echo "Best hyperparameters saved to: $HPO_DIR/best.json"
    if [ -f "$HPO_DIR/best.json" ]; then
        echo ""
        echo "Best trial from Stage 1:"
        cat "$HPO_DIR/best.json"
        echo ""
    fi
    
    echo "=============================================================================="
    echo "NEXT STEP: Run Stage 2 Refinement"
    echo "=============================================================================="
    echo ""
    echo "Stage 2 will train top configs on FULL dataset (100K samples)."
    echo "The legacy run_stage2_refinement scripts were removed during MVDR cleanup."
    echo ""
    echo "Manual full-data training:"
    echo "  python -m ris_pytorch_pipeline.ris_pipeline train --epochs 50 --use_shards --n_train 100000 --n_val 10000"
    echo ""
    echo "Optional Stage 2b (MVDR SpectrumRefiner):"
    echo "  python -m ris_pytorch_pipeline.ris_pipeline train-refiner --backbone_ckpt <path_to_best.pt> --epochs 10 --use_shards"
    echo ""
    echo "  python train_ris.py --epochs 50 --batch-size 64"
    echo ""
    echo "Log saved to: $LOG_FILE"
else
    echo "Check log for errors: $LOG_FILE"
fi
