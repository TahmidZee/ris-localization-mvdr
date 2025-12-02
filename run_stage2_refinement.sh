#!/bin/bash
# =============================================================================
# STAGE 2: REFINEMENT ON FULL DATA
# =============================================================================
# Train top 5 configs from Stage 1 HPO on full 100K dataset
# Date: 2025-11-26
#
# PREREQUISITES:
#   - Stage 1 HPO completed (./run_hpo_manual.sh)
#   - results_final/hpo/hpo.db exists with completed trials
#
# WHAT THIS DOES:
#   - Extract top 5 configs by K_loc score from Stage 1
#   - Train each on full dataset (100K train / 10K val)
#   - 50 epochs with early stopping (patience 8-10)
#   - Save best model for each config
#   - Compare final metrics and pick winner

set -e  # Exit on error

echo "=============================================================================="
echo "STAGE 2: REFINEMENT ON FULL DATA"
echo "=============================================================================="
echo ""

# Configuration
TOP_K=5              # Number of top configs to refine
EPOCHS=50            # Full training epochs
EARLY_STOP=10        # Early stopping patience (longer than Stage 1)
BATCH_SIZE=64        # Batch size for full training

# Paths
PROJECT_DIR="/home/tahit/ris/MainMusic"
HPO_DIR="$PROJECT_DIR/results_final/hpo"
STAGE2_DIR="$PROJECT_DIR/results_final/stage2"
LOG_DIR="$PROJECT_DIR/results_final/logs"

echo "Configuration:"
echo "  Top configs to refine: $TOP_K"
echo "  Epochs per run: $EPOCHS"
echo "  Early stopping patience: $EARLY_STOP epochs"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Verify Stage 1 completed
if [ ! -f "$HPO_DIR/hpo.db" ]; then
    echo "✗ ERROR: Stage 1 HPO database not found!"
    echo "  Expected: $HPO_DIR/hpo.db"
    echo "  Please run ./run_hpo_manual.sh first"
    exit 1
fi
echo "✓ Stage 1 HPO database found"

# Create output directories
mkdir -p "$STAGE2_DIR"
mkdir -p "$LOG_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/stage2_refinement_${TIMESTAMP}.log"

echo ""
echo "Extracting top $TOP_K configs from Stage 1..."

cd "$PROJECT_DIR"

# Extract top K configs using Python
python << 'EOF' > "$STAGE2_DIR/top_configs.json"
import json
import optuna

# Load Stage 1 study
study = optuna.load_study(
    study_name="L16_M64_wide_v2_optimfix",
    storage="sqlite:///results_final/hpo/hpo.db"
)

# Get completed trials sorted by value (lower is better)
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
completed.sort(key=lambda t: t.value)

# Take top 5
top_k = 5
top_trials = completed[:top_k]

# Format for output
configs = []
for i, trial in enumerate(top_trials):
    configs.append({
        "rank": i + 1,
        "trial_number": trial.number,
        "objective": trial.value,
        "params": trial.params
    })

print(json.dumps(configs, indent=2))
EOF

if [ $? -ne 0 ]; then
    echo "✗ ERROR: Failed to extract top configs"
    exit 1
fi

echo "✓ Top $TOP_K configs extracted to: $STAGE2_DIR/top_configs.json"
echo ""
cat "$STAGE2_DIR/top_configs.json"
echo ""

echo "=============================================================================="
echo "Starting Stage 2 Refinement Training"
echo "=============================================================================="
echo ""
echo "This will train $TOP_K models on full dataset (100K samples)"
echo "Expected time: ~4-6 hours per model = ~20-30 hours total"
echo ""
echo "Press Ctrl+C within 5 seconds to abort..."
sleep 5
echo ""

# Train each top config
for RANK in $(seq 1 $TOP_K); do
    echo ""
    echo "======================================================================"
    echo "Training Config #$RANK of $TOP_K"
    echo "======================================================================"
    
    # Extract this config's parameters
    CONFIG=$(python -c "
import json
with open('$STAGE2_DIR/top_configs.json') as f:
    configs = json.load(f)
for c in configs:
    if c['rank'] == $RANK:
        print(json.dumps(c['params']))
        break
")
    
    TRIAL_NUM=$(python -c "
import json
with open('$STAGE2_DIR/top_configs.json') as f:
    configs = json.load(f)
for c in configs:
    if c['rank'] == $RANK:
        print(c['trial_number'])
        break
")
    
    echo "Trial #$TRIAL_NUM params: $CONFIG"
    echo ""
    
    # Create config file for this run
    CONFIG_FILE="$STAGE2_DIR/config_rank${RANK}.json"
    echo "$CONFIG" > "$CONFIG_FILE"
    
    # Output directory for this run
    RUN_DIR="$STAGE2_DIR/rank${RANK}_trial${TRIAL_NUM}"
    mkdir -p "$RUN_DIR"
    
    RUN_LOG="$LOG_DIR/stage2_rank${RANK}_${TIMESTAMP}.log"
    
    echo "Output: $RUN_DIR"
    echo "Log: $RUN_LOG"
    echo ""
    
    # Run full training with this config
    python -m ris_pytorch_pipeline.ris_pipeline train \
        --epochs $EPOCHS \
        --use_shards \
        --from_hpo "$CONFIG_FILE" \
        --batch-size $BATCH_SIZE \
        --early-stop-patience $EARLY_STOP \
        --output-dir "$RUN_DIR" \
        2>&1 | tee "$RUN_LOG"
    
    RUN_EXIT=$?
    
    if [ $RUN_EXIT -eq 0 ]; then
        echo "✓ Config #$RANK training completed"
    else
        echo "⚠️  Config #$RANK training failed (exit code: $RUN_EXIT)"
    fi
done

echo ""
echo "=============================================================================="
echo "STAGE 2 REFINEMENT COMPLETE"
echo "=============================================================================="
echo ""

# Compare results
echo "Comparing final metrics..."
echo ""

python << 'EOF'
import json
import os
from pathlib import Path

stage2_dir = Path("results_final/stage2")
results = []

for rank in range(1, 6):
    # Find the run directory
    run_dirs = list(stage2_dir.glob(f"rank{rank}_trial*"))
    if not run_dirs:
        continue
    run_dir = run_dirs[0]
    
    # Look for metrics file
    metrics_file = run_dir / "final_metrics.json"
    if not metrics_file.exists():
        # Try checkpoint directory
        ckpt_dirs = list(run_dir.glob("checkpoints"))
        if ckpt_dirs:
            metrics_file = ckpt_dirs[0] / "best_metrics.json"
    
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        results.append({
            "rank": rank,
            "dir": str(run_dir),
            "metrics": metrics
        })
    else:
        # Try to read from training log
        results.append({
            "rank": rank,
            "dir": str(run_dir),
            "metrics": "Not found - check training log"
        })

print("=" * 70)
print("STAGE 2 RESULTS COMPARISON")
print("=" * 70)
print()

for r in results:
    print(f"Rank #{r['rank']}: {r['dir']}")
    if isinstance(r['metrics'], dict):
        print(f"  K_acc: {r['metrics'].get('K_acc', 'N/A')}")
        print(f"  K_under: {r['metrics'].get('K_under', 'N/A')}")
        print(f"  AoA_RMSE: {r['metrics'].get('AoA_RMSE', 'N/A')}")
        print(f"  Range_RMSE: {r['metrics'].get('Range_RMSE', 'N/A')}")
        print(f"  Success_rate: {r['metrics'].get('Success_rate', 'N/A')}")
    else:
        print(f"  {r['metrics']}")
    print()

if results:
    print("=" * 70)
    print("RECOMMENDATION: Check each run's log for final validation metrics")
    print("Pick the model with highest K_acc and lowest K_under")
    print("=" * 70)
EOF

echo ""
echo "Stage 2 logs saved to: $LOG_DIR/stage2_*.log"
echo "Model checkpoints in: $STAGE2_DIR/rank*/"
echo ""
echo "Next steps:"
echo "  1. Compare metrics above (or check individual logs)"
echo "  2. Pick best model based on K_acc, K_under, RMSE"
echo "  3. Run final evaluation on test set:"
echo "     python -m ris_pytorch_pipeline.benchmark_test --checkpoint <best_model.pt>"



