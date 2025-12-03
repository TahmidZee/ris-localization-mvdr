#!/bin/bash
# Step 3: Evaluate Model with Stratified Metrics
# Runs acceptance gates and generates paper tables

set -e  # Exit on error

cd "$(dirname "$0")"

echo "========================================="
echo "Step 3: Model Evaluation"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Checkpoint: results_M32_L16_production/checkpoints/best.pt"
echo "  - Data: data_shards_M32_L16/"
echo "  - Samples: 2000"
echo "  - Metrics: Stratified by SNR bins"
echo ""
echo "Acceptance Gates (SNR ≥ 0 dB):"
echo "  - φ median ≤ 0.6°"
echo "  - φ 90% ≤ 0.9°"
echo "  - θ median ≤ 0.9°"
echo "  - θ 90% ≤ 1.2°"
echo "  - r median ≤ 0.8m"
echo "  - K accuracy ≥ 90%"
echo ""

# Check checkpoint exists
if [ ! -f "results_M32_L16_production/checkpoints/best.pt" ]; then
    echo "Error: Checkpoint not found!"
    echo "Run 2_train_model.sh first."
    exit 1
fi

# Check data exists
if [ ! -d "data_shards_M32_L16" ]; then
    echo "Error: data_shards_M32_L16/ not found!"
    exit 1
fi

# Confirm
read -p "Start evaluation? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Evaluation cancelled."
    exit 1
fi

echo ""
echo "Running stratified evaluation..."
echo ""

# Run evaluation
python stratified_evaluation.py \
  --checkpoint results_M32_L16_production/checkpoints/best.pt \
  --data_root data_shards_M32_L16 \
  --n_samples 2000 \
  --device cuda \
  --k_conf_thresh 0.8

echo ""
echo "========================================="
echo "Evaluation Complete!"
echo "========================================="
echo ""
echo "Check the output above for:"
echo "  - Stratified metrics by SNR bins"
echo "  - Acceptance gate results"
echo "  - Overall performance summary"
echo ""
echo "If all acceptance gates PASS:"
echo "  ✅ Sub-1° performance achieved!"
echo "  ✅ Ready for paper submission"
echo ""
echo "If any gates FAIL:"
echo "  ⚠️  Review training logs"
echo "  ⚠️  Check data generation"
echo "  ⚠️  Consider increasing M_beams or training longer"
echo ""

