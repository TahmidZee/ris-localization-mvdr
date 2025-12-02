#!/bin/bash
# HPO for L=16 12x12 system with optimized budget

echo "🔍 Starting L=16 12x12 HPO for substantial performance gains"
echo "Expected: ~45-65% angle RMSE reduction, ~30-55% range RMSE reduction"
echo ""

# Configuration
export CUDA_VISIBLE_DEVICES=0
cd /home/tahit/ris/MainMusic

# Ensure directories exist
mkdir -p results_final_L16_12x12/{hpo,logs}

echo "⚙️ HPO Configuration:"
echo "  • System: 12×12 UPA, L=16 (144 elements, K≤15)"
echo "  • Budget: 40 trials (quick) → 80 trials (full)"
echo "  • Epochs per trial: 12 (with pruning after epoch 4)"
echo "  • Train subset: 32K samples (40% of 80K)"
echo "  • Val set: 16K samples"
echo "  • Batch size: 64, grad accumulation: 2 (effective 128)"
echo "  • Optimizer: AdamW 3e-4, warmup 2, cosine decay"
echo "  • Pruning: Median pruner (40% speedup)"
echo "  • Loss weights: λ_cov, λ_ang, λ_rng, λ_K, shrink_α, softmax_τ"
echo ""

# Quick HPO (40 trials)
echo "🚀 Starting quick HPO (40 trials)..."
python -m ris_pytorch_pipeline.hpo \
    --n_trials 40 \
    --epochs_per_trial 12 \
    --space medium \
    --export_csv

echo ""
echo "📊 Quick HPO completed! Results in results_final_L16_12x12/hpo/"

# Check if we should continue to full HPO
echo ""
echo "🤔 Continue to full HPO (80 trials)? [y/N]"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo ""
    echo "🎯 Starting full HPO (80 trials)..."
    python -m ris_pytorch_pipeline.hpo \
        --n_trials 80 \
        --epochs_per_trial 12 \
        --space full \
        --export_csv
    
    echo ""
    echo "✅ Full HPO completed!"
else
    echo "✅ Quick HPO only. Use results for training."
fi

echo ""
echo "📋 HPO Results Summary:"
echo "  • Best trial: results_final_L16_12x12/hpo/best.json"
echo "  • Full database: results_final_L16_12x12/hpo/hpo.db"
echo "  • Export CSV: results_final_L16_12x12/hpo/trials.csv"
echo ""
echo "🎯 Ready for final training with optimized hyperparameters!"
echo "Expected substantial gains over L=8 7×7 system!"


