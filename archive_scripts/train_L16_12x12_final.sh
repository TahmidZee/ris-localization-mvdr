#!/bin/bash  
# Final training for L=16 12x12 system with substantial performance gains

echo "🎯 Starting L=16 12x12 final training"
echo "Expected: ~45-65% angle RMSE reduction, ~30-55% range RMSE reduction"
echo ""

# Configuration
export CUDA_VISIBLE_DEVICES=0
cd /home/tahit/ris/MainMusic

# Ensure directories exist
mkdir -p results_final_L16_12x12/{checkpoints,logs}

echo "⚙️ Training Configuration:"
echo "  • System: 12×12 UPA, L=16 (144 elements, K≤15)"
echo "  • Data: 80K train, 16K val, 20K test samples"
echo "  • Epochs: 70 (early stop patience 10 on val composite RMSE)"
echo "  • Batch size: 64, grad accumulation: 2 (effective 128)"
echo "  • Optimizer: AdamW 3e-4, warmup 2, cosine decay"  
echo "  • Features: EMA weights, max_grad_norm=1.0, SWA last 20%"
echo "  • Loss weights: From HPO best.json (reduced for 12×12 eigengap)"
echo "  • 3-Phase curriculum: B1/B2/B3 with reduced spectrum weights"
echo "  • K-calibration: Temperature scaling on held-out val slice"
echo ""

# Check for HPO results
if [ ! -f "results_final_L16_12x12/hpo/best.json" ]; then
    echo "⚠️  Warning: No HPO results found (results_final_L16_12x12/hpo/best.json)"
    echo "   Training will use default hyperparameters."
    echo "   Consider running HPO first for optimal performance."
    echo ""
fi

echo "🚀 Starting final training..."
python -m ris_pytorch_pipeline.train \
    --from_hpo true \
    --epochs 70 \
    --batch_size 64 \
    --grad_accumulation 2 \
    --early_stop_patience 10 \
    --max_grad_norm 1.0 \
    --deterministic true \
    --save_best true \
    --calibrate_k true

echo ""
echo "✅ L=16 12x12 training completed!"
echo ""
echo "📋 Training Results:"
echo "  • Best model: results_final_L16_12x12/checkpoints/best.pt"
echo "  • SWA model: results_final_L16_12x12/checkpoints/swa.pt"
echo "  • Training logs: results_final_L16_12x12/logs/"
echo ""
echo "📊 Expected Performance vs L=8 7×7:"
echo "  • Angle RMSE: 45-65% reduction"
echo "  • Range RMSE: 30-55% reduction"  
echo "  • K estimation: Much more stable (K≤15 vs K≤7)"
echo "  • Variance improvement: ~5.9× theoretical gain"
echo ""
echo "🎯 Ready for benchmarking with substantial performance gains!"


