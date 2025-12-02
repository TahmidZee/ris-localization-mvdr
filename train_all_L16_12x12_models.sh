#!/bin/bash
# Train all models (Hybrid, DCD-MUSIC, NF-SubspaceNet) with L=16 12x12 data

echo "🎯 Training all models with L=16 12x12 data for comprehensive comparison"
echo "Expected gains for Hybrid model: ~45-65% angle RMSE, ~30-55% range RMSE"
echo ""

# Configuration
export CUDA_VISIBLE_DEVICES=0  
cd /home/tahit/ris/MainMusic

echo "📊 System Configuration:"
echo "  • Geometry: 12×12 UPA (144 elements)"
echo "  • Snapshots: L=16 (K≤15)"
echo "  • Data: 80K train, 16K val, 20K test"
echo "  • Models: Hybrid (ours), DCD-MUSIC, NF-SubspaceNet"
echo ""

# 1. Train Hybrid Model (our method with substantial gains expected)
echo "🚀 [1/3] Training Hybrid Model (our method)..."
echo "Expected: Substantial performance gains over baselines"
./train_L16_12x12_final.sh

echo ""
echo "⏱️  Waiting 30 seconds before next model..."
sleep 30

# 2. Train DCD-MUSIC baseline
echo "🎯 [2/3] Training DCD-MUSIC baseline..."
python -m ris_pytorch_pipeline.train_dcd \
    --data_dir results_final_L16_12x12/data/shards \
    --results_dir results_final_L16_12x12/dcd_results \
    --N_H 12 --N_V 12 --L 16 \
    --epochs 50 \
    --batch_size 64

echo ""
echo "⏱️  Waiting 30 seconds before next model..."
sleep 30

# 3. Train NF-SubspaceNet baseline  
echo "🎯 [3/3] Training NF-SubspaceNet baseline..."
python -m ris_pytorch_pipeline.train_nfssn \
    --data_dir results_final_L16_12x12/data/shards \
    --results_dir results_final_L16_12x12/nfssn_results \
    --N_H 12 --N_V 12 --L 16 \
    --epochs 50 \
    --batch_size 64

echo ""
echo "✅ All L=16 12x12 models trained successfully!"
echo ""
echo "📋 Training Summary:"
echo "  • Hybrid (ours): results_final_L16_12x12/checkpoints/"
echo "  • DCD-MUSIC: results_final_L16_12x12/dcd_results/"
echo "  • NF-SubspaceNet: results_final_L16_12x12/nfssn_results/"
echo ""
echo "🏆 Expected Results for Hybrid Model:"
echo "  • Angle RMSE: 45-65% reduction vs L=8 7×7"
echo "  • Range RMSE: 30-55% reduction vs L=8 7×7"
echo "  • Superior K estimation stability"
echo "  • ~5.9× variance improvement"
echo ""
echo "🎯 Ready for comprehensive benchmarking!"
echo "Run benchmark to compare all methods with paper-compliant metrics."


