#!/bin/bash
# Generate L=16 12x12 data for substantial performance gains

echo "🚀 Generating L=16 12x12 UPA data for near-field localization"
echo "Expected gains: ~45-65% angle RMSE reduction, ~30-55% range RMSE reduction"
echo ""

# Configuration 
export CUDA_VISIBLE_DEVICES=0
cd /home/tahit/ris/MainMusic

# Create results directory
mkdir -p results_final_L16_12x12/{data/shards/{train,val,test},hpo,logs,checkpoints}

echo "📊 System Configuration:"
echo "  • Geometry: 12×12 UPA (144 elements)"  
echo "  • Snapshots: L=16 (K ≤ 15)"
echo "  • Frequency: 1 GHz (λ=0.3m)"
echo "  • Range: 0.5-10.0m"
echo "  • Element spacing: λ/2 = 0.15m"
echo "  • Expected variance improvement: ~5.9×"
echo "  • Expected RMSE improvement: ~2.4×"
echo ""

# Generate training data (large set for final training)
echo "🎯 Generating training data..."
python -m ris_pytorch_pipeline.ris_pipeline generate \
    --N_H 12 --N_V 12 --L 16 \
    --range_min 0.5 --range_max 10.0 \
    --split train \
    --samples 80000 \
    --K_range 1,5 \
    --snr_range -5,20 \
    --coherent_prob 0.3 \
    --save_dir results_final_L16_12x12/data/shards/train

echo ""

# Generate validation data (for HPO and monitoring)
echo "🔍 Generating validation data..."  
python -m ris_pytorch_pipeline.ris_pipeline generate \
    --N_H 12 --N_V 12 --L 16 \
    --range_min 0.5 --range_max 10.0 \
    --split val \
    --samples 16000 \
    --K_range 1,5 \
    --snr_range -5,20 \
    --coherent_prob 0.3 \
    --save_dir results_final_L16_12x12/data/shards/val

echo ""

# Generate test data (for final benchmarking)
echo "📈 Generating test data..."
python -m ris_pytorch_pipeline.ris_pipeline generate \
    --N_H 12 --N_V 12 --L 16 \
    --range_min 0.5 --range_max 10.0 \
    --split test \
    --samples 20000 \
    --K_range 1,5 \
    --snr_range -5,20 \
    --coherent_prob 0.3 \
    --save_dir results_final_L16_12x12/data/shards/test

echo ""
echo "✅ L=16 12x12 data generation completed!"
echo "📁 Data location: results_final_L16_12x12/data/shards/"
echo "📊 Train: 80K samples, Val: 16K samples, Test: 20K samples"
echo ""
echo "🎯 Ready for HPO with expected substantial performance gains!"


