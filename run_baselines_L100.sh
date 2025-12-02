#!/bin/bash

# Run L=100 Baseline Training in Parallel Screens
# This script trains DCD-MUSIC and NF-SubspaceNet baselines with L=100 snapshots

echo "🚀 Starting L=100 Baseline Training"
echo "=================================="

# Check if we have the required L=100 baseline data
if [ ! -d "data_shards_L100_baseline" ]; then
    echo "❌ Error: data_shards_L100_baseline not found!"
    echo "Please generate L=100 baseline data first:"
    echo "./generate_L100_baseline_data.sh"
    exit 1
fi

echo "✅ Data shards found"

# Create results directories
mkdir -p results_final/baselines_L100/dcd_music
mkdir -p results_final/baselines_L100/nf_subspacenet

echo "📊 Training DCD-MUSIC Baseline (L=100, M_BS=16)"
echo "=============================================="

# Train DCD-MUSIC with L=100 snapshots (advantage for baseline)
screen -dmS dcd_music_L100 bash -c "
    echo '🎯 Starting DCD-MUSIC L=100 training...'
    python train_baselines_L100_optimal.py \
        --model dcd \
        --data_dir data_shards_L100_baseline \
        --tau 100 \
        --epochs 100 \
        --batch_size 32 \
        --lr 1e-3 \
        --results_dir results_final/baselines_L100/dcd_music
    echo '✅ DCD-MUSIC L=100 training completed'
"

echo "📊 Training NF-SubspaceNet Baseline (L=100, N=144)"
echo "================================================"

# Train NF-SubspaceNet with L=100 snapshots (advantage for baseline)
screen -dmS nf_subspacenet_L100 bash -c "
    echo '🎯 Starting NF-SubspaceNet L=100 training...'
    python train_baselines_L100_optimal.py \
        --model nfssn \
        --data_dir data_shards_L100_baseline \
        --tau 100 \
        --epochs 100 \
        --batch_size 16 \
        --lr 5e-4 \
        --results_dir results_final/baselines_L100/nf_subspacenet
    echo '✅ NF-SubspaceNet L=100 training completed'
"

echo "🎯 L=100 Baseline training started in background screens:"
echo "  - DCD-MUSIC: screen -r dcd_music_L100"
echo "  - NF-SubspaceNet: screen -r nf_subspacenet_L100"
echo ""
echo "📊 Monitor progress with:"
echo "  screen -list"
echo "  screen -r dcd_music_L100      # DCD-MUSIC L=100 training"
echo "  screen -r nf_subspacenet_L100 # NF-SubspaceNet L=100 training"
echo ""
echo "⏱️  Expected training time: 2-4 hours per baseline"
echo "🎯 All baselines will be ready for benchmarking when complete"
echo ""
echo "📈 Comparison Summary:"
echo "  - Our L=16 model: 256 measurements (16×16)"
echo "  - DCD-MUSIC L=100: 1,600 measurements (100×16)"
echo "  - NF-SubspaceNet L=100: 14,400 measurements (100×144)"
echo "  - Measurement reduction: 6.25× vs DCD-MUSIC, 56× vs NF-SubspaceNet"
