#!/bin/bash

# Train Baseline Models in Parallel
# This script trains DCD-MUSIC and NF-SubspaceNet baselines

echo "🚀 Starting Baseline Model Training"
echo "=================================="

# Check if we have the required data
if [ ! -d "data_shards_M64_L16" ]; then
    echo "❌ Error: data_shards_M64_L16 not found!"
    echo "Please run data generation first:"
    echo "python -m ris_pytorch_pipeline.ris_pipeline pregen-split --L 16 --n-train 80000 --n-val 16000 --n-test 4000 --snr-min -5 --snr-max 20 --out_dir data_shards_M64_L16 --phi-fov-deg 60 --theta-fov-deg 30 --r-min 0.5 --r-max 10.0"
    exit 1
fi

echo "✅ Data shards found"

# Create results directories
mkdir -p results_final/baselines/dcd_music
mkdir -p results_final/baselines/nf_subspacenet

echo "📊 Training DCD-MUSIC Baseline (L=100, M_BS=16)"
echo "=============================================="

# Train DCD-MUSIC with L=100 snapshots (advantage for baseline)
screen -dmS dcd_music bash -c "
    echo '🎯 Starting DCD-MUSIC training...'
    python -m ris_pytorch_pipeline.train_baseline_dcd_music \
        --L 100 \
        --epochs 100 \
        --batch_size 32 \
        --lr 1e-3 \
        --data_dir data_shards_M64_L16 \
        --results_dir results_final/baselines/dcd_music \
        --use_3_stage_training \
        --newton_iterations 5 \
        --range_grid_size 201
    echo '✅ DCD-MUSIC training completed'
"

echo "📊 Training NF-SubspaceNet Baseline (L=100, N=144)"
echo "================================================"

# Train NF-SubspaceNet with L=100 snapshots (advantage for baseline)
screen -dmS nf_subspacenet bash -c "
    echo '🎯 Starting NF-SubspaceNet training...'
    python -m ris_pytorch_pipeline.train_baseline_nf_subspacenet \
        --L 100 \
        --epochs 100 \
        --batch_size 16 \
        --lr 5e-4 \
        --data_dir data_shards_M64_L16 \
        --results_dir results_final/baselines/nf_subspacenet \
        --hidden_dim 512 \
        --num_layers 4
    echo '✅ NF-SubspaceNet training completed'
"

echo "🎯 Baseline training started in background screens:"
echo "  - DCD-MUSIC: screen -r dcd_music"
echo "  - NF-SubspaceNet: screen -r nf_subspacenet"
echo ""
echo "📊 Monitor progress with:"
echo "  screen -list"
echo "  screen -r dcd_music    # DCD-MUSIC training"
echo "  screen -r nf_subspacenet  # NF-SubspaceNet training"
echo ""
echo "⏱️  Expected training time: 2-4 hours per baseline"
echo "🎯 All baselines will be ready for benchmarking when complete"
