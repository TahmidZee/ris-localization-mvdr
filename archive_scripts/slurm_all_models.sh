#!/bin/bash
#SBATCH --job-name=l8_all_models
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/l8_all_models_%j.out
#SBATCH --error=logs/l8_all_models_%j.err

echo "🚀 Training all 3 models (Hybrid, DCD, NFSSN) with L=8 on GPU node"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo "Time: $(date)"

cd /home/tahit/ris/MainMusic

# Create logs directory
mkdir -p logs

echo "📊 Step 1: Converting features for DCD/NFSSN..."
python convert_L8_features_fast.py

echo "🎯 Step 2: Training Hybrid model..."
python run_final_training.py

echo "🔬 Step 3: Training DCD model..."
python train_dcd_cluster.py \
    --root results_final_L8/baselines/features_dcd_nf \
    --epochs 70 \
    --bs 64 \
    --lr 3e-4 \
    --dtype float32 \
    --workers 4

echo "🧠 Step 4: Training NFSSN model..."
python train_nfssn_cluster.py \
    --root results_final_L8/baselines/features_dcd_nf \
    --epochs 70 \
    --bs 64 \
    --lr 3e-4 \
    --dtype float32 \
    --workers 4

echo "✅ All model training completed at $(date)"


