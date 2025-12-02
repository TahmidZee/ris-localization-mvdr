#!/bin/bash
#SBATCH --job-name=l8_features
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/l8_features_%j.out
#SBATCH --error=logs/l8_features_%j.err

echo "🚀 Starting L=8 feature conversion on GPU node"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo "Time: $(date)"

cd /home/tahit/ris/MainMusic

# Create logs directory
mkdir -p logs

# Feature conversion for DCD/NFSSN
echo "📊 Converting L=8 features for DCD/NFSSN..."
python convert_L8_features_fast.py

echo "✅ Feature conversion completed at $(date)"


