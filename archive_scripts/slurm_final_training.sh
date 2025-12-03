#!/bin/bash
#SBATCH --job-name=l16_12x12_final
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/l16_12x12_final_%j.out
#SBATCH --error=logs/l16_12x12_final_%j.err

echo "🎯 Starting L=16 12x12 final training on GPU node"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo "Time: $(date)"

cd /home/tahit/ris/MainMusic

# Create logs directory
mkdir -p logs

# Final training with HPO results
echo "🚀 Running final training with all improvements..."
python run_final_training.py

echo "✅ Final training completed at $(date)"
