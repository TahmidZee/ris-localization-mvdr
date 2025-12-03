#!/bin/bash
#SBATCH --job-name=l16_12x12_hpo
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/l16_12x12_hpo_%j.out
#SBATCH --error=logs/l16_12x12_hpo_%j.err

echo "🔍 Starting L=16 12x12 HPO on GPU node"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo "Time: $(date)"

cd /home/tahit/ris/MainMusic

# Create logs directory
mkdir -p logs

# HPO with budget recommendations
echo "🎯 Running HPO with optimized budget (40 trials, pruning enabled)..."
python -m ris_pytorch_pipeline.hpo \
    --n_trials 40 \
    --epochs_per_trial 12 \
    --space wide

echo "✅ HPO completed at $(date)"
