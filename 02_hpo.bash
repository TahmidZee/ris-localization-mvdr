#!/bin/bash
# Step 2: Hyperparameter Optimization (HPO)
# Expected time: 1-2 days (automated)
# Output: Best hyperparameters for training

set -e

cd "$(dirname "$0")"

echo "========================================="
echo "Step 2: Hyperparameter Optimization"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - L (snapshots): 16"
echo "  - M_beams (codebook): 32 (4×8 vertical-tilted)"
echo "  - M_BS (BS antennas): 16"
echo "  - HPO trials: 50"
echo "  - Validation samples: 2,000"
echo "  - Output: results_final_L16_12x12/hpo/"
echo ""

# Check if data exists
if [ ! -d "data_shards_M32_L16" ]; then
    echo "❌ ERROR: data_shards_M32_L16 not found!"
    echo "Please run ./01_regenerate_data.bash first."
    exit 1
fi

# Check if ris_pytorch_pipeline exists
if [ ! -d "ris_pytorch_pipeline" ]; then
    echo "❌ ERROR: ris_pytorch_pipeline directory not found!"
    exit 1
fi

# Check if old HPO results exist
if [ -d "results_final_L16_12x12/hpo" ]; then
    echo "⚠️  WARNING: results_final_L16_12x12/hpo already exists!"
    echo "This will be overwritten. Continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    echo "Removing old HPO results..."
    rm -rf results_final_L16_12x12/hpo
fi

echo "Starting hyperparameter optimization..."
echo "This will take approximately 1-2 days."
echo ""

# Set environment variables
export PYTHONPATH="${PWD}/ris_pytorch_pipeline:${PYTHONPATH}"

# Run HPO using the existing optimized code
python -c "
import sys
sys.path.append('ris_pytorch_pipeline')
from ris_pytorch_pipeline.hpo import run_hpo
from ris_pytorch_pipeline.configs import SysConfig, ModelConfig
import torch

print('Initializing configuration...')
sys_cfg = SysConfig()
mdl_cfg = ModelConfig()

print('Setting M_beams=32 for 2D DFT codebook...')
sys_cfg.M_BEAMS_TARGET = 32

print('Setting data paths for L=16 M_beams=32 system...')
sys_cfg.DATA_SHARDS_DIR = 'data_shards_M32_L16'

print('HPO configuration:')
print('  - L=16 snapshots')
print('  - M_beams=32 codebook (4×8 vertical-tilted)')
print('  - M_BS=16 antennas')
print('  - HPO trials: 50')
print('  - Epochs per trial: 15')
print('  - Search space: wide (comprehensive)')
print('  - Optimizations: TPE sampler, median pruning, loss weight tuning')
print('')

print('Starting optimized HPO...')
run_hpo(
    n_trials=50,
    epochs_per_trial=15,
    space='wide',  # Comprehensive search space
    export_csv=True
)

print('')
print('✅ HPO complete!')
print('Best parameters saved to: results_final_L16_12x12/hpo/best.json')
print('All trials saved to: results_final_L16_12x12/hpo/trials.csv')
print('')
"

echo ""
echo "========================================="
echo "HPO Complete!"
echo "========================================="
echo ""
echo "Generated:"
echo "  - results_final_L16_12x12/hpo/best.json"
echo "  - results_final_L16_12x12/hpo/trials.csv"
echo "  - results_final_L16_12x12/hpo/hpo.db"
echo ""
echo "Best parameters found:"
echo "  - D_MODEL: [from HPO]"
echo "  - NUM_HEADS: [from HPO]"
echo "  - Dropout: [from HPO]"
echo "  - Learning rate: [from HPO]"
echo "  - Batch size: [from HPO]"
echo "  - Range grid: [from HPO]"
echo "  - Newton iterations: [from HPO]"
echo "  - Loss weights: [from HPO]"
echo ""
echo "Next step: Run ./03_train_model.bash (will use best HPO parameters)"
echo ""
