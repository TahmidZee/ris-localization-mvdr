#!/bin/bash
# Step 1: Regenerate data with M_beams=32, L=16, measured SNR, strided beam selection
# Expected time: 1 day (automated)
# Output: data_shards_M32_L16/

set -e

cd "$(dirname "$0")"

echo "========================================="
echo "Step 1: Data Regeneration"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - L (snapshots): 16"
echo "  - M_beams (codebook): 32 (4×8 vertical-tilted)"
echo "  - M_BS (BS antennas): 16"
echo "  - Beam selection: Strided (balanced coverage)"
echo "  - SNR: Measured (post-channel/codes)"
echo "  - Output: data_shards_M32_L16/"
echo ""

# Check if ris_pytorch_pipeline exists
if [ ! -d "ris_pytorch_pipeline" ]; then
    echo "❌ ERROR: ris_pytorch_pipeline directory not found!"
    echo "Please ensure you're in the correct directory."
    exit 1
fi

# Check if old data exists
if [ -d "data_shards_M32_L16" ]; then
    echo "⚠️  WARNING: data_shards_M32_L16 already exists!"
    echo "This will be overwritten. Continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    echo "Removing old data..."
    rm -rf data_shards_M32_L16
fi

echo "Starting data generation..."
echo "This will take approximately 1 day."
echo ""

# Set environment variables for the data generation
export PYTHONPATH="${PWD}/ris_pytorch_pipeline:${PYTHONPATH}"

# Run the data generation
python -c "
import sys
sys.path.append('ris_pytorch_pipeline')
from ris_pytorch_pipeline.dataset import prepare_split_shards, set_sampling_overrides_from_cfg
from ris_pytorch_pipeline.configs import SysConfig, ModelConfig
from pathlib import Path

print('Initializing configuration...')
sys_cfg = SysConfig()
mdl_cfg = ModelConfig()

print('Setting M_beams=32 for 2D DFT codebook...')
sys_cfg.M_BEAMS_TARGET = 32

print('Building 2D DFT codebook...')
# The codebook is built in SysConfig.__init__() and stored as sys_cfg.RIS_2D_DFT_COLS
print(f'  - Codebook shape: {sys_cfg.RIS_2D_DFT_COLS.shape}')
print(f'  - M_beams: {sys_cfg.M_BEAMS_TARGET}')

print('Setting up sampling overrides...')
set_sampling_overrides_from_cfg(mdl_cfg)

print('Generating data shards...')
print('  - Train: 80,000 samples')
print('  - Val: 16,000 samples')
print('  - Test: 4,000 samples')
print('  - L=16 snapshots')
print('  - M_beams=64 codebook (8×8 balanced)')
print('  - Strided beam selection')
print('  - Measured SNR labels')
print('')

prepare_split_shards(
    root_dir=Path('data_shards_M64_L16'),
    n_train=80000,
    n_val=16000,
    n_test=4000,
    shard_size=25000,
    seed=42
)

print('')
print('✅ Data generation complete!')
print('Output: data_shards_M64_L16/')
"

echo ""
echo "========================================="
echo "Data Generation Complete!"
echo "========================================="
echo ""
echo "Generated:"
echo "  - data_shards_M32_L16/train/ (80,000 samples)"
echo "  - data_shards_M32_L16/val/ (16,000 samples)"
echo "  - data_shards_M32_L16/test/ (4,000 samples)"
echo ""
echo "Features:"
echo "  - L=16 snapshots"
echo "  - M_beams=32 codebook (4×8 vertical-tilted)"
echo "  - Strided beam selection (balanced coverage)"
echo "  - Measured SNR labels (post-channel/codes)"
echo "  - 2D separable DFT codebook"
echo ""
echo "Next step: Run ./02_train_model.bash"
echo ""
