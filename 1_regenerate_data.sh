#!/bin/bash
# Step 1: Regenerate Training Data with M_beams=32, L=16
# Uses 2D vertical-tilted 4×8 DFT codebook + measured SNR

set -e  # Exit on error

cd "$(dirname "$0")"

echo "========================================="
echo "Step 1: Data Regeneration"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - L (snapshots): 16"
echo "  - M_beams (codebook size): 32 (4×8 vertical-tilted)"
echo "  - M_BS (BS antennas): 16"
echo "  - N (RIS elements): 144 (12×12 UPA)"
echo "  - Codebook: 2D separable DFT"
echo "  - SNR: Measured (post-channel)"
echo ""
echo "Dataset:"
echo "  - Train: 80,000 samples"
echo "  - Val: 16,000 samples"
echo "  - Test: 4,000 samples"
echo ""
echo "Output: data_shards_M32_L16/"
echo ""

# Confirm
read -p "Start data generation? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Data generation cancelled."
    exit 1
fi

echo ""
echo "Starting data generation..."
echo ""

# Run data generation
python -m ris_pytorch_pipeline.ris_pipeline pregen \
  --n_train 80000 \
  --n_val 16000 \
  --n_test 4000 \
  --data_root data_shards_M32_L16 \
  --seed 42

echo ""
echo "========================================="
echo "Data Generation Complete!"
echo "========================================="
echo ""
echo "Verifying generated data..."
python -c "
import numpy as np
import sys

try:
    # Check train shard
    d = np.load('data_shards_M32_L16/train_000.npz', allow_pickle=True)
    print(f'✓ Train shard loaded')
    print(f'  - Codes shape: {d[\"codes\"].shape}')  # Should be (L, N, 2) = (16, 144, 2)
    print(f'  - SNR range: [{d[\"snr_db\"].min():.1f}, {d[\"snr_db\"].max():.1f}] dB')
    print(f'  - θ range: [{d[\"θ\"].min():.3f}, {d[\"θ\"].max():.3f}] rad')
    
    # Check val shard
    d = np.load('data_shards_M32_L16/val_000.npz', allow_pickle=True)
    print(f'✓ Val shard loaded')
    print(f'  - Codes shape: {d[\"codes\"].shape}')
    
    print('')
    print('✓ Data generation successful!')
    print('')
    print('Expected:')
    print('  - Codes: (16, 144, 2) - L=16 snapshots, N=144 RIS elements')
    print('  - SNR: Approximately [-5, 20] dB')
    print('  - θ: [-π/6, π/6] rad (±30°)')
    print('')
    print('Next step: Run 2_train_model.sh')
    
except Exception as e:
    print(f'✗ Verification failed: {e}')
    sys.exit(1)
"

echo ""

