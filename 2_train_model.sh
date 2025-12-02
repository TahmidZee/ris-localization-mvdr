#!/bin/bash
# Step 2: Train Model with All Fixes
# Uses regenerated data with M_beams=32, 2D codebook, measured SNR

set -e  # Exit on error

cd "$(dirname "$0")"

echo "========================================="
echo "Step 2: Model Training"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Data: data_shards_M32_L16/"
echo "  - Epochs: 60"
echo "  - Batch size: 128"
echo "  - Optimizer: AdamW"
echo "  - Learning rate: 3e-4"
echo "  - Features:"
echo "    ✓ 2D DFT codebook (4×8 vertical-tilted)"
echo "    ✓ Measured SNR"
echo "    ✓ θ loss emphasis (1.5×)"
echo "    ✓ Auxiliary angle head"
echo "    ✓ Convex shrinkage"
echo ""
echo "Results: results_M32_L16_production/"
echo ""

# Check data exists
if [ ! -d "data_shards_M32_L16" ]; then
    echo "Error: data_shards_M32_L16/ not found!"
    echo "Run 1_regenerate_data.sh first."
    exit 1
fi

# Confirm
read -p "Start training? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Training cancelled."
    exit 1
fi

echo ""
echo "Starting training..."
echo ""

# Run training
python -m ris_pytorch_pipeline.ris_pipeline train \
  --use_shards \
  --data_root data_shards_M32_L16 \
  --epochs 60 \
  --n_train 80000 \
  --n_val 16000 \
  --batch_size 128 \
  --results_dir results_M32_L16_production \
  --device cuda

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo ""
echo "Results saved to: results_M32_L16_production/"
echo ""
echo "  - Checkpoints: results_M32_L16_production/checkpoints/"
echo "  - Logs: results_M32_L16_production/logs/"
echo "  - Best model: results_M32_L16_production/checkpoints/best.pt"
echo ""
echo "Next step: Run 3_evaluate_model.sh"
echo ""

