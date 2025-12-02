#!/bin/bash
# Step 3: Train the L=16 model with M_beams=32 (using HPO results)
# Expected time: 3-5 days (automated)
# Output: results_M32_L16_production/

set -e

cd "$(dirname "$0")"

echo "========================================="
echo "Step 3: Model Training (with HPO results)"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - L (snapshots): 16"
echo "  - M_beams (codebook): 32 (4×8 vertical-tilted)"
echo "  - M_BS (BS antennas): 16"
echo "  - Training epochs: 60"
echo "  - Batch size: 32"
echo "  - Learning rate: 1e-3"
echo "  - Output: results_M32_L16_production/"
echo ""

# Check if data exists
if [ ! -d "data_shards_M32_L16" ]; then
    echo "❌ ERROR: data_shards_M32_L16 not found!"
    echo "Please run ./01_regenerate_data.bash first."
    exit 1
fi

# Check if HPO results exist
if [ ! -f "results_final_L16_12x12/hpo/best.json" ]; then
    echo "❌ ERROR: HPO results not found!"
    echo "Expected: results_final_L16_12x12/hpo/best.json"
    echo "Please run ./02_hpo.bash first."
    exit 1
fi

# Check if ris_pytorch_pipeline exists
if [ ! -d "ris_pytorch_pipeline" ]; then
    echo "❌ ERROR: ris_pytorch_pipeline directory not found!"
    exit 1
fi

# Check if old results exist
if [ -d "results_M32_L16_production" ]; then
    echo "⚠️  WARNING: results_M32_L16_production already exists!"
    echo "This will be overwritten. Continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    echo "Removing old results..."
    rm -rf results_M32_L16_production
fi

echo "Starting model training..."
echo "This will take approximately 3-5 days."
echo ""

# Set environment variables
export PYTHONPATH="${PWD}/ris_pytorch_pipeline:${PYTHONPATH}"

# Run the training
python -c "
import sys
sys.path.append('ris_pytorch_pipeline')
from ris_pytorch_pipeline.train import main as train_main
from ris_pytorch_pipeline.configs import SysConfig, ModelConfig
import torch

print('Initializing configuration...')
sys_cfg = SysConfig()
mdl_cfg = ModelConfig()

print('Setting M_beams=32 for 2D DFT codebook...')
sys_cfg.M_BEAMS_TARGET = 32

print('Setting data paths for L=16 M_beams=32 system...')
sys_cfg.DATA_SHARDS_DIR = 'data_shards_M32_L16'

print('Loading HPO results...')
import json
with open('results_final_L16_12x12/hpo/best.json', 'r') as f:
    hpo_result = json.load(f)
    best_params = hpo_result['params']

print('Training configuration:')
print('  - L=16 snapshots')
print('  - M_beams=32 codebook (4×8 vertical-tilted)')
print('  - M_BS=16 antennas')
print(f'  - Batch size: {best_params[\"batch_size\"]} (from HPO)')
print(f'  - Learning rate: {best_params[\"lr\"]} (from HPO)')
print(f'  - D_MODEL: {best_params[\"D_MODEL\"]} (from HPO)')
print(f'  - NUM_HEADS: {best_params[\"NUM_HEADS\"]} (from HPO)')
print(f'  - Dropout: {best_params[\"dropout\"]} (from HPO)')
print(f'  - Range grid: {best_params[\"range_grid\"]} (from HPO)')
print(f'  - Newton iterations: {best_params[\"newton_iter\"]} (from HPO)')
print('  - Epochs: 60')
print('  - THETA_LOSS_SCALE: 1.5')
print('  - K_CONF_THRESH: 0.6')
print('')

# Apply HPO results to model config
mdl_cfg.D_MODEL = best_params['D_MODEL']
mdl_cfg.NUM_HEADS = best_params['NUM_HEADS']
mdl_cfg.DROPOUT = best_params['dropout']
mdl_cfg.LR_INIT = best_params['lr']
mdl_cfg.BATCH_SIZE = best_params['batch_size']
mdl_cfg.INFERENCE_GRID_SIZE_RANGE = best_params['range_grid']
mdl_cfg.NEWTON_ITER = best_params['newton_iter']
mdl_cfg.NEWTON_LR = best_params['newton_lr']
mdl_cfg.SHRINK_BASE_ALPHA = best_params['shrink_alpha']
mdl_cfg.SOFTMAX_TAU = best_params['softmax_tau']

# Training arguments (using HPO results)
args = type('Args', (), {
    'data_root': 'data_shards_M32_L16',
    'output_dir': 'results_M32_L16_production',
    'batch_size': best_params['batch_size'],
    'learning_rate': best_params['lr'],
    'epochs': 60,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'pin_memory': True,
    'num_workers': 4,
    'save_every': 10,
    'validate_every': 5,
    'early_stopping_patience': 15,
    'use_swa': True,
    'swa_start': 40,
    'swa_lr': best_params['lr'] * 0.1,  # 10% of main LR
    'use_ema': True,
    'ema_decay': 0.999,
    'gradient_clip': 1.0,  # Default value
    'weight_decay': 1e-4,  # Default value
    'scheduler': 'cosine',  # Default value
    'warmup_epochs': 5,
    'seed': 42
})()

print('Starting training...')
# Create trainer and run training
trainer = Trainer(from_hpo=False)  # Don't apply HPO overrides since we set them manually
best_val = trainer.fit(
    epochs=60,
    use_shards=True,
    n_train=80000,  # Full training set
    n_val=16000,    # Full validation set
    gpu_cache=True,
    grad_accumulation=1,
    early_stop_patience=15
)

print('')
print('✅ Training complete!')
print('Output: results_M32_L16_production/')
"

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo ""
echo "Generated:"
echo "  - results_M32_L16_production/checkpoints/best.pt"
echo "  - results_M32_L16_production/logs/"
echo "  - results_M32_L16_production/configs/"
echo ""
echo "Model features:"
echo "  - L=16 snapshots"
echo "  - M_beams=32 codebook (4×8 vertical-tilted)"
echo "  - THETA_LOSS_SCALE=1.5 (elevation emphasis)"
echo "  - K_CONF_THRESH=0.6 (in-distribution)"
echo "  - SWA + EMA training"
echo "  - Cosine annealing scheduler"
echo ""
echo "Next step: Run ./03_evaluate_model.bash"
echo ""
