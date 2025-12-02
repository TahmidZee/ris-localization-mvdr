#!/bin/bash
# Step 4: Evaluate the trained model
# Expected time: 30 minutes
# Output: Stratified metrics by SNR and range bins

set -e

cd "$(dirname "$0")"

echo "========================================="
echo "Step 4: Model Evaluation"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Model: results_M32_L16_production/checkpoints/best.pt"
echo "  - Data: data_shards_M32_L16/val/"
echo "  - Samples: 2,000"
echo "  - K-gate: τ=0.8 (OOD robustness)"
echo "  - Device: CUDA (if available)"
echo ""

# Check if model exists
if [ ! -f "results_M32_L16_production/checkpoints/best.pt" ]; then
    echo "❌ ERROR: Model checkpoint not found!"
    echo "Expected: results_M32_L16_production/checkpoints/best.pt"
    echo "Please run ./02_train_model.bash first."
    exit 1
fi

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

echo "Starting evaluation..."
echo "This will take approximately 30 minutes."
echo ""

# Set environment variables
export PYTHONPATH="${PWD}/ris_pytorch_pipeline:${PYTHONPATH}"

# Run the evaluation
python -c "
import sys
sys.path.append('ris_pytorch_pipeline')
from ris_pytorch_pipeline.benchmark import load_model
from ris_pytorch_pipeline.dataset import ShardedDataset
import torch

print('Loading model...')
model = load_model('results_M32_L16_production/checkpoints/best.pt')

print('Loading validation dataset...')
dataset = ShardedDataset('data_shards_M32_L16', split='val', max_samples=2000)

print('Using device: cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('')
print('Running stratified evaluation...')
print('K-gate threshold: τ = 0.8 (OOD robustness)')
print('')

# Import and run stratified evaluation
from stratified_evaluation import stratified_evaluation

results = stratified_evaluation(
    model=model,
    dataset=dataset,
    n_samples=2000,
    device=device,
    k_conf_thresh=0.8
)

print('')
print('✅ Evaluation complete!')
print('')
print('Results summary:')
print('================')

# Print overall metrics
overall = results['overall']
print(f'Overall Performance:')
print(f'  φ RMSE: {overall[\"phi_rmse\"]:.3f}°')
print(f'  θ RMSE: {overall[\"theta_rmse\"]:.3f}°')
print(f'  r RMSE: {overall[\"r_rmse\"]:.3f}m')
print(f'  K accuracy: {overall[\"k_accuracy\"]:.1%}')
print('')

# Print stratified metrics
print('Stratified by SNR:')
for snr_range, metrics in results['by_snr'].items():
    print(f'  SNR {snr_range[0]}-{snr_range[1]} dB:')
    print(f'    φ: {metrics[\"phi_rmse\"]:.3f}° (med: {metrics[\"phi_median\"]:.3f}°)')
    print(f'    θ: {metrics[\"theta_rmse\"]:.3f}° (med: {metrics[\"theta_median\"]:.3f}°)')
    print(f'    r: {metrics[\"r_rmse\"]:.3f}m (med: {metrics[\"r_median\"]:.3f}m)')
    print(f'    K: {metrics[\"k_accuracy\"]:.1%}')
print('')

print('Stratified by Range:')
for range_bin, metrics in results['by_range'].items():
    print(f'  Range {range_bin[0]}-{range_bin[1]}m:')
    print(f'    φ: {metrics[\"phi_rmse\"]:.3f}° (med: {metrics[\"phi_median\"]:.3f}°)')
    print(f'    θ: {metrics[\"theta_rmse\"]:.3f}° (med: {metrics[\"theta_median\"]:.3f}°)')
    print(f'    r: {metrics[\"r_rmse\"]:.3f}m (med: {metrics[\"r_median\"]:.3f}m)')
    print(f'    K: {metrics[\"k_accuracy\"]:.1%}')
print('')

# Check acceptance gates
print('Acceptance Gates (SNR ≥ 0 dB):')
snr_0_20 = results['by_snr'][(0, 20)]
phi_ok = snr_0_20['phi_rmse'] <= 0.6
theta_ok = snr_0_20['theta_rmse'] <= 0.9
r_ok = snr_0_20['r_rmse'] <= 0.8
k_ok = snr_0_20['k_accuracy'] >= 0.90

print(f'  φ ≤ 0.6°: {snr_0_20[\"phi_rmse\"]:.3f}° {\"✅\" if phi_ok else \"❌\"}')
print(f'  θ ≤ 0.9°: {snr_0_20[\"theta_rmse\"]:.3f}° {\"✅\" if theta_ok else \"❌\"}')
print(f'  r ≤ 0.8m: {snr_0_20[\"r_rmse\"]:.3f}m {\"✅\" if r_ok else \"❌\"}')
print(f'  K ≥ 90%: {snr_0_20[\"k_accuracy\"]:.1%} {\"✅\" if k_ok else \"❌\"}')
print('')

if phi_ok and theta_ok and r_ok and k_ok:
    print('🎉 ALL ACCEPTANCE GATES PASSED!')
    print('Sub-1° performance achieved!')
else:
    print('⚠️  Some acceptance gates failed.')
    print('Check the results above for details.')
"

echo ""
echo "========================================="
echo "Evaluation Complete!"
echo "========================================="
echo ""
echo "Results saved to console output above."
echo ""
echo "Acceptance Gates (SNR ≥ 0 dB):"
echo "  - φ RMSE ≤ 0.6°"
echo "  - θ RMSE ≤ 0.9°"
echo "  - r RMSE ≤ 0.8m"
echo "  - K accuracy ≥ 90%"
echo ""
echo "If all gates pass: Sub-1° performance achieved! 🎉"
echo ""
