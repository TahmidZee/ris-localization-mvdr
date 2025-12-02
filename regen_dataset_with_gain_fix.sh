#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  REGENERATING DATASET WITH SIGNAL GAIN FIX                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "🔧 Fix Applied: SIGNAL_GAIN = 1000.0 in dataset.py"
echo "   This boosts y from O(1e-3) to O(1) for numerical stability"
echo "   SNR is preserved (noise scales with signal power)"
echo ""

read -p "⚠️  This will DELETE existing dataset. Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "❌ Aborted"
    exit 1
fi

echo ""
echo "🗑️  Removing old dataset..."
rm -rf data_shards_M64_L16

echo "✅ Old dataset removed"
echo ""
echo "🔄 Regenerating dataset (this will take ~30-40 minutes)..."
echo ""

python -m ris_pytorch_pipeline.ris_pipeline regen

echo ""
echo "✅ Dataset regeneration complete!"
echo ""
echo "📊 Verifying signal magnitudes..."

python << 'PYEOF'
import numpy as np

z = np.load("data_shards_M64_L16/train/shard_000.npz")

print("\n" + "="*70)
print("SIGNAL MAGNITUDE VERIFICATION")
print("="*70)

for i in range(5):
    y = z['y'][i]
    y_cplx = y[:, :, 0] + 1j * y[:, :, 1]
    snr_db = z['snr'][i]
    y_norm = np.linalg.norm(y_cplx)
    
    status = "✅" if 0.1 < y_norm < 100 else "❌"
    print(f"{status} Sample {i}: SNR={snr_db:5.2f}dB, ||y||={y_norm:.2f}")

print("\n" + "="*70)
print("Expected: 0.1 < ||y|| < 100 (reasonable for LS inversion)")
print("="*70)
PYEOF

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ DATASET REGENERATION COMPLETE!                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Verify classical MUSIC on R_true gives < 2° error"
echo "  2. Launch HPO"

