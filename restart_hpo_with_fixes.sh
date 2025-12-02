#!/bin/bash
# Restart HPO with optimization fixes
# Date: 2025-10-10

echo "======================================================================="
echo "RESTARTING HPO WITH OPTIMIZATION FIXES"
echo "======================================================================="
echo ""
echo "Changes applied:"
echo "  - Learning rate: 3e-4 to 1e-3 (was 1e-4 to 5e-4)"
echo "  - Batch size: 32-64 (was 64-128)"
echo "  - Grad accumulation: 1 (was 2)"
echo "  - K-loss weight: 0.6-1.0 (was 0.25-0.45)"
echo "  - Cov weight: 0.2-0.4 (was 0.4-0.7)"
echo "  - Dropout: 0.15-0.35 (was 0.25-0.4)"
echo ""
echo "Expected: Val loss should drop below 1.0 by epoch 5"
echo "======================================================================="
echo ""

# Navigate to project directory
cd /home/tahit/ris/MainMusic

# Create new study name to start fresh
STUDY_NAME="L16_M64_wide_v2_optimfix"
DB_PATH="results_final_L16_12x12/hpo/hpo_v2.db"

echo "New study: $STUDY_NAME"
echo "Database: $DB_PATH"
echo ""

# Create HPO command
HPO_CMD="python -m ris_pytorch_pipeline.ris_pipeline hpo \
  --study-name $STUDY_NAME \
  --storage sqlite:///$DB_PATH \
  --n-trials 20 \
  --space wide \
  --epochs-per-trial 12 \
  --early-stop-patience 5"

echo "HPO command:"
echo "$HPO_CMD"
echo ""
echo "======================================================================="
echo "INSTRUCTIONS:"
echo "======================================================================="
echo ""
echo "1. Stop current HPO:"
echo "   screen -r ris_train"
echo "   Press Ctrl+C"
echo "   exit"
echo ""
echo "2. Start new HPO in screen:"
echo "   screen -S ris_train_v2 bash"
echo "   cd /home/tahit/ris/MainMusic"
echo "   $HPO_CMD"
echo ""
echo "3. Detach from screen: Ctrl+A then D"
echo ""
echo "4. Monitor progress:"
echo "   screen -r ris_train_v2"
echo ""
echo "======================================================================="
echo "WATCH FOR:"
echo "======================================================================="
echo ""
echo "✓ Trial 0 should use lr ~0.0003-0.001 (higher than before)"
echo "✓ Trial 0 should use batch_size 32 or 64 (smaller than before)"
echo "✓ Trial 0 should use lam_K ~0.6-1.0 (much higher than before)"
echo ""
echo "✓ Epoch 1: val_loss should be < 1.20"
echo "✓ Epoch 3: val_loss should be < 1.00"
echo "✓ Epoch 5: val_loss should be < 0.85"
echo "✓ Epoch 5: K-accuracy should be > 40%"
echo ""
echo "❌ If val_loss still stuck at ~1.28, there's a deeper issue"
echo ""
echo "======================================================================="

