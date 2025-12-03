#!/bin/bash
# HPO Launch Script with Complete Logging and Cache Clearing
# Usage: ./run_hpo.sh

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║              🚀 HPO LAUNCH - FRESH START                                 ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Navigate to project directory
cd /home/tahit/ris/MainMusic

echo "STEP 1: Clearing caches..."
echo "════════════════════════════════════════════════════════════════════════════"

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Clear GPU cache
python << 'PYTHON_EOF'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✅ GPU cache cleared")
else:
    print("ℹ️  No GPU available")
PYTHON_EOF

echo "✅ All caches cleared"
echo ""

echo "STEP 2: Starting HPO..."
echo "════════════════════════════════════════════════════════════════════════════"

# Create HPO directory
mkdir -p results_final/hpo

# Generate log filename with timestamp
LOGFILE="results_final/hpo/hpo_$(date +%Y%m%d_%H%M%S).log"

# Run HPO in background with full logging (stdout + stderr)
nohup python -m ris_pytorch_pipeline.ris_pipeline hpo \
  --trials 20 \
  --hpo-epochs 40 \
  --early-stop-patience 15 \
  > "$LOGFILE" 2>&1 &

# Save PID
PID=$!
echo $PID > results_final/hpo/hpo.pid

echo "✅ HPO started successfully!"
echo ""
echo "Details:"
echo "  PID:  $PID"
echo "  Log:  $LOGFILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOGFILE"
echo ""
echo "Check status:"
echo "  ps -p $PID"
echo ""
echo "Stop HPO:"
echo "  kill $PID"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "🎯 HPO is running! Expected runtime: 8-12 hours"
echo "═══════════════════════════════════════════════════════════════════════════"


