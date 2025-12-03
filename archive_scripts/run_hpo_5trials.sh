#!/bin/bash
# Run 5-trial HPO with full logging
# Date: October 14, 2025

set -e  # Exit on error

cd /home/tahit/ris/MainMusic

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="results_final_L16_12x12/hpo/logs"
mkdir -p "$LOGDIR"

# Log file with timestamp
LOGFILE="${LOGDIR}/hpo_5trials_${TIMESTAMP}.log"

echo "================================================================================================"
echo "HPO 5-Trial Run - Post Bug Fixes"
echo "================================================================================================"
echo "Timestamp: $TIMESTAMP"
echo "Log file: $LOGFILE"
echo ""
echo "Configuration:"
echo "  - Trials: 5"
echo "  - Epochs per trial: 12"
echo "  - Search space: wide"
echo "  - Early stopping patience: 5"
echo ""
echo "Critical fixes applied:"
echo "  ✓ K-head in optimizer"
echo "  ✓ lam_cov weight applied"
echo "  ✓ Curriculum modulates all weights"
echo "  ✓ Hungarian metrics integrated"
echo "  ✓ Debug/validation parity verified"
echo ""
echo "Expected results:"
echo "  - K-acc: 0.45-0.65 (up from 0.20)"
echo "  - Val loss: 1.0-1.3 (down from 1.67)"
echo "  - K-CE: 1.2-1.4 (down from 1.61)"
echo ""
echo "Starting HPO in 3 seconds..."
echo "Press Ctrl+C to cancel"
sleep 3
echo ""
echo "================================================================================================"

# Run HPO with logging (both to console and file)
python -u -m ris_pytorch_pipeline.ris_pipeline hpo \
  --trials 5 \
  --hpo-epochs 12 \
  --space wide \
  --early-stop-patience 5 \
  2>&1 | tee "$LOGFILE"

# Capture exit code
EXIT_CODE=$?

echo ""
echo "================================================================================================"
echo "HPO Run Complete"
echo "================================================================================================"
echo "Exit code: $EXIT_CODE"
echo "Log saved to: $LOGFILE"
echo ""

# Show summary
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ HPO completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Check results: cat results_final_L16_12x12/hpo/best.json"
    echo "  2. Review log: cat $LOGFILE | grep -E '(K-acc|Val loss|K-CE|composite)'"
    echo "  3. Check trials: cat results_final_L16_12x12/hpo/*_trials.csv"
else
    echo "✗ HPO failed with exit code $EXIT_CODE"
    echo "Check log for errors: cat $LOGFILE"
fi

echo ""
echo "Log file: $LOGFILE"
echo "================================================================================================"

exit $EXIT_CODE

