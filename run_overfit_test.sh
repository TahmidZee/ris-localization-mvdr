#!/bin/bash
# Overfit Test Runner with Logging
# Usage: ./run_overfit_test.sh

echo "Starting Overfit Test with Logging..."
echo "======================================"

# Create timestamp for unique log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="overfit_test_${TIMESTAMP}.log"

echo "Log file: ${LOG_FILE}"
echo ""

# Run the overfit test and capture all output
cd /home/tahit/ris/MainMusic
python test_overfit.py 2>&1 | tee "${LOG_FILE}"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================"
echo "Overfit Test Complete"
echo "Exit code: ${EXIT_CODE}"
echo "Log saved to: ${LOG_FILE}"

# Show last 20 lines of log for quick review
echo ""
echo "Last 20 lines of output:"
echo "------------------------"
tail -20 "${LOG_FILE}"

# Interpret results
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✅ OVERFIT TEST PASSED"
    echo "   → Ready for HPO"
    echo "   → Check ${LOG_FILE} for full details"
else
    echo ""
    echo "❌ OVERFIT TEST FAILED"
    echo "   → Fix issues before HPO"
    echo "   → Check ${LOG_FILE} for error details"
fi

exit ${EXIT_CODE}


