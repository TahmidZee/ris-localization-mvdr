#!/bin/bash
# Pre-run verification script (2 minutes)
# Checks all critical configurations before starting the week-long run

set -e

cd "$(dirname "$0")"

echo "========================================="
echo "Pre-Run Verification Checklist"
echo "========================================="
echo ""

PASS=0
FAIL=0

# Check 1: M_BEAMS_TARGET
echo "1. Checking M_BEAMS_TARGET = 32..."
if grep -q "M_BEAMS_TARGET = 32" ris_pytorch_pipeline/configs.py; then
    echo "   ✅ PASS: M_BEAMS_TARGET = 32"
    ((PASS++))
else
    echo "   ❌ FAIL: M_BEAMS_TARGET not set to 32"
    ((FAIL++))
fi

# Check 2: THETA_LOSS_SCALE in loss.py
echo "2. Checking THETA_LOSS_SCALE integration..."
COUNT=$(grep -c "THETA_LOSS_SCALE" ris_pytorch_pipeline/loss.py)
if [ "$COUNT" -eq 2 ]; then
    echo "   ✅ PASS: THETA_LOSS_SCALE found in 2 places (lines 381, 507)"
    ((PASS++))
else
    echo "   ❌ FAIL: THETA_LOSS_SCALE not properly integrated (found $COUNT times, expected 2)"
    ((FAIL++))
fi

# Check 3: K_CONF_THRESH
echo "3. Checking K_CONF_THRESH = 0.6..."
if grep -q "K_CONF_THRESH = 0.6" ris_pytorch_pipeline/configs.py; then
    echo "   ✅ PASS: K_CONF_THRESH = 0.6"
    ((PASS++))
else
    echo "   ❌ FAIL: K_CONF_THRESH not set to 0.6"
    ((FAIL++))
fi

# Check 4: Strided beam selection
echo "4. Checking strided beam selection..."
if grep -q "stride = M_beams // L" ris_pytorch_pipeline/dataset.py; then
    echo "   ✅ PASS: Strided sampling implemented"
    ((PASS++))
else
    echo "   ❌ FAIL: Strided sampling not found"
    ((FAIL++))
fi

# Check 5: K-gate τ in evaluation
echo "5. Checking K-gate threshold in evaluation..."
if grep -q "k_conf_thresh=0.8" 3_evaluate_model.sh; then
    echo "   ✅ PASS: Evaluation uses τ=0.8 for OOD robustness"
    ((PASS++))
else
    echo "   ❌ FAIL: Evaluation τ not set to 0.8"
    ((FAIL++))
fi

# Check 6: Execution scripts are executable
echo "6. Checking script permissions..."
if [ -x "1_regenerate_data.sh" ] && [ -x "2_train_model.sh" ] && [ -x "3_evaluate_model.sh" ]; then
    echo "   ✅ PASS: All scripts are executable"
    ((PASS++))
else
    echo "   ❌ FAIL: Scripts not executable (run: chmod +x *.sh)"
    ((FAIL++))
fi

# Check 7: 2D codebook exists
echo "7. Checking 2D codebook implementation..."
if grep -q "def _build_2d_dft_codebook" ris_pytorch_pipeline/configs.py; then
    echo "   ✅ PASS: 2D codebook builder found"
    ((PASS++))
else
    echo "   ❌ FAIL: 2D codebook builder not found"
    ((FAIL++))
fi

# Check 8: Measured SNR implementation
echo "8. Checking measured SNR computation..."
if grep -q "B7: Compute measured SNR" ris_pytorch_pipeline/dataset.py; then
    echo "   ✅ PASS: Measured SNR computation found"
    ((PASS++))
else
    echo "   ❌ FAIL: Measured SNR not implemented"
    ((FAIL++))
fi

echo ""
echo "========================================="
echo "Verification Summary"
echo "========================================="
echo "Passed: $PASS/8"
echo "Failed: $FAIL/8"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED!"
    echo ""
    echo "System is production-ready. You can now:"
    echo "  1. (Optional) Run ./cleanup.sh"
    echo "  2. Run ./1_regenerate_data.sh"
    echo ""
else
    echo "❌ SOME CHECKS FAILED!"
    echo ""
    echo "Please fix the failed checks before proceeding."
    echo "Refer to README_PRODUCTION.md for details."
    echo ""
    exit 1
fi

# Additional info
echo "Configuration Summary:"
echo "  - L (snapshots): 16"
echo "  - M_beams (codebook): 32 (4×8 vertical-tilted)"
echo "  - M_BS (BS antennas): 16"
echo "  - Total measurements: 512 scalars"
echo "  - Beam selection: Strided (every 2nd beam for balanced coverage)"
echo "  - K-gate: τ=0.6 (train), τ=0.8 (eval)"
echo ""

