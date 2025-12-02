#!/bin/bash
# Quick wrapper to check data distribution

cd "$(dirname "$0")"

echo "🔍 Checking Train/Val Data Distribution..."
echo ""

# Run the diagnostic tool
python ris_pytorch_pipeline/check_data_distribution.py \
    --max_samples 10000 \
    --plot \
    --save_plot distribution_comparison.png

echo ""
echo "Done! Check distribution_comparison.png for visual comparison."


