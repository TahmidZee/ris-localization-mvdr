#!/bin/bash

# HPO with comprehensive logging
# Usage: ./run_hpo_with_logging.sh

echo "🚀 Starting HPO with comprehensive logging..."
echo "📅 Start time: $(date)"
echo "💾 Memory before HPO:"
free -h

# Create logs directory if it doesn't exist
mkdir -p logs

# Run HPO with comprehensive logging
echo "🔧 Running HPO with 3 trials, 12 epochs, early-stop patience 8..."
python -m ris_pytorch_pipeline.ris_pipeline hpo --trials 3 --hpo-epochs 12 --early-stop-patience 8 \
    2>&1 | tee logs/hpo_$(date +%Y%m%d_%H%M%S).log

# Capture exit code
HPO_EXIT_CODE=$?

echo "📅 End time: $(date)"
echo "💾 Memory after HPO:"
free -h

if [ $HPO_EXIT_CODE -eq 0 ]; then
    echo "✅ HPO completed successfully!"
else
    echo "❌ HPO failed with exit code: $HPO_EXIT_CODE"
fi

echo "📊 Final memory status:"
free -h
echo "🔍 GPU memory status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "No GPU available"

echo "📁 Log files created in: logs/"
ls -la logs/
