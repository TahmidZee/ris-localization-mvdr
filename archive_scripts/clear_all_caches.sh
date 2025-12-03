#!/bin/bash
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║              🧹 COMPREHENSIVE CACHE CLEARING                            ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "STEP 1: Python Cache Clearing"
echo "════════════════════════════════════════════════════════════════════════════"

# Remove __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "✅ __pycache__ directories cleared"

# Remove .pyc files
echo "Removing .pyc files..."
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "✅ .pyc files cleared"

# Remove .pyo files
echo "Removing .pyo files..."
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "✅ .pyo files cleared"

echo ""
echo "STEP 2: GPU Cache Clearing"
echo "════════════════════════════════════════════════════════════════════════════"

python << 'PYTHON_EOF'
import torch
import gc

print("Checking GPU availability...")
if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    print("✅ GPU cache cleared")
    
    # Force garbage collection
    gc.collect()
    print("✅ Garbage collection completed")
else:
    print("ℹ️  No GPU available - skipping GPU cache clear")
    gc.collect()
    print("✅ CPU garbage collection completed")
PYTHON_EOF

echo ""
echo "STEP 3: HPO Database and Log Clearing"
echo "════════════════════════════════════════════════════════════════════════════"

# Remove old HPO databases
echo "Removing old HPO databases..."
rm -f results_final/hpo/hpo.db* 2>/dev/null || true
rm -f results_final/hpo/*.db 2>/dev/null || true
echo "✅ HPO databases cleared"

# Remove old log files
echo "Removing old log files..."
rm -f results_final/hpo/*.log 2>/dev/null || true
echo "✅ Old log files cleared"

# Remove PID files
echo "Removing PID files..."
rm -f results_final/hpo/hpo.pid 2>/dev/null || true
echo "✅ PID files cleared"

echo ""
echo "STEP 4: Model Checkpoint Clearing (Optional)"
echo "════════════════════════════════════════════════════════════════════════════"
echo "ℹ️  Keeping model checkpoints (uncomment to remove):"
echo "# rm -f results_final/models/*.pt 2>/dev/null || true"
echo "# rm -f checkpoints/*.pt 2>/dev/null || true"

echo ""
echo "STEP 5: Temporary Files Clearing"
echo "════════════════════════════════════════════════════════════════════════════"

# Remove any temporary files
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.temp" -delete 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true
echo "✅ Temporary files cleared"

echo ""
echo "STEP 6: Verification"
echo "════════════════════════════════════════════════════════════════════════════"

echo "Remaining cache files:"
PYCACHE_COUNT=$(find . -name "__pycache__" -type d | wc -l)
PYC_COUNT=$(find . -name "*.pyc" | wc -l)
echo "  __pycache__ directories: $PYCACHE_COUNT"
echo "  .pyc files: $PYC_COUNT"

if [ $PYCACHE_COUNT -eq 0 ] && [ $PYC_COUNT -eq 0 ]; then
    echo "✅ All Python cache cleared successfully!"
else
    echo "⚠️  Some cache files remain - may need manual cleanup"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "🎯 CACHE CLEARING COMPLETE - READY FOR FRESH HPO RUN!"
echo "═══════════════════════════════════════════════════════════════════════════"
