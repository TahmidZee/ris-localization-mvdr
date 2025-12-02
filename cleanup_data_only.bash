#!/bin/bash
# Clean up only old data shards, logs, and checkpoints
# This ensures clean regeneration without conflicts

set -e

cd "$(dirname "$0")"

echo "========================================="
echo "CLEANING OLD DATA ONLY"
echo "========================================="
echo ""

echo "Removing old data shards..."
rm -rf data_shards_* 2>/dev/null || true

echo "Removing old results..."
rm -rf results_* 2>/dev/null || true

echo "Removing old checkpoints..."
rm -rf checkpoints 2>/dev/null || true

echo "Removing old logs..."
rm -rf logs 2>/dev/null || true

echo "Removing any remaining shard directories..."
find . -maxdepth 1 -type d -name "*shard*" -exec rm -rf {} + 2>/dev/null || true

echo "Removing any remaining result directories..."
find . -maxdepth 1 -type d -name "*result*" -exec rm -rf {} + 2>/dev/null || true

echo "Removing any remaining checkpoint directories..."
find . -maxdepth 1 -type d -name "*checkpoint*" -exec rm -rf {} + 2>/dev/null || true

echo "Removing any remaining log directories..."
find . -maxdepth 1 -type d -name "*log*" -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "========================================="
echo "CLEANUP COMPLETE"
echo "========================================="
echo ""

echo "Checking for remaining data directories..."
REMAINING=$(find . -maxdepth 1 -type d | grep -E "(shard|result|checkpoint|log)" | wc -l)

if [ "$REMAINING" -eq 0 ]; then
    echo "✅ SUCCESS: No old data directories found"
    echo ""
    echo "Ready for clean data regeneration:"
    echo "  ./01_regenerate_data.bash"
else
    echo "⚠️  WARNING: $REMAINING old data directories still found:"
    find . -maxdepth 1 -type d | grep -E "(shard|result|checkpoint|log)"
    echo ""
    echo "You may need to remove these manually."
fi

echo ""
echo "Current directory contents:"
ls -la | head -10
echo ""
