#!/bin/bash
# Comprehensive cleanup script - removes everything except core pipeline

set -e

cd "$(dirname "$0")"

echo "========================================="
echo "COMPREHENSIVE CLEANUP"
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

echo "Removing backup directories..."
rm -rf backup_* 2>/dev/null || true

echo "Removing old code directories..."
rm -rf AI-Subspace-Methods* 2>/dev/null || true
rm -rf OldCode 2>/dev/null || true
rm -rf env 2>/dev/null || true

echo "Removing Python cache..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Removing old scripts and test files..."
rm -f *.sh 2>/dev/null || true
rm -f test_*.py 2>/dev/null || true
rm -f debug_*.py 2>/dev/null || true
rm -f check_*.py 2>/dev/null || true
rm -f benchmark_*.py 2>/dev/null || true
rm -f *_fixed.py 2>/dev/null || true
rm -f *_old.py 2>/dev/null || true
rm -f *_backup.py 2>/dev/null || true
rm -f convert_*.py 2>/dev/null || true
rm -f train_*.py 2>/dev/null || true
rm -f run_*.py 2>/dev/null || true
rm -f simple_*.py 2>/dev/null || true
rm -f proper_*.py 2>/dev/null || true
rm -f comprehensive_*.py 2>/dev/null || true
rm -f compare_*.py 2>/dev/null || true
rm -f quick_*.py 2>/dev/null || true

echo "Removing documentation files..."
rm -f *.md 2>/dev/null || true

echo "Removing log files..."
rm -f *.log *.out *.err 2>/dev/null || true

echo ""
echo "========================================="
echo "CLEANUP COMPLETE"
echo "========================================="
echo ""
echo "Remaining files:"
ls -la

echo ""
echo "Only keeping:"
echo "  - ris_pytorch_pipeline/ (core code)"
echo "  - New bash scripts (will be created next)"
echo ""
