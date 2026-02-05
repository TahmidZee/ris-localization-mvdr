#!/bin/bash
# Cleanup script for fresh training run after architectural changes
# This clears checkpoints, caches, and resume state to prevent conflicts

set -e

# Configuration (matches configs.py)
RESULTS_DIR="results_M64_N256_L64"
CKPT_DIR="${RESULTS_DIR}/checkpoints"
LOGS_DIR="${RESULTS_DIR}/logs"
HPO_DIR="${RESULTS_DIR}/hpo"

echo "=========================================="
echo "CLEANUP FOR FRESH TRAINING"
echo "=========================================="
echo ""
echo "This will remove:"
echo "  - Checkpoints (*.pt files in ${CKPT_DIR}/)"
echo "  - Training resume state (train_state.pt)"
echo "  - Python bytecode caches (__pycache__)"
echo "  - Old log files (optional)"
echo ""
echo "It will NOT remove:"
echo "  - Data shards (data_shards_*/)"
echo "  - HPO database (${HPO_DIR}/hpo.db)"
echo "  - Git history"
echo ""

# Ask for confirmation
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Optional: Backup checkpoints before deletion
read -p "Backup checkpoints before deletion? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    BACKUP_DIR="checkpoints_backup_$(date +%Y%m%d_%H%M%S)"
    if [ -d "$CKPT_DIR" ] && [ "$(ls -A $CKPT_DIR 2>/dev/null)" ]; then
        echo "Creating backup: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
        cp -r "$CKPT_DIR"/* "$BACKUP_DIR/" 2>/dev/null || true
        echo "✅ Backup created: $BACKUP_DIR"
    else
        echo "⚠️  No checkpoints to backup"
    fi
fi

# 1. Clear checkpoints
echo ""
echo "1. Clearing checkpoints..."
if [ -d "$CKPT_DIR" ]; then
    rm -f "$CKPT_DIR"/*.pt 2>/dev/null || true
    rm -f "$CKPT_DIR"/train_state.pt 2>/dev/null || true
    echo "✅ Cleared: $CKPT_DIR/*.pt"
else
    echo "⚠️  Checkpoint directory not found: $CKPT_DIR"
fi

# 2. Clear Python bytecode caches
echo ""
echo "2. Clearing Python bytecode caches..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "✅ Cleared: __pycache__ directories and .pyc/.pyo files"

# 3. Clear GPU cache files (if they exist as separate files)
echo ""
echo "3. Checking for GPU cache files..."
# GPU cache is typically built on-the-fly, but check for any cached files
if [ -d "$RESULTS_DIR" ]; then
    find "$RESULTS_DIR" -type f -name "*gpu_cache*" -delete 2>/dev/null || true
    find "$RESULTS_DIR" -type f -name "*cache*.pt" -delete 2>/dev/null || true
    echo "✅ Cleared: GPU cache files (if any)"
else
    echo "⚠️  Results directory not found: $RESULTS_DIR"
fi

# 4. Optional: Clear old log files
read -p "Clear old log files in ${LOGS_DIR}/? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "$LOGS_DIR" ]; then
        rm -f "$LOGS_DIR"/*.log 2>/dev/null || true
        echo "✅ Cleared: $LOGS_DIR/*.log"
    else
        echo "⚠️  Logs directory not found: $LOGS_DIR"
    fi
fi

# 5. Clear any .pth cache files in ris_pytorch_pipeline
echo ""
echo "5. Clearing any .pth cache files..."
find ris_pytorch_pipeline -type f -name "*.pth" -delete 2>/dev/null || true
echo "✅ Cleared: .pth cache files (if any)"

# Summary
echo ""
echo "=========================================="
echo "CLEANUP COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. git pull origin fix/covariance-learning-and-slim"
echo "  2. python -m ris_pytorch_pipeline.ris_pipeline train --epochs 10 --use_shards"
echo ""
