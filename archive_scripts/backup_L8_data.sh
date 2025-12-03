#!/bin/bash
# Backup L=8 data before upgrading to L=16 12x12 system

echo "📦 Backing up L=8 data and checkpoints..."

# Create backup directory  
BACKUP_DIR="/home/tahit/ris/MainMusic/backup_L8_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup L=8 results directory if it exists
if [ -d "results_final_L8" ]; then
    echo "📋 Backing up results_final_L8..."
    cp -r results_final_L8 "$BACKUP_DIR/"
    echo "✅ L=8 results backed up to $BACKUP_DIR/results_final_L8"
else
    echo "⚠️  No results_final_L8 directory found"
fi

# Backup any L=8 checkpoints
if [ -d "checkpoints" ]; then
    echo "📋 Backing up checkpoints..."
    cp -r checkpoints "$BACKUP_DIR/"
    echo "✅ Checkpoints backed up to $BACKUP_DIR/checkpoints"
fi

# Create a backup manifest
cat > "$BACKUP_DIR/backup_manifest.txt" << EOF
L=8 System Backup - $(date)
=======================================
- Geometry: 7x7 UPA (49 elements)
- Snapshots: L=8
- Frequency: 1 GHz (λ=0.3m)
- Range: 0.5-5.0m
- Created before upgrading to L=16 12x12 system

Backup contains:
$(ls -la "$BACKUP_DIR/")
EOF

echo "✅ Backup completed successfully!"
echo "📁 Backup location: $BACKUP_DIR"
echo "📋 Manifest created: $BACKUP_DIR/backup_manifest.txt"
echo ""
echo "🔄 Ready to generate L=16 12x12 data!"


