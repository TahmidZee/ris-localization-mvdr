#!/bin/bash

# Complete Pipeline: HPO + Baseline Training
# This script runs the complete pipeline in parallel

echo "🚀 Complete Pipeline: HPO + Baseline Training"
echo "============================================="

# Check current status
echo "📊 Current Status Check:"
echo "  - HPO running: $(screen -list | grep -q hpo && echo '✅ Yes' || echo '❌ No')"
echo "  - L=16 data: $(test -d data_shards_M64_L16 && echo '✅ Yes' || echo '❌ No')"
echo "  - L=100 data: $(test -d data_shards_L100_baseline && echo '✅ Yes' || echo '❌ No')"
echo ""

# Step 1: Generate L=100 baseline data if needed
if [ ! -d "data_shards_L100_baseline" ]; then
    echo "📊 Step 1: Generating L=100 baseline data..."
    echo "============================================="
    ./generate_L100_baseline_data.sh
    echo ""
else
    echo "✅ Step 1: L=100 baseline data already exists"
    echo ""
fi

# Step 2: Start baseline training
echo "📊 Step 2: Starting L=100 baseline training..."
echo "============================================="
./run_baselines_L100.sh
echo ""

# Step 3: Show current status
echo "📊 Step 3: Pipeline Status"
echo "========================="
echo "🎯 Currently Running:"
echo "  - HPO (L=16 model): screen -r hpo"
echo "  - DCD-MUSIC L=100: screen -r dcd_music_L100"
echo "  - NF-SubspaceNet L=100: screen -r nf_subspacenet_L100"
echo ""
echo "📊 Monitor all processes:"
echo "  screen -list"
echo ""
echo "⏱️  Expected Timeline:"
echo "  - HPO: 2-4 hours (60 trials × 15 epochs)"
echo "  - DCD-MUSIC: 2-3 hours (100 epochs)"
echo "  - NF-SubspaceNet: 3-4 hours (100 epochs)"
echo ""
echo "🎯 Next Steps (after completion):"
echo "  1. Train best L=16 model from HPO results"
echo "  2. Run benchmark comparison"
echo "  3. Generate paper results"
echo ""
echo "📈 Final Comparison:"
echo "  - Our L=16 model: 256 measurements (16×16)"
echo "  - DCD-MUSIC L=100: 1,600 measurements (100×16)"
echo "  - NF-SubspaceNet L=100: 14,400 measurements (100×144)"
echo "  - Target: Sub-1° θ accuracy with 56× measurement reduction"
