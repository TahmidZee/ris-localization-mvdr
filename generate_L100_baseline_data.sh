#!/bin/bash

# Generate L=100 Baseline Data Shards
# This creates data with L=100 snapshots for DCD-MUSIC and NF-SubspaceNet baselines

echo "🚀 Generating L=100 Baseline Data Shards"
echo "======================================="

# Create directory for L=100 baseline data
mkdir -p data_shards_L100_baseline

echo "📊 Generating L=100 data shards for baseline comparison"
echo "  - L=100 snapshots (vs our L=16)"
echo "  - Same array geometry (12×12 UPA, 16 BS antennas)"
echo "  - Same range coverage (0.5-10.0m)"
echo "  - Same SNR range (-5 to 20 dB)"
echo "  - Same FOV (φ±60°, θ±30°)"
echo ""

# Generate L=100 data shards
python -m ris_pytorch_pipeline.ris_pipeline pregen-split \
    --L 100 \
    --n-train 80000 \
    --n-val 16000 \
    --n-test 4000 \
    --snr-min -5 \
    --snr-max 20 \
    --out_dir data_shards_L100_baseline \
    --phi-fov-deg 60 \
    --theta-fov-deg 30 \
    --r-min 0.5 \
    --r-max 10.0

echo ""
echo "✅ L=100 baseline data generation completed!"
echo ""
echo "📊 Data Summary:"
echo "  - Train: 80,000 samples"
echo "  - Val: 16,000 samples" 
echo "  - Test: 4,000 samples"
echo "  - L=100 snapshots per sample"
echo "  - M_beams=64 codebook (8×8 balanced)"
echo ""
echo "🎯 Ready for baseline training:"
echo "  ./run_baselines_L100.sh"
echo ""
echo "📈 Measurement Comparison:"
echo "  - Our L=16 model: 256 measurements (16×16)"
echo "  - DCD-MUSIC L=100: 1,600 measurements (100×16)"
echo "  - NF-SubspaceNet L=100: 14,400 measurements (100×144)"
echo "  - Fair comparison: Baselines get L=100 advantage"
