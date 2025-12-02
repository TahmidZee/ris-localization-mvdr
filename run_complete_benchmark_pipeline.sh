#!/bin/bash

# Complete Benchmark Pipeline
# Generates data, trains models, and benchmarks all three methods

echo "🎯 Complete Benchmark Pipeline for Fair Baseline Comparison"
echo "============================================================"

# Step 1: Generate common test scenarios
echo ""
echo "📊 Step 1: Generating Common Test Scenarios"
echo "============================================"
python generate_common_scenarios.py \
    --n_train 80000 \
    --n_val 16000 \
    --n_test 4000 \
    --output_dir benchmark_scenarios \
    --seed 42

# Step 2: Generate method-specific data
echo ""
echo "📊 Step 2: Generating Method-Specific Measurements"
echo "================================================="

echo ""
echo "2a. Generating Our L=16 RIS-based measurements..."
python generate_method_data.py \
    --method ours \
    --scenarios benchmark_scenarios \
    --output data_ours_benchmark \
    --split all

echo ""
echo "2b. Generating DCD-MUSIC ULA measurements (N=15, L=100)..."
python generate_method_data.py \
    --method dcd \
    --scenarios benchmark_scenarios \
    --output data_dcd_benchmark \
    --split all

echo ""
echo "2c. Generating NF-SubspaceNet UPA measurements (N=144, L=100)..."
python generate_method_data.py \
    --method nfssn \
    --scenarios benchmark_scenarios \
    --output data_nfssn_benchmark \
    --split all

# Step 3: Train models
echo ""
echo "📊 Step 3: Training Models"
echo "========================="

echo ""
echo "3a. Training Our L=16 model..."
echo "   (Using HPO results, or train from scratch if no HPO)"
# Check if HPO results exist
if [ -f "results_final/hpo/best.json" ]; then
    echo "   Using HPO results for L=16 model..."
    python -m ris_pytorch_pipeline.train \
        --from_hpo results_final/hpo/best.json \
        --data_dir data_ours_benchmark \
        --epochs 100 \
        --use_shards
else
    echo "   Training L=16 model from scratch (no HPO results)..."
    python -m ris_pytorch_pipeline.train \
        --data_dir data_ours_benchmark \
        --epochs 100 \
        --use_shards
fi

echo ""
echo "3b. Training DCD-MUSIC baseline (L=100)..."
python train_baselines_L100_optimal.py \
    --model dcd \
    --data_dir data_dcd_benchmark \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --results_dir results_final/baselines/dcd_music

echo ""
echo "3c. Training NF-SubspaceNet baseline (L=100)..."
python train_baselines_L100_optimal.py \
    --model nfssn \
    --data_dir data_nfssn_benchmark \
    --epochs 100 \
    --batch_size 16 \
    --lr 5e-4 \
    --results_dir results_final/baselines/nf_subspacenet

# Step 4: Run unified benchmark
echo ""
echo "📊 Step 4: Running Unified Benchmark"
echo "===================================="
python run_unified_benchmark.py \
    --scenarios benchmark_scenarios/test_scenarios.pkl \
    --model_ours results_final/checkpoints/best.pt \
    --model_dcd results_final/baselines/dcd_music/best.pt \
    --model_nfssn results_final/baselines/nf_subspacenet/best.pt \
    --data_ours data_ours_benchmark/test \
    --data_dcd data_dcd_benchmark/test \
    --data_nfssn data_nfssn_benchmark/test \
    --output benchmark_results.csv

# Step 5: Generate paper figures
echo ""
echo "📊 Step 5: Generating Paper-Ready Figures"
echo "========================================"
python plot_benchmark_results.py \
    --results benchmark_results.csv \
    --output figures/

echo ""
echo "✅ Complete Benchmark Pipeline Finished!"
echo "========================================"
echo ""
echo "📊 Results Summary:"
echo "  - Benchmark results: benchmark_results.csv"
echo "  - Figures: figures/"
echo ""
echo "📈 Next Steps:"
echo "  1. Review benchmark_results.csv"
echo "  2. Check figures/ for paper-ready plots"
echo "  3. Use results for paper writing"
echo ""
echo "🎯 Key Comparison:"
echo "  - Our L=16: 256 measurements"
echo "  - DCD-MUSIC: 1,500 measurements (5.9× more)"
echo "  - NF-SubspaceNet: 14,400 measurements (56× more)"
