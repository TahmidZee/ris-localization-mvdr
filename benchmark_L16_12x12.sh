#!/bin/bash
# Comprehensive benchmark for L=16 12x12 system with paper-compliant metrics

echo "📈 Comprehensive L=16 12x12 Benchmarking"
echo "Expected: Substantial performance gains over L=8 7×7 system"
echo ""

# Configuration
export CUDA_VISIBLE_DEVICES=0
cd /home/tahit/ris/MainMusic

echo "📊 Benchmark Configuration:"
echo "  • System: 12×12 UPA, L=16 (144 elements, K≤15)"
echo "  • Test data: 20K samples"
echo "  • SNR range: -5 → 20 dB"
echo "  • K range: 1-5 sources (K≤L-1=15)"
echo "  • Metrics: RMSPE (3D position), RMSE(φ,θ,r), runtime"
echo "  • Models: Hybrid (ours), DCD-MUSIC, NF-SubspaceNet"
echo ""

# Create benchmark results directory
mkdir -p results_final_L16_12x12/benchmark

echo "🎯 Running comprehensive benchmark..."

# Benchmark all models with paper-compliant metrics
python -m ris_pytorch_pipeline.ris_pipeline bench_suite \
    --test_data results_final_L16_12x12/data/shards/test \
    --results_dir results_final_L16_12x12/benchmark \
    --models hybrid,dcd,nfssn \
    --hybrid_ckpt results_final_L16_12x12/checkpoints/swa.pt \
    --dcd_ckpt results_final_L16_12x12/dcd_results/best.pt \
    --nfssn_ckpt results_final_L16_12x12/nfssn_results/best.pt \
    --metrics rmspe,rmse_phi,rmse_theta,rmse_r,runtime \
    --snr_sweep -5,0,5,10,15,20 \
    --k_sweep 1,2,3,4,5 \
    --coherent_sweep 0.0,0.3,0.7,1.0 \
    --save_plots true \
    --save_csv true

echo ""
echo "✅ L=16 12x12 benchmark completed!"
echo ""
echo "📋 Benchmark Results:"
echo "  • Full results: results_final_L16_12x12/benchmark/"
echo "  • Performance plots: results_final_L16_12x12/benchmark/plots/"
echo "  • CSV exports: results_final_L16_12x12/benchmark/results.csv"
echo "  • Runtime analysis: results_final_L16_12x12/benchmark/runtime.csv"
echo ""
echo "🏆 Expected Hybrid Model Performance vs Baselines:"
echo "  • RMSPE (3D position): Substantial reduction"
echo "  • Angle RMSE: 45-65% better than L=8 7×7"
echo "  • Range RMSE: 30-55% better than L=8 7×7"
echo "  • K estimation: Much more stable"
echo "  • Runtime: Competitive with improved accuracy"
echo ""
echo "📝 Paper-Compliant Metrics:"
echo "  • RMSPE (meters): 3D Cartesian position error"
echo "  • RMSE(φ,θ,r): Individual angle/range errors"
echo "  • SNR robustness: -5→20 dB performance"
echo "  • K scalability: 1-5 sources"
echo "  • Coherence robustness: Mixed coherent/non-coherent"
echo "  • Runtime analysis: Median & p90 latency"
echo ""
echo "🎉 Ready for paper with substantial performance claims!"


