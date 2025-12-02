#!/bin/bash
# Production workflow using existing Python infrastructure with full features
# (EMA, SWA, 3-phase training, K-calibration, etc.)

cd "$(dirname "$0")"

echo "========================================="
echo "Production Workflow - L=16 M_beams=32"
echo "========================================="
echo ""
echo "This uses the existing Python infrastructure with:"
echo "  ✅ EMA (Exponential Moving Average)"
echo "  ✅ SWA (Stochastic Weight Averaging)"
echo "  ✅ 3-phase training curriculum"
echo "  ✅ K-calibration with temperature scaling"
echo "  ✅ All optimizations from the original system"
echo ""
echo "========================================="
echo ""

# Step 1: Generate data
echo "Step 1: Generate Data (with 10m range)"
echo "Command:"
echo "  python -m ris_pytorch_pipeline.ris_pipeline pregen-split \\"
echo "      --n-train 80000 --n-val 16000 --n-test 4000 \\"
echo "      --r-min 0.5 --r-max 10.0 \\"
echo "      --phi-fov-deg 60.0 --theta-fov-deg 30.0 \\"
echo "      --snr-min -5.0 --snr-max 20.0 \\"
echo "      --out_dir data_shards_M32_L16 --seed 42"
echo ""
read -p "Press Enter to run, or Ctrl+C to cancel..."
python -m ris_pytorch_pipeline.ris_pipeline pregen-split \
    --n-train 80000 --n-val 16000 --n-test 4000 \
    --r-min 0.5 --r-max 10.0 \
    --phi-fov-deg 60.0 --theta-fov-deg 30.0 \
    --snr-min -5.0 --snr-max 20.0 \
    --out_dir data_shards_M32_L16 --seed 42

echo ""
echo "========================================="
echo ""

# Step 2: Run HPO
echo "Step 2: Run HPO"
echo "Command:"
echo "  python -m ris_pytorch_pipeline.hpo \\"
echo "      --n_trials 50 --epochs_per_trial 15 \\"
echo "      --space wide --export_csv"
echo ""
read -p "Press Enter to run, or Ctrl+C to cancel..."
python -m ris_pytorch_pipeline.hpo \
    --n_trials 50 --epochs_per_trial 15 \
    --space wide --export_csv

echo ""
echo "========================================="
echo ""

# Step 3: Train full model
echo "Step 3: Train Full Model (with all features)"
echo "Command:"
echo "  python -m ris_pytorch_pipeline.train \\"
echo "      --from_hpo true --epochs 70 \\"
echo "      --batch_size 64 --grad_accumulation 2 \\"
echo "      --early_stop_patience 10 --max_grad_norm 1.0 \\"
echo "      --deterministic true --save_best true \\"
echo "      --calibrate_k true"
echo ""
read -p "Press Enter to run, or Ctrl+C to cancel..."
python -m ris_pytorch_pipeline.train \
    --from_hpo true --epochs 70 \
    --batch_size 64 --grad_accumulation 2 \
    --early_stop_patience 10 --max_grad_norm 1.0 \
    --deterministic true --save_best true \
    --calibrate_k true

echo ""
echo "========================================="
echo ""

# Step 4: Evaluate
echo "Step 4: Evaluate Model"
echo "Command:"
echo "  python -m ris_pytorch_pipeline.ris_pipeline bench --n 1000"
echo "  python -m ris_pytorch_pipeline.ris_pipeline suite"
echo ""
read -p "Press Enter to run, or Ctrl+C to cancel..."
python -m ris_pytorch_pipeline.ris_pipeline bench --n 1000
python -m ris_pytorch_pipeline.ris_pipeline suite

echo ""
echo "========================================="
echo "✅ Production workflow complete!"
echo "========================================="
echo ""
