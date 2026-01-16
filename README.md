# RIS Localization with MVDR (K-free)

Deep learning + **MVDR (Minimum Variance Distortionless Response)** K-free localization for RIS (Reconfigurable Intelligent Surface) systems.

## Overview

This pipeline implements a hybrid approach combining:
- **Neural network backbone** for covariance prediction from RIS measurements
- **MVDR spectral estimation** for K-free multi-source localization
- **Optional CNN refinement head** (`SpectrumRefiner`) for peak sharpening

**Key Features:**
- ✅ **K-free**: No explicit K estimation needed (uses MVDR peak detection)
- ✅ **Hybrid covariance**: Blends NN-predicted and sample covariance for robustness
- ✅ **Two-stage training**: Backbone → optional SpectrumRefiner refinement
- ✅ **Production-ready**: Complete HPO → training → evaluation pipeline

## Architecture

```
RIS Measurements (y, H, codes)
    ↓
Neural Network Backbone
    ↓
Covariance Factors (A_angle, A_range)
    ↓
Hybrid Covariance (R_blend = β·R_pred + (1-β)·R_samp)
    ↓
MVDR Spectrum Computation
    ↓
[Optional] SpectrumRefiner CNN
    ↓
Peak Detection (2D NMS)
    ↓
Source Locations (φ, θ, r)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Data Shards

```bash
python -m ris_pytorch_pipeline.ris_pipeline pregen-split \
    --n-train 160000 \
    --n-val 40000 \
    --n-test 8000 \
    --shard 25000 \
    --out_dir data_shards_M64_L16
```

### 3. Run HPO (Stage 1)

```bash
python -m ris_pytorch_pipeline.ris_pipeline hpo \
    --trials 50 \
    --hpo-epochs 20 \
    --space wide \
    --early-stop-patience 6
```

### 4. Train Backbone (Stage 2)

```bash
python -m ris_pytorch_pipeline.ris_pipeline train \
    --epochs 50 \
    --n_train 160000 \
    --n_val 40000 \
    --use_shards \
    --from_hpo results_final_L16_12x12/hpo/best.json
```

### 5. Train Refiner (Optional, Stage 3)

```bash
python -m ris_pytorch_pipeline.ris_pipeline train-refiner \
    --backbone_ckpt results_final_L16_12x12/checkpoints/best.pt \
    --epochs 10 \
    --n_train 160000 \
    --n_val 40000 \
    --use_shards \
    --lam_heatmap 0.1
```

### 6. Evaluate

```bash
python -m ris_pytorch_pipeline.ris_pipeline suite --tag final_eval
```

## Documentation

- **[PRODUCTION_COMMAND_SEQUENCE.md](PRODUCTION_COMMAND_SEQUENCE.md)**: Complete production workflow with exact commands
- **[TWO_STAGE_HPO_GUIDE.md](TWO_STAGE_HPO_GUIDE.md)**: Two-stage HPO strategy guide
- **[IMPLEMENTATION_CHANGELOG.md](IMPLEMENTATION_CHANGELOG.md)**: Detailed changelog of all changes

## Project Structure

```
ris_pytorch_pipeline/
├── model.py              # HybridModel + SpectrumRefiner
├── loss.py               # UltimateHybridLoss (covariance, angle, range, heatmap)
├── train.py              # Trainer class (backbone + refiner stages)
├── infer.py              # Inference pipeline (MVDR + optional refiner)
├── eval_angles.py        # Evaluation metrics (MVDR localization)
├── music_gpu.py          # MVDR/MUSIC GPU implementations
├── configs.py            # System and model configurations
├── hpo.py                # Hyperparameter optimization
├── dataset.py            # Data loading and shard management
└── ris_pipeline.py       # CLI entry point
```

## Key Configuration

Main configuration in `ris_pytorch_pipeline/configs.py`:

```python
# System geometry
M = 16              # BS antennas
N_H, N_V = 12, 12   # RIS elements (12×12 UPA)
L = 16              # Temporal snapshots

# MVDR inference
MVDR_GRID_PHI = 181
MVDR_GRID_THETA = 361
MVDR_THRESH_DB = -10.0
HYBRID_COV_BETA = 0.5

# SpectrumRefiner (optional)
USE_SPECTRUM_REFINER_IN_INFER = False
REFINER_PEAK_THRESH = 0.5
REFINER_NMS_MIN_SEP = 3.0
```

## Results Directory

```
results_final_L16_12x12/
├── checkpoints/
│   ├── best.pt          # Best model checkpoint
│   └── swa.pt           # SWA checkpoint (if enabled)
├── hpo/
│   ├── best.json        # Best HPO config
│   └── hpo.db           # Optuna study database
└── logs/
    └── train_*.log      # Training logs
```

## Citation

If you use this code, please cite:

```bibtex
@article{ris_mvdr_localization,
  title={K-free RIS Localization using MVDR Spectral Estimation},
  author={...},
  year={2025}
}
```

## License

[Add your license here]

## Contact

[Add your contact information here]
