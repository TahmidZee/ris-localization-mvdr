# RIS MUSIC Pipeline

Deep learning + MUSIC-based localization for RIS (Reconfigurable Intelligent Surface) systems.

## Architecture Overview

This pipeline separates concerns cleanly:

### Training & HPO (MUSIC-free)
- Uses **surrogate metrics** (K accuracy, aux angle/range RMSE)
- Fast, stable, no gating/penalty artifacts
- Config: `VAL_PRIMARY="surrogate"`, `USE_MUSIC_METRICS_IN_VAL=False`

### Final Evaluation (Full MUSIC)
- Uses **2.5D GPU MUSIC** with hybrid covariance
- Hungarian matching, MDL baseline comparison
- Run separately via `eval_music_final.py`

## Quick Start

### 1. Generate Data Shards
```bash
python regenerate_shards_python.py
```

### 2. Run HPO (MUSIC-free)
```bash
python run_hpo_manual.py --n_trials 50 --epochs 20
```

### 3. Evaluate Best Checkpoint (Full MUSIC)
```bash
python eval_music_final.py --checkpoint results_final_L16_12x12/checkpoints/best.pt
```

## Key Files

| File | Purpose |
|------|---------|
| `run_hpo_manual.py` | HPO with surrogate metrics (fast) |
| `eval_music_final.py` | Full MUSIC evaluation (final) |
| `train_manual.py` | Manual training script |
| `verify_shards.py` | Verify data shard integrity |
| `test_gpu_music.py` | Test GPU MUSIC implementation |

## Configuration

Key config flags in `ris_pytorch_pipeline/configs.py`:

```python
# Validation mode
VAL_PRIMARY = "surrogate"  # "surrogate" (fast), "k_loc" (MUSIC), "loss"
USE_MUSIC_METRICS_IN_VAL = False  # False for training/HPO, True for final eval

# Surrogate metric weights
SURROGATE_METRIC_WEIGHTS = {
    "w_k_acc": 1.0,      # K accuracy weight
    "w_aux_ang": 0.01,   # Aux angle RMSE penalty
    "w_aux_r": 0.01,     # Aux range RMSE penalty
}
```

## Results Directory Structure

```
results_final_L16_12x12/
├── checkpoints/
│   ├── best.pt          # Best model checkpoint
│   └── last.pt          # Latest checkpoint
├── hpo/
│   ├── hpo.db           # Optuna study database
│   ├── best.json        # Best trial hyperparameters
│   └── hpo_*.log        # HPO logs
└── logs/
    └── hpo_stage1_*.log # Training logs
```

## Branch: feature/music-free-hpo

This branch implements clean separation of:
- **Training/HPO**: Surrogate metrics only (no MUSIC)
- **Final Evaluation**: Full MUSIC pipeline

Key changes:
1. Added `_validate_surrogate_epoch()` - MUSIC-free validation
2. Config flags `VAL_PRIMARY`, `USE_MUSIC_METRICS_IN_VAL`
3. HPO uses surrogate score (K_acc - penalty)
4. `eval_music_final.py` for offline MUSIC evaluation
