# Production Command Sequence
**Date:** 2025-01-13  
**Status:** ✅ Ready for training runs (post-smoke-test gate)
**Updated:** 2026-02-11 (Training schedule tuned: GEOM_ONLY=10, lam_cov capped 0.3, sorted-only geometry, bias LR slow)

---

## Overview

This document provides the exact command sequence to achieve best results using the MVDR-based localization pipeline with optional SpectrumRefiner refinement.

**Workflow:**
1. **Stage 1 HPO** (10% data, fast exploration)
2. **Stage 2 Backbone Training** (full data, top configs)
3. **Stage 3 Refiner Training** (optional, freeze backbone, train SpectrumRefiner)
4. **Final Evaluation** (test set, compare MVDR-only vs refiner-assisted)

---

## Prerequisites

```bash
cd /home/tahit/ris/MainMusic

# Verify data shards exist
# NOTE: val/ is optional; training will reuse train shards if val/ is missing.
ls -lh data_shards_M64_N256_L64/train/*.npz | head -5
ls -lh data_shards_M64_N256_L64/test/*.npz | head -5

# Verify environment is activated (if using venv)
# source env/bin/activate  # if needed

# Gate: pipeline smoke test (must PASS before long runs)
python diagnose_pipeline_smoke.py --device cuda --steps 150
```

---

## Stage 1: HPO Exploration (10% Data)

**Goal:** Find promising hyperparameter regions  
**Time:** ~6-10 hours  
**Output:** `results_M64_N256_L64/hpo/best.json`

### Option A: Using the manual script (recommended)
```bash
cd /home/tahit/ris/MainMusic
./run_hpo_manual.sh
```

### Option B: Direct command
```bash
cd /home/tahit/ris/MainMusic
python -m ris_pytorch_pipeline.ris_pipeline hpo \
    --trials 50 \
    --hpo-epochs 20 \
    --space wide \
    --early-stop-patience 6
```

### Expected Output
- `results_M64_N256_L64/hpo/best.json` - Best single config
- `results_M64_N256_L64/hpo/hpo.db` - Full study database
- `results_M64_N256_L64/hpo/*_trials.csv` - Trial results

### After Completion
Review top 5 configs from CSV:
```bash
cat results_M64_N256_L64/hpo/*_trials.csv | sort -t, -k2 -n | head -6
```

---

## Stage 2: Backbone Training (Full Data)

**Goal:** Train top configs on full dataset  
**Time:** ~4-6 hours per run  
**Output:** `results_M64_N256_L64/checkpoints/best.pt`

### Train with Default Config (Recommended — 2026-02-11)
```bash
cd /home/tahit/ris/MainMusic
python -m ris_pytorch_pipeline.ris_pipeline train \
    --epochs 60 \
    --n_train 100000 \
    --n_val 10000 \
    --use_shards
```

**What to watch for in the first 10 epochs (geometry-only):**
- `aux_φ_rmse` should decrease from ~20° toward 15°
- If φ stalls at 20°+ for 10 epochs → model is stuck (investigate)
- `GRADPATH` probe should show non-zero gradient (confirms slot head is learning)

**Epoch 11+ (NMSE ramp conditional):**
- NMSE only ramps in if `best_aux_phi_rmse < 15°`
- If φ hasn't reached 15° → stays geometry-only indefinitely
- Capped at `lam_cov=0.3` (won't drown geometry)

### Train from HPO Config (Optional)
```bash
cd /home/tahit/ris/MainMusic
python -m ris_pytorch_pipeline.ris_pipeline train \
    --epochs 60 \
    --n_train 100000 \
    --n_val 10000 \
    --use_shards \
    --from_hpo results_M64_N256_L64/hpo/best.json
```

### Monitor Training
```bash
# Watch logs in real-time
tail -f results_M64_N256_L64/logs/full_train_*.log

# Check validation metrics
grep "VAL SURROGATE" results_M64_N256_L64/logs/full_train_*.log | tail -20
```

### Expected Output
- `results_M64_N256_L64/checkpoints/best.pt` - Best model checkpoint
- `results_M64_N256_L64/checkpoints/swa.pt` - SWA checkpoint (if enabled)
- `results_M64_N256_L64/checkpoints/run_config.json` - Training config

---

## Stage 3: SpectrumRefiner Training (Optional)

**Goal:** Train CNN refinement head on MVDR spectra (freeze backbone)  
**Time:** ~1-2 hours  
**Output:** `results_M64_N256_L64/checkpoints/best.pt` (with refiner weights)

### Train Refiner on Best Backbone
```bash
cd /home/tahit/ris/MainMusic
python -m ris_pytorch_pipeline.ris_pipeline train-refiner \
    --backbone_ckpt results_M64_N256_L64/checkpoints/best.pt \
    --epochs 10 \
    --n_train 160000 \
    --n_val 40000 \
    --use_shards \
    --lam_heatmap 0.1 \
    --grid_phi 61 \
    --grid_theta 41
```

### Configuration Options
- `--lam_heatmap`: Heatmap loss weight (default: 0.1, try 0.05-0.2)
- `--grid_phi`: Azimuth grid size (default: 61, should match MVDR grid)
- `--grid_theta`: Elevation grid size (default: 41, should match MVDR grid)
- `--out_ckpt_dir`: Override checkpoint directory (optional)

### Expected Output
- Checkpoint saved as `{"backbone": ..., "refiner": ...}` format
- Refiner weights attached to model for inference

---

## Stage 4: Final Evaluation

**Goal:** Evaluate on test set, compare MVDR-only vs refiner-assisted  
**Time:** ~10-30 minutes

### Evaluate MVDR-Only (Baseline)
```bash
cd /home/tahit/ris/MainMusic
# Run a fast, hybrid-only pass (no baselines) on a capped subset:
python -m ris_pytorch_pipeline.ris_pipeline suite --bench B1 --limit 1000 --no-baselines

# Save a copy (suite overwrites the same CSV names on reruns)
cp results_M64_N256_L64/benches/B1_all_blind.csv results_M64_N256_L64/benches/B1_all_blind_mvdr.csv
```

### Evaluate Refiner-Assisted (If Trained)
First, enable refiner in config or via environment:
```bash
cd /home/tahit/ris/MainMusic
# Edit configs.py or set environment variable
export USE_SPECTRUM_REFINER_IN_INFER=True
export REFINER_PEAK_THRESH=0.5
export REFINER_NMS_MIN_SEP=3.0

python -m ris_pytorch_pipeline.ris_pipeline suite \
    --bench B1 --limit 1000 --no-baselines

cp results_M64_N256_L64/benches/B1_all_blind.csv results_M64_N256_L64/benches/B1_all_blind_refiner.csv
```

### Compare Results
```bash
# View benchmark results
ls -lh results_M64_N256_L64/benches/*.csv

# Compare metrics
python -c "
import pandas as pd
df1 = pd.read_csv('results_M64_N256_L64/benches/B1_all_blind_mvdr.csv')
df2 = pd.read_csv('results_M64_N256_L64/benches/B1_all_blind_refiner.csv')
print('MVDR-only (Hybrid):')
print(df1.groupby('who')[['phi','theta','rng','rmspe','f1']].mean())
print('\\nRefiner-assisted (Hybrid):')
print(df2.groupby('who')[['phi','theta','rng','rmspe','f1']].mean())
"
```

---

## Quick Reference: Key Configurations

### MVDR Inference Parameters
Located in `ris_pytorch_pipeline/configs.py`:
- `MVDR_GRID_PHI = 181` - Azimuth grid resolution
- `MVDR_GRID_THETA = 361` - Elevation grid resolution
- `MVDR_THRESH_DB = -10.0` - Peak detection threshold (dB)
- `MVDR_DELTA_SCALE = 1e-2` - Regularization for MVDR
- `HYBRID_COV_BETA = 0.5` - Blending weight (R_pred vs R_samp)

### Refiner Parameters
- `USE_SPECTRUM_REFINER_IN_INFER = False` - Enable refiner in inference
- `REFINER_PEAK_THRESH = 0.5` - Peak threshold (0-1)
- `REFINER_NMS_MIN_SEP = 3.0` - NMS minimum separation (degrees)
- `REFINER_GRID_PHI = 61` - Refiner input grid (azimuth)
- `REFINER_GRID_THETA = 41` - Refiner input grid (elevation)

### Training Parameters
- `EPOCHS = 60` - Full training epochs
- `EARLY_STOP_PATIENCE = 20` - Patience increased for DETR-style models + NMSE ramp
- `BATCH_SIZE = 64` - Batch size (adjust based on GPU memory)
- `GEOM_ONLY_EPOCHS = 10` - First 10 epochs: geometry-only (no NMSE)
- `LAM_AUX_SORTED = 2.0` - Sorted canonical is the sole geometry loss
- `PHASE_LOSS["joint"]["lam_cov"] = 0.3` - NMSE capped low (gentle regularizer)
- `NMSE_RAMP_AUX_PHI_THRESHOLD = 15.0` - NMSE ramp delayed until φ < 15°
- `BIAS_LR_FINAL_MULTIPLIER = 0.1` - Bias stays slow the entire run
- `EMA_EVAL_WARMUP_EPOCHS = 999` - EMA disabled for debugging (re-enable later)

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size in configs.py or via CLI
python -m ris_pytorch_pipeline.ris_pipeline train \
    --epochs 50 \
    --n_train 160000 \
    --use_shards
# Then edit configs.py: BATCH_SIZE = 32
```

### Slow Training
- Verify GPU is being used: `nvidia-smi`
- Check data loading: ensure shards are on fast storage (SSD)
- Reduce `n_train` for quick tests: `--n_train 10000`

### Checkpoint Loading Issues
```bash
# Verify checkpoint exists
ls -lh results_M64_N256_L64/checkpoints/best.pt

# Test loading
python -c "
from ris_pytorch_pipeline.infer import load_model
model = load_model('results_M64_N256_L64/checkpoints', 'best.pt')
print('✅ Checkpoint loaded successfully')
"
```

---

## Expected Timeline

| Stage | Time | Description |
|-------|------|-------------|
| Stage 1 HPO | 6-10 hours | Hyperparameter exploration |
| Stage 2 Backbone | 4-6 hours | Full backbone training |
| Stage 3 Refiner | 1-2 hours | Refiner training (optional) |
| Stage 4 Evaluation | 10-30 min | Test set evaluation |
| **Total** | **~12-19 hours** | Complete pipeline |

---

## Next Steps After Training

1. **Analyze Results**: Review benchmark CSVs and logs
2. **Tune Thresholds**: Adjust `MVDR_THRESH_DB`, `REFINER_PEAK_THRESH` based on validation
3. **Deploy**: Use `infer.py` for inference on new data
4. **Paper**: Document best configs and results

---

## Notes

- All paths assume working directory: `/home/tahit/ris/MainMusic`
- Data shards should be in: `data_shards_M64_N256_L64/{train,val,test}/`
- Results will be saved to: `results_M64_N256_L64/`
- For production, consider using `swa.pt` checkpoint (if SWA enabled) instead of `best.pt`
