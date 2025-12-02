# HPO Execution Guide
**Date:** 2025-11-26  
**Status:** ✅ READY TO RUN

---

## Quick Start

### Option 1: Bash Script (Recommended)
```bash
cd /home/tahit/ris/MainMusic
./run_hpo_manual.sh
```

### Option 2: Python Script
```bash
cd /home/tahit/ris/MainMusic
python run_hpo_manual.py
```

### Option 3: Direct Command
```bash
cd /home/tahit/ris/MainMusic
python -m ris_pytorch_pipeline.ris_pipeline hpo \
    --trials 40 \
    --hpo-epochs 12 \
    --space wide \
    --early-stop-patience 15
```

---

## What HPO Does

### High-Level Flow:
1. **Optuna suggests hyperparameters** (architecture, learning rate, loss weights)
2. **Trainer runs a short training** (12 epochs, 10K train / 1K val samples)
3. **Returns composite K/localization score** (NOT raw loss)
4. **Optuna minimizes the score:** `(1 - K_acc) + RMSE_norm - success_rate`
5. **Repeats 40 times**, keeping best hyperparameters

### What Gets Optimized:
- **Architecture:** D_MODEL (448/512), NUM_HEADS (6/8), dropout (0.20-0.32)
- **Learning rate:** 1e-4 to 4e-4 (log-uniform)
- **Loss weights:** lam_cov, lam_ang, lam_rng, **lam_K (0.05-0.25)**
- **Inference:** range_grid, newton_iter, newton_lr, shrink_alpha
- **Batch size:** 64 or 80

### What's Being Minimized:
**Composite penalty** (lower is better):
```python
penalty = (1.0 - K_acc) + (AoA_RMSE / 5°) + (Elevation_RMSE / 5°) + (Range_RMSE / 1m) - Success_rate
```

**Why this matters:**
- High K_acc → low penalty ✓
- Low AoA/Range RMSE → low penalty ✓
- High success rate → low penalty ✓
- **NOT optimizing raw NMSE loss** (that's just used for gradients)

---

## Configuration

### Default Settings (in scripts):
```bash
N_TRIALS=40          # 40-80 recommended for good coverage
EPOCHS_PER_TRIAL=12  # Short with early stopping (pruning saves time)
SPACE="wide"         # Full search space
EARLY_STOP=15        # Stop if no improvement for 15 epochs
```

### Dataset Subset (10% for HPO speed):
- **Train:** 10,000 samples (10% of 100K)
- **Val:** 1,000 samples (10% of 10K)
- **Why 10%?** Balances exploration speed vs signal quality

### Search Space ("wide"):
```python
D_MODEL:   [448, 512]
NUM_HEADS: [6, 8]
dropout:   [0.20, 0.32]
lr:        [1e-4, 4e-4] (log-uniform)
batch_size: [64, 80]
lam_cov:   [0.10, 0.25]
lam_ang:   [0.50, 1.00]
lam_rng:   [0.30, 0.60]
lam_K:     [0.05, 0.25]  # UPDATED: Was [0.05, 0.15], now allows stronger K supervision
```

---

## Time & Resources

### Expected Runtime:
- **Per trial:** ~8-15 minutes (12 epochs, 10K samples, early stopping)
- **40 trials:** ~6-10 hours (with pruning, some trials stop early)
- **80 trials:** ~12-20 hours

### Resource Requirements:
- **GPU:** 1× RTX 3090 / A100 (16GB+ VRAM)
- **RAM:** 32 GB+ (for GPU cache)
- **Disk:** ~2 GB (HPO database + logs)

### Pruning:
- **MedianPruner** stops bad trials early (after 6 epochs)
- Saves ~30-40% of total time
- Need ≥8 completed trials before pruning activates

---

## What to Watch During HPO

### Good Trial Example:
```
[HPO Trial 15] Training completed! Best objective: 0.234
  Hyperparameters: D_MODEL=512, BS=80, LR=0.000245
  Metrics: K_acc=0.89, AoA_RMSE=1.2°, Success_rate=0.81
```

### Bad Trial (Will Be Pruned):
```
[HPO Trial 8] Pruned at epoch 7
  Reason: Median objective (0.89) > median of completed trials (0.45)
```

### Warning Signs:
- **All trials fail:** Check shard paths, GPU memory
- **No pruning after 15 trials:** Check pruner settings
- **Objective values all ~1.0:** K-head not learning (check lam_K range)
- **Objective values all ~0.0:** Suspiciously low (check metric computation)

---

## After HPO Completes

### 1. Review Best Hyperparameters
```bash
cat results_final/hpo/best.json
```

**Example output:**
```json
{
  "value": 0.234,
  "number": 27,
  "params": {
    "D_MODEL": 512,
    "NUM_HEADS": 8,
    "dropout": 0.25,
    "lr": 0.000245,
    "lam_cov": 0.18,
    "lam_ang": 0.85,
    "lam_rng": 0.45,
    "lam_K": 0.18,
    ...
  }
}
```

### 2. Run Full Training (with best hyperparameters)
```bash
# These are automatically loaded from best.json
python train_ris.py --epochs 50 --batch-size 64
```

**Or via pipeline:**
```bash
python -m ris_pytorch_pipeline.ris_pipeline train \
    --epochs 50 \
    --use_shards \
    --from_hpo results_final/hpo/best.json
```

### 3. Tune K_CONF_THRESH and β on Validation
After training, sweep confidence threshold and hybrid beta:
```python
# In a script or notebook:
from ris_pytorch_pipeline.infer import load_model, hybrid_estimate_final

for thresh in [0.5, 0.6, 0.65, 0.7, 0.8]:
    cfg.K_CONF_THRESH = thresh
    # Evaluate on validation, record K_acc, K_under, Success_rate
    # Pick threshold that minimizes K_under while keeping K_acc high
```

---

## Troubleshooting

### "No training shards found"
**Cause:** Shards not generated yet  
**Fix:** Run `./regenerate_shards.sh` first (wait ~60 min)

### "R_samp not found in shards"
**Cause:** Old shards or generation failed  
**Fix:** Regenerate shards or set `cfg.HYBRID_COV_BETA = 0.0` temporarily

### "CUDA out of memory"
**Cause:** GPU cache too large or batch size too big  
**Fix:** 
- Reduce `batch_size` in HPO range to `[32, 64]`
- Or disable GPU cache (slower): set `gpu_cache=False` in hpo.py

### "Study already exists with different direction"
**Cause:** Old HPO database with `direction="maximize"`  
**Fix:**
```bash
# Delete old database
rm results_final/hpo/hpo.db*
# Or use a new study name in hpo.py
```

### "K_acc stuck at 0.2 across all trials"
**Cause:** K-head not learning  
**Fix:**
- Check `lam_K` range (should be 0.05-0.25, not 0.0)
- Verify `cfg.VAL_PRIMARY = "k_loc"` is set in hpo.py
- Check cost-sensitive K loss is enabled in loss.py

### "All trials get pruned"
**Cause:** Objective too high in early epochs  
**Fix:**
- Increase `n_warmup_steps` in MedianPruner (from 6 to 8-10)
- Or reduce `early_stop_patience` (from 15 to 10)

---

## Understanding HPO vs Training

### HPO (Hyperparameter Optimization):
- **Purpose:** Find best architecture and loss weights
- **How:** Run many SHORT trainings (12 epochs each)
- **Dataset:** Small subset (10K train / 1K val)
- **Output:** `best.json` with optimal hyperparameters
- **Time:** 6-10 hours for 40 trials

### Full Training (After HPO):
- **Purpose:** Train production model with best hyperparameters
- **How:** Run ONE LONG training (50+ epochs)
- **Dataset:** Full dataset (100K train / 10K val)
- **Input:** Loads hyperparameters from `best.json`
- **Output:** Final model checkpoint for deployment
- **Time:** 4-8 hours (depends on GPU)

**Workflow:**
```
1. Regenerate shards (✓ DONE - 35 min)
2. Run HPO (6-10 hours) ← YOU ARE HERE
3. Run full training with best.json (4-8 hours)
4. Tune K_CONF_THRESH and β on validation (1 hour)
5. Evaluate on test set (final results!)
```

---

## HPO Output Files

### After HPO completes:
```
results_final/hpo/
├── hpo.db              # Optuna study database
├── best.json           # Best hyperparameters (YOU NEED THIS)
├── L16_M64_wide_v2_optimfix_trials.csv  # All trial results
└── hpo_20251126_230500.log  # Full HPO log
```

### best.json Structure:
```json
{
  "value": 0.234,        # Best objective (composite K/loc penalty)
  "number": 27,          # Trial number that achieved this
  "params": {            # Hyperparameters to use for full training
    "D_MODEL": 512,
    "lr": 0.000245,
    "lam_K": 0.18,
    ...
  }
}
```

---

## Resuming Interrupted HPO

If HPO is interrupted (Ctrl+C, system kill, etc.), you can resume:

```bash
# Just run the same command again
./run_hpo_manual.sh

# Optuna will:
# - Load existing study from hpo.db
# - Count completed trials
# - Run remaining trials to reach target count
```

**Example:**
```
[HPO] Completed trials: 23, Remaining trials: 17
[HPO] Starting optimization with 17 trials...
```

---

## After HPO: What to Do

### 1. Inspect Results
```bash
# View best hyperparameters
cat results_final/hpo/best.json

# View all trials (if exported)
head -20 results_final/hpo/*_trials.csv
```

### 2. Run Full Training
```bash
# Automatically loads best.json
python train_ris.py --epochs 50 --batch-size 64

# Watch for:
#   K_acc, K_under, AoA_RMSE, Range_RMSE, Success_rate
# Best checkpoint selected by composite K/loc score
```

### 3. Calibrate K-Logits
```python
# After training, calibration runs automatically
# Temperature T_opt is saved in best.pt checkpoint
# Use this for inference
```

### 4. Tune Thresholds
```python
# Sweep K_CONF_THRESH on validation
for thresh in [0.5, 0.6, 0.65, 0.7, 0.8]:
    cfg.K_CONF_THRESH = thresh
    # Evaluate: K_acc, K_under, Success_rate
    # Pick threshold that minimizes K_under

# Sweep HYBRID_COV_BETA on validation
for beta in [0.0, 0.2, 0.3, 0.5]:
    cfg.HYBRID_COV_BETA = beta
    # Evaluate: stratify by SNR if possible
    # Pick beta that helps low-SNR without hurting high-SNR
```

### 5. Final Evaluation
```bash
# Evaluate on test set with tuned thresholds
python -m ris_pytorch_pipeline.benchmark_test

# Report:
#   - K_acc, K_under, K_over
#   - AoA_RMSE, Elevation_RMSE, Range_RMSE
#   - Success_rate
#   - Comparison: K-head vs MDL vs confidence-gated fusion
```

---

## FAQ

**Q: Do I run HPO before or after full training?**  
A: **HPO first** (finds best hyperparameters), then full training (uses those hyperparameters).

**Q: How long does HPO take?**  
A: ~6-10 hours for 40 trials (with pruning). Can run 80 trials for ~12-20 hours.

**Q: Can I stop and resume HPO?**  
A: Yes! Optuna saves progress to `hpo.db`. Just run the same command again.

**Q: What if all trials have similar objective values?**  
A: The search space may be too narrow, or the metric is insensitive. Check trial CSV to see variance.

**Q: Should I increase lam_K range to [0.10, 0.40]?**  
A: Not for first HPO run. Start with [0.05, 0.25]. If K_under remains high (>10%) after full training, run a second HPO with higher lam_K range.

**Q: What's a "good" objective value?**  
A: Lower is better. Rough guide:
- **Excellent:** < 0.30 (K_acc > 0.85, RMSE < 1°, Success > 0.75)
- **Good:** 0.30-0.50 (K_acc > 0.75, RMSE < 2°, Success > 0.60)
- **Poor:** > 0.70 (K_acc < 0.60, RMSE > 3°, Success < 0.40)

**Q: Can I change the objective formula?**  
A: Yes, edit `train.py` line 1926. Current formula:
```python
val_score = (1.0 - k_acc) + (rmse_phi / 5.0) + (rmse_theta / 5.0) + (rmse_r / 1.0) - success_rate
```
You can adjust the normalization constants (5°, 1m) or add weights.

---

## Verification Before Running

### 1. Check Shards Are Ready:
```bash
ls -lh data_shards_M64_L16/train/*.npz
# Should show 4 shards, each ~8.7 GB

python -c "import numpy as np; z=np.load('data_shards_M64_L16/train/shard_000.npz'); print('R_samp:', 'R_samp' in z); z.close()"
# Should print: R_samp: True
```

**✅ YOUR STATUS:** Shards are ready! (Generated Nov 26, 22:30-23:05)

### 2. Check Covariance Alignment:
```bash
python audit_covariance_paths.py
# Should report tr(R_eff) = 144.0000 for both paths
# Off-diagonal difference due to shrink is EXPECTED
```

**✅ YOUR STATUS:** Alignment verified! (tr = 144, shrink mismatch is intentional)

### 3. Verify Imports:
```bash
python -c "from ris_pytorch_pipeline.hpo import run_hpo; print('✓ Imports OK')"
```

**✅ YOUR STATUS:** All imports working!

---

## Expected HPO Output

### Console Output (First Trial):
```
================================================================================
HYPERPARAMETER OPTIMIZATION (HPO) - Manual Execution
================================================================================

Configuration:
  Trials: 40
  Epochs per trial: 12
  Search space: wide

[HPO Trial 0] Starting trial...
[HPO Trial 0] Hyperparameters suggested: D_MODEL=512, BS=64, LR=0.000234
[HPO Trial 0] Creating Trainer (VAL_PRIMARY=k_loc)...
[Beta Warmup] Annealing β from 0.00 → 0.30 over 2 epochs
[Training] Starting 12 epochs...

Epoch 001/012 [no-curriculum] train 0.425612  val 0.389234
  🧭 Metrics: K_acc=0.672, K_mdl_acc=0.548, succ_rate=0.512, 
     φ_med=2.34°, θ_med=1.89°, r_med=0.67m,
     φ_RMSE≈3.12°, θ_RMSE≈2.45°, r_RMSE≈0.89m, score=0.678

Epoch 002/012 [no-curriculum] train 0.312456  val 0.298765
  🧭 Metrics: K_acc=0.734, K_mdl_acc=0.612, succ_rate=0.623, 
     ...

...

Epoch 012/012 [no-curriculum] train 0.187234  val 0.192456
  🧭 Metrics: K_acc=0.856, K_mdl_acc=0.734, succ_rate=0.789, 
     φ_med=0.89°, θ_med=0.67°, r_med=0.34m,
     φ_RMSE≈1.23°, θ_RMSE≈0.98°, r_RMSE≈0.45m, score=0.312

🌡️ Running post-train K calibration on validation set...
🎯 Optimal K temperature: 1.234
📈 After calibration: confidence=0.712, accuracy=0.867

[HPO Trial 0] Training completed! Best objective: 0.312

[HPO Trial 1] Starting trial...
...
```

### Final Summary:
```
[HPO] best value=0.234 trial=27
[HPO] Saved best trial → results_final/hpo/best.json

Best trial:
{
  "value": 0.234,
  "number": 27,
  "params": {
    "D_MODEL": 512,
    "NUM_HEADS": 8,
    "dropout": 0.26,
    "lr": 0.000245,
    "lam_K": 0.18,
    ...
  }
}
```

---

## Next Steps After HPO

1. **✅ Review best.json** - Check if hyperparameters make sense
2. **🚀 Full training** - 50-100 epochs with full 100K dataset
3. **🎯 Tune thresholds** - K_CONF_THRESH and HYBRID_COV_BETA
4. **📊 Test evaluation** - Final metrics for paper

---

## Your Current Status

### ✅ Completed:
- [x] Shard regeneration with R_samp (100K/10K/10K)
- [x] Covariance path audit (alignment verified)
- [x] Code fixes (metric-driven selection, canonical builder)
- [x] HPO scripts ready

### 📋 Next:
- [ ] Run HPO (6-10 hours) ← **YOU ARE HERE**
- [ ] Full training with best.json (4-8 hours)
- [ ] Tune thresholds on validation (1 hour)
- [ ] Test evaluation (final results)

---

## Quick Reference Commands

```bash
# Run HPO (choose one):
./run_hpo_manual.sh              # Bash
python run_hpo_manual.py         # Python
python -m ris_pytorch_pipeline.ris_pipeline hpo --trials 40  # Direct

# Monitor progress:
tail -f results_final/logs/hpo_*.log

# Check completed trials:
python -c "import optuna; s=optuna.load_study(study_name='L16_M64_wide_v2_optimfix', storage='sqlite:///results_final/hpo/hpo.db'); print(f'Completed: {len([t for t in s.trials if t.state.is_finished()])}')"

# After HPO, run full training:
python train_ris.py --epochs 50 --batch-size 64
```

---

**Ready to start HPO?** Run `./run_hpo_manual.sh` now!

**Estimated completion:** 6-10 hours (overnight run recommended)



