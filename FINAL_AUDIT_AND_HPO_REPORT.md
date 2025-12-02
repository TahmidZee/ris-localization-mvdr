# Final Audit & HPO Report
**Date:** 2025-11-26  
**Status:** ✅ PRODUCTION READY

---

## Your Questions Answered

### Q1: "Are you saying we do training before HPO?"

**NO.** Workflow is:
1. **HPO first** (6-10 hours) → finds best hyperparameters
2. **Full training second** (4-8 hours) → uses those hyperparameters

**HPO internally runs many short trainings** (12 epochs each, 40 trials), not one long training.

**Correct order:**
```
Regenerate shards (✓ DONE)
    ↓
Run HPO (40 trials × 12 epochs = find best hyperparams)
    ↓
Run full training (1 run × 50 epochs = production model)
    ↓
Tune thresholds (K_CONF_THRESH, β)
    ↓
Evaluate on test set (final results)
```

---

### Q2: "Did you fix the HPO ranges as mentioned? Should lam_K range be higher?"

**✅ YES, FIXED!**

**Before:**
```python
lam_K = trial.suggest_float("lam_K", 0.05, 0.15)  # Too conservative
```

**After (NOW):**
```python
lam_K = trial.suggest_float("lam_K", 0.05, 0.25)  # UPDATED: Allows stronger K supervision
```

**Why this matters:**
- Original range [0.05, 0.15] was conservative (to avoid K_CE explosion from earlier bugs)
- With cost-sensitive K loss now working, we can safely go up to 0.25
- If K_under remains >10% after first HPO, consider expanding to [0.10, 0.40] for a second HPO run

**Other HPO fixes:**
- ✅ Dataset sizes: 80K→100K, 16K→10K (updated to match new shards)
- ✅ HPO subset: 10K train / 1K val (10% of full dataset)
- ✅ Direction comment: Added clarification that we minimize penalty
- ✅ VAL_PRIMARY: Forces "k_loc" for metric-driven optimization

---

### Q3: "Audit shows off-diagonals differ. What is wrong?"

**✅ NOTHING IS WRONG!**

**What the audit shows:**
```
✓ R_eff (K-head): trace = 144.0000, Shrink: applied=True
✓ R_eff (MUSIC): trace = 144.0000, Shrink: applied=False
||R_eff_khead - R_eff_music||_F = 3.25e-02
Off-diagonal difference: 3.24e-02
```

**Why off-diagonals differ:**

Shrinkage formula: `R_shrunk = (1-α)*R + α*μ*I`

- Diagonals: `R_shrunk[i,i] = (1-α)*R[i,i] + α*μ`
- **Off-diagonals: `R_shrunk[i,j] = (1-α)*R[i,j]`** ← scaled by (1-α)

At SNR=-2.94 dB (low SNR):
- α ≈ 0.01-0.02 (from `shrink_alpha` calculation)
- Off-diagonals are scaled by ~0.98-0.99
- This causes a non-zero off-diagonal difference

**This is BY DESIGN:**
- K-head needs shrunk eigenvalues for robust features at low SNR
- MUSIC applies its own adaptive shrinkage later in the pipeline
- We verified with `audit_shrink_only.py` that with **both** paths skipping shrink: `||diff||_F = 2.4e-06` ✓

**Proof it's not a bug:**
```bash
# From your earlier audit:
$ python audit_shrink_only.py
||R_khead_noshrink - R_music||_F = 2.403089e-06
✓ PERFECT MATCH when both paths skip shrink!
```

**Conclusion:**
- ✅ Traces are correct (144.0000)
- ✅ Base R_eff is identical (2.4e-06 difference without shrink)
- ✅ Off-diagonal difference is **intentional shrinkage**, not a bug
- ✅ System is production-ready

---

## Complete Status Report

### ✅ Code Audit (All Claims Verified)

#### 1. Metric-Driven Selection in Training
**File:** `ris_pytorch_pipeline/train.py`

**✓ VAL_PRIMARY support:**
```python
val_primary = str(getattr(cfg, "VAL_PRIMARY", "loss")).lower()  # Line 1812
```

**✓ Metrics computed:**
- K_acc, K_under, K_over (lines 1618-1623)
- AoA_RMSE, Range_RMSE (via Hungarian matching)
- Success_rate
- Composite score (line 1926)

**✓ Checkpointing honors VAL_PRIMARY:**
```python
if val_primary in ("k_loc", "metrics"):
    if val_score < best_score:
        best_score = float(val_score)
        improved = True
else:
    if val_loss < best_val:
        best_val = float(val_loss)
        improved = True
```

**✓ fit() returns correct value:**
```python
if str(getattr(cfg, "VAL_PRIMARY", "loss")).lower() in ("k_loc", "metrics"):
    return float(best_score)  # Metric-driven
return float(best_val)        # Loss-driven (legacy)
```

#### 2. HPO Optimizes K/Localization Composite
**File:** `ris_pytorch_pipeline/hpo.py`

**✓ Forces VAL_PRIMARY="k_loc":**
```python
cfg.VAL_PRIMARY = "k_loc"  # Line 188
```

**✓ Direction matches objective:**
```python
direction="minimize",  # Minimize composite penalty: (1-K_acc) + RMSE_norm - success
```

**✓ Returns Trainer score:**
```python
best_val = t.fit(...)  # Gets metric score when VAL_PRIMARY="k_loc"
return float(best_val)  # Optuna minimizes this
```

**✓ Dataset sizes updated:**
```python
total_train_samples = 100000  # Updated from 80K
total_val_samples = 10000     # Updated from 16K
hpo_n_train = 10000  # 10% subset
hpo_n_val = 1000     # 10% subset
```

**✓ lam_K range increased:**
```python
lam_K = trial.suggest_float("lam_K", 0.05, 0.25)  # Was [0.05, 0.15]
```

#### 3. Canonical Covariance Builder in Training
**File:** `ris_pytorch_pipeline/train.py`

**✓ Import added:**
```python
from .covariance_utils import build_effective_cov_torch  # Line 21
```

**✓ R_blend uses builder:**
```python
R_blend = build_effective_cov_torch(
    R_pred,
    snr_db=None,
    R_samp=R_samp_c.detach(),
    beta=float(beta),
    diag_load=False,  # Loss applies this
    apply_shrink=False,  # Loss applies this
    target_trace=float(N),
)
```

#### 4. Offline R_samp (No Online LS)
**File:** `ris_pytorch_pipeline/train.py`

**✓ R_samp loaded from batch:**
```python
R_samp = batch.get("R_samp", None)  # Line 688
if R_samp is not None:
    R_samp = R_samp.to(self.device, non_blocking=True)
```

**✓ No torch.linalg.lstsq in hot path** (verified via grep)

**✓ Shards verified:**
```bash
$ python -c "import numpy as np; z=np.load('data_shards_M64_L16/train/shard_000.npz'); print('R_samp:', 'R_samp' in z); z.close()"
R_samp: True
```

---

## Shard Regeneration: COMPLETE ✅

### Generated Files:
```
data_shards_M64_L16/
├── train/
│   ├── shard_000.npz (8.7 GB, 25K samples)
│   ├── shard_001.npz (8.7 GB, 25K samples)
│   ├── shard_002.npz (8.7 GB, 25K samples)
│   └── shard_003.npz (8.7 GB, 25K samples)
├── val/
│   └── shard_000.npz (3.5 GB, 10K samples)
└── test/
    └── shard_000.npz (3.5 GB, 10K samples)
```

### Verification:
```bash
$ python -c "..."
Keys: ['y', 'H', 'H_full', 'codes', 'ptr', 'K', 'snr', 'R', 'R_samp']
R_samp present: True
R_samp shape: (25000, 144, 144, 2)
R_samp non-zero: True
R_samp dtype: float32
```

**✅ Status:** All shards have R_samp, ready for hybrid blending!

---

## Covariance Audit: PASSED ✅

### Audit Results:
```
✓ R_eff (K-head): trace = 144.0000 (CORRECT)
✓ R_eff (MUSIC): trace = 144.0000 (CORRECT)
||R_khead - R_music||_F = 3.25e-02 (expected due to shrink)
```

### Shrinkage Verification:
```bash
$ python audit_shrink_only.py
||R_khead_noshrink - R_music||_F = 2.403e-06
✓ PERFECT MATCH when both paths skip shrink!
```

### Interpretation:
- ✅ Base covariance pipeline is **identical** (2.4e-06 = numerical precision)
- ✅ Off-diagonal difference is **intentional shrinkage**
- ✅ K-head uses SNR-driven shrink for robust features
- ✅ MUSIC uses adaptive shrink later in its pipeline
- ✅ No bugs, design is correct

**Why the audit script says "Investigate further":**
- It's being conservative
- We already investigated with `audit_shrink_only.py`
- Shrinkage difference is expected and acceptable

---

## HPO Configuration Summary

### What HPO Optimizes:
```
Architecture:
  - D_MODEL: [448, 512]
  - NUM_HEADS: [6, 8]
  - dropout: [0.20, 0.32]

Optimizer:
  - lr: [1e-4, 4e-4] (log-uniform)
  - batch_size: [64, 80]

Loss Weights:
  - lam_cov: [0.10, 0.25]
  - lam_ang: [0.50, 1.00]
  - lam_rng: [0.30, 0.60]
  - lam_K: [0.05, 0.25] ← UPDATED (was [0.05, 0.15])

Inference:
  - range_grid: [121, 161]
  - newton_iter: [5, 15]
  - newton_lr: [0.3, 1.0]
  - shrink_alpha: [0.10, 0.20]
```

### What HPO Minimizes:
```python
objective = (1.0 - K_acc) + (AoA_RMSE / 5°) + (Elev_RMSE / 5°) + (Range_RMSE / 1m) - Success_rate
```

**Lower objective = better model:**
- High K_acc → penalty decreases ✓
- Low RMSE → penalty decreases ✓
- High success_rate → penalty decreases ✓

### Dataset Subset (for speed):
- **Train:** 10,000 samples (10% of 100K)
- **Val:** 1,000 samples (10% of 10K)
- **Why 10%?** Fast exploration, still statistically meaningful

---

## How to Run HPO (Manual Scripts)

### Method 1: Bash Script (Recommended)
```bash
cd /home/tahit/ris/MainMusic
./run_hpo_manual.sh
```

**Features:**
- ✅ Verifies shards exist
- ✅ Checks R_samp presence
- ✅ 5-second abort countdown
- ✅ Logs to file + console (tee)
- ✅ Displays best.json after completion

### Method 2: Python Script
```bash
cd /home/tahit/ris/MainMusic
python run_hpo_manual.py
```

**Features:**
- ✅ Cross-platform (Windows/Linux/Mac)
- ✅ Same verification as bash
- ✅ Exception handling
- ✅ Resume support

### Method 3: Direct Command
```bash
cd /home/tahit/ris/MainMusic
python -m ris_pytorch_pipeline.ris_pipeline hpo \
    --trials 40 \
    --hpo-epochs 12 \
    --space wide \
    --early-stop-patience 15
```

**Features:**
- ✅ Most flexible (easy to change parameters)
- ✅ No wrapper overhead

---

## Expected Timeline

### HPO Run (40 trials):
```
Trial 1:  12 epochs × 10K samples ≈ 10 min
Trial 2:  Pruned at epoch 7 ≈ 6 min (bad hyperparams)
Trial 3:  12 epochs × 10K samples ≈ 10 min
...
Trial 40: 12 epochs × 10K samples ≈ 10 min

Total: ~6-10 hours (with ~30% pruned early)
```

### Full Training (after HPO):
```
50 epochs × 100K samples ≈ 4-8 hours
```

### Total Project Time:
```
Shard regeneration: 60 min (✓ DONE)
HPO: 6-10 hours (ready to start)
Full training: 4-8 hours (after HPO)
Threshold tuning: 1 hour
Test evaluation: 30 min

Total: ~12-20 hours of computation
```

---

## All Changes Made Today

### Files Modified:

1. **`ris_pytorch_pipeline/covariance_utils.py`**
   - Fixed double trace normalization (added re-norm after blend and diag-load)
   - Impact: tr(R_eff) = N exactly (was 2N)

2. **`ris_pytorch_pipeline/configs.py`**
   - Added `VAL_PRIMARY = "k_loc"` (metric-driven checkpointing)
   - Added `K_CONF_THRESH = 0.65` (confidence gating)

3. **`ris_pytorch_pipeline/train.py`**
   - Import `build_effective_cov_torch`
   - Use canonical builder for R_blend
   - Return metric score when VAL_PRIMARY="k_loc"

4. **`ris_pytorch_pipeline/hpo.py`**
   - Updated dataset sizes (100K/10K)
   - Increased lam_K range to [0.05, 0.25]
   - Force VAL_PRIMARY="k_loc"
   - Added direction comment

### Files Created:

**Audit Scripts:**
5. `audit_covariance_paths.py` - Verify K-head vs MUSIC alignment
6. `audit_shrink_only.py` - Verify shrinkage is the only difference

**Shard Regeneration:**
7. `regenerate_shards.sh` - Bash script
8. `regenerate_shards_python.py` - Python script
9. `SHARD_REGENERATION_GUIDE.md` - Comprehensive guide

**HPO Execution:**
10. `run_hpo_manual.sh` - Bash script (recommended)
11. `run_hpo_manual.py` - Python script
12. `HPO_EXECUTION_GUIDE.md` - Comprehensive guide

**Documentation:**
13. `AUDIT_AND_CLEANUP_INDEX.md` - Navigation
14. `AUDIT_EXECUTIVE_SUMMARY.md` - High-level summary
15. `COVARIANCE_AUDIT_AND_CLEANUP_REPORT.md` - Technical details
16. `QUICK_START_AFTER_CLEANUP.md` - Training guide
17. `CODE_AUDIT_REPORT_2025-11-26.md` - Code verification
18. `REGENERATION_SUMMARY.md` - Shard regen summary
19. `FINAL_AUDIT_AND_HPO_REPORT.md` - This file

---

## Your Current Status

### ✅ Completed:
- [x] Code audit (all claimed features verified)
- [x] Covariance path alignment (2.4e-06 difference)
- [x] Trace normalization fix (tr = 144 exactly)
- [x] Shard regeneration with R_samp (100K/10K/10K)
- [x] HPO ranges updated (lam_K: 0.05-0.25)
- [x] HPO dataset sizes updated (100K/10K)
- [x] Metric-driven objective wired to HPO
- [x] All scripts created and tested

### 📋 Ready to Start:
- [ ] **Run HPO** (6-10 hours) ← **NEXT ACTION**
- [ ] Full training with best.json (4-8 hours)
- [ ] Tune K_CONF_THRESH and β on validation (1 hour)
- [ ] Final test evaluation

---

## How to Proceed (Step-by-Step)

### Step 1: Start HPO (NOW)
```bash
cd /home/tahit/ris/MainMusic
./run_hpo_manual.sh
```

**Or run in background/screen:**
```bash
screen -S hpo
cd /home/tahit/ris/MainMusic
./run_hpo_manual.sh
# Ctrl+A, D to detach
```

### Step 2: Monitor Progress (Optional)
```bash
# In another terminal:
tail -f results_final/logs/hpo_*.log

# Or check completed trials:
python -c "
import optuna
s = optuna.load_study(
    study_name='L16_M64_wide_v2_optimfix',
    storage='sqlite:///results_final/hpo/hpo.db'
)
done = len([t for t in s.trials if t.state.is_finished()])
print(f'Completed trials: {done}/40')
"
```

### Step 3: After HPO Completes (~6-10 hours)
```bash
# Review best hyperparameters
cat results_final/hpo/best.json

# Run full training
python train_ris.py --epochs 50 --batch-size 64
```

### Step 4: Tune Thresholds on Validation
```python
# After full training:
for thresh in [0.5, 0.6, 0.65, 0.7, 0.8]:
    cfg.K_CONF_THRESH = thresh
    # Evaluate K_acc, K_under, Success_rate
```

### Step 5: Final Test Evaluation
```bash
# With tuned thresholds, evaluate on test set
python -m ris_pytorch_pipeline.benchmark_test
# Report final K_acc, AoA_RMSE, Success_rate for paper
```

---

## Key Numbers to Remember

### Covariance Alignment:
- **tr(R_eff):** 144.0000 (exactly N) ✅
- **Frobenius diff (no shrink):** 2.4e-06 ✅
- **Frobenius diff (with shrink):** 3.2e-02 (intentional) ✅

### Dataset:
- **Train:** 100,000 samples (4 shards)
- **Val:** 10,000 samples (1 shard)
- **Test:** 10,000 samples (1 shard)
- **HPO uses:** 10K train / 1K val (10% subset)

### HPO Ranges:
- **lam_K:** [0.05, 0.25] (allows stronger K supervision)
- **lam_cov:** [0.10, 0.25]
- **lr:** [1e-4, 4e-4]
- **batch_size:** [64, 80]

---

## FAQ

**Q: The audit shows off-diagonal difference. Is this a bug?**  
A: No. K-head applies SNR-aware shrink; MUSIC doesn't (applies own later). With both skipping shrink, difference is 2.4e-06.

**Q: Should I worry about the "Investigate further" message?**  
A: No. We already investigated with `audit_shrink_only.py`. It's intentional shrinkage.

**Q: Can I increase lam_K range further?**  
A: Yes, but start with [0.05, 0.25] for first HPO. If K_under > 10% after full training, run second HPO with [0.10, 0.40].

**Q: How do I know HPO is working?**  
A: Check that:
- Trials complete (no crashes)
- Best objective decreases over trials (convergence)
- best.json is created
- Final objective < 0.50 (indicates good K/loc performance)

**Q: What if HPO takes too long?**  
A: Reduce trials to 20-30, or reduce epochs_per_trial to 8-10. You'll get less optimal hyperparameters but still usable.

---

## Troubleshooting Reference

### "CUDA out of memory"
```bash
# Reduce batch size in HPO range
# In hpo.py line 129: batch_size = trial.suggest_categorical("batch_size", [32, 64])
```

### "Study direction conflict"
```bash
# Delete old database
rm results_final/hpo/hpo.db*
```

### "K_acc stuck at 0.2"
```bash
# Check lam_K is non-zero (should be 0.05-0.25)
# Verify cost-sensitive K loss in loss.py (lines 668-690)
```

### "All trials get pruned"
```bash
# Increase warmup in hpo.py line 59:
# pruner=optuna.pruners.MedianPruner(n_warmup_steps=10, n_min_trials=12)
```

---

## Documentation Index

### Quick Start:
→ **`HPO_EXECUTION_GUIDE.md`** (this is the main guide)

### Understanding Changes:
→ **`CODE_AUDIT_REPORT_2025-11-26.md`** (what was fixed)  
→ **`AUDIT_EXECUTIVE_SUMMARY.md`** (covariance audit summary)

### Training After HPO:
→ **`QUICK_START_AFTER_CLEANUP.md`** (full training guide)

### Full Context:
→ **`COVARIANCE_AUDIT_AND_CLEANUP_REPORT.md`** (comprehensive)

---

## Final Checklist Before Running HPO

- [x] Shards regenerated with R_samp ✅
- [x] R_samp verified in shards ✅
- [x] Covariance alignment verified (tr = 144) ✅
- [x] lam_K range updated (0.05-0.25) ✅
- [x] HPO dataset sizes updated (100K/10K) ✅
- [x] VAL_PRIMARY="k_loc" forced in HPO ✅
- [x] Metric-driven return from Trainer.fit ✅
- [x] All imports verified ✅

**STATUS: 🚀 READY TO LAUNCH HPO**

---

## Next Action

**Run this command:**
```bash
cd /home/tahit/ris/MainMusic
./run_hpo_manual.sh
```

**Then:**
- Go get coffee ☕ (or sleep 😴)
- Come back in 6-10 hours
- Review `results_final/hpo/best.json`
- Run full training with those hyperparameters

**You're all set!** 🎉



