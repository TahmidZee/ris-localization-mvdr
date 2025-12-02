# Quick Start Guide After Audit & Cleanup
**Updated:** 2025-11-26

---

## TL;DR

**What changed:**  
- Fixed critical trace normalization bug (`tr(R_eff) = 2N` → `N`)
- K-head and MUSIC now use identical covariance (verified)
- Added metric-driven checkpointing (`VAL_PRIMARY = "k_loc"`)

**What you need to do:**  
Choose one of two paths below, then train.

---

## Path A: Full Production (with Hybrid Blending)

**Use this if:** You want best low-SNR performance with hybrid `R_pred + R_samp` blending.

### Step 1: Regenerate Shards with R_samp (~30-60 min)
```bash
cd /home/tahit/ris/MainMusic

python -c "
from ris_pytorch_pipeline.dataset import prepare_split_shards, set_sampling_overrides_from_cfg
from ris_pytorch_pipeline.configs import mdl_cfg
from pathlib import Path

set_sampling_overrides_from_cfg(mdl_cfg)
prepare_split_shards(
    root_dir=Path('data_shards_M64_L16'),
    n_train=100000,
    n_val=10000,
    n_test=10000,
    shard_size=25000,
    seed=42,
    eta_perturb=0.0,
    override_L=16
)
"
```

### Step 2: Verify Shards Have R_samp
```bash
python -c "import numpy as np; z = np.load('data_shards_M64_L16/train/shard_000.npz'); print('R_samp present:', 'R_samp' in z.keys()); z.close()"
# Should print: R_samp present: True
```

### Step 3: Train
```bash
# cfg.HYBRID_COV_BETA = 0.3 is already set
python train_ris.py --epochs 50 --batch-size 64
```

---

## Path B: Quick Test (no Hybrid, train immediately)

**Use this if:** You want to train NOW and don't need hybrid blending yet.

### Step 1: Disable Hybrid in Config
```python
# In ris_pytorch_pipeline/configs.py (or via CLI override):
self.HYBRID_COV_BETA = 0.0  # Disables hybrid, uses pure R_pred
```

Or via environment variable:
```bash
export HYBRID_COV_BETA=0.0
```

### Step 2: Train
```bash
python train_ris.py --epochs 50 --batch-size 64
```

**Note:** This is still production-ready, just without the hybrid blend. You can add hybrid later by regenerating shards.

---

## Common Commands

### Run Audit (verify covariance paths are aligned)
```bash
python audit_covariance_paths.py
# Exit 0 = pass, exit 1 = fail
# Check that tr(R_eff) = 144 in output
```

### Run Overfit Test (sanity check)
```bash
python test_overfit.py
# Should see val_loss < 0.1 within a few epochs
```

### Check Shard Keys
```bash
python -c "import numpy as np; z = np.load('data_shards_M64_L16/train/shard_000.npz'); print('Keys:', list(z.keys())); z.close()"
```

### Tune K_CONF_THRESH (after training)
```python
# In a notebook or script:
from ris_pytorch_pipeline.infer import hybrid_estimate_final
from ris_pytorch_pipeline.eval_angles import eval_scene_angles_ranges

for thresh in [0.5, 0.6, 0.65, 0.7, 0.8]:
    cfg.K_CONF_THRESH = thresh
    # Run validation, compute K_acc, K_under, Success_rate
    # Pick threshold that minimizes K_under while keeping K_acc high
```

---

## What to Watch During Training

### Old (NMSE-driven, NOT recommended):
```
Epoch 10: val_loss=0.234, saving best
```

### New (metric-driven, RECOMMENDED):
```
Epoch 10:
  val_loss=0.234
  K_acc=0.87, K_under=0.06, K_over=0.07
  AoA_RMSE=1.23°, Range_RMSE=0.45m
  Success_rate=0.78
  → New best (K_loc score: 0.812)
```

**Key metrics:**
- `K_acc`: Fraction with K̂ == K_true
- `K_under`: Fraction with K̂ < K_true (WORST error, missed sources)
- `AoA_RMSE`: Angular error (degrees)
- `Success_rate`: % with K̂ correct AND angles/range within threshold

---

## Expected Results After Training

### Good Training Run (50 epochs, β=0.3):
```
Best checkpoint (epoch 42):
  K_acc: 0.89
  K_under: 0.04 (minimize this!)
  AoA_RMSE: 0.8° (sub-degree!)
  Range_RMSE: 0.3m
  Success_rate: 0.83
```

### If You See This, Something's Wrong:
```
Epoch 50:
  K_acc: 0.45 (too low)
  K_under: 0.40 (way too high, missing sources)
  AoA_RMSE: 5.0° (too coarse)
```
→ Check:
- Is `lam_K` too small? (should be ≥ 0.5)
- Is K-head frozen? (should be trainable)
- Is cost-sensitive K loss working? (check `loss.py`)

---

## Troubleshooting

### "ImportError: cannot import build_effective_cov_torch"
→ Make sure you're running from the correct directory:
```bash
cd /home/tahit/ris/MainMusic
python train_ris.py ...
```

### "KeyError: 'R_samp'"
→ Your shards don't have `R_samp`. Either:
- Regenerate shards (Path A above), or
- Set `HYBRID_COV_BETA = 0.0` (Path B above)

### "tr(R_eff) = 288 instead of 144"
→ This bug was fixed. Make sure you're using the updated `covariance_utils.py`. Re-run audit:
```bash
python audit_covariance_paths.py
```

### "K_acc stuck at 0.2 (random guessing)"
→ Check:
- Is `lam_K` non-zero? (should be ≥ 0.5)
- Is K-head in optimizer? (`model.k_mlp` and `model.k_direct_mlp` trainable)
- Is cost-sensitive loss enabled? (check `UltimateHybridLoss` in `loss.py`)

---

## File Overview (What Each Document Does)

- **`AUDIT_EXECUTIVE_SUMMARY.md`** ← **START HERE** (what I did, 2-page summary)
- **`COVARIANCE_AUDIT_AND_CLEANUP_REPORT.md`** ← Comprehensive 7-section report
- **`QUICK_START_AFTER_CLEANUP.md`** ← This file (how to train now)
- **`audit_covariance_paths.py`** ← Audit script (run anytime to verify alignment)
- **`audit_shrink_only.py`** ← Verify shrinkage is the only difference

---

## Next Steps (Checklist)

- [ ] Choose Path A (with hybrid) or Path B (without hybrid)
- [ ] Run audit to verify: `python audit_covariance_paths.py`
- [ ] Run overfit test: `python test_overfit.py`
- [ ] Full training: `python train_ris.py --epochs 50`
- [ ] Tune `K_CONF_THRESH` on validation (sweep 0.5-0.8)
- [ ] Tune `HYBRID_COV_BETA` on validation (sweep 0.0-0.5)
- [ ] Evaluate on test set: K_acc, AoA_RMSE, Success_rate
- [ ] Write paper results section 🎉

---

**Questions?** Read `AUDIT_EXECUTIVE_SUMMARY.md` first, then `COVARIANCE_AUDIT_AND_CLEANUP_REPORT.md` for details.

**Status:** ✅ READY TO TRAIN




