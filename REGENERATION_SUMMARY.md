# Shard Regeneration - Executive Summary
**Date:** 2025-11-26

---

## ✅ You asked for:
1. Manual scripts to regenerate shards
2. Check if `ris_pipeline.py` needs updates
3. Confirmation about clearing old shards

## ✅ What I provided:

### 1. Three Ways to Regenerate Shards:

#### **Option A: Bash Script** (Easiest)
```bash
cd /home/tahit/ris/MainMusic
./regenerate_shards.sh
```

#### **Option B: Python Script** (Cross-platform)
```bash
cd /home/tahit/ris/MainMusic
python regenerate_shards_python.py
```

#### **Option C: Manual Command** (Full control)
```bash
cd /home/tahit/ris/MainMusic
python -m ris_pytorch_pipeline.ris_pipeline pregen-split \
    --out_dir data_shards_M64_L16 \
    --n-train 100000 --n-val 10000 --n-test 10000 \
    --shard 25000 --eta 0.0 --L 16 --seed 42 \
    --phi-fov-deg 60.0 --theta-fov-deg 30.0 \
    --r-min 0.5 --r-max 10.0 \
    --snr-min -5.0 --snr-max 20.0
```

### 2. About `ris_pipeline.py`:
**✅ NO UPDATES NEEDED!**

The existing `ris_pipeline.py` already calls `prepare_split_shards()` from `dataset.py`, which we updated to include offline `R_samp` computation (lines 326-339). It will automatically generate shards with `R_samp`.

### 3. About Clearing Old Shards:
**✅ AUTOMATIC BACKUP!**

Both scripts automatically:
- Back up old `data_shards_M64_L16` → `data_shards_M64_L16_backup_TIMESTAMP`
- Create fresh directory structure
- Generate new shards with `R_samp`

**No manual deletion needed.** Your old data is safe in the backup.

---

## What Happens During Regeneration

### Step-by-Step:
1. **Backup** (if old shards exist)
   - `data_shards_M64_L16` → `data_shards_M64_L16_backup_20251126_141234`

2. **Create fresh structure**
   - `data_shards_M64_L16/{train,val,test}/`

3. **Generate shards** (~30-60 min)
   - Train: 100,000 samples (4 shards)
   - Val: 10,000 samples (1 shard)
   - Test: 10,000 samples (1 shard)
   - **Each sample now includes `R_samp`** (pre-computed offline)

4. **Verify**
   - Check shard counts
   - Verify `R_samp` is present and non-zero
   - Print shape: `(25000, 144, 144, 2)` for first shard

---

## What's New in These Shards

### Old Shards:
```python
Keys: ['y', 'H', 'H_full', 'codes', 'ptr', 'K', 'snr', 'R']
# NO R_samp
```

### New Shards:
```python
Keys: ['y', 'H', 'H_full', 'codes', 'ptr', 'K', 'snr', 'R', 'R_samp']
# R_samp is pre-computed offline (NumPy/CPU)
# tr(R_samp) = N = 144 (trace-normalized)
# Hermitized and ready for hybrid blending
```

### Why This Matters:
- **Hybrid blending:** `R_eff = (1-β)*R_pred + β*R_samp` with `β=0.3`
- **K-head and MUSIC alignment:** Both use the same `R_eff` now
- **Performance:** Offline pre-compute avoids expensive GPU `torch.linalg.lstsq` during training

---

## Time & Resources

### Generation Time:
- **Total:** ~30-60 minutes (CPU-bound)
- **Train (100K):** ~25-45 min
- **Val (10K):** ~3-5 min
- **Test (10K):** ~3-5 min

### Disk Space:
- **New shards:** ~6-8 GB
- **Backup (if exists):** ~6-8 GB
- **Total:** ~12-16 GB (if you have old shards)

### RAM:
- **Recommended:** 16 GB+
- **Minimum:** 8 GB (may need smaller shard size)

---

## After Regeneration

### 1. Verify Success:
```bash
python audit_covariance_paths.py
# Should report: tr(R_eff) = 144.0000, R_samp_used=True
```

### 2. Run Overfit Test:
```bash
python test_overfit.py
# Should see val_loss < 0.1 within a few epochs
```

### 3. Full Training:
```bash
python train_ris.py --epochs 50 --batch-size 64
# cfg.HYBRID_COV_BETA = 0.3 is already set
# Watch for: K_acc, AoA_RMSE, Success_rate
```

---

## Quick Reference

### Files Created:
1. **`regenerate_shards.sh`** - Bash script (Linux/Mac)
2. **`regenerate_shards_python.py`** - Python script (all platforms)
3. **`SHARD_REGENERATION_GUIDE.md`** - Comprehensive guide (troubleshooting, FAQ)
4. **`REGENERATION_SUMMARY.md`** - This file (quick reference)

### Key Commands:
```bash
# Regenerate (choose one):
./regenerate_shards.sh
python regenerate_shards_python.py

# Verify R_samp:
python -c "import numpy as np; z=np.load('data_shards_M64_L16/train/shard_000.npz'); print('R_samp:', 'R_samp' in z); z.close()"

# Audit alignment:
python audit_covariance_paths.py

# Train:
python train_ris.py --epochs 50 --batch-size 64
```

---

## Rollback (if needed)

If something goes wrong:
```bash
# Find your backup
ls -d data_shards_M64_L16_backup_*

# Restore (replace TIMESTAMP)
rm -rf data_shards_M64_L16
mv data_shards_M64_L16_backup_TIMESTAMP data_shards_M64_L16
```

---

## FAQ

**Q: Which script should I use?**  
A: `./regenerate_shards.sh` on Linux/Mac. `python regenerate_shards_python.py` if you prefer Python or on Windows.

**Q: Can I abort during generation?**  
A: Yes, Ctrl+C within first 5 seconds. After that, let it finish (or you'll have partial shards).

**Q: Will this delete my model checkpoints?**  
A: No. Shards are in `data_shards_M64_L16/`, checkpoints are in `results_final/models/`.

**Q: Can I change the parameters (SNR range, etc.)?**  
A: Yes. Edit the scripts or use the manual command with different `--snr-min`, etc.

---

## Next Steps

1. **Choose your script** (bash or python)
2. **Run it:** `./regenerate_shards.sh`
3. **Wait:** ~30-60 minutes (go get coffee ☕)
4. **Verify:** `python audit_covariance_paths.py`
5. **Train:** `python train_ris.py --epochs 50`

---

**Ready?** Run your chosen script now!

**Need details?** Read `SHARD_REGENERATION_GUIDE.md`

**Need help?** All scripts have built-in verification and error messages.




