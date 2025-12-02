# Shard Regeneration Guide
**Date:** 2025-11-26  
**Purpose:** Regenerate shards with offline R_samp computation for hybrid blending

---

## Quick Start

### Option 1: Bash Script (Recommended)
```bash
cd /home/tahit/ris/MainMusic
./regenerate_shards.sh
```

### Option 2: Python Script
```bash
cd /home/tahit/ris/MainMusic
python regenerate_shards_python.py
```

### Option 3: Manual Command
```bash
cd /home/tahit/ris/MainMusic
python -m ris_pytorch_pipeline.ris_pipeline pregen-split \
    --out_dir data_shards_M64_L16 \
    --n-train 100000 \
    --n-val 10000 \
    --n-test 10000 \
    --shard 25000 \
    --eta 0.0 \
    --L 16 \
    --seed 42 \
    --phi-fov-deg 60.0 \
    --theta-fov-deg 30.0 \
    --r-min 0.5 \
    --r-max 10.0 \
    --snr-min -5.0 \
    --snr-max 20.0
```

---

## What These Scripts Do

### 1. Backup Old Shards
- Moves `data_shards_M64_L16` → `data_shards_M64_L16_backup_TIMESTAMP`
- Safe: your old data is preserved if you need to roll back

### 2. Create Fresh Directory Structure
- Creates `data_shards_M64_L16/{train,val,test}/`

### 3. Generate Shards with R_samp
- Calls `prepare_split_shards()` from `dataset.py`
- **Key difference from old shards:** Each sample now includes:
  - `R_samp`: Offline pre-computed sample covariance from snapshots
  - Built using `build_sample_covariance_from_snapshots()` (NumPy/CPU)
  - Hermitized and trace-normalized to `tr(R_samp) = N = 144`

### 4. Verify R_samp Presence
- Checks first training shard for `R_samp` key
- Verifies it's non-zero (not all zeros)
- Prints shape and dtype

---

## Configuration

### Dataset Size:
- **Train:** 100,000 samples (4 shards @ 25K each)
- **Val:** 10,000 samples (1 shard)
- **Test:** 10,000 samples (1 shard)
- **Total:** 120,000 samples

### Physical Parameters:
- **L:** 16 snapshots (few-snapshot regime)
- **PHI FOV:** ±60° (azimuth)
- **THETA FOV:** ±30° (elevation)
- **Range:** 0.5m to 10.0m
- **SNR:** -5 dB to 20 dB (targeted distribution)

### Why These Values?
- **L=16:** Paper focuses on few-snapshot regime (vs baselines at L=100+)
- **PHI ±60°, THETA ±30°:** Realistic wall-mounted RIS panel
- **SNR -5 to 20 dB:** Covers low-SNR robustness and high-SNR precision
- **100K train:** Sufficient for 16M parameter model without overfitting

---

## Time Estimates

### Generation Time (~30-60 minutes):
- **Breakdown:**
  - Train (100K samples): ~25-45 min
  - Val (10K samples): ~3-5 min
  - Test (10K samples): ~3-5 min

### Bottlenecks:
- **R_samp computation:** Least-squares solve per sample (16×16 → 144)
- **CPU-bound:** NumPy/CPU (no GPU acceleration for this step)
- **Disk I/O:** Writing 120K samples @ ~50KB each ≈ 6GB

### Speed Tips:
- Run on machine with many CPU cores (parallelizes across shards)
- Use fast SSD for `data_shards_M64_L16/`
- Don't run other heavy processes during generation

---

## Verification Checklist

After regeneration completes, verify:

### 1. Shard Counts
```bash
ls -1 data_shards_M64_L16/train/*.npz | wc -l  # Should be 4
ls -1 data_shards_M64_L16/val/*.npz | wc -l    # Should be 1
ls -1 data_shards_M64_L16/test/*.npz | wc -l   # Should be 1
```

### 2. R_samp Present
```bash
python -c "
import numpy as np
z = np.load('data_shards_M64_L16/train/shard_000.npz')
print('Keys:', list(z.keys()))
print('R_samp present:', 'R_samp' in z.keys())
if 'R_samp' in z.keys():
    print('R_samp shape:', z['R_samp'].shape)
    print('R_samp non-zero:', np.any(z['R_samp'] != 0))
z.close()
"
```

**Expected output:**
```
Keys: ['y', 'H', 'H_full', 'codes', 'ptr', 'K', 'snr', 'R', 'R_samp']
R_samp present: True
R_samp shape: (25000, 144, 144, 2)
R_samp non-zero: True
```

### 3. Run Audit
```bash
python audit_covariance_paths.py
# Exit 0 = pass
# Check that tr(R_eff) = 144 in output
```

---

## Troubleshooting

### "MemoryError during R_samp computation"
- **Cause:** Insufficient RAM for NumPy lstsq on large batches
- **Fix:** Reduce shard size (e.g., `--shard 10000` instead of 25000)

### "Process killed (exit code 137)"
- **Cause:** OOM killer terminated the process
- **Fix:** 
  - Close other applications
  - Reduce shard size
  - Use a machine with more RAM (need ~16GB+ for full generation)

### "R_samp is all zeros"
- **Cause:** Bug in `build_sample_covariance_from_snapshots`
- **Fix:** 
  - Check `angle_pipeline.py` line 326-339 in `prepare_shards`
  - Verify `H_full` is being passed correctly (not `H_eff`)

### "ImportError: cannot import build_sample_covariance_from_snapshots"
- **Cause:** Path issue
- **Fix:** Run from `/home/tahit/ris/MainMusic` (not a subdirectory)

### "Shards generated but audit fails with tr(R_eff) = 288"
- **Cause:** You're using old `covariance_utils.py` (pre-fix)
- **Fix:** Pull latest `covariance_utils.py` with trace normalization fixes

---

## After Regeneration

### 1. Run Overfit Test
```bash
python test_overfit.py
# Should see val_loss < 0.1 within a few epochs
```

### 2. Full Training
```bash
python train_ris.py --epochs 50 --batch-size 64
# cfg.HYBRID_COV_BETA = 0.3 is already set
# K-head and MUSIC will now use the same hybrid R_eff
```

### 3. Monitor Metrics
Watch for these in training logs:
- `K_acc`: K̂ accuracy (target: >0.85)
- `K_under`: K̂ underestimation (minimize, target: <0.10)
- `AoA_RMSE`: Angular error in degrees (target: <1.0°)
- `Range_RMSE`: Range error in meters (target: <0.5m)
- `Success_rate`: End-to-end accuracy (target: >0.80)

---

## Rollback (if needed)

If something goes wrong and you need the old shards:

```bash
# Find backup directory
ls -d data_shards_M64_L16_backup_*

# Restore (replace TIMESTAMP with actual timestamp)
rm -rf data_shards_M64_L16
mv data_shards_M64_L16_backup_TIMESTAMP data_shards_M64_L16

# Disable hybrid blending to use old shards
# In configs.py: self.HYBRID_COV_BETA = 0.0
```

---

## FAQ

**Q: Do I have to regenerate shards?**  
A: Only if you want hybrid blending (`β > 0`). If you set `β=0` in config, old shards work fine.

**Q: Can I regenerate just train or just val?**  
A: Yes, but easier to regenerate all three (train/val/test) for consistency.

**Q: Will this overwrite my model checkpoints?**  
A: No. Shards are separate from model weights. Your `results_final/models/` is untouched.

**Q: Can I use different ranges (e.g., SNR -10 to 25 dB)?**  
A: Yes. Edit the script or pass different `--snr-min` / `--snr-max` to manual command.

**Q: How much disk space do I need?**  
A: ~6-8 GB for 120K samples (50-70 KB per sample).

---

## Script Comparison

| Feature | Bash Script | Python Script | Manual Command |
|---------|-------------|---------------|----------------|
| Backup old shards | ✓ | ✓ | ✗ (manual) |
| Progress messages | ✓ | ✓ | ✗ |
| Verification | ✓ | ✓ | ✗ (manual) |
| 5-sec countdown | ✓ | ✓ | ✗ |
| Cross-platform | Linux/Mac | All | All |
| Easiest to modify | ✓ | ✓ | ✓ |

**Recommendation:** Use bash script on Linux/Mac, Python script on Windows or if you prefer Python.

---

## Related Documentation

- **`AUDIT_AND_CLEANUP_INDEX.md`** - Navigation guide for all audit docs
- **`QUICK_START_AFTER_CLEANUP.md`** - Step-by-step training guide
- **`COVARIANCE_AUDIT_AND_CLEANUP_REPORT.md`** - Why R_samp is needed

---

**Ready to regenerate?** Run `./regenerate_shards.sh` or `python regenerate_shards_python.py`

**Need help?** Check troubleshooting section above or run audit after generation.




