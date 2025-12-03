# Bug Fixes Report - December 3, 2025

## Summary

Multiple critical bugs were discovered during HPO runs that caused validation metrics to fail completely. All bugs have been fixed and committed.

---

## Bug 1: Validation Count Mismatch Assertion

### Symptom
```
⚠️  WARNING: Expected 1600 val samples, got 1000!
⚠️  600 scenes were dropped or missing!
[VAL METRICS] Skipped due to error: CRITICAL: Validation count mismatch! Expected 1600, got 1000.
```

### Root Cause
The function `_eval_hungarian_metrics` in `train.py` had a **hardcoded assertion** expecting exactly 1600 validation samples:

```python
expected_val_count = 1600  # HARDCODED!
assert actual_count == expected_val_count, ...
```

During HPO, we use a **subset** of validation data (1000 samples instead of full 10K), so this assertion always failed.

### Fix
Removed the hardcoded assertion and replaced with a simple log:

```python
# Before: assert actual_count == expected_val_count
# After:
print(f"[VAL METRICS] Processed {actual_count} samples", flush=True)
```

### Impact
- Validation metrics were being **completely skipped** due to the assertion error
- HPO was optimizing raw loss instead of K_acc/RMSE metrics
- `VAL_PRIMARY=k_loc` was effectively broken

---

## Bug 2: `k_mdl_correct` Not Initialized

### Symptom
```
[VAL METRICS] Skipped due to error: local variable 'k_mdl_correct' referenced before assignment
```

### Root Cause
The variable `k_mdl_correct` was used to track MDL baseline accuracy but was **never initialized**:

```python
# Line 1577: k_mdl_correct += 1  # ERROR: never initialized!
```

### Fix
Added initialization at the start of `_eval_hungarian_metrics`:

```python
success_count = 0
k_mdl_correct = 0  # Track MDL baseline accuracy  # ADDED
```

### Impact
- Validation metrics computation crashed after processing samples
- K_mdl_acc metric was unavailable

---

## Bug 3: All Angle Errors = 90° (CRITICAL)

### Symptom
```
📊 Overall Angle/Range Errors (Hungarian-matched, N=1000):
  Azimuth (φ):     median=90.000°,   95th=90.000°
  Elevation (θ):   median=90.000°,   95th=90.000°
  Range (r):       median=10.000m,   95th=10.000m
```

ALL 1000 validation samples showed maximum penalty values (90° for angles, 10m for range).

### Root Cause
The `blend_beta` variable passed to `angle_pipeline_gpu` was **always 0.0**:

```python
# Line 1445:
blend_beta = 0.0  # Set to 0.0 by default

# Only changed if HYBRID_COV_BLEND is True (which is for CPU path)
if getattr(cfg, "HYBRID_COV_BLEND", False):
    blend_beta = ...

# Line 1523: GPU path uses blend_beta which is still 0.0!
phi_music, theta_music, _ = angle_pipeline_gpu(
    ...,
    beta=blend_beta,  # ALWAYS 0.0!
)
```

With `beta=0.0`, the GPU MUSIC was running on **pure R_pred** without hybrid blending with R_samp. The rank-5 R_pred from the untrained network was too low-rank to produce meaningful MUSIC estimates.

### Fix
Use `cfg.HYBRID_COV_BETA` directly instead of `blend_beta`:

```python
if use_gpu_music:
    hybrid_beta = float(getattr(cfg, "HYBRID_COV_BETA", 0.3))  # NEW
    phi_music, theta_music, _ = angle_pipeline_gpu(
        ...,
        beta=hybrid_beta,  # Use cfg value, not blend_beta
    )
```

### Impact
- GPU MUSIC was effectively broken (no hybrid blending)
- All MUSIC estimates were garbage → all Hungarian matches failed
- All samples assigned 90°/10m penalty → metrics useless

---

## Bug 4: Silent MUSIC Failures

### Symptom
No `[MUSIC DEBUG]` output in logs, even though MUSIC should be running.

### Root Cause
MUSIC exceptions were caught silently unless `MUSIC_DEBUG=True`:

```python
except Exception as e:
    if getattr(cfg, "MUSIC_DEBUG", False):  # Usually False!
        print(f"[MUSIC] Warning: angle_pipeline failed: {e}")
```

### Fix
Always print the first failure with full traceback:

```python
except Exception as e:
    # ALWAYS print first failure to help debug
    if i == 0 or getattr(cfg, "MUSIC_DEBUG", False):
        import traceback
        print(f"[MUSIC] Warning: angle_pipeline failed for sample {i}: {e}")
        if i == 0:
            traceback.print_exc()
```

Also added debug prints to verify MUSIC execution:
- `[VAL MUSIC] Entering MUSIC block for sample 0, batch 0`
- `[MUSIC DEBUG] Sample 0: K_hat=..., phi=..., theta=...`

---

## Commits

| Commit | Description |
|--------|-------------|
| `e0e4017` | Fix validation count mismatch bug |
| `528f22b` | Add debug prints for MUSIC failures, initialize k_mdl_correct |
| `cde8d62` | Fix validation MUSIC: use cfg.HYBRID_COV_BETA instead of blend_beta |

---

## Verification Checklist

After these fixes, the HPO logs should show:

- [x] `[VAL METRICS] Processed 1000 samples` (no assertion error)
- [x] `[VAL MUSIC] Entering MUSIC block...` (MUSIC is being called)
- [x] `[MUSIC DEBUG] Sample 0: K_hat=..., phi=..., theta=...` (MUSIC output)
- [x] Angle errors < 90° (reasonable values, not all max penalty)
- [x] K_mdl_acc metric available
- [x] No `referenced before assignment` errors

---

## How to Re-run HPO

```bash
cd /home/tahit/ris/MainMusic

# Clear old trials
rm -rf results_final/hpo/*.db results_final/hpo/*.journal
rm -rf results_final_L16_12x12/hpo/*.db results_final_L16_12x12/hpo/*.journal
rm -rf results_final/checkpoints/*.pt results_final_L16_12x12/checkpoints/*.pt

# Restart HPO
./run_hpo_manual.sh
```

---

## Lessons Learned

1. **Never hardcode expected counts** - Use dynamic checks or make them configurable
2. **Initialize all counter variables** - Easy to miss in complex functions
3. **Trace variable flow carefully** - `blend_beta` was set correctly for CPU path but not GPU path
4. **Always log first failure** - Silent exceptions make debugging impossible
5. **Add debug prints for critical paths** - MUSIC execution should be visible in logs

