# Bug Fixes Report - December 3, 2025 (Part 2)

## Summary

This report documents critical fixes to the evaluation protocol that were hiding actual MUSIC performance.

---

## The Core Problem

**Symptom:** All medians showed 90°/10m (penalty values) even though RMSE was small (~13°/9°/2m).

**Root Cause:** The evaluation protocol was:
1. Computing Hungarian-matched errors
2. Applying strict gating (30°/20°/5m)
3. **Replacing failed matches with 90°/10m penalties**
4. Computing medians on this **penalty-contaminated** distribution

This meant that when success rate was 0% (expected for untrained models), the medians were 100% penalty values, hiding the actual MUSIC performance.

---

## Fix 1: Use RAW Errors (No Penalties)

### Changed: `eval_angles.py::eval_scene_angles_ranges`

**Before:**
- Returned gated errors (only matches within tolerance)
- Caller added 90°/10m penalties for failed scenes

**After:**
- Returns ALL raw errors from Hungarian matching (no gating)
- Returns `success_flag` separately (strict: all sources within tolerance)
- Returns `raw_phi_errors`, `raw_theta_errors`, `raw_r_errors` arrays

```python
return {
    # RAW metrics (no gating, no penalties)
    "rmse_phi": float(np.sqrt(np.mean(dphi_raw**2))),
    "med_phi": float(np.median(dphi_raw)),
    ...
    # Success flag (strict: ALL sources within tolerance)
    "success_flag": all_within_tol and len(dphi_raw) == Kg,
    # Raw error arrays for aggregation
    "raw_phi_errors": dphi_raw.tolist(),
    ...
}
```

### Changed: `train.py::_eval_hungarian_metrics`

**Before:**
```python
if metrics["med_phi"] is not None:
    all_errors["phi"].append(metrics["med_phi"])
else:
    all_errors["phi"].append(90.0)  # PENALTY!
```

**After:**
```python
if metrics.get("raw_phi_errors") and len(metrics["raw_phi_errors"]) > 0:
    all_errors["phi"].extend(metrics["raw_phi_errors"])  # RAW errors!
    if metrics.get("success_flag", False) and (k_hat == k_true):
        success_count += 1
# NO MORE PENALTIES!
```

---

## Fix 2: Updated Logging

**Before:**
```
📊 Overall Angle/Range Errors (Hungarian-matched, N=1000):
  Azimuth (φ):     median=90.000°,   95th=90.000°
```

**After:**
```
📊 RAW Angle/Range Errors (no penalties, N=2500 source-pairs from 1000 scenes):
  Azimuth (φ):     median=15.23°,   95th=45.67°,   RMSE=18.45°
```

Now you can see:
- Actual MUSIC performance (not penalty artifacts)
- Number of source-pairs vs scenes
- RMSE directly computed from raw errors

---

## Fix 3: LR Scheduler Warning

**Problem:** `UserWarning: Detected call of lr_scheduler.step() before optimizer.step()`

**Fix:** Initialize scheduler step counter to suppress the warning:
```python
self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
self.sched._step_count = 1  # Suppress warning
```

---

## What to Expect Now

1. **Medians will reflect actual MUSIC performance**, not penalties
2. **Early epochs will show high errors** (expected for untrained models)
3. **Errors should decrease** as training progresses
4. **success_rate will be 0** early but should increase as model improves
5. **No more LR scheduler warning**

---

## How to Interpret New Metrics

| Metric | Meaning |
|--------|---------|
| `med_phi` | Median azimuth error across all matched source-pairs (RAW) |
| `RMSE_phi` | Root mean square error (RAW) |
| `success_rate` | Fraction of scenes where K correct AND all sources within tolerance |
| `K_acc` | Fraction of scenes with correct K estimate |
| `n_scenes_with_matches` | Scenes with at least one Hungarian match |

**Good progress indicators:**
- `med_phi`, `med_theta`, `med_r` decreasing over epochs
- `RMSE_phi`, `RMSE_theta`, `RMSE_r` decreasing
- `K_acc` increasing
- `success_rate` increasing (will be 0 early, that's OK)

---

## Re-run HPO

```bash
cd /home/tahit/ris/MainMusic
rm -rf results_final/hpo/*.db results_final/hpo/*.journal
rm -rf results_final_L16_12x12/hpo/*.db results_final_L16_12x12/hpo/*.journal
./run_hpo_manual.sh
```

You should now see actual MUSIC performance in the logs, not 90°/10m everywhere.

