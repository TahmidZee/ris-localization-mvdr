# 🔍 Zero Loss Diagnosis Report

**Issue:** Train loss is 0 and validation loss doesn't change
**Status:** Investigating

---

## 🚨 Most Likely Causes

Based on the symptoms, here are the most likely causes ranked by probability:

### 1. **Loss Weight Configuration Issue** ⭐ MOST LIKELY
**Probability: 90%**

**Symptoms match if:**
- All loss weights are set to 0
- `lam_cov` is 0 (critical weight for NMSE)
- Loss components cancel out

**Check:**
```python
# In loss.py, the forward() returns:
total = (
    self.lam_cov * loss_nmse +          # If lam_cov=0, main loss is 0!
    self.lam_cov_pred * loss_nmse_pred +
    ... other terms
)
```

**If `self.lam_cov = 0`, the entire main loss term is zero!**

---

### 2. **Loss Function Initialization Issue**
**Probability: 5%**

The loss function is initialized with:
```python
self.loss_fn = UltimateHybridLoss(
    lam_subspace_align=getattr(mdl_cfg, 'LAM_SUBSPACE_ALIGN', 0.05),
    lam_peak_contrast=getattr(mdl_cfg, 'LAM_PEAK_CONTRAST', 0.1),
    lam_cov_pred=getattr(mdl_cfg, 'LAM_COV_PRED', 0.03)
)
```

**CRITICAL: `lam_cov` is NOT passed during initialization!**

This means it uses the default value from `__init__`:
```python
def __init__(self, lam_cov: float = 0.3, ...):
```

**BUT:** If HPO loss weights are applied later, they might set `lam_cov=0`.

---

### 3. **R_blend Not Being Passed**
**Probability: 3%**

If `R_blend` is not in `y_pred`, the loss falls back to constructing `R_hat` from factors.
If factors are all zeros, loss would be zero.

---

### 4. **All Model Outputs are Zero**
**Probability: 1%**

If the model is outputting all zeros, the loss would be zero.
This is unlikely given the model architecture.

---

### 5. **Numerical Instability (NaN → 0)**  
**Probability: 1%**

If gradients are NaN and being sanitized to 0, this could cause zero loss.

---

## 🔧 Diagnostic Steps

### Step 1: Check Loss Weights
```python
# Add this at the beginning of _train_one_epoch in train.py
if epoch == 1:
    print(f"[DEBUG] Loss weights:")
    print(f"  lam_cov: {self.loss_fn.lam_cov}")
    print(f"  lam_cov_pred: {self.loss_fn.lam_cov_pred}")
    print(f"  lam_subspace_align: {self.loss_fn.lam_subspace_align}")
    print(f"  lam_peak_contrast: {self.loss_fn.lam_peak_contrast}")
    print(f"  lam_aux: {self.loss_fn.lam_aux}")
    print(f"  lam_K: {self.loss_fn.lam_K}")
```

### Step 2: Check Loss Components
```python
# Add this in loss.py forward() before return
if not hasattr(self, '_debug_once'):
    print(f"[LOSS] Components:")
    print(f"  loss_nmse: {loss_nmse.item():.6f} * {self.lam_cov} = {(self.lam_cov * loss_nmse).item():.6f}")
    print(f"  loss_nmse_pred: {loss_nmse_pred.item():.6f} * {self.lam_cov_pred}")
    print(f"  loss_ortho: {loss_ortho.item():.6f} * {self.lam_ortho}")
    print(f"  loss_K: {loss_K.item():.6f} * {self.lam_K}")
    print(f"  aux_l2: {aux_l2.item():.6f} * {self.lam_aux}")
    print(f"  TOTAL: {total.item():.6f}")
    self._debug_once = True
```

### Step 3: Check Data Pipeline
```python
# In _train_one_epoch, first batch
if epoch == 1 and bi == 0:
    print(f"[DATA] y: min={y.min():.4f}, max={y.max():.4f}")
    print(f"[DATA] R_in: min={R_in.min():.4f}, max={R_in.max():.4f}")
```

---

## 💡 Most Likely Root Cause

**HYPOTHESIS:** The loss weight schedule we just implemented is setting `lam_subspace_align` and `lam_peak_contrast`, but **NOT setting `lam_cov`** (the critical main loss weight).

**Evidence:**
1. We added `_update_structure_loss_weights()` which only sets:
   - `self.loss_fn.lam_subspace_align`
   - `self.loss_fn.lam_peak_contrast`
   
2. We did NOT set `lam_cov` in the schedule!

3. If `lam_cov = 0` (or very small), the main NMSE loss would be zero.

---

## 🔨 Proposed Fix

### Option 1: Ensure `lam_cov` is set correctly
```python
# In train.py, __init__:
self.loss_fn = UltimateHybridLoss(
    lam_cov=1.0,  # ← ADD THIS! Default to 1.0
    lam_subspace_align=getattr(mdl_cfg, 'LAM_SUBSPACE_ALIGN', 0.05),
    lam_peak_contrast=getattr(mdl_cfg, 'LAM_PEAK_CONTRAST', 0.1),
    lam_cov_pred=getattr(mdl_cfg, 'LAM_COV_PRED', 0.03)
)
```

### Option 2: Check if HPO weights are overwriting `lam_cov`
```python
# In _apply_hpo_loss_weights:
if "lam_cov" in weights:
    self.loss_fn.lam_cov = weights["lam_cov"]
else:
    self.loss_fn.lam_cov = 1.0  # Default if not in HPO weights
```

---

## 🎯 Action Plan

1. **IMMEDIATE:** Add debug prints to verify loss weights
2. **VERIFY:** Check if `lam_cov` is being set to 0
3. **FIX:** Ensure `lam_cov` is always set to a reasonable value (>= 0.1)
4. **TEST:** Re-run overfit test to verify fix

---

## 📊 Expected Behavior After Fix

- **Train loss:** Should start around 0.5-2.0 and decrease
- **Val loss:** Should decrease over epochs (overfit test)
- **Gradients:** Should be non-zero and finite

---

**Next Step:** Run diagnostic script to verify hypothesis



