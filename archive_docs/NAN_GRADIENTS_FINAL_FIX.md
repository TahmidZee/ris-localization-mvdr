# NaN Gradients - Final Fix

**Date**: October 23, 2025, 16:10  
**Issue**: After fixing R_samp/R_blend normalization, loss values are now reasonable (~2696) but gradients are still NaN

---

## **Progress from Previous Fixes** ✅

| Metric | Before | After Fix #1-2 | Status |
|--------|--------|----------------|--------|
| R_samp norm | 0.000285 | 41.03 | ✅ **FIXED** |
| R_blend norm | 3.46 × 10^12 | 22.96 | ✅ **FIXED** |
| Loss | 1.7 × 10^25 | 2695.97 | ✅ **FIXED** |
| Gradients | NaN | NaN | ❌ **STILL BROKEN** |

---

## **Root Cause #3: Eigendecomposition Backward Pass**

### **Problem**:
The `_eigengap_hinge` loss function (line 388-409 in `loss.py`) uses `torch.linalg.eigh()` on `R_blend` to compute eigenvalues and encourage a gap at K.

```python
def _eigengap_hinge(self, R_hat_c: torch.Tensor, K_true: torch.Tensor):
    eps_psd = getattr(mdl_cfg, 'EPS_PSD', 1e-4)
    R_stable = R_hat_c + eps_psd * torch.eye(...)
    
    evals, _ = torch.linalg.eigh(R_stable)     # ← NaN in backward pass!
    # ... compute hinge loss on eigenvalue gap ...
```

### **Why This Causes NaN Gradients**:

1. **Complex Hermitian Eigendecomposition**: `torch.linalg.eigh()` on complex 144×144 matrices is numerically sensitive in the backward pass

2. **Full-Rank Matrix**: After adding diagonal loading (`C_EPS=1.0`), `R_blend` is full rank (144/144), making the eigendecomposition more challenging

3. **Repeated/Close Eigenvalues**: With diagonal loading, many eigenvalues become close or repeated, causing numerical instability in the gradient computation of eigenvectors

4. **PyTorch Implementation**: The backward pass of `eigh()` for complex matrices involves solving systems with eigenvalue differences in denominators → division by near-zero when eigenvalues are close → NaN

### **Evidence from Log**:
- Line 65: `[DEBUG] R_blend rank: 144` ← Full rank after diagonal loading
- Line 66: `||R_samp||_F: 41.028347` ← Correct
- Line 67: `||R_blend||_F: 22.955914` ← Correct
- Line 72: `Loss is finite: 2695.967041` ← Finite loss!
- Line 73: `[GRAD] backbone=nan head=nan ok=False` ← **NaN gradients**

The loss is **finite and reasonable**, but gradients are NaN → points to numerical issue in backward pass, not forward pass.

---

## **Fix #3: Disable Eigengap Loss** ✅

**Location**: `ris_pytorch_pipeline/loss.py`, line 81

**Change**:
```python
# Before
lam_gap: float   = 0.012,  # RE-ENABLED: Now uses well-conditioned blended covariance

# After  
lam_gap: float   = 0.0,  # TEMPORARILY DISABLED: eigh() backward causing NaN gradients
```

### **Rationale**:
- The eigengap loss is a **nice-to-have** regularizer, not critical for training
- Other loss terms (NMSE, K classification, auxiliary losses) are sufficient for training
- We can re-enable it later with a more stable implementation (e.g., using stop-gradient on eigenvectors)

### **Alternative Solutions** (for later):
1. **Stop-gradient on eigenvectors**: Only use eigenvalues in loss, detach eigenvectors
2. **Use SVD instead of eigh**: `torch.svd()` might have more stable backward pass
3. **Reduce diagonal loading**: Lower `C_EPS` from 1.0 to 0.01 to reduce rank
4. **Use finite differences**: Compute eigenvalue gap without gradients through eigh

---

## **Expected Results After Fix #3**:

| Metric | Current | Expected |
|--------|---------|----------|
| R_samp norm | 41.03 ✅ | ~30-60 ✅ |
| R_blend norm | 22.96 ✅ | ~20-80 ✅ |
| Loss | 2695.97 ✅ | ~1000-3000 ✅ |
| Gradients | **NaN** ❌ | **Finite** ✅ |
| Model Learning | No ❌ | **Yes** ✅ |

---

## **Summary of All Fixes**

### **Fix #1**: Add R_samp trace normalization (train.py, lines 846-849)
- **Problem**: R_samp 10^5 times too small
- **Solution**: Hermitize + trace normalize to N=144
- **Impact**: R_samp norm 0.000285 → 41.03 ✅

### **Fix #2**: Remove R_blend trace normalization (train.py, lines 872-875)
- **Problem**: R_blend 10^12 times too large
- **Solution**: Only add diagonal loading, no trace normalization
- **Impact**: R_blend norm 3.46 × 10^12 → 22.96, Loss 1.7 × 10^25 → 2696 ✅

### **Fix #3**: Disable eigengap loss (loss.py, line 81)
- **Problem**: `torch.linalg.eigh()` backward pass produces NaN gradients
- **Solution**: Set `lam_gap=0.0` to disable the loss term
- **Impact**: Gradients NaN → Finite (expected) ✅

---

## **Files Modified**:
1. `ris_pytorch_pipeline/train.py`: Lines 846-849, 867-875
2. `ris_pytorch_pipeline/loss.py`: Line 81

---

## **Status**: ✅ **READY TO TEST**

Run HPO again and verify:
1. ✅ Loss stays reasonable (~1000-3000)
2. ✅ Gradients are finite
3. ✅ Training loss decreases over epochs
4. ✅ Model actually learns (weights change)

If gradients are still NaN, check:
- Other eigendecomposition calls in the loss function
- Division by zero in auxiliary losses
- sqrt/log of negative numbers




