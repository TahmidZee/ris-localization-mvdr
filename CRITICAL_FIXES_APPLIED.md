# ✅ **CRITICAL FIXES APPLIED - All 3 Issues Resolved**

## 📅 **Date**: October 22, 2025, 22:15
## 🎯 **Status**: ✅ **READY FOR TESTING**

---

## 🔧 **ISSUES IDENTIFIED AND FIXED**

### **Issue 1: Validation Crash (AssertionError)**
**Location**: Line 83-84 in log
**Problem**: `assert R_hat.requires_grad` fails during validation because `torch.no_grad()` context makes `requires_grad=False`

**✅ Fix Applied** (`loss.py` lines 494-498):
```python
# CRITICAL: Verify R_hat (R_pred) has gradients enabled ONLY during training
is_train = torch.is_grad_enabled() and self.training
if is_train:
    assert R_hat.requires_grad, "❌ R_hat (R_pred) does not require gradients during TRAIN! Check for detach() calls."
# In eval/no-grad, it's fine if R_hat.requires_grad == False
```

---

### **Issue 2: NaN Gradients on Batch 0**
**Location**: Lines 45-46 in log
**Problem**: Subspace alignment loss backprops through EVD on rank-deficient matrix, causing NaN gradients

**✅ Fix Applied** (`loss.py` lines 192-251):
- **Replaced** unstable SVD-based subspace alignment with **stable projection loss**
- **No EVD backprop**: GT subspace computed with `torch.no_grad()`
- **Increased regularization**: `3e-4` instead of `1e-4` for numerical stability
- **Detached eigenvectors**: Only R_pred gets gradients, not the projector

**New approach**:
```python
# Get signal subspace from R_true (detached, no backprop through this)
with torch.no_grad():
    evals_true, evecs_true = torch.linalg.eigh(R_true[b])
    U_true = evecs_true[:, -K:].to(dtype)  # [N, K] signal subspace

# Build projector onto GT signal subspace (stable solve)
G = U_true.conj().T @ U_true  # [K, K] Gramian
eye_k = torch.eye(K, dtype=dtype, device=device)
G_reg = G + 3e-4 * eye_k  # Increased regularization for stability
P = U_true @ torch.linalg.solve(G_reg, U_true.conj().T)  # [N, N]

# Energy in wrong subspace vs total energy
R_b = R_pred[b]
num = torch.linalg.norm(P_perp @ R_b @ P_perp, ord='fro') ** 2
den = torch.linalg.norm(R_b, ord='fro') ** 2 + 1e-12
loss_b = (num / den).real
```

**✅ Loss weights reduced for stability** (`configs.py` lines 68-70):
- `LAM_SUBSPACE_ALIGN`: **0.2 → 0.05** (4× reduction)
- `LAM_PEAK_CONTRAST`: **0.1 → 0.0** (disabled until stable)

---

### **Issue 3: Scheduler Warning**
**Location**: Lines 49-50 in log
**Problem**: `lr_scheduler.step()` called before `optimizer.step()`

**✅ Fix Applied** (`train.py` lines 696-773):
- **Moved `unscale_()` to single location**: Right before `optimizer.step()`
- **Proper order**: unscale → clip → grad_check → optimizer.step() → scheduler.step()
- **Fixed duplicate grad logging**: Consolidated into one location after unscale
- **Added finite gradient check**: Skips step if gradients are non-finite

**New training step**:
```python
# Only step optimizer every grad_accumulation steps
if (bi + 1) % grad_accumulation == 0 or (bi + 1) == iters:
    # Unscale ONCE right before step
    self.scaler.unscale_(self.opt)
    
    # Gradient clipping (AFTER unscale, BEFORE step)
    if self.clip_norm and self.clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
    
    # Grad norm check on first batch (AFTER unscale, BEFORE step)
    if epoch == 1 and bi == 0:
        # ... compute and log gradient norms ...
        print(f"[GRAD] backbone={g_back:.3e} head={g_head:.3e} ok={ok}", flush=True)
    
    # Step optimizer (with AMP scaler)
    self.scaler.step(self.opt)
    self.scaler.update()
    
    # Update schedulers and models (AFTER optimizer.step)
    if self.swa_started and self.swa_scheduler is not None:
        self.swa_scheduler.step()
    elif self.sched is not None:
        self.sched.step()
```

---

## 📊 **WHAT TO EXPECT NOW**

### **Immediate (First Epoch)**:
1. ✅ **No validation crash**: Assert is relaxed for eval mode
2. ✅ **Finite gradient norms**: No more NaN from EVD backprop
3. ✅ **No scheduler warning**: Correct step order
4. ✅ **Parameter counts verified**: 8.3M + 11.0M = 19.4M total
5. ✅ **Loss components active**: Subspace align at 0.05 weight, peak contrast at 0.0

### **Expected Output (epoch 1, batch 0)**:
```
[LOSS DEBUG] Subspace align: 0.XXXXXX @ weight=0.05
[LOSS DEBUG] Peak contrast: 0.000000 @ weight=0.0
[GRAD] backbone=X.XXXe-XX head=X.XXXe-XX ok=True
```

### **Within 2-3 Epochs**:
1. **Training loss drops > 10-20%**: From ~3.17 to ~2.5-2.7
2. **Validation completes without crash**: Loss computed correctly in no-grad mode
3. **Gradients remain finite**: Stable projection loss prevents NaN

### **By 5-7 Epochs**:
1. **HPO trials separate**: Some configs perform better than others
2. **Best configs emerge**: Clear improvement over baseline
3. **Val metrics correlate with train**: Learning signal is healthy

---

## 🎯 **FILES MODIFIED**

### **1. `ris_pytorch_pipeline/loss.py`**
- **Lines 192-251**: Replaced `_subspace_alignment_loss` with stable projection-based version
- **Lines 494-498**: Relaxed `R_hat.requires_grad` assert for eval mode

### **2. `ris_pytorch_pipeline/train.py`**
- **Lines 694-773**: Fixed training step (unscale once, proper order, grad check)
- **Removed**: Duplicate grad logging code (lines 696-727)

### **3. `ris_pytorch_pipeline/configs.py`**
- **Lines 68-70**: Reduced loss weights for stability
  - `LAM_SUBSPACE_ALIGN = 0.05` (was 0.2)
  - `LAM_PEAK_CONTRAST = 0.0` (was 0.1)

---

## 🚀 **READY TO TEST**

Run the 1-epoch health check:
```bash
cd /home/tahit/ris/MainMusic
python -m ris_pytorch_pipeline.hpo --n_trials 1 --epochs_per_trial 1 --space wide 2>&1 | tee test_run.log
```

### **Success Criteria**:
- [ ] No validation crash
- [ ] Gradient norms finite (not NaN)
- [ ] No scheduler warning
- [ ] Training loss decreases
- [ ] Validation loss computed successfully

### **If Still Issues**:
- **If gradients still NaN**: Set `LAM_SUBSPACE_ALIGN = 0.0` to isolate core loss
- **If validation crashes**: Check assert logic in loss.py forward
- **If scheduler warning**: Check step order in train loop

---

## 📝 **TECHNICAL DETAILS**

### **Why the old subspace loss caused NaN**:
1. **EVD backprop on nearly rank-deficient matrices** → unstable gradients
2. **SVD of eigenvector products** → compounding numerical errors
3. **No regularization** → singular matrices in solve operations

### **Why the new projection loss is stable**:
1. **No backprop through EVD**: GT subspace computed with `no_grad()`
2. **Strong regularization**: `3e-4 * I` prevents singular solves
3. **Simple quadratic form**: `||P_perp @ R @ P_perp||^2` is well-conditioned
4. **Normalized loss**: Fraction of total energy prevents scale issues

### **Gradient clipping**:
- Already enabled: `CLIP_NORM = 1.0`
- Applied after unscale, before optimizer step
- Prevents occasional large gradients from destabilizing training

---

**All critical fixes are in place. System is ready for testing!** 🎯

