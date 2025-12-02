# 🎯 Expert-Recommended Fixes Implementation

**Date:** October 27, 2025  
**Status:** 🔧 **IMPLEMENTING EXPERT FIXES**

---

## 🐛 **Critical Bug Found by Expert**

### **Issue:** `running += loss` Inside Gradient Accumulation Block
**Location:** `train.py:991`  
**Problem:** Loss accumulation was inside the `if (bi + 1) % grad_accumulation == 0` block  
**Impact:** Loss only accumulated when optimizer step was taken  
**Fix:** Moved `running += loss` outside the gradient accumulation block (indent 8 instead of 12)  
**Status:** ✅ **FIXED**

---

## 📋 **Expert's Recommendations**

### **1. Gradient Flow Checks** ✅ TODO

Add assertions and checks:
```python
# Right after loss computation
assert loss.requires_grad, "Loss has no grad path"
assert torch.isfinite(loss), "Loss is NaN/Inf"

# After backward(), before step()
no_grad = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
print("[GRAD] params with no grad:", no_grad[:20])
```

### **2. Step Counter** ✅ TODO

Add tracking to verify optimizer steps:
```python
self._steps_taken = 0
self._batches_seen = 0

# Inside batch loop
self._batches_seen += 1

# When stepping
self._steps_taken += 1
if (epoch == 1 and bi < 3):
    print(f"[OPT] step {self._steps_taken} / batch {self._batches_seen} "
          f"LRs={[g['lr'] for g in self.opt.param_groups]}")
```

### **3. R_blend Gradient Flow** ✅ TODO

Ensure gradients flow through R_pred:
```python
# CORRECT: Only detach R_samp
R_blend = (1 - beta) * R_pred + beta * R_samp.detach()

# WRONG: Don't detach R_pred!
# R_blend = (1 - beta) * R_pred.detach() + beta * R_samp
```

### **4. Hermitian + Jitter Before EVD** ✅ TODO

Add numerical stability:
```python
# Before any eigendecomposition
R = 0.5*(R + R.conj().transpose(-2, -1)) + 1e-6*torch.eye(N, device=R.device)
```

### **5. Loss in FP32 with Eps** ✅ TODO

Ensure NMSE has epsilon:
```python
nmse = ||R̂−R||² / (||R||² + 1e-8)
```

---

## 🔧 **Fixes to Implement**

### **Priority 1: Critical (Blocking Training)**
- [x] Move `running += loss` outside gradient accumulation block
- [ ] Add step counter and logging
- [ ] Verify R_blend doesn't detach R_pred
- [ ] Add gradient flow checks

### **Priority 2: Important (Numerical Stability)**
- [ ] Add Hermitian + jitter before EVD
- [ ] Ensure loss NMSE uses eps
- [ ] Add finite checks for inputs

### **Priority 3: Nice-to-Have (Debugging)**
- [ ] Add anomaly detection for one batch
- [ ] Add parameter update verification
- [ ] Add gradient coverage check

---

## 📊 **Expected Results After Fixes**

### **Before Fixes:**
```
Epoch 001/030: train 0.216999  val 1.921642  ← NOT LEARNING
Epoch 030/030: train 0.217593  val 1.921642  ← STUCK!
```

### **After Critical Fix (running += loss):**
```
[OPT] step 1 / batch 1 LRs=[2e-04, 8e-04]
[GRAD] backbone=1.234e-02 head=2.456e-02 ok=True

Epoch 001/030: train 0.806  val 1.921  ← SHOULD START LEARNING!
Epoch 002/030: train 0.723  val 1.789  ← SHOULD DECREASE!
Epoch 030/030: train 0.198  val 0.187  ← SHOULD CONVERGE!
```

---

## 🎯 **Next Steps**

1. ✅ **DONE:** Fix `running += loss` indentation
2. **TODO:** Add step counter and gradient logging
3. **TODO:** Verify R_blend gradient flow
4. **TODO:** Run overfit test with new fixes
5. **TODO:** If still stuck, add anomaly detection

---

**Critical fix implemented! Now need to add debugging instrumentation to verify training works.** 🎯



