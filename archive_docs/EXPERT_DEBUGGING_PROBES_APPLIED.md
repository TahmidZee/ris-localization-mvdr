# 🔍 EXPERT SURGICAL DEBUGGING PROBES APPLIED

**Date:** October 27, 2025  
**Status:** ✅ **ALL EXPERT DEBUGGING PROBES IMPLEMENTED**  
**Source:** Expert's surgical debugging checklist

---

## 🎯 **Critical Issues Being Debugged**

The expert identified **two separate symptoms**:
1. **Train loss ≈0.21±0.01 and doesn't move** → parameters aren't effectively changing
2. **Val loss is fixed constant (1.921642)** → validation path computes same number each time

---

## 🔍 **Expert's Surgical Debugging Probes Applied**

### **1. Parameter Drift Probe** ✅ APPLIED

**Location:** `train.py:686-687, 1000-1010`  
**Purpose:** Prove optimizer is (not) moving parameters  

**Implementation:**
```python
# At start of epoch
param_snapshot = {n: p.detach().clone() for n,p in self.model.named_parameters() if p.requires_grad}

# After successful optimizer step
with torch.no_grad():
    import numpy as np
    deltas = []
    for n, p in self.model.named_parameters():
        if p.requires_grad and n in param_snapshot:
            deltas.append((p - param_snapshot[n]).norm().item())
    if deltas:
        print(f"[STEP] Δparam L2: median={np.median(deltas):.3e}  max={np.max(deltas):.3e}", flush=True)
    param_snapshot = {n: p.detach().clone() for n,p in self.model.named_parameters() if p.requires_grad}
```

**Expected Output:**
- If `median/max ≈ 0`: grads are ~0, LR≈0, or never take step
- If `median/max > 0`: parameters are actually changing

---

### **2. Gradient Path Test** ✅ APPLIED

**Location:** `train.py:923-935`  
**Purpose:** Guarantee live gradient path from loss → heads  

**Implementation:**
```python
if epoch == 1 and bi == 0:
    assert 'R_blend' in preds_fp32, "R_blend missing from preds!"
    assert preds_fp32['R_blend'].requires_grad, "R_blend lost grad path!"
    
    # Unit-test the gradient path from R_blend to any head parameter
    try:
        any_head = next(p for n,p in self.model.named_parameters() if "cov_fact_angle" in n and p.requires_grad)
        s = preds_fp32['R_blend'].real.mean()  # cheap scalar depending on R_pred
        g = torch.autograd.grad(s, any_head, retain_graph=True, allow_unused=True)[0]
        print(f"[GRADPATH] d<R_blend>/d(cov_fact_angle) = {0.0 if g is None else g.norm().item():.3e}", flush=True)
    except StopIteration:
        print(f"[GRADPATH] No cov_fact_angle parameter found!", flush=True)
```

**Expected Output:**
- If `d<R_blend>/d(cov_fact_angle) = 0.000`: gradient path is broken
- If `d<R_blend>/d(cov_fact_angle) > 0`: gradient path is live

---

### **3. LR Printing Every Epoch** ✅ APPLIED

**Location:** `train.py:689-691`  
**Purpose:** Verify learning rates are non-zero  

**Implementation:**
```python
# Expert debug: Print LR every epoch
lrs = [g['lr'] for g in self.opt.param_groups]
print(f"[LR] epoch={epoch} groups={['backbone','head']} lr={lrs}", flush=True)
```

**Expected Output:**
- `[LR] epoch=1 groups=['backbone','head'] lr=[0.0002, 0.0008]` (non-zero)
- If `lr=[0.0, 0.0]`: scheduler killed learning rates

---

### **4. Radians vs Degrees Bug Fix** ✅ APPLIED

**Location:** `loss.py:259-262`  
**Issue:** `ptr_gt` is already in radians, but code was converting degrees→radians again  
**Impact:** Steering vectors nearly constant → `valid_batches==0` → subspace loss = 0.0  

**Fix Applied:**
```python
# BEFORE (WRONG):
phi_deg = ptr_gt[b, k, 0].item()
theta_deg = ptr_gt[b, k, 1].item()
phi_rad = np.deg2rad(phi_deg)      # DOUBLE CONVERSION!
theta_rad = np.deg2rad(theta_deg)  # DOUBLE CONVERSION!

# AFTER (CORRECT):
phi_rad = float(ptr_gt[b, k, 0].item())    # Already radians!
theta_rad = float(ptr_gt[b, k, 1].item())  # Already radians!
```

**Expected Result:**
- Subspace alignment loss becomes **non-zero**
- Peak contrast loss becomes **non-zero**

---

### **5. Validation Loss Printing** ✅ APPLIED

**Location:** `train.py:1124-1126`  
**Purpose:** Verify validation loss changes per epoch (not constant 1.921642)  

**Implementation:**
```python
# Expert debug: Print validation loss for first batch each epoch
if bi == 0:
    print(f"[VAL] loss(first-batch)={float(loss):.6f}", flush=True)
```

**Expected Output:**
- `[VAL] loss(first-batch)=0.921642` (changes each epoch)
- If constant: validation path has bug

---

### **6. Grad_ok Logging Every Step** ✅ APPLIED

**Location:** `train.py:990-991`  
**Purpose:** Track when steps are skipped due to bad gradients  

**Implementation:**
```python
# Expert debug: Log grad_ok every step
print(f"[GRAD] step_ok={ok} g_back={g_back:.2e} g_head={g_head:.2e}", flush=True)
```

**Expected Output:**
- `[GRAD] step_ok=True g_back=1.23e-02 g_head=2.45e-02` (steps taken)
- `[GRAD] step_ok=False g_back=0.00e+00 g_head=0.00e+00` (steps skipped)

---

## 📊 **Expected Results After Probes**

### **Before Probes:**
```
[LOSS DEBUG] Subspace align: 0.000000 @ weight=0.02  ← Broken (radians bug)
[LOSS DEBUG] Peak contrast: 0.000000 @ weight=0.05   ← Broken (radians bug)

Epoch 001/030: train 0.217  val 1.921642  ← No learning
Epoch 030/030: train 0.218  val 1.921642  ← Still no learning
```

### **After Probes (Expected):**
```
[LR] epoch=1 groups=['backbone','head'] lr=[0.0002, 0.0008]  ← Non-zero LR ✅
[GRADPATH] d<R_blend>/d(cov_fact_angle) = 1.234e-02  ← Live gradient path ✅
[GRAD] step_ok=True g_back=1.23e-02 g_head=2.45e-02  ← Steps taken ✅
[STEP] Δparam L2: median=1.234e-04  max=2.456e-04  ← Parameters changing ✅
[VAL] loss(first-batch)=0.921642  ← Changes each epoch ✅

[LOSS DEBUG] Subspace align: 0.042 @ weight=0.02  ← Working! ✅
[LOSS DEBUG] Peak contrast: 0.031 @ weight=0.05   ← Working! ✅

Epoch 001/030: train 0.806  val 0.921  ← LEARNING! ✅
Epoch 010/030: train 0.412  val 0.487  ← DECREASING! ✅
Epoch 030/030: train 0.198  val 0.187  ← CONVERGED! ✅
```

---

## 🎯 **Summary of All Fixes**

| Issue | Location | Status |
|-------|----------|--------|
| **Parameter drift probe** | `train.py:686-687, 1000-1010` | ✅ Applied |
| **Gradient path test** | `train.py:923-935` | ✅ Applied |
| **LR printing** | `train.py:689-691` | ✅ Applied |
| **Radians bug fix** | `loss.py:259-262` | ✅ Applied |
| **Val loss printing** | `train.py:1124-1126` | ✅ Applied |
| **Grad_ok logging** | `train.py:990-991` | ✅ Applied |
| **Eigengap tiny floor** | `loss.py:430` | ✅ Applied |
| **Val loss FP32** | `train.py:1053-1091` | ✅ Applied |

---

## 🚀 **Ready for Testing**

**Run:**
```bash
cd /home/tahit/ris/MainMusic
python test_overfit.py 2>&1 | tee overfit_test_debugging_probes.log
```

**Look For:**
1. ✅ `[LR]` shows non-zero learning rates
2. ✅ `[GRADPATH]` shows non-zero gradient path
3. ✅ `[GRAD] step_ok=True` shows steps being taken
4. ✅ `[STEP] Δparam L2` shows non-zero parameter changes
5. ✅ `[VAL]` shows changing validation loss
6. ✅ Subspace align & peak contrast **non-zero**
7. ✅ Training loss **decreasing**
8. ✅ Validation loss **decreasing**

---

**All expert debugging probes applied! This will pinpoint exactly why the model isn't learning!** 🔍🎯



