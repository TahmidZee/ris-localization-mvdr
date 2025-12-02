# 🔍 **DEBUGGING STEPS - Manual Check Required**

## 📅 **Date**: October 23, 2025, 11:25
## 🎯 **Status**: ⏳ **WAITING FOR MANUAL CHECK**

---

## 🔧 **CHANGES MADE**

### **1. Temporarily Disabled Subspace Alignment Loss**
**File**: `ris_pytorch_pipeline/configs.py` (lines 68-70)
```python
# Training-inference alignment (TEMPORARILY DISABLED to isolate NaN issue)
self.LAM_SUBSPACE_ALIGN = 0.0  # DISABLED: Subspace alignment loss weight (was 0.05)
self.LAM_PEAK_CONTRAST = 0.0  # DISABLED: Peak contrast loss weight (was 0.0)
```

### **2. Added Debug Prints**
**File**: `ris_pytorch_pipeline/loss.py` (lines 644-669)
- Added debug prints to show when losses are disabled
- Will show `[LOSS DEBUG] Subspace align: DISABLED (weight=0.0)`
- Will show `[LOSS DEBUG] Peak contrast: DISABLED (weight=0.0)`

---

## 🎯 **WHAT TO CHECK MANUALLY**

### **Run This Command:**
```bash
cd /home/tahit/ris/MainMusic
python -m ris_pytorch_pipeline.hpo --n_trials 1 --epochs_per_trial 1 --space wide 2>&1 | tee debug_run.log
```

### **Look For These Key Indicators:**

#### **✅ Success Indicators:**
1. **Parameter grouping**: Should show `8.3M + 11.0M = 19.4M params`
2. **Loss debug prints**: Should show `DISABLED` messages
3. **Gradient norms**: Should show `backbone=X.XXXe-XX head=X.XXXe-XX ok=True` (NOT NaN)
4. **Training loss**: Should decrease from ~3.7 to ~3.5 or lower
5. **Validation loss**: Should be finite (not `inf`)

#### **❌ Failure Indicators:**
1. **Still NaN gradients**: `backbone=nan head=nan ok=False`
2. **Still scheduler warning**: `lr_scheduler.step() before optimizer.step()`
3. **Still validation crash**: `AssertionError: R_hat does not require gradients`
4. **Still infinite loss**: `Best val loss: inf`

---

## 🔍 **DIAGNOSIS GUIDE**

### **If Gradients Are Still NaN:**
- **Root cause**: The issue is NOT in the subspace alignment loss
- **Next step**: Check other loss components (covariance NMSE, orthogonality, etc.)
- **Likely culprits**: 
  - Division by zero in loss computation
  - Invalid values in R_hat or R_true
  - Numerical instability in other loss terms

### **If Gradients Are Now Finite:**
- **Root cause**: The subspace alignment loss was causing NaN
- **Next step**: Re-enable with better error handling
- **Fix needed**: Add try-catch in subspace loss, check ptr_gt data

### **If Scheduler Warning Persists:**
- **Root cause**: Scheduler step happening before optimizer step somewhere else
- **Next step**: Find where scheduler.step() is called early
- **Fix needed**: Move scheduler step to after optimizer step

### **If Validation Still Crashes:**
- **Root cause**: Assert still failing in eval mode
- **Next step**: Check if assert logic is correct
- **Fix needed**: Verify `torch.is_grad_enabled()` and `self.training` logic

---

## 📊 **EXPECTED OUTPUT**

### **Good Run Should Show:**
```
[LOSS DEBUG] Subspace align: DISABLED (weight=0.0)
[LOSS DEBUG] Peak contrast: DISABLED (weight=0.0)
[GRAD] backbone=X.XXXe-XX head=X.XXXe-XX ok=True
Epoch 001/001 [no-curriculum] train 3.XXXXXX  val 3.XXXXXX
[HPO Trial 0] Training completed! Best val loss: 3.XXXXXX
```

### **Bad Run Will Show:**
```
[LOSS DEBUG] Subspace align: DISABLED (weight=0.0)
[LOSS DEBUG] Peak contrast: DISABLED (weight=0.0)
[GRAD] backbone=nan head=nan ok=False
Epoch 001/001 [no-curriculum] train 3.XXXXXX  val inf
[HPO Trial 0] Training completed! Best val loss: inf
```

---

## 🚀 **NEXT STEPS BASED ON RESULTS**

### **If Success (Gradients Finite):**
1. **Re-enable subspace alignment loss** with better error handling
2. **Add try-catch** around steering vector construction
3. **Check ptr_gt data** format and validity
4. **Run full HPO** with working system

### **If Still Failing (Gradients NaN):**
1. **Check other loss components** for numerical issues
2. **Add debug prints** to each loss term
3. **Check R_hat and R_true** for invalid values
4. **Look for division by zero** in loss computation

### **If Scheduler Warning Persists:**
1. **Search for early scheduler.step()** calls
2. **Check if LambdaLR** needs different handling
3. **Verify step order** in training loop

---

## 📝 **FILES TO CHECK**

### **If You Need to Investigate Further:**
1. **`ris_pytorch_pipeline/loss.py`** - Check other loss components
2. **`ris_pytorch_pipeline/train.py`** - Check training loop and scheduler
3. **`ris_pytorch_pipeline/configs.py`** - Verify all loss weights
4. **`debug_run.log`** - Check the actual output

### **If You Want to Re-enable Subspace Loss:**
1. **Set `LAM_SUBSPACE_ALIGN = 0.05`** in configs.py
2. **Add try-catch** around steering vector construction
3. **Add validation** for ptr_gt data
4. **Test incrementally** with small weights

---

**Please run the test and let me know what you see in the logs!** 🔍




