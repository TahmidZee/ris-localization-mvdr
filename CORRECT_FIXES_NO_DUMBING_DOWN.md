# ✅ **CORRECT FIXES APPLIED - No Dumbing Down, Physics-Aligned**

## 📅 **Date**: October 22, 2025, 22:30
## 🎯 **Status**: ✅ **PHYSICS-ALIGNED, SOTA-READY**

---

## 🔑 **KEY PRINCIPLE: MAINTAIN SOTA TARGET**

### **What We Did Right**:
- ✅ **Stable projection loss** (no EVD backprop on learned matrix)
- ✅ **GT steering vectors** from angles/ranges (physics-aligned)
- ✅ **Gradients flow to R_pred** (learning signal preserved)
- ✅ **Same steering as inference** (training-inference alignment)
- ✅ **Conservative HPO weights, stronger for final train**

### **What We AVOIDED (Good!)**:
- ❌ **NO dumbing down** of the objective
- ❌ **NO detaching R_pred** (would kill learning)
- ❌ **NO removing alignment losses** (critical for SOTA)
- ❌ **NO circular reasoning** (R_true eigenvectors → wrong!)

---

## 🔧 **CRITICAL FIX: Use GT Steering Vectors, Not R_true Eigenvectors**

### **❌ What Was Wrong (First Attempt)**:
```python
# WRONG: Using R_true eigenvectors as GT subspace
with torch.no_grad():
    evals_true, evecs_true = torch.linalg.eigh(R_true[b])
    U_true = evecs_true[:, -K:]  # ← Circular reasoning!
```

**Problem**: R_true's eigenvectors are noisy and don't match the physical steering manifold. This is circular reasoning that doesn't align training with inference.

### **✅ What's Correct Now**:
```python
# CORRECT: Build A_gt from GT angles/ranges using canonical steering
for k in range(K):
    phi_deg = ptr_gt[b, k, 0].item()
    theta_deg = ptr_gt[b, k, 1].item()
    r_m = ptr_gt[b, k, 2].item()
    
    # Convert to radians
    phi_rad = np.deg2rad(phi_deg)
    theta_rad = np.deg2rad(theta_deg)
    
    # Build near-field steering vector (SAME AS INFERENCE)
    sin_phi = np.sin(phi_rad)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    # Phase (curvature term with correct sign)
    dist = r_m - x_np * sin_phi * cos_theta - y_np * sin_theta + (x_np**2 + y_np**2) / (2.0 * r_m)
    phase = k0 * (r_m - dist)
    
    # Steering vector (unit-norm)
    a = np.exp(1j * phase)
    a = a / np.sqrt(np.sum(np.abs(a)**2))
    
    A_cols.append(torch.from_numpy(a).to(device, dtype=dtype))

A_gt = torch.stack(A_cols, dim=1)  # [N, K] GT signal subspace
```

**Why This is Correct**:
1. **Physics-aligned**: Uses the EXACT steering formula as inference (MUSIC/Newton)
2. **Training-inference match**: Network learns to produce R that MUSIC will use correctly
3. **No circular reasoning**: GT subspace is independent of learned covariance
4. **Stable**: No grad through GT steering (detached), only through R_pred
5. **Near-field aware**: Includes range and curvature (critical for 0.5-10m)

---

## 🎯 **THE THREE CRITICAL ISSUES AND CORRECT FIXES**

### **Issue 1: Validation Crash**
**Problem**: `assert R_hat.requires_grad` failed during validation (no-grad context)

**✅ Correct Fix** (`loss.py` lines 494-498):
```python
# Relax assert for eval mode ONLY
is_train = torch.is_grad_enabled() and self.training
if is_train:
    assert R_hat.requires_grad, "R_hat lost gradients during TRAIN!"
# In eval/no-grad, it's fine if R_hat.requires_grad == False
```

**Why This Maintains SOTA**:
- Learning signal preserved in training
- No dumbing down of the objective
- Just allows validation to run (which doesn't need gradients anyway)

---

### **Issue 2: NaN Gradients**
**Problem**: EVD backprop on learned matrix causes NaN at low SNR/small eigengaps

**✅ Correct Fix** (`loss.py` lines 192-297):

**Key Changes**:
1. **GT subspace from steering vectors** (not R_true eigenvectors)
2. **No grad through GT subspace** (stable, physics-aligned)
3. **Gradients only through R_pred** (preserves learning signal)
4. **Stable projection**: `||P_perp @ R_pred @ P_perp||^2 / ||R_pred||^2`

**Mathematical Form**:
```
L_subspace = ||P_perp @ R_pred @ P_perp||_F^2 / ||R_pred||_F^2

where:
- A_gt = [a(φ₁,θ₁,r₁), ..., a(φₖ,θₖ,rₖ)]  ← GT steering vectors
- P = A_gt @ (A_gt^H @ A_gt + εI)^{-1} @ A_gt^H  ← projector onto GT subspace  
- P_perp = I - P  ← orthogonal projector
```

**Why This Maintains SOTA**:
- **Same physical objective**: Penalize energy outside GT signal subspace
- **Better optimization**: Stable gradients → better convergence
- **Training-inference alignment**: Uses EXACT steering as MUSIC
- **No dumbing down**: Physics-based objective preserved
- **Actually improves SOTA potential**: More stable → reaches better optimum

---

### **Issue 3: Scheduler Warning + AMP Issues**
**Problem**: `lr_scheduler.step()` before `optimizer.step()` + duplicate `unscale_()`

**✅ Correct Fix** (`train.py` lines 696-773):

**Proper Order**:
```python
if do_step:
    # 1. Unscale ONCE
    scaler.unscale_(optimizer)
    
    # 2. Gradient clipping (AFTER unscale)
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # 3. Grad norm check (AFTER unscale, for logging)
    if epoch == 1 and bi == 0:
        g_back = group_norm(backbone_params)
        g_head = group_norm(head_params)
        print(f"[GRAD] backbone={g_back:.3e} head={g_head:.3e} ok={ok}")
    
    # 4. Optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    # 5. Scheduler step (AFTER optimizer.step)
    if scheduler:
        scheduler.step()
```

**Why This Maintains SOTA**:
- Pure engineering hygiene (no effect on optimum)
- Prevents NaN propagation (skips step if gradients non-finite)
- Correct AMP protocol (stable training)
- No change to objective or learning signal

---

## 📊 **LOSS WEIGHTS STRATEGY**

### **HPO Phase (Stabilize)**:
```python
LAM_SUBSPACE_ALIGN = 0.05  # Conservative for stability
LAM_PEAK_CONTRAST = 0.0    # Disabled until stable
CLIP_NORM = 1.0            # Prevent gradient explosion
```

### **Final Train Phase (SOTA)**:
```python
LAM_SUBSPACE_ALIGN = 0.10 - 0.20  # Stronger alignment
LAM_PEAK_CONTRAST = 0.05 - 0.10   # Enable peak sharpening
CLIP_NORM = 1.0                   # Keep for safety
```

**Why This Strategy**:
- **HPO**: Find hyperparameters with stable training
- **Final**: Push performance with stronger alignment losses
- **Progressive**: Build confidence before full strength
- **No compromise**: Eventually use full physics-aligned objective

---

## 🔬 **PHYSICS-ALIGNMENT VERIFICATION**

### **Steering Vector Convention (Exact Match)**:
```python
# Phase formula (same as physics.py and inference):
dist = r - x*sin(φ)*cos(θ) - y*sin(θ) + (x²+y²)/(2r)
phase = k₀*(r - dist)
a = exp(1j*phase) / sqrt(N)

# Convention:
# - x-axis: sin(φ)*cos(θ)  ← azimuth + elevation coupling
# - y-axis: sin(θ)         ← elevation only
# - Curvature: +(x²+y²)/(2r)  ← near-field term (positive!)
```

### **Key Properties Preserved**:
1. **Unit-norm**: `||a|| = 1` (prevents scale ambiguity)
2. **Near-field**: Range-dependent curvature included
3. **Centered**: Sensor array centered at origin
4. **Wavelengths**: Spacing in wavelengths for phase calculation

---

## 💡 **WHY THIS APPROACH IS BETTER THAN ALTERNATIVES**

### **vs. Using R_true Eigenvectors**:
- ❌ R_true eigenvectors: Noisy, circular reasoning, no physics alignment
- ✅ GT steering vectors: Clean, physics-based, training-inference match

### **vs. Backprop Through EVD**:
- ❌ EVD backprop: Unstable gradients, NaN at low SNR
- ✅ Stable projection: Same objective, better optimization

### **vs. Removing Alignment Loss**:
- ❌ No alignment: Network doesn't know about classical backend
- ✅ With alignment: Direct optimization for MUSIC performance

### **vs. Detaching R_pred**:
- ❌ Detach R_pred: Kills learning signal completely
- ✅ Keep gradient: Network actually learns from alignment loss

---

## 🎯 **EXPECTED IMPROVEMENTS**

### **Immediate (Epoch 1)**:
- ✅ No validation crash
- ✅ Finite gradients (not NaN)
- ✅ No scheduler warning
- ✅ **Subspace loss actually working** (using GT steering)

### **Early Training (Epochs 1-10)**:
- ✅ Training loss decreases steadily
- ✅ Validation loss tracks training
- ✅ Gradients remain healthy
- ✅ **Network learns MUSIC-aligned covariances**

### **HPO (50 trials × 12 epochs)**:
- ✅ Trials separate (good vs bad configs)
- ✅ Best configs emerge
- ✅ **β, shrink_alpha, NF-MLE params optimized**

### **Final Train (80-120 epochs)**:
- ✅ Sub-degree @ mid/high SNR (≥5 dB)
- ✅ Near-degree @ low SNR (0-5 dB)
- ✅ **SOTA performance** (physics-aligned objective working)

---

## 📁 **FILES MODIFIED (Physics-Aligned Version)**

### **1. `ris_pytorch_pipeline/loss.py`**
- **Lines 192-297**: `_subspace_alignment_loss` - Now uses GT steering vectors
  - Builds `A_gt` from `ptr_gt` (phi, theta, r)
  - Uses canonical near-field steering (same as inference)
  - Stable projection loss (no EVD backprop on learned matrix)
  - Gradients only through R_pred (learning signal preserved)

- **Lines 494-498**: Relaxed `R_hat.requires_grad` assert for eval mode

- **Lines 642, 767**: Updated calls to pass `ptr_gt` parameter

### **2. `ris_pytorch_pipeline/train.py`**
- **Lines 696-773**: Fixed training step (unscale once, proper order)
  - Single unscale location (before optimizer.step)
  - Gradient clipping after unscale
  - Grad norm logging after unscale
  - Scheduler step after optimizer.step

### **3. `ris_pytorch_pipeline/configs.py`**
- **Lines 68-70**: Conservative HPO weights
  - `LAM_SUBSPACE_ALIGN = 0.05` (will increase to 0.10-0.20 for final)
  - `LAM_PEAK_CONTRAST = 0.0` (will enable 0.05-0.10 for final)

---

## 🚀 **READY FOR TESTING**

```bash
cd /home/tahit/ris/MainMusic
python -m ris_pytorch_pipeline.hpo --n_trials 1 --epochs_per_trial 1 --space wide 2>&1 | tee test_run.log
```

### **Success Criteria**:
- [ ] No validation crash
- [ ] Gradient norms finite (not NaN)  
- [ ] No scheduler warning
- [ ] **Subspace align loss computed from GT steering** (check log)
- [ ] Training loss decreases

### **If Success, Run Full HPO**:
```bash
python -m ris_pytorch_pipeline.hpo --n_trials 50 --epochs_per_trial 12 --space wide 2>&1 | tee hpo_run.log
```

---

## 📝 **SUMMARY: NO DUMBING DOWN**

### **What We Did**:
1. **Stable math** (projection instead of EVD backprop)
2. **Physics-aligned** (GT steering vectors from canonical formula)
3. **Training-inference match** (exact steering as MUSIC uses)
4. **Preserved learning signal** (gradients through R_pred)
5. **Engineering hygiene** (AMP, clipping, step order)

### **What We Did NOT Do**:
1. ❌ Remove alignment losses
2. ❌ Detach R_pred (kill learning)
3. ❌ Use noisy eigenvectors instead of physics
4. ❌ Simplify objective
5. ❌ Compromise SOTA target

**Bottom Line**: Same physics-based objective, better optimization, SOTA target maintained! 🎯

