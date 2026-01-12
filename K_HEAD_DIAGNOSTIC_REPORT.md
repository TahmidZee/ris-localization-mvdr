# K-Head Diagnostic Report

## Executive Summary

The K-head was **stuck at chance accuracy (~0.21)** due to **NaN gradients** from an unstable eigendecomposition backward pass. The fix detaches the eigendecomp from the gradient graph, allowing the K-head to learn through alternative gradient paths.

---

## 1. Problem Analysis

### Symptoms Observed (from `run_k_diagnostic.log`)

1. **K_acc stuck at chance level (0.21)** for all 10 epochs
2. **NaN gradients detected** in epoch 1:
   ```
   [GRAD] batch=77 ||g||_2=nan ok=False overflow=False (grads=21, none=72)
   [STEP] batch=77 SKIPPED - overflow=False ||g||_2=nan
   ```
3. **Zero parameter updates** from epoch 3-9:
   ```
   Epoch 2:  Δparam ||·||₂ = 7.151e-01  ← some update
   Epoch 3:  Δparam ||·||₂ = 0.000e+00  ← NOTHING
   ...
   Epoch 9:  Δparam ||·||₂ = 0.000e+00  ← NOTHING
   ```
4. **K_over = 0.78-0.79**: Network always predicts high K (degenerate classifier)

### Root Cause

The K-head extracts spectral features from the covariance matrix eigenvalues:

```python
# In model.py forward()
evals, _ = torch.linalg.eigh(R_eff_k)  # Eigendecomposition
```

**The Problem**: The backward pass of `torch.linalg.eigh` involves terms like:
```
dλᵢ/dR = vᵢvᵢᴴ
dV/dR ∝ 1/(λᵢ - λⱼ)  for i ≠ j
```

When eigenvalues are **close or repeated**, the `1/(λᵢ - λⱼ)` term **explodes to infinity**, producing NaN gradients.

**Why this happens in our model**:
- Our covariance predictor outputs `R_pred = A·Aᴴ` where A is [N, K_MAX]
- This makes R_pred **rank at most K_MAX = 5**
- With N = 144 antennas, we have **139 near-zero eigenvalues** that are essentially identical
- This causes massive numerical instability in the eigendecomp backward

From the diagnostic log:
```
Estimated rank: 5/144 (based on eigenvalue threshold)
Eigenvalues: Σ(top-5)=144.00 (100.0% of total)
❌ RANK-DEFICIENT! (rank ≤ 5)
```

---

## 2. The Fix

### Change Made in `model.py`

```python
# BEFORE (causes NaN):
evals, _ = torch.linalg.eigh(R_eff_k)

# AFTER (stable):
R_eff_k_detached = R_eff_k.detach()  # Stop gradient here
evals, _ = torch.linalg.eigh(R_eff_k_detached)
```

### Why This Works

The K-head has **two gradient paths**:

1. **Spectral path** (now detached):
   - `k_logits_spectral = k_mlp(k_feats)` where `k_feats` comes from eigenvalues
   - k_mlp parameters still get gradients from cross-entropy loss
   - Just no gradient flows back through the eigendecomp

2. **Direct path** (intact):
   - `k_logits_direct = k_direct_mlp(feats_enhanced)`
   - `feats_enhanced` depends on `cov_fact_angle` through the antidiag pooling
   - Gradients flow: `loss → k_direct_mlp → feats_enhanced → antidiag_pool → R_learned → cov_fact_angle`

The **ensemble output** is:
```python
k_logits = 0.5 * k_logits_spectral + 0.5 * k_logits_direct
```

So K-head can still learn through:
- The direct path gradients to `cov_fact_angle`
- The spectral MLP parameters (operating on stable, detached features)

---

## 3. What the Diagnostic Tests

The `run_k_diagnostic.py` script:

| Setting | Value | Purpose |
|---------|-------|---------|
| `TRAIN_PHASE` | `"k_only"` | Freeze backbone, focus on K-head |
| `FREEZE_BACKBONE` | `True` | Stable feature space |
| `lam_K` | `1.5` | K is the dominant loss term |
| `lam_aux` | `0.5` | Some aux supervision |
| `lam_cov` | `0.1` | Just a regularizer |
| `epochs` | `10` | Enough to see learning signal |
| `n_train/n_val` | `5000/1000` | Meaningful sample size |

### Expected Outcome

| Result | Interpretation |
|--------|----------------|
| K_acc rises above 0.3-0.4 | ✅ K-head CAN learn, proceed with 3-phase training |
| K_acc stays at ~0.2 | ❌ Still broken, need further investigation |

---

## 4. Run the Diagnostic

On your screen terminal:

```bash
cd /home/tahit/ris/MainMusic
python run_k_diagnostic.py 2>&1 | tee run_k_diagnostic_v3.log
```

### What to Watch

1. **No more NaN gradients** - should see `[STEP] batch=X STEP TAKEN` instead of `SKIPPED`
2. **Δparam > 0** for every epoch (parameters are actually updating)
3. **K_acc climbing** above 0.25, 0.30, ideally 0.35+
4. **K_over decreasing** from 0.78 toward 0.5 or lower

---

## 5. Next Steps

| Diagnostic Result | Action |
|-------------------|--------|
| K_acc > 0.3 | Proceed with `run_3phase_training.py` |
| K_acc ≈ 0.2-0.25 | Check label indexing (K vs K-1), confusion matrix |
| Still NaN | Something else is broken, need deeper debugging |

---

## 6. Files Changed

| File | Change |
|------|--------|
| `ris_pytorch_pipeline/model.py` | Added `R_eff_k.detach()` before eigendecomp (line ~477) |
| `ris_pytorch_pipeline/configs.py` | Added `lam_gap: 0.0`, `lam_margin: 0.0` to k_only phase |
| `ris_pytorch_pipeline/train.py` | Fixed `_apply_phase_loss_weights` to set lam_gap/lam_margin; Skip HPO overrides for phase-controlled runs |
| `run_k_diagnostic.py` | Unfreeze backbone for diagnostic, AMP off, optional RESUME_CKPT |

---

## 7. Second Root Cause Found (SVD in Loss Function)

After the first fix, NaN gradients still appeared. Investigation revealed:

1. **`_eigengap_hinge()`** uses `torch.linalg.svd()` - same backward instability!
2. **`_subspace_margin_regularizer()`** uses `torch.linalg.svdvals()` - same issue!
3. **`lam_gap = 0.07`** was enabled in the training loop, triggering these SVD calls

### Why lam_gap was 0.07 despite phase config

The epoch loop had this bug:
```python
if hasattr(self, '_hpo_loss_weights'):  # Always True (dict is initialized)
    hpo_lam_cov = self._hpo_loss_weights.get("lam_cov", 1.0)  # Gets default 1.0
    self.loss_fn.lam_gap = 0.065 * hpo_lam_cov  # Sets 0.065 → prints as 0.07
```

### The Fix

1. Added `lam_gap: 0.0` and `lam_margin: 0.0` to k_only phase config
2. Made `_apply_phase_loss_weights()` apply these settings
3. Skip HPO overrides when in phase-controlled training (k_only, geom, joint)

---

## Technical Note: Why Not Just Remove the Spectral Path?

The spectral path provides valuable information:
- Eigengaps, eigenvalue ratios, log-slopes, MDL scores
- These are theory-grounded features for source counting
- The direct path alone might work, but the ensemble is more robust

By detaching rather than removing, we keep the spectral features as **input to k_mlp** (which learns weights on them), we just don't backprop through the eigendecomp itself.

