# Training-Inference Alignment Fixes

## Problem Summary
The model was not learning during HPO due to architectural misalignment between training and inference:

- **Training**: Loss functions operated on raw `R_hat` (rank-deficient, ~rank 5)
- **Inference**: Uses hybrid-blended `R_blend` (well-conditioned, ~rank 20)
- **Result**: Loss gradients computed on wrong representation, no learning signal

## Critical Fixes Applied

### 1. R_blend Construction in Training Loop (`train.py`)
```python
# Construct R_pred (rank-deficient)
R_pred = (A_angle @ A_angle.conj().transpose(-2, -1)) + 0.3 * (A_range @ A_range.conj().transpose(-2, -1))
R_pred = 0.5 * (R_pred + R_pred.conj().transpose(-2, -1))

# Trace normalize to N (same as inference)
tr_pred = torch.diagonal(R_pred, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-9)
scale_pred = (R_pred.new_tensor(N) / tr_pred).view(-1, 1, 1)
R_pred = R_pred * scale_pred

# Construct R_samp from snapshots (NEVER use R_true!)
# ... LS solve with H_full, y, C ...

# Hybrid blending (same as inference pipeline)
beta = 0.3
R_blend = (1.0 - beta) * R_pred + beta * R_samp.detach()

# Hermitize and trace normalize to N (same as inference)
R_blend = 0.5 * (R_blend + R_blend.conj().transpose(-2, -1))
tr_blend = torch.diagonal(R_blend, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-9)
scale = (R_blend.new_tensor(N) / tr_blend).view(-1, 1, 1)
R_blend = R_blend * scale

# Add to preds for loss function
preds_fp32['R_blend'] = R_blend
```

### 2. Loss Function Updates (`loss.py`)
Modified critical loss functions to use `R_blend` when available:

```python
# Eigengap loss
if 'R_blend' in y_pred and self.lam_gap > 0.0:
    loss_gap = self._eigengap_hinge(y_pred['R_blend'], K_true)
else:
    loss_gap = torch.tensor(0.0, device=device)

# Covariance NMSE loss
if 'R_blend' in y_pred:
    loss_nmse = self._nmse_cov(y_pred['R_blend'], R_true).mean()
else:
    loss_nmse = self._nmse_cov(R_hat, R_true).mean()

# Subspace alignment loss
if self.lam_subspace_align > 0.0:
    if 'R_blend' in y_pred:
        loss_subspace_align = self._subspace_alignment_loss(y_pred['R_blend'], R_true, K_true, ptr_gt)
    else:
        loss_subspace_align = self._subspace_alignment_loss(R_hat, R_true, K_true, ptr_gt)
```

### 3. Trace Normalization Fix
**Critical**: Both `R_pred` and `R_blend` now scale to N (not just divide by trace):

```python
# OLD (wrong): R = R / tr
# NEW (correct): R = R * (N / tr)
scale = (R.new_tensor(N) / tr).view(-1, 1, 1)
R = R * scale
```

This matches the inference pipeline exactly: `R = R * (N / trace_R)`

### 4. Snapshot-based R_samp Construction
Uses actual snapshots via LS solve, not `R_true` fallback:

```python
# Solve for complex amplitudes: Y = H @ A + noise
# A = (H^H @ H + αI)^(-1) @ H^H @ Y
H_H = H_full.conj().transpose(-2, -1)
HtH = H_H @ H_full
HtH_reg = HtH + tikhonov_alpha * torch.eye(N, device=device, dtype=torch.complex64)
A_ls = torch.linalg.solve(HtH_reg, H_H @ y)

# Build sample covariance
R_samp = (A_ls @ A_ls.conj().transpose(-2, -1)) / L
```

## Verification Checklist

✅ **R_blend Construction**: Training now constructs `R_blend` exactly like inference  
✅ **Loss Function Alignment**: Critical losses use `R_blend` when available  
✅ **Trace Normalization**: Both `R_pred` and `R_blend` scale to N (not just normalize)  
✅ **Snapshot-based R_samp**: Uses actual snapshots, not `R_true` fallback  
✅ **Gradient Preservation**: `R_pred` stays in computational graph  
✅ **No Detachment**: Loss functions don't detach `R_hat` unnecessarily  

## Expected Results

1. **Learning Signal**: Training losses now operate on same well-conditioned `R_blend` used in inference
2. **No NaN Gradients**: Eigendecomposition on well-conditioned `R_blend` (rank ~20) instead of rank-deficient `R_hat` (rank ~5)
3. **Proper Learning**: Model should now learn effectively during HPO
4. **Training-Inference Consistency**: Loss gradients computed on same representation as inference

## Files Modified

- `train.py`: Added `R_blend` construction in training loop
- `loss.py`: Updated loss functions to use `R_blend` when available
- `COMPLETE_DEBUGGING_REPORT.md`: Documented the architectural alignment fix

---

**Status**: Training-inference alignment implemented - model should now learn properly  
**Last Updated**: October 23, 2025, 1:00 PM




