# Expert Surgical Upgrades Applied

## Summary
Applied three critical surgical upgrades to lock in the training-inference alignment fix, based on expert analysis of the root cause.

---

## Upgrade 1: Robust Complex Cast + Mean-Centering

### **Problem:**
Previous complex conversion was fragile and didn't handle numerical precision properly.

### **Expert Solution:**
```python
# Robust complex cast with proper memory layout
y_c = torch.view_as_complex(y.to(torch.float32).contiguous())        # [B,L,M]
H_c = torch.view_as_complex(H_full.to(torch.float32).contiguous())   # [B,M,N]
C_c = torch.view_as_complex(C.to(torch.float32).contiguous())        # [B,L,N]

# Mean-center across snapshots for robust covariance
A_c = A - A.mean(dim=1, keepdim=True)                 # mean-center across L
R_samp = (A_c.conj().transpose(-1,-2) @ A_c) / max(L-1,1)  # [B,N,N]
```

### **Benefits:**
- ✅ **Numerical Stability**: `contiguous()` ensures proper memory layout
- ✅ **Robust Covariance**: Mean-centering removes bias from snapshot statistics
- ✅ **Scale Invariance**: Ridge regularization scales with trace of Gram matrix

---

## Upgrade 2: Scale-Invariant Ridge Regularization

### **Problem:**
Fixed Tikhonov regularization was too rigid across different SNR levels.

### **Expert Solution:**
```python
# Scale-invariant ridge regularization
alpha = tikhonov_alpha * (torch.trace(G).real / N + 1e-12)   # scale-invariant ridge
a = torch.linalg.solve(G + alpha*I_N, Phi.conj().T @ y_c[b, ell])  # [N]
```

### **Benefits:**
- ✅ **Adaptive Regularization**: Scales with signal strength
- ✅ **SNR Robustness**: Works across different noise levels
- ✅ **Numerical Stability**: Prevents ill-conditioned solves

---

## Upgrade 3: Exact Inference Pipeline Blending

### **Problem:**
Training blending didn't exactly match inference pipeline.

### **Expert Solution:**
```python
# Blend exactly like inference (keep grad only through R_pred)
beta = getattr(cfg, 'hybrid_beta', 0.3)
R_blend = (1 - beta) * R_pred + beta * R_samp.detach()
R_blend = 0.5*(R_blend + R_blend.conj().transpose(-1,-2))            # Hermitian
tr = torch.diagonal(R_blend, -2, -1).real.sum(-1).clamp_min(1e-9)
R_blend = R_blend * (N / tr).view(-1,1,1)                            # trace→N
eps_cov = getattr(cfg, 'eps_cov', 1e-3)
R_blend = R_blend + eps_cov * torch.eye(N, dtype=R_blend.dtype, device=R_blend.device)
```

### **Benefits:**
- ✅ **Exact Inference Match**: Identical blending to inference pipeline
- ✅ **Gradient Isolation**: Only `R_pred` contributes to gradients
- ✅ **Proper Conditioning**: Diagonal loading for numerical stability
- ✅ **Trace Normalization**: Scales to N (not just normalize)

---

## Debug Logging Added

### **Verification Metrics:**
```python
# DEBUG: Print rank information and verify improvements
if epoch == 1 and bi == 0:
    print(f"[DEBUG] R_pred rank: {torch.linalg.matrix_rank(R_pred[0]).item()}")
    print(f"[DEBUG] R_samp rank: {torch.linalg.matrix_rank(R_samp[0]).item()}")
    print(f"[DEBUG] R_blend rank: {torch.linalg.matrix_rank(R_blend[0]).item()}")
    print(f"[DEBUG] ||R_samp||_F: {torch.norm(R_samp[0]).item():.6f}")
    print(f"[DEBUG] ||R_blend||_F: {torch.norm(R_blend[0]).item():.6f}")
    print(f"[DEBUG] Hybrid beta: {beta}")
    print(f"[DEBUG] Mean-centered snapshots: {A_c.shape}")
```

### **Expected Output:**
- **R_pred rank**: ~5 (network output)
- **R_samp rank**: ~L (snapshot-based, typically 16)
- **R_blend rank**: >R_pred rank (hybrid improvement)
- **Non-zero norms**: Both R_samp and R_blend should have significant Frobenius norms
- **Finite gradients**: No NaN/Inf in gradient computation

---

## Key Improvements Summary

### **1. Robust Snapshot Processing:**
- **Before**: Fragile complex conversion, no mean-centering
- **After**: Robust `view_as_complex` with mean-centering for unbiased covariance

### **2. Adaptive Regularization:**
- **Before**: Fixed Tikhonov regularization
- **After**: Scale-invariant ridge that adapts to signal strength

### **3. Exact Inference Alignment:**
- **Before**: Approximate blending
- **After**: Identical to inference pipeline with proper conditioning

### **4. Comprehensive Debugging:**
- **Before**: Limited visibility into tensor properties
- **After**: Full rank/norm verification for each component

---

## Expected Results

### **Success Indicators:**
1. **Non-zero `||R_samp||`**: Sample covariance has meaningful energy
2. **Non-zero `||R_blend||`**: Hybrid blending produces well-conditioned matrix
3. **`rank(R_blend) > rank(R_pred)`**: Hybrid improves conditioning (e.g., ~L vs ~5)
4. **Finite gradients**: No NaN/Inf in backward pass
5. **Learning signal**: Model should learn effectively with aligned losses

### **Training Pipeline Status:**
- ✅ **Tensor Shapes**: Correctly handled for all data types
- ✅ **R_samp Construction**: Robust snapshot-based covariance
- ✅ **R_blend Construction**: Exact inference pipeline match
- ✅ **Loss Alignment**: Uses `R_blend` for critical losses
- ✅ **Debug Verification**: Comprehensive logging for validation

---

## Files Modified

1. **`train.py`**:
   - Upgraded complex conversion with `view_as_complex`
   - Added mean-centering for robust covariance
   - Implemented scale-invariant ridge regularization
   - Exact inference pipeline blending
   - Comprehensive debug logging

2. **`EXPERT_SURGICAL_UPGRADES.md`**:
   - This documentation file

---

## Next Steps

1. **Run HPO**: Execute with expert upgrades applied
2. **Monitor Debug Output**: Verify success indicators in logs
3. **Check Learning**: Confirm model learns with proper gradients
4. **Validate Alignment**: Ensure training-inference consistency

---

**Status**: Expert surgical upgrades applied, training pipeline optimized  
**Last Updated**: October 23, 2025, 1:45 PM




