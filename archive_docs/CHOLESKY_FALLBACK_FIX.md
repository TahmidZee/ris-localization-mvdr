# Cholesky Fallback Fix for R_samp Construction

## Problem
The Cholesky decomposition in the R_samp construction was failing with:
```
torch._C._LinAlgError: linalg.cholesky: (Batch element 408): The factorization could not be completed because the input is not positive-definite (the leading minor of order 142 is not positive-definite).
```

This occurred because the regularized Gram matrix `G_reg = G + alpha * I_N` was still not positive definite for some batches.

## Root Cause
- The Tikhonov regularization `alpha = 1e-6` was too small for ill-conditioned matrices
- Some batches had Gram matrices that were still not positive definite even after regularization
- No fallback mechanism existed for when Cholesky fails

## Solution Applied

### 1. Increased Tikhonov Regularization
```python
alpha_base = getattr(cfg, 'NF_MLE_TIKHONOV_LAMBDA', 1e-3)  # Increased from 1e-6
```

### 2. Added SVD Fallback
```python
try:
    L_chol = torch.linalg.cholesky(G_reg)
    a = torch.cholesky_solve(rhs.unsqueeze(-1), L_chol).squeeze(-1)
except torch._C._LinAlgError:
    # Fallback to SVD when Cholesky fails (not positive definite)
    U, S, Vh = torch.linalg.svd(G_reg, full_matrices=False)
    S_reg = torch.clamp(S, min=1e-8)
    S_inv = 1.0 / S_reg
    G_inv = Vh.conj().transpose(-1, -2) @ (S_inv.unsqueeze(-1) * U.conj().transpose(-1, -2))
    a = torch.matmul(G_inv, rhs.unsqueeze(-1)).squeeze(-1)
```

### 3. Key Features of the Fix
- **Robust**: Handles both well-conditioned and ill-conditioned matrices
- **Numerically Stable**: Uses SVD with proper regularization (clamp to 1e-8)
- **Complex-Safe**: Properly handles complex matrices with conjugate transposes
- **Efficient**: Only falls back to SVD when Cholesky fails
- **Maintains Gradients**: SVD is differentiable and preserves the computational graph

## Testing
✅ Tested with ill-conditioned matrices that fail Cholesky
✅ Verified SVD fallback produces finite results
✅ Confirmed proper handling of complex numbers
✅ Validated gradient flow is maintained

## Impact
- **Eliminates HPO crashes** due to Cholesky failures
- **Maintains numerical stability** for all matrix conditions
- **Preserves training dynamics** with proper gradient flow
- **No performance penalty** for well-conditioned cases (Cholesky still preferred)

## Files Modified
- `/home/tahit/ris/MainMusic/ris_pytorch_pipeline/train.py` (lines 851-864)

## Status: ✅ **READY FOR HPO**
The system is now robust against Cholesky failures and ready for stable HPO runs.



