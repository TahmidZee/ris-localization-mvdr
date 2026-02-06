# Fix: Symmetric Equilibrium — Model Not Learning (2026-02-05)

## Symptom

After previous fixes (slot head gradient reconnection + Huber delta fix), the model
still failed to learn:

- `aux_φ_rmse` stuck at **~35°** across all 15 epochs (= dataset mean baseline)
- `aux_θ_rmse` stuck at **~17.4°**, `aux_r_rmse` at **~2.76m**
- Validation loss **increased** after epoch 5–6, triggering early stopping
- All 5 slot predictions nearly identical (masks ≈ 0.18, power ≈ 0.12)

## Root Cause: The Mask Warmup + GEOM_ONLY + Low Init Vicious Cycle

Three design choices interacted to create a **symmetric equilibrium** where all slots
converge to the dataset mean:

### The Chain of Failure

| Step | Setting | Effect |
|------|---------|--------|
| 1 | `mask_logit` bias = **-2.0** | Masks start at sigmoid(-2) ≈ **0.12** |
| 2 | `MASK_LOSS_WARMUP_EPOCHS = 10` | Mask losses OFF for 10 epochs → masks get **zero gradient** → stay at 0.12 |
| 3 | `power_eff = power × mask` | ≈ 0.13 × 0.12 = **0.015** → R_pred ≈ σ²I (noise floor) |
| 4 | `GEOM_ONLY_EPOCHS = 5` | lam_cov = 0 → NMSE provides **zero gradient** for 5 epochs |
| 5 | Only `aux_l2` drives learning | Perm-invariant matching → each slot gets matched to **different GT** per sample → average gradient → **dataset mean** |
| 6 | Unmatched slots get **no gradient** | Unlike DETR which pushes unmatched queries toward "no object" |

### Why It's a Stable Equilibrium

With 5 slots and K_true often 1–3:
- Only K of K_MAX=5 slots get geometry gradient per sample
- Different samples match different slots to GT via Hungarian matching
- Over many batches, each slot's average gradient points toward the **dataset mean**
- Diversity loss (weight 0.02) is far too weak vs aux (weight 1.5) to counteract
- Without mask BCE, unmatched slots have no gradient to "turn off"
- The model converges to: all slots ≈ mean(φ), all masks ≈ 0.18, RMSE ≈ 35°

This is exactly the **symmetric set prediction problem** that DETR solved by always
assigning unmatched queries to a "no object" class.

## Fixes Applied

### 1. `configs.py` — Remove Warmup Delays

```python
# BEFORE:
self.GEOM_ONLY_EPOCHS = 5
self.MASK_LOSS_WARMUP_EPOCHS = 10

# AFTER:
self.GEOM_ONLY_EPOCHS = 0   # Allow NMSE gradient from epoch 0
self.MASK_LOSS_WARMUP_EPOCHS = 0  # Enable mask BCE immediately
```

**Rationale**: Mask BCE from epoch 0 provides the "no object" gradient for unmatched
slots. NMSE gradient from epoch 0 provides a second gradient path through R_pred.

### 2. `model.py` — Balanced Bias Initialization

```python
# BEFORE:
last.bias[3].fill_(-2.0)  # power: softplus(-2) ≈ 0.13
last.bias[4].fill_(-2.0)  # mask:  sigmoid(-2) ≈ 0.12

# AFTER:
last.bias[3].fill_(0.0)   # power: softplus(0) ≈ 0.69
last.bias[4].fill_(0.0)   # mask:  sigmoid(0) = 0.50
```

**Rationale**: Starting masks at 0.5 (not 0.12) gives R_pred meaningful signal
components from the start. The mask BCE loss will push matched slots → 1 and
unmatched → 0, providing natural differentiation.

### 3. `loss.py` — Fix Dead Code Path + Default Scale

```python
# mask_loss_scale default: 0.0 → 1.0
self.mask_loss_scale = 1.0  # Active by default

# loss_nmse_pred: added structural R mode path
# Previously only worked with legacy factor heads (cov_fact_angle/range)
# which don't exist in structural R mode → was always 0.0
if "R_pred" in y_pred and "R_blend" not in y_pred:
    R_pred_eff = build_effective_cov_torch(y_pred['R_pred'], ...)
    loss_nmse_pred = self._nmse_cov(R_pred_eff, R_eff_true).mean()
```

## Expected Training Behavior After Fix

### Epoch 0–5 (Early Training)
- **Mask BCE**: Matched slots → target=1, unmatched → target=0 (DETR-style)
- **Masks differentiate**: Some slots → high mask (active), others → low (inactive)
- **power_eff becomes meaningful**: Active slots have power_eff ≈ 0.7 × 0.8 = 0.56
- **R_pred has signal**: Rank-K structure emerges
- **lam_cov** ramps from 0.02 → 0.10 (STRUCTURED_COV_WARMUP over 15 epochs)
- **lam_aux = 1.5** drives geometry learning with symmetry broken by mask signals

### Epoch 5–15 (Refinement)
- **lam_cov** reaches 0.3 — NMSE provides strong gradient for R_pred alignment
- **Masks stabilize**: K slots active, K_MAX−K inactive
- **Geometry improves**: aux_φ_rmse should decrease significantly (target: <10°)
- **lam_cov_pred = 0.05** provides constant additional NMSE pressure

## Current Loss Weights (No Changes Needed)

| Weight | Value | Status | Notes |
|--------|-------|--------|-------|
| `lam_cov` | 0.3 (warmed up over 15 ep) | ✅ OK | Meaningful now that R_pred has signal |
| `lam_cov_pred` | 0.05 | ✅ OK | Provides constant NMSE floor |
| `lam_aux` | 1.5 | ✅ OK | Primary geometry driver |
| `lam_subspace_align` | 0.0 | ✅ OK | **Redundant** in structural R mode — subspace is correct by construction |
| `lam_peak_contrast` | 0.0 | ✅ OK | Useful for fine-tuning, not needed for initial learning |
| `LAM_SLOT_DIVERSITY` | 0.02 | ✅ OK | Prevents slot collapse, small relative to aux |
| `LAM_AUX_MASK_BCE` | 0.2 | ✅ OK | Permutation-aware mask supervision |
| `LAM_AUX_MASK` | 0.3 | ✅ OK | Count-based mask loss |

### Why Subspace Alignment Is Not Needed

In structural R mode, `build_structured_R` constructs the covariance as:

```
R_pred = Σ_k power_k · a(φ_k, θ_k, r_k) · a(φ_k, θ_k, r_k)^H + σ²I
```

The signal subspace is **steering vectors by construction**. If the geometry (φ, θ, r)
is correct, the subspace is automatically correct. Subspace alignment loss is only
useful in legacy mode where the covariance is learned as free-form factors.

## Files Changed

| File | Changes |
|------|---------|
| `ris_pytorch_pipeline/configs.py` | `GEOM_ONLY_EPOCHS` 5→0, `MASK_LOSS_WARMUP_EPOCHS` 10→0 |
| `ris_pytorch_pipeline/model.py` | mask_logit bias -2→0, power bias -2→0 |
| `ris_pytorch_pipeline/loss.py` | `mask_loss_scale` default 0→1, added R_pred path to `loss_nmse_pred` |

## Prior Fixes (2026-02-03)

These were applied earlier and remain in place:

1. **`model.py` output mismatch**: `phi_soft`/`theta_soft` were returning soft-argmax
   outputs instead of slot head outputs, disconnecting the slot head from aux loss
   gradients entirely.

2. **`loss.py` Huber delta**: `_wrapped_huber_loss` delta was 0.25° (π/720 rad),
   causing all errors >0.25° to fall in the linear regime (constant gradient).
   Changed to 10° (0.175 rad) for quadratic gradients on small errors.

## Diagnostic Commands

```python
# Check slot head bias initialization
from ris_pytorch_pipeline.model import HybridModel
import torch
model = HybridModel()
last = model.slot_head[-1]
print(f"power bias: {last.bias[3].item():.2f} → softplus = {torch.nn.functional.softplus(last.bias[3]).item():.4f}")
print(f"mask bias:  {last.bias[4].item():.2f} → sigmoid = {torch.sigmoid(last.bias[4]).item():.4f}")

# Check loss config
from ris_pytorch_pipeline.loss import UltimateHybridLoss
loss_fn = UltimateHybridLoss()
print(f"mask_loss_scale: {loss_fn.mask_loss_scale}")
```
