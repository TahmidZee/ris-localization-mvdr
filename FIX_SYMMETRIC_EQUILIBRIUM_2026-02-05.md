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

---

## Round 1 Fixes (Initial Pass)

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
components from the start.

### 3. `loss.py` — Fix Dead Code Path + Default Scale

```python
# mask_loss_scale default: 0.0 → 1.0
self.mask_loss_scale = 1.0  # Active by default
```

---

## Round 2 Fixes (After First Training Still Failed)

The first training run showed the model *still* not learning:
- `aux_φ_rmse` ≈ 35–38°, validation loss increasing
- `lam_cov = 0.02` (should be much higher)
- `loss_nmse_pred = 0.000000` (dead code — not contributing at all)
- **Training loss was NEGATIVE** (e.g., train = -0.400983)

Investigation uncovered **3 critical bugs**:

### Bug 1: `loss_nmse_pred` Always Zero (Dead Code)

**File**: `loss.py`, line 828

```python
# BEFORE (BROKEN):
if "R_pred" in y_pred and "R_blend" not in y_pred:
    # This branch was NEVER entered because train.py always creates R_blend
    ...

# AFTER (FIXED):
if "R_pred" in y_pred:
    # Now enters correctly for structural R mode
    R_pred_eff = build_effective_cov_torch(y_pred['R_pred'], ...)
    loss_nmse_pred = self._nmse_cov(R_pred_eff, R_eff_true).mean()
```

**Impact**: `lam_cov_pred = 0.05` was multiplied by 0.0 every batch → zero gradient
from this auxiliary NMSE path. Now provides constant gradient pressure through
`R_pred → geometry`.

### Bug 2: Diversity Loss Returning NEGATIVE Values → Negative Total Loss

**File**: `loss.py`, `_slot_diversity_loss()`

```python
# BEFORE (BROKEN):
return -(dist_offdiag * eye_mask).sum() / (eye_mask.sum() + 1e-9)
# This was NEGATIVE, which when added to total loss, REWARDED slot spread
# instead of penalizing slot collapse

# AFTER (FIXED):
# Replaced with margin-based violation loss (always non-negative)
violation = torch.relu(margin - dist) * eye_mask
loss = violation.sum() / (eye_mask.sum() + 1e-9)
return loss  # Positive: penalizes slots closer than margin
```

**Impact**: The old negative return made the total training loss negative (-0.40 to
-0.46). The optimizer was minimizing a negative loss, which meant it was trying to
MAXIMIZE slot spread while IGNORING accuracy. This is why the model diverged.

### Bug 3: `STRUCTURED_COV_WARMUP_EPOCHS = 15` Keeping `lam_cov` at 0.02

**File**: `configs.py`

```python
# BEFORE (BROKEN):
STRUCTURED_COV_WARMUP_EPOCHS = 15  # Ramp lam_cov from 0 → 0.3 over 15 epochs
# At epoch 1: lam_cov = 0.3 × (1/15) = 0.02 ← almost zero!

PHASE_LOSS["joint"]["lam_cov"] = 0.3  # Target was only 0.3 even after warmup

# AFTER (FIXED):
STRUCTURED_COV_WARMUP_EPOCHS = 0   # No warmup: lam_cov = 1.0 from epoch 1

PHASE_LOSS["joint"]["lam_cov"] = 1.0  # Full weight, equal to aux
```

**Impact**: With `lam_cov = 0.02`, the covariance NMSE (≈1.4) contributed only
`0.02 × 1.4 = 0.028` to the loss, vs `1.5 × 1.02 = 1.53` from aux_l2. The NMSE
was effectively silenced. At `lam_cov = 1.0`, NMSE contributes `1.0 × 1.4 = 1.4`,
giving it comparable weight to aux_l2 and providing real gradient pressure.

---

## Summary of All Changes

| File | Change | Round |
|------|--------|-------|
| `configs.py` | `GEOM_ONLY_EPOCHS` 5 → 0 | Round 1 |
| `configs.py` | `MASK_LOSS_WARMUP_EPOCHS` 10 → 0 | Round 1 |
| `configs.py` | `STRUCTURED_COV_WARMUP_EPOCHS` 15 → 0 | Round 2 |
| `configs.py` | `PHASE_LOSS["joint"]["lam_cov"]` 0.3 → 1.0 | Round 2 |
| `model.py` | `mask_logit` bias -2.0 → 0.0 | Round 1 |
| `model.py` | `power` bias -2.0 → 0.0 | Round 1 |
| `loss.py` | `mask_loss_scale` default 0.0 → 1.0 | Round 1 |
| `loss.py` | `loss_nmse_pred` condition: remove `R_blend not in` guard | Round 2 |
| `loss.py` | `_slot_diversity_loss` sign fix: negative → positive | Round 2 |

## Current Loss Weights (After All Fixes)

| Weight | Value | Status | Notes |
|--------|-------|--------|-------|
| `lam_cov` | **1.0** (no warmup) | ✅ Fixed | Full weight from epoch 1 |
| `lam_cov_pred` | 0.05 | ✅ Fixed | Now actually computes NMSE |
| `lam_aux` | 1.5 | ✅ OK | Primary geometry driver |
| `lam_subspace_align` | 0.0 | ✅ OK | Redundant in structural R mode |
| `lam_peak_contrast` | 0.0 | ✅ OK | Not needed for initial learning |
| `LAM_SLOT_DIVERSITY` | 0.02 | ✅ Fixed | Now correctly positive |
| `LAM_AUX_MASK_BCE` | 0.2 | ✅ OK | Active from epoch 0 |
| `LAM_AUX_MASK` | 0.3 | ✅ OK | Active from epoch 0 |

## Expected Training Behavior After All Fixes

### Loss Breakdown (Epoch 1)
```
lam_cov  * loss_nmse     = 1.0  × ~1.0  = 1.00  (was 0.02 × 1.4 = 0.03)
lam_aux  * aux_l2        = 1.5  × ~1.0  = 1.50
lam_pred * loss_nmse_pred= 0.05 × ~1.0  = 0.05  (was 0.05 × 0.0 = 0.00)
diversity                                = 0.00+ (was NEGATIVE)
total                                   ≈ 2.55+ (was going negative!)
```

### What Should Improve
1. **NMSE gradient flows** through `R_pred → build_structured_R → geometry`
2. **Masks differentiate** via BCE from epoch 0 (matched → 1, unmatched → 0)
3. **Total loss is positive** and monotonically decreasing
4. **aux_φ_rmse should decrease** below 35° (dataset mean) within first 5 epochs

### Why Subspace Alignment Is Not Needed

In structural R mode, `build_structured_R` constructs the covariance as:

```
R_pred = Σ_k power_k · a(φ_k, θ_k, r_k) · a(φ_k, θ_k, r_k)^H + σ²I
```

The signal subspace is **steering vectors by construction**. If the geometry (φ, θ, r)
is correct, the subspace is automatically correct.

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
