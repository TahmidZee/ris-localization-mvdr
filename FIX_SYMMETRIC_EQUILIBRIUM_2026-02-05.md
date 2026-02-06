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
print(f"lam_slot_diversity: {loss_fn.lam_slot_diversity}")
print(f"lam_aux_sorted: {loss_fn.lam_aux_sorted}")
```

---

# Round 3: Full Codebase Audit & Refactor (2026-02-06)

## Context

After rounds 1 and 2, training showed:
- `Slot diversity loss: enabled @ weight=0.020` despite `configs.py` setting `LAM_SLOT_DIVERSITY = 0.5`
- `aux_φ_rmse` still stuck at ~35° (dataset mean baseline)
- Sorted canonical loss (`LAM_AUX_SORTED = 1.0`) may not have been active

A full end-to-end audit of all files was performed. Here are the findings and fixes.

---

## Bug Fix 1: `LAM_SLOT_DIVERSITY` / `LAM_AUX_SORTED` Not Reliably Picked Up

**File**: `loss.py` (lines 1046, 1063 — old; now fixed)

**Root Cause**: These weights were read from `mdl_cfg` at **forward time** via:
```python
lam_aux_sorted = float(getattr(mdl_cfg, "LAM_AUX_SORTED", 0.0))
lam_diversity  = float(getattr(mdl_cfg, "LAM_SLOT_DIVERSITY", 0.1))
```

While `mdl_cfg` is a singleton (so changes *should* propagate), this pattern is:
1. **Fragile** — the Trainer cannot override these without modifying the global singleton
2. **Opaque** — the actual weight used is never logged at construction time
3. **Inconsistent** — all other loss weights (lam_cov, lam_aux, etc.) are explicit constructor params

**Fix**: Made both weights **explicit constructor parameters** of `UltimateHybridLoss`:
```python
# loss.py __init__:
lam_slot_diversity: float = None,  # Read from mdl_cfg if None
lam_aux_sorted: float = None,      # Read from mdl_cfg if None

# In __init__ body:
self.lam_slot_diversity = float(getattr(mdl_cfg, "LAM_SLOT_DIVERSITY", 0.1)) if lam_slot_diversity is None else float(lam_slot_diversity)
self.lam_aux_sorted = float(getattr(mdl_cfg, "LAM_AUX_SORTED", 0.0)) if lam_aux_sorted is None else float(lam_aux_sorted)

# In forward:
lam_aux_sorted = self.lam_aux_sorted    # ← was: getattr(mdl_cfg, ...)
lam_diversity  = self.lam_slot_diversity # ← was: getattr(mdl_cfg, ...)
```

**Also added**: `_apply_phase_loss_weights` in `train.py` now supports setting
`lam_slot_diversity` and `lam_aux_sorted` per-phase if needed.

**Also added**: `log_config_summary()` method on `UltimateHybridLoss` that prints ALL
loss weights once after construction. Called automatically from `Trainer.__init__`.
This replaces the scattered `if not hasattr(self, "_xxx_logged")` debug prints and
gives a single authoritative snapshot of the configuration.

---

## Bug Fix 2: Train/Val `R_true` Trace Normalization Mismatch

**Files**: `train.py` — `_validate_one_epoch` (line 1454) and `_validate_surrogate_epoch` (line 1735)

**Root Cause**: In `_train_one_epoch`, `R_true` was trace-normalized to `trace=N` before
being passed to the loss function:
```python
# _train_one_epoch (line 1004):
N = R_true_c.shape[-1]
tr_true = torch.diagonal(R_true_c, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-9)
R_true_c = R_true_c * (N / tr_true).view(-1, 1, 1)
```

This normalization was **missing** in `_validate_one_epoch` and `_validate_surrogate_epoch`:
```python
# _validate_one_epoch (line 1454) — BEFORE:
R_true_c = _ri_to_c(R_in)
R_true_c = 0.5 * (R_true_c + R_true_c.conj().transpose(-2, -1))
R_true   = _c_to_ri(R_true_c).float()  # No trace normalization!
```

**Impact**: While `loss.py` does its own internal trace normalization (divides by trace),
having inconsistent pre-normalization between train and val means the raw `R_true` values
in the labels dict have different scales. This can cause subtle numerical precision
differences in float32 RI conversion and affects any code that inspects labels directly.

**Fix**: Added matching trace normalization to both validation functions:
```python
# _validate_one_epoch and _validate_surrogate_epoch — AFTER:
R_true_c = _ri_to_c(R_in)
R_true_c = 0.5 * (R_true_c + R_true_c.conj().transpose(-2, -1))
N = R_true_c.shape[-1]
tr_true = torch.diagonal(R_true_c, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-9)
R_true_c = R_true_c * (N / tr_true).view(-1, 1, 1)
R_true   = _c_to_ri(R_true_c).float()
```

---

## Full Audit Results: No Other Bugs Found

The following files were audited end-to-end for logical and numerical bugs:

| File | Status | Notes |
|------|--------|-------|
| `model.py` | ✅ Clean | Slot head, structural R, steering vectors all correct |
| `loss.py` | ✅ Fixed | 2 bugs fixed (see above), all loss terms verified |
| `train.py` | ✅ Fixed | 1 bug fixed (val normalization), training loop verified |
| `configs.py` | ✅ Clean | All values consistent with intended behavior |
| `covariance_utils.py` | ✅ Clean | `build_effective_cov_torch` correctly chains hermitize→trace-norm→blend→diag-load→shrink |
| `physics.py` | ✅ Clean | `nearfield_vec` and `shrink` consistent with model's steering |

### Key Verified Behaviors

1. **Steering vector convention** is consistent across `model.py` (`build_steering_matrix_batch`),
   `loss.py` (`_steer_torch`), `physics.py` (`nearfield_vec`), and `covariance_utils.py`:
   - `phase = k0 * (planar - curvature)`
   - `planar = x * sin(φ) * cos(θ) + y * sin(θ)`
   - `curvature = (x² + y²) / (2r)`
   - Unit-normalized: `a / sqrt(N)`

2. **R_true processing in loss.py** is self-contained and correct:
   - Hermitizes, trace-normalizes to trace=1, then passes through `build_effective_cov_torch`
   - Same pipeline applied to both R_pred and R_true for NMSE computation

3. **Slot head gradient flow** verified:
   - `phi_soft`/`theta_soft` correctly return slot head outputs (not soft-argmax)
   - `R_pred = build_structured_R(aux_phi, aux_theta, aux_r, power_eff, cfg)`
   - Full gradient path: `loss → R_pred → build_structured_R → aux_phi/theta/r → slot_head → backbone`

4. **Permutation-invariant matching** is correct:
   - Brute-force over `permutations(range(K_MAX), k)` — correct for K_MAX=5
   - Cost uses same Huber structure as loss (consistent)
   - Hard assignment under `no_grad` — gradients flow through selected permutation only

5. **Sorted canonical loss** is correct:
   - Sorts both GT and predictions by phi
   - Takes first k sorted predictions → first k sorted GT
   - Provides deterministic gradients to break symmetric equilibrium

---

## Updated Current Loss Weights (After Round 3)

| Weight | Value | Source | Notes |
|--------|-------|--------|-------|
| `lam_cov` | **1.0** | `PHASE_LOSS["joint"]` | Full weight, no warmup |
| `lam_cov_pred` | 0.05 | `configs.py` | Aux NMSE on R_pred |
| `lam_aux` | 1.5 | `PHASE_LOSS["joint"]` | Primary geometry driver |
| `lam_slot_diversity` | **0.5** | `configs.py` → loss constructor | Repulsive force between slots |
| `lam_aux_sorted` | **1.0** | `configs.py` → loss constructor | Sorted matching to break symmetry |
| `lam_subspace_align` | 0.0 | `PHASE_LOSS["joint"]` | Off (redundant in structural R) |
| `lam_peak_contrast` | 0.0 | `PHASE_LOSS["joint"]` | Off (not needed for initial learning) |
| `LAM_AUX_MASK_BCE` | 0.2 | `configs.py` | Permutation-aware mask BCE |
| `LAM_AUX_MASK` | 0.3 | `configs.py` | Mask count loss |
| `LAM_AUX_MASK_BIN` | 0.1 | `configs.py` | Mask binarization penalty |
| `mask_loss_scale` | 1.0 | loss constructor | Active from epoch 0 |

### Expected Loss Breakdown (Epoch 1)
```
lam_cov    * loss_nmse        = 1.0  × ~1.4  = 1.40
lam_aux    * aux_l2           = 1.5  × ~1.0  = 1.50
lam_sorted * sorted_loss      = 1.0  × ~1.0  = 1.00  (NEW: breaks symmetry)
lam_div    * diversity_loss    = 0.5  × ~0.15 = 0.08  (NEW: slot repulsion)
lam_pred   * loss_nmse_pred   = 0.05 × ~1.4  = 0.07
mask_count                                    ≈ 0.03
mask_bce                                      ≈ 0.02
total                                        ≈ 4.10
```

---

## Architecture Summary (For Reference)

```
Input: y[B,L,M,2], H_full[B,M,N,2], codes[B,L,N,2], snr_db[B]
  │
  ├── Conv1d tokenization → [B,D,L] → Transformer → [B,L,D]
  ├── H_full → Conv2d stack → [B,D/2]
  ├── SNR → MLP embed → [B,snr_dim]
  │
  └── Fusion → [B,D]
        │
        ├── Slot queries (K=5) × 3-round cross-attention → [B,K,D]
        │     └── slot_head MLP → [B,K,5]: (φ,θ,r,power,mask)
        │           │
        │           ├── φ = tanh(raw) × 60°
        │           ├── θ = tanh(raw) × 30°
        │           ├── r = R_MIN + (R_MAX-R_MIN) × softplus(raw)/(1+softplus(raw))
        │           ├── power = softplus(raw)
        │           └── mask = sigmoid(raw)
        │
        └── build_structured_R(φ,θ,r, power×mask) → R_pred [B,N,N]
              │
              └── build_effective_cov_torch → R_blend [B,N,N]
```

### Loss Terms
```
total = lam_cov * NMSE(R_eff_pred, R_eff_true)          # Primary: covariance
      + lam_aux * perm_invariant_aux(φ,θ,r)              # Geometry: Huber on angles/log-range
      + lam_aux_sorted * sorted_canonical_aux(φ,θ,r)     # Symmetry breaker
      + lam_slot_diversity * margin_diversity(φ,θ,r)     # Slot repulsion
      + lam_cov_pred * NMSE(R_pred, R_eff_true)          # Aux gradient path
      + mask_bce * perm_invariant_mask_bce               # "No object" for unmatched slots
      + mask_count * (sum(mask) - K_true)²               # Count supervision
      + mask_bin * mask*(1-mask)                          # Binarization
      + lam_ortho * ortho_penalty(A_angle)               # Stiefel regularizer
```

---

# Round 4: Pre-Training Cleanup (2026-02-06)

## Context

Before the first training run with all Round 1–3 fixes, a full codebase audit was performed
to catch any remaining issues that could cause crashes, incorrect behavior, or wasted compute
during training.

## Fixes Applied

### Fix 1: Mixed Tab/Space Indentation in `train.py` (Potential Crash)

**Lines**: 1367, 1378, 1385, 1388, 1396, 1405

Six `print()` statements inside `_train_one_epoch` used a TAB character before spaces for
indentation. Python 3 forbids inconsistent mixing of tabs and spaces within the same block,
which could trigger a `TabError` depending on the Python interpreter version.

**Fix**: Replaced all 6 tab-indented lines with consistent space indentation.

### Fix 2: Missing R_blend in Validation for Structural R Mode (Silent Metric Loss)

**Files**: `train.py` — `_validate_one_epoch` (line ~1553) and `_validate_surrogate_epoch` (line ~1792)

**Root Cause**: Both validation methods only built `R_blend` when legacy factor heads
(`cov_fact_angle`, `cov_fact_range`) were present. In structural R mode (the default),
these keys don't exist in the model output, so `R_blend` was never set.

**Impact**: Surrogate validation silently skipped all MVDR peak-level detection metrics
(precision/recall/F1) and subspace overlap metrics, because they check for `"R_blend" in preds`.
This meant the validation feedback loop was less informative than intended.

**Fix**: Added an `elif "R_pred" in preds` branch that builds `R_blend` from `R_pred` via
`build_effective_cov_torch`. The legacy factor path is preserved as a fallback for ablations.

### Fix 3: Wasted Soft-Argmax Computation in `model.py` (GPU Waste)

**File**: `model.py` — `HybridModel.forward()` (lines 670–696)

**Root Cause**: When `USE_SLOT_HEAD=True` AND `USE_STRUCTURED_R=True` (both default), the
factored soft-argmax heads (`phi_logits`, `theta_logits`) were computed every forward pass,
producing `phi_soft` and `theta_soft` values. These local variables were then immediately
overwritten by the slot head outputs in the return dict (lines 768–769).

**Impact**: ~0.3M parameters worth of GPU computation wasted on every forward pass (linear
projection + softmax + weighted sum, for both φ and θ, across K_MAX×G grid points).

**Fix**: Wrapped the soft-argmax block in `if not (self.use_slot_head and self.use_structured_R):`
so it only runs when the legacy path actually needs the results. The soft-argmax `nn.Linear`
layers remain in the model for checkpoint compatibility and ablation use.

### Fix 4: Silenced R_samp Warnings (Log Noise)

**File**: `train.py` — `_aggregate_cpu_then_gpu()` and `_train_one_epoch()`

**Root Cause**: Three warning messages printed during training when R_samp was absent from
shards. Since R_samp is intentionally not stored (`HYBRID_COV_BETA=0.0`,
`STORE_RSAMP_IN_SHARDS=False`), these warnings were pure noise.

**Silenced messages**:
- `⚠️ WARNING: Shard '...' lacks 'R_samp'` — printed per-shard during GPU cache build
- `⚠️ WARNING: No R_samp found in dataset!` — printed once after all shards loaded
- `[Hybrid] R_samp not available; using pure R_pred for loss.` — printed on epoch 1 batch 0

## Verified Correct (No Changes Needed)

| Component | Status |
|-----------|--------|
| Steering vector convention (model/loss/physics/covariance_utils) | ✅ Consistent |
| `build_structured_R` (geometry → R_pred) | ✅ Correct |
| `build_effective_cov_torch` chain | ✅ Correct |
| Permutation-invariant matching | ✅ Correct |
| Sorted canonical loss | ✅ Correct |
| Slot diversity loss (margin-based, non-negative) | ✅ Correct |
| Mask BCE loss (DETR "no object") | ✅ Correct |
| Loss weight application order (constructor → HPO → phase → log) | ✅ Correct |
| Train/val R_true trace normalization | ✅ Consistent |
| EMA/SWA swap logic | ✅ Correct |
| Auto-resume from train_state.pt | ✅ Correct |

## Known Dormant Issues (Deferred to Phase 1 Cleanup)

These do NOT affect the current training run:

1. **Refiner-only training crashes in structural R mode** — `TRAIN_PHASE="refiner"` accesses
   `cov_fact_angle` which doesn't exist. Won't be used until after backbone training succeeds.

2. **Eigenspectrum diagnostic crash** — only triggers with `TRAIN_EPOCH_DEBUG=True` (default False).

3. **Dead code** (~800 lines) — 3-phase curriculum, legacy factor helpers, unused functions.
   Already planned in `REFACTOR_PROGRESS.md`.

---

# Round 5: NMSE Gradient Domination Fix (2026-02-06)

## Symptom

After all Round 1–4 fixes, training run showed:
- `aux_φ_rmse` stuck at **~34.7°** (= dataset mean for ±60° uniform)
- `aux_θ_rmse` stuck at **~17.5°** (= dataset mean for ±30° uniform)
- `aux_r_rmse` stuck at **~2.75m** (= dataset mean for [0.5, 10] uniform)
- Train loss decreased slightly (3.54 → 3.49) but aux metrics were FLAT for 6 epochs
- Gradients were finite (||g||₂ = 43.2), steps were taken — model appears to learn but doesn't

## Root Cause: NMSE Gradient Dominates Clip Budget

The total gradient norm was **43.2**, and `CLIP_NORM = 1.0`. Gradient clipping scales
ALL gradients by `1.0 / 43.2 = 0.023` — a **43× reduction**.

The NMSE loss on R_pred generates a massive gradient because:
- R_pred is [B, 256, 256] complex (65,536 entries per sample)
- The Jacobian ∂R_pred/∂(phi,theta,r) involves 256-dimensional steering vector derivatives
- This gradient dominates the total norm

The aux/sorted/diversity gradients operate on [B, 5] tensors and are comparatively tiny.
After clipping, their effective learning rate is:

```
effective_head_LR = 1.2e-3 × (1.0 / 43.2) ≈ 2.8e-5
```

Over 1562 batches/epoch: total parameter change ≈ 0.044. With conflicting gradient directions
across batches (permutation matching instability), the NET signal for geometry is near zero.

### Why This Wasn't a Problem Before Round 1

In the original code, `GEOM_ONLY_EPOCHS = 5` disabled NMSE for the first 5 epochs.
But back then, `MASK_LOSS_WARMUP_EPOCHS = 10` also disabled mask losses, leaving only
aux_l2 — which created a symmetric equilibrium.

Round 1 fixed masks (warmup=0, bias=0.0) but ALSO set `GEOM_ONLY_EPOCHS = 0`,
reasoning that "NMSE provides a second gradient path." In practice, the NMSE gradient
is too large and drowns out the symmetry-breaking signals.

## Fix

### Change 1: `GEOM_ONLY_EPOCHS = 3` (was 0)

Force `lam_cov = 0` for the first 3 epochs. Without the 256×256 NMSE gradient:
- Total gradient norm drops from ~43 to ~5-10
- Clip scale improves from 1/43 to 1/5–1/10
- Aux/sorted/diversity get **5-10× more effective learning rate**
- Slots should break symmetry within 3 epochs

This is safe because mask BCE is active from epoch 0 (unlike the original setup).

### Change 2: `STRUCTURED_COV_WARMUP_EPOCHS = 5` (was 0)

After the geometry-only phase (epoch 3), ramp `lam_cov` from 0 → 1.0 over 5 epochs.
This prevents the NMSE gradient from suddenly overwhelming the still-young aux signals.

```
Epoch  1-3:  lam_cov = 0.0  (geometry-only: aux + sorted + diversity + mask)
Epoch  4:    lam_cov = 0.2  (NMSE starts ramping in)
Epoch  5:    lam_cov = 0.4
Epoch  6:    lam_cov = 0.6
Epoch  7:    lam_cov = 0.8
Epoch  8+:   lam_cov = 1.0  (full weight)
```

### Change 3: Fix double-processing bug in surrogate validation

`_validate_surrogate_epoch` was building R_blend with `diag_load=True, apply_shrink=True`,
then loss.py applied `build_effective_cov_torch` AGAIN with `diag_load=True, apply_shrink=True`.
This double-processed R_blend and inflated the validation loss (~8.6 vs train ~3.5).

Fixed to match training: `diag_load=False, apply_shrink=False` in the validation R_blend
construction (loss.py does its own processing).

## Expected Behavior

- **Epochs 1-3**: Slots spread apart (diversity), sorted loss assigns each to different
  order statistic of GT distribution. `aux_φ_rmse` should start decreasing below 35°.
- **Epochs 4-8**: NMSE ramps in gradually. Geometry is already differentiated, so NMSE
  gradient pushes R_pred toward correct per-sample covariance (not dataset average).
- **Epochs 9+**: Full training with all losses. Expect continued improvement.
