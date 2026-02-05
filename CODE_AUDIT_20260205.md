# Comprehensive Code Audit: RIS Localization Pipeline
**Date:** 2026-02-05  
**Status:** Critical refactoring recommended before wideband upgrade

---

## Executive Summary

After extensive debugging and patching, the pipeline **functionally works** (smoke tests pass), but has accumulated **significant technical debt**:

| Metric | train.py | loss.py | model.py |
|--------|----------|---------|----------|
| **Lines of code** | 3,420 | 1,338 | 1,111 |
| **Conditionals** | 471 | 114 | 51 |
| **Warning comments** | 72 | 14 | 10 |
| **Largest method** | 696 lines (fit) | 463 lines (forward) | 215 lines (forward) |

**Recommendation**: Refactor before adding wideband (F dimension), or the codebase will become unmaintainable.

---

## Critical Issues Found

### 1. **train.py is Unmaintainably Large** 🔴 CRITICAL

#### Method Sizes
| Method | Lines | Problem |
|--------|-------|---------|
| `fit()` | **696** | Should be <100; handles too many responsibilities |
| `_train_one_epoch()` | **476** | Should be <150; mixes forward/backward/logging/scheduling |
| `_validate_surrogate_epoch()` | **445** | Should be <150; duplicates logic with `_eval_hungarian_metrics` |
| `_eval_hungarian_metrics()` | **333** | Overlaps with surrogate validation |
| `__init__()` | **315** | Should be <100; mixes config/HPO/optimizer/loss setup |

#### Redundant Code Paths
- **3 validation methods** that do similar things: `_validate_one_epoch`, `_validate_surrogate_epoch`, `_eval_hungarian_metrics`
- **2 dataloader builders**: `_build_loaders_cpu_io`, `_build_loaders_gpu_cache`
- **Refiner-only path**: 42 conditionals scattered throughout instead of a separate trainer class

#### Disabled-by-Default Features (Dead Weight)
- **3-phase curriculum**: 12 references, `USE_3_PHASE_CURRICULUM = False`
- **SWA (Stochastic Weight Averaging)**: 72 lines of code, enabled but adds complexity
- **Auto-resume logic**: Complex state saving/loading that often breaks on arch changes

#### Debug Logging Pollution
- 35 debug print statements with `epoch_dbg`, `loopcheck_dbg`, `_should_log_batch` checks
- 43 "CRITICAL FIX" / "Expert fix" comments (should be cleaned up post-fix)

---

### 2. **loss.py `forward()` is 463 Lines** 🟠 HIGH

#### Problems
| Issue | Lines Wasted |
|-------|--------------|
| **Inline helper functions** (e.g., `_vec2c`, `_col_norm`) | ~20 |
| **Factor head fallback** (unused when `USE_STRUCTURED_R=True`) | ~40 |
| **Debug logging** (selftest, shape checks, breakdown printing) | ~60 |
| **Disabled terms** (gap/margin with warnings) | ~15 |
| **Duplicated FP32 casting logic** | ~30 |

#### Redundant Loss Terms
- `_eigengap_hinge()` and `_subspace_margin_regularizer()` both do similar things
- `lam_gap` and `lam_margin` are **always 0** (disabled globally) but still computed
- `_angle_chamfer()` and `_perm_invariant_aux_loss()` both penalize angle errors

---

### 3. **model.py Has Unused Legacy Paths** 🟡 MEDIUM

#### Dead Code (When `USE_STRUCTURED_R=True`, which is the default)
```python
# Lines 387-398: Covariance factor heads (unused in structural R mode)
if not self.use_structured_R:
    self.cov_fact_angle = nn.Linear(D, cfg.N * cfg.K_MAX * 2)
    self.cov_fact_range = nn.Linear(D, cfg.N * cfg.K_MAX * 2)
else:
    # Dummy linear layers (not used, but kept for API compatibility)
    # This saves 2.6M parameters!
    self.cov_fact_angle = None
    self.cov_fact_range = None
```

#### Redundant Angle Heads
- Slot head produces `aux_phi`, `aux_theta`, `aux_r`
- Soft-argmax grid head produces `phi_soft`, `theta_soft` (but these are overwritten to use slot outputs!)
- The grid head is **9.2M params** but only used for an ablation baseline

---

## Specific Redundancies & Inefficiencies

### train.py

#### 1. **Duplicate Validation Logic** (3 methods doing similar things)

| Method | Purpose | Lines |
|--------|---------|-------|
| `_validate_one_epoch` | Basic loss + debug terms | 184 |
| `_validate_surrogate_epoch` | Aux RMSE + optional peak metrics | 445 |
| `_eval_hungarian_metrics` | MUSIC-based metrics | 333 |

**Problem**: All three loop over batches, run forward(), compute errors. ~90% code overlap.

**Fix**: Extract common logic to `_validate_batch(batch) -> metrics_dict`, then each validation method just aggregates different metrics.

#### 2. **GPU Cache Logic Mixed into Trainer** (165 lines)

`_aggregate_cpu_then_gpu()` is a 165-line method that:
- Samples from ShardNPZDataset
- Loads .npz files
- Concatenates tensors
- Creates TensorDataset

**Fix**: Extract to `ris_pytorch_pipeline/data_utils.py` as a standalone function.

#### 3. **HPO Config Application Scattered** (3 methods)

- `_apply_hpo_to_mdl_cfg()`: 35 lines
- `_apply_hpo_loss_weights()`: 30 lines  
- `_apply_phase_loss_weights()`: 28 lines

**Fix**: Merge into single `_configure_from_hpo(hpo_dict)` method.

#### 4. **Refiner-Only Training Mixed In**

42 conditionals like:
```python
if self.train_refiner_only:
    # ... special path
else:
    # ... normal path
```

**Fix**: Create `RefinerTrainer(Trainer)` subclass that overrides only what's different.

#### 5. **Excessive Debugging Scaffolding**

```python
epoch_dbg = bool(getattr(cfg, "TRAIN_EPOCH_DEBUG", False))
loopcheck_dbg = bool(getattr(cfg, "TRAIN_LOOPCHECK_DEBUG", False))
def _should_log_batch(bi): ...
if epoch_dbg:
    print(...)
if loopcheck_dbg:
    print(...)
# ... 35 such blocks
```

**Fix**: Use Python's `logging` module with configurable levels, not ad-hoc `if` blocks.

---

### loss.py

#### 1. **`forward()` is 463 Lines** (should be <200)

**Breakdown:**
- Lines 660-740: Setup (80 lines) — reasonable
- Lines 741-855: NMSE losses (115 lines) — could extract `_compute_nmse_losses()`
- Lines 856-1000: Aux losses (145 lines) — could extract `_compute_aux_losses()`
- Lines 1001-1070: Aggregate losses (70 lines) — reasonable
- Lines 1071-1122: Debug logging (52 lines) — **remove in production**

**Fix**: Break into 3-4 submethods.

#### 2. **Disabled Loss Terms Still Computed**

```python
# Eigengap / margin terms are disabled in this system.
if (self.lam_gap != 0.0) or (self.lam_margin != 0.0):
    if not hasattr(self, "_gap_margin_disabled_warned"):
        print("[WARN] lam_gap/lam_margin are disabled...")
        self._gap_margin_disabled_warned = True
loss_gap = torch.tensor(0.0, device=device)
loss_margin = torch.tensor(0.0, device=device)
```

**Fix**: Remove `_eigengap_hinge()` and `_subspace_margin_regularizer()` entirely (they're never used).

#### 3. **Two Aux Loss Implementations**

- `_perm_invariant_aux_loss()`: 101 lines (optimal matching)
- `_sorted_aux_loss()`: 65 lines (sorted matching, **now disabled**)

**Fix**: Remove `_sorted_aux_loss()` since `USE_SORTED_MATCHING=False` is now the correct default.

---

### model.py

#### 1. **Soft-Argmax Grid Head is Unused** (when slot head active)

Lines 663-690: Compute `phi_soft`, `theta_soft` from grid head, then **immediately overwrite** with slot outputs (lines 755-757).

**Fix**: Make grid head fully optional when `USE_SLOT_HEAD=True`.

#### 2. **Factor Heads Are Dummy Objects**

```python
if not self.use_structured_R:
    self.cov_fact_angle = nn.Linear(...)
    self.cov_fact_range = nn.Linear(...)
else:
    self.cov_fact_angle = None  # Dummy
    self.cov_fact_range = None  # Dummy
```

**Fix**: Remove them entirely when `USE_STRUCTURED_R=True`. Add a `LegacyFactorModel` for ablations if needed.

#### 3. **AntiDiagPool is Disabled in Structural R Mode**

Lines 651-657: Big comment saying AntiDiagPool is disabled, but the code still computes features and has a 328K-param `fusion_with_antidiag` layer.

**Fix**: Actually skip AntiDiagPool construction when `USE_STRUCTURED_R=True`.

---

## Major Refactoring Opportunities

### 1. **Split Trainer into Multiple Classes** (HIGH IMPACT)

```python
# Proposed structure:
class BaseTrainer:
    """Core training loop, loss computation, optimization step"""
    
class BackboneTrainer(BaseTrainer):
    """Full model training (current default)"""
    
class RefinerTrainer(BaseTrainer):
    """Freeze backbone, train SpectrumRefiner only"""
```

**Lines saved**: ~300+ (remove 42 `if self.train_refiner_only` checks)

### 2. **Extract Data Loading to Separate Module** (MEDIUM IMPACT)

```python
# New file: ris_pytorch_pipeline/data_loader.py
class ShardDataLoader:
    @staticmethod
    def build_cpu_io_loaders(...) -> Tuple[DataLoader, DataLoader]
    
    @staticmethod
    def build_gpu_cache_loaders(...) -> Tuple[DataLoader, DataLoader]
```

**Lines saved**: ~250

### 3. **Extract Validation Logic** (MEDIUM IMPACT)

```python
# New file: ris_pytorch_pipeline/validation.py
class Validator:
    def validate_batch(self, batch) -> Dict[str, float]:
        """Common forward + metric computation"""
    
    def compute_surrogate_metrics(self, all_batches) -> Dict:
    
    def compute_music_metrics(self, all_batches) -> Dict:
```

**Lines saved**: ~400 (remove duplicated batch loops)

### 4. **Use Proper Logging** (SMALL IMPACT, BIG QUALITY IMPROVEMENT)

```python
import logging
logger = logging.getLogger(__name__)

# Instead of:
if epoch_dbg:
    print(f"[EPOCH DEBUG] ...")

# Use:
logger.debug(f"[EPOCH] ...")
```

**Lines saved**: ~100 (remove debug flag checks)

### 5. **Remove Dead Loss Terms** (SMALL IMPACT)

Delete entirely:
- `_eigengap_hinge()` (never called with lam_gap>0)
- `_subspace_margin_regularizer()` (never called with lam_margin>0)
- `_sorted_aux_loss()` (USE_SORTED_MATCHING=False)
- `debug_terms()` method (144 lines, only used for logging)

**Lines saved**: ~300

---

## Remaining Bugs / Concerns

### 1. **Training Instability at Epoch 4** (YOUR CURRENT ISSUE)

**Symptom**: φ RMSE jumps from 34° → 48° at epoch 4.

**Root cause**: Hungarian matching can still flip assignments when predictions are clustered. You need **soft matching warmup**.

**Fix applied** (just pushed): Disabled `lam_peak` and `range_raw` terms under permutation-invariant aux.

**Recommended additional fix**:
```python
# In configs.py:
mdl_cfg.AUX_MATCH_SOFT_EPOCHS = 3  # Use softmin for first 3 epochs
```

### 2. **Validation Uses Model Predictions, Not phi_theta_r Output** ⚠️

Lines 1817-1852 in `train.py`:
```python
if "phi_theta_r" in preds:
    phi_theta_r_pred = preds["phi_theta_r"].float().cpu()
    # ... compute aux RMSE
```

But `phi_theta_r` is the **concatenated aux_ptr from the slot head**, which goes through tanh/softplus **bounding**. If you want to measure **unbounded model learning**, you'd log `phi_raw`/`theta_raw`/`r_raw` instead.

**Current behavior is correct** for measuring "what would inference see" (bounded outputs), but the metric name (`aux_φ_rmse`) is misleading if you think it's measuring "raw model capability."

### 3. **No Gradient Accumulation Validation** ⚠️

`_train_one_epoch()` has `grad_accumulation` parameter (line 951), but it's never validated/tested and the default is 1. If someone sets it >1, the optimizer step logic might break.

### 4. **Memory Leak Risk in Validation** ⚠️

Lines 1807-1808, 2245-2246: Validation builds full `R_blend` tensors **per batch** but doesn't explicitly delete them. At validation time with 100+ batches and N=256, this could accumulate 50+ GB.

**Fix**: Add `del preds, R_eff, R_blend` at end of batch loop.

### 5. **SpectrumRefiner is Built Even When Not Used**

`HybridModel.__init__()` doesn't create a refiner, but `Trainer.__init__()` creates one when `train_refiner_only=True`. However, the normal training path **never uses it**, so it's wasted memory.

---

## Specific Redundancies to Remove

### train.py

| Code | Lines | Status | Recommendation |
|------|-------|--------|----------------|
| 3-phase curriculum (`_apply_phase_weights`) | 73 | Disabled | **DELETE** (USE_3_PHASE_CURRICULUM=False) |
| SWA logic | 72 | Enabled | **Keep**, but extract to mixin/helper |
| EMA logic | 49 | Enabled | **Keep**, but extract to mixin/helper |
| `_param_vec_prev` drift tracking | ~20 | Debug only | **DELETE** or gate behind DEBUG flag |
| Scale/overflow logging | ~15 | Debug only | **DELETE** or use logging.debug() |
| `_classical_music_nearfield()` | 125 | Only used in one test | **Move to test file** |

### loss.py

| Code | Lines | Status | Recommendation |
|------|-------|--------|----------------|
| `_sorted_aux_loss()` | 65 | Disabled | **DELETE** (USE_SORTED_MATCHING=False) |
| `_eigengap_hinge()` | 28 | Never used | **DELETE** (lam_gap always 0) |
| `_subspace_margin_regularizer()` | 23 | Never used | **DELETE** (lam_margin always 0) |
| `debug_terms()` | 144 | Logging only | **DELETE** or simplify to 20 lines |
| Debug prints in `forward()` | ~60 | One-time prints | **Move to unit tests** |
| Factor head fallback | ~40 | Unused | **DELETE** when USE_STRUCTURED_R=True |

### model.py

| Code | Lines | Status | Recommendation |
|------|-------|--------|----------------|
| Soft-argmax grid head | ~90 | Redundant | **Make optional** when USE_SLOT_HEAD=True |
| Factor heads (dummy) | ~40 | Unused | **Remove** when USE_STRUCTURED_R=True |
| AntiDiagPool (disabled) | ~80 | Disabled | **Make conditional** on USE_STRUCTURED_R |
| SpectrumRefiner in model.py | ~300 | Separate concern | **Move to separate file** |

---

## Architectural Issues

### 1. **No Separation of Concerns**

`Trainer` class does:
- Data loading (2 strategies)
- Model construction
- Loss configuration
- Optimizer setup
- Learning rate scheduling
- Training loop
- Validation (3 different kinds)
- Checkpoint saving/loading
- HPO config application
- EMA/SWA weight averaging
- Curriculum/phase scheduling
- Refiner-only training

**That's 13 responsibilities in one class!**

### 2. **Global Config Mutation**

Many places do:
```python
setattr(mdl_cfg, "LAM_HEATMAP", 0.1)
setattr(cfg, "TRAIN_PHASE", "refiner")
```

This mutates **global state** and makes the code hard to test/reason about.

**Fix**: Pass config objects explicitly, don't mutate globals.

### 3. **Tight Coupling Between train.py and loss.py**

`Trainer` directly sets:
```python
self.loss_fn.lam_cov = 0.0
self.loss_fn.set_mask_loss_scale(mask_scale)
self.loss_fn.set_aux_match(soft, tau)
```

This makes it impossible to use `UltimateHybridLoss` without a `Trainer`.

**Fix**: Loss function should accept a `LossSchedule` object that encapsulates all the dynamic weight updates.

---

## Recommended Refactoring Plan

### Phase 1: Immediate Cleanup (Before Wideband) — 2-3 days

1. **Delete dead code**:
   - Remove `_sorted_aux_loss`, `_eigengap_hinge`, `_subspace_margin_regularizer`
   - Remove 3-phase curriculum (`USE_3_PHASE_CURRICULUM` paths)
   - Remove factor head code when `USE_STRUCTURED_R=True`

2. **Extract data loading**:
   - Move `_build_loaders_*` and `_aggregate_cpu_then_gpu` to `data_loader.py`

3. **Simplify validation**:
   - Extract `_validate_batch()` helper
   - Make `_validate_one_epoch`, `_validate_surrogate_epoch` thin wrappers

4. **Clean up logging**:
   - Replace ad-hoc `if epoch_dbg: print(...)` with `logging.debug(...)`
   - Remove "CRITICAL FIX" comments (move to git history)

**Expected result**: ~800 lines removed, train.py down to ~2,600 lines.

### Phase 2: Structural Refactoring (Optional, Post-Wideband) — 1 week

1. **Split Trainer**:
   ```
   BaseTrainer (core loop + optimization)
   ├── BackboneTrainer (current default)
   └── RefinerTrainer (freeze backbone path)
   ```

2. **Extract validation**:
   ```
   Validator class with pluggable metrics
   ```

3. **Config management**:
   ```
   TrainingConfig dataclass (immutable, no global mutation)
   ```

**Expected result**: train.py down to ~1,500 lines, much better testability.

---

## Critical Bugs Still Present

### 1. **Hungarian Matching Can Still Flip** (Current Training Failure)

Even with optimal assignment, when all 5 slots predict φ≈6°±4°, the assignment can flip batch-to-batch based on tiny prediction noise.

**Evidence**: Your epoch 4 jump (φ RMSE 34° → 48°).

**Fix** (URGENT):
```python
# In configs.py:
mdl_cfg.AUX_MATCH_SOFT_EPOCHS = 5  # Soft matching for first 5 epochs
```

This uses **softmin over all permutations** (weighted average of gradients) instead of hard argmin, preventing abrupt flips.

### 2. **Chamfer Loss Uses Unwr apped Angles** ⚠️

`_angle_chamfer()` computes `(P - T)**2` directly without angle wrapping. For angles near ±π, this gives wrong distances.

**Fix**: Wrap phi differences:
```python
def _wrap_angle_diff(a, b):
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))
    
# In _angle_chamfer:
dphi = _wrap_angle_diff(phi_p.unsqueeze(2), phi_t.unsqueeze(1))
dth = _wrap_angle_diff(theta_p.unsqueeze(2), theta_t.unsqueeze(1))
D = (dphi**2 + dth**2)
```

(Note: I already disabled `lam_peak` under perm-invariant aux, so this won't affect current training, but it's still technically wrong.)

### 3. **R_blend Not Always Built in Validation**

Lines 1807-1808: `R_blend` only built if `cov_fact_angle` and `cov_fact_range` exist. But with `USE_STRUCTURED_R=True`, those are `None`!

**Current workaround**: loss.py uses `R_pred` directly when `R_blend` missing.

**Proper fix**: Always build `R_blend` from `R_pred` in structural R mode.

---

## Does the Code Need Major Changes?

### Short Answer: **Not for basic functionality, but YES for maintainability**

| Aspect | Status |
|--------|--------|
| **Core architecture** | ✅ Sound (slot head + structural R is correct) |
| **Learning works** | ✅ After today's fixes (smoke test passes) |
| **Wideband-ready** | ❌ Adding F dimension to current spaghetti = disaster |
| **Code quality** | 🔴 Poor (3420-line methods, 471 branches) |
| **Maintainability** | 🔴 Critical (any new feature = 50+ line changes) |

### What Will Break When Adding Wideband

When you add the F (frequency) dimension, you'll need to modify:
- `dataset.py`: Load `y[L,F,M,2]` instead of `y[L,M,2]`
- `model.py`: Add frequency pooling layer
- `train.py`: Handle new tensor shapes in **~15 places** (every `_unpack_batch`, every validation loop, every forward call)
- `loss.py`: Potentially handle per-tone losses

With the current code structure, that's **100+ line changes across 40+ locations**. High risk of bugs.

### Recommended Approach

**Option A (Pragmatic)**: Do Phase 1 cleanup first, then add wideband
- Estimated time: 3 days cleanup + 2 days wideband
- Risk: Medium

**Option B (Clean Slate)**: Refactor now, then add wideband
- Estimated time: 1 week refactor + 1 day wideband
- Risk: Low, but more upfront work

**Option C (Ship It)**: Skip refactoring, add wideband carefully
- Estimated time: 3-4 days wideband (many edge cases)
- Risk: High, code becomes unmaintainable

---

## Immediate Action Items (if you want to proceed with current codebase)

### 1. Enable Soft Matching Warmup (URGENT — fixes epoch 4 jump)

```python
# In ris_pytorch_pipeline/configs.py, line 569:
self.AUX_MATCH_SOFT_EPOCHS = 5  # Was: 0
```

### 2. Verify Training on Goose

After pulling latest + above fix:
```bash
cd /home/tahit/ris/MainMusic
./cleanup_for_fresh_training.sh  # Clear old checkpoints
python -m ris_pytorch_pipeline.ris_pipeline train --epochs 15 --use_shards
```

**Expected**: aux_φ_rmse should decrease smoothly (no jumps to 48°).

### 3. Run Smoke Test on Goose (Sanity Check)

```bash
python diagnose_pipeline_smoke.py --device cuda --steps 150 --batch_size 32
```

**Expected**: All 4 tests PASS.

---

## My Recommendation

**For your PhD timeline**: Do **Phase 1 cleanup only** (delete dead code, extract data loading, simplify validation). This is:
- Low risk (mostly deletions)
- High ROI (removes 800 lines of confusion)
- Sets you up for wideband without making train.py a 5000-line monster

Then proceed with wideband upgrade on the cleaner codebase.

Want me to prepare the Phase 1 cleanup as a separate branch? I can do it systematically (one commit per cleanup category) so you can review/test incrementally.