# Phase 1 Refactoring Progress
**Branch:** `refactor/phase1-cleanup`  
**Started:** 2026-02-05  
**Goal:** Remove ~800 lines of dead code before wideband upgrade

---

## Completed ✅

### 1. loss.py Dead Code Removal (Commit `016a4a3`)
- ❌ Deleted `_sorted_aux_loss()` (65 lines) — USE_SORTED_MATCHING=False
- ❌ Deleted `_eigengap_hinge()` (28 lines) — lam_gap always 0
- ❌ Deleted `_subspace_margin_regularizer()` (23 lines) — lam_margin always 0
- ❌ Removed sorted matching conditionals in `forward()` and `debug_terms()`
- ❌ Removed gap/margin warning code

**Result**: `1338 → 1186 lines (-152, -11%)`  
**Status**: Smoke tests PASS ✅

---

## In Progress 🔄

### 2. train.py Curriculum Code Removal
**Target**: Remove 3-phase curriculum (disabled by USE_3_PHASE_CURRICULUM=False)

Methods to delete:
- `_apply_phase_weights()` (73 lines)
- `_tau_schedule()` (8 lines)
- `_apply_curriculum()` (13 lines)
- `_update_structure_loss_weights()` (45 lines)

Call sites to remove (in `fit()`):
- Lines 2795-2797: `if getattr(mdl_cfg, 'USE_3_PHASE_CURRICULUM'...`
- Lines 2863-2864: Same check

**Expected**: ~140 lines removed

---

## Remaining TODOs

### 3. Debug Scaffolding Removal (~100 lines)
- Remove `epoch_dbg`, `loopcheck_dbg`, `_should_log_batch` checks
- Convert to Python `logging` module
- Remove "CRITICAL FIX" / "Expert fix" comments (43 total)

### 4. Extract Data Loading (~250 lines)
- Move `_build_loaders_cpu_io`, `_build_loaders_gpu_cache`, `_aggregate_cpu_then_gpu` to new `data_loader.py`
- Simplify Trainer to just call `data_loader.build_loaders(...)`

### 5. Simplify Validation (~400 lines)
- Extract common `_validate_batch()` helper
- Merge overlapping logic in 3 validation methods
- Remove duplicate batch loops

### 6. model.py Dead Parameters & Unused Heads (~774K params, ~200 lines)
⚠️ **HIGH PRIORITY** — These are dead parameters receiving no gradient but consuming memory and weight decay.

When `USE_STRUCTURED_R=True` + `USE_SLOT_HEAD=True` (current default), the following are dead:

| Module | Params | Why Dead |
|--------|--------|----------|
| `cov_ln` | 1,024 | `feats_cov` computed but `cf_ang=None` |
| `heads_ln` | 1,024 | `feats_final` computed but soft-argmax gated off |
| `phi_logits` | 156,465 | Soft-argmax gated off |
| `theta_logits` | 156,465 | Soft-argmax gated off |
| `antidiag_pool.proj` | 130,944 | `cf_ang=None` → antidiag path never entered |
| `fusion_with_antidiag` | 328,192 | `cf_ang=None` → never called |
| **Total** | **~774K (7.3%)** | |

**Cleanup tasks:**
- Remove `cov_ln`, `heads_ln` when structural-R is default
- Remove `phi_logits`, `theta_logits`, `soft_argmax`, `logits_gg` (soft-argmax path)
- Remove `antidiag_pool`, `fusion_with_antidiag` (AntiDiagPool path)
- Remove `cov_fact_angle`, `cov_fact_range` stubs (already None)
- Remove `_apply_psd_parameterization` method (legacy path only)
- Remove `_whiten_covariance` method (only used in dead antidiag path)
- Guard: keep behind `if not self.use_structured_R:` for ablation compatibility

### 7. Final Testing
- Run smoke tests
- Run 10-epoch training
- Verify metrics match pre-refactor

---

## Target Line Counts

| File | Before | After Phase 1 | Reduction |
|------|--------|---------------|-----------|
| train.py | 3,420 | ~2,600 | -820 (-24%) |
| loss.py | 1,338 | ~1,050 | -288 (-22%) |
| model.py | 1,111 | ~750 | -361 (-32%) |
| **Total** | **5,869** | **~4,400** | **-1,469 (-25%)** |

> **Note:** model.py reduction increased due to dead parameter cleanup (~774K params, ~150 lines).

---

## Testing Strategy

After each commit:
1. `python -m py_compile <file>` — verify compilation
2. `python diagnose_pipeline_smoke.py --device cpu --steps 30` — verify smoke tests
3. Push to GitHub

After all Phase 1 commits:
1. Run 10-epoch training on goose
2. Compare metrics to baseline (before refactor)
3. If metrics match: proceed to Phase 2 (structural refactoring)

---

## Commits So Far

| Commit | Description | Lines Removed |
|--------|-------------|---------------|
| `016a4a3` | Remove dead loss terms | -152 |
| (next) | Remove 3-phase curriculum | ~-140 |
| (next) | Remove debug scaffolding | ~-100 |
| (next) | Extract data loading | ~0 (move, not delete) |
| (next) | Simplify validation | ~-400 |
| (next) | Clean up model.py | ~-200 |

---

## Risk Assessment

- **Low Risk**: Dead code removal (already done)
- **Low Risk**: Curriculum removal (disabled by default)
- **Medium Risk**: Debug removal (need to preserve critical logging)
- **Medium Risk**: Data loading extraction (changes import paths)
- **High Risk**: Validation refactor (core training logic)

Strategy: Test after each commit, revert if broken.
