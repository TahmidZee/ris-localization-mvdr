# Constant Prediction Trap: Full Diagnosis & Status

**Date:** 2026-02-11  
**Branch:** `refactor/phase1-cleanup`  
**Status:** ⚠️ UNRESOLVED — model still cannot learn geometry in overfit test

---

## 1. The Problem

The model predicts **near-constant values** for azimuth angle (φ), elevation (θ), and range (r) across all samples. Instead of learning input-dependent geometry, each slot converges to a fixed location and stays there.

### Evidence (Latest Logs)

**Run 1** — Full training (100K samples, 38 epochs):
```
Slot 0: φ= -34.3°±2.06°   θ= -2.1°±0.88°   r=5.20m±0.067m
Slot 1: φ=  -0.1°±3.07°   θ= -1.6°±0.88°   r=5.26m±0.066m
Slot 2: φ= +22.9°±2.64°   θ= -1.7°±0.88°   r=5.21m±0.066m
Slot 3: φ= +40.4°±1.69°   θ= -2.0°±0.88°   r=5.25m±0.066m
Slot 4: φ= +54.2°±0.57°   θ= -2.1°±0.88°   r=5.21m±0.066m
⚠️ LOW φ VARIANCE (2.00°), LOW θ VARIANCE (0.88°), LOW r VARIANCE (0.066m)
aux_φ_rmse = 24.67°  (should be < 5° if learning)
```

Slots have spread apart (thanks to `slot_output_bias`), but predictions are **sample-independent** — the model ignores its input entirely.

**Run 2** — Overfit test (200 samples, 200 epochs) after `tanh` → `atan` + init fixes:
```
Slot 0–4: φ= +60.0°±0.00°  (ALL slots locked to max value!)
aux_φ_rmse = 72.04°  (catastrophic regression)
```

The model saturated all predictions to the maximum φ output, confirming the activation function was the bottleneck.

---

## 2. Root Cause Chain (from first principles)

The constant prediction trap is caused by a **chain of interacting failures**, not one single bug:

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. SYMMETRIC INIT  →  All slots start at same/similar predictions   │
│    ↓                                                                 │
│ 2. PERMUTATION MATCHING  →  Different slots match different GT per  │
│    sample; average gradient → dataset mean for each slot             │
│    ↓                                                                 │
│ 3. GRADIENT SATURATION  →  tanh activation kills gradients when     │
│    predictions drift to extremes (tanh'(2)=0.07, tanh'(5)=3.6e-5)  │
│    ↓                                                                 │
│ 4. SIGNAL DROWNING  →  NMSE gradient (256×256 Jacobian) overwhelms  │
│    geometry gradient (5-dim), especially under CLIP_NORM             │
│    ↓                                                                 │
│ 5. BIAS DOMINANCE  →  slot_output_bias receives all gradient while  │
│    backbone/MLP weights receive negligible signal                    │
│    ↓                                                                 │
│ 6. STABLE EQUILIBRIUM  →  All feedback loops reinforce the constant │
│    prediction; no gradient force can escape this basin               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Fixes Applied (Rounds 1–18)

### Round 1–4: Loss & Initialization (2026-02-05 → 02-06)
| Fix | What | Why |
|-----|------|-----|
| `GEOM_ONLY_EPOCHS` 5→0 | NMSE from epoch 0 | Provide covariance gradient immediately |
| `MASK_LOSS_WARMUP` 10→0 | Mask BCE from epoch 0 | DETR "no-object" signal for unmatched slots |
| `mask_logit` bias -2→0 | Masks start at 0.5 not 0.12 | R_pred gets signal from day 1 |
| `loss_nmse_pred` condition | Fixed dead code | R_pred NMSE was always 0 |
| `_slot_diversity_loss` | Fixed sign bug | Was NEGATIVE → optimizer maximized spread |
| `lam_cov` 0.3→1.0 | Full NMSE weight | Was only 0.02 effective |

### Round 5–6: Gradient Clipping (2026-02-07)
| Fix | What | Why |
|-----|------|-----|
| `GEOM_ONLY_EPOCHS` 0→3 | Re-enable geometry warmup | NMSE gradient (norm=43) drowns aux at CLIP_NORM=1 |
| `CLIP_NORM` 1.0→5.0 | Relax clipping | Head effective LR was 3.4e-5 (43× reduction) |
| Zero `lam_cov_pred` during GEOM | Remove residual NMSE | Even 0.05 weight reinforces equilibrium |

### Round 7: Per-Slot Output Bias (2026-02-07)
| Fix | What | Why |
|-----|------|-----|
| `slot_output_bias` | Learnable [K,5] bias | Breaks symmetric equilibrium by construction |
| Init: φ ∈ {-48°, -24°, 0°, +24°, +48°} | Spread across FOV | Gives sorted loss a stable sort order |

### Round 8–11: Training Schedule (2026-02-08)
| Fix | What | Why |
|-----|------|-----|
| `GEOM_ONLY_EPOCHS` 3→8 | Longer geometry warmup | Slots need time to specialize |
| Surrogate weights rebalanced | aux dominates score | Prevent early stopping on NMSE jump |
| NMSE warmup starts after GEOM_ONLY | Prevent step-function | Smooth transition |

### Round 12–16: Init & Signal Path (2026-02-10)
| Fix | What | Why |
|-----|------|-----|
| `_slot_last_linear` init 0→0.01 | Fix zero-init gradient block | `W=0` means `d(out)/d(input)=0` always |
| `BIAS_LR_FINAL_MULTIPLIER` 1.0→0.1 | Keep bias slow | 10× LR jump at epoch 5 was undoing learning |
| Conditional NMSE ramp | Only ramp if φ < threshold | Prevent NMSE from drowning immature geometry |

### Round 17: Training Schedule Tuning (2026-02-11)
| Fix | What | Why |
|-----|------|-----|
| `GEOM_ONLY_EPOCHS` 2→10 | 10 epochs geometry only | Log showed φ=17.68° at ep3 → 21° with NMSE |
| `lam_cov` 1.0→0.3 | Cap NMSE low | NMSE overwhelms geometry at full weight |
| `lam_aux` restored to 1.0 | Perm-invariant aux back on | Sorted-only wasn't enough |
| `LAM_AUX_SORTED` 1.5→2.0 | Stronger sorted loss | Primary symmetry-breaking signal |
| `EMA_EVAL_WARMUP_EPOCHS`=999 | Disable EMA | See raw model during debugging |

### Round 18: Activation Function Fix (2026-02-12, latest)
| Fix | What | Why |
|-----|------|-----|
| `tanh` → `(2/π)·atan` for φ,θ | Replace activation | tanh'(2)=0.07 → atan'(2)=0.127 (1.8×); tanh'(5)=3.6e-5 → atan'(5)=0.024 (667×) |
| `slot_output_bias` init: `atanh` → `tan` | Match new activation | Inverse function must match |
| Bias spread: ±48° → ±30° | Reduce bias magnitude | Outer slots had dead gradients at tanh(1.1) |
| `_slot_last_linear` std: 0.10→0.05 | Scale for atan regime | MLP output ~0.8 + bias ~1.0 → total ~1.8, atan'(1.8)=0.15 |
| `lam_cov=0.0` in overfit test | Pure geometry test | Isolate geometry learning from NMSE interference |

### Round 19: Overfit Test Correctness Fixes (2026-02-12)
| Fix | What | Why |
|-----|------|-----|
| `LR` → `LR_INIT` in `overfit_test.py` | Correct LR override key | `Trainer` reads `mdl_cfg.LR_INIT`; setting `mdl_cfg.LR` had **no effect**, making prior overfit results potentially underpowered |
| Disable `lam_cov_pred` in overfit | `mdl_cfg.LAM_COV_PRED=0.0` and `cfg.LAM_COV_PRED=0.0` | Even with `lam_cov=0.0`, `lam_cov_pred` adds an NMSE gradient path that can re-create the constant-prediction equilibrium |
| Full-batch + no regularization for memorization | `BATCH_SIZE=200`, `WEIGHT_DECAY=0.0`, `DROPOUT=0.0`, `USE_AMP=False` | Make the overfit test a clean “can we memorize?” diagnostic (remove stochasticity/regularization confounds) |
| Never start NMSE ramp in overfit | `NMSE_RAMP_AUX_PHI_THRESHOLD=0.0` | Prevent the structured-cov schedule from re-enabling `lam_cov_pred` during the run |

---

## 4. Current Configuration (After All Fixes)

### Model Architecture
```
Input: y[B,L,M,2], H_full[B,M,N,2], codes[B,L,N,2], snr_db[B]
  │
  ├── Conv1d tokenization → [B,D,L] → Transformer (8.6M params) → [B,L,D]
  ├── H_full → Conv2d stack → [B,D/2]
  ├── SNR → MLP embed → [B,snr_dim]
  │
  └── Fusion → [B,D]
        │
        ├── Slot queries (K=5, init_std=1.0) × 3-round cross-attention → [B,K,D]
        │     └── slot_head MLP → [B,K,5] + slot_output_bias [K,5]
        │           │
        │           ├── φ = (2/π)·atan(raw) × 60°    ← NEW (was tanh)
        │           ├── θ = (2/π)·atan(raw) × 30°    ← NEW (was tanh)
        │           ├── r = R_MIN + (R_MAX-R_MIN) × softplus(raw)/(1+softplus(raw))
        │           ├── power = softplus(raw)
        │           └── mask = sigmoid(raw)
        │
        └── build_structured_R(φ,θ,r, power×mask) → R_pred [B,N,N]
```

### Loss Function
```
total = lam_cov    * NMSE(R_eff_pred, R_eff_true)      # 0.3 (capped)
      + lam_aux    * perm_invariant_aux(φ,θ,r)          # 1.0
      + lam_sorted * sorted_canonical_aux(φ,θ,r)        # 2.0  (symmetry breaker)
      + lam_div    * margin_diversity(φ,θ,r)             # 0.5  (slot repulsion)
      + lam_pred   * NMSE(R_pred, R_eff_true)            # 0.05 (aux gradient path)
      + mask_bce   * perm_invariant_mask_bce              # 0.2  ("no object" for unmatched)
      + mask_count * (sum(mask) - K_true)²                # 0.3
      + mask_bin   * mask*(1-mask)                        # 0.1
```

### Training Schedule
```
Epochs  1–10:  lam_cov=0.0 (geometry only)
Epoch  11+:    NMSE ramp starts IF best aux_φ_rmse < 22.0°
               Ramp over 5 epochs: 0 → 0.3
Epoch  16+:    lam_cov = 0.3 (capped)
```

### Key Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| `CLIP_NORM` | 5.0 | Was 1.0 (43× gradient suppression) |
| `SLOT_QUERY_INIT_STD` | 1.0 | Was 3.0 (drowned cross-attention) |
| `_slot_last_linear` init std | 0.05 | v1: 0.01 (dead), v2: 0.10 (tanh saturation) |
| `BIAS_LR_FINAL_MULTIPLIER` | 0.1 | Keeps bias slow entire run |
| `EMA_EVAL_WARMUP_EPOCHS` | 999 | Disabled for debugging |
| Activation | `(2/π)·atan` | Was `tanh` (saturates exponentially) |
| Bias spread | ±30° | Was ±48° |

---

## 5. What We Know

### ✅ Confirmed Working
- Gradient path from loss → R_pred → geometry → slot head → backbone is connected (`GRADPATH` probe shows non-zero)
- Slots DO spread apart thanks to `slot_output_bias` (different φ per slot)
- Loss is finite and decreasing
- Steering vectors are consistent across model/loss/physics modules
- Permutation-invariant matching is correct
- Sorted canonical loss provides deterministic gradients

### ❌ Confirmed Broken
- **Slots do not respond to input**: cross-sample variance is < 3° (should be ~20° for ±60° FOV)
- **Overfit test fails**: 200 samples, 200 epochs, cannot memorize → architecture issue, not optimization
- **θ and r also constant**: not just a φ problem
- **All gradient fixes have NOT solved the core issue**: 18 rounds of fixes have improved symptoms but not the fundamental problem

### 🔍 Key Observation
The model behaves as if `slot_output_bias` is the ONLY thing determining predictions. The backbone, cross-attention, and MLP contribute negligible corrections. This suggests the **information bottleneck is in the slot mechanism** (query-based cross-attention), not just initialization or activation.

---

## 6. Hypotheses for Next Steps

### Hypothesis A: Slot Cross-Attention Is the Bottleneck
The slot query mechanism (3 rounds of cross-attention) may not effectively route input information to individual slots. If queries are too similar or the attention is too diffuse, all slots receive the same pooled global feature, making input-dependent prediction impossible.

**Test:** Log per-slot attention entropy. If attention is near-uniform across tokens, the slots are not specializing.

### Hypothesis B: Decoupled Pipeline
Separate the problem into two stages:
1. **Stage 1 — Angle/geometry prediction**: A simpler head (e.g., direct MLP or 1D conv on transformer features) predicts a fixed-size output of K_MAX angles. No slot mechanism needed.
2. **Stage 2 — Source counting**: Determine which of the K_MAX predictions are "active" (mask).

**Rationale:** The slot mechanism was designed for set prediction (like DETR for object detection), but our problem may not need it. The input is a global covariance feature, not a spatial image with objects at different locations. A direct regression might be simpler and more effective.

### Hypothesis C: Feature Quality Issue
The backbone transformer may not be producing features that carry geometry information. If the transformer bottleneck erases angular information, no amount of head tuning will help.

**Test:** Freeze a randomly initialized backbone and train ONLY the slot head. If the overfit test passes → backbone features carry information. If not → backbone is the problem.

### Hypothesis D: Loss Landscape Pathology
Even with `atan`, the loss landscape for geometry prediction may have a flat basin around the dataset mean. The permutation-invariant matching creates a combinatorial structure that makes the gradient landscape non-smooth.

**Test:** Replace permutation-invariant matching with a fixed assignment (sort GT by φ, assign to slots in order) for the overfit test. This removes all matching ambiguity and should make the overfit test trivially solvable.

---

## 7. The Decoupling Question

The user asked: *"Should we decouple the pipeline and let angle prediction and slot prediction be two separate tasks?"*

### Arguments For Decoupling
1. The slot mechanism adds complexity (cross-attention, permutation matching) that may not be needed
2. DETR-style slot heads work well for **spatially grounded** detection (images), but our input is a **global covariance feature** — there's no spatial structure for attention to exploit
3. A simple MLP head predicting K_MAX × 3 outputs (sorted by φ) eliminates the entire permutation problem
4. The overfit test failure after 18 rounds of fixes suggests the problem is structural, not just tuning

### Arguments Against Decoupling
1. Slot-based attention is more principled for set prediction (order invariance)
2. The slot mechanism theoretically allows attending to different parts of the feature for different sources
3. Decoupling requires significant refactoring effort
4. We haven't yet tested the latest `atan` fix in a clean overfit test

### Recommendation
**Run the overfit test with the latest `atan` fixes first.** If it still fails (φ RMSE > 10° after 100 epochs on 200 samples), then decoupling is strongly indicated. The overfit test on 200 samples should be decisive — if the architecture can't memorize 200 samples, no amount of training schedule tuning will help.

---

## 8. Immediate Action Items

1. **Run overfit test** with latest `atan` + `lam_cov=0.0` changes:
   ```bash
   cd ~/ris/MainMusic && python overfit_test.py
   ```
   Look for: φ RMSE dropping below 10° within 50 epochs

2. **If overfit test fails** → try Hypothesis D first (fixed sorted assignment, remove permutation matching entirely for the overfit test). This is a 5-line code change.

3. **If that also fails** → implement Hypothesis C (freeze backbone, train head only). If head can't learn with frozen random backbone → problem is definitely architectural.

4. **If all above fail** → implement decoupled pipeline (Hypothesis B): direct MLP regression for K_MAX sorted angles, separate mask head.

---

## 9. Dead Parameters (Pending Cleanup)

~774K parameters (7.3% of 10.5M) are created but never contribute to the gradient. See `REFACTOR_PROGRESS.md` §6 for details. These include `phi_logits`, `theta_logits`, `antidiag_pool`, `fusion_with_antidiag`, and associated LayerNorms. Cleanup is deferred to after the constant prediction trap is resolved.

---

## 10. File Reference

| File | Role | Key Changes |
|------|------|-------------|
| `ris_pytorch_pipeline/model.py` | Architecture | `atan` activation, `slot_output_bias`, init std |
| `ris_pytorch_pipeline/loss.py` | Loss functions | Diversity, sorted canonical, perm-invariant aux |
| `ris_pytorch_pipeline/configs.py` | Hyperparameters | GEOM_ONLY, CLIP_NORM, NMSE ramp, bias LR |
| `ris_pytorch_pipeline/train.py` | Training loop | Conditional NMSE ramp, bias param group |
| `overfit_test.py` | Diagnostic | 200-sample memorization test, pure geometry |
| `FIX_SYMMETRIC_EQUILIBRIUM_2026-02-05.md` | History | Detailed Rounds 1–11 |
| `IMPLEMENTATION_CHANGELOG.md` | History | Full project changelog (Rounds 12–18) |
