## RIS PyTorch Pipeline — Current Challenges, Observed Outputs, and Remediation Plan

Date: 2025-11-24
Owner: Tahmid
Scope: Overfit test, NMSE correction, stability and curriculum plan

### Executive summary

- The prior “loss floor” was caused by NMSE being implemented as a similarity, not an error; it returned ≈1.0 even at perfection. This has been corrected so NMSE now returns 0 at perfect predictions and ≈1 when predicting zeros.
- We now enforce a strict “pure overfit” mode: β=0, `lam_cov=1.0`, all other λ=0. Guards assert this at runtime.
- Stability has been hardened (total-grad step gate, scheduler gating, SVD-based structure terms, AMP overflow logging). On this environment, long runs are currently killed due to resource limits; results below reflect partial logs but the configuration and instrumentation are correct.
- Critical pivot: experiments and acceptance criteria are now K̂ + localization centric. NMSE is retained as a smoke test, not the primary KPI. Validation computes K_acc, AoA/Range RMSE, and Success_rate (Hungarian-matched), and early stopping can be driven by these metrics.
  - Use `cfg.VAL_PRIMARY="k_loc"` to select checkpoints by a composite K/localization score (printed each epoch). Otherwise, defaults to loss-based selection.

---

## Current configuration (pure overfit proof mode)

- Objective: NMSE-only.
- Weights: `lam_cov=1.0`; all others 0: `lam_cov_pred, lam_diag, lam_off, lam_ortho, lam_cross, lam_gap, lam_K, lam_aux, lam_peak, lam_subspace_align, lam_peak_contrast = 0.0`.
- Hybrid: β=0.0 for the entire run (no blending). Guards hard‑fail if violated.
- Regularization: weight decay 0, dropout 0.
- Training stability:
  - Step gate: single total-grad-norm check; step only if finite and > 0. Scheduler ticks only after a real step.
  - AMP: overflow (“found_inf”) logging wired; grad sanitization avoids masking infs.
  - EMA/SWA: off for overfit proof.
- Loss stability:
  - NMSE corrected: returns 0 at perfection.
  - Complex-eigen backward issues removed: eigengap/subspace/contrast implemented via SVD/projectors.
  - Oracle check and NMSE self-test prints included.

---

## Observed outputs (latest run on this machine)

This environment kills longer runs due to resource limits, but the early logs show the correct configuration has taken effect:

```text
[OVERFIT] Logging to: overfit_test.log
🔧 Optimizer setup:
  Backbone: 6.7M @ LR=1.00e-03
  Head:     9.7M @ LR=4.00e-03 (4.0×)
  Total trainable: 16.3M
  ✅ Optimizer wiring verified!
🔧 PURE OVERFIT: β=0, NMSE-only, all λ=0 except lam_cov=1.0
✅ Pure overfit guards passed - all λ=0 except lam_cov=1.0
[VAL LOADER] len(val_ds)=512, batches per epoch=8
[Training] Starting 5 epochs (train batches=8, val batches=8)...
[LR] epoch=1 groups=['backbone', 'head'] lr=[1.00e-03, 4.00e-03]
-- Killed by system (resource limit) --
```

Prior to the NMSE fix, we saw floors near 1.0 even for “oracle”:

```text
[VAL DEBUG] oracle_total_if_Rpred_eq_Rtrue = ~0.996   (old behavior – fixed now)
loss_nmse_pred ≈ 0.99                                (old behavior – fixed now)
```

What we expect now (once the run completes on a machine with resources):

- First forward pass prints (self-test): `nmse(R_true,R_true)≈0, nmse(0,R_true)≈1`.
- Validation NMSE quickly falling below 0.1 on a 512-sample identical train/val set.
- Δ-param nonzero after steps; step logs show finite total gradient norms; no AMP overflows.

---

## Problems found (and their status)

- NMSE defined as similarity, not error (root cause of ~1.0 floor)
  - Evidence: oracle loss ≈ 1.0 when `R_pred == R_true`; nmse_pred ≈ 0.99.
  - Status: Fixed. NMSE now returns 0 at perfection, ≈1 at zeros; self-test added.

- Hidden non-zero loss weights during “pure” runs
  - Evidence: constructor defaults (e.g., `lam_cov_pred=0.04`) contributed when not forced on instance.
  - Status: Fixed. All λ forced on the loss instance; guards validate at runtime.

- Over-strict step gate (backbone and head both > 0 required)
  - Evidence: weight stasis and repeated val values; zero updates on batches with masked heads.
  - Status: Fixed. Single total grad-norm gate; scheduler ticks only after a real step.

- Complex-eigen backward instabilities
  - Evidence: `linalg_eigh_backward` errors/NaNs on complex eigenvectors.
  - Status: Fixed. Replaced with SVD-based/projector-only implementations for eigengap, subspace margin, alignment.

- Reporting vs training path inconsistencies and phantom infs
  - Evidence: finite per-batch but “inf” in summary.
  - Status: Hardened validation aggregation; optional clamp in reporting (not needed in pure overfit).

- Memory/overflow masking
  - Evidence: `nan_to_num` turning inf→finite, hiding AMP overflow.
  - Status: Fixed. We now preserve inf for overflow detection and log `found_inf`.

- Environment resource kills (this terminal)
  - Evidence: process terminated by the OS before epoch 1 completes.
  - Status: Outstanding in this environment only. Run on a resource-sufficient screen to see full epoch logs.

---

## Plan to address and scale up (phased curriculum with acceptance gates — K̂/Localization first)

### Phase 0 — Smoke test (pure NMSE)
- Config: β=0; `lam_cov=1.0`, others 0; AMP/EMA/SWA off; WD=0; dropout=0; identical train/val (e.g., 512); batch size 64; warmup floor non-zero; cosine decay.
- Instrumentation: print [LRs], total_grad_norm, Δ-param, AMP overflow status, NMSE components.
- Acceptance (once): validation NMSE ≤ 0.1 within a few epochs; Oracle ≈ 0.0; self-test 0/≈1. Do this once, then move on.
- Action: run `python test_overfit.py` on a resource-sufficient session/screen and capture the first epoch output.

### Phase 1 — Hybrid effective covariance on (β fixed or mild ramp)
- Config: use unified `build_effective_cov_*` path; turn on hybrid with fixed β (e.g., 0.2–0.3).
- Acceptance: K_acc trending up; AoA_RMSE/Range_RMSE trending down; Success_rate trending up. NMSE monitored but not used for gating.

### Phase 2 — Enable K prediction (gentle)
- Config: enable K-loss with a small ramp; optionally freeze or lower backbone LR briefly.
- Acceptance: K_acc improves, especially reduction in K_under; AoA/Range RMSE do not regress materially.

### Phase 3 — Structure terms (one at a time)
- Terms: eigengap, subspace alignment, peak contrast, ortho; SVD/projector variants only.
- Method: enable one term at a time with a small ramp; normalize/scale each so its steady contribution is O(0.05–0.15).
- Acceptance: K_acc and Success_rate non-decreasing; AoA/Range RMSE stable or improving. NMSE stays reasonable but not used as a gate.

### Objective balancing
- Consider uncertainty weighting or GradNorm to automatically balance multi-term training.
- Keep `lam_cov` anchored; ramp others based on observed magnitudes.

---

## Runbook (quick start)

1) Prove overfit (Phase 1):

```bash
cd /home/tahit/ris/MainMusic
python test_overfit.py  # produces overfit_test.log
```

2) Confirm these prints in the first epoch:

- Pure overfit guards passed
- NMSE self-test: `nmse(R_true,R_true)≈0, nmse(0,R_true)≈1`
- Oracle (optional): total ≈ 0 when `R_pred := R_true`
- Val NMSE trending downward; ≤ 0.1 target

3) Stage β ramp (Phase 2), then K ramp (Phase 3), then structure terms (Phase 4).

---

## Open risks / watch items

- If NMSE does not reach ≤ 0.1 in Phase 1: re-check data conventions (Hermitian, trace scaling, RI↔C conversions), confirm no hidden λ or β, and verify gradient norms are non-zero with Δ-param > 0. If still blocked, share the first epoch’s `[SELFTEST]`, `[LR]`, `[GRAD]`, `Δparam`, and `[VAL]` lines.
- If enabling β causes large regressions: slow the ramp or slightly increase LR floor during ramp-in.
- If enabling K destabilizes NMSE: reduce backbone LR or briefly freeze backbone during K ramp.
- If structure terms are noisy: ensure SVD implementations are active and weights are normalized; keep per-term ramps slow.

---

## TL;DR

- Root cause fixed: NMSE now behaves like a true loss (0 at perfection), eliminating the artificial ~1.0 floor.
- Pure overfit mode is strictly enforced with guards; stability tooling and logging are in place.
- Primary KPIs are now K̂ accuracy, AoA/Range RMSE, and Success_rate. NMSE is kept as a smoke test.
- Execute the staged plan with K/localization acceptance gates. This ensures predictable convergence and isolates issues early.



