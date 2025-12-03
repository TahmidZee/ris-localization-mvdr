## Implementation changes (done) and next steps (actionable)

Date: 2025-11-25
Owner: Tahmid

### What I changed (today + recent critical fixes)

- Covariance alignment inside the loss (train == eval == infer)
  - In `ris_pytorch_pipeline/loss.py`:
    - Added `\_trace_norm_torch` and `\_shrink_torch` (torch-native, per-sample SNR, differentiable).
    - The main NMSE now compares the same object that inference/MUSIC sees:
      - `R_true_eff = trace_norm(R_true) → shrink(R_true_eff, snr_db)`
      - `R_pred_eff = trace_norm(R_blend or R_hat) → shrink(R_pred_eff, snr_db)`
      - `loss_nmse = NMSE(R_pred_eff, R_true_eff)`
    - Auxiliary NMSE on raw `R_pred` (if enabled) also runs on trace-normalized + shrunk prediction for consistency.
    - Kept the NMSE self-test print (should show `≈0` at perfection and `≈1` for zero-predictions).

- Inference uses the same hybrid path as train‑eval
  - In `ris_pytorch_pipeline/infer.py`:
    - `angle_pipeline(...)` now receives `y_snapshots`, `H_snapshots`, `codes_snapshots`, and `blend_beta=cfg.HYBRID_COV_BETA`. This makes inference use the same hybrid covariance logic you evaluate during training.

- K‑head SNR handling is vectorized (per-sample, not just sample 0)
  - In `ris_pytorch_pipeline/model.py`:
    - Replaced the “use first sample’s SNR” shrink with a vectorized per‑sample shrinkage for building the spectral features used by the K‑head.

- Previously completed, still important
  - NMSE fixed to be a true “error” (0 at perfection). Self-test added.
  - Pure overfit mode enforced in the harness (`test_overfit.py`): `β=0`, `lam_cov=1.0`, all other λ=0 with hard guards.
  - Step gate simplified to a single total-grad norm + scheduler ticks only after a real step.
  - Eigengap/subspace losses made stable (SVD/projector based) to avoid complex-eigen backprop issues.
  - AMP overflow logging, Δ‑param probe, LR prints, and safer validation aggregation in place.

### Why this matters

- The loss now optimizes the exact covariance that MUSIC consumes at inference (trace-normalized + SNR‑aware shrink + optional hybrid), removing train/infer mismatch.
- The NMSE “floor” caused by the old definition is gone; with pure overfit you should see validation loss plunge well below 0.1.
- K‑head spectral features are consistent across the batch when SNR varies.

### What outputs to expect

- First pass self-test in logs:
  - `[SELFTEST] nmse(R_true,R_true)≈0, nmse(0,R_true)≈1`
- Pure overfit (β=0, λ only on cov) should reach val NMSE < 0.1 in a few epochs on the 512‑sample identical train/val set (smoke test only).
- Δ‑param prints become non‑zero on successful steps; total grad norm finite and > 0; no AMP overflow.

### Next steps (actionable checklist)

- Run the pure overfit probe on a resource-sufficient session (one-time smoke test)
  - Command: run `test_overfit.py` (already configured). Capture `overfit_test.log`.
  - Acceptance: val NMSE ≤ 0.1. If not, treat as bug; share the first epoch’s `[SELFTEST]`, `[LR]`, `[GRAD]`, `Δparam`, `[VAL]` lines.

- Switch validation to inference-like metrics (K̂ + localization)
  - Compute per-epoch: `K_acc`, `K_under`, `K_over`, `AoA_RMSE`, `Range_RMSE`, `Success_rate` (Hungarian).
  - Drive early stopping by a composite K/loc score (or directly by Success_rate + K_acc), not by NMSE.
  - Keep NMSE as a sanity log only.
  - Tip: set `cfg.VAL_PRIMARY="k_loc"` to enable metric-driven checkpointing/early-stopping; default remains loss-based.

- Stage hybrid blending (β fixed or mild ramp)
  - Fix β in [0.2, 0.3] for a baseline; optionally mild ramp later.
  - Acceptance: `K_acc` increases and `Success_rate` improves; `AoA/Range RMSE` decreases or remains stable.

- Add K prediction gently
  - Turn on K‑loss with a small ramp. Optionally freeze the backbone or use a lower backbone LR for a few epochs.
  - Acceptance: `K_acc` improves, especially reduced `K_under`; `AoA/Range RMSE` not degraded materially.

- Add structure terms one at a time with small ramps
  - Eigengap → Subspace align → Peak contrast → Ortho (SVD/projector implementations already wired).
  - Normalize weights so each term contributes O(0.05–0.15) to the total.
  - Consider uncertainty weighting or GradNorm once multiple terms are on.

- Optional refactor (recommended): single shared builder for “effective covariance”
  - Create `covariance_utils.py` with `build_effective_cov_for_music(...)` that performs:
    - factor/full R → hermitize → trace-normalize → (optional) hybrid with snapshots → diag load → shrink (SNR) → return R
  - Replace duplicated logic in `loss.py`, `angle_pipeline.py`, and `infer.py` with that function to eliminate drift.

### Open risks / things to watch

- Environment in this terminal kills long runs (resource limits). Use your high‑resource screen to confirm convergence.
- Ensure your inference callsites can pass snapshots/codes if you want hybrid at inference (done inside `infer.py`, but the outer application must supply those arrays).
- Keep an eye on SNR provenance (per-sample alignment). If SNR labels are synthetic/noisy, shrinkage may need retuning.

### Files changed in this pass

- `ris_pytorch_pipeline/loss.py`: added `\_trace_norm_torch`, `\_shrink_torch`, aligned NMSE on effective covariances; kept NMSE self‑test.
- `ris_pytorch_pipeline/infer.py`: angle pipeline now receives snapshots and `HYBRID_COV_BETA` for hybrid at inference.
+- `ris_pytorch_pipeline/model.py`: vectorized per-sample SNR shrinkage for K‑head spectral features.

### Quick runbook

- Pure overfit (prove learning):
  - `python test_overfit.py`
  - Expect NMSE self‑test print, guards pass, val NMSE → < 0.1 quickly (smoke test).

- Then:
  - Train with hybrid enabled and track: `K_acc`, `K_under/K_over`, `AoA_RMSE`, `Range_RMSE`, `Success_rate`.
  - Enable K‑loss with ramp → acceptance based on improved K metrics and stable RMSE.
  - Introduce structure terms one by one; accept only if K/loc metrics improve or stay flat.
  - After training: run `trainer.calibrate_k_logits(val_loader)` to temperature-scale K logits; store `k_calibration_temp` in the checkpoint and use at inference.



