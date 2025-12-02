## Implementation Change Log — Alignment and Hybrid Covariance (Nov 25, 2025)

Owner: Tahmid
Scope: Unify effective-covariance across train/val/infer, remove online R_samp LS from hot path, reweight loss toward K̂ and localization

### Why these changes

- Prior instability and “loss floor” were driven by training/inference covariance mismatch and an NMSE definition that didn’t reflect localization quality.
- We now optimize the same effective covariance the inference stack consumes and shift emphasis toward K̂ and subspace/angle accuracy, with NMSE kept as a regularizer.

### High-level outcomes

- A single, consistent “effective covariance” path: hermitize → trace-normalize (trace=N) → optional hybrid blend → diagonal loading → optional per‑sample shrink.
- Loss NMSE compares effective covariances (pred vs target) built the same way; NMSE remains 0 at perfection.
- Inference now uses hybrid (when configured) just like train/val.
- R_samp is precomputed offline and loaded from shards; no per-batch LS in the training loop.
- Loss weights reweighted to prioritize K/subspace/aux over NMSE.

---

### Changes by file

- `ris_pytorch_pipeline/covariance_utils.py` [NEW]
  - Torch helpers: `trace_norm_torch`, `shrink_torch`, `build_effective_cov_torch` (per-sample SNR, differentiable).
  - NumPy helpers: `trace_norm_np`, `shrink_np`, `build_effective_cov_np` (for eval/infer paths).

- `ris_pytorch_pipeline/loss.py` [UPDATED]
  - NMSE now computed on effective covariances (trace=N, per-sample shrink) for both target and prediction.
  - Uses `build_effective_cov_torch` to align with inference pipeline.
  - NMSE remains a true error (0 at R̂=R_true); self-test print kept.
  - Loss reweight defaults (starting points, tune by K_acc/AoA RMSE):
    - `lam_cov=0.10`, `lam_cov_pred=0.02`, `lam_K=0.50`, `lam_aux=1.00`, `lam_subspace_align=0.50`, `lam_peak=0.20`.
  - **Cost-sensitive K loss**: Underestimation is penalized more than overestimation (configurable via `mdl_cfg.K_UNDER_WEIGHT`, default 2.0; `mdl_cfg.K_OVER_WEIGHT`, default 1.0). This reflects the physics: missed sources are worse than false alarms.

- `ris_pytorch_pipeline/angle_pipeline.py` [UPDATED]
  - Builds the effective covariance once via `build_effective_cov_np` (hybrid blending + norm + diag-load). Avoids duplicated norm/load logic.
  - Leaves MUSIC’s own adaptive shrinkage behavior intact (no double-shrink).

- `ris_pytorch_pipeline/infer.py` [UPDATED]
  - Calls `angle_pipeline(...)` with `y_snapshots`, `H_snapshots`, `codes_snapshots`, and `blend_beta=cfg.HYBRID_COV_BETA` so inference uses hybrid like train/val.

- `ris_pytorch_pipeline/model.py` [UPDATED]
  - K‑head spectral shrinkage is vectorized per-sample SNR (no more "first sample only" in batch).
  - **K-head now uses R_eff** (same effective covariance as inference/MUSIC): hermitize → trace-norm → diag-load → shrink via `build_effective_cov_torch`. This aligns K features with what MUSIC sees.
  - **MDL one-hot added to K features**: The argmin of MDL scores is encoded as a one-hot vector and concatenated to the spectral features. This lets the network learn residual corrections to MDL, improving low-SNR robustness.
  - K feature dimension updated: `5*K_MAX + 2` (was `4*K_MAX + 2`).

- `ris_pytorch_pipeline/dataset.py` [UPDATED]
  - `prepare_shards(...)` now precomputes `R_samp` offline (NumPy/CPU) and stores it in shard `.npz` files (field `R_samp`).
  - `ShardNPZDataset.__getitem__` returns `R_samp` tensor when present.

- `ris_pytorch_pipeline/train.py` [UPDATED]
  - `_unpack_any_batch(...)` now returns `R_samp` (if present) along with `H_full`.
  - Removed per-batch LS for `R_samp` from the hot path; if `R_samp` exists, blend `(1-β)R_pred + βR_samp`; else fall back to pure `R_pred`.
  - Blended covariance is hermitized and diagonally loaded (no extra trace norm applied here; loss handles effective normalization/shrink).
  - Added inference-like validation:
    - Per-epoch Hungarian-matched metrics: `K_acc`, `K_under/K_over`, `AoA_RMSE`, `Range_RMSE`, `Success_rate`.
    - MDL baseline K on the same effective covariance; logs `K_mdl_acc`.
    - Option to drive selection/early stopping by K/localization composite via `cfg.VAL_PRIMARY="k_loc"`.

---

### Behavioral notes and guardrails

- Effective-covariance alignment:
  - Loss, validation, and inference now share the same normalization/shrink conventions (no double-shrink, no mismatched scaling).
  - If hybrid is used in training, it is used in inference too.

- Hybrid blending:
  - Uses offline `R_samp` when available. If `R_samp` is missing in a run, training falls back to pure `R_pred` with β messages logged.
  - β warmup/jitter behavior preserved; logs print the β in first batch.

- Loss weighting (defaults) now emphasize K and localization:
  - Adjust in config or per-experiment as you start tracking K_acc, AoA/Range RMSE, and Success_rate.

---

### How to use

1) Regenerate shards (to include offline `R_samp`):
- Use `prepare_shards(...)` in `ris_pytorch_pipeline/dataset.py` to recreate train/val/test shards so `R_samp` is saved alongside `y/H/codes/ptr/K/snr/R`.

2) Run the pure sanity probe (optional, quick): 
- Verify NMSE self-test (`≈0` for R_true vs R_true; `≈1` for zeros) and that the loop is stable.

3) Train with hybrid + localization emphasis:
- Keep the reweighted loss defaults; monitor K_acc, AoA_RMSE, Range_RMSE, Success_rate in validation.

4) K‑centric fine-tune:
- Freeze most of backbone; train K-heads; calibrate logits (temperature). Optionally fuse K̂ with MDL/AIC at inference for robustness.

---

### Expected logs

- `[SELFTEST] nmse(R_true,R_true)≈0, nmse(0,R_true)≈1` on first forward.
- `[Beta] epoch=..., beta=... (offline R_samp)` if `R_samp` is provided.
- Loss prints once per run with reweighted λ’s and per-term magnitudes.

---

### Risks and mitigations

- Snapshots/audio path must be available at inference to use hybrid. If not feasible, set β=0 everywhere (train/val/infer/loss).
- Avoid adding separate per-site normalization/shrink; the builders should remain the single source of truth.

---

### Summary

- Unified the effective covariance used across train/val/infer and the loss (no mismatched objects).
- Removed the heavy per-batch LS from training; `R_samp` is now precomputed offline and loaded from shards.
- Reweighted the loss to focus on K̂ and localization; NMSE retained as a conditioning regularizer.
- K-head improvements:
  - Uses `R_eff` (same as inference) for eigenfeatures, aligning K with what MUSIC sees.
  - MDL one-hot fused at feature level for decision-level ensemble.
  - Cost-sensitive K CE loss penalizes underestimation more than overestimation.



