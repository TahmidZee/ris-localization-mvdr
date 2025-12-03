## Recent Updates — K-Head Upgrades and Hybrid R_eff Alignment (Nov 25, 2025)

Owner: Tahmid
Scope: Summarize the last two updates: (1) K-head upgrades, (2) Ideal fix to make K-head use the exact same hybrid effective covariance as MUSIC, plus calibration and MDL gating.

### Update 1 — K-Head Upgrades

Why
- K was being inferred from a covariance object different from what MUSIC uses; underestimation wasn’t penalized strongly; MDL wasn’t leveraged at decision time.

What changed
- R_eff-aligned K features (pred-only): K-head eigenfeatures now come from `build_effective_cov_torch(R_pred, snr_db, ...)` instead of raw `R_hat` (hermitize → trace-norm → diag-load → per-sample shrink). This removed per-batch SNR issues and aligned construction with inference conventions.
- MDL one-hot fusion (feature-level): The argmin MDL over candidate K is encoded as a one-hot and concatenated to spectral features. The K MLP can now learn residual corrections to MDL.
- Cost-sensitive K loss: Underestimation is penalized more than overestimation.
  - `mdl_cfg.K_UNDER_WEIGHT` (default 2.0)
  - `mdl_cfg.K_OVER_WEIGHT` (default 1.0)
  - `mdl_cfg.K_LABEL_SMOOTHING` retained (default 0.05)

Files
- `ris_pytorch_pipeline/model.py`: K feature construction updated; MDL one-hot added; feature dim `5*K_MAX + 2`.
- `ris_pytorch_pipeline/loss.py`: Cost-sensitive CE for K.
- `K_HEAD_UPGRADES_2025-11-25.md`: Documented changes, config knobs, expected impact.

Impact
- Lower `K_under` at similar/better `K_acc`. More robust mixed-SNR behavior. MDL signal available to the network.

---

### Update 2 — Ideal Fix: K-Head Uses Exactly the Same Hybrid R_eff as MUSIC

Why
- Final conceptual mismatch: K-head saw `R_pred` (with shrink) while MUSIC used the hybrid blend `(1−β)R_pred + βR_samp`. K should learn from the same covariance used to localize.

What changed
- Pass `R_samp` into the model:
  - `HybridModel.forward(..., R_samp=None)` now accepts optional sample covariance (RI or complex). No gradients flow through `R_samp`.
- K-head builds hybrid R_eff internally (when `β>0` and `R_samp` present):
  - Uses `build_effective_cov_torch(R_pred, snr_db, R_samp, beta, diag_load, shrink)` for K eigenfeatures.
  - This matches MUSIC’s covariance path by construction.
- Training/validation/inference plumbing:
  - Train/val: pass `R_samp` (if available) into `model(...)` so K-head sees hybrid R_eff.
  - Inference: build `R_samp` from snapshots (`build_sample_covariance_from_snapshots(...)`) and pass it into `model(...)` for hybrid-aware K logits.
- Calibration and MDL gating:
  - Post-train K temperature scaling auto-runs and stores `k_calibration_temp` in `best.pt` (`mdl_cfg.CALIBRATE_K_AFTER_TRAIN = True` default).
  - Confidence-gated K selection at inference/validation:
    - Compute calibrated `p(K)`; if `max p(K) < cfg.K_CONF_THRESH` (e.g., 0.65), fall back to MDL K on the same R_eff; else use NN K.

Files
- `ris_pytorch_pipeline/model.py`:
  - `forward(..., R_samp=None)`; K-head uses hybrid R_eff when available.
- `ris_pytorch_pipeline/train.py`:
  - Pass `R_samp` to `model(...)` in train and validation.
  - Auto-run K calibration after training; save temperature into checkpoint.
- `ris_pytorch_pipeline/infer.py`:
  - Build `R_samp` from snapshots and pass to `model(...)` for hybrid-aware K logits.
  - Confidence-gated NN vs MDL K with calibrated logits.

Config knobs
- `mdl_cfg.CALIBRATE_K_AFTER_TRAIN` (bool, default True): run K calibration on val and save `k_calibration_temp` into `best.pt`.
- `cfg.K_CONF_THRESH` (float, e.g., 0.65): threshold on calibrated max p(K) for MDL fallback.
- `mdl_cfg.K_UNDER_WEIGHT`, `mdl_cfg.K_OVER_WEIGHT`, `mdl_cfg.K_LABEL_SMOOTHING`: cost-sensitive K loss settings.

How to use
- Training: nothing special; if shards contain `R_samp`, it will be used for hybrid-aware K features; otherwise K uses `R_pred` path.
- Validation:
  - Inference-like evaluation computes and logs: `K_acc`, `K_under`, `K_over`, `Success_rate`, `AoA_RMSE`, `Range_RMSE`, and `K_mdl_acc`.
  - Set `cfg.VAL_PRIMARY="k_loc"` to drive checkpoint selection by K/localization metrics.
- Inference:
  - Model’s `k_logits` are calibrated (`k_calibration_temp`) and confidence-gated against MDL K on the same R_eff.

Expected impact
- K-head and MUSIC now use the same statistical object (R_eff), eliminating the final mismatch.
- Better K stability under hybrid (β>0) and mixed-SNR. Lower `K_under` with minimal AoA/Range trade-off.

---

### Quick references (where to look)

- K features from hybrid R_eff (model):
  - `ris_pytorch_pipeline/model.py` → `forward(..., R_samp=...)` K-head block.
- Pass `R_samp` into model (train/val):
  - `ris_pytorch_pipeline/train.py` → `_train_one_epoch`, `_validate_one_epoch` calls to `model(...)`.
- Inference — hybrid-aware `k_logits` + gating:
  - `ris_pytorch_pipeline/infer.py` → build `R_samp` from snapshots; pass to `model(...)`; gate NN vs MDL with calibration.
- K calibration auto-run:
  - `ris_pytorch_pipeline/train.py` → after training loop, `calibrate_k_logits(...)` and save `k_calibration_temp`.


