## K-Head Upgrades — Alignment, MDL Fusion, Cost-Sensitive Loss (Nov 25, 2025)

Owner: Tahmid
Scope: Make K̂ prediction consistent with inference, leverage MDL at decision time, and penalize missed sources more than false alarms.

### Why

- K features were computed from a different covariance (`R_hat` from angle factors) than the one MUSIC uses at inference (`R_eff`). This misalignment costs accuracy and stability.
- Underestimation (K̂ < K_true) is worse than overestimation in practice; symmetric CE ignores this asymmetry.
- MDL already provides a strong signal; fusing it at decision level helps low‑SNR robustness and reduces pathological misses.

---

### What changed (code)

- `ris_pytorch_pipeline/model.py`
  - K-head eigenfeatures now use `R_eff` built via `build_effective_cov_torch` (hermitize → trace-normalize → diag-load → per-sample shrink).
  - Added MDL argmin as a one-hot vector concatenated to spectral features (lets the network learn residual corrections to MDL).
  - Updated K feature dimensionality: from `4*K_MAX + 2` to `5*K_MAX + 2`.
  - Notes:
    - Hybrid with `R_samp` is still performed outside the model (in train/val/infer). The model uses `R_pred` + shrink for K features to avoid baking data IO into forward().

- `ris_pytorch_pipeline/loss.py`
  - Cost-sensitive K cross-entropy:
    - Underestimation penalty: `mdl_cfg.K_UNDER_WEIGHT` (default 2.0).
    - Overestimation penalty: `mdl_cfg.K_OVER_WEIGHT` (default 1.0).
    - Label smoothing remains (default `K_LABEL_SMOOTHING = 0.05`).

No interface changes; training/inference calls remain the same.

---

### Config knobs

- `mdl_cfg.K_UNDER_WEIGHT` (float, default 2.0): multiplier for K̂ < K_true errors.
- `mdl_cfg.K_OVER_WEIGHT` (float, default 1.0): multiplier for K̂ > K_true errors.
- `mdl_cfg.K_LABEL_SMOOTHING` (float, default 0.05): label smoothing for K CE.
- Validation selection (unchanged; recommended):
  - `cfg.VAL_PRIMARY = "k_loc"` to drive checkpointing by K/localization composite instead of loss.

---

### How to use

1) Train as usual. K-head will now learn from `R_eff` features and MDL one-hot.
2) After training, run temperature scaling for K:
   - `trainer.calibrate_k_logits(val_loader, save_path=best_ckpt)`
   - Inference will use `k_calibration_temp` from the checkpoint.
3) (Auto) Calibration after training:
   - Set `mdl_cfg.CALIBRATE_K_AFTER_TRAIN = True` (default). Training will auto-run calibration on the val set and save `k_calibration_temp` into `best.pt`.
4) Confidence-gated K at inference/validation:
   - Set `cfg.K_CONF_THRESH` (e.g., 0.65). If the calibrated max p(K) < threshold, we fall back to MDL K on the same `R_eff`.
5) Keep validation on inference-like metrics:
   - `K_acc`, `K_under`, `K_over`, `Success_rate`, `AoA_RMSE`, `Range_RMSE`, and MDL baseline (`K_mdl_acc`).

---

### Expected impact

- Lower `K_under` at comparable or better `K_acc`.
- More stable K in mixed-SNR batches (per-sample shrink respected).
- Better low-confidence behavior via MDL signal fused at feature level.
- Downstream AoA/Range RMSE stable or improving due to better K.

---

### Acceptance checks (quick)

- Logs show per-epoch metrics line including `K_acc`, `K_mdl_acc`, `succ_rate`, medians and RMSE; score printed when `VAL_PRIMARY="k_loc"`.
- `K_under` decreases vs prior runs at similar SNR.
- No regressions in AoA/Range RMSE.

---

### Notes and next options

- Decision-level MDL fusion can be extended at inference (gating by calibrated confidence: use MDL when max p(K) < τ). Current change fuses MDL as features; gating can be added later if needed.
- If R_samp is available in-model in the future, consider computing K features from hybrid `R_eff` directly inside forward(). Today, hybrid remains in the train/val/infer stack for cleanliness.


