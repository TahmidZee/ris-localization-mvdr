## Recent Findings & HPO Progress Report (2026-01-28)

### Scope
This report summarizes the most recent work around the MVDR localization pipeline, with emphasis on:
- What we changed/fixed in the pipeline (notably `R_samp`, L=64 migration, shrinkage consistency)
- How Stage-1 and Stage-2 HPO progressed
- What the Stage-2 results *mean* (and what they do **not** mean)
- What we are doing now (current recommended execution path)

---

## 1) Where we are in the pipeline (current “production” target)

### Core production inference path
- **Neural network predicts covariance** (`R_pred`) from compressive measurements.
- **MVDR-first inference** runs on `R_pred` (effective covariance with conditioning + shrinkage).
- Optional: **SpectrumRefiner** (heatmap supervision) refines MVDR spectra after the backbone is trained.

### `R_samp` status / decision
- We previously invested effort into fixing `R_samp` construction end-to-end (dataset → shard writing → inference usage).
- Empirically and structurally, with **`M=16` BS antennas and `N=144` RIS elements**, `R_samp` is fundamentally limited (rank deficiency / poor subspace overlap) and **can be MVDR-useless even when numerically “sane”**.
- Current practical stance for SOTA progress: **run production with `R_pred` only** (hybrid beta effectively 0).

---

## 2) Snapshot count migration (L=64) and shard generation challenges

### Why we moved to L=64
- More snapshots improve sample quality and stabilize downstream learning/inference (especially shrinkage/conditioning sensitivity).

### What broke / felt “stuck”
- Regenerating shards with L=64 initially appeared hung because of heavy compute + large preallocation (and minimal progress output).

### What we changed
- Added more robust shard generation behavior (progress visibility, safer defaults for L=64).
- Default stance is **not to store `R_samp` in shards**, both for speed and because the production path is `R_pred`-only.

---

## 3) Shrinkage consistency (NumPy vs Torch)

### Problem
- Shrinkage semantics differed between NumPy and Torch paths, causing silent train/infer mismatch.

### Fix
- Unified shrinkage semantics so `mdl_cfg.SHRINK_BASE_ALPHA` drives shrinkage consistently.
- Made alpha scaling snapshot-aware (L-aware behavior).

---

## 4) HPO progress summary

### Stage-1 HPO (surrogate)
Stage-1 was run as a broad exploration using **fast surrogate validation**:
- Study:
  - `L64_M64_wide_2stage_20260126_154123_stage1_surrogate`
  - 300 planned trials
- Surrogate objective is fast and designed to correlate with later MVDR-final behavior, but it is still a proxy.

### Stage-2 HPO (MVDR-final rerank)
Stage-2 was launched as a rerank of Stage-1 top-K candidates using the **production-aligned MVDR-final objective**:
- Study:
  - `L64_M64_wide_2stage_20260126_154123_stage1_surrogate_stage2_mvdr_final`
  - 40 trials planned
- Implementation detail (important):
  - Training loop still uses **fast surrogate validation** for early stopping.
  - The **Optuna objective** returned for Stage-2 is computed by a final **E2E MVDR-final evaluation** on a fixed val subset.

#### Stage-2 results observed so far (from `hpo_goose/hpo_20260127_170834.log`)
Across multiple trials, MVDR-final metrics are consistently very poor, e.g.:
- Trial 0:
  - `E2E(MVDR-final) obj=12.1576 rmse_xyz=4.433m F1=0.006 ... TP=47/6012 FP=9953 FN=5965`
- Trial 1:
  - `E2E(MVDR-final) obj=11.9234 rmse_xyz=3.967m F1=0.006 ... TP=48/6012 FP=9952 FN=5964`

Interpretation:
- Stage-2 is **executing MVDR-final evaluation** (it is not “stuck in Stage-1 mode”).
- The issue is that **the backbone is not learning a usable covariance under the tiny per-trial training budget**, so MVDR-final scores are near-flat across candidates → reranking signal is weak.

---

## 5) What the Stage-2 HPO results mean (and what they don’t)

### What they mean
- Under the current per-trial training regime (10k train / 1k val; early-stop around ~16–17 epochs), the model is not reaching a regime where MVDR-final detection/localization is competitive.
- Because the objective is poor for nearly all candidates, Stage-2 cannot effectively separate “good” from “bad” hyperparameter sets yet.

### What they do NOT mean
- This does **not** prove the pipeline is fundamentally broken.
- It does **not** prove the MVDR-final objective is wrong.
- It does **not** imply `R_samp` needs to be reintroduced.

It indicates we need a **proper learning confirmation** first (full training) so HPO has a meaningful signal.

---

## 6) What we are doing now (current plan)

### Immediate goal: confirm learnability with a full training run
Before spending more compute on HPO reranking, run backbone training at the intended scale:
- `n_train=100000`, `n_val=10000`
- Log output to file (use `tee`) for reproducibility and debugging.

### Loss policy for backbone (updated)
To prioritize MVDR-usability and stability:
- **Enabled**: **subspace alignment** (`lam_subspace_align`) for backbone training.
- **Disabled**: **peak contrast** (`lam_peak_contrast`) during backbone training, because peak shaping is deferred to SpectrumRefiner heatmap supervision.
- `lam_gap` / `lam_margin` remain off/ignored (SVD-backprop instability and currently disabled in code).

This policy was pushed to GitHub on branch `fix/rsamp-end2end-20260125`.

### After backbone training passes the “learnability check”
If full training produces a usable MVDR-final result (improving F1 and rmse_xyz):
- Train SpectrumRefiner (heatmap supervision).
- Re-run HPO with a budget that has meaningful signal (or narrow search around a working baseline).

---

## 7) Key “do / don’t” checklist (as of today)

### Do
- **Proceed with `R_pred`-only** in production inference (β≈0).
- **Run full training once** to validate learnability before more Stage-2 reranking.
- Keep logs of full training runs (`tee` to `results_final_L64_12x12/logs/...`).

### Don’t (for now)
- Don’t spend large compute on Stage-2 MVDR-final reranking while MVDR-final is near-flat across candidates.
- Don’t re-enable `lam_gap` / `lam_margin` (currently disabled/ignored and historically unstable).
- Don’t rely on `R_samp` for MVDR in the current `M<<N` regime.

