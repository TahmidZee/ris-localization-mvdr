## Recent Findings & Challenges Report (2026-01-29)

### TL;DR
- **MVDR itself is fine**: on `R_true` we get strong detection (e.g., **F1≈0.95** on 500 test scenes).
- **The bottleneck is `R_pred`**: MVDR on `R_pred` collapses (e.g., **F1≈0.08**, huge FP/FN).
- This is **not** a factor-unpacking bug; the factor layout is **interleaved RI** and consistent with training.
- The practical fix path is to **make training/validation optimize MVDR-relevant structure**:
  - disable unstable AMP by default,
  - increase **subspace alignment** pressure,
  - enable a **small, stable peak-contrast** term,
  - add **subspace/peak metrics in validation** so early stopping + HPO have a meaningful signal.

---

## 1) Key diagnostic results (what we know now)

### 1.1 Oracle upper bound: MVDR on `R_true` works
From `diagnose_hybrid_vs_oracle_mvdr.py` (test split, 500 scenes):
- **Oracle MVDR on `R_true`**:
  - TP=1373, FP=11, FN=122
  - Precision=0.992, Recall=0.918, **F1=0.954**

Interpretation:
- The scenario, manifold, range planes, and MVDR detector are not the limiting factor.
- Any poor end-to-end result is dominated by the quality of the covariance fed into MVDR.

### 1.2 MVDR on `R_pred` is currently unusable
Same diagnostic:
- **Hybrid MVDR on `R_pred`** (Blind-K):
  - TP=151, FP=2069, FN=1344
  - Precision=0.068, Recall=0.101, **F1=0.081**

Interpretation:
- `R_pred` is not producing an MVDR spectrum with correct peaks; it generates many spurious peaks and misses true ones.
- This also explains why B1/B2 suite metrics looked poor even when aux RMSE was improving: the aux head is not the production estimator.

### 1.3 Factor layout check: not an unpacking mismatch
We tested two unpackings of `cov_fact_*`:
- interleaved RI: `[re0, im0, re1, im1, ...]`
- split-halves: `[re...(N*K), im...(N*K)]`

Results (500 scenes, effective covariance vs effective `R_true`):
- **NMSE**:
  - interleaved: median≈1.34
  - split:       median≈1.32
- **Top-K subspace overlap** (higher is better):
  - interleaved: median≈0.25
  - split:       median≈0.02

Interpretation:
- The correct layout is **interleaved** (split layout is clearly wrong on subspace overlap).
- The problem is not unpacking; it is that the model is not learning a good signal subspace yet.

---

## 2) HPO + training challenges encountered

### 2.1 Stage-2 HPO “looked like Stage-1”
Cause:
- Stage-2 uses surrogate validation during training for early stopping, then runs MVDR-final scoring once post-training.
Fix:
- Added explicit Stage-2 logging so it’s clear MVDR-final scoring is happening.

### 2.2 Full training had a 3-phase curriculum that broke early stopping
Observed:
- Phase 2 jumped loss scale (because `lam_cov` increased), making surrogate scores non-comparable across phases.
Fix:
- Disabled `USE_3_PHASE_CURRICULUM` by default so full training isn’t derailed by phase-dependent score shifts.

### 2.3 Benchmarking issues obscured interpretation
Fixes applied:
- Added `--no-baselines` to run Hybrid-only suites (no Ramezani/DCD/NFSSN dependency noise).
- Fixed baseline call signatures and added baseline failure guards.
- Fixed RMSPE reporting by matching in 3D when reporting RMSPE (avoids range-mismatch inflation).
- Rate-limited repeated “refiner fallback” logs to avoid spam.

---

## 3) What we are doing now (solution path)

We will make training and validation reflect MVDR-relevant success conditions:

### Step 1 — Disable AMP by default
Motivation:
- AMP overflow/step-skips were observed during long runs. Covariance + eigenspace-related optimization is numerically sensitive.
Action:
- Default `mdl_cfg.AMP=False` for backbone training.

### Step 2 — Increase subspace alignment pressure
Motivation:
- MVDR depends on the signal subspace; measured subspace overlap is far too low.
Action:
- Increase `lam_subspace_align` (joint phase) from 0.5 to **2.0**.

### Step 3 — Enable a small, stable peak-contrast term
Motivation:
- Even with improved subspace, we need MVDR spectra that peak at correct locations with fewer spurious maxima.
Action:
- Enable `lam_peak_contrast=0.02` with stable `PEAK_CONTRAST_TAU=0.5`.

### Step 4 — Validate using MVDR-relevant metrics during training
Motivation:
- Surrogate loss/aux RMSE alone can improve while MVDR on `R_pred` remains unusable.
Action:
- Always build `R_blend` (effective covariance) from factor heads even when `R_samp` is absent.
- Log:
  - MVDR peak detection proxy metrics (optional; capped scenes)
  - subspace overlap median/mean (capped scenes)
  - (optionally) incorporate overlap into the surrogate score for early stopping/HPO.

---

## 4) Are we sure the model is “fine”?

We are confident the *pipeline physics and MVDR* are fine (oracle MVDR is strong).
However, the model path **does require focused investigation** because:
- `R_pred` is currently not MVDR-usable (F1≈0.08).
- Subspace overlap is low (~0.25 median).

The changes above (Steps 1–4) are the correct first intervention because they:
- eliminate a known numerical instability driver (AMP),
- directly optimize the MVDR-critical subspace,
- add a small peak-shaping cue,
- and, critically, make validation reflect MVDR success.

If after these changes Hybrid F1 remains low, the next things to inspect would be:
- whether the factor heads are saturating / collapsing (column norms, rank usage),
- whether `LAM_RANGE_FACTOR` and range-plane selection are mis-calibrated,
- whether the `H`/`codes` feature pathways are sufficient (information bottleneck),
- and whether we need to incorporate more MVDR-first supervision earlier (larger `SURROGATE_PEAK_MAX_SCENES` or a slightly higher `lam_peak_contrast`).

---

## 5) Next commands (goose)

After pulling the latest changes, the “go/no-go” gate is:
- `diagnose_hybrid_vs_oracle_mvdr.py` Hybrid F1 should climb materially (from ~0.08 toward the oracle).

Suggested run order:
1. Full backbone training (stable defaults).
2. Run:
   - `python diagnose_hybrid_vs_oracle_mvdr.py --split test --n 500`
3. Only after Hybrid F1 is reasonable:
   - train SpectrumRefiner,
   - revisit Stage-2 HPO.

