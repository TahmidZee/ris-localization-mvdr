## R_samp Fix Report (2026-01-25)

### Context / Symptom
- During end-to-end MVDR evaluation we observed:
  - **MVDR on `R` (ground-truth covariance): F1 ≈ 0.76** ✅
  - **MVDR on `R_samp` (sample covariance from snapshots): F1 = 0.00** ❌
- This meant Stage-2 HPO (MVDR-final objective) could be “poisoned” whenever hybrid blending used `R_samp`.

### Root Causes Found
- **(A) Dataset snapshot model was effectively coherent across snapshots**
  - In `ris_pytorch_pipeline/dataset.py`, the source symbol vector `s` was constant across the `L` snapshots.
  - With constant symbols, multi-snapshot covariance estimation collapses; `R_samp` becomes degenerate / uninformative.

- **(B) `build_sample_covariance_from_snapshots()` removed the signal via mean-centering**
  - In `ris_pytorch_pipeline/angle_pipeline.py`, the older implementation mean-centered per-snapshot RIS-domain estimates.
  - Mean-centering can subtract the dominant signal component and leave mostly noise covariance.

- **(C) A second “sample covariance” implementation was mathematically wrong**
  - `ris_pytorch_pipeline/baseline.py:incident_cov_from_snaps()` solved a single joint LS system and returned **rank-1** covariance (`x x^H`).
  - Benchmarks used this function, so benchmark `R_samp` behavior was inconsistent with the intended hybrid design.

- **(D) Inference path `hybrid_estimate_raw()` built `R_samp` using the wrong channel input**
  - It used `H` from the dataset, which is an *effective per-snapshot* vector `[L,M]`, not the full BS→RIS matrix `[M,N]` required to solve for RIS-domain signals.

### Follow-up Finding (Important): why “numerically sane” still gave MVDR F1=0
- Even after fixing coherence + mean-centering and making `R_samp` Hermitian/trace-normalized, **MVDR on `R_samp` could still be F1≈0**.
- Reason: per-snapshot inversion is fundamentally ill-posed here (**M < N**). Estimating a full spatial covariance from a handful of underdetermined snapshots requires an explicitly **joint low-rank** estimator; naive per-snapshot `x_hat` heuristics can be MVDR-useless while still looking “reasonable” under NMSE/trace checks.

### Fixes Implemented
- **Dataset generation fix (required for correct `R_samp`)**
  - `ris_pytorch_pipeline/dataset.py`
  - Generate time-varying source symbols: `s` is now **shape `[L,K]`** (per-snapshot symbols).
  - Update `y_clean` generation to use `s[l]`.
  - Update `R_true` to use average per-source power across snapshots.

- **Correct `R_samp` construction**
  - `ris_pytorch_pipeline/angle_pipeline.py`
  - Rewrote `build_sample_covariance_from_snapshots()` and added a stronger solver:
    - Accepts `H` as `[M,N]` or `[L,M,N]`
    - **Default (recommended) solver: `als_lowrank`**
      - Joint alternating least squares across snapshots to estimate a low-rank factor \(F\\) such that
        \(y_l \\approx (H\\,\\mathrm{diag}(c_l))\\,(F s_l)\).
      - Sets \(R_{samp} = F F^H\) then hermitizes + trace-normalizes.
    - Keeps `ridge_ls` and `matched_filter` as cheaper heuristics (not recommended for MVDR validation).
    - Hermitizes and trace-normalizes to `tr(R)=N`

- **Naming / call-site consistency**
  - `ris_pytorch_pipeline/baseline.py`
    - `incident_cov_from_snaps()` now delegates to `build_sample_covariance_from_snapshots()` (no more rank-1 bug).
  - `ris_pytorch_pipeline/infer.py`
    - `hybrid_estimate_raw()` now prefers precomputed `R_samp`, otherwise builds using `H_full` (correct shape).
    - `hybrid_estimate_final()` no longer repeats `H_full` across snapshots (builder accepts `[M,N]` directly).

- **Regression test added**
  - `ris_pytorch_pipeline/regression_tests.py`
  - New command: `python -m ris_pytorch_pipeline.regression_tests r_samp`
    - Checks Hermitian property, trace normalization, non-degeneracy, and loose NMSE correlation vs `R`.
    - Implemented to read **one shard via mmap** to avoid OOM on machines with many shards.
  - New command: `python -m ris_pytorch_pipeline.regression_tests r_samp_mvdr`
    - Recomputes `R_samp` from snapshots and runs **tiny MVDR** to ensure **TP>0** (stronger guardrail).

### Operational Notes
- **You must regenerate shards** for these fixes to take effect, because `R_samp` is stored in `.npz` shards:

```bash
python -m ris_pytorch_pipeline.ris_pipeline pregen-split \
  --n-train 160000 --n-val 40000 --n-test 8000 \
  --shard 25000 --seed 42
```

- After regeneration, validate:

```bash
python -m ris_pytorch_pipeline.regression_tests r_samp --n 16 --seed 0
```

### Current Verification (local)
- `r_samp` regression test: **PASS**
- `mvdr_lowrank` regression test: **PASS**

