### K-estimation debugging report (MUSIC-free training/HPO)

**Date**: 2026-01-12  
**Scope**: Diagnose why the model’s **K-head** is stuck near chance during MUSIC-free training/HPO, and validate whether classical K-estimation is feasible under the current **L=16, N=144** regime.

---

### Executive summary

- **K estimation is the blocker**: the logs show **K accuracy hovering near chance** (\(\approx 0.2\) for `K_MAX=5`) and predictions often collapsing to a single class.
- We found **two “instrumentation/diagnostic” issues** that made the situation look even worse (and hid what was happening):
  - **Train-loss logging was wrong** (only the last batch contributed to `running`), so the printed `train_loss` was not the loss being optimized.
  - The “hybrid blending failed” message was misleading when **β=0** (hybrid blending was disabled by config, so rank cannot increase).
- We added an **MDL sanity check** and observed **MDL collapses to always predicting K=1** on `R_samp` in the current **N=144, L=16** setting (small sample test). This strongly suggests **classical MDL on the 144×144 covariance is statistically ill-conditioned at L=16**.
- The next highest-leverage steps are:
  - **Re-run the K diagnostic with hybrid enabled** (β>0 so K spectral features see `R_samp`).
  - **Evaluate K-estimation in a lower-dimensional domain** (e.g., measurement domain or a reduced subspace) and/or switch to an **ordinal / cumulative K head**.
  - Consider a clean **L ablation** (L=16 vs 32/64) if K remains unstable even with hybrid + improved head.

---

### What changed (code edits)

- **Fixed train loss accumulation/logging** so `running` reflects all batches, not just the final batch:
  - `ris_pytorch_pipeline/train.py`
- **Clarified hybrid blending diagnostic output** so β=0 is treated as “disabled”, not “failed”:
  - `ris_pytorch_pipeline/train.py`
- **Updated K diagnostic to stop forcing β=0 in EASY_OVERFIT** (and allow `K_DIAG_BETA` override):
  - `run_k_diagnostic.py`
- **Added an MDL sanity-check script** to evaluate whether K is even identifiable with current shards:
  - `mdl_sanity_check.py`

---

### What we observed in the logs (high-signal points)

From `run_k_diagnostic_trace.log` (EASY_OVERFIT mode):

- **The optimized loss terms show K is basically at chance**:
  - `CE_raw ≈ ln(5) ≈ 1.61` and `lam_K * CE ≈ 3.2–3.3`, matching chance-level behavior.
  - Example symptom: `K_pred dist` collapsing to a single class (e.g. all predicted K=5).
- **The printed `train_loss` was not trustworthy** in that run:
  - The per-batch `running` counter was only updated inside the optimizer-step block, so it effectively reflected only the last batch (we fixed this).
- **Hybrid covariance diagnostic said “FAILED” because β=0**:
  - In EASY_OVERFIT, the script forced `cfg.HYBRID_COV_BETA = 0.0`.
  - With β=0, `R_blend = R_pred`, so the rank stays ≤ `K_MAX` and “rank didn’t increase” is expected.
  - We updated the message to avoid false alarms when β=0.

---

### MDL sanity check (what it found)

We added `mdl_sanity_check.py` to compute classical MDL K-hat on the **offline** sample covariance `R_samp` stored in shards.

**Small-run result (32 samples)**:

- MDL predicted **K=1 for every sample**, giving accuracy equal to the fraction of true K=1 in that mini-batch (\(\approx 0.156\)).
- Disabling diagonal loading and shrinkage did **not** change this behavior.

**Interpretation**:

- This is consistent with the known issue that **MDL becomes dominated by its penalty term** when the covariance dimension is large (**N=144**) and the “snapshot count” is small (**L=16**).
- This does **not** prove “L=16 is impossible”, but it strongly suggests:
  - **MDL on the 144×144 covariance is a bad baseline at L=16**, and
  - A K estimator should likely operate in a **lower-dimensional domain** (or use a different criterion than MDL).

**Note on this environment**:

- Larger MDL sanity runs were killed (exit code 137) in this runtime environment unless we used very small settings (`batch_size=1` with `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1`). On your V100 box you can run this at scale on GPU.

---

### Thoughts on the user’s hypothesis (freeze vs snapshots vs head design)

Your points are mostly right, with one important nuance:

- **Yes**: Starting `TRAIN_PHASE="k_only"` with `FREEZE_BACKBONE_FOR_K_PHASE=True` **from scratch** is a guaranteed failure mode (the K head sees random features → chance forever).
- **But**: in the **latest diagnostic logs**, backbone freezing was already disabled (`FREEZE_BACKBONE=False` in the diagnostic run), and K still did not improve. So **freezing alone does not explain the current behavior**.
- **The bigger issues are**:
  - The K spectral path needs **informative covariances** (hybrid β>0 so it can “see” `R_samp`), and
  - Classical IC methods (MDL) are very shaky on **144-dim** covariances at **L=16**, which also implies that “K from eigenspectrum” is intrinsically hard in that space.

Your suggestions are solid:

- **Estimating K in a lower-dimensional domain** (e.g. BS measurement covariance) is statistically more sane.
- **Ordinal / cumulative K prediction** is often easier than 5-way softmax and gives better control over under/over penalties.

---

### What to do next (recommended, in order)

- **1) Re-run K diagnostic with hybrid enabled**:

```bash
cd /home/tahit/ris/MainMusic
EASY_OVERFIT=1 K_DIAG_BETA=0.5 python run_k_diagnostic.py
```

- **2) Run MDL sanity at scale on your GPU machine** (keep threads low; batch_size can be >1 on GPU):

```bash
cd /home/tahit/ris/MainMusic
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python mdl_sanity_check.py --n_val 2000 --batch_size 64 --device cuda
```

- **3) If K is still bad**, prioritize one of these “stop bleeding time” options:
  - **Lower-dimensional K estimator** (measurement-domain or reduced subspace).
  - **Ordinal K head** (predict \(P(K>t)\) for t=1..4 and sum).
  - **Slot-existence (`k_mask`) head** (BCE/Focal over K slots, then sum).
  - **L ablation**: regenerate shards for L=32 (then L=64) and measure K stability.

---

### Files referenced

- `run_k_diagnostic.py`
- `ris_pytorch_pipeline/train.py`
- `ris_pytorch_pipeline/model.py`
- `mdl_sanity_check.py`


