### Implementation updates (2026-01-12)

This document summarizes **all code changes and new artifacts added in this debugging/upgrade pass**, plus the key conclusions from the logs.

Primary objective: move toward a **production-ready, MUSIC-free training/HPO loop** while unblocking the current bottleneck: **K estimation**.

---

### 1) High-level conclusions (from logs + sanity checks)

- **K estimation is the bottleneck**:
  - K accuracy stayed near chance (≈0.2 for `K_MAX=5`) and often collapsed to a single predicted class.
- **RIS-domain (144×144) MDL is not reliable at L=16**:
  - The MDL sanity check on `R_samp` showed MDL collapsing to always predicting `K=1` (on a small sample), consistent with the known regime where IC penalties dominate when **dimension is large (N=144)** and **snapshots are few (L=16)**.
- **Several “debugging artifacts” were masking reality**:
  - Train loss logging was incorrect (only last batch contributed to `running` in the debug build), making training appear healthier than it was.
  - A “hybrid blending failed” message was misleading when β=0 (blending was disabled by config, so rank cannot increase).

---

### 2) What we changed (major behavior changes)

#### 2.1 Optimizer grouping fix (real bug)
Your optimizer uses **two param groups** (backbone vs head) with higher LR for the head.  
Previously, the grouping key list missed several K-head components, so parts of the K-head could be treated as backbone params.

- **Change**: Expanded `HEAD_KEYS` to include missing K-head pieces (direct path + scale + fusion/proj).
- **File**: `ris_pytorch_pipeline/train.py`
- **Impact**: K-head (including `k_direct_mlp` and `k_logit_scale`) is now consistently trained with the intended “head LR”.

#### 2.2 Stop relying on RIS-domain MDL (144×144) for K
We moved “K-from-spectrum” to a statistically better-conditioned domain:

- **Change**: K spectral features now come from measurement-domain covariance:
  - \(y \in \mathbb{C}^{L \times M}\) (here M=16)
  - \(R_{yy} = \frac{1}{L} y^H y \in \mathbb{C}^{M \times M}\)
  - compute eigenvalue features / MDL on \(R_{yy}\) (16×16), not on RIS-domain 144×144 objects.
- **File**: `ris_pytorch_pipeline/model.py`
- **Impact**: The K-head gets features from a domain where **L=16 is far less pathological** than for N=144.

#### 2.3 Make K prediction easier: ordinal/cumulative K head (default)
We added an ordinal K head to avoid common 5-way softmax collapse modes.

- **Change**:
  - Model now outputs:
    - `k_logits` (softmax/classic) — kept for backward compatibility
    - `k_ord_logits` (ordinal) — logits for \(P(K > t)\), \(t=1..K_{max}-1\)
  - Loss supports:
    - `cfg.K_HEAD_MODE="ordinal"` → weighted BCE on `k_ord_logits`
    - else → existing cost-sensitive CE on `k_logits`
  - Surrogate validation K metrics now decode from ordinal logits when enabled.
  - Inference also decodes ordinal logits (with a simple confidence heuristic), then gates vs IC fallback.
- **Files**:
  - `ris_pytorch_pipeline/configs.py` (new knobs)
  - `ris_pytorch_pipeline/model.py` (new heads + output)
  - `ris_pytorch_pipeline/loss.py` (ordinal loss path)
  - `ris_pytorch_pipeline/train.py` (ordinal K metric decode + skip softmax temperature calibration)
  - `ris_pytorch_pipeline/infer.py` (ordinal decode in inference)
- **Impact**: K training signal becomes easier and better-aligned with “under/over” semantics.

#### 2.4 Default training phase: run joint training now (avoid “k_only from scratch”)
We changed defaults so the backbone is allowed to learn K-correlated features immediately.

- **Change**:
  - `cfg.TRAIN_PHASE` default: `"geom"` → `"joint"`
  - `cfg.FREEZE_BACKBONE_FOR_K_PHASE` default: `True` → `False` (safer if you ever run `k_only`)
  - Boosted `PHASE_LOSS["joint"]["lam_K"]`: `0.5` → `1.0`
- **File**: `ris_pytorch_pipeline/configs.py`
- **Impact**: You don’t accidentally do an unproductive `k_only` run without a warm-start.

#### 2.5 Warm-start plumbing (`INIT_CKPT`)
We added a safe “escape hatch” to warm-start the model for staged training.

- **Change**:
  - Added `cfg.INIT_CKPT = ""` (optional path)
  - `Trainer.__init__` loads the checkpoint weights (supports either weights-only dict or `{ "model": state_dict }`)
  - Load happens **before** phase freezing so it doesn’t break module consistency.
- **Files**:
  - `ris_pytorch_pipeline/configs.py`
  - `ris_pytorch_pipeline/train.py`
- **Impact**: Enables clean workflows like `geom → joint` or `geom → k_only → joint` without hacking scripts.

#### 2.6 Inference: compute MDL/AIC on measurement-domain \(R_{yy}\)
We updated inference K selection to avoid the unreliable RIS-domain MDL.

- **Change**: In `hybrid_estimate_final()`:
  - Compute \(R_{yy}\) from measurement snapshots and run `estimate_k_ic_from_cov(Ryy, ...)`
  - Fallback to old RIS-domain behavior if an exception occurs.
- **File**: `ris_pytorch_pipeline/infer.py`
- **Impact**: Baseline K estimate becomes far more stable under L=16.

---

### 3) Diagnostics/tools added or updated

#### 3.1 MDL sanity check script
Added a standalone script to evaluate “is K identifiable” under current shards using the stored `R_samp`.

- **File**: `mdl_sanity_check.py`
- **Behavior**: Computes MDL K-hat and reports accuracy + confusion (intended for offline sanity).
- **Note**: In this environment, large runs can get killed; on your V100 machine you can run it at scale.

#### 3.2 K diagnostic script update
The diagnostic previously forced β=0 in EASY_OVERFIT mode, which removed the hybrid signal path.

- **Change**: `EASY_OVERFIT` now uses `K_DIAG_BETA` (default 0.50) instead of forcing 0.0.
- **File**: `run_k_diagnostic.py`

#### 3.3 Debug report artifact
- **File**: `K_ESTIMATION_DEBUG_REPORT.md`
- **Purpose**: Captures the investigative narrative and main findings around K estimation.

---

### 4) Files changed (index)

- **Updated**
  - `ris_pytorch_pipeline/train.py`
    - Optimizer grouping `HEAD_KEYS` fix
    - `INIT_CKPT` warm-start load
    - Surrogate K metric decode supports ordinal mode
    - Skip softmax temperature calibration when `K_HEAD_MODE=ordinal`
    - (Earlier) train loss logging correctness fixes + hybrid diagnostic messaging fix
  - `ris_pytorch_pipeline/configs.py`
    - Default phase set to `joint`
    - Safer freeze defaults
    - `lam_K` boosted in joint
    - Added `INIT_CKPT`, `K_HEAD_MODE`, `K_ORD_THRESH`
  - `ris_pytorch_pipeline/model.py`
    - K spectral features moved to measurement-domain \(R_{yy}\)
    - Added ordinal heads and `k_ord_logits` output
  - `ris_pytorch_pipeline/loss.py`
    - Added ordinal K loss path (weighted BCE)
  - `ris_pytorch_pipeline/infer.py`
    - MDL/AIC now computed on measurement-domain \(R_{yy}\)
    - Ordinal K decode added (plus legacy softmax path)
  - `run_k_diagnostic.py`
    - β behavior in EASY_OVERFIT updated (uses env `K_DIAG_BETA`)

- **Added**
  - `mdl_sanity_check.py`
  - `K_ESTIMATION_DEBUG_REPORT.md`
  - `IMPLEMENTATION_UPDATES_2026-01-12.md` (this file)

---

### 5) How to run (quick reference)

#### 5.1 K diagnostic (recommended)

```bash
cd /home/tahit/ris/MainMusic
EASY_OVERFIT=1 K_DIAG_BETA=0.5 python run_k_diagnostic.py |& tee run_k_diagnostic_after_kfix.log
```

#### 5.2 MDL sanity check on current shards

```bash
cd /home/tahit/ris/MainMusic
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python mdl_sanity_check.py --n_val 2000 --batch_size 64 --device cuda
```

#### 5.3 Joint training (now default)

```bash
cd /home/tahit/ris/MainMusic
python -m ris_pytorch_pipeline.ris_pipeline train
```

#### 5.4 Warm-start (optional)
Set `cfg.INIT_CKPT` to a `.pt`/`.ckpt` containing a model state dict, then run training as usual.

---

### 6) Recommended next move

If K is still unstable after these changes, the next decision should be data-side:
- Run an **L ablation** (`L=16 → 32 → 64`) and measure K accuracy vs compute cost,
while keeping the measurement-domain K estimator and ordinal head in place.


