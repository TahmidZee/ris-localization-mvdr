## Globecom Pipeline Readiness Notes (2026-01-26)

### Context (what we discussed)
- **Electrically large RIS**: With `WAVEL=0.3m` and `12×12` elements at `d=λ/2`, the aperture per side is \((12-1)\cdot 0.15=1.65\) m ≈ **5.5λ**, i.e., electrically large.
- **mmWave impact**:
  - If you keep **element count fixed** and keep **`d=λ/2`**, the **physical** RIS size shrinks with \(λ\), and the **Rayleigh distance** \(R\approx 2D^2/λ\) also shrinks roughly with \(λ\) (because \(D\propto λ\)). Result: for the same 0.5–10 m ranges you may drift toward **far-field** unless you increase physical aperture / element count.
  - If you keep **physical aperture fixed** and move to smaller \(λ\), the RIS becomes **more electrically large**, typically improving resolvability—but you should model more realistic mmWave effects (impairments, blockage, multipath) to avoid “too ideal” results.
- **Target venue**: Globecom (expects clear novelty, strong baselines, and credible simulation assumptions + ablations).

### What the repo already models (verified in code)
- **RIS phase quantization**:
  - Implemented via `quantise_phase()` in `ris_pytorch_pipeline/physics.py`.
  - Driven by `mdl_cfg.PHASE_BITS` (default **3**).
- **Geometry / calibration-ish mismatch**:
  - RIS element position jitter via `mdl_cfg.ETA_PERTURB` in `ris_pytorch_pipeline/dataset.py` (jitter in meters proportional to `WAVEL/2`).
- **Domain-randomization / hardware non-idealities on RIS codes** (in `ris_pytorch_pipeline/dataset.py`):
  - Per-element **phase jitter** (`DR_PHASE_SIGMA`)
  - **Amplitude jitter** (`DR_AMP_JITTER`)
  - Element **dropout** (`DR_DROPOUT_P`)
  - Optional **grid offset** (sub-grid phase shift) (`GRID_OFFSET_*`)
  - Optional wavelength / spacing jitter knobs are present in overrides.

### What is *not* modeled (important Globecom realism gaps)
- **Receiver-chain impairments**: CFO, phase noise, IQ imbalance, timing offsets (not present in snapshot generator today).
- **Wideband effects**: beam squint / frequency-dependent steering; current pipeline is effectively **narrowband**.
- **Richer channel models**: current BS→RIS is a simplified Rician model; not a clustered multipath model with blockage statistics.

### Readiness verdict (for Globecom)
The pipeline is **research-grade and close**, but not yet “reviewer-proof” for a top conference unless you add (or at least ablate) a small set of realism + robustness items and present a clean experimental story.

This aligns with prior audits:
- See `SYSTEM_REALISM_AUDIT_20260126.md` (rank-deficiency + realism notes)
- See `R_SAMP_FIX_REPORT_20260125.md` (why `R_samp` can be MVDR-useless when `M<<N`, and what was fixed)

### Update (2026-01-27): Stage-2 HPO interpretation + loss policy
- **Stage-2 HPO was running MVDR-final correctly**, but the **MVDR-final signal was near-flat** because the backbone was not learning a usable covariance under the small per-trial budget (10k train / 1k val, early-stop ~16 epochs).
- **Interpretation**: not an Optuna bug; it’s “no learning ⇒ no reranking signal”.
- **Action**: validate learning with a **full training run** (100k train / 10k val) before spending more compute on Stage-2 reranking.
- **Backbone loss policy**: enable **subspace alignment** (`lam_subspace_align`) and keep **peak contrast** (`lam_peak_contrast`) **off** (defer peak shaping to SpectrumRefiner via heatmap supervision).

---

## Globecom Checklist (Minimum vs Stretch)

### 0) Paper framing (do first)
- [ ] **Claim**: One crisp sentence describing your contribution (e.g., “Learning an MVDR-usable covariance/whitener from compressive RIS measurements in an ill-posed `M<<N` regime, with detection-aware e2e selection.”).
- [ ] **Problem regime**: Explicitly state `M`, `N`, `L`, carrier, range regime (near-field vs far-field), and why it matters.
- [ ] **Define SOTA target**: “simulation SOTA” vs “deployment realism” (reviewers care that you’re honest and thorough).

### 1) Dataset + physics realism (Minimum viable)
- [ ] **Lock a frequency & justify**: choose sub-6 or mmWave, but be consistent with aperture/range regime.
- [ ] **Ablate quantization bits**: at least `{1,2,3,4}` bits.
- [ ] **Ablate calibration mismatch**: sweep `ETA_PERTURB`, `DR_PHASE_SIGMA`, `DR_AMP_JITTER`, `DR_DROPOUT_P`.
- [ ] **Blockage / intermittency**: add a simple on/off LoS blocker model (even if synthetic) and report sensitivity.
- [ ] **Noisy CSI**: add noise to `H_full` (or a simple estimator proxy) and ablate.

### 2) Wideband diversity (Stretch, but very strong)
- [ ] Add `S` subcarriers and make steering frequency-dependent; use effective diversity `M × L × S`.
- [ ] Report performance vs number of subcarriers; show how it resolves rank-deficiency.

### 3) Receiver impairments (Minimum viable = add one)
- [ ] Add **phase noise** OR **CFO** to snapshots (one impairment is enough for Globecom credibility if well-ablated).
- [ ] Ablate impairment magnitude and show graceful degradation.

### 4) Baselines (Minimum viable)
- [ ] **Oracle upper bound**: MVDR/MUSIC with `R_true` (already used in diagnostics).
- [ ] **Classical**: MVDR on `R_samp` + diagonal loading/shrinkage (report and explain failure in `M<<N`).
- [ ] **Matched filter / beamformer peaks** baseline.
- [ ] **Your method**: NN → `R_pred` → MVDR-first inference (production-aligned).
- [ ] **Hybrid**: optional `R_eff = (1-β)R_pred + β R_samp` (if enabled, show when it helps/hurts).

### 5) Ablations reviewers will expect (Minimum viable)
- [ ] **Snapshots**: `L ∈ {16, 32, 64, 128}` (you’re moving to **64** already).
- [ ] **Measurements**: `M` sweep (beams/BS antennas as appropriate).
- [ ] **#sources**: `K` sweep, and performance stratified by `K`.
- [ ] **SNR sweep** and report not only mean metrics but also error CDFs.
- [ ] **Generalization**: train on one geometry distribution, test on shifted distributions (ranges/FOV/angles).

### 6) Metrics & objective alignment (Minimum viable)
- [ ] Primary: **Precision/Recall/F1** with a clearly defined tolerance.
- [ ] Localization: **RMSE / median error** *conditioned on matched detections*, plus a miss-penalized aggregate metric.
- [ ] Ensure the **model selection / HPO objective** matches the paper’s primary metric (you already moved toward this; keep the narrative consistent).

### 7) Reproducibility (Minimum viable)
- [ ] One-command scripts to:
  - regenerate shards (with `L=64`)
  - run HPO
  - train final model from `best.json`
  - run benchmark suite and export tables/figs
- [ ] Fixed seeds, version pinning, and a minimal regression test gate.

**Note**: Some docs may still contain legacy “K-logit calibration” language from the removed K-head; before submission, clean docs so the repo story is consistent.

---

## Recommended “Globecom-ready in 2–3 weeks” execution plan
- **Week 1 (stability + baselines)**:
  - Lock frequency + scenario; regenerate shards for `L=64`.
  - Run HPO → train final → benchmarks.
  - Add 2–3 strong baselines and produce initial tables.
- **Week 2 (realism toggles + ablations)**:
  - Add (A) noisy `H_full` and (B) blockage OR phase noise/CFO.
  - Run ablation sweeps for `L`, `M`, bits, SNR; produce plots.
- **Week 3 (paper polish)**:
  - Tighten narrative around `M<<N` and learned priors; finalize reproducibility scripts and tables.

