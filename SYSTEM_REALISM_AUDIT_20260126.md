## System Realism Audit (2026-01-26)

### Goal
Assess whether the current simulation/setup is “unrealistic” relative to typical RIS near-field localization literature, and what to change (if anything) to stay on a credible path to SOTA.

### Current Setup (as implemented)
- **RIS size**: `N = 12×12 = 144` elements
- **BS antennas**: `M = 16` (per config used in diagnostics)
- **Time snapshots**: `L = 16`
- **Inference path**: NN predicts `R_pred` → MVDR peak detection (`hybrid_estimate_final`)

### Finding 1: Rank-deficiency is expected in RIS problems
In many RIS-aided positioning/localization formulations, the RIS element-domain is high dimensional (e.g., 64–256+ elements) while the BS has fewer RF chains/antennas and the UE often has 1–few antennas. Therefore **M < N is not inherently unrealistic**.

**What matters** is the *total measurement diversity* across:
- time snapshots / pilot symbols (L),
- frequency subcarriers (wideband),
- multiple frames,
- multiple BS arrays / panels,
- multiple UE antennas,
- multiple RIS phase configurations.

If you only use a small number of narrowband snapshots, the inverse problem becomes extremely ill-posed for physics-only covariance estimation (`R_samp`), which is exactly what we observed.

### Finding 2: Why `R_samp` being poor is not a red flag for realism
With `M << N`, reconstructing a RIS-domain covariance purely from `(y, H_full, codes)` without a strong prior can fail to recover the true near-field steering subspace. This is a limitation of the measurement model, not automatically “unrealistic simulation”.

**Implication for our pipeline:** the NN is the component that supplies the necessary prior. The design “NN → `R_pred` → MVDR” is a legitimate way to address rank-deficiency.

### Potential Unrealistic Assumptions to be aware of
These are common in papers, but still assumptions:
- **Perfect/known BS→RIS channel (`H_full`)**: many simulations assume perfect CSI. In practice, CSI acquisition is hard and costs overhead.
- **Idealized noise / hardware**: Gaussian noise, ideal calibration, perfect synchronization; real systems have impairments.
- **Near-field model fidelity**: steering vector conventions must match the simulated physics and any hardware array geometry.

None of these block SOTA in *simulation*; they just define what “SOTA” means (paper-SOTA vs deployment-SOTA).

### Recommended Path to SOTA with this design
1. **Treat `R_samp` as non-critical** (do not require it to be MVDR-good standalone).
2. **Optimize `R_pred` + MVDR-first inference**, using MVDR-final objective in HPO and full training.
3. If you want to improve realism and/or performance headroom:
   - **Add wideband diversity** (multiple subcarriers) so effective measurements scale like `M × (#subcarriers) × L`.
   - **Increase BS antennas** (e.g., 32/64) if your target system plausibly supports it.
   - **Model CSI acquisition** (noisy `H_full`, or replace `H_full` with an estimator) if deployment realism is required.

### Notes on literature comparison
I attempted to automatically fetch parameter tables from recent papers, but the web search tool in this environment returned irrelevant results. If you paste 2–3 target papers you want to match (PDF links or titles), I can extract their BS/RIS/L/frequency assumptions and benchmark our setup against them precisely.

