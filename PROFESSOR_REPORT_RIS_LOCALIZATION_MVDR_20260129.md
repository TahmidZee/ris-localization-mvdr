# RIS-based Multi-Source Localization using MVDR-End Learning
**Progress report for Prof. \[Name\]**  
Date: 2026-01-29  
Author: Tahit / Team

---

## 1) Problem statement (what we are trying to solve)
We want to localize **multiple wireless sources** in 3D (azimuth \(\phi\), elevation \(\theta\), and range \(r\)) using a **Reconfigurable Intelligent Surface (RIS)** and a **base station (BS)**.

Key challenges:
- **Multiple sources** can be active at the same time (up to \(K_{\max}=5\)).
- We operate in **near-field** conditions (range matters; far-field approximations are not enough).
- Measurements are **noisy**, and performance must remain robust across **different SNRs**.

Our central design choice is to learn an internal representation (a covariance matrix) that is good for a physics-grounded estimator (MVDR), instead of directly regressing final positions only.

---

## 2) Big picture: how our pipeline works (end-to-end)
Each scene provides **L time snapshots**. For snapshot \(\ell\), the RIS applies a known phase-code \(c_\ell\) and the BS measures a complex vector \(y_\ell\).

### 2.1 Inputs per scene (what the model sees)
- **BS snapshots**: \(y \in \mathbb{C}^{L \times M}\) (M BS antennas)
- **RIS codes**: \(c \in \mathbb{C}^{L \times N}\) (N RIS elements)
- **BS→RIS channel (full operator)**: \(H_{\text{full}} \in \mathbb{C}^{M \times N}\)
- **SNR value**: `snr_db` (we pass it to the model so it can adapt to noise level)

### 2.2 Outputs and inference (what we do with the model output)
1. **Backbone network predicts a low-rank factorization** of the RIS-domain covariance matrix \(R\).
2. We construct a complex Hermitian covariance prediction:
   \[
   R_{\text{pred}} \approx A A^H
   \]
   (low-rank by construction, stable and PSD-like).
3. We create an **effective covariance** \(R_{\text{eff}}\) for inference by applying:
   - Hermitian symmetrization, trace normalization
   - diagonal loading (stability)
   - SNR-aware shrinkage (noise-robustness)
4. We run **MVDR beamforming** on \(R_{\text{eff}}\) to get a 2D angular spectrum over \((\phi,\theta)\).
5. We **detect peaks** (multi-source) directly on the MVDR spectrum.
6. We assign **range** by scanning a set of near-field range planes and selecting the \(r\) that maximizes MVDR at the chosen \((\phi,\theta)\).  
   **Important clarification**: in the current implementation, *range selection is MVDR-based*, not MUSIC-based.
7. (Optional) We run a **local Newton refinement** step that uses a MUSIC-style cost for sub-grid polishing. This is a refinement trick; the core detection remains MVDR-driven.

Outcome: we output a set of source hypotheses \(\{(\phi_i,\theta_i,r_i)\}\) with \(i=1..K\).

---

## 3) What MVDR is (explained from scratch)
### 3.1 What problem does MVDR solve?
When signals come from particular directions, the array response (steering vector) \(a(\phi,\theta,r)\) describes what the array “should” measure for that hypothesis.

MVDR asks:
> “If a signal came from this candidate direction, what is the best linear filter that passes that direction without distortion, while suppressing everything else (noise + other directions) as much as possible?”

### 3.2 MVDR spectrum (the core equation)
Given a covariance matrix \(R\) and a candidate steering vector \(a\), the MVDR score is:
\[
P_{\text{MVDR}} = \frac{1}{a^H R^{-1} a}
\]
Interpretation:
- If \(a\) matches a true source direction well, the denominator becomes small \(\Rightarrow\) **large peak**.
- \(R^{-1}\) makes the method **adaptive**: it suppresses interference and noise based on the statistics in \(R\).

### 3.3 Why MVDR is sensitive (and why it helps us)
MVDR performance depends strongly on the quality of the covariance matrix:
- If \(R\) has the wrong structure (wrong signal subspace), MVDR peaks collapse.
- If \(R\) is ill-conditioned, inversion becomes unstable.

So, if we can learn \(R_{\text{pred}}\) that is “MVDR-usable”, we get strong localization even under noise.

---

## 4) MVDR vs MUSIC (why we focus on MVDR for end metrics)
Both MVDR and MUSIC are classical array processing methods. Both use the covariance matrix \(R\), but in different ways.

### 4.1 MUSIC in simple words
MUSIC separates the space into:
- a **signal subspace** (spanned by top eigenvectors)
- a **noise subspace** (spanned by the remaining eigenvectors)

Then it scores a direction by checking how orthogonal the steering vector is to the noise subspace. MUSIC usually needs a good estimate of **how many sources \(K\)** exist (or it risks using the wrong subspace split).

### 4.2 MVDR in simple words
MVDR does not rely on an explicit signal/noise subspace split in the same way. It uses \(R^{-1}\) to form an adaptive beamformer and tends to be very sensitive to whether the covariance can be inverted stably.

### 4.3 Why we use MVDR as our primary “end-to-end” criterion
We use MVDR because:
- **It matches our production inference pipeline**: we detect peaks on an MVDR spectrum.
- **It is a strong stress-test** of whether \(R_{\text{pred}}\) is truly usable (subspace + conditioning).
- We verified that **MVDR works well on ground-truth covariance** \(R_{\text{true}}\). This means the classical beamformer and evaluation stack are correct; if performance is poor, the bottleneck is almost certainly the learned covariance.

We still keep MUSIC-based tools for:
- **local refinement** (Newton step on a MUSIC-style cost),
- diagnostics and comparisons,
but MVDR is the main “end metric” we care about.

---

## 5) Our neural architecture (what we learned and why it matters)
### 5.1 Backbone summary
The backbone is a **CNN + Transformer** hybrid that processes **L snapshot tokens**.

The most important modeling idea is that each snapshot has a *pair*:
- measurement \(y_\ell\)
- RIS code \(c_\ell\)

So we create **joint tokens per snapshot**: the model sees \((y_\ell, c_\ell)\) together, then learns patterns across \(\ell=1..L\) using a Transformer.

### 5.2 Key fixes we implemented after diagnosing MVDR collapse
We identified that earlier models could look “good” on auxiliary losses but still produce unusable covariances for MVDR. We implemented:
- **Joint snapshot tokens** (couple \(y_\ell\) and \(c_\ell\) directly): prevents losing the per-snapshot measurement/code relationship.
- **SNR embedding**: the model receives `snr_db` so it can learn noise-robust behavior.
- **LayerNorm before covariance heads**: stabilizes factor prediction and reduces training instability.
- **Factor column-norm leash**: prevents rare very-large factor outputs that can create unstable covariances.

These changes target the root cause: producing a covariance whose *signal subspace structure* and numerical properties are good enough for MVDR.

---

## 6) Losses we use (training objective) — explained simply
Our backbone does not directly output the final location. Instead, it outputs a covariance surrogate that must work well for MVDR. Therefore, we train using losses that encourage MVDR-usable covariance structure.

### 6.1 Covariance NMSE (`lam_cov`)
This compares predicted covariance to ground-truth covariance:
\[
\text{NMSE}(R_{\text{pred}},R_{\text{true}}) = \frac{\|R_{\text{pred}}-R_{\text{true}}\|_F^2}{\|R_{\text{true}}\|_F^2}
\]
This ensures the overall covariance is moving toward the correct target.

### 6.2 Subspace alignment (`lam_subspace_align`)
MVDR depends heavily on the **signal subspace**. Subspace alignment explicitly encourages the predicted covariance to carry the correct signal subspace structure.

This term was critical because low NMSE alone does not always imply the correct subspace geometry.

### 6.3 Peak contrast (`lam_peak_contrast`, small)
This is a light MVDR-aligned shaping term that nudges the spectrum to be more “peak-like” at true source locations. We keep it small for stability.

### 6.4 Auxiliary loss (`lam_aux`)
We also train auxiliary heads that predict \(\phi,\theta,r\) (as a regularizer).
This helps the model learn geometry early, but auxiliary accuracy alone is not the final goal; the final goal is MVDR-end performance.

### 6.5 Loss terms we intentionally disabled
Some “structure losses” (eigengap/margin terms) required differentiating through SVD/eigen operations and repeatedly caused numerical instability (NaNs). We disabled them to keep training stable and focused on MVDR-relevant structure.

---

## 7) Evaluation (how we measure progress)
### 7.1 Why we do end-to-end MVDR benchmarks
We evaluate on benchmark suites (B1/B2) that run the full inference pipeline and report:
- detection metrics: precision/recall/F1
- localization error in meters: RMSPE (3D position error)
- angular and range errors

This is the only reliable way to confirm that the learned covariance is actually usable for MVDR localization.

### 7.2 Matching multi-source predictions to ground truth
Because sources are unordered, we use Hungarian matching with **3D Cartesian distance** for a fair multi-source RMSPE evaluation.

---

## 8) Current progress (what we have achieved so far)
### 8.1 We isolated the core bottleneck
We confirmed:
- MVDR on the true covariance \(R_{\text{true}}\) works well.
- Therefore, poor results must come from the learned covariance \(R_{\text{pred}}\), not from the MVDR code.

### 8.2 We fixed major stability and architecture issues
We implemented the architecture upgrades listed above (joint tokens, SNR embedding, LayerNorm, factor leash), and simplified training by disabling a curriculum that caused phase-to-phase metric drift.

### 8.3 Critical discovery (root-cause bug): we were feeding a lossy channel surrogate to the model
We found a major information bottleneck in the original shard/model interface:
- The shards stored `H` as a per-snapshot **collapsed effective response** \(H_{\text{eff},\ell} = H_{\text{full}} c_\ell \in \mathbb{C}^{M}\).
- The backbone was built to consume only this collapsed \(H_{\text{eff}}\) (shape \([L,M]\)), not the true operator \(H_{\text{full}}\) (shape \([M,N]\)).

This collapses the RIS dimension and removes critical information needed to recover an RIS-domain covariance reliably. We implemented a **breaking-but-correct fix**:
- the backbone now consumes **\(H_{\text{full}}\in\mathbb{C}^{M\times N}\)** (stored in shards as `H_full`)
- training/inference/benchmarks now require `H_full`
- old checkpoints are not compatible (the model input dimensionality changed)

This fix matters because it changes the upstream inference problem from “guess RIS covariance from a collapsed proxy” to “predict covariance with access to the true sensing operator.”

### 8.4 Diagnostic result: R_samp vs R_true (decisive evidence about identifiability)
We ran a targeted diagnostic that compares MVDR localization using:
- \(R_{\text{true}}\): oracle RIS-domain covariance (ground truth)
- \(R_{\text{samp}}\): covariance estimated from the actual snapshots **on-the-fly** using \(y\), \(H_{\text{full}}\), and \(c\)

On a small test subset (N=200 scenes):
- **MVDR on \(R_{\text{true}}\)**: essentially perfect (cm-level RMSPE, ~0.02° angles)
- **MVDR on \(R_{\text{samp}}\)**:
  - Blind-K mode: median RMSPE ≈ **1.18 m** (angles ~1°)
  - Oracle-K mode: median RMSPE ≈ **3.96 m** (angles ~19°)

Interpretation (important):
- \(R_{\text{samp}}\) contains useful information but its spectrum is often **cluttered**; forcing exactly K peaks (Oracle-K) can pick weak/spurious peaks.
- This shows that in our current regime, “perfect covariance → perfect MVDR,” but “covariance estimated from limited measurements” is a major bottleneck.

### 8.5 Covariance failure diagnostic (decisive evidence of architectural flaw)
We ran a detailed covariance diagnostic on a 10-epoch checkpoint (with \(H_{\text{full}}\) input) comparing \(R_{\text{pred}}\) to \(R_{\text{true}}\):

| Metric | \(R_{\text{pred}}\) | \(R_{\text{true}}\) | Problem |
|--------|---------------------|---------------------|---------|
| NMSE | 1.32 | 0 | Very high — R_pred not matching R_true |
| Top-K energy concentration | 77% | 100% | Too diffuse — energy leaks to noise eigenvectors |
| Subspace overlap (top-K eigenvectors) | 29% | 100% | **Critical**: eigenvector directions are wrong |
| MDL K estimation accuracy | 24% | 100% | Model-order estimation failing |
| MVDR at GT location | 0.59 (higher baseline) | 0.33 (sharp peak) | Smooth spectrum, no sharp peaks |

**Root cause identified**: The covariance head (`cov_fact_angle`, `cov_fact_range`) was predicting ~5,120 free parameters disconnected from the geometry predictions (aux heads). Even though aux heads were learning well (angles/range improving), the covariance factors were drifting to near-identity-like matrices that erase MVDR peaks.

### 8.6 Current status: implementing structural fix
We identified that **loss weight tuning alone has ~60% confidence** of fixing this, whereas a **structural fix has ~90% confidence**.

The structural fix changes the architecture so that covariance is **constructed from geometry predictions** rather than predicted as free factors:

\[
R_{\text{pred}} = \sum_{k=1}^{K_{\max}} p_k \, a(\phi_k, \theta_k, r_k) \, a(\phi_k, \theta_k, r_k)^H + \sigma^2 I
\]

where \((\phi_k, \theta_k, r_k, p_k)\) are predicted by the network, and \(a(\cdot)\) is the near-field steering vector.

**Why this solves the diagnosed issues:**
| Audit finding | How structural fix solves it |
|---------------|------------------------------|
| Subspace overlap 29% | **By construction**: eigenvectors are steering vectors at predicted (φ,θ,r) |
| Top-K energy 77% vs 100% | **Guaranteed**: R is rank-K by construction |
| MDL accuracy 24% | **Follows automatically**: clean eigenvalue gap from low-rank structure |
| Cov head disconnected from aux | **Eliminated**: R is built *from* aux predictions |

This reduces the effective output dimensionality from \(O(N^2)\) free parameters to \(O(K)\) geometry parameters.

---

## 9) Paper narrative (geometry-aware covariance prediction)

> We propose a geometry-aware covariance prediction architecture where the neural network predicts source locations \((\phi_k, \theta_k, r_k)\) and powers \(p_k\), and the covariance is constructed analytically from near-field steering vectors. This design enforces the correct signal subspace structure by construction, ensuring MVDR compatibility without requiring explicit subspace alignment losses. The approach reduces the effective output dimensionality from \(O(N^2)\) to \(O(K)\), dramatically simplifying the learning problem while matching the physical forward model. By leveraging the known steering manifold, the network focuses on the well-posed geometry estimation task rather than the ill-posed high-dimensional covariance recovery problem.

---

## 10) Phased plan forward

### Phase A: Structured R for LOS-dominant UE→RIS (current narrowband + future wideband)
- Predict geometry \((\phi, \theta, r)\) + per-source power \(p\) from backbone
- Construct \(R\) analytically using near-field steering vectors
- Works for current narrowband M64/N256/L64 setup
- Extends directly to wideband Phase A (OFDM with LOS-dominant UE→RIS)

### Phase B: Controlled multipath extension (future wideband with TR 38.901)
- If multipath on UE→RIS is needed, extend to multi-subpath per source:
  - Predict \(P\) subpaths per source: \((\phi_{k,p}, \theta_{k,p}, r_{k,p}, p_{k,p})\)
  - Rank increases from \(K\) to \(K \cdot P\), still structured and MVDR-compatible
- Alternative: structured + residual hybrid for non-idealities

---

## 11) Immediate next steps
1. **Implement the structural fix** (geometry-aware covariance construction):
   - Add `aux_power_head` to predict per-source power/confidence
   - Replace free-form covariance factors with `build_structured_R()` function
   - Remove disconnected `cov_angle_head` and `cov_range_head`
2. **Run 10-epoch validation** on current M64/N256/L64 shards
3. **Run diagnostic** — expect subspace overlap > 0.8 immediately (by construction)
4. **Run B1 benchmark** — expect F1 > 0, RMSPE approaching R_samp baseline (~0.4m)
5. **Decision gate for wideband**:
   - If narrowband works well → document results, move to OFDM Phase A
   - If still struggling → analyze residual errors before adding wideband complexity

---

## Appendix: “Is range refinement MUSIC-based?”
**In the current inference implementation, range selection is MVDR-based**:
- after selecting a \((\phi,\theta)\) peak, we scan a set of candidate ranges (range planes) and choose the range that maximizes MVDR at that direction.

We do have an optional Newton refinement routine that uses a MUSIC-style cost for local optimization, but that is a polishing step and not the primary range estimator.

