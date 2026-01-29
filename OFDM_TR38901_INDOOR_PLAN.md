# Plan: NR-like OFDM + 3GPP TR 38.901 Indoor Office Channel for RIS Localization
Date: 2026-01-29  
Repo: `ris/MainMusic` (RIS localization + MVDR pipeline)

This document is a **paper-ready implementation plan** for upgrading our simulator and pipeline from a narrowband snapshot model to a **realistic, defensible OFDM wideband model** aligned with:
- **3GPP TR 38.901** (Indoor Office clustered multipath channel model)
- **3GPP TS 38.211** (NR OFDM numerology and slot structure)
- **3GPP TS 38.104** (NRB tables for channel bandwidth / SCS → occupied RBs)

It also records **explicit caveats** (what we simplify) and **what we do not implement** (and why).

---

## 1) Motivation (why OFDM + TR 38.901)

### 1.1 What problem we are fixing
Our narrowband setup is physically plausible for some regimes, but for **range estimation** and **multipath robustness**, narrowband is inherently fragile:
- Range is primarily inferred from **near-field curvature** (spherical wavefront effects), which is sensitive to SNR, calibration, and aperture.
- In indoor environments, **multipath** is real and can corrupt covariance subspace structure; narrowband makes separation difficult.

### 1.2 What OFDM gives us
Moving to wideband OFDM provides:
- **Frequency diversity**: more independent observations per RIS configuration.
- **Delay-resolved information**: range becomes (partly) supported by frequency-dependent phase slopes, not only curvature.
- A realistic link to deployable systems: **FR1 mid-band NR-like**.

The goal is not to perfectly emulate a full NR PHY stack; the goal is a **defensible “NR-like pilot OFDM measurement model”** that increases identifiability while staying computationally feasible.

---

## 2) Scenario definition (what we claim we simulate)

### 2.1 Deployment
- **Environment**: Indoor office / lab / industrial hall (room scale 10–30 m).
- **Carrier**: FR1 mid-band, e.g. **3.5 GHz**.
- **Topology**: **TDD uplink pilots** (UE transmits pilots, BS receives), RIS applies a programmable phase profile per pilot symbol.
- **Ranges**: Typically **2–10 m** UE-to-RIS/BS-scale distances (adjustable).
- **Motion assumption**: UE quasi-static over the pilot burst (coherence over a few ms).

### 2.2 Array/RIS dimensions (baseline vs moderate regime)
We will report results over at least two regimes:

- **Moderate identifiability regime (recommended baseline for showing method works)**:
  - BS antennas \(M\): **32 or 64** (e.g., 4×8 or 8×8 UPA)
  - RIS elements \(N\): **16×16 (256)** or **20×20 (400)**
- **Stress regime (hard case)**:
  - BS \(M=16\) (4×4)
  - RIS \(N=12×12=144\)

**Why**: many “headline” results in literature use large aperture and/or large diversity. Reporting a moderate regime and a stress regime is standard and strengthens the story.

---

## 3) OFDM numerology (NR-like, FR1)

We adopt an NR-like OFDM profile.

### 3.1 Numerology (TS 38.211-aligned)
- **Subcarrier spacing (SCS)**: **30 kHz** (NR μ=1).
- **Cyclic prefix**: **Normal CP**.
- **Slot structure**: **14 OFDM symbols per slot**, slot duration **0.5 ms** for μ=1.

### 3.2 Bandwidth configuration (TS 38.104 tables)
Choose one of:

#### Option A (recommended compute/realism sweet spot): 50 MHz
- Channel BW: **50 MHz**
- At 30 kHz SCS: **NRB ≈ 133**
- Active subcarriers: \(N_{sc} = 12 \cdot 133 = 1596\)

#### Option B (heavier, “serious NR carrier”): 100 MHz
- Channel BW: **100 MHz**
- At 30 kHz SCS: **NRB ≈ 273**
- Active subcarriers: \(N_{sc} = 12 \cdot 273 = 3276\)

### 3.3 FFT size (implementation detail; not mandated by 3GPP)
We choose telecom-friendly sampling rates:
- 50 MHz: **NFFT=2048**, \(F_s = 2048 \cdot 30\,\text{kHz} = 61.44\,\text{MHz}\)
- 100 MHz: **NFFT=4096**, \(F_s = 4096 \cdot 30\,\text{kHz} = 122.88\,\text{MHz}\)

**Caveat**: NFFT is a design choice; in the paper we should not claim it is mandated.

---

## 4) Pilot model (what we actually simulate)

### 4.1 We do NOT simulate all subcarriers
Simulating all 1596/3276 active tones is overkill for our ML pipeline and not required to be realistic.

Instead, we model **pilot/CSI-RS-like frequency samples**:
- Select **F pilot tones**, uniformly spaced across the occupied band.

Recommended:
- 50 MHz: **F = 256**
- 100 MHz: **F = 512** (or 1024 if feasible)

### 4.2 Mapping into our pipeline (paper sentence)
**One RIS code = one pilot OFDM symbol.**

For RIS configuration \(\ell\), the BS observes:
\[
y[\ell, k, m], \quad \ell=1..L,\; k=1..F,\; m=1..M
\]

We store:
- `y`: **[L, F, M, 2]**
- `codes`: **[L, N, 2]**
- `H_full`: channel parameters sufficient to compute \(H[k]\) for each pilot tone (see Section 5).

Defensible paper sentence:
> “We apply **L** RIS configurations across **L pilot OFDM symbols** within a coherence interval. Each symbol provides **F** frequency samples, yielding diversity proportional to **L×F**.”

### 4.3 Coherence check (why L=32–64 is realistic)
At μ=1, one slot is 0.5 ms with 14 symbols. If we use **one pilot symbol per RIS configuration**, then:
- L=64 pilot symbols spans ~64/14 ≈ 4.6 slots ≈ **2.3 ms**.
This is plausible indoors for slow motion.

---

## 5) Channel model: TR 38.901 Indoor Office (what we implement)

### 5.1 What “TR 38.901 style” means in practice
We simulate clustered multipath with:
- LOS + NLOS components (Rician/Rayleigh mixture depending on scenario)
- multiple clusters, each with multiple rays
- realistic delay spread and angular spreads

We will select the **Indoor Office** model family (TR 38.901) and parameterize:
- RMS delay spread (tens to hundreds of ns depending on office type)
- number of clusters/rays
- K-factor / LOS probability

### 5.2 Frequency-selective channel representation
For realism, we should model frequency selectivity:

Option 1 (recommended): **Tap-domain channel**
- Generate per-path delays and complex gains.
- Build per-tone response:
\[
H[k] = \sum_{p} \alpha_p \exp(-j2\pi f_k \tau_p) \, a_{\text{BS}}(\cdot)\, a_{\text{RIS}}(\cdot)^H
\]
- This is the cleanest way to incorporate TR 38.901 delay behavior.

Option 2 (simplification for early ablation): **Frequency-flat H**
- Assume \(H[k] \approx H\) across 50 MHz.
- This is not strictly realistic indoors, but can be used for a controlled ablation.

**Paper caveat**: If we use Option 2, we must state it explicitly and treat it as an ablation, not the final “TR 38.901 wideband” result.

---

## 6) Measurement equation (OFDM pilot observation model)

For each RIS configuration \(\ell\) and pilot tone \(k\):

1) RIS applies phase code \(c_\ell \in \mathbb{C}^{N}\) (unit-modulus with quantization if modeled).
2) Effective per-tone sensing matrix:
\[
G_{\ell,k} = H[k] \,\mathrm{diag}(c_\ell)
\]
3) BS measurement:
\[
y[\ell,k] = G_{\ell,k}\,x[\ell,k] + n[\ell,k]
\]

Where \(x[\ell,k]\) represents the RIS-element domain excitation induced by the sources for that tone (depends on source symbols, geometry, and possibly frequency).

---

## 7) RIS hardware realism knobs (recommended)

To match real prototypes and avoid “ideal RIS” criticism:
- **Phase quantization**: 2–3 bits (configurable).
- **Reflection loss**: magnitude < 1 (e.g., 0.7–0.95).
- **Static per-element phase offsets** (calibration error).
- Optional slow drift over the burst (small).

---

## 8) What we will NOT implement (and why)

### 8.1 Full NR PHY (channel coding, DMRS mapping, MCS, HARQ)
Not required for localization physics or identifiability; it adds huge complexity and does not change the core measurement model we need.

### 8.2 Simulating all subcarriers
Unnecessary. We will simulate a **pilot subset F**; this is realistic and computationally manageable.

### 8.3 mmWave beam squint and FR2 numerology
We are targeting **FR1 indoor sub-6**. Beam squint is far less severe than FR2; adding it would be a different story.

### 8.4 “SOTA-by-default” claims
We will not claim SOTA unless we match the information budget (aperture/transmissions/bandwidth) of the strongest baselines in literature.

---

## 9) Evaluation protocol (what we will report)

### 9.1 Regimes
Report at least:
- Moderate identifiability regime (M up and/or N down)
- Stress regime (original M=16, N=144) for robustness

### 9.2 Metrics
- End-to-end MVDR localization: RMSPE (3D), angle RMSE, range RMSE
- Detection: TP/FP/FN, precision/recall/F1, and K_pred
- Blind-K vs Oracle-K (with correct interpretation)

### 9.3 Diagnostics
- Compare MVDR using:
  - \(R_{\text{true}}\)
  - \(R_{\text{samp}}\) (from snapshots using \(H_{\text{full}}\))
  - \(R_{\text{pred}}\) (learned)

This isolates whether the bottleneck is **physics**, **classical estimation**, or **learning**.

---

## 10) Implementation plan (engineering steps)

### Phase A: Few-tone wideband (fastest high-leverage)
1. Extend dataset to store `y[L,F,M,2]` for a chosen F (e.g., 16→64→256).
2. Keep the existing pipeline structure; treat frequency as an extra axis.
3. Update R_samp construction to use all tones (aggregate across k).
4. Re-run the `R_samp vs R_true` diagnostic under wideband.

### Phase B: Full NR-like tone set (still pilots, not full subcarriers)
1. Use TS 38.104-derived occupied bandwidth for 50/100 MHz.
2. Sample pilot tones uniformly across that band (F=256/512).
3. Implement TR 38.901 Indoor Office delay taps.

### Phase C: (Optional) full OFDM subcarrier simulation
Only if absolutely needed; likely unnecessary.

---

## 11) Key caveats to state in the paper
- We simulate an **NR-like pilot OFDM measurement model**, not the full NR PHY.
- Pilot-tone subsampling is realistic and compute-motivated.
- If using frequency-flat \(H\) initially, state it as an ablation; final results should use TR 38.901 delay taps.
- Coherence assumption: the pilot burst duration fits within indoor coherence for slow motion.

---

## 12) Paper paragraph templates (ready to paste)

### 12.1 Simulation scenario (deployment + geometry)
> **Scenario.** We consider RIS-assisted localization in an **indoor office** environment. A BS equipped with an \(M\)-element array and an RIS with \(N\) passive elements are deployed at fixed, known locations. A user equipment (UE) transmits uplink pilots while the RIS applies programmable phase configurations. The UE is assumed quasi-static over the pilot burst duration.

### 12.2 OFDM numerology and pilot sampling (NR-like, FR1)
> **OFDM numerology.** We adopt an NR-like OFDM profile in FR1 with **subcarrier spacing \(\Delta f=30\) kHz (μ=1)** and **normal cyclic prefix**, corresponding to **14 OFDM symbols per slot** and **0.5 ms slot duration** (TS 38.211). We consider channel bandwidths of **50 MHz** (NRB≈133) or **100 MHz** (NRB≈273), where the occupied subcarriers are \(N_{sc}=12\cdot\text{NRB}\) (TS 38.104).
>
> **Pilot-tone subsampling.** Rather than simulating all occupied subcarriers, we sample **\(F\)** pilot tones uniformly across the occupied band (CSI-RS-like). This yields frequency diversity while keeping computation tractable; unless stated otherwise we use \(F=256\) for 50 MHz or \(F=512\) for 100 MHz.

### 12.3 RIS coding protocol (mapping to L “snapshots”)
> **RIS coding over pilot symbols.** We apply **\(L\)** RIS phase configurations across **\(L\)** pilot OFDM symbols within a coherence interval (TDD uplink pilots). For RIS configuration \(\ell\) and pilot tone \(k\), the BS collects frequency-domain measurements \(y[\ell,k,m]\) across antennas \(m=1..M\). We store measurements as \(y\in\mathbb{C}^{L\times F\times M}\), RIS codes as \(c\in\mathbb{C}^{L\times N}\), and use the BS–RIS channel operator \(H\in\mathbb{C}^{M\times N}\) (or a tap-domain representation) to form the per-tone effective sensing matrix \(G_{\ell,k}=H[k]\mathrm{diag}(c_\ell)\).

### 12.4 Channel model (TR 38.901 Indoor Office)
> **Channel model.** The BS–RIS and RIS–UE channels are generated using a **3GPP TR 38.901 Indoor Office** clustered multipath model with realistic delay and angular spreads. To capture frequency selectivity over 50–100 MHz, we generate multipath taps and compute per-tone frequency responses via the standard phase rotation \(\exp(-j2\pi f_k \tau)\) (TR 38.901). A frequency-flat channel assumption, when used, is treated as an ablation and is explicitly labeled as such.

### 12.5 Model realism knobs (RIS non-idealities)
> **RIS non-idealities.** To avoid idealized assumptions, we optionally include RIS phase quantization (e.g., 2–3 bits), reflection magnitude \(<1\), and static per-element phase offsets. These settings are reported alongside results.

### 12.6 What we do NOT simulate (explicit scope statement)
> **Scope.** We do not simulate the full NR PHY (coding, HARQ, scheduler) because it does not change the core pilot measurement model for localization. We also do not require simulation of all occupied subcarriers; pilot-tone subsampling is used to retain realism while controlling computational cost.

