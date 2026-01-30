# Plan: NR-like OFDM + 3GPP TR 38.901 Indoor Office Channel for RIS Localization
Date: 2026-01-30 (Updated)  
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
- **Critical discovery**: With narrowband, even perfect sample covariance (`R_samp`) yields poor MVDR results (Oracle-K RMSPE ~4m), indicating fundamental identifiability limits.

### 1.2 What OFDM gives us
Moving to wideband OFDM provides:
- **Frequency diversity**: more independent observations per RIS configuration.
- **Delay-resolved information**: range becomes (partly) supported by frequency-dependent phase slopes, not only curvature.
- **More equations**: With F tones, we get L×F×M measurements instead of L×M, making the inverse problem better-posed.
- A realistic link to deployable systems: **FR1 mid-band NR-like**.

The goal is not to perfectly emulate a full NR PHY stack; the goal is a **defensible "NR-like pilot OFDM measurement model"** that increases identifiability while staying computationally feasible.

---

## 2) Scenario definition (what we claim we simulate)

### 2.1 Deployment
- **Environment**: Indoor office / lab / industrial hall (room scale 10–30 m).
- **Carrier**: FR1 mid-band, **3.5 GHz** (λ ≈ 0.086 m).
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

**Why**: many "headline" results in literature use large aperture and/or large diversity. Reporting a moderate regime and a stress regime is standard and strengthens the story.

### 2.3 Near-field / far-field check
For 3.5 GHz with RIS aperture D:
- N=256 (16×16), element spacing λ/2 → D ≈ 8 × 0.086 = 0.69 m
- Fraunhofer distance: \(d_F = \frac{2D^2}{\lambda} = \frac{2 \times 0.69^2}{0.086} \approx 11 \text{ m}\)

At ranges 2–10 m, we are in the **near-field** of the RIS, which is correct for our localization setup.

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

#### Option B (heavier, "serious NR carrier"): 100 MHz
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
- `H_taps`: tap-domain channel representation (see §5.2)

Defensible paper sentence:
> "We apply **L** RIS configurations across **L pilot OFDM symbols** within a coherence interval. Each symbol provides **F** frequency samples, yielding diversity proportional to **L×F**."

### 4.3 L and F are independent parameters
**Critical clarification**: L (number of RIS configurations) and F (number of frequency samples) are **orthogonal** diversity axes:
- **L** controls temporal/spatial diversity (different RIS phase patterns)
- **F** controls frequency diversity (different tones within one OFDM symbol)
- Total measurements scale as **L × F × M**

Changing from narrowband to wideband does **not** change L; it adds F as a new axis.

### 4.4 Coherence check (why L=32–64 is realistic)
At μ=1, one slot is 0.5 ms with 14 symbols. If we use **one pilot symbol per RIS configuration**, then:
- L=64 pilot symbols spans ~64/14 ≈ 4.6 slots ≈ **2.3 ms**.

**Coherence budget check (3.5 GHz, walking speed 1 m/s):**
\[
T_c \approx \frac{9 \lambda}{16 \pi v} = \frac{9 \times 0.086}{16 \pi \times 1} \approx 15 \text{ ms}
\]
Our pilot burst (2.3 ms) << \(T_c\), so the quasi-static assumption holds.

---

## 5) Channel model: TR 38.901 Indoor Office (what we implement)

### 5.1 What "TR 38.901 style" means in practice
We simulate clustered multipath with:
- LOS + NLOS components (Rician/Rayleigh mixture depending on scenario)
- multiple clusters, each with multiple rays
- realistic delay spread and angular spreads

We will select the **Indoor Office** model family (TR 38.901) and parameterize:
- RMS delay spread (tens to hundreds of ns depending on office type)
- number of clusters/rays
- K-factor / LOS probability

### 5.2 Frequency-selective channel representation (CRITICAL)

**Recommended approach: Tap-domain channel storage**

Rather than storing `H[k]` for every tone (huge: `[F, M, N, 2]`), we store the **tap-domain representation**:

```
H_taps = {
    "n_paths": int,                    # number of multipath components
    "alphas": [n_paths, 2],            # complex path gains (real, imag)
    "taus": [n_paths],                 # path delays in seconds
    "aod_az": [n_paths],               # AoD azimuth (BS side)
    "aod_el": [n_paths],               # AoD elevation (BS side)
    "aoa_az": [n_paths],               # AoA azimuth (RIS side)
    "aoa_el": [n_paths],               # AoA elevation (RIS side)
}
```

**On-the-fly H[k] computation** (in DataLoader or model forward):
\[
H[k] = \sum_{p=1}^{P} \alpha_p \exp(-j2\pi f_k \tau_p) \, a_{\text{BS}}(\text{AoD}_p)\, a_{\text{RIS}}(\text{AoA}_p)^H
\]

where \(f_k\) is the frequency of the k-th pilot tone.

**Why tap-domain**:
- Compact storage (P paths << F tones)
- Physically interpretable
- Easy to compute H[k] for any k

### 5.3 Frequency-flat ablation (optional baseline)
For controlled ablation, we can assume \(H[k] \approx H\) across 50 MHz (frequency-flat).
- This is not strictly realistic indoors, but can be used for a controlled ablation.
- **Paper caveat**: If we use this, we must state it explicitly and treat it as an ablation, not the final "TR 38.901 wideband" result.

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

**Per-tone effective channel** (useful for model input):
\[
H_{\text{eff}}[\ell,k] = H[k] \, c_\ell \in \mathbb{C}^{M}
\]

---

## 7) RIS hardware realism knobs (recommended)

To match real prototypes and avoid "ideal RIS" criticism:
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

### 8.4 "SOTA-by-default" claims
We will not claim SOTA unless we match the information budget (aperture/transmissions/bandwidth) of the strongest baselines in literature.

---

## 9) Model Interface for Wideband

### 9.1 Tensor shapes: Narrowband vs Wideband

| Tensor | Narrowband (current) | Wideband (Phase A) | Notes |
|--------|---------------------|---------------------|-------|
| `y` | `[L, M, 2]` | `[L, F, M, 2]` | F is new frequency axis |
| `codes` | `[L, N, 2]` | `[L, N, 2]` | Unchanged |
| `H_full` | `[M, N, 2]` | Deprecated | Use `H_taps` instead |
| `H_taps` | N/A | dict (see §5.2) | Compact tap-domain |
| `R_true` | `[N, N, 2]` | `[N, N, 2]` | Unchanged |

### 9.2 How the model consumes wideband data (Phase A)

**Option 1 (Recommended): Pool over F before transformer**
- Per snapshot ℓ: compute `y_pooled[ℓ] = pool_k(y[ℓ, k, :])` 
- Pooling can be: mean, learned 1D conv, or attention over k
- Transformer sees L tokens as before
- Output: single covariance `R_pred ∈ ℂ^{N×N}`

```
Input processing (Phase A, Option 1):
  y[L, F, M, 2] → freq_pool → y_pooled[L, M, 2]
  codes[L, N, 2] → (unchanged)
  H_taps → compute H[k] → pool → H_pooled[M, N, 2] (or just use H[0])
  
Transformer input: L tokens, each = concat(y_pooled[ℓ], codes[ℓ], H_proj)
Output: R_pred[N, N, 2]
```

**Option 2 (More expressive, higher cost): Flatten (L, F) into sequence**
- Each token is (y[ℓ,k], c_ℓ) — note: same code c_ℓ for all k in snapshot ℓ
- Sequence length = L × F
- Much more compute, but fully utilizes frequency information
- Output: single covariance `R_pred ∈ ℂ^{N×N}`

**Recommendation**: Start with Option 1 (pool over F). If performance is insufficient, try Option 2.

### 9.3 R_samp construction for wideband

The sample covariance aggregates over both snapshots and tones:
\[
R_{\text{samp}} = \frac{1}{LF} \sum_{\ell=1}^{L} \sum_{k=1}^{F} \hat{x}[\ell,k] \hat{x}[\ell,k]^H
\]

where \(\hat{x}[\ell,k]\) is the pseudo-inverse recovery of the RIS-domain signal:
\[
\hat{x}[\ell,k] = G_{\ell,k}^{\dagger} y[\ell,k]
\]

**Implementation**: This can be done in the DataLoader or as a separate preprocessing step.

### 9.4 H_eff vs H_full vs H_taps (clarification)

| Representation | Shape | When to use |
|----------------|-------|-------------|
| `H_eff` | `[L, M, 2]` | **Deprecated** — collapses information |
| `H_full` | `[M, N, 2]` | Narrowband only (frequency-flat) |
| `H_taps` | dict | Wideband (recommended) |
| `H[k]` | `[F, M, N, 2]` | Computed on-the-fly from `H_taps` |

**Key insight**: For wideband, the model needs access to per-tone channel information. Storing `H_taps` and computing `H[k]` on-the-fly is most efficient.

---

## 10) Phase A vs Phase B: Implementation Strategy

### 10.1 Phase A: "Wideband helps estimation, MVDR stays single-covariance"

**Goal**: Use OFDM tones to make the inverse problem better-posed, without rewriting MVDR.

**What changes**:
- Dataset tensors: `y[L, F, M, 2]`, `H_taps`
- Model input: pool over F, then process as before
- Model output: single `R_pred[N, N, 2]`
- MVDR inference: unchanged (single covariance)

**Implementation steps**:
1. Update `pregen.py` to generate wideband shards with `y[L, F, M, 2]` and `H_taps`
2. Update `dataset.py` to load wideband format
3. Update `model.py` to add frequency pooling layer
4. Update `train.py` and `infer.py` to handle new shapes
5. Run R_samp vs R_true diagnostic under wideband

**Paper claim**: "Wideband improves covariance estimation quality."

### 10.2 Phase B: "True wideband beamforming"

**Goal**: Make MVDR itself wideband for even better localization.

**What changes** (on top of Phase A):
- Model optionally outputs per-tone covariance: `R_pred[k]` for k=1..F (or subband)
- MVDR runs per-tone:
\[
P(\phi,\theta,r) = \sum_{k \in \mathcal{K}} w_k \, P_{\text{MVDR}}^{(k)}(\phi,\theta,r)
\]
- Peaks detected on combined spectrum

**Implementation steps**:
1. Modify model to output per-tone factors (optional)
2. Modify MVDR to loop over tones and combine spectra
3. Use noncoherent combining (sum power spectra)

**Paper claim**: "Wideband improves both estimation AND beamforming."

### 10.3 Decision gate: Phase A → Phase B

After completing Phase A (short training + R_samp diagnostic):

| R_samp MVDR-end result | Interpretation | Action |
|------------------------|----------------|--------|
| RMSPE ≤ 1.5 m, F1 ≥ 0.6 | Wideband physics is viable | Proceed to full training |
| RMSPE > 3 m | Still ill-posed | Increase F, L, M, or N |
| Full training works well | Phase A success | Phase B is optional polish |

### 10.4 Recommended implementation order

1. **Phase A, F=16** (minimal wideband): quick validation that pipeline works
2. **Phase A, F=64** (moderate wideband): proper diversity gain
3. **R_samp diagnostic**: verify physics works before training
4. **Full training** with Phase A
5. **Phase B** only if Phase A plateau is unsatisfactory

---

## 11) Config Parameter Updates (CRITICAL before implementation)

### 11.1 Current config (narrowband, outdated)
```python
cfg.WAVEL = 0.3              # 1 GHz — WRONG for 3.5 GHz
cfg.M = 16                   # BS antennas — too small
cfg.N_H, cfg.N_V = 12, 12    # RIS 144 elements — stress regime
cfg.L = 64                   # snapshots — OK
# F is missing!
```

### 11.2 Updated config (wideband, recommended)
```python
# === Carrier / wavelength ===
cfg.CARRIER_HZ = 3.5e9                          # 3.5 GHz FR1 mid-band
cfg.WAVEL = 3e8 / cfg.CARRIER_HZ                # ≈ 0.0857 m

# === Array dimensions (moderate regime) ===
cfg.M = 64                                       # BS antennas (8×8 UPA)
cfg.M_H, cfg.M_V = 8, 8                          # BS UPA layout
cfg.N_H, cfg.N_V = 16, 16                        # RIS elements → N = 256
cfg.N = cfg.N_H * cfg.N_V                        # 256

# === OFDM parameters (NEW) ===
cfg.F = 64                                       # pilot tones (start small; scale to 256)
cfg.BW_HZ = 50e6                                 # 50 MHz channel BW
cfg.SCS_HZ = 30e3                                # 30 kHz SCS (μ=1)
cfg.NFFT = 2048                                  # FFT size

# === Derived from OFDM ===
cfg.F_SPACING_HZ = cfg.BW_HZ / cfg.F             # pilot tone spacing
cfg.PILOT_FREQS = np.linspace(                   # pilot tone frequencies
    -cfg.BW_HZ/2, cfg.BW_HZ/2, cfg.F
) + cfg.CARRIER_HZ

# === Temporal snapshots (unchanged) ===
cfg.L = 64

# === Geometry derived ===
cfg.k0 = 2 * math.pi / cfg.WAVEL
cfg.d_H = cfg.d_V = 0.5 * cfg.WAVEL              # λ/2 spacing

# === Range (unchanged) ===
cfg.RANGE_R = (0.5, 10.0)
```

### 11.3 Stress regime config (for comparison)
```python
# Stress regime (original dimensions, wideband)
cfg.M = 16
cfg.N_H, cfg.N_V = 12, 12   # N = 144
cfg.F = 64                  # wideband helps even here
```

### 11.4 Memory estimate for M=64, N=256
- `H_full` shape: `[M, N, 2] = [64, 256, 2]` = 32 KB per sample
- `y` shape (wideband): `[L, F, M, 2] = [64, 64, 64, 2]` = 2 MB per sample
- For 100k training samples: ~200 GB shards (F=64)

Start with F=16 to validate pipeline, then scale up.

---

## 12) Evaluation protocol (what we will report)

### 12.1 Regimes
Report at least:
- Moderate identifiability regime (M=32, N=256, F=64 or 256)
- Stress regime (M=16, N=144, F=64) for robustness

### 12.2 Metrics
- End-to-end MVDR localization: RMSPE (3D), angle RMSE, range RMSE
- Detection: TP/FP/FN, precision/recall/F1, and K_pred
- Blind-K vs Oracle-K (with correct interpretation)

### 12.3 Diagnostics (CRITICAL before training)
- Compare MVDR using:
  - \(R_{\text{true}}\) — oracle covariance
  - \(R_{\text{samp}}\) (from wideband snapshots using \(H_{\text{taps}}\))
  - \(R_{\text{pred}}\) (learned)

This isolates whether the bottleneck is **physics**, **classical estimation**, or **learning**.

---

## 13) Implementation Roadmap

### Phase 0: Narrowband at new dimensions (FIRST)
**Goal**: Verify pipeline works at M=32, N=256 before adding frequency axis.

1. Update `configs.py` with new M, N, wavelength
2. Update `pregen.py` to generate new shards
3. Run `doctor` command to verify
4. Quick training run (10 epochs) to verify model trains
5. R_samp vs R_true diagnostic at new dimensions

**Exit criterion**: Narrowband R_true MVDR works (F1 > 0.9).

### Phase A1: Minimal wideband (F=16)
**Goal**: Validate wideband tensor flow.

1. Add `cfg.F = 16` to config
2. Update `pregen.py` to generate `y[L, F, M, 2]` and `H_taps`
3. Update `dataset.py` to load new format
4. Add frequency pooling to `model.py`
5. Run short training (10 epochs)
6. R_samp diagnostic

**Exit criterion**: Pipeline runs without errors; R_samp improves vs narrowband.

### Phase A2: Full wideband (F=64 or 256)
**Goal**: Get actual benefit.

1. Scale F to 64 or 256
2. Regenerate shards
3. Full training run (60+ epochs)
4. MVDR-end benchmarks (B1, B2)

**Exit criterion**: RMSPE ≤ 1.5 m, F1 ≥ 0.7

### Phase B: Wideband MVDR (optional)
1. Modify MVDR to run per-tone
2. Implement spectrum combination
3. Compare against Phase A

---

## 14) Key caveats to state in the paper

- We simulate an **NR-like pilot OFDM measurement model**, not the full NR PHY.
- Pilot-tone subsampling is realistic and compute-motivated.
- If using frequency-flat \(H\) initially, state it as an ablation; final results should use TR 38.901 delay taps.
- Coherence assumption: the pilot burst duration fits within indoor coherence for slow motion.
- We report both **moderate** and **stress** regimes to show robustness.

---

## 15) Paper paragraph templates (ready to paste)

### 15.1 Simulation scenario (deployment + geometry)
> **Scenario.** We consider RIS-assisted localization in an **indoor office** environment. A BS equipped with an \(M\)-element array and an RIS with \(N\) passive elements are deployed at fixed, known locations. A user equipment (UE) transmits uplink pilots while the RIS applies programmable phase configurations. The UE is assumed quasi-static over the pilot burst duration.

### 15.2 OFDM numerology and pilot sampling (NR-like, FR1)
> **OFDM numerology.** We adopt an NR-like OFDM profile in FR1 with **subcarrier spacing \(\Delta f=30\) kHz (μ=1)** and **normal cyclic prefix**, corresponding to **14 OFDM symbols per slot** and **0.5 ms slot duration** (TS 38.211). We consider channel bandwidths of **50 MHz** (NRB≈133) or **100 MHz** (NRB≈273), where the occupied subcarriers are \(N_{sc}=12\cdot\text{NRB}\) (TS 38.104).
>
> **Pilot-tone subsampling.** Rather than simulating all occupied subcarriers, we sample **\(F\)** pilot tones uniformly across the occupied band (CSI-RS-like). This yields frequency diversity while keeping computation tractable; unless stated otherwise we use \(F=256\) for 50 MHz or \(F=512\) for 100 MHz.

### 15.3 RIS coding protocol (mapping to L "snapshots")
> **RIS coding over pilot symbols.** We apply **\(L\)** RIS phase configurations across **\(L\)** pilot OFDM symbols within a coherence interval (TDD uplink pilots). For RIS configuration \(\ell\) and pilot tone \(k\), the BS collects frequency-domain measurements \(y[\ell,k,m]\) across antennas \(m=1..M\). We store measurements as \(y\in\mathbb{C}^{L\times F\times M}\), RIS codes as \(c\in\mathbb{C}^{L\times N}\), and use a tap-domain channel representation to compute per-tone sensing matrices \(G_{\ell,k}=H[k]\mathrm{diag}(c_\ell)\).

### 15.4 Channel model (TR 38.901 Indoor Office)
> **Channel model.** The BS–RIS and RIS–UE channels are generated using a **3GPP TR 38.901 Indoor Office** clustered multipath model with realistic delay and angular spreads. To capture frequency selectivity over 50–100 MHz, we generate multipath taps and compute per-tone frequency responses via the standard phase rotation \(\exp(-j2\pi f_k \tau)\) (TR 38.901). A frequency-flat channel assumption, when used, is treated as an ablation and is explicitly labeled as such.

### 15.5 Model realism knobs (RIS non-idealities)
> **RIS non-idealities.** To avoid idealized assumptions, we optionally include RIS phase quantization (e.g., 2–3 bits), reflection magnitude \(<1\), and static per-element phase offsets. These settings are reported alongside results.

### 15.6 What we do NOT simulate (explicit scope statement)
> **Scope.** We do not simulate the full NR PHY (coding, HARQ, scheduler) because it does not change the core pilot measurement model for localization. We also do not require simulation of all occupied subcarriers; pilot-tone subsampling is used to retain realism while controlling computational cost.

---

## 16) Appendix: Quick Reference Tables

### A1: Dimension mapping (narrowband → wideband)

| Symbol | Meaning | Narrowband | Wideband |
|--------|---------|------------|----------|
| M | BS antennas | 16 | 64 (8×8 UPA) |
| N | RIS elements | 144 | 256 (16×16) |
| L | RIS configurations | 64 | 64 |
| F | Pilot tones | 1 | 64–256 |
| λ | Wavelength | 0.3 m (1 GHz) | 0.086 m (3.5 GHz) |

### A2: File changes required

| File | Changes |
|------|---------|
| `configs.py` | Add CARRIER_HZ, F, BW_HZ, SCS_HZ; update M, N, WAVEL |
| `pregen.py` | Generate `y[L,F,M,2]`, store `H_taps` dict |
| `dataset.py` | Load `H_taps`, compute `H[k]` on-the-fly |
| `collate_fn.py` | Handle new `y` shape |
| `model.py` | Add freq pooling layer, update input projection |
| `train.py` | Pass new tensors to model |
| `infer.py` | Handle wideband inference |
| `loss.py` | May need wideband R_samp formula |
| `music_gpu.py` | Phase B only: per-tone MVDR |

### A3: Shard storage estimate

| Config | y size | H_taps size | Total per sample |
|--------|--------|-------------|------------------|
| L=64, F=64, M=64 | 64×64×64×2×4 = 2 MB | ~10 KB | ~2 MB |
| L=64, F=256, M=64 | 64×256×64×2×4 = 8 MB | ~10 KB | ~8 MB |

For 100k training samples at F=64: ~200 GB
For 100k training samples at F=256: ~800 GB

**Recommendation**: Start with F=16 (~50 GB), then scale to F=64 for full training.

