# K-Free Multi-Source Near-Field Localization via MVDR

## Executive Summary

This document describes a PhD-level implementation plan for **K-free multi-source near-field localization** using RIS-coded measurements. The key innovation is replacing K-dependent MUSIC with **MVDR/Capon spectrum** for robust source detection, while retaining the learned covariance enhancement (hybrid `R_blend`).

**Publication-ready narrative:**
> "We propose a K-free multi-source localization framework for RIS-assisted near-field sensing. By combining neural network-enhanced covariance estimation with MVDR spectrum analysis and coarse-to-fine search, we achieve state-of-the-art localization accuracy without requiring explicit model-order estimation."

---

## 1. Problem Statement

### Current Limitations
1. **K-dependency**: MUSIC requires knowing the number of sources K to split signal/noise subspaces
2. **Mode collapse**: K-head consistently predicts wrong K, causing MUSIC to fail
3. **Multi-source fragility**: Per-slot angle/range heads suffer from permutation ambiguity and duplicate predictions
4. **Eigenvalue identifiability**: With fixed source signals across snapshots, Ryy eigenvalues don't reliably indicate K

### Our Solution
Replace the "predict K → run MUSIC" paradigm with:
1. **K-free MVDR spectrum** for candidate generation
2. **Detection-based source counting** (peaks emerge from thresholding)
3. **Coarse-to-fine search** for computational efficiency

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXISTING (KEEP)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  RIS-Coded Measurements y[L×M]                                              │
│         ↓                                                                   │
│  NN Backbone (Transformer + CNN)                                            │
│         ↓                                                                   │
│  R_pred [N×N] ←──── Learned covariance prediction                           │
│         ↓                                                                   │
│  R_blend = β·R_pred + (1-β)·R_samp ←── Hybrid blending                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                           NEW (IMPLEMENT)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stage 1: 2.5D MVDR Spectrum                                                │
│         ↓                                                                   │
│  Stage 2: 3D NMS + Candidate Selection                                      │
│         ↓                                                                   │
│  Stage 3: Local 3D Refinement                                               │
│         ↓                                                                   │
│  Stage 4: Peak Detection (CFAR/Threshold)                                   │
│         ↓                                                                   │
│  Output: [(φ₁,θ₁,r₁), (φ₂,θ₂,r₂), ...] with confidences                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. MVDR vs MUSIC: Why MVDR?

### MUSIC Spectrum
```
P_MUSIC(φ,θ,r) = 1 / (a^H · U_n · U_n^H · a)
```
- **Requires K**: Must know noise subspace dimension (N - K)
- **Super-resolution**: Very sharp peaks when K is correct
- **Failure mode**: Wrong K → completely wrong peaks

### MVDR/Capon Spectrum
```
P_MVDR(φ,θ,r) = 1 / (a^H · (R + δI)^{-1} · a)
```
- **K-free**: No subspace split needed
- **Robust**: Diagonal loading (δ) handles ill-conditioning
- **Slightly broader peaks**: But stable and reliable

### Decision Matrix

| Criterion | MUSIC | MVDR |
|-----------|-------|------|
| Needs K | ✗ Yes | ✓ No |
| Resolution | Higher | Good |
| Robustness | Fragile | Strong |
| Multi-source | K-dependent | Natural |
| **Recommendation** | Refinement only | **Primary** |

---

## 4. Grid Design

### Current Grid (Suboptimal)
- φ: 361 points over [-60°, +60°]
- θ: 181 points over [-30°, +30°]
- r: 5 planes at [0.6, 1.5, 3.0, 6.0, 9.0] m

### Improved Grid (Recommended)

#### Range Planes (Log-Spaced for Near-Field)
```python
# Old (linear-ish, sparse near-field)
RANGE_PLANES_OLD = [0.6, 1.5, 3.0, 6.0, 9.0]  # 5 planes

# New (log-spaced, dense near-field)
RANGE_PLANES_NEW = [0.55, 0.85, 1.3, 2.0, 3.5, 6.0, 9.5]  # 7 planes
```

**Rationale**: Near-field localization is most sensitive at close range. Phase curvature ∝ 1/r², so errors at r=0.5m matter more than at r=9m.

#### Total Grid Points
| Config | Points | Compute |
|--------|--------|---------|
| 2.5D (7 planes) | 361 × 181 × 7 = **457k** | Fast |
| Full 3D (fine) | 361 × 181 × 121 = **7.9M** | Too slow |
| Local refinement | 41 × 41 × 41 = **69k/candidate** | Fast |

---

## 5. Algorithm Details

### Stage 1: 2.5D MVDR Candidate Generation

**Input**: `R_blend` [N×N] from NN backbone

**Algorithm**:
```python
def compute_mvdr_spectrum_2_5d(R_blend, steering_vectors, range_planes, delta=1e-2):
    N = R_blend.shape[0]
    
    # Step 1: Regularize and invert (ONCE per sample)
    R_reg = R_blend + delta * torch.trace(R_blend) / N * torch.eye(N)
    R_inv = torch.linalg.inv(R_reg)
    
    # Step 2: For each range plane, compute 2D spectrum
    spectra = []
    for r in range_planes:
        A = steering_nearfield_grid(phi_grid, theta_grid, r)  # [G_phi, G_theta, N]
        P = 1.0 / real(sum_n(A * (A @ R_inv).conj()))  # MVDR formula
        spectra.append(P)
    
    # Step 3: Stack into 3D volume
    return torch.stack(spectra, dim=-1)  # [G_phi, G_theta, n_planes]
```

**Output**: MVDR spectrum `P[361, 181, 7]`

### Stage 2: 3D NMS + Candidate Selection

**Algorithm**:
```python
def extract_candidates(spectrum, n_candidates=20, min_sep_deg=5.0, min_sep_r=0.5):
    # Step 1: Find all local maxima (3D)
    maxima = find_local_maxima_3d(spectrum)
    
    # Step 2: Sort by spectrum value
    maxima = sorted(maxima, key=lambda x: spectrum[x], reverse=True)
    
    # Step 3: NMS - suppress peaks too close to stronger peaks
    candidates = []
    for peak in maxima:
        if not too_close_to_existing(peak, candidates, min_sep_deg, min_sep_r):
            candidates.append(peak)
        if len(candidates) >= n_candidates:
            break
    
    return candidates
```

**Output**: List of `(φ_idx, θ_idx, r_idx)` candidates

### Stage 3: Local 3D Refinement

**Per candidate** `(φ₀, θ₀, r₀)`:

```python
def local_refine(R_inv, phi0, theta0, r0, method='mvdr'):
    # Local grid: ±2° at 0.1° for angles, ±1m at 0.05m for range
    phi_local = linspace(phi0 - 2, phi0 + 2, 41)    # degrees
    theta_local = linspace(theta0 - 2, theta0 + 2, 41)
    r_local = linspace(max(0.5, r0-1), min(10, r0+1), 41)
    
    # Compute local spectrum
    if method == 'mvdr':
        P_local = compute_mvdr_spectrum_3d(R_inv, phi_local, theta_local, r_local)
    elif method == 'music_ensemble':
        P_local = compute_music_ensemble_3d(R, phi_local, theta_local, r_local, K_range=[1,2,3,4,5])
    
    # Find peak with sub-grid refinement
    peak_idx = argmax(P_local)
    phi_ref, theta_ref, r_ref = parabolic_refine_3d(P_local, peak_idx, phi_local, theta_local, r_local)
    
    return phi_ref, theta_ref, r_ref, P_local[peak_idx]
```

**Output**: Refined `(φ, θ, r, confidence)` per candidate

### Stage 4: Peak Detection (Source Selection)

**No explicit K estimation.** Use detection logic:

```python
def detect_sources(candidates, threshold_db=12.0, max_sources=5):
    # Sort by confidence
    candidates = sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    max_conf = candidates[0].confidence
    threshold = max_conf / (10 ** (threshold_db / 10))  # X dB below max
    
    # Keep peaks above threshold
    sources = [c for c in candidates if c.confidence >= threshold]
    
    # Optional: cap at max_sources
    return sources[:max_sources]
```

**Output**: Final source list `[(φ₁,θ₁,r₁,conf₁), ...]`

---

## 6. Effect of Increasing L (Snapshots)

### Current: L = 16

| Aspect | Status |
|--------|--------|
| Sample covariance quality | Poor (L << N = 144) |
| NN contribution | Critical (compensates for noisy R_samp) |
| R_blend ratio | β ≈ 0.3-0.4 works best |
| Eigenvalue estimation | Unreliable |

### L = 32

| Aspect | Expected Change |
|--------|-----------------|
| Sample covariance quality | **Improved** (~√2 better) |
| NN contribution | Still important |
| R_blend ratio | Can increase β toward 0.5 |
| MVDR spectrum | **Sharper peaks** |
| **φ/θ RMSE** | **~20-30% improvement** |
| **r RMSE** | **~20-30% improvement** |

### L = 64

| Aspect | Expected Change |
|--------|-----------------|
| Sample covariance quality | **Good** (but still L < N) |
| NN contribution | Less critical, but still helps |
| R_blend ratio | Can push β toward 0.6-0.7 |
| MVDR spectrum | **Much sharper peaks** |
| **φ/θ RMSE** | **~40-50% improvement vs L=16** |
| **r RMSE** | **~40-50% improvement** |
| Multi-source separation | **Significantly better** |

### L = 144+ (Full Rank)

| Aspect | Expected |
|--------|----------|
| Sample covariance | Invertible without heavy regularization |
| NN contribution | Optional (nice-to-have) |
| MVDR = MUSIC (approx) | Subspaces become clear |

### Recommendation

| L | Use Case | Thesis Story |
|---|----------|--------------|
| **16** | Ultra-low pilot, challenging | "Works even at L=16" |
| **32** | Sweet spot for demos | "Reliable performance" |
| **64** | Maximum quality | "SOTA accuracy" |

**For PhD publication**: Show results at L=16, 32, 64 to demonstrate robustness across pilot budgets.

---

## 7. Implementation Checklist

### Step 1: MVDRSpectrumEstimator (music_gpu.py)
- [ ] Add `_compute_spectrum_mvdr()` method
- [ ] Add `compute_mvdr_2_5d()` for 2.5D search
- [ ] Add log-spaced range planes option
- [ ] Unit test on synthetic data

### Step 2: Visualization (eval_mvdr.py)
- [ ] Load test samples
- [ ] Compute MVDR spectrum
- [ ] Plot 2D slices at each range plane
- [ ] Overlay ground truth source locations
- [ ] Assess spectrum quality

### Step 3: Local Refinement (music_gpu.py)
- [ ] Add `local_refine_3d()` method
- [ ] Add 3D NMS implementation
- [ ] Add `extract_candidates()` 
- [ ] Benchmark speed

### Step 4: Detection Logic (music_gpu.py)
- [ ] Add `detect_sources()` with CFAR threshold
- [ ] Add confidence scoring
- [ ] Test on multi-source scenes

### Step 5: Benchmark (benchmark.py)
- [ ] Compare MVDR vs current MUSIC
- [ ] Measure φ, θ, r RMSE
- [ ] Measure multi-source recall
- [ ] Generate publication-ready plots

### Step 6: Optional CNN Refinement (model.py)
- [ ] Add `SpectrumRefiner` module
- [ ] Gaussian blob supervision
- [ ] Training loop integration
- [ ] Ablation study

---

## 8. Expected Results

### Localization Accuracy (L=16)

| Metric | Current (MUSIC+K) | MVDR K-free | With CNN Refiner |
|--------|-------------------|-------------|------------------|
| φ RMSE | 1.5-2.5° | 0.8-1.5° | 0.5-1.0° |
| θ RMSE | 1.5-2.5° | 0.8-1.5° | 0.5-1.0° |
| r RMSE | 0.5-1.0m | 0.3-0.6m | 0.2-0.4m |
| K accuracy | ~70% (fragile) | N/A | N/A |
| Multi-source recall | ~50% | ~80% | ~90% |

### Localization Accuracy (L=32)

| Metric | MVDR K-free | With CNN Refiner |
|--------|-------------|------------------|
| φ RMSE | 0.5-1.0° | 0.3-0.6° |
| θ RMSE | 0.5-1.0° | 0.3-0.6° |
| r RMSE | 0.2-0.4m | 0.1-0.3m |
| Multi-source recall | ~85% | ~95% |

### Localization Accuracy (L=64)

| Metric | MVDR K-free | With CNN Refiner |
|--------|-------------|------------------|
| φ RMSE | 0.3-0.6° | 0.2-0.4° |
| θ RMSE | 0.3-0.6° | 0.2-0.4° |
| r RMSE | 0.1-0.3m | 0.05-0.15m |
| Multi-source recall | ~90% | ~98% |

---

## 9. Publication Contributions

1. **Hybrid Covariance Enhancement**: NN-predicted covariance blended with sample covariance for low-pilot regime

2. **K-Free Localization**: MVDR-based spectrum avoids fragile model-order estimation

3. **Coarse-to-Fine Search**: 2.5D candidate generation + 3D local refinement for efficiency

4. **Multi-Source Detection**: Peak-based detection with CFAR thresholding

5. **(Optional) Learned Spectrum Refinement**: CNN denoiser for SOTA accuracy

---

## 10. Files to Modify/Create

| File | Changes |
|------|---------|
| `music_gpu.py` | Add MVDR methods, 3D NMS, local refinement |
| `eval_mvdr.py` (new) | Visualization and debugging |
| `model.py` | (Optional) Add SpectrumRefiner |
| `benchmark.py` | Add MVDR vs MUSIC comparison |
| `configs.py` | Add MVDR-specific hyperparameters |

---

## 11. Timeline

| Week | Deliverable |
|------|-------------|
| 1 | Steps 1-2: MVDR implementation + visualization |
| 2 | Steps 3-4: Refinement + detection logic |
| 3 | Step 5: Benchmarking + tuning |
| 4 | Step 6: (Optional) CNN refinement head |
| 5 | Paper writing + plots |

---

*Document created: 2026-01-14*
*Status: Ready for implementation*

