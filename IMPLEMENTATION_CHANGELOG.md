# Implementation Changelog: K-Free MVDR Localization

**Date:** January 16, 2026  
**Reference:** `MVDR_LOCALIZATION_PLAN.md`  
**Last Updated:** January 16, 2026 (Refiner mandatory + robust MVDR thresholding + repo cleanup)

---

## Executive Summary

This document details all code changes made to transition from a K-head classification approach to a **K-free MVDR peak detection** approach for multi-source near-field localization.

### Key Achievements
- ✅ Implemented MVDR spectrum computation (K-free)
- ✅ Added 2.5D → 3D refinement pipeline
- ✅ Created visualization/verification tools
- ✅ Removed all K-head related code (clean slate)
- ✅ **Made SpectrumRefiner mandatory in production inference** (MVDR → Refiner → peak picking)
- ✅ Added robust MVDR thresholding mode (`MVDR_THRESH_MODE="mad"`)
- ✅ Cleaned up redundant files
- ✅ Added SpectrumRefiner CNN for learned spectrum denoising
- ✅ Added heatmap supervision loss (Gaussian blob targets)

### Comprehensive Check Fixes (Post-Plan Audit)
- ✅ **Fixed SpectrumRefiner U-Net shape bug** on odd grid sizes (e.g. 181×361) by adding center-crop / symmetric pad before skip concatenations.
- ✅ **Relaxed overly-strict gradient assertions** in `UltimateHybridLoss` so refiner-only / frozen-backbone training doesn’t crash.
- ✅ **Switched inference to MVDR-first** (K-free) by rewriting `infer.py:hybrid_estimate_final()` to run MVDR peak detection on `R_eff`.
- ✅ **Removed remaining K-metric logic** from training surrogate scoring (updated `SURROGATE_METRIC_WEIGHTS` accordingly).
- ✅ **Gated noisy debug prints** in `angle_pipeline.build_sample_covariance_from_snapshots()` behind `cfg.DEBUG_BUILD_R_SAMP`.
- ✅ **Deleted legacy directories/scripts** (`archive_docs/`, `archive_scripts/`, `OldCode/`, outdated K-estimation reports/scripts, `train.py.bak`, stray CLI artifact files).
- ✅ **Implemented Option B Stage-2 SpectrumRefiner training** (freeze backbone, generate MVDR spectrum from low-rank factors, train refiner with heatmap loss).
- ✅ **Added CLI entrypoint** `python -m ris_pytorch_pipeline.ris_pipeline train-refiner` for Stage-2 refiner training.
- ✅ **Removed MDL from the production inference decision path** (kept only for ablations/debug where needed).
- ✅ **Repo hygiene cleanup**: committed deletion of legacy archives + removed a dangling submodule-style gitlink (`AI-Subspace-Methods`) and ignored local copies.

### Stage-2 SpectrumRefiner Training (Option B) — New Additions

**Goal:** After training a strong covariance backbone, train a lightweight CNN to denoise/sharpen the physics MVDR spectrum.

**Key Design Choice:** MVDR is treated as a *frozen physics transform* during Stage-2.
We do **not** backpropagate through the MVDR inverse; we train the `SpectrumRefiner` on MVDR spectrum images computed from the backbone’s low-rank factors.

#### New MVDR Low-Rank (Woodbury) Spectrum for Training

**File:** `ris_pytorch_pipeline/music_gpu.py`

Added fast low-rank MVDR spectrum utilities:
- `_compute_spectrum_mvdr_lowrank(...)`: Woodbury identity for \(R = F F^H + \delta I\)
- `mvdr_spectrum_max_2_5d_lowrank(...)`: max-over-range-planes spectrum for grid \((\phi,\theta)\)

This makes spectrum generation feasible at training time with coarse grids (e.g. 61×41).

#### Trainer Support: Refiner-Only Phase

**File:** `ris_pytorch_pipeline/train.py`

Added a new training phase:
- `cfg.TRAIN_PHASE="refiner"`:
  - freezes the backbone (`HybridModel`)
  - constructs `self.refiner = SpectrumRefiner()`
  - computes MVDR spectrum from predicted factors \(A_{ang}, A_{rng}\) using low-rank MVDR
  - runs heatmap supervision loss only (fast path)
  - saves checkpoints as `{"backbone": ..., "refiner": ...}`

Also:
- disabled EMA/SWA during refiner-only stage
- added refiner-aware gradient clipping/sanitization and optimizer wiring

#### Loss Fast-Path for Heatmap-Only Training

**File:** `ris_pytorch_pipeline/loss.py`

Added a fast path so refiner-only training does **not** require `R_true` / covariance losses:
- if only `lam_heatmap>0` and `refined_spectrum` present, returns heatmap loss directly

#### Config Knobs Added

**File:** `ris_pytorch_pipeline/configs.py`

Added Stage-2 defaults:
- `REFINER_GRID_PHI`, `REFINER_GRID_THETA`, `REFINER_R_PLANES`
- `PHASE_LOSS["refiner"]` (zeros other weights)

#### New CLI Command

**File:** `ris_pytorch_pipeline/ris_pipeline.py`

Added:
- `train-refiner` command, which sets `TRAIN_PHASE="refiner"` and loads the backbone checkpoint via `cfg.INIT_CKPT`.

Example:
```bash
python -m ris_pytorch_pipeline.ris_pipeline train-refiner \
  --backbone_ckpt results_final_L16_12x12/checkpoints/best.pt \
  --epochs 10 --use_shards --n_train 100000 --n_val 10000 \
  --lam_heatmap 0.1 --grid_phi 61 --grid_theta 41
```

### Refiner-Assisted Inference (NEW)

**Goal:** Use the trained `SpectrumRefiner` at inference time to sharpen the MVDR spectrum before peak picking.

**File:** `ris_pytorch_pipeline/infer.py`

Changes:
- `load_model(...)` now supports loading Stage-2 checkpoints saved as:
  - raw backbone state_dict
  - `{"model": ...}`
  - `{"backbone": ..., "refiner": ...}` (attaches `model._spectrum_refiner`)
- `hybrid_estimate_final(...)` now supports an optional refiner-assisted path when:
  - `cfg.USE_SPECTRUM_REFINER_IN_INFER = True`
  - the loaded model has `model._spectrum_refiner`

Pipeline:
1. Build low-rank factors from backbone outputs: \(F=[A_{ang}, \sqrt{\lambda}A_{rng}]\)
2. Compute MVDR max-over-range spectrum using Woodbury MVDR
3. Apply `SpectrumRefiner` → probability heatmap
4. 2D NMS peak detection on refined heatmap
5. For each selected peak, choose the best range plane by evaluating MVDR across `r_planes`

Config knobs added:
- `USE_SPECTRUM_REFINER_IN_INFER`
- `REFINER_PEAK_THRESH`, `REFINER_NMS_MIN_SEP`

---

## 1. New Files Created

### 1.1 MVDR Implementation in `music_gpu.py`

**Added Methods to `GPUMusicEstimator` class:**

| Method | Description |
|--------|-------------|
| `_compute_R_inv()` | Regularized inverse of covariance matrix |
| `_compute_spectrum_mvdr()` | Core MVDR spectrum: `1 / (a^H R^{-1} a)` |
| `_compute_spectrum_mvdr_2_5d()` | 2.5D MVDR over range planes |
| `_compute_spectrum_mvdr_3d_local()` | Local 3D grid for refinement |
| `_find_local_maxima_3d()` | Peak detection in 3D volume |
| `_nms_3d()` | Non-Maximum Suppression |
| `estimate_mvdr_2_5d()` | Main 2.5D candidate generation |
| `refine_candidate_3d()` | Local 3D Newton-like refinement |
| `detect_sources_mvdr()` | **Full pipeline**: 2.5D → refine → NMS |

**Added Convenience Functions:**
```python
def mvdr_detect_sources(R, cfg, device='cuda', **kwargs):
    """K-free multi-source detection using MVDR."""
    
def mvdr_spectrum_2_5d(R, cfg, device='cuda', **kwargs):
    """Compute 2.5D MVDR spectrum for visualization."""
```

### 1.2 Evaluation Wrappers in `eval_angles.py`

**New Functions Added:**
```python
def mvdr_localize(R, cfg, *, grid_phi=361, grid_theta=181, 
                  threshold_db=12.0, max_sources=5, ...):
    """K-free multi-source localization using MVDR spectrum."""
    
def eval_scene_mvdr(R, phi_gt, theta_gt, r_gt, cfg, *, 
                    threshold_db=12.0, max_sources=5, ...):
    """Evaluate MVDR localization on a single scene with Hungarian matching."""
```

### 1.3 Visualization Script `eval_mvdr.py`

New standalone script for MVDR testing and visualization:
- Loads trained model checkpoint
- Generates test data with known ground truth
- Computes `R_blend` from network predictions
- Runs `mvdr_detect_sources()`
- Plots spectrum slices with GT overlay
- Saves PNG visualizations

### 1.4 SpectrumRefiner CNN in `model.py`

**New Class: `SpectrumRefiner`**

A small U-Net-like CNN that takes physics-based MVDR spectrum and learns to sharpen peaks / suppress noise.

```python
class SpectrumRefiner(nn.Module):
    """
    Architecture:
        - Input: [B, 1, H, W] MVDR spectrum (2D slice)
        - Encoder: 3 conv blocks with downsampling (1→32→64→128 channels)
        - Bottleneck: Dilated convolutions (receptive field expansion)
        - Decoder: 3 conv blocks with upsampling + skip connections
        - Output: [B, 1, H, W] refined probability map (sigmoid)
    """
```

**Key Features:**
- Skip connections for fine detail preservation
- Dilated convolutions in bottleneck for large receptive field
- BatchNorm for stable training
- Sigmoid output for probability interpretation

**Supporting Classes:**
```python
class SpectrumRefinerLoss(nn.Module):
    """Focal loss with Gaussian blob GT targets."""

def create_spectrum_refiner_with_pretrained_backbone(...):
    """Factory for combined backbone + refiner model."""
```

### 1.5 Heatmap Supervision Loss in `loss.py`

**New Method: `_heatmap_loss()`**

Computes focal loss between refined spectrum and GT Gaussian blobs.

```python
def _heatmap_loss(self, refined_spectrum, phi_gt, theta_gt, K_true, 
                  grid_phi, grid_theta):
    """
    1. Create Gaussian blob heatmap at each GT source location
    2. Compute focal loss (α=0.25, γ=2.0) for sparse target handling
    3. Max-blend overlapping sources
    """
```

**New Loss Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lam_heatmap` | 0.0 | Weight for heatmap supervision |
| `heatmap_sigma_phi` | 2.0 | Gaussian blob σ (grid cells) |
| `heatmap_sigma_theta` | 2.0 | Gaussian blob σ (grid cells) |

---

## 2. Files Modified

### 2.1 `model.py` — K-Head Removal

**Removed from `__init__`:**
```python
# DELETED:
self.k_spec_proj = nn.Linear(N + K_MAX, 64)
self.k_fuse = nn.Linear(64 + backbone_out, 128)
self.k_mlp = nn.Sequential(...)           # 5-way softmax
self.k_ord_mlp = nn.Sequential(...)       # Ordinal P(K>t)
self.k_direct_mlp = nn.Sequential(...)    # Direct path
self.k_direct_ord_mlp = nn.Sequential(...)
self.k_logit_scale = nn.Parameter(...)
```

**Removed from `forward()`:**
```python
# DELETED: ~80 lines of K-head forward pass
# - Spectral feature extraction (eigenvalues of Ryy)
# - MDL/AIC one-hot encoding
# - k_feats concatenation
# - Dual-path logit computation
# - k_logits, k_ord_logits output
```

**Removed from `set_trainable_for_phase()`:**
```python
# DELETED: freeze_k parameter and K-head freezing logic
```

### 2.2 `loss.py` — K Loss Removal

**Removed from `__init__`:**
```python
# DELETED:
self.lam_K = lam_K  # K classification weight
```

**Removed Methods:**
```python
# DELETED:
def _eigengap_hinge(...)      # Still exists but for cov quality, not K
def _subspace_margin_regularizer(...)  # Still exists for cov quality
def _blind_k_regularizer(...)  # Fully removed
```

**Removed from `forward()`:**
```python
# DELETED: ~60 lines of K loss computation
# - K ordinal BCE loss
# - K softmax CE loss  
# - Confidence weighting
# - Under/over prediction penalties
# - loss_K term in total
```

**Updated Total Loss:**
```python
# BEFORE:
total = (self.lam_cov * loss_nmse + ... + self.lam_K * loss_K + ...)

# AFTER:
total = (self.lam_cov * loss_nmse + ... )  # No lam_K term
```

### 2.3 `train.py` — K Training Code Removal

**Removed from `Trainer.__init__`:**
```python
# DELETED from HEAD_KEYS:
'k_head', 'k_mlp', 'k_direct_mlp', 'k_ord', 'k_direct_ord', 
'k_spec_proj', 'k_fuse', 'k_logit'
```

**Removed Methods:**
```python
# DELETED:
def _ramp_k_weight(self, epoch):
    """Curriculum learning for K weight."""
    
def calibrate_k_logits(self, val_loader, save_path=None):
    """Temperature scaling for K logits."""
```

**Removed from `_apply_hpo_loss_weights()`:**
```python
# DELETED:
self.loss_fn.lam_K = self._hpo_loss_weights.get("lam_K", 0.1)
```

**Updated Validation:**
```python
# BEFORE: K metrics from k_logits/k_ord_logits
k_mode = str(getattr(cfg, "K_HEAD_MODE", "softmax")).lower()
if k_mode == "ordinal" and ("k_ord_logits" in preds):
    probs = torch.sigmoid(preds["k_ord_logits"].float())
    ...

# AFTER: Use GT K for validation, MDL at inference
k_pred = None  # K estimated via MDL during inference
```

**Removed Post-Training Calibration:**
```python
# DELETED:
if bool(getattr(mdl_cfg, "CALIBRATE_K_AFTER_TRAIN", True)):
    optimal_temp, calib_acc, high_snr_acc = self.calibrate_k_logits(...)
```

### 2.4 `configs.py` — K Config Removal

**Removed Parameters:**
```python
# DELETED:
self.K_HEAD_MODE = "ordinal"
self.K_ORD_THRESH = 0.5
self.K_CONF_THRESH = 0.65

# DELETED from PHASE_LOSS_WEIGHTS:
"k_only": {..., "lam_K": 1.5, ...}
"geom": {..., "lam_K": 0.1, ...}
"joint": {..., "lam_K": 1.0, ...}
```

### 2.5 `infer.py` — K Inference Update

**Removed K-Logits Logic:**
```python
# DELETED: ~30 lines
k_mode = str(getattr(cfg, "K_HEAD_MODE", "softmax")).lower()
if k_mode == "ordinal" and ("k_ord_logits" in pred):
    ord_logits = pred["k_ord_logits"][0].detach().cpu()
    probs = torch.sigmoid(ord_logits)
    ...
elif "k_logits" in pred:
    k_logits_scaled = pred["k_logits"][0].detach().cpu() / ...
    ...
```

**Updated K Selection:**
```python
# BEFORE:
if prefer_logits and (k_from_logits is not None):
    thr = float(getattr(cfg, "K_CONF_THRESH", 0.65))
    K_hat = K_mdl if (nn_conf < thr) else K_nn

# AFTER:
# K estimation: use MDL/AIC (K-head removed)
K_hat = K_mdl
```

### 2.6 `hpo.py` — K HPO Removal

**Removed from Search Space:**
```python
# DELETED:
lam_K = trial.suggest_float("lam_K", 0.05, 0.30)
```

**Updated Return Dict:**
```python
# DELETED from return:
"lam_K": lam_K
```

**Updated Trainer Config:**
```python
# DELETED:
t.loss_fn.lam_K = s["lam_K"]
t._hpo_loss_weights["lam_K"] = s["lam_K"]
```

---

## 3. Files Deleted

### 3.1 Diagnostic Scripts
| File | Reason |
|------|--------|
| `run_k_diagnostic.py` | K-head diagnostic no longer needed |

### 3.2 Documentation
| File | Reason |
|------|--------|
| `K_HEAD_DIAGNOSTIC_REPORT.md` | Obsolete K-head analysis |

### 3.3 Log Files (10 files)
| File | Reason |
|------|--------|
| `kdiag_separability.log` | K diagnostic logs |
| `run_k_diagnostic.log` | K diagnostic logs |
| `run_k_diagnostic_v2.log` | K diagnostic logs |
| `run_k_diagnostic_v3.log` | K diagnostic logs |
| `run_k_diagnostic_v4.log` | K diagnostic logs |
| `run_k_diagnostic_v5.log` | K diagnostic logs |
| `run_k_diagnostic_debug.log` | K diagnostic logs |
| `run_k_diagnostic_easy.log` | K diagnostic logs |
| `run_k_diagnostic_trace.log` | K diagnostic logs |
| `run_k_diagnostic_after_kfix.log` | K diagnostic logs |

---

## 4. Architecture Changes

### Before (K-Head Approach)
```
Input → Backbone → K-Head → K̂ → MUSIC(K̂) → Sources
                     ↓
              R_pred → R_blend
```

**Problems:**
- K estimation from Ryy eigenvalues was ill-posed (coherent sources)
- Mode collapse to K=3 or K=4
- Training-inference mismatch

### After (MVDR Approach)
```
Input → Backbone → R_pred → R_blend → MVDR Spectrum → Peak Detection → Sources
                                           ↓
                                    NMS + 3D Refinement
```

**Advantages:**
- K-free: number of sources determined by peak count
- Robust to coherent sources (MVDR handles rank deficiency)
- Training-inference aligned (same R_blend used everywhere)

---

## 5. Loss Function Summary

### Current Loss Terms (Post-Cleanup)

| Term | Weight | Purpose |
|------|--------|---------|
| `loss_nmse` | `lam_cov` | Primary: Covariance NMSE |
| `loss_nmse_pred` | `lam_cov_pred` | Aux: R_pred regularization |
| `loss_ortho` | `lam_ortho` | Stiefel manifold penalty |
| `loss_gap` | `lam_gap` | Eigengap for covariance quality |
| `loss_margin` | `lam_margin` | Subspace margin regularizer |
| `aux_l2` | `lam_aux` | Angle/range Huber loss |
| `peak_l2` | `lam_peak` | Chamfer distance on angles |
| `loss_subspace_align` | `lam_subspace_align` | GT subspace projection |
| `loss_peak_contrast` | `lam_peak_contrast` | MUSIC peak sharpness |
| `loss_heatmap` | `lam_heatmap` | **NEW:** SpectrumRefiner supervision |

### New Loss Term: Heatmap Supervision
```python
# Gaussian blob GT target + Focal Loss
# Only active when refined_spectrum is in predictions
if self.lam_heatmap > 0.0 and 'refined_spectrum' in y_pred:
    loss_heatmap = self._heatmap_loss(refined_spectrum, phi_gt, theta_gt, K_true, ...)
```

### Removed Terms
| Term | Former Weight | Reason Removed |
|------|---------------|----------------|
| `loss_K` | `lam_K` | K-head removed |
| `loss_blind_k` | (internal) | K-head removed |

---

## 6. Verification Results

### MVDR Test Output (from terminal)
```
============================================================
SUMMARY STATISTICS
============================================================
Mean φ RMSE: 12.83°
Mean θ RMSE: 22.72°
Mean r RMSE: 4.963m
Detection: 24/12 sources (200.0%)
============================================================
```

**Notes:**
- Detection > 100% indicates false positives (expected with untrained model)
- Errors are higher than target but MVDR pipeline is functional
- Full training with new loss landscape expected to improve

### Code Verification
- ✅ All imports successful
- ✅ Model has no K-head attributes
- ✅ Loss has no `lam_K`
- ✅ MVDR functions callable
- ✅ No linter errors

---

## 7. Next Steps (Per Plan)

| Step | Status | Description |
|------|--------|-------------|
| 1 | ✅ Done | MVDR spectrum computation |
| 2 | ✅ Done | Visualization & verification |
| 3 | ✅ Done | Remove K-head (clean slate) |
| 4 | 🔲 Pending | Train with new loss landscape |
| 5 | ✅ Done | SpectrumRefiner CNN |
| 6 | ✅ Done | Heatmap supervision |
| 7 | 🔲 Pending | Benchmarking vs old approach |

---

## 8. File Summary

### Modified Files (7)
1. `ris_pytorch_pipeline/model.py` — K-head removal + SpectrumRefiner added
2. `ris_pytorch_pipeline/loss.py` — K loss removal + heatmap loss added
3. `ris_pytorch_pipeline/train.py` — K training code removal
4. `ris_pytorch_pipeline/configs.py` — K config removal
5. `ris_pytorch_pipeline/infer.py` — MDL fallback for K
6. `ris_pytorch_pipeline/hpo.py` — lam_K removal
7. `ris_pytorch_pipeline/eval_angles.py` — MVDR eval functions added

### New Files (2)
1. `ris_pytorch_pipeline/music_gpu.py` (MVDR methods added)
2. `eval_mvdr.py`

### New Classes Added
1. `SpectrumRefiner` — U-Net CNN for spectrum denoising
2. `SpectrumRefinerLoss` — Focal loss with Gaussian blob targets
3. `create_spectrum_refiner_with_pretrained_backbone()` — Factory function

### Deleted Files (12)
1. `run_k_diagnostic.py`
2. `K_HEAD_DIAGNOSTIC_REPORT.md`
3. 10 log files (`*k_diag*.log`, `kdiag*.log`)

---

## Appendix: Key Code Snippets

### MVDR Spectrum Formula
```python
def _compute_spectrum_mvdr(self, R_inv, A):
    """
    MVDR/Capon spectrum: P(θ) = 1 / (a^H R^{-1} a)
    
    Unlike MUSIC, this doesn't require knowing K.
    """
    # A: [N, G] steering vectors
    # R_inv: [N, N] regularized inverse
    numerator = torch.ones(A.shape[1], device=A.device)
    denominator = torch.einsum('ng,nm,mg->g', A.conj(), R_inv, A).real
    return numerator / denominator.clamp(min=1e-12)
```

### Full Detection Pipeline
```python
def detect_sources_mvdr(self, R, num_candidates=10, min_sep_deg=1.0, 
                        threshold_db=10.0, ...):
    """
    K-free multi-source detection:
    1. Compute 2.5D MVDR spectrum over range planes
    2. Find candidates via peak detection
    3. Refine each candidate in local 3D grid
    4. Apply NMS to remove duplicates
    5. Return sources above threshold
    """
```

### SpectrumRefiner Architecture
```python
class SpectrumRefiner(nn.Module):
    """
    U-Net-like CNN for spectrum denoising/sharpening.
    
    Encoder:
        [B, 1, H, W] → conv(32) → conv(64) → conv(128)
                        ↓          ↓          ↓
                      skip1      skip2      skip3
    
    Bottleneck:
        Dilated conv (d=2, d=4) for large receptive field
    
    Decoder:
        upsample + concat(skip) → conv → upsample → ... → [B, 1, H, W]
    
    Output: sigmoid for probability map
    """
```

### Heatmap Loss (Focal + Gaussian Blob)
```python
def _heatmap_loss(self, refined_spectrum, phi_gt, theta_gt, K_true, ...):
    # 1. Create Gaussian blobs at GT locations
    for k in range(K):
        blob = exp(-0.5 * ((φ - φ_gt[k])/σ_φ)² + ((θ - θ_gt[k])/σ_θ)²)
        heatmap = max(heatmap, blob)  # Max-blend for overlap
    
    # 2. Focal loss (handles sparse targets)
    pos_loss = -α(1-p)^γ * gt * log(p)
    neg_loss = -(1-α)p^γ * (1-gt) * log(1-p)
    return mean(pos_loss + neg_loss)
```

---

## 9. Usage Examples

### Basic MVDR Detection (No Refiner)
```python
from ris_pytorch_pipeline.music_gpu import mvdr_detect_sources

# Get R_blend from model
pred = model(y, H, codes, R_samp=R_samp)
R_blend = pred['R_blend'][0].cpu().numpy()

# Detect sources (K-free!)
sources, spectrum = mvdr_detect_sources(R_blend, cfg, threshold_db=12.0)
for phi, theta, r, conf in sources:
    print(f"Source: φ={phi:.1f}°, θ={theta:.1f}°, r={r:.2f}m, conf={conf:.3f}")
```

### With SpectrumRefiner (Learned Denoising)
```python
from ris_pytorch_pipeline.model import SpectrumRefiner, create_spectrum_refiner_with_pretrained_backbone

# Create combined model
combined = create_spectrum_refiner_with_pretrained_backbone(backbone, freeze_backbone=True)

# Forward pass
mvdr_spectrum = compute_mvdr_spectrum(R_blend)  # [B, 1, G_phi, G_theta]
out = combined(y, H, codes, mvdr_spectrum)
refined = out['refined_spectrum']  # [B, 1, G_phi, G_theta]

# Peak detection on refined map
peaks = nms_2d(refined[0, 0], threshold=0.5)
```

### Training with Heatmap Supervision
```python
# Set loss weight > 0 to enable
loss_fn = UltimateHybridLoss(lam_heatmap=0.1, heatmap_sigma_phi=2.0)

# Model must output 'refined_spectrum'
pred = combined_model(y, H, codes, mvdr_spectrum)
loss = loss_fn(pred, y_true)  # Automatically includes heatmap loss
```

---

*End of Changelog*
