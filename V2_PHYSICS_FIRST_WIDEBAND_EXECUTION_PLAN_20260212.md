# V2 Physics-First Wideband Execution Plan

**Date:** 2026-02-12  
**Branch:** `v2/physics-first-wideband`  
**Parent:** `main`  
**Related:** `CONSTANT_PREDICTION_TRAP_DIAGNOSIS_20260211.md`, `OFDM_TR38901_INDOOR_PLAN.md`

---

## Status Snapshot (Branch Reality)

This section tracks what is actually implemented on `v2/physics-first-wideband` so there is no ambiguity between plan vs code state.

- Implemented scaffold files: `ris_pytorch_pipeline/v2/{config_v2.py,model_v2.py,loss_v2.py,dataset_v2.py,train_v2.py,overfit_v2.py,run_v2.py}`.
- Implemented model objective: covariance-only prediction via low-rank PSD factor (`R_pred = A A^H + eps I`), no slot heads.
- Implemented trainer behavior: no curriculum, no permutation loss stack, NMSE-primary loss path.
- Not implemented yet: Phase-2 wideband data generator (`F>1` shards), tap-domain `H_tap` ingestion, and production-wideband eval runs.

Practical meaning: Phase-0/Phase-1 code skeleton is in place, but end-to-end validation gates still need to be run on Goose.

**Correction (authoritative direction):**
- `OFDM_TR38901_INDOOR_PLAN.md` is the primary simulation/system plan to follow.
- V2 execution is **wideband-first** (start at F=16), not narrowband-first.
- Narrowband is only an optional fallback for debugging, not the main path.

---

## 0. Why V2?

The v1 pipeline predicts geometry (φ, θ, r) **directly** from neural network heads.  
After 19 rounds of fixes, the model still fails to overfit 200 samples (φ_rmse stuck ≈ 23°,
θ and r variances near-constant). Root causes:

| Failure Mode | Explanation |
|---|---|
| **Permutation matching** | With K_MAX=5 slots, each sample's matching permutation differs → averaged gradient pulls every slot toward dataset mean |
| **Gradient competition** | NMSE loss (144×144 Jacobian) drowns geometry loss (15-dim) under shared backbone + CLIP_NORM |
| **Symmetric equilibrium** | Slot diversity + sorted-canonical losses are band-aids; the fundamental landscape has a deep basin at constant predictions |
| **Circular feature extraction** | AntiDiagPool extracts features from R_learned that is built from the same backbone features — no new information |

**Key insight:** Covariance prediction is a regression problem with a smooth, convex-like
NMSE loss and **no permutation ambiguity** (R = Σ aₖaₖᴴ is order-invariant by construction).
The model should predict R, and physics-based MUSIC/MVDR should extract geometry.

---

## 1. Design Principles

1. **Physics-first**: NN learns the **covariance matrix** (or its low-rank factors).  
   Geometry extraction uses classical MUSIC/MVDR — no learned angle/range heads.
2. **Single loss objective**: NMSE on covariance dominates training. Optional physics-aligned
   structure losses (subspace, peak-contrast) added only after covariance converges.
3. **No slots, no permutation matching**: The output is a single matrix R̂, not K slots.
4. **Wideband-first**: Architecture trains on OFDM tensors from day 1 (start at F=16,
   then scale to F=64+). Narrowband is optional fallback only.
5. **Incremental validation**: Each phase has a hard gate. No proceeding without pass.
6. **Reuse correct v1 code**: Physics utilities, MUSIC/MVDR backends, data loading,
   covariance utilities, evaluation scripts — all reused.

---

## 2. Architecture

### 2.1 Tensor Shapes (narrowband, F=1)

| Tensor | Shape | Description |
|---|---|---|
| `y` | `[B, L, M, 2]` | Received signal (RI), L=16 snapshots, M=16 BS antennas |
| `H` | `[B, L, M, 2]` | Direct channel estimate (RI) |
| `codes` | `[B, L, N, 2]` | RIS codebook (RI), N=144 elements |
| `R_true` | `[B, N, N]` complex | Ground-truth covariance (from simulator) |
| `snr_db` | `[B]` | Per-sample SNR |

### 2.2 Tensor Shapes (wideband, F>1)

| Tensor | Shape | Description |
|---|---|---|
| `y` | `[B, L, F, M, 2]` | OFDM received signal, F subcarriers |
| `H_tap` | `[B, D_taps, M, N, 2]` | Tap-domain channel (compact), D_taps ≤ 8 |
| `codes` | `[B, L, N, 2]` | RIS codebook (same across subcarriers) |
| `R_true` | `[B, N, N]` complex | Broadband covariance (aggregated) |
| `R_f_true` | `[B, F, N, N]` complex | Per-subcarrier covariance (optional) |
| `snr_db` | `[B]` | Per-sample average SNR |

### 2.3 Model Blueprint — `CovariancePredictor`

```
Input: y [B, L, (F), M, 2], H [B, L, M, 2], codes [B, L, N, 2]
                │                    │                │
                ▼                    ▼                ▼
         ┌──────────┐        ┌───────────┐    ┌───────────┐
         │ y_encoder │        │ H_proj    │    │ code_conv │
         │ Conv1D→   │        │ Linear    │    │ Conv1D→   │
         │ Transformer│        │ → D/2     │    │ pool → D/2│
         │ → pool→ D │        └───────────┘    └───────────┘
         └──────────┘              │                │
                │                  ▼                ▼
                └──────────> [ concat → Linear → D ] ← fusion
                                   │
                     ┌─────────────┤
                     ▼             ▼
              ┌────────────┐  (optional, Phase 2+)
              │ factor_head│  ┌────────────────┐
              │ Linear(D,  │  │ freq_pool_head │
              │  N*R*2)    │  │ Attention(F→1) │
              └────────────┘  └────────────────┘
                     │
                     ▼
              A [B, N, R] complex  (R = rank, typically 2*K_MAX = 10)
                     │
                     ▼
              R̂ = A Aᴴ + ε I   [B, N, N] complex (guaranteed PSD)
                     │
                     ▼
              hermitize + trace-normalize → R_pred [B, N, N]
```

**Key design choices:**
- **Single linear head** outputs the low-rank factor A. No separate angle/range factors.
- **Rank R = 2×K_MAX = 10**: Overparameterized relative to K_MAX=5 to give the NN
  freedom to represent noise subspace nuances. The `A Aᴴ` construction ensures PSD.
- **No AntiDiagPool feedback loop**: Features come only from raw inputs.
- **No softmax grid, no aux_angles, no aux_range**: Geometry comes from MUSIC post-hoc.

### 2.4 y-Encoder Details

```python
class YEncoder(nn.Module):
    """Encode received signal snapshots into a fixed-dim feature vector."""
    def __init__(self, M, L, D, n_heads=8, n_layers=4, dropout=0.1):
        # Conv1D stem: [B, 2M, L] → [B, D, L]
        self.conv1 = nn.Conv1d(M * 2, D // 2, kernel_size=5, padding=2)
        self.dw    = nn.Conv1d(D // 2, D // 2, kernel_size=3, padding=1, groups=D // 2)
        self.conv2 = nn.Conv1d(D // 2, D, kernel_size=1)
        # Transformer: [B, L, D] → [B, L, D]
        layer = nn.TransformerEncoderLayer(D, n_heads, D*4, dropout, 'gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
    
    def forward(self, y):
        # y: [B, L, M, 2] → [B, 2M, L]
        x = y.reshape(B, L, M*2).permute(0, 2, 1)
        x = F.gelu(self.conv1(x))
        x = self.dw(x)
        x = F.gelu(self.conv2(x))       # [B, D, L]
        x = x.permute(0, 2, 1)          # [B, L, D]
        x = self.transformer(x).mean(1)  # [B, D] (temporal pooling)
        return x
```

### 2.5 Wideband Extension — Frequency Pooling (Phase 2+)

For F>1, the y-encoder produces **per-subcarrier features** `[B, F, D]`, then a learned
frequency attention module pools them:

```python
class FreqPool(nn.Module):
    """Attention-weighted pooling over F subcarriers."""
    def __init__(self, D):
        self.query = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.key_proj = nn.Linear(D, D)
    
    def forward(self, x_f):
        # x_f: [B, F, D]
        keys = self.key_proj(x_f)                   # [B, F, D]
        attn = (self.query * keys).sum(-1) / D**0.5  # [B, F]
        attn = F.softmax(attn, dim=1).unsqueeze(-1)  # [B, F, 1]
        return (attn * x_f).sum(1)                    # [B, D]
```

---

## 3. Loss Design

### 3.1 Primary: Covariance NMSE

```
L_cov = || R̂ - R_true ||²_F  /  || R_true ||²_F
```

This is the **sole loss** for Phase 1. It's smooth, convex in R̂, and has no permutation
ambiguity. The model's only job is to predict a good covariance matrix.

### 3.2 Optional Structure Losses (Phase 1b+)

Added **only after** NMSE overfit is confirmed:

| Loss | Formula | Weight | Purpose |
|---|---|---|---|
| `L_subspace` | `|| (I - P_sig) A_gt ||² / || A_gt ||²` | 0.05 | Align predicted R's signal subspace with GT steering vectors |
| `L_peak_contrast` | `1 - MVDR(R̂, φ_gt, θ_gt)` | 0.02 | Encourage MVDR peaks at GT angles |

**Explicitly excluded** from v2:
- Slot diversity loss
- Sorted canonical auxiliary loss
- Permutation-invariant auxiliary loss
- Direct geometry loss (wrapped Huber on angles, log-range Huber)
- Eigengap hinge
- Chamfer angle loss
- Mask/BCE losses

### 3.3 Loss Code (Simplified)

```python
class V2CovarianceLoss(nn.Module):
    """Pure covariance NMSE + optional structure terms."""
    
    def __init__(self, lam_subspace=0.0, lam_peak=0.0):
        super().__init__()
        self.lam_subspace = lam_subspace
        self.lam_peak = lam_peak
    
    def forward(self, R_pred, R_true, K_true=None, ptr_gt=None, snr_db=None):
        # Primary NMSE
        diff = R_pred - R_true
        nmse = (diff.conj() * diff).real.sum((-2, -1)) / \
               (R_true.conj() * R_true).real.sum((-2, -1)).clamp_min(1e-12)
        loss = nmse.mean()
        
        info = {"nmse": loss.item()}
        
        # Optional: subspace alignment (uses GT steering vectors, no EVD backprop)
        if self.lam_subspace > 0 and ptr_gt is not None:
            L_sub = self._subspace_alignment(R_pred, K_true, ptr_gt)
            loss = loss + self.lam_subspace * L_sub
            info["subspace"] = L_sub.item()
        
        # Optional: peak contrast (encourages MVDR peaks at GT angles)
        if self.lam_peak > 0 and ptr_gt is not None:
            L_peak = self._peak_contrast(R_pred, K_true, ptr_gt)
            loss = loss + self.lam_peak * L_peak
            info["peak"] = L_peak.item()
        
        return loss, info
```

---

## 4. Training Protocol

### 4.1 Trainer (Simplified)

The v2 trainer is stripped down:
- **No curriculum phases** (geom_only, joint, refiner)
- **No NMSE ramp** (NMSE is always on)
- **No mask warmup** (no masks)
- **No slot diversity scheduling**
- **No sorted canonical loss**

```python
class V2Trainer:
    def __init__(self, model, train_ds, val_ds):
        self.model = model
        self.loss_fn = V2CovarianceLoss()
        self.optimizer = torch.optim.AdamW(
            self._param_groups(), weight_decay=1e-4
        )
        self.scheduler = CosineAnnealingWarmRestarts(...)
    
    def _param_groups(self):
        backbone = [p for n, p in self.model.named_parameters() if 'factor_head' not in n]
        head = [p for n, p in self.model.named_parameters() if 'factor_head' in n]
        return [
            {"params": backbone, "lr": 3e-4},
            {"params": head, "lr": 3e-4 * 4.0},  # 4× head LR
        ]
    
    def train_one_epoch(self, loader):
        for y, H, codes, R_true, K_true, ptr_gt, snr_db in loader:
            out = self.model(y, H, codes)        # → {"R_pred": [B, N, N]}
            loss, info = self.loss_fn(out["R_pred"], R_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
```

### 4.2 Overfit Test Protocol

**THE decisive diagnostic.** Must pass before any large-scale training.

```
Dataset:   200 samples (fixed seed=1337 subset)
Batch:     200 (full batch, no shuffling noise)
LR:        1e-3 backbone, 4e-3 head
Epochs:    200
Loss:      Pure NMSE only (lam_subspace=0, lam_peak=0)
Dropout:   0.0
AMP:       Off
Weight decay: 0.0

PASS criteria:
  - NMSE < 0.01 by epoch 100 (memorize covariances)
  - NMSE < 0.001 by epoch 200 (near-perfect reconstruction)

FAIL action:
  - If NMSE stuck > 0.1: backbone capacity issue → increase D_MODEL or layers
  - If NMSE stuck > 0.5: data pipeline issue → check R_true integrity
  - If NMSE goes NaN: numerical issue → check factor construction
```

### 4.3 Full Training Protocol

```
Phase 1a: Pure NMSE
  - Loss: NMSE only
  - Epochs: 30-50
  - Gate: val_NMSE < 0.1

Phase 1b: NMSE + Structure
  - Loss: NMSE + 0.05*L_subspace + 0.02*L_peak
  - Epochs: 20-30  
  - Gate: val_NMSE < 0.08, MUSIC φ_rmse < 10°

Phase 1c: SpectrumRefiner (optional, parallel)
  - Freeze backbone
  - Train U-Net refiner on MVDR spectra from R_pred
  - Gate: refiner peak-recall > 0.8
```

---

## 5. Inference Pipeline

Inference is **identical** to v1 except the covariance source:

```
R_pred (from NN) → shrink(R_pred, snr_db)
                  → MUSIC 2D scan → peaks
                  → Parabolic refinement
                  → Newton refinement (optional near-field)
                  → Range MUSIC (1D per angle)
                  → Joint Newton polish
                  → (φ, θ, r) per source
```

The existing `infer.py`, `music_gpu.py`, `eval_angles.py` are reused unchanged.
Only the model changes — it produces R_pred instead of (R_pred + aux geometry).

---

## 6. Wideband OFDM Extension (Phase 2–3)

### 6.1 Motivation

Narrowband setup (F=1) has limited range identifiability — range comes only from
near-field phase curvature across the 12×12 RIS array. Wideband OFDM (F>1) adds
**time-of-arrival** information that dramatically improves range estimation.

### 6.2 OFDM Numerology (3GPP TR 38.901 Indoor)

| Parameter | Value |
|---|---|
| Carrier frequency | 3.5 GHz |
| Subcarrier spacing (Δf) | 30 kHz |
| FFT size | 256 |
| Active subcarriers | 200 |
| Bandwidth | 6 MHz (pilot subset) |
| Pilot subcarriers (F) | 16 / 32 / 64 (phased) |
| CP length | 4.69 μs (normal) |
| Channel taps (D_taps) | 4–8 (indoor) |
| Max delay spread | ~100 ns (indoor office) |
| Range resolution | c / (2 × BW) ≈ 25 m (coarse), improved by super-resolution |

### 6.3 Wideband Data Generation

```python
def generate_ofdm_sample(phi, theta, r, snr_db, F=64):
    """Generate wideband OFDM measurements for multi-source scene."""
    # Per-subcarrier steering (frequency-dependent near-field)
    for f_idx, f in enumerate(subcarrier_freqs):
        k_f = 2 * pi * f / c  # Frequency-dependent wavenumber
        a_f = nearfield_vec_wideband(phi, theta, r, k_f)  # [N] per source
        # Build per-subcarrier covariance
        R_f[f_idx] = sum(p_k * a_f_k @ a_f_k^H) + sigma^2 * I
    # Broadband covariance: average across subcarriers
    R_broadband = mean(R_f, dim=0)
    return y_ofdm, R_f, R_broadband
```

### 6.4 Wideband Model Changes

```
Phase 2 (F=16):
  - y_encoder processes each subcarrier independently, then FreqPool
  - Factor head outputs broadband A [B, N, R] (same as Phase 1)
  - Loss: NMSE on broadband R_pred vs broadband R_true

Phase 3 (F=64):
  - Optional per-subcarrier factor heads → per-subcarrier R̂(f)
  - Joint wideband MUSIC using all R̂(f) for coherent range estimation
  - Tap-domain channel input for compact representation
```

---

## 7. Module Ownership

### 7.1 New Files (in `ris_pytorch_pipeline/v2/`)

| File | Description | Status |
|---|---|---|
| `__init__.py` | Package init | Scaffold |
| `config_v2.py` | V2-specific config (extends v1 SysConfig/ModelConfig) | Phase 0 |
| `model_v2.py` | `CovariancePredictor` — backbone + factor head | Phase 1 |
| `loss_v2.py` | `V2CovarianceLoss` — NMSE + optional structure | Phase 1 |
| `train_v2.py` | `V2Trainer` — simplified training loop | Phase 1 |
| `overfit_v2.py` | Standalone overfit test script | Phase 1 |
| `dataset_v2.py` | Thin wrapper (Phase 1: reuse v1 ShardNPZDataset; Phase 2+: OFDM loader) | Phase 1/2 |
| `run_v2.py` | Entry point (train / eval / overfit) | Phase 1 |

### 7.2 Reused from v1 (unchanged)

| File | What's Reused |
|---|---|
| `physics.py` | `nearfield_vec`, `shrink`, `alpha_from_snr_db`, `quantise_phase` |
| `covariance_utils.py` | `hermitize_torch`, `trace_norm_torch`, `shrink_torch`, `build_effective_cov_torch` |
| `music_gpu.py` | GPU-accelerated MUSIC/MVDR 2D scan |
| `eval_angles.py` | `eval_scene_angles_ranges`, `eval_batch_angles_ranges` |
| `infer.py` | `hybrid_estimate_final` (MVDR→MUSIC→Newton→Range pipeline) |
| `dataset.py` | `ShardNPZDataset` (data loading from .npz shards) |
| `configs.py` | `SysConfig`, `ModelConfig` (v2 config extends these) |
| `model.py` | `SpectrumRefiner` class (U-Net for spectrum sharpening, Phase 1c) |

### 7.3 Deprecated from v1 (not used in v2)

| Component | Reason |
|---|---|
| `HybridModel` (full class) | Replaced by `CovariancePredictor` |
| `SoftArgmax2D` | No grid-based angle prediction in v2 |
| `AntiDiagPool` | Circular feature loop; not needed |
| `UltimateHybridLoss` | 15+ loss terms → replaced by 1–3 terms |
| Slot mechanism (queries, cross-attention, bias) | Eliminated entirely |
| Permutation matching | Not needed (covariance is order-invariant) |
| `overfit_test.py` | Replaced by `overfit_v2.py` |

---

## 8. Execution Phases

### Phase 0: Scaffolding (Day 1, ~2h)

**Tasks:**
- [ ] Create `ris_pytorch_pipeline/v2/` directory with `__init__.py`
- [ ] Create `config_v2.py` extending v1 configs with v2-specific flags
- [ ] Create `model_v2.py` with `CovariancePredictor` skeleton
- [ ] Create `loss_v2.py` with `V2CovarianceLoss` skeleton
- [ ] Create `train_v2.py` with `V2Trainer` skeleton
- [ ] Create `overfit_v2.py` standalone test
- [ ] Create `run_v2.py` entry point
- [ ] Verify: `python -m ris_pytorch_pipeline.v2.run_v2 --check` prints config and exits

**Gate:** All files importable, no runtime errors on `--check`.

### Phase 1: Wideband Minimal (F=16) Covariance Learning (Day 1–2)

**Goal:** 200-sample overfit on covariance NMSE with OFDM inputs (F=16) → NMSE < 0.01.

**Tasks:**
- [ ] Implement `CovariancePredictor.forward()` (y-encoder → fusion → factor → R̂)
- [ ] Implement `V2CovarianceLoss.forward()` (NMSE only)
- [ ] Implement `V2Trainer` training loop (simple, no curriculum)
- [ ] Run overfit test with wideband data (F=16)
- [ ] Verify NMSE converges to < 0.01 on 200 samples
- [ ] Run MUSIC on overfit R̂ predictions → verify φ_rmse < 5°

**Gate:** Overfit NMSE < 0.01 by epoch 100. MUSIC φ_rmse < 5° on overfit set.

**If gate fails:**
- Increase D_MODEL (512 → 768)
- Increase rank R (10 → 20)
- Check R_true data integrity (print norms, Hermiticity)
- Try removing H/codes features (pure y → R̂) to isolate

### Phase 1b: Structure Losses + Wideband Validation (Day 2–3)

**Goal:** Val NMSE < 0.1 on full dataset. MUSIC φ_rmse < 10° on validation.

**Tasks:**
- [ ] Add subspace alignment loss (reuse `_subspace_alignment_loss` from v1 loss.py)
- [ ] Add peak contrast loss (reuse `_peak_contrast_loss` from v1 loss.py)
- [ ] Train on full wideband F=16 dataset with NMSE + structure losses
- [ ] Evaluate with MUSIC pipeline (reuse v1 eval)

**Gate:** Val NMSE < 0.1. MUSIC φ_rmse < 10° on val set.

### Phase 2: Wideband Scale-Up F=64 (Day 3–4)

**Goal:** Full-band pilot subset training with 64 subcarriers.

**Tasks:**
- [ ] Implement OFDM data generator (extend existing pregen scripts)
- [ ] Add FreqPool module to model
- [ ] Update y-encoder for `[B, L, F, M, 2]` input
- [ ] Generate F=64 training data (20K+ samples)
- [ ] Overfit test with F=64 data
- [ ] Compare range RMSE: F=16 vs. F=64

**Gate:** Overfit NMSE < 0.01. Range RMSE improvement over F=16 baseline.

### Phase 3: Full Wideband F=64 (Day 4–6)

**Goal:** Full bandwidth with 64 subcarriers.

**Tasks:**
- [ ] Scale data generation to F=64
- [ ] Optional: per-subcarrier covariance prediction R̂(f)
- [ ] Wideband MUSIC (coherent processing across subcarriers)
- [ ] Full evaluation on test set
- [ ] Compare with v1 performance metrics

**Gate:** End-to-end φ < 3°, θ < 3°, r < 0.5m on test set.

### Phase 4: SpectrumRefiner + Production (Day 6–7)

**Goal:** Polish with spectrum refiner and production-ready inference.

**Tasks:**
- [ ] Attach SpectrumRefiner (reuse v1 U-Net architecture)
- [ ] Train refiner on MVDR spectra from converged backbone
- [ ] Full MVDR→Refiner→NMS→Newton inference pipeline
- [ ] Final evaluation and comparison with v1

---

## 9. Validation Protocol

### 9.1 Covariance Sanity

Run after every model change:
```python
def cov_sanity(R_pred):
    assert R_pred.shape[-2:] == (N, N)
    assert torch.isfinite(R_pred).all(), "Non-finite entries"
    # Hermitian check
    asym = (R_pred - R_pred.conj().transpose(-2,-1)).abs().max()
    assert asym < 1e-3, f"Non-Hermitian: max asymmetry = {asym}"
    # PSD check (all eigenvalues ≥ 0)
    eigs = torch.linalg.eigvalsh(R_pred.float())
    assert (eigs > -1e-6).all(), f"Not PSD: min eigenvalue = {eigs.min()}"
    # Trace check
    tr = torch.diagonal(R_pred, dim1=-2, dim2=-1).real.sum(-1)
    assert (tr > 0).all(), "Zero or negative trace"
```

### 9.2 Physics Sanity

After overfit succeeds:
```python
def physics_sanity(model, test_sample):
    """Verify that MUSIC on R_pred produces correct angles."""
    R_pred = model(y, H, codes)["R_pred"]
    R_eff = build_effective_cov_torch(R_pred, snr_db)
    # Run MUSIC
    phi_hat, theta_hat = music_2d_scan(R_eff, K_true)
    # Check against GT
    phi_err = abs(phi_hat - phi_gt)
    assert phi_err < 5.0, f"MUSIC φ error = {phi_err}° (should be < 5°)"
```

### 9.3 Overfit Sanity

```
200 samples, 200 epochs, full-batch:
  ✓ NMSE < 0.01 by epoch 100
  ✓ NMSE < 0.001 by epoch 200
  ✓ MUSIC φ_rmse < 5° on overfit set
  ✓ All 200 samples have distinct R_pred (no constant prediction)
```

### 9.4 End Metrics (on held-out test set)

| Metric | Target (narrowband) | Target (wideband F=64) |
|---|---|---|
| φ_rmse | < 5° | < 2° |
| θ_rmse | < 5° | < 2° |
| r_rmse | < 1.0 m | < 0.3 m |
| Cov NMSE | < 0.1 | < 0.08 |
| K detection rate | > 80% | > 90% |

---

## 10. Risk Register

| Risk | Mitigation |
|---|---|
| NMSE overfit fails | Increase model capacity; check data pipeline; try direct R prediction (no factor) |
| NMSE converges but MUSIC fails | Add subspace/peak losses; check steering vector consistency between loss and MUSIC |
| Wideband data generation slow | Start with F=16 subset; parallelize with multiprocessing |
| Memory issues with F=64 | Use gradient checkpointing; frequency-batched forward pass |
| Range still poor with wideband | Verify tap-domain model; consider explicit delay-domain processing |

---

## Appendix A: v1 → v2 Migration Checklist

```
[x] Branch created: v2/physics-first-wideband
[x] Diagnosis document carried over  
[ ] v2/ directory scaffold created
[ ] CovariancePredictor model implemented
[ ] V2CovarianceLoss implemented  
[ ] V2Trainer implemented
[ ] Overfit test passes (NMSE < 0.01)
[ ] MUSIC evaluation on R_pred works
[ ] Full narrowband training converges
[ ] Wideband data generator implemented
[ ] Wideband training converges
[ ] End-to-end evaluation complete
```

## Appendix B: Quick Reference — Key Config Values

```python
# System
N_H, N_V = 12, 12    # RIS UPA dimensions
N = 144               # RIS elements  
M = 16                # BS antennas
L = 16                # Temporal snapshots
K_MAX = 5             # Max sources
WAVEL = 0.3           # 1 GHz carrier (v1)
# WAVEL = 0.0857      # 3.5 GHz carrier (v2 wideband)

# Model
D_MODEL = 512         # Feature dimension
FACTOR_RANK = 10      # Low-rank factor rank (2 × K_MAX)
N_HEADS = 8           # Transformer heads
N_LAYERS = 4          # Transformer layers

# Training
LR_BACKBONE = 3e-4
LR_HEAD = 12e-4       # 4× backbone
BATCH_SIZE = 64
EPOCHS = 60
CLIP_NORM = 5.0
WEIGHT_DECAY = 1e-4
```
