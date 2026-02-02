# Wideband OFDM Implementation Checklist
Date: 2026-02-02 (Updated with capacity fix diagnosis)  
Reference: `OFDM_TR38901_INDOOR_PLAN.md`

This is a **step-by-step implementation checklist** for upgrading the pipeline from narrowband to wideband OFDM.

---

## CRITICAL FIX #6 APPLIED (2026-02-02): Slot Head + Presence Mask (Best Structural-R Head)

### Why the previous head was structurally wrong (even with more MLP capacity)
Structural-R makes the aux path the *entire* geometry/covariance path. Predicting \(K\) unordered sources from a **single pooled scene vector** is a poor inductive bias for:
- multi-source set prediction,
- permutation invariance,
- variable \(K\),
- suppressing unused slots (phantom sources).

### Fix: DETR-style learned queries (slot head)
We replaced “pool-to-one-vector then regress \(K\) slots” with:
- **K learned query vectors** \([K_{max}, D]\)
- **cross-attention** into the backbone’s **snapshot token sequence** \([B, L, D]\)
- output **K slot embeddings** \([B, K_{max}, D]\)
- per-slot prediction of: \((\phi, \theta, r, p, m)\) where \(m\in(0,1)\) is a presence probability.

This makes the backbone capacity **actually usable**: each predicted source is produced by attending to the right evidence in the per-snapshot token sequence, rather than forcing a tiny MLP to “decode the whole scene” from a single vector.

### Presence/mask head (phantom source fix)
We gate the structured covariance contribution:
\[
  p^{eff}_k = p_k \cdot m_k \quad,\quad
  R_{pred}=\sum_k p^{eff}_k\,a_k a_k^H + \sigma^2 I
\]
This makes unused slots naturally vanish.

### Permutation-safe supervision for mask
Slots are unordered, so we **do not** use per-slot BCE targets. Instead we supervise the **count**:
\[
  \left(\sum_k m_k - K_{true}\right)^2
\]
This is stable, cheap, and avoids assignment complexity.

### New parameter count (with slot head)
- **Total trainable**: **10.53M**
- **Slot head** (cross-attn + per-slot MLP + queries): **1.61M**
- Transformer backbone: **6.31M**

---

## CRITICAL FIX #5 APPLIED (2026-02-02): Head Capacity Collapse

### ROOT CAUSE: Model Has No Capacity to Learn Geometry!

**Comparison of working vs broken runs:**

| Metric | Working (2026-01-30) | Broken (2026-02-01) |
|--------|---------------------|---------------------|
| Total params | **28.9M** | **8.9M** (3× smaller) |
| Backbone | 16.7M | 8.6M (2× smaller) |
| **Head** | **12.18M** | **0.32M** (40× smaller!) |
| aux_φ_rmse | 19.38° → 16.40° ✅ | ~18° (FLAT) ❌ |
| aux_θ_rmse | 16.90° → 12.60° ✅ | ~17° (FLAT) ❌ |

**What happened:**
When we implemented structural R, we removed the covariance factor heads (2.6M params) because R is now built from aux predictions. We also "slimmed" the model:
- `USE_FACTORED_SOFTARGMAX=True` → saved 9.2M params
- `USE_CONV_HPROJ=True` → saved 8.3M params

But this left the aux heads as **TINY single linear layers**:
```python
# BROKEN: These have NO CAPACITY!
self.aux_angles = nn.Linear(512, 10)   # 5K params
self.aux_range = nn.Linear(512, 5)     # 2.5K params  
self.aux_power = nn.Linear(512, 5)     # 2.5K params
# Total: ~10K params for ALL geometry prediction!
```

A single linear layer cannot learn the complex features → geometry mapping!

### Fix Applied in `model.py`:
```python
# FIXED: MLP with hidden layer gives real capacity
aux_hidden = mdl_cfg.AUX_HEAD_HIDDEN_DIM  # default 256

self.aux_angles = nn.Sequential(
    nn.Linear(D, aux_hidden),     # 512 → 256
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(aux_hidden, 10),    # 256 → 10
)  # ~131K params per head (13× more capacity)

# Same pattern for aux_range and aux_power
```

**New parameter count:**
- aux_angles: ~134K params (up from 5K)
- aux_range: ~133K params (up from 2.5K)
- aux_power: ~133K params (up from 2.5K)
- Total head: ~0.7M (up from 0.32M)

### Config option added:
```python
# In ModelConfig:
self.AUX_HEAD_HIDDEN_DIM = 256  # Set to 0 for single linear (not recommended)
```

### If still not learning:
Consider reverting more capacity:
1. `USE_FACTORED_SOFTARGMAX = False` → adds back 9.2M params
2. `USE_CONV_HPROJ = False` → adds back 8.3M params

---

## CRITICAL FIX #4 (Revised): Physics Mismatch - Handled by aux_power

### Note: cov_nmse CAN work with structural R

**R_true (generated in shards):**
```python
A0 = sqrt(path_loss(r) * p_src) * unit_steer_vec(...)
R_true = A0 @ A0^H
```

**R_pred (structural model):**
```python
A = unit_steer_vec(...) / sqrt(N)
R_pred = A @ diag(aux_power) @ A^H + sigma2 * I
```

If `aux_power` learns `effective_received_power = path_loss(r) * p_src`, then R_pred CAN match R_true!

**Current config (lam_cov=0.3 with warmup):**
- `lam_cov` warms up from 0 → 0.3 over 5 epochs
- This lets geometry stabilize first before cov_nmse kicks in
- The physics mismatch is NOT fundamental - aux_power absorbs it

---

## CRITICAL FIX #3 APPLIED (2026-02-01): Huber Delta Too Small

### Issue: Loss in Linear Regime (Constant Gradient)
The Huber loss delta for angles was `π/720 = 0.25°` which is **40× too small**. With initial errors of ~19°, ALL samples were in the linear regime (constant gradient regardless of error magnitude).

### Symptoms:
- aux_φ_rmse stuck at ~19°
- aux_θ_rmse stuck at ~16°
- aux_r_rmse stuck at ~2.4m
- Loss decreasing very slowly (1.754 → 1.737 over 27 epochs)

### Fix Applied in `loss.py`:
```python
# OLD (broken):
delta_ang = math.pi/720  # 0.25° = 0.0044 rad

# NEW (fixed):
delta_ang = 0.175  # 10° in radians
delta_logr = 0.5   # (was 0.2)
```

### Why This Matters:
- Huber loss: quadratic for |error| < delta, linear for |error| > delta
- With delta = 0.25°, any error > 0.25° gets constant gradient = 1
- With delta = 10°, errors < 10° get quadratic (proportional) gradients
- This provides meaningful learning signal early in training

---

## CRITICAL FIX #2 APPLIED (2026-01-31): Bounded Aux Outputs for Structural R

### Issue: Structural R Model Had Flat Aux RMSE Despite Gradients Flowing

With `USE_STRUCTURED_R=True`, the aux heads (φ, θ, r) feed into `build_structured_R()` to construct R_pred. At random init, unbounded outputs caused:
- `phi_pred`: -94° to +68° (expected: ±30°)
- `theta_pred`: -94° to +50° (expected: ±15°)
- `r_pred`: 0.3-1.6m (expected: 1-5m)

**Problem**: With errors of 60-90°, Huber loss was in its **linear zone** (constant gradient). The model got the same gradient whether error was 10° or 90°—very slow learning!

**Root Cause**: In the old free-form model, aux heads only fed into aux_l2 loss. In structural R mode, aux heads feed into BOTH aux_l2 AND steering vector construction. Very wrong angles → wrong steering vectors → unstable gradients from cov_nmse that fight against aux_l2 gradients.

### Fix Applied in `model.py`:
```python
# Angle outputs: bound with tanh
aux_phi = torch.tanh(aux_phi_raw) * 0.7      # ±40° (0.7 rad)
aux_theta = torch.tanh(aux_theta_raw) * 0.35  # ±20° (0.35 rad)

# Range outputs: offset + scale
R_MIN, R_SCALE = 1.0, 3.0
aux_range = R_MIN + R_SCALE * Softplus(raw)   # 1-5m range
```

### Additional Fixes:
1. **HEAD_KEYS updated** in `train.py`: Added `aux_power`, `phi_logits`, `theta_logits` to head group for 4× higher LR
2. **LR schedule**: Start at 0.8× (was 0.5×), warmup in 1-2 epochs (was 3), floor at 0.2× (was 0.1×)

### Result After Fix:
- `phi_pred`: ±26° at random init ✓
- `theta_pred`: ±15° at random init ✓
- `r_pred`: 2.5-4.5m at random init ✓

Errors now in Huber **quadratic zone** → gradients proportional to error → faster learning.

---

## CRITICAL FIX #1 APPLIED (2026-01-31): Range Loss Clamp

Before running any training, verify the range loss fix is in place:

### Issue: Training Stalled with Flat Aux RMSE
The `_range_huber_loss` was clamping predicted ranges to `RANGE_R[0] * 0.9 = 0.45m`, which zeroed gradients for ~28% of predictions early in training.

### Fix Applied in `loss.py`:
```python
# OLD (broken):
pred_r_clamped = pred_r.clamp(min=cfg.RANGE_R[0] * 0.9)

# NEW (fixed):
eps_m = getattr(cfg, "RANGE_EPS_M", 1e-3)
pred_r_clamped = pred_r.clamp(min=eps_m)
```

### Verification:
```bash
cd ris/MainMusic
python -c "
from ris_pytorch_pipeline.configs import cfg
from ris_pytorch_pipeline.loss import _perm_invariant_aux_loss
import torch

# Check range gradients flow
r_p = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], requires_grad=True)
r_t = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
phi_p = torch.zeros(1, cfg.K_MAX, requires_grad=True)
theta_p = torch.zeros(1, cfg.K_MAX, requires_grad=True)
phi_t = torch.zeros(1, cfg.K_MAX)
theta_t = torch.zeros(1, cfg.K_MAX)
K = torch.tensor([2])

loss = _perm_invariant_aux_loss(phi_p, theta_p, r_p, phi_t, theta_t, r_t, K)
loss.backward()
print(f'r_p.grad.norm() = {r_p.grad.norm().item():.4f}')
if r_p.grad.norm().item() > 1e-6:
    print('✅ Range gradients flow!')
else:
    print('❌ FIX NOT APPLIED!')
"
```

---

## PHASE 0: Narrowband at New Dimensions (M=64, N=256, 3.5 GHz)

**Goal**: Verify pipeline works at new dimensions before adding frequency axis.

### Step 0.1: Update configs.py (SysConfig.__init__)

**Current values (line ~13):**
```python
self.M, self.N_H, self.N_V = 16, 12, 12  # M=16, N=144
self.WAVEL = 0.3  # 1 GHz
```

**New values:**
```python
# === Carrier / wavelength ===
self.CARRIER_HZ = 3.5e9                    # 3.5 GHz FR1 mid-band
self.WAVEL = 3e8 / self.CARRIER_HZ         # ≈ 0.0857 m

# === Array dimensions ===
self.M, self.N_H, self.N_V = 64, 16, 16    # M=64 BS antennas (8×8), N=256 RIS elements
self.M_BS = self.M
self.N = self.N_H * self.N_V               # = 256

# === Derived geometry (these use WAVEL) ===
self.k0 = 2 * math.pi / self.WAVEL
self.d_H = self.d_V = 0.5 * self.WAVEL     # λ/2 spacing ≈ 0.043 m

# === Update paths ===
self.DATA_SHARDS_DIR = f"data_shards_M{self.M_BEAMS_TARGET}_L{self.L}_N{self.N}"
self.RESULTS_DIR = f"results_final_L{self.L}_{self.N_H}x{self.N_V}"
```

**Checklist:**
- [ ] Add `CARRIER_HZ = 3.5e9`
- [ ] Update `WAVEL = 3e8 / CARRIER_HZ`
- [ ] Change `M = 64` (was 16)
- [ ] Change `N_H, N_V = 16, 16` (was 12, 12)
- [ ] Verify `N = N_H * N_V` = 256
- [ ] Verify `d_H, d_V, k0` are re-derived from `WAVEL`
- [ ] Update `DATA_SHARDS_DIR` path
- [ ] Update `RESULTS_DIR` path

### Step 0.2: Verify physics.py (no changes needed)
The physics module uses `cfg.k0`, `cfg.d_H`, etc., which are derived from `cfg.WAVEL`.
As long as configs.py is updated, physics.py will use the correct values automatically.

**Checklist:**
- [ ] Confirm `nearfield_vec()` uses `cfg.k0` (line 17)
- [ ] Confirm `_rician_bs2ris()` uses `k0, d_H` from caller (line 20)

### Step 0.3: Verify dataset.py (pregen) (minor changes)

**Current (line 106-109):**
```python
h_idx = np.arange(-(cfg.N_H - 1)//2, (cfg.N_H + 1)//2) * cfg.d_H
v_idx = np.arange(-(cfg.N_V - 1)//2, (cfg.N_V + 1)//2) * cfg.d_V
```
This uses `cfg.N_H`, `cfg.N_V`, `cfg.d_H`, `cfg.d_V` — will auto-update.

**IMPORTANT (line 306-308):**
```python
y = np.zeros((n_this, chosen_L, cfg.M, 2), np.float32)
H = np.zeros((n_this, chosen_L, cfg.M, 2), np.float32)
H_full = np.zeros((n_this, cfg.M, cfg.N, 2), np.float32)
```
These use `cfg.M` and `cfg.N` — will auto-update.

**Checklist:**
- [ ] Confirm array shapes use `cfg.M`, `cfg.N` (they do)
- [ ] No code changes needed in dataset.py for Phase 0

### Step 0.4: Update model.py (CRITICAL)

**Current H_proj dimension (search for `H_proj` or `M * N`):**
The model's `H_proj` layer takes flattened `H_full` as input.
Input size = `M * N * 2` = `16 * 144 * 2 = 4608` (old)
New size = `64 * 256 * 2 = 32768`

**Find and update:**
```python
# In HybridModel.__init__, look for:
self.H_proj = nn.Linear(cfg.M * cfg.N * 2, ...)
# This should auto-update if it uses cfg.M and cfg.N
```

**Checklist:**
- [ ] Verify `H_proj` input dimension uses `cfg.M * cfg.N * 2`
- [ ] Verify covariance factor outputs use `cfg.N` for shape
- [ ] Verify any hardcoded `144` or `16` is replaced with `cfg.N` or `cfg.M`

### Step 0.4: Regenerate shards
```bash
# Generate small test set first
python -m ris_pytorch_pipeline.ris_pipeline pregen \
    --split test --n-samples 1000 --shard-size 500

# Verify
python -m ris_pytorch_pipeline.ris_pipeline doctor

# If OK, generate train/val
python -m ris_pytorch_pipeline.ris_pipeline pregen \
    --split train --n-samples 100000 --shard-size 2000
python -m ris_pytorch_pipeline.ris_pipeline pregen \
    --split val --n-samples 10000 --shard-size 2000
```

### Step 0.5: Verify pipeline
```bash
# Doctor check
python -m ris_pytorch_pipeline.ris_pipeline doctor

# Quick training (10 epochs)
python -m ris_pytorch_pipeline.ris_pipeline train --epochs 10

# R_samp vs R_true diagnostic
python diagnose_rsamp_vs_rtrue_mvdr.py
```

**Exit criterion**: R_true MVDR works (F1 > 0.9), training runs without errors.

---

## PHASE A1: Minimal Wideband (F=16)

**Goal**: Validate wideband tensor flow with minimal compute.

### Step A1.1: Add OFDM config parameters
```python
# In configs.py:
cfg.F = 16                          # pilot tones (start small)
cfg.BW_HZ = 50e6                    # 50 MHz
cfg.SCS_HZ = 30e3                   # 30 kHz SCS
cfg.NFFT = 2048                     # FFT size

# Derived
cfg.PILOT_FREQS_OFFSET = np.linspace(-cfg.BW_HZ/2, cfg.BW_HZ/2, cfg.F)
```

### Step A1.2: Update pregen.py for wideband
- [ ] Add `H_taps` generation (TR 38.901 style or simplified)
- [ ] For each sample, store:
  ```python
  {
      "y": np.array([L, F, M, 2], dtype=float32),
      "codes": np.array([L, N, 2], dtype=float32),
      # IMPORTANT: do NOT store dicts inside NPZ (pickling). Store fixed-shape arrays + mask:
      "n_paths": int,
      "alphas": np.array([P_MAX, 2], dtype=float32),
      "taus_s": np.array([P_MAX], dtype=float32),
      "aod_az": np.array([P_MAX], dtype=float32),
      "aod_el": np.array([P_MAX], dtype=float32),
      "aoa_az": np.array([P_MAX], dtype=float32),
      "aoa_el": np.array([P_MAX], dtype=float32),
      "path_mask": np.array([P_MAX], dtype=bool),
      "R_true": np.array([N, N, 2], dtype=float32),
      # ... other fields
  }
  ```
- [ ] Add function to compute `H[k]` from `H_taps`:
  ```python
  def compute_Hk_from_taps(H_taps, freq_offsets, a_bs, a_ris):
      """Compute per-tone channel H[k] from tap-domain representation."""
      # H[k] = sum_p alpha_p * exp(-j*2*pi*f_k*tau_p) * a_bs(aod_p) @ a_ris(aoa_p).H
  ```
- [ ] Generate `y[L, F, M, 2]` by looping over tones

### Step A1.3: Update dataset.py
- [ ] Load `H_taps` from shard
- [ ] Optionally compute `H[k]` on-the-fly
- [ ] Return `y` with shape `[L, F, M, 2]`

### Step A1.4: Update collate_fn.py
- [ ] Handle new `y` shape in batching
- [ ] Stack `y` to `[B, L, F, M, 2]`

### Step A1.5: Update model.py - Add frequency pooling
```python
# In HybridModel.__init__:
# IMPORTANT: avoid naive mean pooling over frequency of complex samples (it can cancel phase and
# destroy wideband cues). Prefer a small learned pooler:
#
# Option A (simple): treat real/imag as channels and apply a 1D conv over k to produce pooled features.
# Option B (slightly heavier): attention over k.

# In forward():
# y: [B, L, F, M, 2]
# y_pooled = learned_pool(y)  # [B, L, M, 2] or [B, L, D] depending on design
# ... rest of processing unchanged
```

### Step A1.6: Update train.py
- [ ] Ensure batch unpacking handles new shapes
- [ ] Pass pooled y to model (or let model do pooling)

### Step A1.7: Update infer.py
- [ ] Handle wideband sample format
- [ ] Pool over F before running model

### Step A1.8: Regenerate wideband shards
```bash
python -m ris_pytorch_pipeline.ris_pipeline pregen \
    --split test --n-samples 1000 --shard-size 500 --wideband
```

### Step A1.9: Verify wideband pipeline
```bash
python -m ris_pytorch_pipeline.ris_pipeline doctor
python -m ris_pytorch_pipeline.ris_pipeline train --epochs 5
```

**Exit criterion**: Pipeline runs without errors; loss decreases.

---

## PHASE A2: Full Wideband (F=64 or 256)

### Step A2.1: Scale up F
```python
cfg.F = 64  # or 256
```

### Step A2.2: Regenerate larger shards
```bash
# Expect ~100 GB for F=64, ~400 GB for F=256
python -m ris_pytorch_pipeline.ris_pipeline pregen \
    --split train --n-samples 100000 --shard-size 1000 --wideband
```

### Step A2.3: Run R_samp diagnostic
```bash
python diagnose_rsamp_vs_rtrue_mvdr.py --wideband
```

### Step A2.4: Full training
```bash
python -m ris_pytorch_pipeline.ris_pipeline train --epochs 60
```

### Step A2.5: MVDR-end benchmarks
```bash
python -m ris_pytorch_pipeline.ris_pipeline suite --bench B1 --limit 1000 --no-baselines
python -m ris_pytorch_pipeline.ris_pipeline suite --bench B2 --limit 1000 --no-baselines
```

**Exit criterion**: RMSPE ≤ 1.5 m, F1 ≥ 0.7

---

## Common Pitfalls to Avoid

1. **Wavelength mismatch**: Make sure ALL steering vector computations use the new `WAVEL = 0.0857 m`, not the old `0.3 m`.

2. **Shape mismatches**: After changing M and N, many hardcoded shapes will break. Search for:
   - `16` (old M)
   - `144` (old N)
   - `12` (old N_H, N_V)

3. **H_proj dimension**: The model's `H_proj` layer input size depends on `M * N * 2`. This will change from 4608 to 16384.

4. **Covariance shapes**: The output covariance is `[N, N, 2]`. When N changes from 144 to 256, all covariance-related code must handle this.

5. **Memory**: With larger M, N, F, memory usage increases significantly. Monitor GPU memory during training.

6. **Shard sizes**: Larger tensors mean larger shards. Reduce `--shard-size` to avoid out-of-memory during pregen.

---

## Quick Dimension Reference

### Narrowband (current)
| Param | Value |
|-------|-------|
| M | 16 |
| N | 144 (12×12) |
| L | 64 |
| F | 1 (implicit) |
| λ | 0.3 m |
| y shape | [L, M, 2] = [64, 16, 2] |
| H_full shape | [M, N, 2] = [16, 144, 2] |
| R_true shape | [N, N, 2] = [144, 144, 2] |

### Wideband (target)
| Param | Value |
|-------|-------|
| M | 64 (8×8 UPA) |
| N | 256 (16×16) |
| L | 64 |
| F | 64 (or 256) |
| λ | 0.0857 m |
| y shape | [L, F, M, 2] = [64, 64, 64, 2] |
| H_taps | dict with ~10-20 paths |
| R_true shape | [N, N, 2] = [256, 256, 2] |

---

## Files to Modify (Summary)

| File | Phase 0 | Phase A1 | Phase A2 |
|------|---------|----------|----------|
| `configs.py` | ✓ M, N, λ | ✓ F, BW, SCS | — |
| `pregen.py` | ✓ dimensions | ✓ H_taps, y[L,F,M] | — |
| `dataset.py` | maybe | ✓ load wideband | — |
| `collate_fn.py` | maybe | ✓ new y shape | — |
| `model.py` | ✓ H_proj dim | ✓ freq pooling | — |
| `train.py` | maybe | ✓ new tensors | — |
| `infer.py` | maybe | ✓ wideband infer | — |
| `loss.py` | — | maybe R_samp | — |
| `music_gpu.py` | — | — | Phase B only |

