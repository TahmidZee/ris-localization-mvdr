# Wideband OFDM Implementation Checklist
Date: 2026-01-30  
Reference: `OFDM_TR38901_INDOOR_PLAN.md`

This is a **step-by-step implementation checklist** for upgrading the pipeline from narrowband to wideband OFDM.

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

