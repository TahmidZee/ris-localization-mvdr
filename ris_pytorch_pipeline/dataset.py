# ris_pytorch_pipeline/dataset.py
import numpy as np, math, os
from torch.utils.data import Dataset
import torch
from .configs import cfg, mdl_cfg
from .angle_pipeline import build_sample_covariance_from_snapshots  # For offline R_samp precompute
from .physics import nearfield_vec, _rician_bs2ris, quantise_phase
from pathlib import Path

def _unpack_ptr_np(ptr_vec, K, KMAX):
    a = np.asarray(ptr_vec, dtype=np.float32).reshape(-1)
    if a.size < 3*KMAX:
        pad = np.full(3*KMAX, np.nan, np.float32); pad[:a.size] = a; a = pad
    phi   = a[0:KMAX][:K]
    theta = a[KMAX:2*KMAX][:K]
    r     = a[2*KMAX:3*KMAX][:K]
    return phi, theta, r


# === Sampling overrides (activated only during pregen via ris_pipeline) ===
class _SamplingOverrides:
    __slots__ = ("use", "phi_fov_rad", "theta_fov_rad", "r_min", "r_max",
                 "p_db_range", "noise_std_range", "k_range", "snr_db_range",
                 "dr_enabled", "dr_phase_sigma", "dr_amp_jitter", "dr_dropout_p",
                 "dr_wav_jitter", "dr_dspacing_jitter",
                 "grid_offset_enabled", "grid_offset_frac")
    def __init__(self):
        self.use = False
        self.phi_fov_rad = None; self.theta_fov_rad = None
        self.r_min = None; self.r_max = None
        self.p_db_range = None; self.noise_std_range = None; self.k_range = None
        self.snr_db_range = None
        self.dr_enabled = False; self.dr_phase_sigma = 0.0; self.dr_amp_jitter = 0.0; self.dr_dropout_p = 0.0
        self.dr_wav_jitter = 0.0; self.dr_dspacing_jitter = 0.0
        self.grid_offset_enabled = False; self.grid_offset_frac = 0.0

SAMPLING_OVERRIDES = _SamplingOverrides()

def set_sampling_overrides_from_cfg(mdl_cfg):
    SAMPLING_OVERRIDES.use = True
    SAMPLING_OVERRIDES.phi_fov_rad   = np.deg2rad(getattr(mdl_cfg, "TRAIN_PHI_FOV_DEG", 60.0))
    SAMPLING_OVERRIDES.theta_fov_rad = np.deg2rad(getattr(mdl_cfg, "TRAIN_THETA_FOV_DEG", 30.0))
    rmin, rmax = getattr(mdl_cfg, "TRAIN_R_MIN_MAX", (0.5, 10.0))  # Extended to 10m for L=16 system
    SAMPLING_OVERRIDES.r_min, SAMPLING_OVERRIDES.r_max = float(rmin), float(rmax)
    SAMPLING_OVERRIDES.p_db_range      = getattr(mdl_cfg, "P_DB_RANGE", (-10.0, 10.0))
    SAMPLING_OVERRIDES.noise_std_range = getattr(mdl_cfg, "NOISE_STD_RANGE", (1.5e-3, 7e-3))
    SAMPLING_OVERRIDES.snr_db_range    = getattr(mdl_cfg, "SNR_DB_RANGE", (-5.0, 20.0))
    SAMPLING_OVERRIDES.dr_enabled         = bool(getattr(mdl_cfg, "DR_ENABLED", False))
    SAMPLING_OVERRIDES.dr_phase_sigma     = float(getattr(mdl_cfg, "DR_PHASE_SIGMA", 0.04))
    SAMPLING_OVERRIDES.dr_amp_jitter      = float(getattr(mdl_cfg, "DR_AMP_JITTER", 0.05))
    SAMPLING_OVERRIDES.dr_dropout_p       = float(getattr(mdl_cfg, "DR_DROPOUT_P", 0.03))
    SAMPLING_OVERRIDES.dr_wav_jitter      = float(getattr(mdl_cfg, "DR_WAVELEN_JITTER", 0.002))
    SAMPLING_OVERRIDES.dr_dspacing_jitter = float(getattr(mdl_cfg, "DR_DSPACING_JITTER", 0.005))
    SAMPLING_OVERRIDES.grid_offset_enabled = bool(getattr(mdl_cfg, "GRID_OFFSET_ENABLED", False))
    SAMPLING_OVERRIDES.grid_offset_frac    = float(getattr(mdl_cfg, "GRID_OFFSET_FRAC", 0.5))

# --- helpers to apply CLI overrides during sampling ---
def _sample_angles_and_range():
    if SAMPLING_OVERRIDES.use:
        phi_fov   = float(SAMPLING_OVERRIDES.phi_fov_rad)
        theta_fov = float(SAMPLING_OVERRIDES.theta_fov_rad)
        rmin, rmax= float(SAMPLING_OVERRIDES.r_min), float(SAMPLING_OVERRIDES.r_max)
        phi   = np.random.uniform(-phi_fov,   +phi_fov)
        theta = np.random.uniform(-theta_fov, +theta_fov)
        r     = np.random.uniform(rmin, rmax)
    else:
        phi   = np.random.uniform(-cfg.ANGLE_RANGE_PHI, +cfg.ANGLE_RANGE_PHI)
        theta = np.random.uniform(-cfg.ANGLE_RANGE_THETA, +cfg.ANGLE_RANGE_THETA)
        r     = np.random.uniform(*cfg.RANGE_R)
    return phi, theta, r

def _sample_power_and_noise():
    if SAMPLING_OVERRIDES.use:
        p_lo, p_hi = SAMPLING_OVERRIDES.p_db_range
        n_lo, n_hi = SAMPLING_OVERRIDES.noise_std_range
        p_db   = np.random.uniform(p_lo, p_hi)
        sigma  = np.random.uniform(n_lo, n_hi)
    else:
        p_db   = np.random.uniform(*cfg.P_DB_RANGE)
        sigma  = np.random.uniform(*cfg.NOISE_STD_RANGE)
    return float(p_db), float(sigma)

def _apply_grid_offset(codes):
    if SAMPLING_OVERRIDES.grid_offset_enabled:
        frac = float(SAMPLING_OVERRIDES.grid_offset_frac)
        L, N = codes.shape
        shift = np.exp(1j * 2*np.pi * frac * np.random.uniform(-1,1,size=(L,1))).astype(np.complex64)
        return (codes * shift).astype(np.complex64)
    return codes

def _apply_domain_randomization(H, codes):
    if not SAMPLING_OVERRIDES.dr_enabled:
        return H, codes
    L, N = codes.shape
    drop_p = float(SAMPLING_OVERRIDES.dr_dropout_p)
    amp_j  = float(SAMPLING_OVERRIDES.dr_amp_jitter)
    phsig  = float(SAMPLING_OVERRIDES.dr_phase_sigma)
    mask   = (np.random.rand(L, N) > drop_p).astype(np.float32)
    gain   = 1.0 + np.random.uniform(-amp_j, +amp_j, size=(L, N))
    dphi   = np.random.normal(0.0, phsig, size=(L, N))
    codes  = codes * mask * gain * np.exp(1j*dphi)
    return H, codes


h_idx = np.arange(-(cfg.N_H - 1)//2, (cfg.N_H + 1)//2) * cfg.d_H
v_idx = np.arange(-(cfg.N_V - 1)//2, (cfg.N_V + 1)//2) * cfg.d_V
h_mesh, v_mesh = np.meshgrid(h_idx, v_idx, indexing="xy")
h_flat_pristine = h_mesh.reshape(-1).astype(np.float32)
v_flat_pristine = v_mesh.reshape(-1).astype(np.float32)

class RISDataset(Dataset):
    def __init__(self, n_samples=10000, phase=2, eta_perturb=0.0):
        self.n_samples = n_samples; self.phase = phase; self.eta_perturb = eta_perturb
    def __len__(self): return self.n_samples

    def snapshot(self, phase_bits=None, eta_perturb=None, override_L=None,
                 codebook_type="DFT", coherent=False,
                 K_override=None, snr_db_override=None, p_db_override=None, sigma_override=None):
        if phase_bits is None: phase_bits = mdl_cfg.PHASE_BITS
        if eta_perturb is None: eta_perturb = self.eta_perturb
        L = int(override_L) if override_L is not None else cfg.L
        K = int(K_override) if K_override is not None else np.random.randint(1, cfg.K_MAX + 1)
        phi = np.zeros(K, np.float32); theta = np.zeros(K, np.float32); r = np.zeros(K, np.float32)
        for k in range(K):
            phi[k], theta[k], r[k] = _sample_angles_and_range()

        h_flat, v_flat = h_flat_pristine.copy(), v_flat_pristine.copy()
        if eta_perturb:
            jitter_h = np.random.uniform(-eta_perturb, eta_perturb, cfg.N) * (cfg.WAVEL / 2)
            jitter_v = np.random.uniform(-eta_perturb, eta_perturb, cfg.N) * (cfg.WAVEL / 2)
            h_flat += jitter_h; v_flat += jitter_v

        # 1) (optional) keep p_db to shape absolute signal power, but SNR is set later
        if p_db_override is not None and sigma_override is not None:
            p_db, _ = float(p_db_override), float(sigma_override)
        else:
            p_db, _ = _sample_power_and_noise()  # legacy, only p_db used now
        pW = 10 ** (p_db / 10) / 1e3

        # 2) build s, A, H as before
        if coherent: s = np.ones(K, np.complex64)
        else: s = (np.random.randn(K)+1j*np.random.randn(K)).astype(np.complex64)/np.sqrt(2)

        A = np.stack([ np.sqrt(pW*(cfg.WAVEL**2)/(4*np.pi*r[k])**2) *
                       nearfield_vec(cfg, phi[k], theta[k], r[k], h_flat, v_flat) for k in range(K) ], 1)
        # Use azimuth range for BS-RIS channel (max of the two FOV ranges)
        angle_range = max(cfg.ANGLE_RANGE_PHI, cfg.ANGLE_RANGE_THETA)
        H = _rician_bs2ris(cfg.M, cfg.N, cfg.k0, cfg.d_H, np.random.uniform(*cfg.KFAC_RANGE), angle_range).astype(np.complex64)

        # CRITICAL: Use M_beams for codebook size, L for time snapshots
        # M_beams = total spatial beams available (e.g., 32)
        # L = time snapshots to use (e.g., 16)
        # Use strided sampling to get balanced coverage of the full M_beams pool
        M_beams = getattr(cfg, 'M_BEAMS_TARGET', L)  # Fallback to L if not set
        half = M_beams // 2
        
        if codebook_type == "DFT":
            # Use 2D separable DFT codebook (RECOMMENDED)
            # Handle different cases: M_beams > L, M_beams == L, M_beams < L
            if hasattr(cfg, 'RIS_2D_DFT_COLS'):
                if M_beams > L:
                    # Strided indices: [0, 2, 4, ...] for M_beams=32, L=16 (stride=2)
                    stride = M_beams // L
                    indices = np.arange(0, M_beams, stride)[:L]  # Ensure exactly L beams
                    cod = cfg.RIS_2D_DFT_COLS[indices]  # [L, N] with balanced spatial coverage
                elif M_beams == L:
                    cod = cfg.RIS_2D_DFT_COLS[:L]  # Use all beams
                else:
                    # M_beams < L: Cycle through the codebook to get L beams
                    # Repeat the codebook as needed to get exactly L beams
                    n_repeats = (L + M_beams - 1) // M_beams  # Ceiling division
                    cod_repeated = np.tile(cfg.RIS_2D_DFT_COLS, (n_repeats, 1))  # Repeat codebook
                    cod = cod_repeated[:L]  # Take exactly L beams
            else:
                # Fallback to 1D DFT codebook
                if L <= len(cfg.RIS_CONFIG_DFT_COLS):
                    cod = cfg.RIS_CONFIG_DFT_COLS[:L]
                else:
                    # Cycle through 1D DFT codebook
                    n_repeats = (L + len(cfg.RIS_CONFIG_DFT_COLS) - 1) // len(cfg.RIS_CONFIG_DFT_COLS)
                    cod_repeated = np.tile(cfg.RIS_CONFIG_DFT_COLS, (n_repeats, 1))
                    cod = cod_repeated[:L]
        elif codebook_type == "random":
            cod = np.exp(1j * 2 * np.pi * np.random.rand(L, cfg.N))
        elif codebook_type == "hadamard":
            import scipy.linalg as la
            n0 = 1 << int(np.ceil(np.log2(cfg.N)))
            Hwal = la.hadamard(n0).astype(np.complex64)
            cod = Hwal[:L, :cfg.N]
        else:
            # Mixed: use 2D DFT for first half, random for second half
            # Use strided sampling for DFT part, random for second half
            half_L = L // 2  # Split L snapshots, not M_beams
            if hasattr(cfg, 'RIS_2D_DFT_COLS'):
                if M_beams > half_L:
                    # Strided sampling for balanced coverage
                    stride = M_beams // half_L
                    indices = np.arange(0, M_beams, stride)[:half_L]
                    cod_2d = cfg.RIS_2D_DFT_COLS[indices]
                else:
                    cod_2d = cfg.RIS_2D_DFT_COLS[:half_L]
            else:
                cod_2d = cfg.RIS_CONFIG_DFT_COLS[:half_L]
            cod = np.vstack([ cod_2d, np.exp(1j*2*np.pi*np.random.rand(L-half_L, cfg.N)) ])
        cod = _apply_grid_offset(cod)
        codes = quantise_phase(cod, bits=phase_bits).astype(np.complex64)
        H, codes = _apply_domain_randomization(H, codes)

        # 3) build CLEAN snapshots first (no noise yet)
        # CRITICAL FIX: Apply signal gain for numerical stability
        # Path loss makes A tiny (~1e-4), which makes y tiny (~1e-3)
        # This breaks LS inversion in hybrid covariance (Tikhonov needs ||y|| ≈ O(1))
        # Scale factor doesn't affect SNR (noise is scaled to match p_sig)
        SIGNAL_GAIN = 1000.0  # Boost y to O(1) range for numerical stability
        
        y_clean = np.empty((L, cfg.M), np.complex64)
        for l in range(L):
            Hl = H @ np.diag(codes[l])
            y_clean[l] = SIGNAL_GAIN * (Hl @ (A @ s))

        # 4) compute signal power and choose / hit target SNR
        p_sig = float(np.mean(np.abs(y_clean)**2))  # clean signal avg power
        if snr_db_override is not None:
            snr_db = float(snr_db_override)
        elif SAMPLING_OVERRIDES.use and SAMPLING_OVERRIDES.snr_db_range is not None and cfg.SNR_TARGETED:
            snr_db = float(np.random.uniform(*SAMPLING_OVERRIDES.snr_db_range))
        elif cfg.SNR_TARGETED:
            snr_db = float(np.random.uniform(*cfg.SNR_DB_RANGE))
        else:
            # Legacy non-targeted path (not recommended)
            _, sigma_n = _sample_power_and_noise()
            snr_db = 10.0 * math.log10(max(p_sig, 1e-12) / max(float(sigma_n**2), 1e-12))

        # 5) set noise std to match the target SNR
        snr_lin = 10.0 ** (snr_db / 10.0)
        p_noise = max(p_sig / snr_lin, 1e-12)
        sigma_n = math.sqrt(p_noise)

        # 6) now draw noise and build noisy snapshots
        noise = (np.random.randn(L, cfg.M)+1j*np.random.randn(L, cfg.M)).astype(np.complex64)/np.sqrt(2.0)
        y_snaps = y_clean + sigma_n * noise

        A0 = np.stack([ np.sqrt(pW*(cfg.WAVEL**2)/(4*np.pi*r[k])**2) *
                        nearfield_vec(cfg, phi[k], theta[k], r[k], h_flat, v_flat) for k in range(K) ], 1)
        R_true = A0 @ np.diag(s) @ np.diag(s.conj()) @ A0.conj().T
        R_true = 0.5 * (R_true + R_true.conj().T)
        R_true *= (cfg.N / (np.trace(R_true).real + 1e-9))  # Normalize to tr(R) = N, not 1

        pad = lambda v: np.pad(v.astype(np.float32), (0, cfg.K_MAX - len(v)), 'constant')
        phi_p, theta_p, r_p = pad(phi), pad(theta), pad(r)

        # Fixed measured SNR (don't subtract noise from signal)
        p_sig = float(np.mean(np.abs(y_clean)**2))
        p_noise = float(sigma_n ** 2)
        snr_db = float(10 * math.log10(max(p_sig, 1e-12) / max(p_noise, 1e-12)))
        # Clip to intended training range
        snr_db = float(np.clip(snr_db, cfg.SNR_DB_RANGE[0], cfg.SNR_DB_RANGE[1]))
        return dict(y_cplx=y_snaps.astype(np.complex64), H_cplx=H.astype(np.complex64), codes=codes.astype(np.complex64),
                    R_true_cplx=R_true.astype(np.complex64), φ_padded=phi_p, θ_padded=theta_p, r_padded=r_p,
                    φ=phi.astype(np.float32), θ=theta.astype(np.float32), r=r.astype(np.float32),
                    K=int(K), snr_db=snr_db)

def to_ri(z): return np.stack([z.real, z.imag], axis=-1).astype(np.float32)


def prepare_shards(out_dir, n_samples: int, shard_size: int = 25000,
                   seed: int = 42, eta_perturb: float = 0.0, override_L: int = None):
    """
    Generate uncompressed .npz shards: y, H, codes, ptr, K, snr, R.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    chosen_L = int(override_L) if override_L is not None else int(cfg.L)
    n_shards = (n_samples + shard_size - 1) // shard_size

    for s in range(n_shards):
        n_this = min(shard_size, n_samples - s*shard_size)
        y = np.zeros((n_this, chosen_L, cfg.M, 2), np.float32)
        H = np.zeros((n_this, chosen_L, cfg.M, 2), np.float32)  # H_eff per snapshot (for backward compat)
        H_full = np.zeros((n_this, cfg.M, cfg.N, 2), np.float32)  # NEW: Full BS→RIS channel [M_BS, N]
        codes = np.zeros((n_this, chosen_L, cfg.N, 2), np.float32)
        ptr = np.zeros((n_this, 3*cfg.K_MAX), np.float32)
        Kvec = np.zeros((n_this,), np.int32)
        snr = np.zeros((n_this,), np.float32)
        R = np.zeros((n_this, cfg.N, cfg.N, 2), np.float32)
        R_samp = np.zeros((n_this, cfg.N, cfg.N, 2), np.float32)  # NEW: offline sample covariance

        for i in range(n_this):
            np.random.seed(int(seed) + s*shard_size + i)
            ds = RISDataset(n_samples=1, phase=2, eta_perturb=eta_perturb)
            sdict = ds.snapshot(override_L=chosen_L)
            if sdict['y_cplx'].shape[0] != chosen_L:
                raise ValueError("Snapshot L mismatch")

            y[i]     = to_ri(sdict['y_cplx'])
            
            # Store both H_full (true BS→RIS channel) and H_eff (per-snapshot effective response)
            H_cplx = sdict['H_cplx']  # [M_BS, N] - BS to RIS channel matrix
            C_cplx = sdict['codes']   # [L, N] - RIS codes for L snapshots
            
            # 1) Store the full channel matrix once per sample (NEW!)
            H_full[i] = to_ri(H_cplx.astype(np.complex64))  # [M_BS, N, 2]
            
            # 2) Also store per-snapshot effective H for backward compatibility
            H_eff = np.zeros((chosen_L, cfg.M), dtype=np.complex64)
            for ell in range(chosen_L):
                c_ell = C_cplx[ell, :]           # [N] - code for snapshot ℓ
                H_eff[ell] = H_cplx @ c_ell      # [M_BS] - effective response for snapshot ℓ
            
            H[i] = to_ri(H_eff)  # [L, M_BS, 2]
            codes[i] = to_ri(sdict['codes'])
            ptr[i]   = np.concatenate([sdict['φ_padded'], sdict['θ_padded'], sdict['r_padded']])
            Kvec[i]  = sdict['K']
            snr[i]   = sdict['snr_db']

            Rc = sdict.get('R_true_cplx', None)
            if Rc is None:
                phi = sdict['φ']; theta = sdict['θ']; r = sdict['r']; K = int(sdict['K'])
                A0 = np.stack([ nearfield_vec(cfg, phi[k], theta[k], r[k]) for k in range(K) ], 1)
                svec = np.ones((K,), np.complex64)
                Rc = A0 @ np.diag(svec) @ np.diag(np.conj(svec)) @ A0.conj().T
            Rc = 0.5*(Rc + Rc.conj().T)
            tr = float(np.trace(Rc).real)
            Rc = Rc * (cfg.N / max(tr, 1e-9))  # Normalize to tr(R) = N, not 1
            R[i] = to_ri(Rc.astype(np.complex64))

            # --- NEW: Offline R_samp precompute (NumPy, CPU) for hybrid blending ---
            try:
                y_cplx = sdict['y_cplx'].astype(np.complex64)        # [L, M]
                H_cplx = sdict['H_cplx'].astype(np.complex64)        # [M, N]
                C_cplx = sdict['codes'].astype(np.complex64)         # [L, N]
                # build_sample_covariance_from_snapshots expects shapes [L,M], [L,M], [L,N]
                # Our H_cplx is [M,N]; the builder uses its own per-snapshot formulation
                R_samp_np = build_sample_covariance_from_snapshots(y_cplx, np.repeat(H_cplx[None, :, :], chosen_L, axis=0), C_cplx, cfg)
                # Hermitize (defensive) and convert to RI
                R_samp_np = 0.5 * (R_samp_np + R_samp_np.conj().T)
                R_samp[i] = to_ri(R_samp_np.astype(np.complex64))
            except Exception:
                # If it fails, leave zeros; training will fall back to pure R_pred
                R_samp[i] = 0.0

        np.savez(out_dir / f"shard_{s:03d}.npz",
                 y=y, H=H, H_full=H_full, codes=codes, ptr=ptr, K=Kvec, snr=snr, R=R, R_samp=R_samp)

class ShardNPZDataset(Dataset):
    """
    .npz shard dataset with fields: y, H, codes, ptr, K, snr, R, (optional) R_samp
    """
    def __init__(self, npz_paths_or_dir):
        self.paths, self.meta = [], []
        self._npz_cache = {}
        self._worker_pid = None

        def _add_file(p: Path):
            with np.load(p, mmap_mode="r") as z:
                n = int(z["y"].shape[0]); L = int(z["y"].shape[1])
            self.paths.append(str(p)); self.meta.append((str(p), n, L))

        def _add(x):
            p = Path(x)
            if p.is_dir():
                for f in sorted(p.glob("*.npz")): _add_file(f)
            else:
                if p.suffix.lower() != ".npz": raise ValueError(f"Not an .npz file: {p}")
                _add_file(p)

        if isinstance(npz_paths_or_dir, (list, tuple)):
            for x in npz_paths_or_dir: _add(x)
        else:
            _add(npz_paths_or_dir)

        if not self.meta: raise FileNotFoundError("No .npz shards found.")
        Ls = {L for _,_,L in self.meta}
        if len(Ls) != 1:
            raise ValueError(f"All shards must share the same L. Found {sorted(Ls)}.")

        self.index_map = []
        for si, (p, n, L) in enumerate(self.meta):
            for i in range(n): self.index_map.append((si, i))

    def __len__(self): return len(self.index_map)

    def _ensure_worker_cache(self):
        import os
        pid = os.getpid()
        if self._worker_pid != pid:
            self._npz_cache = {}
            self._worker_pid = pid

    def __getitem__(self, idx):
        self._ensure_worker_cache()
        si, i = self.index_map[idx]
        p, n, L = self.meta[si]

        z = self._npz_cache.get(p)
        if z is None:
            z = np.load(p, allow_pickle=False, mmap_mode="r")
            self._npz_cache[p] = z

        y     = z["y"][i]
        H     = z["H"][i]
        H_full = z["H_full"][i] if ("H_full" in z.files) else None  # NEW: Read full channel if available
        codes = z["codes"][i]
        ptr   = z["ptr"][i]
        K     = int(z["K"][i])
        snr   = float(z["snr"][i])
        R     = z["R"][i] if ("R" in z.files) else None
        R_samp = z["R_samp"][i] if ("R_samp" in z.files) else None

        # Convert to tensors (same as your current code)
        y_t   = torch.from_numpy(y)
        H_t   = torch.from_numpy(H)
        H_full_t = (torch.from_numpy(H_full) if H_full is not None else None)  # NEW
        C_t   = torch.from_numpy(codes)
        ptr_t = torch.from_numpy(ptr)
        K_t   = torch.tensor(K, dtype=torch.long)
        snr_t = torch.tensor(snr, dtype=torch.float32)
        R_t   = (torch.from_numpy(R) if R is not None else None)
        R_samp_t = (torch.from_numpy(R_samp) if R_samp is not None else None)

        # --- NEW: decode ptr -> (phi, theta, r) as float32 tensors, length K ---
        # We work in torch to avoid extra numpy hops.
        KMAX = int(cfg.K_MAX)
        a = ptr_t.reshape(-1).to(torch.float32)
        if a.numel() < 3 * KMAX:
            pad = torch.full((3 * KMAX,), float("nan"), dtype=torch.float32)
            pad[: a.numel()] = a
            a = pad
        k = int(K_t.item())
        phi_t   = a[0:KMAX][:k].clone()
        theta_t = a[KMAX:2 * KMAX][:k].clone()
        r_t     = a[2 * KMAX:3 * KMAX][:k].clone()

        return dict(
            y=y_t, H=H_t, H_full=H_full_t, codes=C_t, ptr=ptr_t, K=K_t, snr=snr_t, R=R_t, R_samp=R_samp_t,
            phi=phi_t, theta=theta_t, r=r_t,
        )



# -------- split helpers --------

def prepare_split_shards(root_dir: Path, n_train: int, n_val: int, n_test: int,
                         shard_size: int = 25000, seed: int = 42,
                         eta_perturb: float = 0.0, override_L: int = None):
    """
    Generate {train,val,test} splits under root_dir using the current SAMPLING_OVERRIDES.
    """
    root = Path(root_dir)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "val").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)

    if n_train > 0:
        prepare_shards(root / "train", n_train, shard_size, seed, eta_perturb, override_L)
    if n_val > 0:
        prepare_shards(root / "val",   n_val,   shard_size, seed+123, eta_perturb, override_L)
    if n_test > 0:
        prepare_shards(root / "test",  n_test,  shard_size, seed+456, eta_perturb, override_L)


