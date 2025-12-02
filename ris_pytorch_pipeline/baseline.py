from pathlib import Path
import os, importlib.util, inspect, numpy as np, numpy.linalg as la
from .configs import cfg, mdl_cfg
from .physics import nearfield_vec

from .ai_subspace_adapters import dcd_run as _dcd_run, nfsubspacenet_run as _nfssn_run
from .ramezani_mod_music import ramezani_modified_music as _ramezani_mm
from . import configs as _cfgmod
cfg = _cfgmod.cfg; mdl_cfg = _cfgmod.mdl_cfg



def _anti_diag_vector(R, N_H, N_V):
    N = N_H * N_V; ybar = np.empty(N, dtype=np.complex64)
    for n in range(N): ybar[n] = R[n, N-1-n]
    return ybar

def modified_music_ris(R_incident, K, N_H, N_V, d_H, d_V, wavelength, DH=6, DV=6, grid_phi=41, grid_theta=41):
    N = N_H * N_V; assert R_incident.shape == (N, N)
    ybar = _anti_diag_vector(R_incident, N_H, N_V)
    ymat = ybar.reshape(N_V, N_H)
    JH, JV = N_H - DH + 1, N_V - DV + 1
    if JH <= 0 or JV <= 0: raise ValueError("Sub-RIS too large")
    J = JH * JV; D = DH * DV
    Ys = np.empty((J, D), np.complex64); idx=0
    for v0 in range(JV):
        for h0 in range(JH):
            Ys[idx] = ymat[v0:v0+DV, h0:h0+DH].reshape(-1); idx+=1
    Rbar = (Ys.conj().T @ Ys) / J
    evals, evecs = la.eigh(Rbar); Un_bar = evecs[:, np.argsort(evals)[:-K]]
    k0 = 2*np.pi / wavelength
    # Match main model FOV: φ ±60°, θ ±30°
    phi_grid = np.linspace(-np.pi/3.0, np.pi/3.0, grid_phi)      # ±60° azimuth
    theta_grid = np.linspace(-np.pi/6.0, np.pi/6.0, grid_theta)  # ±30° elevation
    spec = np.zeros((grid_phi, grid_theta), np.float32)
    def b_vec(alpha, beta, DH, DV):
        h = np.arange(DH, dtype=np.float32); v = np.arange(DV, dtype=np.float32)
        H = np.exp(1j * 2.0 * alpha * h); V = np.exp(1j * 2.0 * beta  * v)
        return (V[:,None]*H[None,:]).reshape(-1).astype(np.complex64)
    for i,phi in enumerate(phi_grid):
        for j,theta in enumerate(theta_grid):
            alpha = k0 * d_H * np.sin(phi) * np.cos(theta)
            beta  = k0 * d_V * np.sin(theta)
            b = b_vec(alpha, beta, DH, DV)
            den = (b.conj() @ Un_bar @ Un_bar.conj().T @ b).real
            spec[i,j] = 1.0 / (den + 1e-12)
    flat = spec.reshape(-1); idxs = np.argpartition(-flat, K)[:K]
    phi_est   = [ float(phi_grid[q // grid_theta]) for q in idxs ]
    theta_est = [ float(theta_grid[q %  grid_theta]) for q in idxs ]
    vals=[float(flat[q]) for q in idxs]; ord2=np.argsort(vals)[::-1]
    return [phi_est[o] for o in ord2], [theta_est[o] for o in ord2], spec

def decoupled_mod_music(R_incident, K, grid_size_ang=41, grid_size_r=61, N_H=None, N_V=None, d_H=0.15, d_V=0.15, wavelength=0.3, DH=6, DV=6):
    N = R_incident.shape[0]
    if N_H is None or N_V is None:
        N_H = int(round(np.sqrt(N))); N_V = int(np.ceil(N/N_H))
        if N_H*N_V!=N: raise ValueError("Provide N_H,N_V for non-rectangular RIS")
    phi, theta, _ = modified_music_ris(R_incident, K, N_H, N_V, d_H, d_V, wavelength, DH, DV, grid_size_ang, grid_size_ang)
    evals, evecs = la.eigh(R_incident); Un = evecs[:, np.argsort(evals)[:-K]]
    r_grid = np.linspace(cfg.RANGE_R[0], cfg.RANGE_R[1], grid_size_r)
    rng = []
    for p,t in zip(phi[:K], theta[:K]):
        best_val, best_r = -1.0, None
        for r in r_grid:
            a = nearfield_vec(cfg, p, t, r).astype(np.complex64)[:,None]
            den = (a.conj().T @ Un @ Un.conj().T @ a).real.item()
            val = 1.0/(den+1e-12)
            if val>best_val: best_val, best_r = val, r
        rng.append(float(best_r))
    return phi[:K], theta[:K], rng, _

def incident_cov_from_snaps(y_lm, H_mn, codes_ln):
    L, M = y_lm.shape; N = H_mn.shape[1]
    GT = np.empty((L*M, N), np.complex64); yT = y_lm.reshape(L*M).astype(np.complex64)
    for l in range(L): GT[l*M:(l+1)*M,:] = H_mn @ np.diag(codes_ln[l])
    A = GT.conj().T @ GT; b = GT.conj().T @ yT
    x = la.solve(A + 1e-6*np.eye(N, dtype=np.complex64), b)
    R = np.outer(x, x.conj()) / max(1,L)
    return R.astype(np.complex64)


def ramezani_mod_music_wrapper(y=None, H=None, codes=None, R=None, *,
                               K=None, N_H=None, N_V=None, d_H=None, d_V=None,
                               wavelength=None, DH=None, DV=None,
                               range_grid=61, r_min=None, r_max=None,
                               phi_grid=None, theta_grid=None):
    assert (R is not None) or (y is not None and H is not None and codes is not None), \
        "Pass either R or (y,H,codes)."
    lamb = float(wavelength)
    k0 = 2*np.pi/lamb
    phi_th, theta_th, rng = _ramezani_mm(
        y=y, H=H, codes=codes, R=R,
        N_H=int(N_H), N_V=int(N_V),
        d_H=float(d_H), d_V=float(d_V),
        lamb=lamb, k0=k0,
        K=int(K) if K is not None else None,
        DH=int(DH) if DH is not None else 3,
        DV=int(DV) if DV is not None else 3,
        grid_phi=phi_grid, grid_theta=theta_grid,
        range_grid=int(range_grid),
        r_min=float(r_min) if r_min is not None else float(cfg.RANGE_R[0]),
        r_max=float(r_max) if r_max is not None else float(cfg.RANGE_R[1]),
    )
    return phi_th, theta_th, rng






def _load_ext(preferred_rel_paths):
    import importlib.util, os, fnmatch
    root = os.environ.get("SUBSPACE_METHODS_PATH", "").strip()
    if not root or not os.path.isdir(root):
        raise ImportError("Set SUBSPACE_METHODS_PATH to your AI-Subspace-Methods/src folder")

    # 1) Try the given relative suggestions first
    for rel in preferred_rel_paths:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location(Path(p).stem, p)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            return mod

    # 2) Otherwise search recursively for a plausible filename
    targets = ["dcd_music.py", "subspacenet.py", "nf_subspacenet.py"]
    for dirpath, _, filenames in os.walk(root):
        for t in targets:
            for fname in fnmatch.filter(filenames, t):
                p = os.path.join(dirpath, fname)
                spec = importlib.util.spec_from_file_location(Path(p).stem, p)
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                return mod

    raise ImportError(f"Could not find any of {targets} under {root}")


def _pick_entry(mod, *names):
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn): return fn
    for attr in dir(mod):
        obj = getattr(mod, attr, None)
        if inspect.isclass(obj):
            try: inst = obj()
            except Exception: continue
            for m in ("__call__","infer","estimate","run"):
                if hasattr(inst, m) and callable(getattr(inst, m)): return getattr(inst, m)
    raise ImportError(f"No suitable entry in {mod.__name__}")

def dcd_music_wrapper(R, K):
    return _dcd_run(R, int(K), cfg)

def nf_subspacenet_wrapper(R, K):
    return _nfssn_run(R, int(K), cfg)
