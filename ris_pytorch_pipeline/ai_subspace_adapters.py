import os, sys, importlib.util
from pathlib import Path
import numpy as np

def _ensure_proj_root_on_path():
    root = os.environ.get("SUBSPACE_METHODS_PATH", "").strip()
    if not root:
        raise ImportError("Set SUBSPACE_METHODS_PATH to the project ROOT (the folder that contains 'src/').")
    proj_root = Path(root)
    if proj_root.name == "src":
        proj_root = proj_root.parent
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))
    return proj_root

def _import_by_path(py_path: Path):
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def _hunt(proj_root: Path, candidates):
    for rel in candidates:
        p = proj_root / rel
        if p.is_file():
            return _import_by_path(p)
    # fallback: crawl
    for p in proj_root.rglob("*.py"):
        if p.name.lower() in ("dcd_music.py", "subspacenet.py", "nf_subspacenet.py"):
            return _import_by_path(p)
    raise ImportError(f"Could not find any of {candidates} under {proj_root}")

def _stdtriplet(pred):
    import numpy as _np
    if isinstance(pred, tuple) and len(pred) == 3:
        return [_np.array(pred[0]).ravel(), _np.array(pred[1]).ravel(), _np.array(pred[2]).ravel()]
    if isinstance(pred, dict):
        for a,b,c in (("phi","theta","r"), ("phis","thetas","ranges")):
            if a in pred and b in pred and c in pred:
                return [_np.array(pred[a]).ravel(), _np.array(pred[b]).ravel(), _np.array(pred[c]).ravel()]
    arr = _np.array(pred)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return [arr[:,0].ravel(), arr[:,1].ravel(), arr[:,2].ravel()]
    raise ValueError("Unexpected prediction format; expected (phi,theta,r), dict with those keys, or (K,3) array.")

# ---------- DCD-MUSIC ----------
def dcd_run(R: np.ndarray, K: int, cfg=None):
    proj_root = _ensure_proj_root_on_path()
    mod = _hunt(proj_root, [
        Path("src/models_pack/dcd_music.py"),
        Path("models_pack/dcd_music.py"),
        Path("src/dcd_music.py"),
    ])

    # simple runner if repo provides it
    for fn in ("run", "infer", "estimate", "dcd_music"):
        if hasattr(mod, fn):
            return _stdtriplet(getattr(mod, fn)(R, K))

    # Class-based path
    Cls = None
    for cls_name in ("DCDMUSIC","DCDMusic","DCD_Model"):
        if hasattr(mod, cls_name):
            Cls = getattr(mod, cls_name); break
    if Cls is None:
        raise RuntimeError("DCD: no simple run() and no DCD class found in repo.")

    # Try to instantiate with minimal args; fall back to cfg
    try:
        model = Cls()
    except TypeError:
        if cfg is None:
            raise RuntimeError("DCDMUSIC requires constructor args; pass cfg or add a simple run(R,K) function in the repo.")
        # Best-effort constructor kwargs
        kw = {}
        for key in ("N_H","N_V","d_H","d_V","WAVEL"):
            if hasattr(cfg, key):
                kw[key if key!="WAVEL" else "wavelength"] = getattr(cfg, key)
        model = Cls(**kw)

    # try common predict methods
    for meth in ("run","infer","estimate","__call__"):
        if hasattr(model, meth):
            try:
                return _stdtriplet(getattr(model, meth)(R, K, cfg=cfg))
            except TypeError:
                # fallback if shim ignores cfg
                return _stdtriplet(getattr(model, meth)(R, K))


    # last resort: forward expects features; we only have R
    raise RuntimeError("DCDMUSIC has no callable run/infer/estimate/predict(R,K). Add a thin adapter in the repo that exposes def run(R,K): ...")

# ---------- NF-SubspaceNet ----------
def nfsubspacenet_run(R: np.ndarray, K: int, cfg=None):
    proj_root = _ensure_proj_root_on_path()
    mod = _hunt(proj_root, [
        Path("src/models_pack/subspacenet.py"),
        Path("models_pack/subspacenet.py"),
        Path("src/nf_subspacenet.py"),
    ])

    # simple runner if provided
    for fn in ("run", "infer", "estimate", "nf_subspacenet"):
        if hasattr(mod, fn):
            return _stdtriplet(getattr(mod, fn)(R, K))

    # Class-based
    Cls = None
    for cls_name in ("SubspaceNet","NFSubspaceNet","SubspaceNetNF"):
        if hasattr(mod, cls_name):
            Cls = getattr(mod, cls_name); break
    if Cls is None:
        raise RuntimeError("NF-SubspaceNet: no simple run() and no class found.")

    # This model typically needs weights and time-series x, not R.
    # We enforce a checkpoint and rely on repo-provided predict method if any.
    ckpt = os.environ.get("NFSSN_CKPT", "").strip()
    if not ckpt:
        raise RuntimeError("Set NFSSN_CKPT to a valid checkpoint for NF-SubspaceNet, or add a simple run(R,K) in the repo.")

    try:
        model = Cls()
    except TypeError:
        if cfg is None:
            raise RuntimeError("NF-SubspaceNet requires constructor args; pass cfg or add a simple run(R,K).")
        kw = {}
        for key in ("N_H","N_V","d_H","d_V","WAVEL"):
            if hasattr(cfg, key):
                kw[key if key!="WAVEL" else "wavelength"] = getattr(cfg, key)
        model = Cls(**kw)

    # try to load checkpoint with common methods
    loaded = False
    for meth in ("load","load_state_dict","from_pretrained"):
        if hasattr(model, meth):
            try: getattr(model, meth)(ckpt); loaded=True; break
            except Exception as e: last_err = e
    if not loaded:
        raise RuntimeError("Could not load NF-SubspaceNet weights via load/load_state_dict/from_pretrained.")

    for meth in ("run","infer","estimate","predict","__call__"):
        if hasattr(model, meth):
            try:
                return _stdtriplet(getattr(model, meth)(R, K, cfg=cfg))
            except TypeError:
                return _stdtriplet(getattr(model, meth)(R, K))


    raise RuntimeError("NF-SubspaceNet model lacks a callable run/infer/estimate/predict(R,K). Add a thin adapter in the repo that exposes def run(R,K): ...")
