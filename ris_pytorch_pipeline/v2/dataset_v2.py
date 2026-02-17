"""
V2 dataset utilities.

Wideband-first behavior:
- Prefer OFDM/TR38.901 shards (`y: [L,F,M,2]`, tap-domain channel fields)
- Reuse v1 shard loader only as an explicit fallback.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ..dataset import ShardNPZDataset
from .config_v2 import v2_cfg, v2_mdl


def _list_npz_files(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.glob("*.npz"))
    if path.suffix.lower() == ".npz" and path.exists():
        return [path]
    return []


def _is_memmap_shard_dir(path: Path) -> bool:
    return path.is_dir() and (path / "y.npy").exists() and (path / "codes.npy").exists()


def _list_memmap_shard_dirs(path: Path) -> List[Path]:
    if _is_memmap_shard_dir(path):
        return [path]
    if path.is_dir():
        out = [p for p in sorted(path.iterdir()) if _is_memmap_shard_dir(p)]
        return out
    return []


def resolve_shards_train_val() -> Tuple[Path, Path, bool]:
    """
    Resolve train/val shard locations.

    Returns:
      train_path, val_path, is_wideband
    """
    prefer_memmap = bool(getattr(v2_cfg, "PREFER_MEMMAP_SHARDS", True))

    # 1) Preferred: mmap shard directories (true O(1) sample access).
    wb_mm_root = Path(getattr(v2_cfg, "DATA_SHARDS_WIDEBAND_MEMMAP_DIR", "data_shards_ofdm_tr38901_mmap"))
    wb_mm_train = Path(getattr(v2_cfg, "DATA_SHARDS_WIDEBAND_MEMMAP_TRAIN", wb_mm_root / "train"))
    wb_mm_val = Path(getattr(v2_cfg, "DATA_SHARDS_WIDEBAND_MEMMAP_VAL", wb_mm_root / "val"))
    if prefer_memmap:
        if _list_memmap_shard_dirs(wb_mm_train) and _list_memmap_shard_dirs(wb_mm_val):
            return wb_mm_train, wb_mm_val, True
        if _list_memmap_shard_dirs(wb_mm_root):
            return wb_mm_root, wb_mm_root, True

    # 2) NPZ wideband split dirs (legacy fallback).
    wb_root = Path(getattr(v2_cfg, "DATA_SHARDS_WIDEBAND_DIR", "data_shards_ofdm_tr38901"))
    wb_train = Path(getattr(v2_cfg, "DATA_SHARDS_WIDEBAND_TRAIN", wb_root / "train"))
    wb_val = Path(getattr(v2_cfg, "DATA_SHARDS_WIDEBAND_VAL", wb_root / "val"))
    if _list_npz_files(wb_train) and _list_npz_files(wb_val):
        return wb_train, wb_val, True

    # 3) NPZ flat dir fallback
    if _list_npz_files(wb_root):
        return wb_root, wb_root, True

    # 4) If memmap not preferred, try memmap after NPZ.
    if (not prefer_memmap):
        if _list_memmap_shard_dirs(wb_mm_train) and _list_memmap_shard_dirs(wb_mm_val):
            return wb_mm_train, wb_mm_val, True
        if _list_memmap_shard_dirs(wb_mm_root):
            return wb_mm_root, wb_mm_root, True

    # 5) Explicitly fail if wideband is required
    if bool(getattr(v2_cfg, "REQUIRE_WIDEBAND_DATA", True)):
        raise FileNotFoundError(
            "Wideband shards were not found. "
            f"Checked mmap root={wb_mm_root} and npz root={wb_root}. "
            "Generate/convert OFDM shards first or disable REQUIRE_WIDEBAND_DATA for fallback."
        )

    # 6) Optional narrowband fallback (v1 format)
    if not bool(getattr(v2_cfg, "ALLOW_NARROWBAND_FALLBACK", False)):
        raise FileNotFoundError(
            "Wideband shards missing and ALLOW_NARROWBAND_FALLBACK is False."
        )

    nb_root = Path(getattr(v2_cfg, "DATA_SHARDS_DIR", "data_shards_M64_L16"))
    nb_train = Path(getattr(v2_cfg, "DATA_SHARDS_TRAIN", nb_root / "train"))
    nb_val = Path(getattr(v2_cfg, "DATA_SHARDS_VAL", nb_root / "val"))
    if _list_npz_files(nb_train) and _list_npz_files(nb_val):
        return nb_train, nb_val, False
    if _list_npz_files(nb_root):
        return nb_root, nb_root, False

    raise FileNotFoundError(
        "No shard dataset found for v2. "
        f"Checked mmap root={wb_mm_root}, npz root={wb_root}, and narrowband root={nb_root}."
    )


def fixed_subset(
    ds,
    n_cap: Optional[int],
    seed: int = 1337,
    mode: str = "random",
    start_idx: int = 0,
):
    """
    Deterministic subset helper.

    mode:
      - "random": seeded random subset (default).
      - "head": contiguous local slice [start_idx : start_idx + n_cap].
    """
    if n_cap is None or int(n_cap) >= len(ds):
        return ds
    n_cap = int(n_cap)
    if mode == "head":
        start = max(0, int(start_idx))
        start = min(start, max(0, len(ds) - n_cap))
        idx = np.arange(start, start + n_cap)
    else:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(ds))[:n_cap]
    return Subset(ds, idx.tolist())


def _to_tensor(x):
    if x is None:
        return None
    # Copy to writable ndarray: np.memmap slices are read-only and trigger
    # a PyTorch warning/undefined behavior if converted directly.
    arr = np.array(x, copy=True)
    return torch.from_numpy(arr)


def _pack_wideband_sample(
    shard_path: str,
    y,
    codes,
    ptr,
    K: int,
    snr: float,
    R,
    R_f,
    H,
    h_taps: Dict[str, Any],
):
    if y.ndim != 4:
        raise ValueError(
            f"Expected wideband sample y shape [L,F,M,2], got {y.shape} from {shard_path}"
        )

    has_h_feature = H is not None
    has_tap_feature = "H_taps_ri" in h_taps
    if bool(getattr(v2_cfg, "REQUIRE_WIDEBAND_DATA", True)) and (not has_h_feature) and (not has_tap_feature):
        raise ValueError(
            f"Wideband shard {shard_path} is missing both H and H_taps_ri; "
            "at least one channel representation is required."
        )
    if bool(getattr(v2_cfg, "REQUIRE_R_F_SUPERVISION", False)) and (R_f is None):
        raise ValueError(
            f"Wideband shard {shard_path} missing per-tone covariance key "
            f"({getattr(v2_cfg, 'WIDEBAND_R_F_KEY', 'R_f')})."
        )

    if H is None:
        # Keep dataloader collation stable: always provide tensor for H.
        # y is [L, F, M, 2] for wideband, so use [L, M, 2] zeros as fallback.
        l_dim, _, m_dim, _ = y.shape
        H = np.zeros((l_dim, m_dim, 2), dtype=np.float32)

    return {
        "y": _to_tensor(y),
        "H": _to_tensor(H),
        "codes": _to_tensor(codes),
        "ptr": _to_tensor(np.asarray(ptr, dtype=np.float32)),
        "K": torch.tensor(int(K), dtype=torch.long),
        "snr": torch.tensor(float(snr), dtype=torch.float32),
        "R": _to_tensor(R),
        # Empty tensor placeholder keeps default collate stable when R_f is absent.
        "R_f": _to_tensor(R_f) if R_f is not None else torch.zeros((0,), dtype=torch.float32),
        "H_taps": {k: _to_tensor(v) for k, v in h_taps.items()},
    }


def _cov_trace_real_scalar(R: Any) -> float:
    """
    Real trace helper for one covariance matrix in complex or RI format.
    """
    if R is None:
        return float("nan")

    if torch.is_tensor(R):
        if R.numel() == 0:
            return float("nan")
        if torch.is_complex(R):
            return float(torch.diagonal(R, dim1=-2, dim2=-1).real.sum().item())
        if R.dim() >= 3 and R.shape[-1] == 2:
            return float(torch.diagonal(R[..., 0], dim1=-2, dim2=-1).sum().item())
        return float(torch.diagonal(R, dim1=-2, dim2=-1).sum().item())

    arr = np.asarray(R)
    if arr.size == 0:
        return float("nan")
    if np.iscomplexobj(arr):
        return float(np.trace(arr, axis1=-2, axis2=-1).real.sum())
    if arr.ndim >= 3 and arr.shape[-1] == 2:
        return float(np.trace(arr[..., 0], axis1=-2, axis2=-1).sum())
    return float(np.trace(arr, axis1=-2, axis2=-1).sum())


def _run_cov_trace_sanity(ds, split_name: str):
    """
    Fail fast if too many shard samples have near-zero covariance trace.
    This catches stale/broken wideband shards before training starts.
    """
    if not bool(getattr(v2_cfg, "COV_TRACE_SANITY_ENABLE", True)):
        return
    n_scan_max = int(getattr(v2_cfg, "COV_TRACE_SANITY_MAX_SAMPLES", 64))
    n_scan = min(max(0, n_scan_max), len(ds))
    if n_scan <= 0:
        return

    min_trace = float(getattr(v2_cfg, "COV_TRACE_SANITY_MIN_TRACE", 1e-8))
    max_bad_ratio = float(getattr(v2_cfg, "COV_TRACE_SANITY_MAX_BAD_RATIO", 0.05))

    n_eval = 0
    n_bad = 0
    traces = []
    for i in range(n_scan):
        sample = ds[i]
        if not isinstance(sample, dict) or ("R" not in sample):
            continue
        n_eval += 1
        tr = _cov_trace_real_scalar(sample["R"])
        if np.isfinite(tr):
            traces.append(float(tr))
        if (not np.isfinite(tr)) or (float(tr) <= min_trace):
            n_bad += 1

    if n_eval <= 0:
        return

    bad_ratio = float(n_bad) / float(n_eval)
    if traces:
        print(
            f"[V2 DATA] trace-sanity split={split_name} scanned={n_eval} bad={n_bad} "
            f"ratio={bad_ratio:.3f} min={min(traces):.3e} median={float(np.median(traces)):.3e} "
            f"max={max(traces):.3e}",
            flush=True,
        )
    else:
        print(
            f"[V2 DATA] trace-sanity split={split_name} scanned={n_eval} bad={n_bad} ratio={bad_ratio:.3f}",
            flush=True,
        )

    if bad_ratio > max_bad_ratio:
        raise RuntimeError(
            "Covariance trace sanity check failed: "
            f"split={split_name}, bad_ratio={bad_ratio:.3f} > {max_bad_ratio:.3f}, "
            f"min_trace={min_trace:.1e}. "
            "Regenerate wideband shards with the patched generator."
        )


class V2WidebandNPZDataset(Dataset):
    """
    Wideband shard dataset for OFDM/TR38.901 pipeline.

    Expected fields per shard (minimum):
      - y: [S, L, F, M, 2]
      - codes: [S, L, N, 2]
      - ptr: [S, 3*K_MAX]
      - K: [S]
      - snr or snr_db: [S]
      - R: [S, N, N, 2]

    Optional channel features:
      - H: [S, L, M, 2] or [S, L, F, M, 2]
      - H_taps_ri (recommended): [S, P, M, N, 2]
      - plus optional per-path metadata arrays.
    """

    _OPTIONAL_TAP_KEYS = (
        "H_taps_ri",
        "path_mask",
        "taus_s",
        "alphas",
        "aod_az",
        "aod_el",
        "aoa_az",
        "aoa_el",
    )
    _OPTIONAL_RF_KEYS_DEFAULT = ("R_f_true", "R_f", "Rf")

    def __init__(self, npz_paths_or_dir, max_cached_shards: Optional[int] = None):
        self.paths: List[str] = []
        self.meta: List[Tuple[str, int]] = []  # (path, n_samples)
        self._npz_cache: Dict[str, Any] = {}
        self._cache_order: List[str] = []
        self._worker_pid = None
        cfg_cap = int(getattr(v2_cfg, "MAX_CACHED_SHARDS", 2))
        self.max_cached_shards = max(1, int(cfg_cap if max_cached_shards is None else max_cached_shards))

        def _add_file(p: Path):
            with np.load(p, mmap_mode="r") as z:
                if "K" in z.files:
                    n = int(z["K"].shape[0])
                elif "y" in z.files:
                    n = int(z["y"].shape[0])
                else:
                    raise ValueError(f"Shard missing both 'K' and 'y': {p}")
            self.paths.append(str(p))
            self.meta.append((str(p), n))

        def _add_path(x):
            p = Path(x)
            if p.is_dir():
                for f in sorted(p.glob("*.npz")):
                    _add_file(f)
            else:
                if p.suffix.lower() != ".npz":
                    raise ValueError(f"Not an .npz file: {p}")
                _add_file(p)

        if isinstance(npz_paths_or_dir, (list, tuple)):
            for item in npz_paths_or_dir:
                _add_path(item)
        else:
            _add_path(npz_paths_or_dir)

        if not self.meta:
            raise FileNotFoundError("No .npz wideband shards found.")

        self.index_map = []
        for shard_idx, (_, n_samples) in enumerate(self.meta):
            for local_idx in range(n_samples):
                self.index_map.append((shard_idx, local_idx))

    def __len__(self):
        return len(self.index_map)

    def _ensure_worker_cache(self):
        import os

        pid = os.getpid()
        if self._worker_pid != pid:
            self._npz_cache = {}
            self._cache_order = []
            self._worker_pid = pid

    def _get_shard(self, path: str):
        z = self._npz_cache.get(path)
        if z is not None:
            if path in self._cache_order:
                self._cache_order.remove(path)
            self._cache_order.append(path)
            return z

        if len(self._npz_cache) >= self.max_cached_shards and self._cache_order:
            old_path = self._cache_order.pop(0)
            old = self._npz_cache.pop(old_path, None)
            if old is not None and hasattr(old, "close"):
                try:
                    old.close()
                except Exception:
                    pass

        if z is None:
            # NOTE: mmap_mode is silently IGNORED for .npz files by numpy;
            # full arrays are materialized on first key access.
            z = np.load(path, allow_pickle=False)
            self._npz_cache[path] = z
            self._cache_order.append(path)
        return z

    def __getitem__(self, idx):
        self._ensure_worker_cache()
        shard_idx, local_idx = self.index_map[idx]
        shard_path, _ = self.meta[shard_idx]
        z = self._get_shard(shard_path)

        y_key = str(getattr(v2_cfg, "WIDEBAND_Y_KEY", "y"))
        y = z[y_key][local_idx]

        codes = z["codes"][local_idx]
        ptr = z["ptr"][local_idx]
        K = int(z["K"][local_idx])
        if "snr_db" in z.files:
            snr = float(z["snr_db"][local_idx])
        elif "snr" in z.files:
            snr = float(z["snr"][local_idx])
        else:
            snr = 0.0
        R = z["R"][local_idx]
        R_f = None
        rf_key_main = str(getattr(v2_cfg, "WIDEBAND_R_F_KEY", "R_f"))
        rf_keys = [rf_key_main]
        rf_keys.extend(list(getattr(v2_cfg, "WIDEBAND_R_F_ALT_KEYS", self._OPTIONAL_RF_KEYS_DEFAULT)))
        seen = set()
        for key in rf_keys:
            if key in seen:
                continue
            seen.add(key)
            if key in z.files:
                R_f = z[key][local_idx]
                break

        H = z["H"][local_idx] if "H" in z.files else None
        h_taps = {}
        for key in self._OPTIONAL_TAP_KEYS:
            if key in z.files:
                h_taps[key] = z[key][local_idx]

        return _pack_wideband_sample(
            shard_path=shard_path,
            y=y,
            codes=codes,
            ptr=ptr,
            K=K,
            snr=snr,
            R=R,
            R_f=R_f,
            H=H,
            h_taps=h_taps,
        )


class V2WidebandMemmapDataset(Dataset):
    """
    Wideband mmap shard dataset.

    Expected layout:
      root_or_split/
        shard_000/
          y.npy, H.npy, codes.npy, ptr.npy, K.npy, snr.npy, R.npy, R_f.npy, ...
        shard_001/
          ...
    """

    _OPTIONAL_TAP_KEYS = V2WidebandNPZDataset._OPTIONAL_TAP_KEYS
    _OPTIONAL_RF_KEYS_DEFAULT = V2WidebandNPZDataset._OPTIONAL_RF_KEYS_DEFAULT

    def __init__(self, shard_dirs_or_root, max_cached_shards: Optional[int] = None):
        self.shard_dirs: List[Path] = []
        self.meta: List[Tuple[str, int, set[str]]] = []  # (dir, n_samples, keys)
        self._shard_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_order: List[str] = []
        self._worker_pid = None

        cfg_cap = int(getattr(v2_cfg, "MAX_CACHED_SHARDS", 1))
        self.max_cached_shards = max(1, int(cfg_cap if max_cached_shards is None else max_cached_shards))

        def _add_dir(p: Path):
            if _is_memmap_shard_dir(p):
                self.shard_dirs.append(p)
                return
            if p.is_dir():
                for d in _list_memmap_shard_dirs(p):
                    self.shard_dirs.append(d)
                return
            raise ValueError(f"Expected memmap shard dir/root, got: {p}")

        if isinstance(shard_dirs_or_root, (list, tuple)):
            for item in shard_dirs_or_root:
                _add_dir(Path(item))
        else:
            _add_dir(Path(shard_dirs_or_root))

        if not self.shard_dirs:
            raise FileNotFoundError("No mmap wideband shard directories found.")

        for d in self.shard_dirs:
            y_file = d / "y.npy"
            if not y_file.exists():
                raise FileNotFoundError(f"Missing required file: {y_file}")
            y_arr = np.load(y_file, mmap_mode="r", allow_pickle=False)
            n_samples = int(y_arr.shape[0])
            keys = {p.stem for p in d.glob("*.npy")}
            self.meta.append((str(d), n_samples, keys))

        self.index_map = []
        for shard_idx, (_, n_samples, _) in enumerate(self.meta):
            for local_idx in range(n_samples):
                self.index_map.append((shard_idx, local_idx))

    def __len__(self):
        return len(self.index_map)

    def _ensure_worker_cache(self):
        import os

        pid = os.getpid()
        if self._worker_pid != pid:
            self._shard_cache = {}
            self._cache_order = []
            self._worker_pid = pid

    @staticmethod
    def _close_cache_entry(cache_entry: Dict[str, Any]):
        for arr in cache_entry.values():
            mm = getattr(arr, "_mmap", None)
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass

    def _get_arr(self, shard_dir: str, key: str):
        entry = self._shard_cache.get(shard_dir)
        if entry is None:
            if len(self._shard_cache) >= self.max_cached_shards and self._cache_order:
                old_dir = self._cache_order.pop(0)
                old_entry = self._shard_cache.pop(old_dir, None)
                if old_entry is not None:
                    self._close_cache_entry(old_entry)
            entry = {}
            self._shard_cache[shard_dir] = entry
        if shard_dir in self._cache_order:
            self._cache_order.remove(shard_dir)
        self._cache_order.append(shard_dir)

        arr = entry.get(key)
        if arr is None:
            file_path = Path(shard_dir) / f"{key}.npy"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing mmap field file: {file_path}")
            arr = np.load(file_path, mmap_mode="r", allow_pickle=False)
            entry[key] = arr
        return arr

    def __getitem__(self, idx):
        self._ensure_worker_cache()
        shard_idx, local_idx = self.index_map[idx]
        shard_dir, _, keys = self.meta[shard_idx]

        y_key = str(getattr(v2_cfg, "WIDEBAND_Y_KEY", "y"))
        if y_key not in keys and "y" in keys:
            y_key = "y"
        y = self._get_arr(shard_dir, y_key)[local_idx]

        codes = self._get_arr(shard_dir, "codes")[local_idx]
        ptr = self._get_arr(shard_dir, "ptr")[local_idx]
        K = int(self._get_arr(shard_dir, "K")[local_idx])

        if "snr_db" in keys:
            snr = float(self._get_arr(shard_dir, "snr_db")[local_idx])
        elif "snr" in keys:
            snr = float(self._get_arr(shard_dir, "snr")[local_idx])
        else:
            snr = 0.0

        R = self._get_arr(shard_dir, "R")[local_idx]
        H = self._get_arr(shard_dir, "H")[local_idx] if "H" in keys else None

        R_f = None
        rf_key_main = str(getattr(v2_cfg, "WIDEBAND_R_F_KEY", "R_f"))
        rf_keys = [rf_key_main]
        rf_keys.extend(list(getattr(v2_cfg, "WIDEBAND_R_F_ALT_KEYS", self._OPTIONAL_RF_KEYS_DEFAULT)))
        seen = set()
        for key in rf_keys:
            if key in seen:
                continue
            seen.add(key)
            if key in keys:
                R_f = self._get_arr(shard_dir, key)[local_idx]
                break

        h_taps: Dict[str, Any] = {}
        for key in self._OPTIONAL_TAP_KEYS:
            if key in keys:
                h_taps[key] = self._get_arr(shard_dir, key)[local_idx]

        return _pack_wideband_sample(
            shard_path=shard_dir,
            y=y,
            codes=codes,
            ptr=ptr,
            K=K,
            snr=snr,
            R=R,
            R_f=R_f,
            H=H,
            h_taps=h_taps,
        )


def build_dataloaders_v2(
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: int = 1337,
    shuffle_train: bool = True,
    train_subset_mode: str = "random",
    val_subset_mode: str = "random",
    max_cached_shards: Optional[int] = None,
):
    """Build train/val loaders for v2 (wideband-first)."""
    tr_dir, va_dir, is_wideband = resolve_shards_train_val()
    if is_wideband:
        has_memmap = bool(_list_memmap_shard_dirs(Path(tr_dir))) and bool(_list_memmap_shard_dirs(Path(va_dir)))
        if has_memmap:
            ds_tr_full = V2WidebandMemmapDataset(tr_dir, max_cached_shards=max_cached_shards)
            ds_va_full = V2WidebandMemmapDataset(va_dir, max_cached_shards=max_cached_shards)
        else:
            ds_tr_full = V2WidebandNPZDataset(tr_dir, max_cached_shards=max_cached_shards)
            ds_va_full = V2WidebandNPZDataset(va_dir, max_cached_shards=max_cached_shards)
    else:
        ds_tr_full = ShardNPZDataset(tr_dir)
        ds_va_full = ShardNPZDataset(va_dir)

    ds_tr = fixed_subset(ds_tr_full, n_train, seed=seed, mode=train_subset_mode)
    ds_va = fixed_subset(ds_va_full, n_val, seed=seed + 1, mode=val_subset_mode)

    _run_cov_trace_sanity(ds_tr, split_name="train")
    _run_cov_trace_sanity(ds_va, split_name="val")

    bs = int(batch_size if batch_size is not None else v2_mdl.BATCH_SIZE)
    num_workers = int(getattr(v2_cfg, "NUM_WORKERS", 0))
    pin_memory = bool(getattr(v2_cfg, "PIN_MEMORY", True))

    tr_loader = DataLoader(
        ds_tr,
        batch_size=bs,
        shuffle=shuffle_train,
        drop_last=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    va_loader = DataLoader(
        ds_va,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return tr_loader, va_loader
