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


def resolve_shards_train_val() -> Tuple[Path, Path, bool]:
    """
    Resolve train/val shard locations.

    Returns:
      train_path, val_path, is_wideband
    """
    # 1) Preferred: wideband split dirs
    wb_root = Path(getattr(v2_cfg, "DATA_SHARDS_WIDEBAND_DIR", "data_shards_ofdm_tr38901"))
    wb_train = Path(getattr(v2_cfg, "DATA_SHARDS_WIDEBAND_TRAIN", wb_root / "train"))
    wb_val = Path(getattr(v2_cfg, "DATA_SHARDS_WIDEBAND_VAL", wb_root / "val"))
    if _list_npz_files(wb_train) and _list_npz_files(wb_val):
        return wb_train, wb_val, True

    # 2) Wideband flat dir fallback
    if _list_npz_files(wb_root):
        return wb_root, wb_root, True

    # 3) Explicitly fail if wideband is required
    if bool(getattr(v2_cfg, "REQUIRE_WIDEBAND_DATA", True)):
        raise FileNotFoundError(
            "Wideband shards were not found. "
            f"Expected under {wb_root} (or train/val subdirs). "
            "Generate OFDM shards first or disable REQUIRE_WIDEBAND_DATA for fallback."
        )

    # 4) Optional narrowband fallback (v1 format)
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
        f"Checked wideband root={wb_root} and narrowband root={nb_root}."
    )


def fixed_subset(ds, n_cap: Optional[int], seed: int = 1337):
    """Deterministic subset used for overfit diagnostics."""
    if n_cap is None or int(n_cap) >= len(ds):
        return ds
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(ds))[: int(n_cap)]
    return Subset(ds, idx.tolist())


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

    def __init__(self, npz_paths_or_dir):
        self.paths: List[str] = []
        self.meta: List[Tuple[str, int]] = []  # (path, n_samples)
        self._npz_cache: Dict[str, Any] = {}
        self._worker_pid = None

        def _add_file(p: Path):
            with np.load(p, mmap_mode="r") as z:
                if "y" not in z.files:
                    raise ValueError(f"Shard missing required key 'y': {p}")
                n = int(z["y"].shape[0])
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
            self._worker_pid = pid

    def _get_shard(self, path: str):
        z = self._npz_cache.get(path)
        if z is None:
            z = np.load(path, allow_pickle=False, mmap_mode="r")
            self._npz_cache[path] = z
        return z

    @staticmethod
    def _to_tensor(x):
        if x is None:
            return None
        return torch.from_numpy(x)

    def __getitem__(self, idx):
        self._ensure_worker_cache()
        shard_idx, local_idx = self.index_map[idx]
        shard_path, _ = self.meta[shard_idx]
        z = self._get_shard(shard_path)

        y_key = str(getattr(v2_cfg, "WIDEBAND_Y_KEY", "y"))
        y = z[y_key][local_idx]
        if y.ndim != 4:
            raise ValueError(
                f"Expected wideband sample y shape [L,F,M,2], got {y.shape} from {shard_path}"
            )

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
                h_taps[key] = self._to_tensor(z[key][local_idx])

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

        sample = {
            "y": self._to_tensor(y),
            "H": self._to_tensor(H),
            "codes": self._to_tensor(codes),
            "ptr": self._to_tensor(ptr.astype(np.float32)),
            "K": torch.tensor(K, dtype=torch.long),
            "snr": torch.tensor(snr, dtype=torch.float32),
            "R": self._to_tensor(R),
            # Empty tensor placeholder keeps default collate stable when R_f is absent.
            "R_f": self._to_tensor(R_f) if R_f is not None else torch.zeros((0,), dtype=torch.float32),
            "H_taps": h_taps,
        }
        return sample


def build_dataloaders_v2(
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: int = 1337,
    shuffle_train: bool = True,
):
    """Build train/val loaders for v2 (wideband-first)."""
    tr_dir, va_dir, is_wideband = resolve_shards_train_val()
    if is_wideband:
        ds_tr_full = V2WidebandNPZDataset(tr_dir)
        ds_va_full = V2WidebandNPZDataset(va_dir)
    else:
        ds_tr_full = ShardNPZDataset(tr_dir)
        ds_va_full = ShardNPZDataset(va_dir)

    ds_tr = fixed_subset(ds_tr_full, n_train, seed=seed)
    ds_va = fixed_subset(ds_va_full, n_val, seed=seed + 1)

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
