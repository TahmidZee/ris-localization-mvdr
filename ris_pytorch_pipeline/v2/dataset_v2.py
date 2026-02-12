"""
V2 dataset utilities.

Phase 1 reuses the existing shard format through `ShardNPZDataset`.
Phase 2+ can extend this module for OFDM-wideband shards.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset

from ..dataset import ShardNPZDataset
from .config_v2 import v2_cfg, v2_mdl


def resolve_shards_train_val() -> Tuple[Path, Path]:
    """
    Resolve train/val shard directories.

    Supports:
      - split dirs: <root>/train and <root>/val
      - flat dir: <root>/*.npz (same dir for both, split via subset caps)
    """
    root = Path(getattr(v2_cfg, "DATA_SHARDS_DIR", "data_shards_M64_L16"))
    train_dir = Path(getattr(v2_cfg, "DATA_SHARDS_TRAIN", root / "train"))
    val_dir = Path(getattr(v2_cfg, "DATA_SHARDS_VAL", root / "val"))

    if train_dir.exists() and list(train_dir.glob("*.npz")):
        if val_dir.exists() and list(val_dir.glob("*.npz")):
            return train_dir, val_dir
        return train_dir, train_dir

    if root.exists() and list(root.glob("*.npz")):
        return root, root

    raise FileNotFoundError(
        f"No shards found under {root}. "
        f"Expected split dirs ({train_dir}, {val_dir}) or flat shards in {root}."
    )


def fixed_subset(ds, n_cap: Optional[int], seed: int = 1337):
    """Deterministic subset used for overfit diagnostics."""
    if n_cap is None or int(n_cap) >= len(ds):
        return ds
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(ds))[: int(n_cap)]
    return Subset(ds, idx.tolist())


def build_dataloaders_v2(
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: int = 1337,
    shuffle_train: bool = True,
):
    """Build train/val loaders for v2."""
    tr_dir, va_dir = resolve_shards_train_val()
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
