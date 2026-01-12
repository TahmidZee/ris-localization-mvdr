#!/usr/bin/env python3
"""
MDL sanity check for K estimation on the CURRENT dataset.

Goal:
  Answer: "Is L (snapshots) too small for K estimation?"

This script computes classical MDL K-hat using the *offline* sample covariance `R_samp`
stored in shards, and reports accuracy overall + by SNR buckets.

Usage examples:
  python mdl_sanity_check.py --n_val 2000 --max_batches 50
  python mdl_sanity_check.py --n_val 5000 --batch_size 256 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ris_pytorch_pipeline.configs import cfg, set_seed
from ris_pytorch_pipeline.dataset import ShardNPZDataset
from ris_pytorch_pipeline.collate_fn import collate_pad_to_kmax_with_snr
from ris_pytorch_pipeline.covariance_utils import build_effective_cov_torch


def _confusion_update(cm: np.ndarray, y_true: torch.Tensor, y_hat: torch.Tensor) -> None:
    yt = y_true.detach().cpu().numpy().astype(np.int64)
    yh = y_hat.detach().cpu().numpy().astype(np.int64)
    for t, h in zip(yt, yh):
        if 1 <= t <= cm.shape[0] and 1 <= h <= cm.shape[1]:
            cm[t - 1, h - 1] += 1


def _mdl_scores_from_eigs(evals_desc: torch.Tensor, L_snap: int, k_max: int) -> torch.Tensor:
    """
    evals_desc: [B, N] real, descending
    returns: mdl scores [B, k_max] for k=1..k_max
    """
    B, N = evals_desc.shape
    T = int(L_snap)
    # Guard: log(0) and negative eigenvalues due to numerical noise
    evals_desc = torch.clamp(evals_desc, min=1e-12)

    mdl = torch.zeros(B, k_max, device=evals_desc.device, dtype=torch.float32)
    logT = torch.log(torch.tensor(float(max(T, 2)), device=evals_desc.device))
    for k in range(1, k_max + 1):
        k_idx = k - 1
        noise = evals_desc[:, k:]  # [B, N-k]
        gm = torch.exp(torch.mean(torch.log(noise), dim=-1))
        am = torch.mean(noise, dim=-1)
        Lk = (T * (N - k)) * torch.log(am / gm + 1e-12)
        mdl[:, k_idx] = Lk + 0.5 * k * (2 * N - k) * logT
    return mdl


def _snr_bucket(snr_db: float) -> str:
    if snr_db < 0.0:
        return "<0dB"
    if snr_db < 10.0:
        return "0-10dB"
    if snr_db < 20.0:
        return "10-20dB"
    return ">=20dB"


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_val", type=int, default=2000, help="Number of val samples to use (subset)")
    ap.add_argument("--max_batches", type=int, default=0, help="Max batches to process (0 = all in subset)")
    ap.add_argument("--batch_size", type=int, default=256, help="Batch size")
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--diag_load", type=int, default=1, help="1=apply diagonal loading in effective cov")
    ap.add_argument("--apply_shrink", type=int, default=1, help="1=apply SNR-aware shrinkage")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Resolve val shard dir
    val_dir = Path(getattr(cfg, "DATA_SHARDS_VAL", Path(getattr(cfg, "DATA_SHARDS_DIR")) / "val"))
    if not val_dir.exists():
        val_dir = Path(getattr(cfg, "DATA_SHARDS_DIR"))
    ds_full = ShardNPZDataset(val_dir)
    n = min(int(args.n_val), len(ds_full))
    idx = np.random.RandomState(1337).permutation(len(ds_full))[:n]
    ds = Subset(ds_full, idx.tolist())

    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        collate_fn=lambda batch: collate_pad_to_kmax_with_snr(batch, cfg.K_MAX),
    )

    total = 0
    correct = 0
    cm = np.zeros((cfg.K_MAX, cfg.K_MAX), dtype=np.int64)
    k_true_hist = np.zeros(cfg.K_MAX, dtype=np.int64)

    buckets: Dict[str, Tuple[int, int]] = {
        "<0dB": (0, 0),
        "0-10dB": (0, 0),
        "10-20dB": (0, 0),
        ">=20dB": (0, 0),
    }

    max_batches = None if int(args.max_batches) <= 0 else int(args.max_batches)
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        if "R_samp" not in batch or batch["R_samp"] is None:
            raise RuntimeError("Dataset batch has no 'R_samp'. Regenerate shards with R_samp enabled.")
        if "snr_db" not in batch:
            raise RuntimeError("Dataset batch has no 'snr_db'.")

        R_samp_ri = batch["R_samp"].to(device=device, dtype=torch.float32)  # [B,N,N,2]
        snr_db = batch["snr_db"].to(device=device, dtype=torch.float32).view(-1)
        K_true = batch["K"].to(device=device).long().view(-1)

        # K distribution
        for k in range(1, cfg.K_MAX + 1):
            k_true_hist[k - 1] += int((K_true == k).sum().item())

        R_samp_c = torch.complex(R_samp_ri[..., 0], R_samp_ri[..., 1]).to(torch.complex64)
        R_eff = build_effective_cov_torch(
            R_samp_c,
            snr_db=snr_db if bool(int(args.apply_shrink)) else None,
            R_samp=None,
            beta=None,
            diag_load=bool(int(args.diag_load)),
            apply_shrink=bool(int(args.apply_shrink)),
            target_trace=float(cfg.N),
        )

        evals = torch.linalg.eigvalsh(R_eff).real  # ascending
        evals = torch.flip(evals, dims=[-1])       # descending

        mdl = _mdl_scores_from_eigs(evals, L_snap=int(cfg.L), k_max=int(cfg.K_MAX))
        K_hat = torch.argmin(mdl, dim=-1).long() + 1  # [B] in 1..K_MAX

        total += int(K_true.numel())
        correct += int((K_hat == K_true).sum().item())
        _confusion_update(cm, K_true, K_hat)

        # Buckets
        snr_cpu = snr_db.detach().cpu().numpy()
        Kt_cpu = K_true.detach().cpu().numpy()
        Kh_cpu = K_hat.detach().cpu().numpy()
        for s, kt, kh in zip(snr_cpu, Kt_cpu, Kh_cpu):
            b = _snr_bucket(float(s))
            c, t = buckets[b]
            buckets[b] = (c + int(kh == kt), t + 1)

    acc = correct / max(1, total)
    print("\n=== MDL sanity check (on offline R_samp) ===")
    print(f"  Dataset: {val_dir}")
    print(f"  N={cfg.N}  L={cfg.L}  K_MAX={cfg.K_MAX}")
    print(f"  Samples={total}  Accuracy={acc:.3f}")

    print("\n  K_true distribution:")
    for k in range(1, cfg.K_MAX + 1):
        frac = k_true_hist[k - 1] / max(1, total)
        print(f"    K={k}: {k_true_hist[k - 1]} ({frac:.1%})")

    print("\n  Accuracy by SNR bucket:")
    for name in ["<0dB", "0-10dB", "10-20dB", ">=20dB"]:
        c, t = buckets[name]
        a = c / max(1, t)
        print(f"    {name:>6}: acc={a:.3f}  (N={t})")

    print("\n  Confusion (rows=true K, cols=MDL K_hat):")
    with np.printoptions(linewidth=140):
        print(cm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


