#!/usr/bin/env python3
"""
Correlation check between surrogate metrics and full MUSIC metrics.

Usage:
    python correlation_check.py --checkpoints ckpt1.pt ckpt2.pt ... --n_val 2000 --max_batches 20

For each checkpoint:
  1) Runs surrogate validation (no MUSIC): K_acc, aux φ/θ/r RMSE, surrogate score
  2) Runs full MUSIC metrics on the same val subset: K_acc_MUSIC, φ/θ/r RMSE, succ_rate, MDL_acc
  3) Reports a table and overall correlations across checkpoints.

Note: This is intended for a small subset (e.g., 1–2k samples) to keep runtime reasonable.
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from ris_pytorch_pipeline.configs import cfg, mdl_cfg, set_seed
from ris_pytorch_pipeline.train import Trainer


def run_for_checkpoint(ckpt_path: Path, n_val: int, max_batches: int):
    # Build trainer (surrogate mode)
    cfg.VAL_PRIMARY = "surrogate"
    cfg.USE_MUSIC_METRICS_IN_VAL = False
    set_seed(int(getattr(mdl_cfg, "SEED", 42)))
    t = Trainer(from_hpo=False)

    # Load checkpoint state dict (weights-only)
    state_dict = torch.load(ckpt_path, map_location=t.device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    t.model.load_state_dict(state_dict, strict=False)
    t.model.eval()

    # Build val loader (GPU cache) with requested subset
    _, va_loader = t._build_loaders_gpu_cache(n_train=1000, n_val=n_val)

    # 1) Surrogate metrics
    surrogate = t._validate_surrogate_epoch(va_loader, max_batches=max_batches)

    # 2) Full MUSIC metrics on same subset
    cfg.USE_MUSIC_METRICS_IN_VAL = True
    music = t._eval_hungarian_metrics(va_loader, max_batches=max_batches)

    return surrogate, music


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint paths (state_dict .pt files)")
    ap.add_argument("--n_val", type=int, default=2000, help="Number of val samples to use (subset)")
    ap.add_argument("--max_batches", type=int, default=20, help="Max val batches to evaluate (keep small)")
    ap.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    args = ap.parse_args()

    set_seed(args.seed)

    rows = []
    for ckpt in args.checkpoints:
        ckpt_path = Path(ckpt)
        print(f"\n=== Evaluating checkpoint: {ckpt_path} ===")
        surrogate, music = run_for_checkpoint(ckpt_path, args.n_val, args.max_batches)
        row = {
            "ckpt": ckpt_path.name,
            "sur_score": float(surrogate.get("score", 0.0)),
            "sur_k_acc": float(surrogate.get("k_acc", 0.0)),
            "sur_phi_rmse": float(surrogate.get("aux_phi_rmse", 0.0)),
            "sur_theta_rmse": float(surrogate.get("aux_theta_rmse", 0.0)),
            "sur_r_rmse": float(surrogate.get("aux_r_rmse", 0.0)),
            "music_k_acc": float(music.get("k_acc", 0.0)),
            "music_k_mdl": float(music.get("k_mdl_acc", 0.0)),
            "music_phi_rmse": float(music.get("rmse_phi_mean", 0.0)),
            "music_theta_rmse": float(music.get("rmse_theta_mean", 0.0)),
            "music_r_rmse": float(music.get("rmse_r_mean", 0.0)),
            "music_succ": float(music.get("success_rate", 0.0)),
        }
        rows.append(row)
        print(f"  Surrogate: score={row['sur_score']:.4f}, k_acc={row['sur_k_acc']:.3f}, "
              f"aux φ/θ/r RMSE=({row['sur_phi_rmse']:.2f}, {row['sur_theta_rmse']:.2f}, {row['sur_r_rmse']:.2f})")
        print(f"  MUSIC:     k_acc={row['music_k_acc']:.3f}, k_mdl={row['music_k_mdl']:.3f}, "
              f"φ/θ/r RMSE=({row['music_phi_rmse']:.2f}, {row['music_theta_rmse']:.2f}, {row['music_r_rmse']:.2f}), "
              f"succ={row['music_succ']:.3f}")

    # Compute correlations across checkpoints
    def corr(x, y):
        if len(x) < 2:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    sur_scores = [r["sur_score"] for r in rows]
    music_kacc = [r["music_k_acc"] for r in rows]
    music_phi = [r["music_phi_rmse"] for r in rows]
    music_theta = [r["music_theta_rmse"] for r in rows]
    music_r = [r["music_r_rmse"] for r in rows]

    print("\n=== Correlations (across checkpoints) ===")
    print(f"  corr(sur_score, MUSIC k_acc)      = {corr(sur_scores, music_kacc):.3f}")
    print(f"  corr(sur_score, MUSIC φ_RMSE)     = {corr(sur_scores, music_phi):.3f}")
    print(f"  corr(sur_score, MUSIC θ_RMSE)     = {corr(sur_scores, music_theta):.3f}")
    print(f"  corr(sur_score, MUSIC r_RMSE)     = {corr(sur_scores, music_r):.3f}")


if __name__ == "__main__":
    main()



