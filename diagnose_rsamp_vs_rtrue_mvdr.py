"""
Diagnose whether snapshot-derived covariance (R_samp) is MVDR-usable.

Why this exists:
- We already verified MVDR works well on R_true in multiple experiments.
- Our current shards typically do NOT store R_samp (too expensive at L=64).
- But we *can* compute R_samp on-the-fly from (y, codes, H_full) for a small subset
  to decide whether a "denoise/correct R_samp" approach (Option B) is viable.

This script:
  - Loads test shards via ShardNPZDataset (requires H_full in shards)
  - For each sample, builds:
      R_true  (from shard field R)
      R_samp  (from snapshots using angle_pipeline.build_sample_covariance_from_snapshots)
  - Runs MVDR detection on each covariance and reports RMSPE / angle / range errors.

Usage (example):
  cd ris/MainMusic
  python diagnose_rsamp_vs_rtrue_mvdr.py --limit 200 --device cuda
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

from ris_pytorch_pipeline.configs import cfg
from ris_pytorch_pipeline.dataset import ShardNPZDataset
from ris_pytorch_pipeline.angle_pipeline import build_sample_covariance_from_snapshots
from ris_pytorch_pipeline.music_gpu import mvdr_detect_sources, get_gpu_estimator


def _ri_to_c_np(x_ri: np.ndarray) -> np.ndarray:
    x_ri = np.asarray(x_ri)
    if np.iscomplexobj(x_ri):
        return x_ri.astype(np.complex64, copy=False)
    if x_ri.shape[-1] == 2:
        return (x_ri[..., 0] + 1j * x_ri[..., 1]).astype(np.complex64, copy=False)
    return x_ri.astype(np.complex64, copy=False)


def _cartesian_from_spherical(phi, theta, r):
    phi, theta, r = np.asarray(phi), np.asarray(theta), np.asarray(r)
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z


def _match_err_rmspe(gt, pr):
    """
    Hungarian matching on Cartesian distance; returns (phi_rmse, theta_rmse, r_rmse, rmspe).
    gt/pr are tuples of lists: (phi_list_rad, theta_list_rad, r_list_m).
    """
    from scipy.optimize import linear_sum_assignment

    phi_t, theta_t, r_t = [np.asarray(x, dtype=np.float64) for x in gt]
    phi_p, theta_p, r_p = [np.asarray(x, dtype=np.float64) for x in pr]
    Kt, Kp = int(phi_t.size), int(phi_p.size)
    K = min(Kt, Kp)
    if K == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    gt_x, gt_y, gt_z = _cartesian_from_spherical(phi_t, theta_t, r_t)
    pr_x, pr_y, pr_z = _cartesian_from_spherical(phi_p, theta_p, r_p)
    dx = gt_x[:, None] - pr_x[None, :]
    dy = gt_y[:, None] - pr_y[None, :]
    dz = gt_z[:, None] - pr_z[None, :]
    C = np.sqrt(dx * dx + dy * dy + dz * dz)

    if Kt <= Kp:
        row_ind, col_ind = linear_sum_assignment(C)
        row_ind, col_ind = row_ind[:K], col_ind[:K]
        t_idx, p_idx = row_ind, col_ind
    else:
        col_ind, row_ind = linear_sum_assignment(C.T)
        row_ind, col_ind = row_ind[:K], col_ind[:K]
        t_idx, p_idx = row_ind, col_ind

    # wrap-safe angular diffs
    def ang_wrap_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    e_phi = ang_wrap_diff(phi_t[t_idx], phi_p[p_idx])
    e_tht = ang_wrap_diff(theta_t[t_idx], theta_p[p_idx])
    e_rng = (r_t[t_idx] - r_p[p_idx])

    phi_rmse = float(np.sqrt(np.mean(e_phi ** 2))) if e_phi.size else float("nan")
    tht_rmse = float(np.sqrt(np.mean(e_tht ** 2))) if e_tht.size else float("nan")
    rng_rmse = float(np.sqrt(np.mean(e_rng ** 2))) if e_rng.size else float("nan")

    # RMSPE on matched pairs
    gx, gy, gz = _cartesian_from_spherical(phi_t[t_idx], theta_t[t_idx], r_t[t_idx])
    px, py, pz = _cartesian_from_spherical(phi_p[p_idx], theta_p[p_idx], r_p[p_idx])
    pe = np.sqrt((gx - px) ** 2 + (gy - py) ** 2 + (gz - pz) ** 2)
    rmspe = float(np.sqrt(np.mean(pe ** 2))) if pe.size else float("nan")
    return phi_rmse, tht_rmse, rng_rmse, rmspe


def _gt_from_item(item):
    # ShardNPZDataset already decodes phi/theta/r (preferred)
    if "phi" in item and "theta" in item and "r" in item:
        return (np.asarray(item["phi"]).tolist(), np.asarray(item["theta"]).tolist(), np.asarray(item["r"]).tolist())
    # fallback: ptr
    a = np.asarray(item["ptr"], dtype=np.float32).reshape(-1)
    K = int(item["K"]); KMAX = int(cfg.K_MAX)
    phi = a[0:KMAX][:K]
    tht = a[KMAX:2 * KMAX][:K]
    rr = a[2 * KMAX:3 * KMAX][:K]
    return (phi.tolist(), tht.tolist(), rr.tolist())


def _mvdr_estimate_from_cov(R: np.ndarray, *, device: str, K_force: int | None):
    est = get_gpu_estimator(cfg, device=device)
    sources, _spec = mvdr_detect_sources(
        R,
        cfg,
        device=device,
        grid_phi=int(getattr(cfg, "MVDR_GRID_PHI", 361)),
        grid_theta=int(getattr(cfg, "MVDR_GRID_THETA", 181)),
        r_planes=getattr(cfg, "REFINER_R_PLANES", None) or est.default_r_planes_mvdr,
        delta_scale=float(getattr(cfg, "MVDR_DELTA_SCALE", 1e-2)),
        threshold_db=float(getattr(cfg, "MVDR_THRESH_DB", 12.0)),
        threshold_mode=str(getattr(cfg, "MVDR_THRESH_MODE", "mad")),
        cfar_z=float(getattr(cfg, "MVDR_CFAR_Z", 5.0)),
        max_sources=int(getattr(cfg, "K_MAX", 5)),
        force_k=(int(K_force) if (K_force is not None) else None),
        do_refinement=bool(getattr(cfg, "MVDR_DO_REFINEMENT", True)),
    )
    if len(sources) == 0:
        return ([], [], [])
    phi_deg = np.array([s[0] for s in sources], dtype=np.float32)
    tht_deg = np.array([s[1] for s in sources], dtype=np.float32)
    r_m = np.array([s[2] for s in sources], dtype=np.float32)
    return (np.deg2rad(phi_deg).tolist(), np.deg2rad(tht_deg).tolist(), r_m.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--oracle-k", action="store_true", help="Force K=K_gt during MVDR detection (top-K, no thresholding).")
    args = ap.parse_args()

    test_dir = str(getattr(cfg, "DATA_SHARDS_TEST", "results_final/data/shards/test"))
    dset = ShardNPZDataset(test_dir)
    N = min(int(args.limit), len(dset))

    rows = []
    for i in range(N):
        it = dset[i]
        gt = _gt_from_item(it)
        K_force = int(it["K"]) if bool(args.oracle_k) else None

        # R_true from shards
        R_true = _ri_to_c_np(it["R"])
        # R_samp from snapshots (requires H_full)
        H_full = it.get("H_full", None)
        if H_full is None:
            raise RuntimeError("This diagnostic requires H_full in shards to build R_samp on-the-fly.")
        y = _ri_to_c_np(it["y"])
        codes = _ri_to_c_np(it["codes"])
        H_full = _ri_to_c_np(H_full)
        R_samp = build_sample_covariance_from_snapshots(y, H_full, codes, cfg)

        pr_true = _mvdr_estimate_from_cov(R_true, device=str(args.device), K_force=K_force)
        pr_samp = _mvdr_estimate_from_cov(R_samp, device=str(args.device), K_force=K_force)

        e_true = _match_err_rmspe(gt, pr_true)
        e_samp = _match_err_rmspe(gt, pr_samp)

        rows.append(("R_true", float(it["snr"]), int(it["K"]), e_true[0], e_true[1], e_true[2], e_true[3]))
        rows.append(("R_samp", float(it["snr"]), int(it["K"]), e_samp[0], e_samp[1], e_samp[2], e_samp[3]))

    # Summaries
    for tag in ("R_true", "R_samp"):
        rs = np.array([r[-1] for r in rows if r[0] == tag], dtype=np.float64)
        ph = np.array([r[3] for r in rows if r[0] == tag], dtype=np.float64)
        th = np.array([r[4] for r in rows if r[0] == tag], dtype=np.float64)
        rg = np.array([r[5] for r in rows if r[0] == tag], dtype=np.float64)
        print(f"\n=== MVDR on {tag} (N={N}) ===")
        print(f"median rmspe(m)= {np.nanmedian(rs):.6f} mean= {np.nanmean(rs):.6f}")
        print(f"median rng(m)= {np.nanmedian(rg):.6f} mean= {np.nanmean(rg):.6f}")
        print(f"median phi(rad)= {np.nanmedian(ph):.6f}  -> deg≈ {np.rad2deg(np.nanmedian(ph)):.2f}")
        print(f"median theta(rad)= {np.nanmedian(th):.6f} -> deg≈ {np.rad2deg(np.nanmedian(th)):.2f}")


if __name__ == "__main__":
    main()

