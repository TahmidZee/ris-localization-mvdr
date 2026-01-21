# SPDX-License-Identifier: MIT
import time
import numpy as np
import torch
from pathlib import Path
from .dataset import ShardNPZDataset
from .infer import load_model, hybrid_estimate_final, estimate_k_blind
from .baseline import (
    ramezani_mod_music_wrapper,
    decoupled_mod_music,
    dcd_music_wrapper,
    nf_subspacenet_wrapper,
    incident_cov_from_snaps,
)
from .configs import cfg

TEST_DIR = Path(getattr(cfg, "DATA_SHARDS_TEST", "results_final/data/shards/test"))
BENCH_DIR = Path(cfg.RESULTS_DIR) / "benches"
BENCH_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Matching (Hungarian, angle-wrap safe) ----------------

def _match_err(gt, pr, r_scale=1.0, w_phi=1.0, w_theta=1.0, w_r=0.0):
    """
    Optimal 1-1 pairing via Hungarian on a weighted cost. Angles wrap to [-pi,pi].
    Returns RMSE for (phi, theta, r) across matched pairs.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    def ang_wrap_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    phi_t, theta_t, r_t = [np.asarray(x, dtype=np.float64) for x in gt]
    phi_p, theta_p, r_p = [np.asarray(x, dtype=np.float64) for x in pr]
    Kt, Kp = len(phi_t), len(phi_p)
    K = min(Kt, Kp)
    if K == 0:
        return float("nan"), float("nan"), float("nan")

    dphi = ang_wrap_diff(phi_t[:, None],   phi_p[None, :])
    dtht = ang_wrap_diff(theta_t[:, None], theta_p[None, :])
    if (r_t.size > 0) and (r_p.size > 0):
        dr = (r_t[:, None] - r_p[None, :]) / max(1e-9, r_scale)
    else:
        dr = np.zeros_like(dphi)

    C = w_phi * np.abs(dphi) + w_theta * np.abs(dtht) + w_r * np.abs(dr)

    if Kt <= Kp:
        row_ind, col_ind = linear_sum_assignment(C)
        row_ind, col_ind = row_ind[:K], col_ind[:K]
        t_idx, p_idx = row_ind, col_ind
    else:
        col_ind, row_ind = linear_sum_assignment(C.T)
        row_ind, col_ind = row_ind[:K], col_ind[:K]
        t_idx, p_idx = row_ind, col_ind

    e_phi = ang_wrap_diff(phi_t[t_idx],   phi_p[p_idx])
    e_tht = ang_wrap_diff(theta_t[t_idx], theta_p[p_idx])
    if (r_t.size > 0) and (r_p.size > 0):
        e_rng = (r_t[t_idx] - r_p[p_idx])
    else:
        e_rng = np.array([], dtype=np.float64)

    dphi_rmse   = float(np.sqrt(np.mean(e_phi**2)))   if e_phi.size else float("nan")
    dtheta_rmse = float(np.sqrt(np.mean(e_tht**2)))   if e_tht.size else float("nan")
    dr_rmse     = float(np.sqrt(np.mean(e_rng**2)))   if e_rng.size else float("nan")
    return dphi_rmse, dtheta_rmse, dr_rmse


# ---------------- GT extraction (works with ptr-decoding in ShardNPZDataset) ----------------

def _gt_from_item(item):
    if "phi" in item and "theta" in item and "r" in item:
        return (np.asarray(item["phi"]).tolist(),
                np.asarray(item["theta"]).tolist(),
                np.asarray(item["r"]).tolist())
    # fallback (shouldn’t happen with current dataset.py)
    if "ptr" in item and "K" in item:
        a = np.asarray(item["ptr"], dtype=np.float32).reshape(-1)
        K = int(item["K"]); KMAX = int(cfg.K_MAX)
        if a.size >= 3*KMAX:
            phi = a[0:KMAX][:K]
            tht = a[KMAX:2*KMAX][:K]
            rr  = a[2*KMAX:3*KMAX][:K]
            return (phi.tolist(), tht.tolist(), rr.tolist())
    return ([], [], [])


# ---------------- Estimators ----------------

def _estimate_hybrid(model, sample, blind_k=True):
    s = {
        "y": sample["y"],         # [L,M,2] torch
        "H": sample["H"],         # [M,N,2] torch
        "H_full": sample.get("H_full", None),  # [M,N,2] torch (optional; enables R_samp blending)
        "codes": sample["codes"], # [L,N,2] torch
        "K": int(sample["K"]),
        "snr_db": float(sample["snr"]),
    }
    force_K = None if blind_k else int(sample["K"])
    t0 = time.time()
    ph, tht, rr = hybrid_estimate_final(model, s, force_K=force_K, k_policy="mdl", do_newton=True)
    ms = (time.time() - t0) * 1000.0
    return (ph, tht, rr), ms


def _estimate_with_baseline(kind, sample, blind_k=True):
    y_ri = sample["y"].numpy()
    C_ri = sample["codes"].numpy()
    H_full_ri = sample.get("H_full", None)
    if H_full_ri is None:
        raise RuntimeError("Baseline evaluation requires H_full in shards (field 'H_full' missing).")
    H_full_ri = H_full_ri.numpy()
    R_inc = incident_cov_from_snaps(y_ri, H_full_ri, C_ri)
    K_hat = estimate_k_blind(R_inc, T=int(y_ri.shape[0]), kmax=int(getattr(cfg, "K_MAX", 5))) if blind_k else int(sample["K"])

    t0 = time.time()
    if kind == "ramezani":
        phi, tht, rr = ramezani_mod_music_wrapper(R_inc, K_hat)
    elif kind == "decoupled_mod":
        phi, tht, rr = decoupled_mod_music(R_inc, K_hat)
    elif kind == "dcd":
        phi, tht, rr = dcd_music_wrapper(R_inc, K_hat)
    elif kind == "nfssn":
        phi, tht, rr = nf_subspacenet_wrapper(R_inc, K_hat)
    else:
        raise ValueError(f"Unknown baseline '{kind}'")
    ms = (time.time() - t0) * 1000.0
    return (phi, tht, rr), ms


# ---------------- Quick bench (kept for compatibility) ----------------

def run_bench_csv(model, n=300, oracle=False, outf="bench_blind.csv"):
    """
    Writes pairs [gt_phi0, pred_phi0] for quick sanity scatter.
    Now reads test shards via ShardNPZDataset (fair).
    """
    import csv
    dset = ShardNPZDataset(TEST_DIR)
    rows = []
    N = min(n, len(dset))
    for i in range(N):
        it = dset[i]
        s = {"y": it["y"], "H": it["H"], "H_full": it.get("H_full", None), "codes": it["codes"], "K": int(it["K"]), "snr_db": float(it["snr"])}
        pr = hybrid_estimate_final(model, s, force_K=(int(it["K"]) if oracle else None), k_policy="mdl", do_newton=True)
        gt_phi = _gt_from_item(it)[0]
        gt_phi0 = float(gt_phi[0]) if len(gt_phi) > 0 else 0.0
        pred_phi0 = float(pr[0][0]) if pr and len(pr[0]) > 0 else 0.0
        rows.append([gt_phi0, pred_phi0])
    with open(outf, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return outf


def load_model_for_bench():
    """Convenience loader that uses CUDA when available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_model(map_location="cpu", device=device)
