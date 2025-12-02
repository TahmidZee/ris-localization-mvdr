# SPDX-License-Identifier: MIT
import time, json
import numpy as np
import pandas as pd
from pathlib import Path

from .configs import cfg
from .dataset import ShardNPZDataset
from .infer import load_model, hybrid_estimate_final, estimate_k_ic_from_cov
from .baseline import (
    ramezani_mod_music_wrapper,
    decoupled_mod_music,
    dcd_music_wrapper,
    nf_subspacenet_wrapper,
    incident_cov_from_snaps,
)

BENCH_DIR = Path("results_final/benches")
BENCH_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR = Path("results_final/data/shards/test")


# --------------- Matching & GT (same as benchmark.py) ---------------

def _cartesian_from_spherical(phi, theta, r):
    """Convert spherical (φ, θ, r) to Cartesian (x, y, z)"""
    import numpy as np
    phi, theta, r = np.asarray(phi), np.asarray(theta), np.asarray(r)
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi) 
    z = r * np.sin(theta)
    return x, y, z

def _match_err(gt, pr, r_scale=1.0, w_phi=1.0, w_theta=1.0, w_r=0.0, return_rmspe=True):
    """
    Optimal 1-1 pairing via Hungarian on weighted cost. Returns both:
    (i) Separate RMSE(φ,θ,r) for detailed analysis
    (ii) RMSPE (3D position error) for paper comparison
    """
    import numpy as np
    import math
    from scipy.optimize import linear_sum_assignment
    
    def ang_wrap_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))
    
    phi_t, theta_t, r_t = [np.asarray(x, dtype=np.float64) for x in gt]
    phi_p, theta_p, r_p = [np.asarray(x, dtype=np.float64) for x in pr]
    Kt, Kp = len(phi_t), len(phi_p)
    K = min(Kt, Kp)
    
    if K == 0:
        if return_rmspe:
            return float("nan"), float("nan"), float("nan"), float("nan")  # φ, θ, r, RMSPE
        return float("nan"), float("nan"), float("nan")
    
    # Hungarian matching
    dphi = ang_wrap_diff(phi_t[:, None], phi_p[None, :])
    dtht = ang_wrap_diff(theta_t[:, None], theta_p[None, :])
    if (r_t.size > 0) and (r_p.size > 0):
        dr = (r_t[:, None] - r_p[None, :]) / max(1e-9, r_scale)
    else:
        dr = np.zeros_like(dphi)
    
    C = w_phi*np.abs(dphi) + w_theta*np.abs(dtht) + w_r*np.abs(dr)
    
    if Kt <= Kp:
        row_ind, col_ind = linear_sum_assignment(C)
        row_ind, col_ind = row_ind[:K], col_ind[:K]
        t_idx, p_idx = row_ind, col_ind
    else:
        col_ind, row_ind = linear_sum_assignment(C.T)
        row_ind, col_ind = row_ind[:K], col_ind[:K]
        t_idx, p_idx = row_ind, col_ind
    
    # Matched errors
    e_phi = ang_wrap_diff(phi_t[t_idx], phi_p[p_idx])
    e_tht = ang_wrap_diff(theta_t[t_idx], theta_p[p_idx])
    if (r_t.size > 0) and (r_p.size > 0):
        e_rng = (r_t[t_idx] - r_p[p_idx])
    else:
        e_rng = np.array([], dtype=np.float64)
    
    # Individual RMSE
    dphi_rmse   = float(np.sqrt(np.mean(e_phi**2)))   if e_phi.size else float("nan")
    dtheta_rmse = float(np.sqrt(np.mean(e_tht**2)))   if e_tht.size else float("nan")
    dr_rmse     = float(np.sqrt(np.mean(e_rng**2)))   if e_rng.size else float("nan")
    
    if not return_rmspe:
        return dphi_rmse, dtheta_rmse, dr_rmse
    
    # RMSPE: 3D position error after matching
    if (r_t.size > 0) and (r_p.size > 0) and len(t_idx) > 0:
        # Convert matched pairs to Cartesian
        gt_x, gt_y, gt_z = _cartesian_from_spherical(phi_t[t_idx], theta_t[t_idx], r_t[t_idx])
        pred_x, pred_y, pred_z = _cartesian_from_spherical(phi_p[p_idx], theta_p[p_idx], r_p[p_idx])
        
        # 3D position errors
        pos_errors = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2 + (gt_z - pred_z)**2)
        rmspe = float(np.sqrt(np.mean(pos_errors**2)))
    else:
        rmspe = float("nan")
    
    return dphi_rmse, dtheta_rmse, dr_rmse, rmspe


def _gt_from_item(item):
    if "phi" in item and "theta" in item and "r" in item:
        return (np.asarray(item["phi"]).tolist(),
                np.asarray(item["theta"]).tolist(),
                np.asarray(item["r"]).tolist())
    if "ptr" in item and "K" in item:
        a = np.asarray(item["ptr"], dtype=np.float32).reshape(-1)
        K = int(item["K"]); KMAX = int(cfg.K_MAX)
        if a.size >= 3*KMAX:
            phi = a[0:KMAX][:K]
            tht = a[KMAX:2*KMAX][:K]
            rr  = a[2*KMAX:3*KMAX][:K]
            return (phi.tolist(), tht.tolist(), rr.tolist())
    return ([], [], [])


# --------------- Estimators ---------------

def _estimate_hybrid(model, sample, blind_k=True):
    # Convert real/imaginary format to complex for hybrid_estimate_final
    def to_complex_numpy(ri_tensor):
        """Convert [real, imag] tensor to complex numpy array"""
        ri_np = ri_tensor.numpy()
        if ri_np.shape[-1] == 2:
            return ri_np[..., 0] + 1j * ri_np[..., 1]
        else:
            return ri_np.astype(np.complex64)
    
    s = {
        "y_cplx": to_complex_numpy(sample["y"]),
        "H_cplx": to_complex_numpy(sample["H"]),
        "codes": to_complex_numpy(sample["codes"]),
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
    H_ri = sample["H"].numpy()
    C_ri = sample["codes"].numpy()
    R_inc = incident_cov_from_snaps(y_ri, H_ri, C_ri)
    K_hat = estimate_k_ic_from_cov(R_inc) if blind_k else int(sample["K"])
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


# --------------- Core runner ---------------

def _run_on_dataset(tag, dset, baselines=("ramezani","dcd","nfssn"), blind_k=True, limit=None):
    model = load_model()
    N = len(dset) if limit is None else min(limit, len(dset))
    recs = []
    t0 = time.time()
    for idx in range(N):
        item = dset[idx]
        gt = _gt_from_item(item)
        # Hybrid
        (ph_h, tht_h, rr_h), t_h = _estimate_hybrid(model, item, blind_k=blind_k)
        err_h = _match_err(gt, (ph_h, tht_h, rr_h), return_rmspe=True)
        recs.append(dict(tag=tag, SNR=float(item["snr"]), K=int(item["K"]), who="Hybrid",
                         phi=err_h[0], theta=err_h[1], rng=err_h[2], rmspe=err_h[3],
                         t_hybrid_ms=t_h, t_mod_ms=np.nan, t_dcd_ms=np.nan, t_nfssn_ms=np.nan,
                         mode=("Blind-K" if blind_k else "Oracle-K")))
        # Baselines
        for base in baselines:
            (ph_b, tht_b, rr_b), t_b = _estimate_with_baseline(base, item, blind_k=blind_k)
            err_b = _match_err(gt, (ph_b, tht_b, rr_b), return_rmspe=True)
            who = ("Ramezani-MOD-MUSIC" if base=="ramezani" else
                   "DCD-MUSIC" if base=="dcd" else
                   "NF-SubspaceNet" if base=="nfssn" else
                   "Decoupled-MOD-MUSIC")
            recs.append(dict(tag=tag, SNR=float(item["snr"]), K=int(item["K"]), who=who,
                             phi=err_b[0], theta=err_b[1], rng=err_b[2], rmspe=err_b[3],
                             t_hybrid_ms=np.nan,
                             t_mod_ms=t_b if base in ("ramezani","decoupled_mod") else np.nan,
                             t_dcd_ms=t_b if base=="dcd" else np.nan,
                             t_nfssn_ms=t_b if base=="nfssn" else np.nan,
                             mode=("Blind-K" if blind_k else "Oracle-K")))
    out = BENCH_DIR / f"{tag}.csv"
    pd.DataFrame(recs).to_csv(out, index=False)
    print(f"[{tag}] Saved {out} in {(time.time()-t0):.1f}s")
    return out


# --------------- Indexed subset ---------------

class _IndexedSubset:
    def __init__(self, base, indices): self.base, self.indices = base, list(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return self.base[self.indices[i]]


# --------------- Suite B1–B7 ---------------

def B1_all_blind(baselines=("ramezani","dcd","nfssn"), limit=None):
    dset = ShardNPZDataset(TEST_DIR); return _run_on_dataset("B1_all_blind", dset, baselines, True, limit)

def B2_all_oracle(baselines=("ramezani","dcd","nfssn"), limit=None):
    dset = ShardNPZDataset(TEST_DIR); return _run_on_dataset("B2_all_oracle", dset, baselines, False, limit)

def B3_by_K_blind(baselines=("ramezani","dcd","nfssn"), limit_per_K=None):
    outs=[]; full=ShardNPZDataset(TEST_DIR)
    for k in range(1, int(cfg.K_MAX)+1):
        idxs=[i for i in range(len(full)) if int(full[i]["K"])==k]
        if not idxs: continue
        sub=_IndexedSubset(full, idxs[:limit_per_K] if limit_per_K else idxs)
        outs.append(_run_on_dataset(f"B3_K{k}_blind", sub, baselines, True))
    return outs

def B4_by_SNR_blind(bins=(-10,0,5,10,15,20,25), baselines=("ramezani","dcd","nfssn"), limit_per_bin=None):
    outs=[]; full=ShardNPZDataset(TEST_DIR)
    snrs=np.array([float(full[i]["snr"]) for i in range(len(full))])
    for lo,hi in zip(bins[:-1], bins[1:]):
        idxs=[i for i in range(len(full)) if lo<=snrs[i]<hi]
        if not idxs: continue
        sub=_IndexedSubset(full, idxs[:limit_per_bin] if limit_per_bin else idxs)
        outs.append(_run_on_dataset(f"B4_SNR_{int(lo)}_{int(hi)}_blind", sub, baselines, True))
    return outs

def B5_by_range_bins_blind(edges=(0.5,1.0,2.0,3.5,5.0), baselines=("ramezani","dcd","nfssn"), limit_per_bin=None):
    outs=[]; full=ShardNPZDataset(TEST_DIR)
    try:
        r_means=[]
        for i in range(len(full)):
            r=np.asarray(_gt_from_item(full[i])[2]); r_means.append(float(r[: int(full[i]["K"])].mean()) if r.size else np.nan)
        r_means=np.asarray(r_means)
    except Exception:
        print("[B5] Missing GT r; skipping."); return outs
    for lo,hi in zip(edges[:-1], edges[1:]):
        idxs=[i for i in range(len(full)) if np.isfinite(r_means[i]) and (lo<=r_means[i]<hi)]
        if not idxs: continue
        sub=_IndexedSubset(full, idxs[:limit_per_bin] if limit_per_bin else idxs)
        outs.append(_run_on_dataset(f"B5_R_{int(lo)}_{int(hi)}_blind", sub, baselines, True))
    return outs

def B6_by_phi_fov_blind(edges=(-np.deg2rad(60), -np.deg2rad(20), 0.0, np.deg2rad(20), np.deg2rad(60)),
                        baselines=("ramezani","dcd","nfssn"), limit_per_bin=None):
    outs=[]; full=ShardNPZDataset(TEST_DIR)
    try:
        phi_means=[]
        for i in range(len(full)):
            phi=np.asarray(_gt_from_item(full[i])[0]); phi_means.append(float(phi[: int(full[i]["K"])].mean()) if phi.size else np.nan)
        phi_means=np.asarray(phi_means)
    except Exception:
        print("[B6] Missing GT phi; skipping."); return outs
    for lo,hi in zip(edges[:-1], edges[1:]):
        idxs=[i for i in range(len(full)) if np.isfinite(phi_means[i]) and (lo<=phi_means[i]<hi)]
        if not idxs: continue
        sub=_IndexedSubset(full, idxs[:limit_per_bin] if limit_per_bin else idxs)
        lab=f"B6_PHI_{int(np.rad2deg(lo))}_{int(np.rad2deg(hi))}_blind"
        outs.append(_run_on_dataset(lab, sub, baselines, True))
    return outs

def B7_oracle_full_sweep(baselines=("ramezani","dcd","nfssn"), limit=None):
    dset = ShardNPZDataset(TEST_DIR); return _run_on_dataset("B7_all_oracle_full", dset, baselines, False, limit)


# --------------- Extra ICC views: K=2 SNR sweep, RMSE vs K @15dB, Heatmap ---------------

def _select_indices_by_K(dset, K): return [i for i in range(len(dset)) if int(dset[i]["K"]) == int(K)]
def _select_indices_by_SNR(dset, lo, hi):
    idxs=[]; 
    for i in range(len(dset)):
        s=float(dset[i]["snr"])
        if lo<=s<hi: idxs.append(i)
    return idxs
def _select_indices_at_SNR(dset, target_snr, tol=0.75): return _select_indices_by_SNR(dset, target_snr-tol, target_snr+tol)

def B4_k2_snr_sweep(edges=(-10,-5,0,5,10,15,20,25), baselines=("ramezani","dcd","nfssn"), limit_per_bin=None):
    outs=[]; allset=ShardNPZDataset(TEST_DIR)
    k2_idx=_select_indices_by_K(allset, 2)
    if not k2_idx: 
        print("[B4_k2] No K=2 samples."); 
        return outs
    k2=_IndexedSubset(allset, k2_idx)
    for lo,hi in zip(edges[:-1], edges[1:]):
        idx=_select_indices_by_SNR(k2, lo, hi)
        if not idx: continue
        sub=_IndexedSubset(k2, idx[:limit_per_bin] if limit_per_bin else idx)
        outs.append(_run_on_dataset(f"B4K2_SNR_{int(lo)}_{int(hi)}_blind", sub, baselines, True))
    return outs

def B8_rmse_vs_K_at_snr(snr_target=15.0, tol=0.75, K_list=None, baselines=("ramezani","dcd","nfssn"), limit_per_K=None):
    allset=ShardNPZDataset(TEST_DIR)
    idx=_select_indices_at_SNR(allset, snr_target, tol)
    if not idx: 
        print(f"[B8] No samples near {snr_target}±{tol} dB.")
        return []
    near=_IndexedSubset(allset, idx)
    if K_list is None: K_list=list(range(1, int(cfg.K_MAX)+1))
    outs=[]
    for k in K_list:
        kidx=_select_indices_by_K(near, k)
        if not kidx: continue
        sub=_IndexedSubset(near, kidx[:limit_per_K] if limit_per_K else kidx)
        outs.append(_run_on_dataset(f"B8_SNR{int(round(snr_target))}_K{k}_blind", sub, baselines, True))
    return outs

def B9_heatmap_K_by_SNR(snr_edges=(-10,-5,0,5,10,15,20,25), K_list=None, baselines=("ramezani","dcd","nfssn"), limit_per_cell=None):
    dset=ShardNPZDataset(TEST_DIR)
    if K_list is None: K_list=list(range(1, int(cfg.K_MAX)+1))
    recs=[]; model=load_model()
    for k in K_list:
        idx_k=_select_indices_by_K(dset, k)
        if not idx_k: continue
        d_k=_IndexedSubset(dset, idx_k)
        for lo,hi in zip(snr_edges[:-1], snr_edges[1:]):
            idx_s=_select_indices_by_SNR(d_k, lo, hi)
            if not idx_s: continue
            if limit_per_cell: idx_s=idx_s[:limit_per_cell]
            cell=_IndexedSubset(d_k, idx_s)
            for i in range(len(cell)):
                item=cell[i]; gt=_gt_from_item(item)
                (ph_h, tht_h, rr_h), t_h=_estimate_hybrid(model, item, True)
                err_h=_match_err(gt, (ph_h, tht_h, rr_h), return_rmspe=True)
                recs.append(dict(tag="B9_heatmap", who="Hybrid",
                                 K=int(item["K"]), SNR=float(item["snr"]),
                                 K_bin=int(item["K"]), SNR_bin=f"{int(lo)}:{int(hi)}",
                                 phi=err_h[0], theta=err_h[1], rng=err_h[2], rmspe=err_h[3],
                                 t_hybrid_ms=t_h, t_mod_ms=np.nan, t_dcd_ms=np.nan, t_nfssn_ms=np.nan,
                                 mode="Blind-K"))
                for base in baselines:
                    (ph_b, tht_b, rr_b), t_b=_estimate_with_baseline(base, item, True)
                    err_b=_match_err(gt, (ph_b, tht_b, rr_b), return_rmspe=True)
                    who=("Ramezani-MOD-MUSIC" if base=="ramezani" else
                         "DCD-MUSIC" if base=="dcd" else
                         "NF-SubspaceNet" if base=="nfssn" else
                         "Decoupled-MOD-MUSIC")
                    recs.append(dict(tag="B9_heatmap", who=who,
                                     K=int(item["K"]), SNR=float(item["snr"]),
                                     K_bin=int(item["K"]), SNR_bin=f"{int(lo)}:{int(hi)}",
                                     phi=err_b[0], theta=err_b[1], rng=err_b[2], rmspe=err_b[3],
                                     t_hybrid_ms=np.nan,
                                     t_mod_ms=t_b if base in ("ramezani","decoupled_mod") else np.nan,
                                     t_dcd_ms=t_b if base=="dcd" else np.nan,
                                     t_nfssn_ms=t_b if base=="nfssn" else np.nan,
                                     mode="Blind-K"))
    out=BENCH_DIR/"B9_heatmap.csv"
    pd.DataFrame(recs).to_csv(out, index=False)
    print(f"[B9] Saved {out}")
    return out


# --------------- Aggregators for CLI ---------------

def run_all_benchmarks(tag="suite", baselines=("ramezani","dcd","nfssn"),
                       blind_k=True, oracle_too=False, save_prefix=None, limit=None):
    if blind_k:
        out=B1_all_blind(baselines, limit)
    else:
        out=B2_all_oracle(baselines, limit)
    if oracle_too:
        B2_all_oracle(baselines, limit)
    return out

def run_full_suite(tag="suite", include_decoupled=False, limit=None):
    bases=["ramezani","dcd","nfssn"] + (["decoupled_mod"] if include_decoupled else [])
    outs=[]
    outs.append(B1_all_blind(tuple(bases), limit))
    outs.append(B2_all_oracle(tuple(bases), limit))
    outs.extend(B3_by_K_blind(tuple(bases)))
    outs.extend(B4_by_SNR_blind(tuple(bases)))
    outs.extend(B5_by_range_bins_blind(tuple(bases)))
    outs.extend(B6_by_phi_fov_blind(tuple(bases)))
    outs.append(B7_oracle_full_sweep(tuple(bases), limit))
    # ICC extras
    outs.extend(B4_k2_snr_sweep(tuple(bases)))
    outs.extend(B8_rmse_vs_K_at_snr(15.0, tol=0.75, baselines=tuple(bases)))
    outs.append(B9_heatmap_K_by_SNR(tuple(bases)))
    return outs


def run_bench_csv(model, n=300, oracle=False, outf="bench_blind.csv"):
    import csv
    dset = ShardNPZDataset(TEST_DIR)
    rows = []
    N = min(n, len(dset))
    for i in range(N):
        it = dset[i]
        s = {"y": it["y"], "H": it["H"], "codes": it["codes"], "K": int(it["K"]), "snr_db": float(it["snr"])}
        pr = hybrid_estimate_final(model, s, force_K=(int(it["K"]) if oracle else None), k_policy="mdl", do_newton=True)
        gt_phi = _gt_from_item(it)[0]
        gt_phi0 = float(gt_phi[0]) if len(gt_phi) > 0 else 0.0
        pred_phi0 = float(pr[0][0]) if pr and len(pr[0]) > 0 else 0.0
        rows.append([gt_phi0, pred_phi0])
    with open(outf, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return outf
