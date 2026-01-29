import argparse
from pathlib import Path
import numpy as np
import torch

from ris_pytorch_pipeline.configs import cfg
from ris_pytorch_pipeline.dataset import ShardNPZDataset
from ris_pytorch_pipeline.infer import load_model, hybrid_estimate_final
from ris_pytorch_pipeline.eval_angles import hungarian_pairs_angles_ranges
from ris_pytorch_pipeline.music_gpu import mvdr_detect_sources, get_gpu_estimator
from ris_pytorch_pipeline.covariance_utils import build_effective_cov_np


def _to_complex_numpy(ri):
    ri = np.asarray(ri)
    if np.iscomplexobj(ri):
        return ri.astype(np.complex64)
    if ri.shape[-1] == 2:
        return (ri[..., 0] + 1j * ri[..., 1]).astype(np.complex64)
    return ri.astype(np.complex64)


def _tp_fp_fn(phi_p_deg, th_p_deg, r_p_m, phi_g_deg, th_g_deg, r_g_m, *, tol_phi=5.0, tol_th=5.0, tol_r=1.0):
    pairs = hungarian_pairs_angles_ranges(phi_p_deg, th_p_deg, r_p_m, phi_g_deg, th_g_deg, r_g_m)
    tp = 0
    for pi, gi in pairs:
        if (
            abs(phi_p_deg[pi] - phi_g_deg[gi]) <= tol_phi
            and abs(th_p_deg[pi] - th_g_deg[gi]) <= tol_th
            and abs(r_p_m[pi] - r_g_m[gi]) <= tol_r
        ):
            tp += 1
    fp = max(0, len(phi_p_deg) - tp)
    fn = max(0, len(phi_g_deg) - tp)
    return tp, fp, fn


def _nmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    num = np.linalg.norm(a - b) ** 2
    den = np.linalg.norm(b) ** 2
    return float(num / max(1e-12, den))


def _unpack_factor(v_flat: np.ndarray, *, N: int, Kmax: int, mode: str) -> np.ndarray:
    """
    Convert factor vector to complex A[N,Kmax].
    mode:
      - "interleaved": [re0, im0, re1, im1, ...]
      - "split":       [re...(N*K), im...(N*K)]
    """
    v_flat = np.asarray(v_flat, dtype=np.float32).reshape(-1)
    NK = int(N * Kmax)
    if mode == "interleaved":
        xr, xi = v_flat[::2], v_flat[1::2]
        if xr.size < NK or xi.size < NK:
            raise ValueError(f"bad factor length for interleaved: got {v_flat.size}, need >= {2*NK}")
        A = (xr[:NK] + 1j * xi[:NK]).reshape(N, Kmax)
        return A.astype(np.complex64)
    if mode == "split":
        if v_flat.size < 2 * NK:
            raise ValueError(f"bad factor length for split: got {v_flat.size}, need >= {2*NK}")
        xr = v_flat[:NK]
        xi = v_flat[NK:2*NK]
        A = (xr + 1j * xi).reshape(N, Kmax)
        return A.astype(np.complex64)
    raise ValueError(f"unknown mode {mode}")


def _build_R_pred_from_factors(cf_ang: np.ndarray, cf_rng: np.ndarray, *, lam_range: float, N: int, Kmax: int, mode: str) -> np.ndarray:
    A_ang = _unpack_factor(cf_ang, N=N, Kmax=Kmax, mode=mode)
    A_rng = _unpack_factor(cf_rng, N=N, Kmax=Kmax, mode=mode)
    R = (A_ang @ A_ang.conj().T) + float(lam_range) * (A_rng @ A_rng.conj().T)
    R = 0.5 * (R + R.conj().T)
    return R.astype(np.complex64)


def _subspace_overlap(R_pred: np.ndarray, R_true: np.ndarray, K: int) -> float:
    """
    Compute average squared cosine of principal angles between top-K eigenspaces.
    Returns in [0,1], higher is better.
    """
    K = int(max(1, K))
    # eigh gives ascending eigenvalues; take last K
    ew_p, ev_p = np.linalg.eigh(R_pred.astype(np.complex128))
    ew_t, ev_t = np.linalg.eigh(R_true.astype(np.complex128))
    Up = ev_p[:, np.argsort(ew_p)[-K:]]  # [N,K]
    Ut = ev_t[:, np.argsort(ew_t)[-K:]]  # [N,K]
    # Singular values of Ut^H Up are cos(principal angles)
    s = np.linalg.svd(Ut.conj().T @ Up, compute_uv=False)
    s = np.clip(np.real(s), 0.0, 1.0)
    return float(np.mean(s**2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val", help="Which shard split to evaluate.")
    ap.add_argument("--n", type=int, default=200, help="Max scenes to evaluate.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", type=str, default=None, help="Override checkpoint directory (defaults to cfg.CKPT_DIR).")
    ap.add_argument("--ckpt-name", type=str, default="best.pt", help="Checkpoint file name under ckpt-dir.")
    ap.add_argument("--oracle-k", action="store_true", help="Use oracle K for Hybrid inference (otherwise blind-K).")
    ap.add_argument("--tol-phi", type=float, default=5.0)
    ap.add_argument("--tol-theta", type=float, default=5.0)
    ap.add_argument("--tol-r", type=float, default=1.0)
    args = ap.parse_args()

    split_dir = Path(cfg.DATA_SHARDS_VAL if args.split == "val" else cfg.DATA_SHARDS_TEST)
    ds = ShardNPZDataset(str(split_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt_dir=args.ckpt_dir, ckpt_name=args.ckpt_name, map_location="cpu", device=device, require_refiner=False)

    est = get_gpu_estimator(cfg, device=device)
    r_planes = est.default_r_planes_mvdr

    rng = np.random.default_rng(int(args.seed))
    idx = rng.choice(len(ds), size=min(int(args.n), len(ds)), replace=False)

    TP_or = FP_or = FN_or = 0
    TP_hy = FP_hy = FN_hy = 0
    total_gt = 0

    # Quality diagnostics for R_pred vs R_true (to catch factor unpacking mismatches)
    nmse_int = []
    nmse_split = []
    ov_int = []
    ov_split = []

    for i in idx:
        it = ds[int(i)]
        K = int(it.get("K", 0))
        if K <= 0:
            continue
        Kmax = int(cfg.K_MAX)
        ptr = np.asarray(it["ptr"], dtype=np.float32).reshape(-1)
        gt_phi_deg = np.rad2deg(ptr[0:Kmax][:K])
        gt_th_deg = np.rad2deg(ptr[Kmax:2 * Kmax][:K])
        gt_r = ptr[2 * Kmax:3 * Kmax][:K]
        total_gt += K
        snr_db = float(it.get("snr", it.get("snr_db", 10.0)))

        # ---- Oracle MVDR on R_true ----
        R = it.get("R")
        if R is not None:
            R = _to_complex_numpy(R)
            sources, _ = mvdr_detect_sources(
                torch.as_tensor(R, device=device, dtype=torch.complex64),
                cfg,
                device=device,
                grid_phi=int(getattr(cfg, "MVDR_GRID_PHI", 361)),
                grid_theta=int(getattr(cfg, "MVDR_GRID_THETA", 181)),
                r_planes=r_planes,
                delta_scale=float(getattr(cfg, "MVDR_DELTA_SCALE", 1e-2)),
                threshold_db=float(getattr(cfg, "MVDR_THRESH_DB", 12.0)),
                threshold_mode=str(getattr(cfg, "MVDR_THRESH_MODE", "mad")),
                cfar_z=float(getattr(cfg, "MVDR_CFAR_Z", 5.0)),
                max_sources=int(cfg.K_MAX),
                do_refinement=bool(getattr(cfg, "MVDR_DO_REFINEMENT", True)),
            )
            phi_p = np.array([s[0] for s in sources], float)
            th_p = np.array([s[1] for s in sources], float)
            r_p = np.array([s[2] for s in sources], float)
            tp, fp, fn = _tp_fp_fn(phi_p, th_p, r_p, gt_phi_deg, gt_th_deg, gt_r, tol_phi=args.tol_phi, tol_th=args.tol_theta, tol_r=args.tol_r)
            TP_or += tp; FP_or += fp; FN_or += fn
        else:
            continue

        # ---- Hybrid MVDR-first on R_pred ----
        sample = {
            "y": it["y"],
            "H": it["H"],
            "codes": it["codes"],
            "snr_db": snr_db,
        }
        force_K = int(K) if args.oracle_k else None
        ph, th, rr = hybrid_estimate_final(model, sample, force_K=force_K, k_policy="mdl")
        phi_p = np.rad2deg(np.asarray(ph, float))
        th_p = np.rad2deg(np.asarray(th, float))
        r_p = np.asarray(rr, float)
        tp, fp, fn = _tp_fp_fn(phi_p, th_p, r_p, gt_phi_deg, gt_th_deg, gt_r, tol_phi=args.tol_phi, tol_th=args.tol_theta, tol_r=args.tol_r)
        TP_hy += tp; FP_hy += fp; FN_hy += fn

        # ---- R_pred quality diagnostics (two unpackings) ----
        # We recompute R_eff under both factor layouts and compare to R_true_eff.
        try:
            with torch.no_grad():
                # Run a forward pass just to get cov_fact vectors (same as infer uses)
                y = _to_complex_numpy(it["y"])
                H = _to_complex_numpy(it["H"])
                C = _to_complex_numpy(it["codes"])
                dev = next(model.parameters()).device
                y_t = torch.from_numpy(y).to(torch.complex64).unsqueeze(0).to(dev)
                H_t = torch.from_numpy(H).to(torch.complex64).unsqueeze(0).to(dev)
                C_t = torch.from_numpy(C).to(torch.complex64).unsqueeze(0).to(dev)
                pred = model(torch.stack([y_t.real, y_t.imag], dim=-1),
                             torch.stack([H_t.real, H_t.imag], dim=-1),
                             torch.stack([C_t.real, C_t.imag], dim=-1),
                             snr_db=snr_db, R_samp=None)
                cf_ang = pred["cov_fact_angle"][0].detach().cpu().numpy()
                cf_rng = pred["cov_fact_range"][0].detach().cpu().numpy()
            lam_range = float(getattr(cfg, "LAM_RANGE_FACTOR", getattr(__import__("ris_pytorch_pipeline.configs", fromlist=["mdl_cfg"]).mdl_cfg, "LAM_RANGE_FACTOR", 0.3)))
            R_pred_int = _build_R_pred_from_factors(cf_ang, cf_rng, lam_range=lam_range, N=int(cfg.N), Kmax=int(cfg.K_MAX), mode="interleaved")
            R_pred_split = _build_R_pred_from_factors(cf_ang, cf_rng, lam_range=lam_range, N=int(cfg.N), Kmax=int(cfg.K_MAX), mode="split")

            R_true_eff = build_effective_cov_np(R, diag_load=True, apply_shrink=True, snr_db=snr_db, target_trace=float(cfg.N))
            R_int_eff = build_effective_cov_np(R_pred_int, diag_load=True, apply_shrink=True, snr_db=snr_db, target_trace=float(cfg.N))
            R_split_eff = build_effective_cov_np(R_pred_split, diag_load=True, apply_shrink=True, snr_db=snr_db, target_trace=float(cfg.N))
            nmse_int.append(_nmse(R_int_eff, R_true_eff))
            nmse_split.append(_nmse(R_split_eff, R_true_eff))
            ov_int.append(_subspace_overlap(R_int_eff, R_true_eff, K=int(K)))
            ov_split.append(_subspace_overlap(R_split_eff, R_true_eff, K=int(K)))
        except Exception:
            pass

    def _prf(tp, fp, fn):
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-12, p + r)
        return p, r, f1

    p_or, r_or, f_or = _prf(TP_or, FP_or, FN_or)
    p_hy, r_hy, f_hy = _prf(TP_hy, FP_hy, FN_hy)

    mode = "Oracle-K" if args.oracle_k else "Blind-K"
    print(f"[DATA] split={args.split} n_eval={len(idx)} total_gt={total_gt}")
    print(f"[ORACLE MVDR on R_true] TP={TP_or} FP={FP_or} FN={FN_or}  P={p_or:.3f} R={r_or:.3f} F1={f_or:.3f}")
    print(f"[HYBRID MVDR on R_pred] ({mode}) TP={TP_hy} FP={FP_hy} FN={FN_hy}  P={p_hy:.3f} R={r_hy:.3f} F1={f_hy:.3f}")

    if nmse_int and nmse_split:
        print("[R_pred quality] (effective cov vs R_true effective cov; lower NMSE is better)")
        print(f"  NMSE interleaved: median={float(np.median(nmse_int)):.3f} mean={float(np.mean(nmse_int)):.3f} n={len(nmse_int)}")
        print(f"  NMSE split:       median={float(np.median(nmse_split)):.3f} mean={float(np.mean(nmse_split)):.3f} n={len(nmse_split)}")
    if ov_int and ov_split:
        print("[R_pred quality] (top-K subspace overlap; higher is better)")
        print(f"  overlap interleaved: median={float(np.median(ov_int)):.3f} mean={float(np.mean(ov_int)):.3f} n={len(ov_int)}")
        print(f"  overlap split:       median={float(np.median(ov_split)):.3f} mean={float(np.mean(ov_split)):.3f} n={len(ov_split)}")


if __name__ == "__main__":
    main()

