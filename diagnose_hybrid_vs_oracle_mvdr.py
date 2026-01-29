import argparse
from pathlib import Path
import numpy as np
import torch

from ris_pytorch_pipeline.configs import cfg
from ris_pytorch_pipeline.dataset import ShardNPZDataset
from ris_pytorch_pipeline.infer import load_model, hybrid_estimate_final
from ris_pytorch_pipeline.eval_angles import hungarian_pairs_angles_ranges
from ris_pytorch_pipeline.music_gpu import mvdr_detect_sources, get_gpu_estimator


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

        # ---- Hybrid MVDR-first on R_pred ----
        sample = {
            "y": it["y"],
            "H": it["H"],
            "codes": it["codes"],
            "snr_db": float(it.get("snr", it.get("snr_db", 10.0))),
        }
        force_K = int(K) if args.oracle_k else None
        ph, th, rr = hybrid_estimate_final(model, sample, force_K=force_K, k_policy="mdl")
        phi_p = np.rad2deg(np.asarray(ph, float))
        th_p = np.rad2deg(np.asarray(th, float))
        r_p = np.asarray(rr, float)
        tp, fp, fn = _tp_fp_fn(phi_p, th_p, r_p, gt_phi_deg, gt_th_deg, gt_r, tol_phi=args.tol_phi, tol_th=args.tol_theta, tol_r=args.tol_r)
        TP_hy += tp; FP_hy += fp; FN_hy += fn

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


if __name__ == "__main__":
    main()

