#!/usr/bin/env python3
"""
Sweep MVDR + SpectrumRefiner inference knobs on a fixed validation slice and write a CSV.

This is intended for *calibration*, not training:
  - MVDR diagonal loading: cfg.MVDR_DELTA_SCALE
  - MVDR thresholding aggressiveness: cfg.MVDR_CFAR_Z (with cfg.MVDR_THRESH_MODE="mad")
  - Refiner peak selection: cfg.REFINER_REL_THRESH, cfg.REFINER_NMS_MIN_SEP
  - Hybrid blend beta: cfg.HYBRID_COV_BETA (requires R_samp in shards)

Usage example:
  cd /home/tahit/ris/MainMusic
  python sweep_knobs.py --ckpt results_final_L16_12x12/checkpoints/refiner_best.pt --n 2000 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

from ris_pytorch_pipeline.configs import cfg
from ris_pytorch_pipeline.dataset import ShardNPZDataset
from ris_pytorch_pipeline.eval_angles import eval_scene_angles_ranges
from ris_pytorch_pipeline.infer import hybrid_estimate_final, load_model


def _parse_csv_floats(s: str) -> List[float]:
    out: List[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("expected a non-empty comma-separated float list")
    return out


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("expected a non-empty comma-separated int list")
    return out


def _safe_mean(x: List[float]) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def _safe_median(x: List[float]) -> float:
    return float(np.median(x)) if len(x) else float("nan")


def _safe_rmse(x: List[float]) -> float:
    if not x:
        return float("nan")
    a = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a)))


@torch.no_grad()
def _eval_one_setting(
    model,
    ds: ShardNPZDataset,
    indices: np.ndarray,
    *,
    success_tol_phi: float,
    success_tol_theta: float,
    success_tol_r: float,
) -> dict:
    succ = 0
    num_pred_list: List[int] = []
    num_gt_list: List[int] = []
    all_phi_err: List[float] = []
    all_theta_err: List[float] = []
    all_r_err: List[float] = []

    for j, idx in enumerate(indices.tolist()):
        s = ds[int(idx)]
        k_gt = int(s["K"].item()) if torch.is_tensor(s["K"]) else int(s["K"])
        num_gt_list.append(k_gt)

        # Ground truth from decoded ptr (already length K_gt)
        phi_gt = s["phi"].detach().cpu().numpy().astype(np.float64) if torch.is_tensor(s["phi"]) else np.asarray(s["phi"], dtype=np.float64)
        theta_gt = s["theta"].detach().cpu().numpy().astype(np.float64) if torch.is_tensor(s["theta"]) else np.asarray(s["theta"], dtype=np.float64)
        r_gt = s["r"].detach().cpu().numpy().astype(np.float64) if torch.is_tensor(s["r"]) else np.asarray(s["r"], dtype=np.float64)

        phi_gt_deg = np.rad2deg(phi_gt)
        theta_gt_deg = np.rad2deg(theta_gt)

        # Prediction (returns radians + meters)
        try:
            phi_p, theta_p, r_p = hybrid_estimate_final(model, s, force_K=None, do_newton=True)
        except Exception:
            phi_p, theta_p, r_p = [], [], []

        num_pred_list.append(int(len(phi_p)))

        phi_p_deg = np.rad2deg(np.asarray(phi_p, dtype=np.float64)) if len(phi_p) else np.asarray([], dtype=np.float64)
        theta_p_deg = np.rad2deg(np.asarray(theta_p, dtype=np.float64)) if len(theta_p) else np.asarray([], dtype=np.float64)
        r_p_arr = np.asarray(r_p, dtype=np.float64) if len(r_p) else np.asarray([], dtype=np.float64)

        m = eval_scene_angles_ranges(
            phi_p_deg, theta_p_deg, r_p_arr,
            phi_gt_deg, theta_gt_deg, r_gt,
            success_tol_phi=success_tol_phi,
            success_tol_theta=success_tol_theta,
            success_tol_r=success_tol_r,
        )

        if bool(m.get("success_flag", False)):
            succ += 1

        all_phi_err.extend(m.get("raw_phi_errors", []) or [])
        all_theta_err.extend(m.get("raw_theta_errors", []) or [])
        all_r_err.extend(m.get("raw_r_errors", []) or [])

        if (j + 1) % 200 == 0:
            print(f"  eval {j+1}/{len(indices)} scenes...", flush=True)

    n_scenes = int(len(indices))
    return {
        "n_scenes": n_scenes,
        "success_rate": float(succ / max(1, n_scenes)),
        "med_phi": _safe_median(all_phi_err),
        "med_theta": _safe_median(all_theta_err),
        "med_r": _safe_median(all_r_err),
        "rmse_phi": _safe_rmse(all_phi_err),
        "rmse_theta": _safe_rmse(all_theta_err),
        "rmse_r": _safe_rmse(all_r_err),
        "avg_num_pred": _safe_mean([float(x) for x in num_pred_list]),
        "avg_num_gt": _safe_mean([float(x) for x in num_gt_list]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to combined checkpoint with {'backbone','refiner'} state dicts.")
    ap.add_argument("--val_dir", default=None, help="Validation shards directory (defaults to cfg.DATA_SHARDS_VAL).")
    ap.add_argument("--n", type=int, default=2000, help="Number of validation scenes to evaluate per setting.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for selecting the validation slice.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for model inference.")

    ap.add_argument("--delta_scales", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    ap.add_argument("--cfar_z", default="3,4,5,6,7")
    ap.add_argument("--refiner_rel_thresh", default="0.10,0.15,0.20,0.25,0.30")
    ap.add_argument("--nms_min_sep", default="1,2,3")
    ap.add_argument("--betas", default="0.0,0.1,0.2,0.3,0.4")

    ap.add_argument("--success_tol_phi", type=float, default=5.0)
    ap.add_argument("--success_tol_theta", type=float, default=5.0)
    ap.add_argument("--success_tol_r", type=float, default=1.0)

    ap.add_argument("--out", default=None, help="Output CSV path (default: results_final_L16_12x12/knob_sweeps/sweep_<timestamp>.csv)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    # Resolve validation dir
    if args.val_dir is None:
        val_dir = Path("/home/tahit/ris/MainMusic") / str(getattr(cfg, "DATA_SHARDS_VAL", "data_shards_M64_L16/val"))
    else:
        val_dir = Path(args.val_dir).expanduser().resolve()
    if not val_dir.exists():
        raise SystemExit(f"val_dir not found: {val_dir}")

    # Output path
    if args.out is None:
        out_dir = Path("/home/tahit/ris/MainMusic") / str(getattr(cfg, "RESULTS_DIR", "results_final_L16_12x12")) / "knob_sweeps"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"sweep_{ts}.csv"
    else:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model (combined backbone+refiner required by design)
    print(f"[sweep] loading checkpoint: {ckpt_path}", flush=True)
    model = load_model(ckpt_dir=str(ckpt_path.parent), ckpt_name=ckpt_path.name, map_location="cpu", prefer_swa=False, require_refiner=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    if hasattr(model, "_spectrum_refiner") and (getattr(model, "_spectrum_refiner") is not None):
        model._spectrum_refiner = model._spectrum_refiner.to(device)
    model.eval()

    # Dataset and fixed slice
    print(f"[sweep] loading val shards: {val_dir}", flush=True)
    ds = ShardNPZDataset(str(val_dir))
    n = min(int(args.n), len(ds))
    rng = np.random.default_rng(int(args.seed))
    indices = rng.choice(len(ds), size=n, replace=False)
    print(f"[sweep] using n={n} scenes (seed={args.seed})", flush=True)

    delta_scales = _parse_csv_floats(args.delta_scales)
    cfar_zs = _parse_csv_floats(args.cfar_z)
    rel_thrs = _parse_csv_floats(args.refiner_rel_thresh)
    nms_seps = _parse_csv_ints(args.nms_min_sep)
    betas = _parse_csv_floats(args.betas)

    # Ensure the intended modes are active
    cfg.MVDR_THRESH_MODE = "mad"
    cfg.USE_SPECTRUM_REFINER_IN_INFER = True

    rows = []
    combos = list(itertools.product(delta_scales, cfar_zs, rel_thrs, nms_seps, betas))
    print(f"[sweep] total settings: {len(combos)}", flush=True)

    t_start = time.time()
    for k_i, (delta_scale, cfar_z, rel_thr, nms_sep, beta) in enumerate(combos):
        cfg.MVDR_DELTA_SCALE = float(delta_scale)
        cfg.MVDR_CFAR_Z = float(cfar_z)
        cfg.REFINER_REL_THRESH = float(rel_thr)
        cfg.REFINER_NMS_MIN_SEP = int(nms_sep)
        cfg.HYBRID_COV_BETA = float(beta)

        print(
            f"\n[sweep] {k_i+1}/{len(combos)} "
            f"delta={cfg.MVDR_DELTA_SCALE:g} cfar_z={cfg.MVDR_CFAR_Z:g} "
            f"rel_thr={cfg.REFINER_REL_THRESH:g} nms={cfg.REFINER_NMS_MIN_SEP} beta={cfg.HYBRID_COV_BETA:g}",
            flush=True,
        )

        t0 = time.time()
        m = _eval_one_setting(
            model,
            ds,
            indices,
            success_tol_phi=float(args.success_tol_phi),
            success_tol_theta=float(args.success_tol_theta),
            success_tol_r=float(args.success_tol_r),
        )
        dt = time.time() - t0

        row = {
            "delta_scale": cfg.MVDR_DELTA_SCALE,
            "cfar_z": cfg.MVDR_CFAR_Z,
            "refiner_rel_thresh": cfg.REFINER_REL_THRESH,
            "nms_min_sep": cfg.REFINER_NMS_MIN_SEP,
            "beta": cfg.HYBRID_COV_BETA,
            "n_scenes": m["n_scenes"],
            "success_rate": m["success_rate"],
            "med_phi": m["med_phi"],
            "med_theta": m["med_theta"],
            "med_r": m["med_r"],
            "rmse_phi": m["rmse_phi"],
            "rmse_theta": m["rmse_theta"],
            "rmse_r": m["rmse_r"],
            "avg_num_pred": m["avg_num_pred"],
            "avg_num_gt": m["avg_num_gt"],
            "seconds": dt,
        }
        rows.append(row)

        print(
            f"[sweep] success={row['success_rate']:.3f} "
            f"med(φ,θ,r)=({row['med_phi']:.2f}°, {row['med_theta']:.2f}°, {row['med_r']:.2f}m) "
            f"rmse(φ,θ,r)=({row['rmse_phi']:.2f}°, {row['rmse_theta']:.2f}°, {row['rmse_r']:.2f}m) "
            f"avgK_pred={row['avg_num_pred']:.2f} time={dt:.1f}s",
            flush=True,
        )

    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else [
        "delta_scale","cfar_z","refiner_rel_thresh","nms_min_sep","beta",
        "n_scenes","success_rate","med_phi","med_theta","med_r","rmse_phi","rmse_theta","rmse_r",
        "avg_num_pred","avg_num_gt","seconds"
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    total_dt = time.time() - t_start
    print(f"\n[sweep] wrote: {out_path}", flush=True)
    print(f"[sweep] total time: {total_dt/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

