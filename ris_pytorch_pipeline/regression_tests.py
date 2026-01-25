"""
Lightweight regression tests (no pytest required).

Usage:
  python -m ris_pytorch_pipeline.regression_tests mvdr_lowrank
"""

from __future__ import annotations

import argparse
import math
import numpy as np
import torch

from .configs import cfg
from .music_gpu import get_gpu_estimator


def r_samp_sanity_test(*, n: int = 8, seed: int = 0) -> None:
    """
    Regression guardrail: R_samp should be a usable covariance proxy.

    We keep this test lightweight (no MVDR), enforcing basic numerical sanity and that
    R_samp is not degenerate / zero and is at least somewhat correlated with R.
    """
    from pathlib import Path

    def _ri_to_c(x):
        if x is None:
            return None
        x = np.asarray(x)
        if x.shape[-1] == 2:
            return (x[..., 0] + 1j * x[..., 1]).astype(np.complex64)
        return x.astype(np.complex64)

    def _nmse(a, b):
        num = np.linalg.norm(a - b, "fro") ** 2
        den = np.linalg.norm(b, "fro") ** 2 + 1e-12
        return float(num / den)

    # IMPORTANT: Avoid loading the full ShardNPZDataset here (can be heavy / memory-fragile on some machines).
    # Instead, read a single shard with mmap and sample a few indices.
    val_dir = Path(str(getattr(cfg, "DATA_SHARDS_VAL", f"{cfg.DATA_SHARDS_DIR}/val")))
    shard_paths = sorted(val_dir.glob("*.npz"))
    if not shard_paths:
        raise RuntimeError(f"[r_samp] No .npz shards found under: {val_dir}")

    rng = np.random.default_rng(int(seed))
    p0 = shard_paths[0]
    z = np.load(p0, allow_pickle=False, mmap_mode="r")
    n0 = int(z["R"].shape[0])
    idx = rng.choice(n0, size=min(int(n), n0), replace=False)

    nmse_list = []
    herm_list = []
    tr_list = []
    nz_list = []
    N = int(getattr(cfg, "N", 144))

    for i in idx:
        R = _ri_to_c(z["R"][int(i)]) if ("R" in z.files) else None
        Rs = _ri_to_c(z["R_samp"][int(i)]) if ("R_samp" in z.files) else None
        if R is None or Rs is None:
            continue
        Rs_h = 0.5 * (Rs + Rs.conj().T)
        R_h = 0.5 * (R + R.conj().T)
        herm_list.append(float(np.linalg.norm(Rs - Rs.conj().T, "fro") / (np.linalg.norm(Rs, "fro") + 1e-12)))
        tr_list.append(float(np.real(np.trace(Rs_h))))
        nz_list.append(float(np.linalg.norm(Rs_h, "fro")))
        nmse_list.append(_nmse(Rs_h, R_h))

    if len(nmse_list) == 0:
        raise RuntimeError("No samples with both R and R_samp found in val shards.")

    # Hermitian should be very close (it is explicitly hermitized).
    if float(np.median(herm_list)) >= 1e-3:
        raise RuntimeError(f"[r_samp] not Hermitian enough: median={np.median(herm_list)}")

    # Trace should be normalized near N.
    if abs(float(np.median(tr_list)) - float(N)) >= 0.10 * float(N):
        raise RuntimeError(f"[r_samp] trace not near N: median_tr={np.median(tr_list)} vs N={N}")

    # Not degenerate
    if float(np.median(nz_list)) <= 1e-3:
        raise RuntimeError("[r_samp] appears degenerate/zero")

    # Loose correlation check (dataset/SNR dependent; keep generous).
    if float(np.median(nmse_list)) >= 10.0:
        raise RuntimeError(f"[r_samp] too far from R: median NMSE={np.median(nmse_list):.3f}")

    print(
        f"✅ r_samp_sanity_test passed (n={len(nmse_list)}, median_nmse={np.median(nmse_list):.3f}, median_tr={np.median(tr_list):.2f})",
        flush=True,
    )


@torch.no_grad()
def mvdr_lowrank_equivalence_test(
    *,
    n_trials: int = 5,
    k_lr: int = 10,
    grid_phi: int = 181,
    grid_theta: int = 91,
    r_planes: list[float] | None = None,
    delta_scale: float = 1e-2,
    min_corr: float = 0.995,
    max_peak_cell_err: int = 1,
    device: str | None = None,
) -> None:
    """
    Validate Woodbury (low-rank) MVDR against full MVDR on random low-rank covariances.

    This catches cases where training (low-rank MVDR) diverges from inference (full MVDR).
    """
    est = get_gpu_estimator(cfg, device=(device or ("cuda" if torch.cuda.is_available() else "cpu")))
    N = int(getattr(cfg, "N", est.N))
    r_planes = r_planes or est.default_r_planes_mvdr

    grid_phi_t = torch.linspace(
        math.radians(float(getattr(cfg, "PHI_MIN_DEG", -60.0))),
        math.radians(float(getattr(cfg, "PHI_MAX_DEG", 60.0))),
        int(grid_phi),
        device=est.device,
        dtype=torch.float32,
    )
    grid_theta_t = torch.linspace(
        math.radians(float(getattr(cfg, "THETA_MIN_DEG", -30.0))),
        math.radians(float(getattr(cfg, "THETA_MAX_DEG", 30.0))),
        int(grid_theta),
        device=est.device,
        dtype=torch.float32,
    )

    for t in range(int(n_trials)):
        # Random complex low-rank factors, scaled so tr(F F^H) ~= N for realism.
        F = (torch.randn(N, k_lr, device=est.device) + 1j * torch.randn(N, k_lr, device=est.device)).to(torch.complex64)
        tr0 = torch.real((F.conj() * F).sum()).clamp_min(1e-9)
        F = F * torch.sqrt(torch.tensor(float(N), device=est.device) / tr0)

        # Low-rank spectrum (Woodbury MVDR)
        spec_lr = est.mvdr_spectrum_max_2_5d_lowrank(F, grid_phi_t, grid_theta_t, r_planes, delta_scale=float(delta_scale))

        # Full spectrum (explicit inverse with same diagonal loading rule)
        R0 = F @ F.conj().transpose(0, 1)  # [N,N]
        R_inv = est._compute_R_inv(R0, delta_scale=float(delta_scale))
        spec_full = torch.zeros_like(spec_lr)
        for r in r_planes:
            A = est._steering_nearfield_grid(grid_phi_t, grid_theta_t, float(r))
            spec = est._compute_spectrum_mvdr(R_inv, A)
            spec_full = torch.maximum(spec_full, spec)

        # Correlation check (flattened)
        a = spec_lr.flatten().float()
        b = spec_full.flatten().float()
        a = (a - a.mean()) / (a.std() + 1e-9)
        b = (b - b.mean()) / (b.std() + 1e-9)
        corr = float((a * b).mean().detach().cpu().item())
        if not np.isfinite(corr) or corr < float(min_corr):
            raise RuntimeError(f"[mvdr_lowrank] corr too low: {corr:.4f} < {min_corr} (trial {t})")

        # Peak location check (grid cell agreement)
        gi_lr = int(torch.argmax(spec_lr).detach().cpu().item())
        gi_f = int(torch.argmax(spec_full).detach().cpu().item())
        lr_phi, lr_th = divmod(gi_lr, int(grid_theta))
        f_phi, f_th = divmod(gi_f, int(grid_theta))
        cell_err = max(abs(lr_phi - f_phi), abs(lr_th - f_th))
        if cell_err > int(max_peak_cell_err):
            raise RuntimeError(
                f"[mvdr_lowrank] peak mismatch: lr=({lr_phi},{lr_th}) full=({f_phi},{f_th}) "
                f"cell_err={cell_err} > {max_peak_cell_err} (trial {t})"
            )

    print(f"✅ mvdr_lowrank_equivalence_test passed (n_trials={n_trials}, min_corr={min_corr})", flush=True)


def _main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_rs = sub.add_parser("r_samp", help="Sanity-check R_samp against R (lightweight)")
    ap_rs.add_argument("--n", type=int, default=8)
    ap_rs.add_argument("--seed", type=int, default=0)

    ap_lr = sub.add_parser("mvdr_lowrank", help="Validate low-rank MVDR vs full MVDR")
    ap_lr.add_argument("--n", type=int, default=5)
    ap_lr.add_argument("--k_lr", type=int, default=10)
    ap_lr.add_argument("--grid_phi", type=int, default=181)
    ap_lr.add_argument("--grid_theta", type=int, default=91)
    ap_lr.add_argument("--delta_scale", type=float, default=1e-2)
    ap_lr.add_argument("--min_corr", type=float, default=0.995)
    ap_lr.add_argument("--max_peak_cell_err", type=int, default=1)
    ap_lr.add_argument("--device", type=str, default=None)

    args = ap.parse_args()
    if args.cmd == "r_samp":
        r_samp_sanity_test(n=args.n, seed=args.seed)
    if args.cmd == "mvdr_lowrank":
        mvdr_lowrank_equivalence_test(
            n_trials=args.n,
            k_lr=args.k_lr,
            grid_phi=args.grid_phi,
            grid_theta=args.grid_theta,
            delta_scale=args.delta_scale,
            min_corr=args.min_corr,
            max_peak_cell_err=args.max_peak_cell_err,
            device=args.device,
        )


if __name__ == "__main__":
    _main()

