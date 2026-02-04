#!/usr/bin/env python3
"""
Pipeline smoke tests (FAST) for RIS localization training.

Goal: catch "looks stuck" failures *before* long training runs.

This script validates 4 invariants:
  1) Forward outputs contain slot-head geometry in phi_soft/theta_soft/r_soft
  2) Aux-only loss produces nonzero gradients on slot-head parameters
  3) Cov-only loss produces nonzero gradients on slot-head parameters (via structured R)
  4) Single-batch overfit (aux-only) decreases loss and RMSE quickly

Run (on goose):
  python diagnose_pipeline_smoke.py --device cuda --steps 150
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import torch

from ris_pytorch_pipeline.configs import cfg, mdl_cfg, set_seed
from ris_pytorch_pipeline.dataset import ShardNPZDataset
from ris_pytorch_pipeline.collate_fn import collate_pad_to_kmax_with_snr
from ris_pytorch_pipeline.model import HybridModel
from ris_pytorch_pipeline.loss import UltimateHybridLoss, _wrap_angle


@dataclass
class SmokeResult:
    ok: bool
    name: str
    details: str


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _slot_geom_rmse_deg(pred_phi: torch.Tensor, pred_th: torch.Tensor, pred_r: torch.Tensor,
                        gt_ptr: torch.Tensor, K_true: torch.Tensor) -> Tuple[float, float, float]:
    """
    RMSE for slot-index matching against GT sorted by phi.
    - pred_*: [B,Kmax] (radians/radians/meters)
    - gt_ptr: [B, 3*Kmax] (chunked)
    - K_true: [B]
    """
    B = int(pred_phi.shape[0])
    Kmax = int(pred_phi.shape[1])
    phi_t = gt_ptr[:, :Kmax]
    th_t = gt_ptr[:, Kmax:2 * Kmax]
    r_t = gt_ptr[:, 2 * Kmax:3 * Kmax]

    phi_err2: List[float] = []
    th_err2: List[float] = []
    r_err2: List[float] = []
    for b in range(B):
        k = int(K_true[b].item())
        if k <= 0:
            continue
        gt_order = torch.argsort(phi_t[b, :k])
        gt_phi = phi_t[b, :k][gt_order]
        gt_th = th_t[b, :k][gt_order]
        gt_r = r_t[b, :k][gt_order]

        pp = pred_phi[b, :k]
        pt = pred_th[b, :k]
        pr = pred_r[b, :k]

        dphi = _wrap_angle(pp - gt_phi)
        dth = _wrap_angle(pt - gt_th)
        dr = (pr - gt_r)

        phi_err2.append(float((dphi * dphi).mean().detach().cpu()))
        th_err2.append(float((dth * dth).mean().detach().cpu()))
        r_err2.append(float((dr * dr).mean().detach().cpu()))

    if not phi_err2:
        return float("nan"), float("nan"), float("nan")
    rmse_phi_deg = math.degrees(math.sqrt(float(np.mean(phi_err2))))
    rmse_th_deg = math.degrees(math.sqrt(float(np.mean(th_err2))))
    rmse_r_m = math.sqrt(float(np.mean(r_err2)))
    return rmse_phi_deg, rmse_th_deg, rmse_r_m


def _named_grad_norms(model: torch.nn.Module) -> Dict[str, float]:
    out = {}
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        out[n] = float(p.grad.detach().norm().cpu())
    return out


def _sum_norm(d: Dict[str, float], include: Tuple[str, ...]) -> float:
    return float(sum(v for k, v in d.items() if any(tok in k for tok in include)))


def smoke_forward_and_keys(model: HybridModel, batch: Dict[str, Any], device: torch.device) -> SmokeResult:
    model.eval()
    with torch.no_grad():
        preds = model(
            batch["y"].to(device),
            batch.get("H_full", None).to(device) if batch.get("H_full", None) is not None else batch["H"].to(device),
            batch["codes"].to(device),
            snr_db=batch.get("snr_db", None).to(device) if torch.is_tensor(batch.get("snr_db", None)) else None,
            R_samp=batch.get("R_samp", None),
        )
    required = ("R_pred", "phi_theta_r", "phi_soft", "theta_soft", "r_soft", "aux_mask", "aux_power")
    missing = [k for k in required if k not in preds]
    if missing:
        return SmokeResult(False, "forward_keys", f"Missing keys in preds: {missing}")

    # Basic shape sanity
    B = int(batch["y"].shape[0])
    Kmax = int(getattr(cfg, "K_MAX", 5))
    if tuple(preds["phi_soft"].shape) != (B, Kmax):
        return SmokeResult(False, "forward_shapes", f"phi_soft shape {tuple(preds['phi_soft'].shape)} != {(B, Kmax)}")
    if tuple(preds["theta_soft"].shape) != (B, Kmax):
        return SmokeResult(False, "forward_shapes", f"theta_soft shape {tuple(preds['theta_soft'].shape)} != {(B, Kmax)}")
    if tuple(preds["r_soft"].shape) != (B, Kmax):
        return SmokeResult(False, "forward_shapes", f"r_soft shape {tuple(preds['r_soft'].shape)} != {(B, Kmax)}")

    return SmokeResult(True, "forward_keys", "OK (slot head geom exposed via phi_soft/theta_soft/r_soft)")


def smoke_grad_connectivity(model: HybridModel, loss_fn: UltimateHybridLoss, batch: Dict[str, Any],
                           device: torch.device, *, mode: str) -> SmokeResult:
    """
    mode:
      - "aux_only": lam_aux=1.0, lam_cov=0.0
      - "cov_only": lam_aux=0.0, lam_cov=1.0
    """
    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    # configure loss
    loss_fn.train()
    loss_fn.lam_cross = 0.0
    loss_fn.lam_gap = 0.0
    loss_fn.lam_margin = 0.0
    loss_fn.lam_peak = 0.0
    loss_fn.lam_subspace_align = 0.0
    loss_fn.lam_peak_contrast = 0.0
    loss_fn.lam_cov_pred = 0.0
    loss_fn.set_mask_loss_scale(0.0)

    if mode == "aux_only":
        loss_fn.lam_aux = 1.0
        loss_fn.lam_cov = 0.0
    elif mode == "cov_only":
        loss_fn.lam_aux = 0.0
        loss_fn.lam_cov = 1.0
    else:
        raise ValueError(mode)

    preds = model(
        batch["y"].to(device),
        batch.get("H_full", None).to(device) if batch.get("H_full", None) is not None else batch["H"].to(device),
        batch["codes"].to(device),
        snr_db=batch.get("snr_db", None).to(device) if torch.is_tensor(batch.get("snr_db", None)) else None,
        R_samp=batch.get("R_samp", None),
    )
    labels = {
        "R_true": batch["R_true"].to(device),
        "ptr": batch["ptr"].to(device),
        "K": batch["K"].to(device),
        "snr_db": batch.get("snr_db", None).to(device) if torch.is_tensor(batch.get("snr_db", None)) else batch.get("snr_db", None),
    }
    loss = loss_fn(preds, labels)
    loss.backward()

    gn = _named_grad_norms(model)
    slot_sum = _sum_norm(gn, ("slot_", "slot_head", "slot_queries"))
    grid_sum = _sum_norm(gn, ("phi_logits", "theta_logits", "logits_gg", "soft_argmax"))

    if slot_sum <= 0.0 or not np.isfinite(slot_sum):
        return SmokeResult(False, f"grad_{mode}", f"slot grad sum is {slot_sum:.3e} (expected > 0). grid_sum={grid_sum:.3e}")
    return SmokeResult(True, f"grad_{mode}", f"OK slot_grad_sum={slot_sum:.3e} (grid_grad_sum={grid_sum:.3e})")


def smoke_overfit_single_batch(model: HybridModel, loss_fn: UltimateHybridLoss, batch: Dict[str, Any],
                              device: torch.device, *, steps: int = 200, lr: float = 1e-3) -> SmokeResult:
    model.train()
    loss_fn.train()

    # aux-only optimization (fast, isolates geometry learning)
    loss_fn.lam_aux = 1.5
    loss_fn.lam_cov = 0.0
    loss_fn.lam_cross = 0.0
    loss_fn.lam_gap = 0.0
    loss_fn.lam_margin = 0.0
    loss_fn.lam_peak = 0.0
    loss_fn.lam_subspace_align = 0.0
    loss_fn.lam_peak_contrast = 0.0
    loss_fn.lam_cov_pred = 0.0
    loss_fn.set_mask_loss_scale(0.0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    labels = {
        "R_true": batch["R_true"].to(device),
        "ptr": batch["ptr"].to(device),
        "K": batch["K"].to(device),
        "snr_db": batch.get("snr_db", None).to(device) if torch.is_tensor(batch.get("snr_db", None)) else batch.get("snr_db", None),
    }

    # baseline
    with torch.no_grad():
        preds0 = model(
            batch["y"].to(device),
            batch.get("H_full", None).to(device) if batch.get("H_full", None) is not None else batch["H"].to(device),
            batch["codes"].to(device),
            snr_db=batch.get("snr_db", None).to(device) if torch.is_tensor(batch.get("snr_db", None)) else None,
            R_samp=batch.get("R_samp", None),
        )
        loss0 = float(loss_fn(preds0, labels).detach().cpu())
        rm0 = _slot_geom_rmse_deg(preds0["phi_soft"], preds0["theta_soft"], preds0["r_soft"], labels["ptr"], labels["K"])

    # train
    last = loss0
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        preds = model(
            batch["y"].to(device),
            batch.get("H_full", None).to(device) if batch.get("H_full", None) is not None else batch["H"].to(device),
            batch["codes"].to(device),
            snr_db=batch.get("snr_db", None).to(device) if torch.is_tensor(batch.get("snr_db", None)) else None,
            R_samp=batch.get("R_samp", None),
        )
        loss = loss_fn(preds, labels)
        if not torch.isfinite(loss):
            return SmokeResult(False, "overfit", f"Non-finite loss at step {it}: {loss}")
        loss.backward()
        opt.step()
        last = float(loss.detach().cpu())

    with torch.no_grad():
        preds1 = model(
            batch["y"].to(device),
            batch.get("H_full", None).to(device) if batch.get("H_full", None) is not None else batch["H"].to(device),
            batch["codes"].to(device),
            snr_db=batch.get("snr_db", None).to(device) if torch.is_tensor(batch.get("snr_db", None)) else None,
            R_samp=batch.get("R_samp", None),
        )
        loss1 = float(loss_fn(preds1, labels).detach().cpu())
        rm1 = _slot_geom_rmse_deg(preds1["phi_soft"], preds1["theta_soft"], preds1["r_soft"], labels["ptr"], labels["K"])

    # Pass criteria: we just want "it learns at all" (robust to noise).
    # Either a noticeable loss drop OR a noticeable phi RMSE drop is acceptable.
    # (The loss can be a bit non-monotone early depending on initialization.)
    loss_drop_ok = (loss1 < 0.90 * loss0)
    phi_drop_ok = (np.isfinite(rm0[0]) and np.isfinite(rm1[0]) and (rm1[0] < (rm0[0] - 5.0)))
    if not (loss_drop_ok or phi_drop_ok):
        return SmokeResult(
            False,
            "overfit",
            f"loss did not drop enough: {loss0:.4f} → {loss1:.4f}. "
            f"RMSE φ/θ/r: {rm0[0]:.2f}/{rm0[1]:.2f}/{rm0[2]:.2f} → {rm1[0]:.2f}/{rm1[1]:.2f}/{rm1[2]:.2f}",
        )
    return SmokeResult(
        True,
        "overfit",
        f"OK loss {loss0:.4f} → {loss1:.4f}. RMSE φ/θ/r: {rm0[0]:.2f}/{rm0[1]:.2f}/{rm0[2]:.2f} → {rm1[0]:.2f}/{rm1[1]:.2f}/{rm1[2]:.2f}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train_dir", default=str(getattr(cfg, "DATA_SHARDS_TRAIN", f"data_shards_M{cfg.M}_N{cfg.N}_L{cfg.L}/train")))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=int(getattr(mdl_cfg, "SEED", 42)))
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Load 1 batch (fast) using the real training shard format.
    ds = ShardNPZDataset(args.train_dir)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda b: collate_pad_to_kmax_with_snr(b, int(getattr(cfg, "K_MAX", 5))),
    )
    batch = next(iter(loader))
    batch = _to_device(batch, device)

    # Build model + loss
    model = HybridModel().to(device)
    loss_fn = UltimateHybridLoss().to(device)

    results: List[SmokeResult] = []
    results.append(smoke_forward_and_keys(model, batch, device))
    results.append(smoke_grad_connectivity(model, loss_fn, batch, device, mode="aux_only"))
    results.append(smoke_grad_connectivity(model, loss_fn, batch, device, mode="cov_only"))
    results.append(smoke_overfit_single_batch(model, loss_fn, batch, device, steps=int(args.steps)))

    ok_all = all(r.ok for r in results)
    print("\n" + "=" * 80)
    print("PIPELINE SMOKE RESULTS:", "PASS ✅" if ok_all else "FAIL ❌")
    print("=" * 80)
    for r in results:
        tag = "PASS" if r.ok else "FAIL"
        print(f"[{tag:4s}] {r.name:14s} :: {r.details}")
    print("=" * 80)

    # Exit code for automation
    raise SystemExit(0 if ok_all else 2)


if __name__ == "__main__":
    main()

