# ris_pytorch_pipeline/train_nfssn_from_features.py
# SPDX-License-Identifier: MIT
import math, argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from .feature_dataset import RxFeatureDataset
from .configs import cfg

# Import from AI-Subspace-Methods (ensure import path is available)
from ai_subspace_methods.src.models_pack.subspacenet import SubspaceNet
from ai_subspace_methods.src.system_model import SystemModel

def build_nfssn(cfg):
    sm = SystemModel(
        number_sensors=cfg.N_H * cfg.N_V,
        sensor_distance_h=cfg.d_H,
        sensor_distance_v=cfg.d_V,
        wavelength=cfg.WAVEL,
        field_type="near",
    )
    model = SubspaceNet(
        tau=getattr(cfg, "TAU", 12),
        diff_method="root_music",
        system_model=sm,
        field_type="near",
        regularization=None,
        variant="small",
        norm_layer=True,
        psd_epsilon=1e-6,
        batch_norm=False,
    )
    return model

def _ensure_batched(x):
    """Ensure tensors are [B, ...]. If 1D, add batch dim."""
    if isinstance(x, torch.Tensor) and x.ndim == 1:
        return x.unsqueeze(0)
    return x

def mse_triplet_per_sample(pred_phi, pred_th, pred_r, gt_phi, gt_th, gt_r, K):
    """
    pred_*, gt_*: [B, >=K] (e.g., [B, K_MAX]) after padding or model output
    K: LongTensor [B] with true source counts
    Returns scalar loss averaged over valid samples (K>0).
    """
    B = K.shape[0]
    loss = pred_phi.new_tensor(0.0)
    valid = 0
    for b in range(B):
        k = int(K[b].item())
        if k <= 0:
            continue
        k_eff = min(k, pred_phi.shape[-1], gt_phi.shape[-1])
        loss = loss + ((pred_phi[b, :k_eff] - gt_phi[b, :k_eff])**2).mean()
        loss = loss + ((pred_th [b, :k_eff] - gt_th [b, :k_eff])**2).mean()
        loss = loss + ((pred_r  [b, :k_eff] - gt_r  [b, :k_eff])**2).mean()
        valid += 1
    if valid == 0:
        return loss
    return loss / valid

def _forward_predict(model, Rx, K):
    """
    Calls model and returns (pred_phi, pred_th, pred_r) as float tensors with a batch dim.
    SubspaceNet.forward(x, sources_num=K)
    """
    out = model(Rx, sources_num=K)
    if not (isinstance(out, tuple) and len(out) >= 3):
        raise RuntimeError("SubspaceNet forward did not return (phi, theta, r)")
    pphi, pth, pr = out[0], out[1], out[2]
    pphi = _ensure_batched(pphi)
    pth  = _ensure_batched(pth)
    pr   = _ensure_batched(pr)
    return pphi.float(), pth.float(), pr.float()

def train_epoch(model, loader, opt, device):
    model.train(); tot = 0.0; n = 0
    for b in loader:
        Rx = b["Rx"].to(device).float()        # [B, C, L, L] or [C, L, L]
        K  = b["K"].to(device).long()          # [B]
        gt_phi = b["phi"].to(device).float()   # [B, K_MAX] (padded)
        gt_th  = b["theta"].to(device).float() # [B, K_MAX]
        gt_r   = b["r"].to(device).float()     # [B, K_MAX]

        if Rx.ndim == 3:  # [C,L,L] -> [1,C,L,L]
            Rx = Rx.unsqueeze(0)
            K = K.view(1)
            gt_phi = gt_phi.unsqueeze(0)
            gt_th  = gt_th.unsqueeze(0)
            gt_r   = gt_r.unsqueeze(0)

        opt.zero_grad()
        pred_phi, pred_th, pred_r = _forward_predict(model, Rx, K)
        loss = mse_triplet_per_sample(pred_phi, pred_th, pred_r, gt_phi, gt_th, gt_r, K)
        loss.backward()
        opt.step()

        tot += float(loss.item()); n += 1
    return tot / max(1, n)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval(); tot = 0.0; n = 0
    for b in loader:
        Rx = b["Rx"].to(device).float()
        K  = b["K"].to(device).long()
        gt_phi = b["phi"].to(device).float()
        gt_th  = b["theta"].to(device).float()
        gt_r   = b["r"].to(device).float()

        if Rx.ndim == 3:
            Rx = Rx.unsqueeze(0)
            K = K.view(1)
            gt_phi = gt_phi.unsqueeze(0)
            gt_th  = gt_th.unsqueeze(0)
            gt_r   = gt_r.unsqueeze(0)

        pred_phi, pred_th, pred_r = _forward_predict(model, Rx, K)
        loss = mse_triplet_per_sample(pred_phi, pred_th, pred_r, gt_phi, gt_th, gt_r, K)

        tot += float(loss.item()); n += 1
    return tot / max(1, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_final/baselines/features_dcd_nf")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_dir", default="checkpoints/nfsubspacenet")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pin-memory", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = RxFeatureDataset("train", root=args.root, use_labels=True)
    va = RxFeatureDataset("val",   root=args.root, use_labels=True)
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True,
                    num_workers=args.num_workers, drop_last=False, pin_memory=args.pin_memory)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False,
                    num_workers=args.num_workers, drop_last=False, pin_memory=args.pin_memory)

    model = build_nfssn(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = math.inf
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.save_dir) / "nfsubspacenet.pt"

    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, tl, opt, device)
        va_loss = eval_epoch(model, vl, device)
        print(f"[NFSSN] epoch {ep:03d}  train {tr_loss:.4f}  val {va_loss:.4f}")
        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(), "cfg": dict(
                N_H=cfg.N_H, N_V=cfg.N_V, d_H=cfg.d_H, d_V=cfg.d_V,
                WAVEL=cfg.WAVEL, K_MAX=cfg.K_MAX, TAU=getattr(cfg, 'TAU', 12)
            )}, ckpt)
            print("  ✓ saved", ckpt)

if __name__ == "__main__":
    main()
