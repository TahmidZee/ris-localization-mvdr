#!/usr/bin/env python3
"""
Test if the K-head vs MUSIC difference is PURELY due to shrinkage.
"""

import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ris_pytorch_pipeline.configs import cfg, mdl_cfg
from ris_pytorch_pipeline.model import HybridModel
from ris_pytorch_pipeline.dataset import ShardNPZDataset
from ris_pytorch_pipeline.covariance_utils import build_effective_cov_torch, build_effective_cov_np

# Load sample
shard_dir = Path("data_shards_M64_L16/val")
ds = ShardNPZDataset(shard_dir)
sample = ds[0]

model = HybridModel()
model.eval()
device = torch.device("cpu")
model = model.to(device)

y = sample["y"].unsqueeze(0).to(device)
H = sample["H"].unsqueeze(0).to(device)
codes = sample["codes"].unsqueeze(0).to(device)
snr_db = sample["snr"].unsqueeze(0).to(device)

# Build R_samp from H_full
from ris_pytorch_pipeline.angle_pipeline import build_sample_covariance_from_snapshots

y_np = y[0].numpy()
codes_np = codes[0].numpy()
y_cplx = y_np[..., 0] + 1j * y_np[..., 1]
codes_cplx = codes_np[..., 0] + 1j * codes_np[..., 1]

H_full_np = sample["H_full"].numpy()
H_full_cplx = H_full_np[..., 0] + 1j * H_full_np[..., 1]
H_for_rsamp = np.repeat(H_full_cplx[None, :, :], y_cplx.shape[0], axis=0)

R_samp_np = build_sample_covariance_from_snapshots(
    y_cplx, H_for_rsamp, codes_cplx, cfg, tikhonov_alpha=1e-3
)
R_samp_np = 0.5 * (R_samp_np + R_samp_np.conj().T)
R_samp_t = torch.from_numpy(R_samp_np).to(torch.complex64).unsqueeze(0).to(device)

# Forward pass
with torch.no_grad():
    preds = model(y, H, codes, snr_db=snr_db, R_samp=R_samp_t)

cf_ang = preds["cov_fact_angle"][0].detach().cpu().numpy()
cf_cplx = (cf_ang[::2] + 1j * cf_ang[1::2]).reshape(cfg.N, cfg.K_MAX)
R_pred_np = cf_cplx @ cf_cplx.conj().T

R_pred_t = torch.from_numpy(R_pred_np).to(torch.complex64).unsqueeze(0).to(device)
beta = float(getattr(cfg, "HYBRID_COV_BETA", 0.0))

# Build R_eff with and without shrink
print("=" * 80)
print("K-HEAD (with shrink):")
R_khead_shrink = build_effective_cov_torch(
    R_pred_t, snr_db=snr_db, R_samp=R_samp_t, beta=beta,
    diag_load=True, apply_shrink=True, target_trace=float(cfg.N),
)[0].detach().cpu().numpy()
print(f"  trace = {np.trace(R_khead_shrink).real:.4f}")
print(f"  evals[:5] = {np.sort(np.linalg.eigvalsh(R_khead_shrink).real)[::-1][:5]}")

print("\nK-HEAD (no shrink):")
R_khead_noshrink = build_effective_cov_torch(
    R_pred_t, snr_db=None, R_samp=R_samp_t, beta=beta,
    diag_load=True, apply_shrink=False, target_trace=float(cfg.N),
)[0].detach().cpu().numpy()
print(f"  trace = {np.trace(R_khead_noshrink).real:.4f}")
print(f"  evals[:5] = {np.sort(np.linalg.eigvalsh(R_khead_noshrink).real)[::-1][:5]}")

print("\nMUSIC (no shrink):")
R_music = build_effective_cov_np(
    R_pred_np, R_samp=R_samp_np, beta=beta,
    diag_load=True, apply_shrink=False, snr_db=None, target_trace=float(cfg.N),
)
print(f"  trace = {np.trace(R_music).real:.4f}")
print(f"  evals[:5] = {np.sort(np.linalg.eigvalsh(R_music).real)[::-1][:5]}")

print("\n" + "=" * 80)
print("COMPARISON (no shrink vs no shrink):")
diff_noshrink = np.linalg.norm(R_khead_noshrink - R_music, 'fro')
print(f"||R_khead_noshrink - R_music||_F = {diff_noshrink:.6e}")
print(f"Relative = {diff_noshrink / np.linalg.norm(R_music, 'fro'):.6e}")

if diff_noshrink < 1e-5:
    print("✓ PERFECT MATCH when both paths skip shrink!")
    print("  → The mismatch is PURELY due to K-head applying shrink and MUSIC not.")
    sys.exit(0)
else:
    print("✗ Still mismatched even without shrink. Deeper investigation needed.")
    sys.exit(1)




