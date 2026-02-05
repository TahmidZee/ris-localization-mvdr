#!/usr/bin/env python3
"""
Diagnostic: What is the model ACTUALLY predicting?

Check if outputs are stuck at initialization or if there's something else going on.
"""
import torch
import numpy as np
from ris_pytorch_pipeline.configs import cfg, mdl_cfg
from ris_pytorch_pipeline.model import HybridModel
from ris_pytorch_pipeline.dataset import ShardNPZDataset
from ris_pytorch_pipeline.collate_fn import collate_pad_to_kmax_with_snr

# Load model
model = HybridModel()
model.eval()

# Load 1 batch
ds = ShardNPZDataset('data_shards_M64_N256_L64/train')
loader = torch.utils.data.DataLoader(
    ds, batch_size=32, shuffle=True,
    collate_fn=lambda b: collate_pad_to_kmax_with_snr(b, cfg.K_MAX)
)
batch = next(iter(loader))

# Forward
with torch.no_grad():
    preds = model(
        y=batch['y'],
        H_full=batch['H_full'],
        codes=batch['codes'],
        snr_db=batch.get('snr_db'),
        R_samp=None,
    )

# Extract predictions
phi_soft = preds['phi_soft'].cpu().numpy()  # [B, K]
theta_soft = preds['theta_soft'].cpu().numpy()
r_soft = preds['r_soft'].cpu().numpy()
aux_power = preds['aux_power'].cpu().numpy()
aux_mask = preds['aux_mask'].cpu().numpy()

print("="*80)
print("MODEL PREDICTIONS (random initialization)")
print("="*80)
print(f"\nFirst 5 samples, all K slots:")
for i in range(min(5, phi_soft.shape[0])):
    print(f"\n--- Sample {i} (K_true={batch['K'][i].item()}) ---")
    print(f"φ (deg):  {np.rad2deg(phi_soft[i])}")
    print(f"θ (deg):  {np.rad2deg(theta_soft[i])}")
    print(f"r (m):    {r_soft[i]}")
    print(f"power:    {aux_power[i]}")
    print(f"mask:     {aux_mask[i]}")

print("\n" + "="*80)
print("STATISTICS ACROSS BATCH")
print("="*80)
print(f"φ mean (deg): {np.rad2deg(phi_soft).mean():.2f} ± {np.rad2deg(phi_soft).std():.2f}")
print(f"θ mean (deg): {np.rad2deg(theta_soft).mean():.2f} ± {np.rad2deg(theta_soft).std():.2f}")
print(f"r mean (m):   {r_soft.mean():.2f} ± {r_soft.std():.2f}")
print(f"power mean:   {aux_power.mean():.3f} ± {aux_power.std():.3f}")
print(f"mask mean:    {aux_mask.mean():.3f} ± {aux_mask.std():.3f}")

print("\n" + "="*80)
print("DIVERSITY CHECK (how different are the 5 slots?)")
print("="*80)
for i in range(min(3, phi_soft.shape[0])):
    # Within-sample slot diversity
    phi_range = np.rad2deg(phi_soft[i].max() - phi_soft[i].min())
    theta_range = np.rad2deg(theta_soft[i].max() - theta_soft[i].min())
    r_range = r_soft[i].max() - r_soft[i].min()
    print(f"Sample {i}: φ_range={phi_range:.1f}°, θ_range={theta_range:.1f}°, r_range={r_range:.2f}m")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
phi_mean = np.rad2deg(phi_soft).mean()
theta_mean = np.rad2deg(theta_soft).mean()
r_mean = r_soft.mean()

# Check if stuck at initialization (all near 0)
if abs(phi_mean) < 5.0 and np.rad2deg(phi_soft).std() < 8.0:
    print("❌ PROBLEM: φ predictions are clustered near 0° (stuck at init)")
else:
    print("✓ φ predictions have spread")

if abs(theta_mean) < 3.0 and np.rad2deg(theta_soft).std() < 5.0:
    print("❌ PROBLEM: θ predictions are clustered near 0° (stuck at init)")
else:
    print("✓ θ predictions have spread")

if 4.5 < r_mean < 5.5 and r_soft.std() < 1.5:
    print("❌ PROBLEM: r predictions are clustered near midpoint (stuck at init)")
else:
    print("✓ r predictions have spread")

# Check slot diversity
phi_diversity = np.mean([np.rad2deg(phi_soft[i].max() - phi_soft[i].min()) for i in range(phi_soft.shape[0])])
if phi_diversity < 10.0:
    print(f"❌ PROBLEM: Slots are too similar (mean φ_range={phi_diversity:.1f}° across batch)")
    print("   → Slots haven't specialized, permutation matching is unstable")
else:
    print(f"✓ Slots are differentiated (mean φ_range={phi_diversity:.1f}°)")
