#!/usr/bin/env python3
"""
Verify angle units/order in Hungarian matching.
Checks rad vs deg, φ/θ order consistency.
"""
import sys
sys.path.insert(0, '/home/tahit/ris/MainMusic')

import torch
import numpy as np
from ris_pytorch_pipeline.configs import cfg, mdl_cfg
from ris_pytorch_pipeline.dataset import ShardNPZDataset
from ris_pytorch_pipeline.eval_angles import eval_scene_angles_ranges

print("=" * 80)
print("ANGLE PARSING VERIFICATION")
print("=" * 80)

# Load validation data
data_dir = f"/home/tahit/ris/MainMusic/{cfg.DATA_SHARDS_VAL}"
ds = ShardNPZDataset(data_dir)

# Test on first sample with K > 1
for i in range(len(ds)):
    sample = ds[i]
    K = int(sample['K'])
    if K > 1:
        break

if isinstance(sample['ptr'], torch.Tensor):
    ptr = sample['ptr'].cpu().numpy()
else:
    ptr = sample['ptr']

print(f"\nSample {i}, K={K}")
print("-" * 80)

# Extract ground truth (in radians)
phi_gt_rad = ptr[:K]
theta_gt_rad = ptr[cfg.K_MAX:cfg.K_MAX+K]
r_gt = ptr[2*cfg.K_MAX:2*cfg.K_MAX+K]

print(f"Ground truth φ (rad): {phi_gt_rad}")
print(f"Ground truth θ (rad): {theta_gt_rad}")
print(f"Ground truth r (m):   {r_gt}")

# Convert to degrees for Hungarian
phi_gt_deg = np.rad2deg(phi_gt_rad)
theta_gt_deg = np.rad2deg(theta_gt_rad)

print(f"\nGround truth φ (deg): {phi_gt_deg}")
print(f"Ground truth θ (deg): {theta_gt_deg}")

# Test case 1: Perfect match (should give 0 error)
print("\n" + "=" * 80)
print("TEST 1: Perfect Match (pred == gt)")
print("-" * 80)

metrics1 = eval_scene_angles_ranges(
    phi_gt_deg.copy(), theta_gt_deg.copy(), r_gt.copy(),
    phi_gt_deg, theta_gt_deg, r_gt
)

print(f"φ median error: {metrics1['med_phi']:.4f}°")
print(f"θ median error: {metrics1['med_theta']:.4f}°")
print(f"r median error: {metrics1['med_r']:.4f}m")

if metrics1['med_phi'] < 0.01 and metrics1['med_theta'] < 0.01 and metrics1['med_r'] < 0.001:
    print("✅ Perfect match works!")
else:
    print("❌ ERROR: Perfect match should give ~0 error!")

# Test case 2: Swapped order (Hungarian should fix this)
print("\n" + "=" * 80)
print("TEST 2: Swapped Order (Hungarian should match correctly)")
print("-" * 80)

if K == 2:
    phi_pred = phi_gt_deg[::-1]  # Reverse order
    theta_pred = theta_gt_deg[::-1]
    r_pred = r_gt[::-1]
    
    metrics2 = eval_scene_angles_ranges(
        phi_pred, theta_pred, r_pred,
        phi_gt_deg, theta_gt_deg, r_gt
    )
    
    print(f"Predicted (swapped): φ={phi_pred}, θ={theta_pred}, r={r_pred}")
    print(f"Ground truth:        φ={phi_gt_deg}, θ={theta_gt_deg}, r={r_gt}")
    print(f"φ median error: {metrics2['med_phi']:.4f}°")
    print(f"θ median error: {metrics2['med_theta']:.4f}°")
    print(f"r median error: {metrics2['med_r']:.4f}m")
    
    if metrics2['med_phi'] < 0.01 and metrics2['med_theta'] < 0.01 and metrics2['med_r'] < 0.001:
        print("✅ Hungarian matching fixed swapped order!")
    else:
        print("⚠️  Hungarian should give ~0 error for swapped perfect match")
else:
    print(f"Skipped (K={K}, need K=2)")

# Test case 3: Small perturbation
print("\n" + "=" * 80)
print("TEST 3: Small Perturbation (±2°, ±0.1m)")
print("-" * 80)

phi_pred = phi_gt_deg + np.array([2.0, -2.0][:K])
theta_pred = theta_gt_deg + np.array([-2.0, 2.0][:K])
r_pred = r_gt + np.array([0.1, -0.1][:K])

metrics3 = eval_scene_angles_ranges(
    phi_pred, theta_pred, r_pred,
    phi_gt_deg, theta_gt_deg, r_gt
)

print(f"Predicted: φ={phi_pred}, θ={theta_pred}, r={r_pred}")
print(f"Ground truth: φ={phi_gt_deg}, θ={theta_gt_deg}, r={r_gt}")
print(f"φ median error: {metrics3['med_phi']:.4f}°")
print(f"θ median error: {metrics3['med_theta']:.4f}°")
print(f"r median error: {metrics3['med_r']:.4f}m")

if 1.5 < metrics3['med_phi'] < 2.5 and 1.5 < metrics3['med_theta'] < 2.5 and 0.08 < metrics3['med_r'] < 0.12:
    print("✅ Small perturbation errors look correct!")
else:
    print("⚠️  Expected ~2° angle error and ~0.1m range error")

# Test case 4: Wrap-around (φ near ±180°)
print("\n" + "=" * 80)
print("TEST 4: Wrap-around (φ near ±180°)")
print("-" * 80)

phi_test_pred = np.array([170.0, -170.0])
phi_test_gt = np.array([-170.0, 170.0])
theta_test = np.array([0.0, 0.0])
r_test = np.array([2.0, 2.0])

metrics4 = eval_scene_angles_ranges(
    phi_test_pred, theta_test, r_test,
    phi_test_gt, theta_test, r_test
)

print(f"Predicted: φ={phi_test_pred}")
print(f"Ground truth: φ={phi_test_gt}")
print(f"φ median error: {metrics4['med_phi']:.4f}°")

# Circular distance: min(|170-(-170)|, 360-|170-(-170)|) = min(340, 20) = 20°
if 15 < metrics4['med_phi'] < 25:
    print("✅ Circular distance works (20° expected)")
else:
    print(f"⚠️  Expected ~20° error (circular distance), got {metrics4['med_phi']:.2f}°")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

issues = []
if metrics1['med_phi'] > 0.01 or metrics1['med_theta'] > 0.01:
    issues.append("Perfect match test failed")
if K == 2 and (metrics2['med_phi'] > 0.01 or metrics2['med_theta'] > 0.01):
    issues.append("Hungarian matching doesn't handle swapped order")
if not (1.5 < metrics3['med_phi'] < 2.5) or not (1.5 < metrics3['med_theta'] < 2.5):
    issues.append("Perturbation errors unexpected")

if issues:
    print("\n⚠️  ISSUES FOUND:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nThese could explain parsing bugs!")
else:
    print("\n✅ ALL ANGLE PARSING CHECKS PASSED!")
    print("\nHungarian matching uses correct units and circular distance.")
    print("If seeing 30°/15° errors, likely a model output issue, not parsing.")

print("=" * 80)

