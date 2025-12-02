#!/usr/bin/env python3
"""Quick angle parsing verification (no dataset loading)"""
import numpy as np
from ris_pytorch_pipeline.eval_angles import eval_scene_angles_ranges

print("=" * 80)
print("ANGLE PARSING VERIFICATION (Unit Tests)")
print("=" * 80)

# Test 1: Perfect match
print("\nTEST 1: Perfect match (should give 0 error)")
phi_gt = np.array([45.0, -30.0])
theta_gt = np.array([10.0, -15.0])
r_gt = np.array([2.5, 3.2])

m1 = eval_scene_angles_ranges(phi_gt, theta_gt, r_gt, phi_gt, theta_gt, r_gt)
print(f"  φ: {m1['med_phi']:.4f}° (expect ~0)")
print(f"  θ: {m1['med_theta']:.4f}° (expect ~0)")
print(f"  r: {m1['med_r']:.4f}m (expect ~0)")
assert m1['med_phi'] < 0.01 and m1['med_theta'] < 0.01, "❌ Perfect match failed!"
print("  ✅ PASS")

# Test 2: Swapped order (Hungarian should fix)
print("\nTEST 2: Swapped order (Hungarian should match)")
phi_pred = phi_gt[::-1]
theta_pred = theta_gt[::-1]
r_pred = r_gt[::-1]

m2 = eval_scene_angles_ranges(phi_pred, theta_pred, r_pred, phi_gt, theta_gt, r_gt)
print(f"  φ: {m2['med_phi']:.4f}° (expect ~0)")
print(f"  θ: {m2['med_theta']:.4f}° (expect ~0)")
print(f"  r: {m2['med_r']:.4f}m (expect ~0)")
assert m2['med_phi'] < 0.01 and m2['med_theta'] < 0.01, "❌ Hungarian failed!"
print("  ✅ PASS")

# Test 3: Small perturbation
print("\nTEST 3: Small perturbation (±2°, ±0.1m)")
phi_pred = phi_gt + np.array([2.0, -2.0])
theta_pred = theta_gt + np.array([-2.0, 2.0])
r_pred = r_gt + np.array([0.1, -0.1])

m3 = eval_scene_angles_ranges(phi_pred, theta_pred, r_pred, phi_gt, theta_gt, r_gt)
print(f"  φ: {m3['med_phi']:.4f}° (expect ~2)")
print(f"  θ: {m3['med_theta']:.4f}° (expect ~2)")
print(f"  r: {m3['med_r']:.4f}m (expect ~0.1)")
assert 1.5 < m3['med_phi'] < 2.5 and 1.5 < m3['med_theta'] < 2.5, "❌ Perturbation unexpected!"
print("  ✅ PASS")

# Test 4: Wrap-around (single source, 170° vs -175°)
print("\nTEST 4: Wrap-around (φ near ±180°)")
phi_pred = np.array([170.0])
phi_gt_wrap = np.array([-175.0])
theta_test = np.array([0.0])
r_test = np.array([2.0])

m4 = eval_scene_angles_ranges(phi_pred, theta_test, r_test, phi_gt_wrap, theta_test, r_test)
print(f"  φ: {m4['med_phi']:.4f}° (expect ~15°, circular distance)")
# 170 - (-175) = 345 → wrap → -15 → abs → 15
assert 13 < m4['med_phi'] < 17, f"❌ Circular distance broken! Got {m4['med_phi']:.2f}°"
print("  ✅ PASS")

print("\n" + "=" * 80)
print("✅ ALL ANGLE PARSING TESTS PASSED!")
print("=" * 80)
print("\nHungarian matching:")
print("  ✓ Uses correct units (degrees)")
print("  ✓ Handles swapped order correctly")
print("  ✓ Applies circular distance for φ")
print("\nIf seeing 30°/15° errors, root cause is model output, NOT parsing.")

