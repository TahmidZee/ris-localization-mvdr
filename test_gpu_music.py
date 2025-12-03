#!/usr/bin/env python
"""
Test script for GPU MUSIC implementation.

Run this on your GPU machine:
    python test_gpu_music.py

Expected output on V100:
- Single sample: 2-5ms (vs 15-30ms CPU)
- Batch of 100: 100-200ms (vs 2-3s CPU)
- Speedup: 10-20x
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import time

print("=" * 60)
print("GPU MUSIC Implementation Test")
print("=" * 60)

# Check CUDA
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    device = 'cuda'
else:
    print("WARNING: CUDA not available, running on CPU (will be slow)")
    device = 'cpu'

# Import
from ris_pytorch_pipeline.configs import cfg
from ris_pytorch_pipeline.music_gpu import GPUMusicEstimator, benchmark

# Create estimator
print(f"\nCreating estimator on {device}...")
estimator = GPUMusicEstimator(cfg, device=device)
print(f"Array: {estimator.N_H}x{estimator.N_V} = {estimator.N} elements")

# Generate test covariance
N = estimator.N
K = 2
np.random.seed(42)
A = np.random.randn(N, K) + 1j * np.random.randn(N, K)
R = A @ A.conj().T
R = R / np.trace(R).real * N

# ============================================================================
# Test 1: Single Sample (2D mode - fast)
# ============================================================================
print("\n" + "-" * 40)
print("Test 1: Single Sample (2D mode)")
print("-" * 40)

# Warmup
_ = estimator.estimate_single(R, K, use_2_5d=False, grid_phi=91, grid_theta=61)
if device == 'cuda':
    torch.cuda.synchronize()

# Timed run
t0 = time.time()
phi, theta, r, spectrum = estimator.estimate_single(R, K, use_2_5d=False, grid_phi=181, grid_theta=121)
if device == 'cuda':
    torch.cuda.synchronize()
t1 = time.time()

print(f"Time: {(t1-t0)*1000:.2f} ms")
print(f"Grid: 181x121 = {181*121} points")
print(f"Estimated phi: {phi}")
print(f"Estimated theta: {theta}")
print(f"Spectrum shape: {spectrum.shape}")

# ============================================================================
# Test 2: Single Sample (2.5D mode - with range search)
# ============================================================================
print("\n" + "-" * 40)
print("Test 2: Single Sample (2.5D mode)")
print("-" * 40)

t0 = time.time()
phi, theta, r, spectrum = estimator.estimate_single(R, K, use_2_5d=True, grid_phi=121, grid_theta=81,
                                                      r_planes=[0.6, 1.5, 3.0, 6.0, 9.0])
if device == 'cuda':
    torch.cuda.synchronize()
t1 = time.time()

print(f"Time: {(t1-t0)*1000:.2f} ms")
print(f"Grid: 121x81 x 5 range planes")
print(f"Estimated phi: {phi}")
print(f"Estimated theta: {theta}")
print(f"Estimated r: {r}")

# ============================================================================
# Test 3: Batch Processing
# ============================================================================
print("\n" + "-" * 40)
print("Test 3: Batch Processing (100 samples)")
print("-" * 40)

B = 100
R_batch = np.stack([R] * B)
K_batch = np.full(B, K)

# Warmup
_ = estimator.estimate_batch(R_batch[:10], K_batch[:10], use_2_5d=False, grid_phi=61, grid_theta=41)
if device == 'cuda':
    torch.cuda.synchronize()

# Timed run
t0 = time.time()
phi_batch, theta_batch, r_batch = estimator.estimate_batch(
    R_batch, K_batch, use_2_5d=False, grid_phi=91, grid_theta=61
)
if device == 'cuda':
    torch.cuda.synchronize()
t1 = time.time()

print(f"Total time: {(t1-t0)*1000:.1f} ms")
print(f"Per sample: {(t1-t0)*1000/B:.2f} ms")
print(f"Output shapes: phi={phi_batch.shape}, theta={theta_batch.shape}")

# ============================================================================
# Test 4: Newton Refinement
# ============================================================================
print("\n" + "-" * 40)
print("Test 4: Newton Refinement")
print("-" * 40)

phi_init = np.array([10.0, -15.0])
theta_init = np.array([5.0, -8.0])
r_init = np.array([3.0, 5.0])

t0 = time.time()
phi_ref, theta_ref, r_ref = estimator.newton_refine(phi_init, theta_init, r_init, R, K, max_iters=5)
if device == 'cuda':
    torch.cuda.synchronize()
t1 = time.time()

print(f"Time: {(t1-t0)*1000:.2f} ms")
print(f"Initial phi: {phi_init} -> Refined: {phi_ref}")
print(f"Initial theta: {theta_init} -> Refined: {theta_ref}")
print(f"Initial r: {r_init} -> Refined: {r_ref}")

# ============================================================================
# Benchmark (if CUDA available)
# ============================================================================
if device == 'cuda':
    print("\n" + "=" * 60)
    print("Full Benchmark: GPU vs CPU")
    print("=" * 60)
    benchmark(cfg, n_samples=100, K=2, device='cuda')

print("\n" + "=" * 60)
print("✓ All tests complete!")
print("=" * 60)

