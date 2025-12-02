#!/usr/bin/env python3
"""Quick sanity test to diagnose zero loss issue"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ris_pytorch_pipeline.train import Trainer
from ris_pytorch_pipeline.configs import cfg, mdl_cfg

print("=" * 80)
print("TRAINING SANITY TEST - Diagnosing Zero Loss Issue")
print("=" * 80)
print()

try:
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer()
    
    # Get first batch
    print("Loading first batch...")
    train_loader, val_loader = trainer._build_loaders_gpu_cache(n_train=100, n_val=100)
    batch = next(iter(train_loader))
    y, H, C, ptr, K, R_in, snr, H_full = trainer._unpack_any_batch(batch)
    
    print("\n=== INPUT DATA SANITY CHECK ===")
    print(f"y shape: {y.shape}")
    print(f"  min={y.min().item():.6f}, max={y.max().item():.6f}, mean={y.mean().item():.6f}")
    print(f"  contains NaN: {torch.isnan(y).any().item()}, contains Inf: {torch.isinf(y).any().item()}")
    
    print(f"\nH shape: {H.shape}")
    print(f"  min={H.min().item():.6f}, max={H.max().item():.6f}, mean={H.mean().item():.6f}")
    print(f"  contains NaN: {torch.isnan(H).any().item()}, contains Inf: {torch.isinf(H).any().item()}")
    
    print(f"\nR_in shape: {R_in.shape}")
    print(f"  min={R_in.min().item():.6f}, max={R_in.max().item():.6f}, mean={R_in.mean().item():.6f}")
    print(f"  contains NaN: {torch.isnan(R_in).any().item()}, contains Inf: {torch.isinf(R_in).any().item()}")
    
    print(f"\nsnr shape: {snr.shape}")
    print(f"  min={snr.min().item():.2f}, max={snr.max().item():.2f}, mean={snr.mean().item():.2f}")
    
    # Model forward pass (with gradients for R_blend construction)
    print("\n=== MODEL FORWARD PASS (with grad) ===")
    trainer.model.eval()
    preds = trainer.model(y=y, H=H, codes=C, snr_db=snr)
    
    print(f"\nModel outputs:")
    for k, v in preds.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}")
            if v.numel() > 0:
                print(f"    min={v.min().item():.6f}, max={v.max().item():.6f}, mean={v.mean().item():.6f}")
                print(f"    NaN: {torch.isnan(v).any().item()}, Inf: {torch.isinf(v).any().item()}")
    
    # Loss computation
    print("\n=== LOSS COMPUTATION ===")
    
    # Need to construct R_blend like in training loop
    print("Constructing R_blend...")
    B, N = R_in.shape[0], R_in.shape[1]
    
    # Build R_pred from model outputs (like in training)
    cov_fact_angle = preds['cov_fact_angle']  # [B, N*K_MAX]
    cov_fact_range = preds['cov_fact_range']  # [B, N*K_MAX]
    
    # Reshape to [B, N, K_MAX] and combine
    cov_fact_angle_reshaped = cov_fact_angle.reshape(B, N, -1)  # [B, N, K_MAX]
    cov_fact_range_reshaped = cov_fact_range.reshape(B, N, -1)  # [B, N, K_MAX]
    
    # Combine angle and range factors
    cov_fact_combined = cov_fact_angle_reshaped + cov_fact_range_reshaped  # [B, N, K_MAX]
    
    # Build R_pred = cov_fact @ cov_fact^H
    R_pred = torch.matmul(cov_fact_combined, cov_fact_combined.transpose(-1, -2))  # [B, N, N]
    R_pred = torch.complex(R_pred, torch.zeros_like(R_pred))  # Make complex
    
    # Build R_samp from input data (like in training)
    print("Building R_samp from input data...")
    # This is a simplified version - in real training it's more complex
    R_samp = torch.complex(torch.randn(B, N, N), torch.randn(B, N, N)) * 0.1
    
    # Blend them (beta=0.1 for testing)
    beta = 0.1
    R_blend = (1.0 - beta) * R_pred + beta * R_samp
    R_blend = 0.5 * (R_blend + R_blend.conj().transpose(-1, -2))  # Make Hermitian
    
    # Add to predictions
    preds_with_blend = preds.copy()
    preds_with_blend['R_blend'] = R_blend
    
    print(f"R_blend shape: {R_blend.shape}")
    print(f"R_blend requires_grad: {R_blend.requires_grad}")
    
    loss = trainer.loss_fn(preds_with_blend, {'R_true': R_in, 'ptr': ptr, 'K': K, 'snr_db': snr})
    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss is NaN: {torch.isnan(loss).item()}")
    print(f"Loss is Inf: {torch.isinf(loss).item()}")
    
    # Backward pass
    print("\n=== BACKWARD PASS ===")
    trainer.model.train()
    trainer.model.zero_grad()
    
    # Forward again in train mode
    preds_train = trainer.model(y=y, H=H, codes=C, snr_db=snr)
    
    # Construct R_blend for train mode too
    cov_fact_angle_train = preds_train['cov_fact_angle']
    cov_fact_range_train = preds_train['cov_fact_range']
    
    cov_fact_angle_reshaped_train = cov_fact_angle_train.reshape(B, N, -1)
    cov_fact_range_reshaped_train = cov_fact_range_train.reshape(B, N, -1)
    cov_fact_combined_train = cov_fact_angle_reshaped_train + cov_fact_range_reshaped_train
    
    R_pred_train = torch.matmul(cov_fact_combined_train, cov_fact_combined_train.transpose(-1, -2))
    R_pred_train = torch.complex(R_pred_train, torch.zeros_like(R_pred_train))
    
    R_blend_train = (1.0 - beta) * R_pred_train + beta * R_samp
    R_blend_train = 0.5 * (R_blend_train + R_blend_train.conj().transpose(-1, -2))
    
    preds_train_with_blend = preds_train.copy()
    preds_train_with_blend['R_blend'] = R_blend_train
    
    loss_train = trainer.loss_fn(preds_train_with_blend, {'R_true': R_in, 'ptr': ptr, 'K': K, 'snr_db': snr})
    
    print(f"Loss (train mode): {loss_train.item():.6f}")
    
    # Backward
    loss_train.backward()
    
    # Check gradients
    print("\n=== GRADIENT CHECK ===")
    grad_count = 0
    zero_grad_count = 0
    no_grad_count = 0
    nan_grad_count = 0
    
    for name, param in trainer.model.named_parameters():
        if param.grad is not None:
            grad_count += 1
            grad_norm = param.grad.norm().item()
            if grad_norm == 0:
                zero_grad_count += 1
            if torch.isnan(param.grad).any():
                nan_grad_count += 1
                print(f"  ❌ {name}: NaN gradient!")
        else:
            no_grad_count += 1
    
    print(f"Total parameters: {grad_count + no_grad_count}")
    print(f"  With gradients: {grad_count}")
    print(f"  Zero gradients: {zero_grad_count}")
    print(f"  No gradients: {no_grad_count}")
    print(f"  NaN gradients: {nan_grad_count}")
    
    # Sample a few gradient norms
    print("\nSample gradient norms:")
    sample_count = 0
    for name, param in trainer.model.named_parameters():
        if param.grad is not None and sample_count < 5:
            print(f"  {name}: {param.grad.norm().item():.6f}")
            sample_count += 1
    
    print("\n=== DIAGNOSIS ===")
    if loss_train.item() == 0:
        print("❌ CRITICAL: Loss is exactly 0")
        print("   Possible causes:")
        print("   1. Loss function is returning 0 regardless of input")
        print("   2. Model outputs are all zeros")
        print("   3. R_true is all zeros")
    elif torch.isnan(loss_train):
        print("❌ CRITICAL: Loss is NaN")
        print("   Possible causes:")
        print("   1. Division by zero in loss computation")
        print("   2. Invalid operations (sqrt of negative, log of zero)")
    elif grad_count == 0:
        print("❌ CRITICAL: No gradients computed")
        print("   Possible causes:")
        print("   1. All parameters detached from computational graph")
        print("   2. Loss not connected to model parameters")
    elif zero_grad_count == grad_count:
        print("❌ CRITICAL: All gradients are zero")
        print("   Possible causes:")
        print("   1. Model outputs are constant (not learning)")
        print("   2. Loss is not sensitive to model parameters")
    else:
        print("✅ Basic sanity checks passed")
        print(f"   Loss: {loss_train.item():.6f}")
        print(f"   Gradients flowing: {grad_count}/{grad_count + no_grad_count}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
except Exception as e:
    print("\n" + "=" * 80)
    print("❌ TEST CRASHED")
    print("=" * 80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

