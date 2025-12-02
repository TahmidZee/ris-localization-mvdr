#!/usr/bin/env python3
"""
Quick verification script to test the lam_cov fix
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ris_pytorch_pipeline.train import Trainer
from ris_pytorch_pipeline.configs import cfg, mdl_cfg

def main():
    print("=" * 60)
    print("VERIFYING LAM_COV FIX")
    print("=" * 60)
    print()
    
    try:
        # Initialize trainer
        print("1. Initializing trainer...")
        trainer = Trainer()
        
        # Check loss weights
        print("2. Checking loss weights:")
        print(f"   lam_cov: {trainer.loss_fn.lam_cov}")
        print(f"   lam_cov_pred: {trainer.loss_fn.lam_cov_pred}")
        print(f"   lam_subspace_align: {trainer.loss_fn.lam_subspace_align}")
        print(f"   lam_peak_contrast: {trainer.loss_fn.lam_peak_contrast}")
        
        # Verify lam_cov is not zero
        if trainer.loss_fn.lam_cov == 0:
            print("❌ CRITICAL: lam_cov is still 0!")
            return 1
        elif trainer.loss_fn.lam_cov < 0.1:
            print(f"⚠️  WARNING: lam_cov is very small ({trainer.loss_fn.lam_cov})")
            return 1
        else:
            print(f"✅ GOOD: lam_cov = {trainer.loss_fn.lam_cov}")
        
        # Test a single forward pass
        print("3. Testing single forward pass...")
        train_loader, val_loader = trainer._build_loaders_gpu_cache(n_train=32, n_val=32)
        batch = next(iter(train_loader))
        y, H, C, ptr, K, R_in, snr, H_full = trainer._unpack_any_batch(batch)
        
        # Model forward
        with torch.no_grad():
            preds = trainer.model(y=y, H=H, codes=C, snr_db=snr)
        
        # Loss computation
        loss = trainer.loss_fn(preds, {'R_true': R_in, 'ptr': ptr, 'K': K, 'snr_db': snr})
        
        print(f"   Loss value: {loss.item():.6f}")
        
        if loss.item() == 0:
            print("❌ CRITICAL: Loss is still 0!")
            return 1
        elif torch.isnan(loss):
            print("❌ CRITICAL: Loss is NaN!")
            return 1
        elif loss.item() < 0.1:
            print(f"⚠️  WARNING: Loss is very small ({loss.item():.6f})")
        else:
            print(f"✅ GOOD: Loss = {loss.item():.6f} (reasonable value)")
        
        # Test backward pass
        print("4. Testing backward pass...")
        trainer.model.train()
        trainer.model.zero_grad()
        
        preds_train = trainer.model(y=y, H=H, codes=C, snr_db=snr)
        loss_train = trainer.loss_fn(preds_train, {'R_true': R_in, 'ptr': ptr, 'K': K, 'snr_db': snr})
        
        loss_train.backward()
        
        # Check gradients
        grad_count = 0
        zero_grad_count = 0
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                if param.grad.norm().item() == 0:
                    zero_grad_count += 1
        
        print(f"   Parameters with gradients: {grad_count}")
        print(f"   Zero gradients: {zero_grad_count}")
        
        if grad_count == 0:
            print("❌ CRITICAL: No gradients computed!")
            return 1
        elif zero_grad_count == grad_count:
            print("❌ CRITICAL: All gradients are zero!")
            return 1
        else:
            print(f"✅ GOOD: {grad_count - zero_grad_count} parameters have non-zero gradients")
        
        print()
        print("=" * 60)
        print("✅ VERIFICATION PASSED")
        print("   → lam_cov fix is working")
        print("   → Loss computation is working")
        print("   → Gradients are flowing")
        print("   → Ready for overfit test")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ VERIFICATION FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())


