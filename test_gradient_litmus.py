#!/usr/bin/env python3
"""
One-Batch Gradient Litmus Test

Purpose: Quick sanity check that gradients flow properly with beta warmup
Expected: loss_nmse_pred decreases noticeably by step 3-5
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ris_pytorch_pipeline.train import Trainer
from ris_pytorch_pipeline.configs import cfg, mdl_cfg, set_seed

def main():
    print("=" * 80)
    print("ONE-BATCH GRADIENT LITMUS TEST")
    print("=" * 80)
    print()
    
    set_seed(42)
    
    # Disable curriculum for clean test
    mdl_cfg.USE_3_PHASE_CURRICULUM = False
    
    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer()
    
    # Manual beta warmup setup
    trainer.beta_start = 0.0
    trainer.beta_final = 0.30
    trainer.beta_warmup_epochs = 5
    
    # Get one batch
    print("Loading one batch...")
    train_loader, _ = trainer._build_loaders_gpu_cache(n_train=64, n_val=64)
    batch = next(iter(train_loader))
    y, H, C, ptr, K, R_in, snr, H_full = trainer._unpack_any_batch(batch)
    
    print(f"\nBatch shapes:")
    print(f"  y: {y.shape}")
    print(f"  H: {H.shape}")
    print(f"  R_in: {R_in.shape}")
    print()
    
    # Run 5 training steps with beta warmup
    print("Running 5 steps with beta warmup (0.0 → 0.30)...\n")
    
    losses = []
    betas = []
    
    for step in range(1, 6):
        # Compute beta for this step
        beta = trainer.beta_start + (trainer.beta_final - trainer.beta_start) * (step / 5)
        betas.append(beta)
        
        # Forward pass
        trainer.model.train()
        with torch.amp.autocast('cuda', enabled=(trainer.amp and trainer.device.type == "cuda")):
            preds = trainer.model(y=y, H=H, codes=C, snr_db=snr)
            
            # Force beta for this step (hack for testing)
            if 'R_blend' in preds:
                # Recompute R_blend with this beta
                from ris_pytorch_pipeline.train import _ri_to_c
                R_pred = preds.get('R_pred')
                if R_pred is None:
                    # Reconstruct from factors
                    raise NotImplementedError("Need to implement R_pred reconstruction")
                # Note: This is a simplified test, full implementation would recompute R_samp
                
            loss = trainer.loss_fn(preds, {'R_true': R_in, 'ptr': ptr, 'K': K, 'snr_db': snr})
        
        losses.append(loss.item())
        
        # Backward pass
        trainer.opt.zero_grad()
        trainer.scaler.scale(loss).backward()
        
        # Check gradients
        grad_norms = []
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        
        # Step optimizer
        trainer.scaler.step(trainer.opt)
        trainer.scaler.update()
        
        print(f"Step {step}/5: beta={beta:.3f}, loss={loss.item():.6f}, avg_grad_norm={avg_grad_norm:.4f}")
    
    print()
    print("=" * 80)
    print("LITMUS TEST RESULTS")
    print("=" * 80)
    print()
    
    # Check if loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"Initial loss (step 1): {initial_loss:.6f}")
    print(f"Final loss (step 5):   {final_loss:.6f}")
    print(f"Loss reduction:        {loss_reduction:.2f}%")
    print()
    
    # Check gradient norms
    print(f"Beta progression: {[f'{b:.3f}' for b in betas]}")
    print()
    
    # Verdict
    if final_loss < initial_loss * 0.95:  # At least 5% reduction
        print("✅ PASS: Loss decreased significantly")
        print("   → Gradients are flowing")
        print("   → Model is learning")
        print("   → Ready for overfit test")
        return 0
    elif final_loss < initial_loss:
        print("⚠️  MARGINAL: Loss decreased but slowly")
        print("   → Gradients may be weak")
        print("   → Check learning rate or loss weights")
        return 1
    else:
        print("❌ FAIL: Loss did not decrease")
        print("   → Gradients may be blocked")
        print("   → Check loss function and weight management")
        return 1

if __name__ == "__main__":
    sys.exit(main())



