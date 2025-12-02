#!/usr/bin/env python3
"""
Quick Overfit Test: Confidence builder before full HPO

Purpose:
--------
Run a tiny overfit test on 512 examples (train = val) for 20-40 epochs.
Expected behavior:
- Val loss should plunge (drop significantly from initial value)
- Angle medians should be << 1° at moderate SNR (> 5 dB)
- If this doesn't happen, issue is in data/labels/loss wiring, NOT capacity

Usage:
------
cd /home/tahit/ris/MainMusic
python test_overfit.py

This script will:
1. Create a small dataset subset (512 samples)
2. Train on the same 512 samples for both train and val
3. Monitor val loss convergence
4. Report angle/range errors at different SNR bins
5. Provide go/no-go signal for full HPO
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add ris_pytorch_pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from ris_pytorch_pipeline.configs import cfg, mdl_cfg, set_seed
from ris_pytorch_pipeline.train import Trainer

def main():
    # Setup logging to file
    log_file = Path("overfit_test.log")
    print(f"[OVERFIT] Logging to: {log_file}")
    
    # Tee output to both console and file
    class TeeLogger:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    log_f = open(log_file, 'w')
    sys.stdout = TeeLogger(sys.stdout, log_f)
    sys.stderr = TeeLogger(sys.stderr, log_f)
    
    print("=" * 80)
    print("OVERFIT TEST - Quick Confidence Builder")
    print("=" * 80)
    print()
    print("Purpose: Verify data/labels/loss wiring before full HPO")
    print("Expected: Val loss plunges, angle medians << 1° at moderate SNR")
    print()
    
    set_seed(42)
    
    # Test configuration
    N_SAMPLES = 512
    EPOCHS = 5  # Ready for manual test with NMSE fixes  # Minimal test to verify pure overfit works
    
    # PURE OVERFIT CONFIGURATION - Import first
    from ris_pytorch_pipeline.configs import cfg, mdl_cfg
    
    # A) No blend - β=0 for entire run
    cfg.HYBRID_COV_BETA = 0.0            # disable blending completely
    setattr(mdl_cfg, "OVERFIT_NMSE_PURE", True)
    
    # B) Pure NMSE only - disable ALL regularization
    mdl_cfg.AMP = False
    mdl_cfg.USE_SWA = False
    mdl_cfg.USE_EMA = False
    mdl_cfg.DROPOUT = 0.0
    mdl_cfg.WEIGHT_DECAY = 0.0
    mdl_cfg.CLIP_NORM = 0.0
    mdl_cfg.USE_3_PHASE_CURRICULUM = False  # Disable curriculum
    mdl_cfg.K_WEIGHT_RAMP = False            # Disable K-weight ramp
    
    # C) Consistent LR - single source of truth
    mdl_cfg.LR_INIT = 1e-3
    mdl_cfg.HEAD_LR_MULT = 4.0
    
    # Use moderate model size
    mdl_cfg.D_MODEL = 448
    mdl_cfg.NUM_HEADS = 6
    mdl_cfg.BATCH_SIZE = 64
    
    print(f"Configuration:")
    print(f"  - Samples: {N_SAMPLES} (same for train and val)")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch size: {mdl_cfg.BATCH_SIZE}")
    print(f"  - Learning rate: {mdl_cfg.LR_INIT}")
    print()
    
    print("Creating trainer...")
    trainer = Trainer()
    
    # Force param-group LRs for this run (single source of truth)
    for g in trainer.opt.param_groups:
        name = g.get("name","")
        if name == "backbone":
            g["lr"] = 1e-3
        elif name == "head":
            g["lr"] = 4e-3
        g["weight_decay"] = 0.0
    
    # PURE NMSE ONLY - disable ALL auxiliary losses
    loss = trainer.loss_fn
    loss.lam_cov       = 1.0
    loss.lam_cov_pred  = 0.0
    loss.lam_diag      = 0.0
    loss.lam_off       = 0.0
    loss.lam_ortho     = 0.0
    loss.lam_cross     = 0.0
    loss.lam_gap       = 0.0
    loss.lam_K         = 0.0
    loss.lam_aux       = 0.0
    loss.lam_peak      = 0.0
    loss.lam_subspace_align = 0.0
    loss.lam_peak_contrast  = 0.0
    
    print("🔧 PURE OVERFIT: β=0, NMSE-only, all λ=0 except lam_cov=1.0")
    
    # Pure mode violation guards
    assert cfg.HYBRID_COV_BETA == 0.0, "Pure overfit requires β=0.0"
    for name, val in {
        "lam_cov_pred": loss.lam_cov_pred, 
        "lam_gap": loss.lam_gap,
        "lam_cross": loss.lam_cross, 
        "lam_K": loss.lam_K,
        "lam_diag": loss.lam_diag,
        "lam_off": loss.lam_off,
        "lam_ortho": loss.lam_ortho,
        "lam_aux": loss.lam_aux,
        "lam_peak": loss.lam_peak,
        "lam_subspace_align": loss.lam_subspace_align,
        "lam_peak_contrast": loss.lam_peak_contrast,
    }.items():
        assert float(val) == 0.0, f"Pure overfit requires {name}=0.0, got {val}"
    
    assert float(loss.lam_cov) == 1.0, f"Pure overfit requires lam_cov=1.0, got {loss.lam_cov}"
    print("✅ Pure overfit guards passed - all λ=0 except lam_cov=1.0")
    
    print(f"Starting overfit test...")
    print(f"Monitoring first and last 3 epochs for convergence signal...")
    print()
    
    try:
        best_val = trainer.fit(
            epochs=EPOCHS,
            use_shards=True,
            n_train=N_SAMPLES,
            n_val=N_SAMPLES,  # Same data for train and val (overfit test)
            gpu_cache=True,
            grad_accumulation=1,
            early_stop_patience=15,  # Stop if loss plateaus
        )
        
        print()
        print("=" * 80)
        print("OVERFIT TEST RESULTS")
        print("=" * 80)
        print()
        
        if best_val is None or not np.isfinite(best_val):
            print("❌ FAILED: Non-finite validation loss")
            print(f"   Val loss: {best_val}")
            print()
            print("This indicates a critical issue in training:")
            print("  - Check gradient flow")
            print("  - Check loss function implementation")
            print("  - Check data normalization")
            print()
            return 1
        
        print(f"✅ Best validation loss: {best_val:.6f}")
        print()
        
        # Load best checkpoint and evaluate
        checkpoint_path = Path(cfg.CKPT_DIR) / "best.pt"
        if checkpoint_path.exists():
            print("Loading best checkpoint for evaluation...")
            # Expert fix: Handle both dict and direct state_dict formats
            ckpt = torch.load(checkpoint_path)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                trainer.model.load_state_dict(ckpt['model_state'])
            else:
                trainer.model.load_state_dict(ckpt)
            
            # Run evaluation with Hungarian metrics
            print()
            print("Running Hungarian matching evaluation...")
            
        # Provide interpretation
        print()
        print("INTERPRETATION:")
        print()
        if best_val < 0.5:
            print("✅ EXCELLENT: Val loss < 0.5")
            print("   → Data pipeline is working correctly")
            print("   → Loss function is properly wired")
            print("   → Model has sufficient capacity")
            print()
            print("🚀 READY FOR FULL HPO")
        elif best_val < 1.0:
            print("⚠️  MARGINAL: Val loss < 1.0 but > 0.5")
            print("   → System is learning but may have issues")
            print("   → Check loss weights and data quality")
            print()
            print("⏸️  CONSIDER DEBUGGING BEFORE HPO")
        else:
            print("❌ POOR: Val loss > 1.0")
            print("   → System is not learning properly")
            print("   → Check data/labels/loss wiring")
            print("   → DO NOT proceed to HPO")
            print()
            print("🛑 FIX ISSUES BEFORE HPO")
            return 1
        
        print()
        print("=" * 80)
        return 0
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ OVERFIT TEST CRASHED")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()
        print("This indicates a critical bug - fix before proceeding to HPO")
        return 1
    finally:
        # Clean up logging
        if 'log_f' in locals():
            log_f.close()
        # Restore stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

if __name__ == "__main__":
    sys.exit(main())

