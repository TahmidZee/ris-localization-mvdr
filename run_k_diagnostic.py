#!/usr/bin/env python3
"""
K-Head Diagnostic Script

Quick test to verify K-head can learn when given proper conditions:
- Backbone frozen (stable features)
- High lam_K (K is the boss)
- Small dataset (fast iteration)

Expected outcome: K_acc should rise above 0.3-0.4 within 5 epochs.
If K_acc stays at ~0.2 (chance), there's a bug in the K-head or labels.
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("K-HEAD DIAGNOSTIC")
    print("Testing if K-head can learn on frozen backbone")
    print("=" * 60)
    
    # Import after path setup
    from ris_pytorch_pipeline.configs import cfg, mdl_cfg
    from ris_pytorch_pipeline.train import Trainer
    
    # === PHASE 1 CONFIG: K-centric ===
    cfg.TRAIN_PHASE = "k_only"
    # UNFREEZE backbone here so the diagnostic can actually learn from scratch.
    cfg.FREEZE_BACKBONE_FOR_K_PHASE = False
    cfg.FREEZE_AUX_IN_K_PHASE = False  # Keep aux trainable
    
    # Surrogate validation (no MUSIC)
    cfg.VAL_PRIMARY = "surrogate"
    cfg.USE_MUSIC_METRICS_IN_VAL = False
    
    # Disable SWA/EMA for diagnostic (cleaner signal)
    mdl_cfg.USE_SWA = False
    mdl_cfg.USE_EMA = False
    # Disable AMP for stability in this diagnostic
    mdl_cfg.USE_AMP = False
    
    # Disable curriculum (we're setting weights manually)
    mdl_cfg.USE_3_PHASE_CURRICULUM = False
    
    # V100-optimized defaults (you can override below)
    mdl_cfg.BATCH_SIZE = 64
    mdl_cfg.NUM_WORKERS = 4
    mdl_cfg.PIN_MEMORY = True
    
    # Optional easy overfit mode (set EASY_OVERFIT=1)
    easy_overfit = os.environ.get("EASY_OVERFIT", "") == "1"
    if easy_overfit:
        print("⚡ EASY_OVERFIT=1 → using tiny split and stronger K/AUX weights")
        # Make it even lighter to avoid OOM in loader iteration
        mdl_cfg.BATCH_SIZE = 8
        mdl_cfg.NUM_WORKERS = 0
        mdl_cfg.PIN_MEMORY = False
        # Make the task easier for K: USE hybrid blend so spectral features actually see R_samp.
        # NOTE: Keep beta < 1.0 so R_blend still has a (small) grad path to R_pred during training.
        beta_default = "0.50"
        cfg.HYBRID_COV_BETA = float(os.environ.get("K_DIAG_BETA", beta_default))
        # Phase loss overrides for this run
        cfg.PHASE_LOSS["k_only"].update({
            "lam_cov": 0.02,
            "lam_aux": 1.0,
            "lam_K": 2.0,
            "lam_gap": 0.0,
            "lam_margin": 0.0,
            "lam_subspace_align": 0.0,
            "lam_peak_contrast": 0.0,
        })
    
    print(f"\n📋 Config:")
    print(f"   TRAIN_PHASE = {cfg.TRAIN_PHASE}")
    print(f"   FREEZE_BACKBONE = {cfg.FREEZE_BACKBONE_FOR_K_PHASE}")
    print(f"   HYBRID_COV_BETA = {getattr(cfg, 'HYBRID_COV_BETA', None)}")
    print(f"   VAL_PRIMARY = {cfg.VAL_PRIMARY}")
    print(f"   USE_MUSIC_METRICS_IN_VAL = {cfg.USE_MUSIC_METRICS_IN_VAL}")
    
    # Create trainer (will apply phase-specific freezing and loss weights)
    print("\n🔧 Creating Trainer...")
    t = Trainer(from_hpo=False)
    
    # Print loss weights being used
    print(f"\n📊 Loss weights (from PHASE_LOSS['{cfg.TRAIN_PHASE}']):")
    print(f"   lam_K = {t.loss_fn.lam_K:.3f}")
    print(f"   lam_aux = {t.loss_fn.lam_aux:.3f}")
    print(f"   lam_cov = {t.loss_fn.lam_cov:.3f}")
    print(f"   lam_subspace_align = {t.loss_fn.lam_subspace_align:.3f}")
    print(f"   lam_peak_contrast = {t.loss_fn.lam_peak_contrast:.3f}")
    
    # Count trainable params
    trainable = sum(p.numel() for p in t.model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in t.model.parameters() if not p.requires_grad)
    print(f"\n🧊 Parameters:")
    print(f"   Trainable: {trainable:,}")
    print(f"   Frozen: {frozen:,}")
    print(f"   Ratio: {frozen/(trainable+frozen)*100:.1f}% frozen")
    
    # Run short training
    print("\n" + "=" * 60)
    print("STARTING K DIAGNOSTIC TRAINING")
    print("Watch for K_acc to rise above 0.3-0.4")
    print("=" * 60 + "\n")
    
    best_score = t.fit(
        epochs=10,
        use_shards=True,
        n_train=5000 if not easy_overfit else 64,
        n_val=1000  if not easy_overfit else 64,
        gpu_cache=not easy_overfit,    # avoid cache in tiny overfit mode
        grad_accumulation=1,
        early_stop_patience=10,  # No early stop for diagnostic
        val_every=1,
    )
    
    print("\n" + "=" * 60)
    print("K DIAGNOSTIC COMPLETE")
    print(f"Final score: {best_score:.4f}")
    print("=" * 60)
    
    print("\n📝 INTERPRETATION:")
    print("   - If K_acc > 0.3: K-head CAN learn ✅")
    print("   - If K_acc ≈ 0.2: K-head is BROKEN ❌ (check labels/indexing)")
    print("\nIf diagnostic passes, proceed with 3-phase full training.")

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        import sys; sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        import sys; sys.exit(130)

