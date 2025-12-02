#!/usr/bin/env python3
"""
Manual Training Script - Quick Test with Fixed Hyperparameters
===============================================================

Uses mid-range hyperparameters from our HPO search space.
Runs 12 epochs with mini-curriculum to quickly verify the fixes.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ris_pytorch_pipeline import configs
from ris_pytorch_pipeline.configs import cfg, mdl_cfg, set_seed
from ris_pytorch_pipeline.train import Trainer

def main():
    print("=" * 80)
    print("MANUAL TRAINING - QUICK TEST WITH FIXED HYPERPARAMETERS")
    print("=" * 80)
    
    set_seed(42)
    
    # Setup config
    cfg.FIELD_TYPE = "near"
    cfg.K_MAX = 5
    cfg.N = 16
    cfg.CKPT_DIR = Path("results_final_L16_12x12/manual_run")
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    
    # SNR-aware data
    mdl_cfg.SNR_TARGETED = True
    mdl_cfg.SNR_DB_RANGE = (-5.0, 20.0)
    
    # Enable mini-curriculum
    mdl_cfg.USE_3_PHASE_CURRICULUM = True
    
    # Disable EMA/SWA for quick run
    mdl_cfg.USE_EMA = False
    mdl_cfg.USE_SWA = False
    
    # Mid-range hyperparameters from HPO search space
    mdl_cfg.LR = 5e-4  # Mid-range of [3e-4, 1e-3]
    mdl_cfg.BATCH_SIZE = 64
    mdl_cfg.DROPOUT = 0.25  # Mid-range of [0.15, 0.35]
    
    # Loss weights (mid-range from HPO)
    lam_cov = 0.3    # [0.2, 0.4]
    lam_ang = 0.2    # [0.15, 0.25]
    lam_rng = 0.25   # [0.15, 0.35]
    lam_K = 0.8      # [0.6, 1.0]
    lam_gap = 0.05   # [0.02, 0.08]
    lam_cross = 0.05  # Fixed
    
    print(f"\n📊 Hyperparameters:")
    print(f"  LR: {mdl_cfg.LR}")
    print(f"  Batch size: {mdl_cfg.BATCH_SIZE}")
    print(f"  Dropout: {mdl_cfg.DROPOUT}")
    print(f"  Loss weights: cov={lam_cov}, ang={lam_ang}, "
          f"rng={lam_rng}, K={lam_K}, gap={lam_gap}")
    print(f"  Mini-curriculum: {'✅ ENABLED' if mdl_cfg.USE_3_PHASE_CURRICULUM else '❌ DISABLED'}")
    
    # Create Trainer (will create its own dataloaders)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📍 Device: {device}")
    
    trainer = Trainer(from_hpo=False)
    
    # Set loss weights
    trainer.loss_fn.lam_diag = lam_cov * 0.2
    trainer.loss_fn.lam_off = lam_cov * 0.8
    trainer.loss_fn.lam_aux = lam_ang + lam_rng
    trainer.loss_fn.lam_K = lam_K
    trainer.loss_fn.lam_cross = lam_cross
    trainer.loss_fn.lam_gap = lam_gap
    
    # Set reasonable defaults for other loss params
    trainer.loss_fn.lam_ortho = 1e-3
    trainer.loss_fn.lam_peak = 0.05
    trainer.loss_fn.lam_margin = 0.1
    trainer.loss_fn.lam_range_factor = 0.3
    mdl_cfg.LAM_ALIGN = 0.002
    
    # Train
    print("\n🚀 Starting training (12 epochs with mini-curriculum)...")
    print("=" * 80)
    
    # Use full dataset (not HPO subset)
    composite_score = trainer.fit(
        epochs=12,
        use_shards=True,
        n_train=80000,  # Full training set
        n_val=16000,    # Full validation set
        gpu_cache=True,
        grad_accumulation=1,
        early_stop_patience=5
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"📊 Final composite score: {composite_score:.4f}")
    print(f"💾 Best model saved to: {cfg.CKPT_DIR / 'best.pt'}")
    
    print("\n" + "=" * 80)
    print("NEXT STEP: Run full diagnostics")
    print("=" * 80)
    print("Commands:")
    print("  cd /home/tahit/ris/MainMusic")
    print("  python diagnostic_probes.py")
    print("  python k_only_probe.py")
    print("=" * 80)

if __name__ == "__main__":
    main()

