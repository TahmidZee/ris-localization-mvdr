#!/usr/bin/env python3
"""
3-Phase Training Script

Phase 0: Geometry warm-up (backbone learns features, K de-emphasized)
Phase 1: K-centric (backbone frozen, K-head learns eigenspectrum patterns)
Phase 2: Joint refinement (everything trainable, balanced weights)

Usage:
    python run_3phase_training.py --phase 0  # Run Phase 0
    python run_3phase_training.py --phase 1  # Run Phase 1 (resumes from Phase 0)
    python run_3phase_training.py --phase 2  # Run Phase 2 (resumes from Phase 1)
    python run_3phase_training.py --all      # Run all phases sequentially
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Phase configurations
PHASE_CONFIG = {
    0: {
        "name": "Geometry Warm-up",
        "train_phase": "geom",
        "freeze_backbone": False,
        "freeze_aux": False,
        "epochs": 6,
        "n_train": None,  # Full dataset
        "n_val": None,
        "resume_from": None,
        "save_dir": "runs/phase0_geom",
    },
    1: {
        "name": "K-Centric Fine-tune",
        "train_phase": "k_only",
        "freeze_backbone": True,
        "freeze_aux": False,
        "epochs": 10,
        "n_train": None,
        "n_val": None,
        "resume_from": "runs/phase0_geom/best.ckpt",
        "save_dir": "runs/phase1_k",
    },
    2: {
        "name": "Joint Refinement",
        "train_phase": "joint",
        "freeze_backbone": False,
        "freeze_aux": False,
        "epochs": 45,
        "n_train": None,
        "n_val": None,
        "resume_from": "runs/phase1_k/best.ckpt",
        "save_dir": "runs/phase2_joint",
    },
}

def run_phase(phase_num: int, use_hpo_config: str = None):
    """Run a single training phase."""
    from ris_pytorch_pipeline.configs import cfg, mdl_cfg
    from ris_pytorch_pipeline.train import Trainer
    import torch
    
    pc = PHASE_CONFIG[phase_num]
    
    print("=" * 70)
    print(f"PHASE {phase_num}: {pc['name']}")
    print("=" * 70)
    
    # Set phase config
    cfg.TRAIN_PHASE = pc["train_phase"]
    cfg.FREEZE_BACKBONE_FOR_K_PHASE = pc["freeze_backbone"]
    cfg.FREEZE_AUX_IN_K_PHASE = pc["freeze_aux"]
    
    # Surrogate validation (no MUSIC during training)
    cfg.VAL_PRIMARY = "surrogate"
    cfg.USE_MUSIC_METRICS_IN_VAL = False
    
    # Phase 2: enable SWA for final polish
    if phase_num == 2:
        mdl_cfg.USE_SWA = True
        mdl_cfg.USE_EMA = True
    else:
        mdl_cfg.USE_SWA = False
        mdl_cfg.USE_EMA = False
    
    # Disable curriculum (we handle phases manually)
    mdl_cfg.USE_3_PHASE_CURRICULUM = False
    
    print(f"\n📋 Phase {phase_num} Config:")
    print(f"   TRAIN_PHASE = {cfg.TRAIN_PHASE}")
    print(f"   FREEZE_BACKBONE = {cfg.FREEZE_BACKBONE_FOR_K_PHASE}")
    print(f"   FREEZE_AUX = {cfg.FREEZE_AUX_IN_K_PHASE}")
    print(f"   Epochs = {pc['epochs']}")
    if pc["resume_from"]:
        print(f"   Resume from = {pc['resume_from']}")
    print(f"   Save to = {pc['save_dir']}")
    
    # Update checkpoint directory
    cfg.CKPT_DIR = pc["save_dir"]
    Path(cfg.CKPT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    print("\n🔧 Creating Trainer...")
    t = Trainer(from_hpo=use_hpo_config if use_hpo_config else False)
    
    # Load checkpoint if resuming
    if pc["resume_from"] and Path(pc["resume_from"]).exists():
        print(f"\n📂 Loading checkpoint: {pc['resume_from']}")
        ckpt = torch.load(pc["resume_from"], map_location=t.device)
        t.model.load_state_dict(ckpt["model"])
        print("   ✅ Model weights loaded")
        # Don't load optimizer state - we're starting fresh with new freeze config
    elif pc["resume_from"]:
        print(f"\n⚠️  Checkpoint not found: {pc['resume_from']}")
        print("   Starting from scratch (this may not be intended!)")
    
    # Print loss weights
    print(f"\n📊 Loss weights:")
    print(f"   lam_K = {t.loss_fn.lam_K:.3f}")
    print(f"   lam_aux = {t.loss_fn.lam_aux:.3f}")
    print(f"   lam_cov = {t.loss_fn.lam_cov:.3f}")
    print(f"   lam_subspace_align = {t.loss_fn.lam_subspace_align:.3f}")
    
    # Count params
    trainable = sum(p.numel() for p in t.model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in t.model.parameters() if not p.requires_grad)
    total = trainable + frozen
    print(f"\n🧊 Parameters: {trainable:,} trainable / {frozen:,} frozen ({frozen/total*100:.1f}% frozen)")
    
    # Run training
    print("\n" + "=" * 70)
    print(f"STARTING PHASE {phase_num} TRAINING")
    print("=" * 70 + "\n")
    
    best_score = t.fit(
        epochs=pc["epochs"],
        use_shards=True,
        n_train=pc["n_train"],
        n_val=pc["n_val"],
        gpu_cache=True,
        grad_accumulation=1,
        early_stop_patience=15 if phase_num == 2 else 8,
        val_every=1,
    )
    
    print("\n" + "=" * 70)
    print(f"PHASE {phase_num} COMPLETE")
    print(f"Best score: {best_score:.4f}")
    print(f"Checkpoint saved to: {pc['save_dir']}/best.ckpt")
    print("=" * 70 + "\n")
    
    return best_score

def main():
    parser = argparse.ArgumentParser(description="3-Phase Training")
    parser.add_argument("--phase", type=int, choices=[0, 1, 2], help="Run specific phase")
    parser.add_argument("--all", action="store_true", help="Run all phases sequentially")
    parser.add_argument("--hpo-config", type=str, default=None, 
                        help="Path to HPO best.json (for Phase 2 loss weights)")
    args = parser.parse_args()
    
    if args.all:
        print("\n" + "=" * 70)
        print("RUNNING ALL 3 PHASES SEQUENTIALLY")
        print("=" * 70 + "\n")
        
        for phase in [0, 1, 2]:
            # Only use HPO config for Phase 2
            hpo = args.hpo_config if phase == 2 else None
            run_phase(phase, use_hpo_config=hpo)
        
        print("\n" + "=" * 70)
        print("ALL PHASES COMPLETE!")
        print("Final checkpoint: runs/phase2_joint/best.ckpt")
        print("\nNext step: Run offline MUSIC evaluation")
        print("  python eval_music_final.py --checkpoint runs/phase2_joint/best.ckpt")
        print("=" * 70)
        
    elif args.phase is not None:
        # Only use HPO config for Phase 2
        hpo = args.hpo_config if args.phase == 2 else None
        run_phase(args.phase, use_hpo_config=hpo)
    else:
        parser.print_help()
        print("\n⚠️  Specify --phase N or --all")

if __name__ == "__main__":
    main()



