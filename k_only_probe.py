#!/usr/bin/env python3
"""
K-Only Probe - Verify K-head Wiring
====================================

Trains ONLY the K-head for 2-3 epochs with lam_K=1.0, all others=0.

Expected:
- Train K-acc > 70%
- Val K-acc > 50%

If not met: K-head wiring/labels are broken.
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
    print("K-ONLY PROBE - VERIFY K-HEAD WIRING")
    print("=" * 80)
    
    set_seed(42)
    
    # Setup
    cfg.FIELD_TYPE = "near"
    cfg.K_MAX = 5
    cfg.N = 16
    cfg.CKPT_DIR = Path("results_final_L16_12x12/k_only_probe")
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    
    mdl_cfg.SNR_TARGETED = True
    mdl_cfg.SNR_DB_RANGE = (-5.0, 20.0)
    mdl_cfg.USE_EMA = False
    mdl_cfg.USE_SWA = False
    mdl_cfg.USE_3_PHASE_CURRICULUM = False  # No curriculum for probe
    
    mdl_cfg.BATCH_SIZE = 64
    mdl_cfg.LR = 5e-4
    mdl_cfg.DROPOUT = 0.0  # No dropout for K-only probe
    mdl_cfg.NUM_WORKERS = 0  # Disable multiprocessing to avoid OOM
    mdl_cfg.PERSISTENT_WORKERS = False
    
    # Override loss weights: K-only
    print("\n🔧 Setting K-only loss weights (lam_K=1.0, others=0.0)...")
    
    # Create trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📍 Device: {device}")
    
    trainer = Trainer(from_hpo=False)
    
    # Set K-only weights
    trainer.loss_fn.lam_diag = 0.0
    trainer.loss_fn.lam_off = 0.0
    trainer.loss_fn.lam_aux = 0.0
    trainer.loss_fn.lam_K = 1.0
    trainer.loss_fn.lam_cross = 0.0
    trainer.loss_fn.lam_gap = 0.0
    trainer.loss_fn.lam_ortho = 0.0
    trainer.loss_fn.lam_peak = 0.0
    trainer.loss_fn.lam_margin = 0.0
    
    print(f"📊 Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    # Train for 3 epochs (use subset for speed)
    print("\n🚀 Starting K-only probe (3 epochs)...")
    print("=" * 80)
    
    try:
        trainer.fit(
            epochs=3,
            use_shards=True,
            n_train=2000,   # Minimal subset (less than 1/10 of one shard)
            n_val=500,
            gpu_cache=False,  # Disable GPU cache to avoid OOM
            grad_accumulation=2,  # Increase to compensate for smaller batch
            early_stop_patience=10  # No early stopping
        )
    except Exception as e:
        print(f"\n❌ ERROR during fit: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("K-ONLY PROBE RESULTS")
    print("=" * 80)
    
    # Load best and check K-acc
    best_path = cfg.CKPT_DIR / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        model.eval()
        
        # Compute train and val K-acc
        def compute_k_acc(loader):
            correct, total = 0, 0
            with torch.no_grad():
                for batch in loader:
                    y = batch['y'].to(device)
                    H = batch['H'].to(device)
                    C = batch['codes'].to(device)
                    K_true = batch['K'].to(device)
                    
                    preds = model(y=y, H=H, codes=C)
                    if 'k_logits' in preds:
                        k_pred = preds['k_logits'].argmax(dim=1) + 1
                        correct += (k_pred == K_true).sum().item()
                        total += K_true.size(0)
            return correct / total if total > 0 else 0.0
        
        print("\n🔬 Computing K-accuracy on train/val...")
        train_k_acc = compute_k_acc(tr_loader)
        val_k_acc = compute_k_acc(va_loader)
        
        print(f"\n📊 K-Accuracy Results:")
        print(f"  Train: {train_k_acc:.1%} {'✅' if train_k_acc > 0.70 else '🔴'} (expect >70%)")
        print(f"  Val:   {val_k_acc:.1%} {'✅' if val_k_acc > 0.50 else '🔴'} (expect >50%)")
        
        if train_k_acc < 0.70 or val_k_acc < 0.50:
            print("\n🔴 FAILED: K-head wiring or labels are broken!")
            print("   Check:")
            print("   - k_head output shape: should be [B, K_MAX=5]")
            print("   - K labels: should be 1-indexed (1..5)")
            print("   - Loss target mapping: K_true - 1 (0-indexed)")
            print("   - Inference mapping: argmax(logits) + 1 (back to 1-indexed)")
        else:
            print("\n✅ PASSED: K-head is wired correctly!")
    else:
        print(f"\n⚠️  No checkpoint saved at {best_path}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

