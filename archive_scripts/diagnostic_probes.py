#!/usr/bin/env python3
"""
Diagnostic Probes - Last-Mile Verification
==========================================

Checks:
1. High-SNR slice (≥10 dB): K-acc, angle medians
2. K class distribution (check for skew)
3. Current model performance breakdown

Decision criteria:
- If high-SNR K-acc < 0.55 → Need Tier 1 (beam tokens)
- If median |Δφ| > 1.4° or |Δθ| > 1.8° → Need Tier 1
- If cov-NMSE dominates → Need Tier 2 (separable PSD)
- If K distribution skewed → Add focal/weighted CE
"""

import sys
import torch
import numpy as np
from pathlib import Path
from collections import Counter

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from ris_pytorch_pipeline import configs
from ris_pytorch_pipeline.configs import cfg, mdl_cfg
from ris_pytorch_pipeline.dataset import make_loader
from ris_pytorch_pipeline.model import HybridModel
from ris_pytorch_pipeline.loss import UltimateHybridLoss

def _wrap_deg(d):
    """Wrap angle to [-180, 180]"""
    return ((d + 180) % 360) - 180

def _angle_dist(pred, true):
    """Angular distance in degrees"""
    return np.abs(_wrap_deg(pred - true))

def main():
    print("=" * 80)
    print("DIAGNOSTIC PROBES - LAST-MILE VERIFICATION")
    print("=" * 80)
    
    # Setup config
    cfg.FIELD_TYPE = "near"
    cfg.K_MAX = 5
    cfg.N = 16
    mdl_cfg.SNR_TARGETED = True
    mdl_cfg.SNR_DB_RANGE = (-5.0, 20.0)
    mdl_cfg.BATCH_SIZE = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📍 Device: {device}")
    
    # Create a Trainer to get data loaders
    print("\n📂 Loading validation dataset...")
    from ris_pytorch_pipeline.train import Trainer
    temp_trainer = Trainer(from_hpo=False)
    _, val_loader = temp_trainer._load_shards(n_train=1000, n_val=10000, gpu_cache=False)
    del temp_trainer  # Free memory
    
    # Check K distribution
    print("\n" + "=" * 80)
    print("1. K CLASS DISTRIBUTION CHECK")
    print("=" * 80)
    
    K_values = []
    SNR_values = []
    for batch in val_loader:
        K_values.append(batch['K'].numpy())
        SNR_values.append(batch['snr_db'].numpy())
    
    K_all = np.concatenate(K_values)
    SNR_all = np.concatenate(SNR_values)
    
    K_counts = Counter(K_all)
    total = len(K_all)
    
    print("\nK Distribution:")
    for k in sorted(K_counts.keys()):
        freq = K_counts[k] / total
        bar = "█" * int(freq * 50)
        print(f"  K={k}: {K_counts[k]:5d} ({freq:5.1%}) {bar}")
    
    # Check for skew
    freqs = np.array([K_counts[k] / total for k in sorted(K_counts.keys())])
    max_freq = freqs.max()
    min_freq = freqs.min()
    skew_ratio = max_freq / min_freq if min_freq > 0 else float('inf')
    
    print(f"\n📊 Skew ratio (max/min): {skew_ratio:.2f}")
    if skew_ratio > 2.0:
        print("⚠️  RECOMMENDATION: Add class-weighted or focal CE")
    else:
        print("✅ Distribution is balanced, standard CE is fine")
    
    print(f"\n📊 SNR range: [{SNR_all.min():.1f}, {SNR_all.max():.1f}] dB")
    print(f"    SNR mean: {SNR_all.mean():.1f} ± {SNR_all.std():.1f} dB")
    
    # Try to load model
    print("\n" + "=" * 80)
    print("2. MODEL PERFORMANCE CHECK")
    print("=" * 80)
    
    best_path = Path(cfg.CKPT_DIR) / "best.pt"
    if not best_path.exists():
        print(f"\n⚠️  No trained model found at {best_path}")
        print("    Run HPO first, then re-run this diagnostic.")
        return
    
    print(f"\n📦 Loading model from {best_path}...")
    model = HybridModel().to(device)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    
    k_temp = ckpt.get('k_calibration_temp', 1.0)
    print(f"🌡️  K calibration temperature: {k_temp:.3f}")
    
    # Prepare loss for per-term analysis
    loss_fn = UltimateHybridLoss()
    
    # Collect predictions
    print("\n🔬 Running inference on validation set...")
    
    all_metrics = {
        'k_pred': [], 'k_true': [],
        'phi_err': [], 'theta_err': [], 'r_err': [],
        'snr': [],
        'loss_K': [], 'loss_cov': [], 'loss_ang': [], 'loss_rng': []
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 50:  # ~1600 samples
                break
            
            y = batch['y'].to(device)
            H = batch['H'].to(device)
            C = batch['codes'].to(device)
            K_true = batch['K'].to(device)
            ptr = batch['ptr'].to(device)
            snr = batch['snr_db'].to(device)
            R_in = batch['R_cov'].to(device)
            
            # Forward
            preds = model(y=y, H=H, codes=C)
            
            # K prediction (with temperature)
            if 'k_logits' in preds:
                k_logits = preds['k_logits'] / k_temp
                k_pred = k_logits.argmax(dim=1) + 1  # 0-indexed → 1-indexed
                all_metrics['k_pred'].append(k_pred.cpu().numpy())
                all_metrics['k_true'].append(K_true.cpu().numpy())
            
            # Per-sample angle/range errors (simple version, no Hungarian)
            if 'phi' in preds and 'r' in preds:
                phi_pred = preds['phi'].cpu().numpy()  # [B, K_MAX]
                r_pred = preds['r'].cpu().numpy()
                
                # Extract ground truth from ptr (simplified)
                B = ptr.shape[0]
                K_MAX = cfg.K_MAX
                ptr_np = ptr.cpu().numpy()  # [B, 3*K_MAX]
                
                for b in range(B):
                    k_b = K_true[b].item()
                    phi_true_b = ptr_np[b, :k_b]
                    r_true_b = ptr_np[b, 2*K_MAX:2*K_MAX+k_b]
                    
                    phi_pred_b = phi_pred[b, :k_b]
                    r_pred_b = r_pred[b, :k_b]
                    
                    # Simple L1 matching (not Hungarian, but fast)
                    for i in range(k_b):
                        all_metrics['phi_err'].append(_angle_dist(phi_pred_b[i], phi_true_b[i]))
                        all_metrics['r_err'].append(np.abs(r_pred_b[i] - r_true_b[i]))
            
            all_metrics['snr'].append(snr.cpu().numpy())
            
            # Compute per-term losses
            R_true_c = torch.view_as_complex(R_in.reshape(R_in.shape[0], R_in.shape[1], R_in.shape[2]//2, 2).contiguous())
            R_true_c = 0.5 * (R_true_c + R_true_c.conj().transpose(-2, -1))
            R_true = torch.view_as_real(R_true_c).reshape(R_in.shape[0], R_in.shape[1], -1).float()
            
            labels = {"R_true": R_true, "ptr": ptr, "K": K_true, "snr_db": snr}
            
            debug = loss_fn.debug_terms(preds, labels)
            all_metrics['loss_K'].append(debug.get('loss_K_raw', 0))
            all_metrics['loss_cov'].append(debug.get('loss_nmse', 0))
            all_metrics['loss_ang'].append(debug.get('phi_huber', 0))
            all_metrics['loss_rng'].append(debug.get('rng_err_log', 0))
    
    # Aggregate metrics
    k_pred_all = np.concatenate(all_metrics['k_pred'])
    k_true_all = np.concatenate(all_metrics['k_true'])
    snr_all = np.concatenate(all_metrics['snr'])
    
    k_acc_overall = (k_pred_all == k_true_all).mean()
    
    # High-SNR slice
    high_snr_mask = snr_all >= 10.0
    k_acc_high_snr = (k_pred_all[high_snr_mask] == k_true_all[high_snr_mask]).mean() if high_snr_mask.sum() > 0 else 0.0
    
    # Mid-SNR slice
    mid_snr_mask = (snr_all >= 3.0) & (snr_all < 10.0)
    k_acc_mid_snr = (k_pred_all[mid_snr_mask] == k_true_all[mid_snr_mask]).mean() if mid_snr_mask.sum() > 0 else 0.0
    
    # Low-SNR slice
    low_snr_mask = snr_all < 3.0
    k_acc_low_snr = (k_pred_all[low_snr_mask] == k_true_all[low_snr_mask]).mean() if low_snr_mask.sum() > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("3. HIGH-SNR SLICE ANALYSIS (Decision Criterion)")
    print("=" * 80)
    
    print(f"\n📊 K-Accuracy Breakdown:")
    print(f"  Overall:        {k_acc_overall:.1%} ({len(k_pred_all)} samples)")
    print(f"  High-SNR (≥10): {k_acc_high_snr:.1%} ({high_snr_mask.sum()} samples) {'🔴' if k_acc_high_snr < 0.55 else '✅'}")
    print(f"  Mid-SNR (3-10): {k_acc_mid_snr:.1%} ({mid_snr_mask.sum()} samples)")
    print(f"  Low-SNR (<3):   {k_acc_low_snr:.1%} ({low_snr_mask.sum()} samples)")
    
    if len(all_metrics['phi_err']) > 0:
        phi_err = np.array(all_metrics['phi_err'])
        r_err = np.array(all_metrics['r_err'])
        
        phi_median = np.median(phi_err)
        r_median = np.median(r_err)
        
        print(f"\n📊 Angle/Range Errors (Median):")
        print(f"  |Δφ|: {phi_median:.2f}° {'🔴' if phi_median > 1.4 else '✅'}")
        print(f"  |Δr|: {r_median:.3f} m")
    
    # Per-term loss breakdown
    print(f"\n📊 Per-Term Loss (Average):")
    print(f"  K-CE:   {np.mean(all_metrics['loss_K']):.4f}")
    print(f"  Cov:    {np.mean(all_metrics['loss_cov']):.4f}")
    print(f"  Angle:  {np.mean(all_metrics['loss_ang']):.4f}")
    print(f"  Range:  {np.mean(all_metrics['loss_rng']):.4f}")
    
    cov_loss_avg = np.mean(all_metrics['loss_cov'])
    if cov_loss_avg > 0.5:
        print("  🔴 Cov loss dominates → Consider Tier 2 (separable PSD)")
    
    # Decision
    print("\n" + "=" * 80)
    print("4. RECOMMENDATION")
    print("=" * 80)
    
    needs_tier1 = k_acc_high_snr < 0.55 or (len(all_metrics['phi_err']) > 0 and phi_median > 1.4)
    needs_tier2 = cov_loss_avg > 0.5
    needs_focal = skew_ratio > 2.0
    
    if not needs_tier1 and not needs_tier2 and not needs_focal:
        print("\n✅ CURRENT ARCHITECTURE IS SUFFICIENT")
        print("   Continue with longer training (30-40 epochs + SWA)")
    else:
        print("\n🔧 RECOMMENDED UPGRADES:")
        if needs_tier1:
            print("   ✓ Tier 1: Beam-aware token encoder (+5-10 pts K-acc, -0.2-0.4° angles)")
        if needs_tier2:
            print("   ✓ Tier 2: Separable PSD covariance head (stabilize cov-NMSE)")
        if needs_focal:
            print("   ✓ Add focal/weighted CE for K (handle class imbalance)")
        
        print("\n   Expected gains:")
        print(f"     K-acc (high-SNR): {k_acc_high_snr:.1%} → ~{min(0.95, k_acc_high_snr + 0.10):.1%}")
        if len(all_metrics['phi_err']) > 0:
            print(f"     |Δφ| median: {phi_median:.2f}° → ~{max(0.5, phi_median - 0.3):.2f}°")
    
    print("\n" + "=" * 80)
    print("END OF DIAGNOSTICS")
    print("=" * 80)

if __name__ == "__main__":
    main()

