#!/usr/bin/env python3
"""
Manual dataset verification script
Run this in your screen terminal to check if the dataset is correct
"""

import numpy as np
from pathlib import Path
import sys

def check_shard(shard_path, split_name, shard_name):
    """Check a single shard and print detailed info"""
    print(f"\n📁 {split_name.upper()} - {shard_name}")
    print("=" * 50)
    
    try:
        # Load shard
        data = np.load(shard_path)
        n_samples = len(data['y'])
        
        # File size
        size_mb = shard_path.stat().st_size / (1024*1024)
        print(f"📊 File size: {size_mb:.1f} MB")
        print(f"📈 Samples: {n_samples:,}")
        
        # Check snapshot dimensions
        y_shape = data['y'].shape
        print(f"🎯 Snapshot shape: {y_shape}")
        print(f"   • L (snapshots): {y_shape[1]}")
        print(f"   • M (BS antennas): {y_shape[2]}")
        print(f"   • Channels: {y_shape[3]}")
        
        # Verify dimensions
        expected_L = 16
        expected_M = 16  # BS antennas (receiving end)
        
        if y_shape[1] == expected_L:
            print(f"✅ L is correct: {expected_L}")
        else:
            print(f"❌ L ERROR: got {y_shape[1]}, expected {expected_L}")
        
        if y_shape[2] == expected_M:
            print(f"✅ M (BS antennas) is correct: {expected_M}")
        else:
            print(f"❌ M ERROR: got {y_shape[2]}, expected {expected_M}")
        
        # Check parameter ranges
        if 'phi' in data:
            phi_deg = np.degrees(data['phi'])
            print(f"📐 φ range: {phi_deg.min():.1f}° to {phi_deg.max():.1f}°")
            print(f"   Expected: ±60°")
        
        if 'theta' in data:
            theta_deg = np.degrees(data['theta'])
            print(f"📐 θ range: {theta_deg.min():.1f}° to {theta_deg.max():.1f}°")
            print(f"   Expected: ±30°")
        
        if 'r' in data:
            r_vals = data['r']
            print(f"📏 Range: {r_vals.min():.2f}m to {r_vals.max():.2f}m")
            print(f"   Expected: 0.5 to 10.0m")
        
        if 'snr_db' in data:
            snr_vals = data['snr_db']
            print(f"📶 SNR: {snr_vals.min():.1f} to {snr_vals.max():.1f} dB")
            print(f"   Expected: -5 to 20 dB")
        
        # Check K distribution
        if 'K' in data:
            k_values = data['K']
            k_counts = np.bincount(k_values)
            print(f"🔢 K distribution:")
            for k, count in enumerate(k_counts):
                if count > 0:
                    print(f"   K={k}: {count} samples ({count/n_samples*100:.1f}%)")
            
            print(f"   K range: {k_values.min()} to {k_values.max()}")
            print(f"   Unique K values: {sorted(set(k_values))}")
        
        data.close()
        return n_samples
        
    except Exception as e:
        print(f"❌ Error loading shard: {e}")
        return 0

def main():
    print("🔍 MANUAL DATASET VERIFICATION")
    print("=" * 60)
    print("Checking if the L=16 M_beams=32 dataset is correct...")
    print()
    
    base_dir = Path("data_shards_M32_L16")
    if not base_dir.exists():
        print("❌ Directory 'data_shards_M32_L16' does not exist!")
        print("   Make sure you're in the MainMusic directory")
        return
    
    print("✅ Dataset directory exists")
    
    total_samples = 0
    
    # Check each split
    for split in ['train', 'val', 'test']:
        split_dir = base_dir / split
        if not split_dir.exists():
            print(f"❌ {split} directory missing")
            continue
        
        print(f"\n📂 {split.upper()} SPLIT")
        print("-" * 40)
        
        shards = list(split_dir.glob("*.npz"))
        print(f"Found {len(shards)} shards")
        
        split_samples = 0
        for i, shard_path in enumerate(sorted(shards)):
            samples = check_shard(shard_path, split, shard_path.name)
            split_samples += samples
            
            # Only show details for first shard of each split
            if i > 0:
                print(f"   {shard_path.name}: {samples:,} samples")
        
        total_samples += split_samples
        print(f"\n📊 {split.upper()} total: {split_samples:,} samples")
    
    print(f"\n🎯 DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples: {total_samples:,}")
    print(f"Expected: 100,000 (80K train + 16K val + 4K test)")
    
    if total_samples == 100000:
        print("✅ Sample count matches expected!")
    else:
        print(f"⚠️  Sample count mismatch: got {total_samples}, expected 100,000")
    
    print(f"\n📋 EXPECTED PARAMETERS:")
    print("   • L (snapshots): 16")
    print("   • N (elements): 144 (12×12 UPA)")
    print("   • φ FOV: ±60°")
    print("   • θ FOV: ±30°")
    print("   • Range: 0.5 to 10.0m")
    print("   • SNR: -5 to 20 dB")
    print("   • K range: 1 to 5 (or 0 to 5)")
    
    print(f"\n✅ Verification complete!")
    print("   If all parameters look correct, you can proceed with HPO.")

if __name__ == "__main__":
    main()
