#!/usr/bin/env python3
import numpy as np
import os

# Check if shards exist
shard_path = '/home/tahit/ris/MainMusic/data_shards_M32_L16/train/shard_000.npz'
if os.path.exists(shard_path):
    print("✅ Shard exists")
    size_mb = os.path.getsize(shard_path) / (1024*1024)
    print(f"Size: {size_mb:.1f} MB")
    
    try:
        data = np.load(shard_path)
        print(f"Keys: {list(data.keys())}")
        print(f"Y shape: {data['y'].shape}")
        print(f"Samples: {len(data['y'])}")
        
        # Check ranges
        if 'phi' in data:
            phi_deg = np.degrees(data['phi'])
            print(f"Phi range: {phi_deg.min():.1f}° to {phi_deg.max():.1f}°")
        
        if 'theta' in data:
            theta_deg = np.degrees(data['theta'])
            print(f"Theta range: {theta_deg.min():.1f}° to {theta_deg.max():.1f}°")
        
        if 'r' in data:
            print(f"Range: {data['r'].min():.2f}m to {data['r'].max():.2f}m")
        
        if 'snr_db' in data:
            print(f"SNR: {data['snr_db'].min():.1f} to {data['snr_db'].max():.1f} dB")
        
        data.close()
        print("✅ Shard loaded successfully")
    except Exception as e:
        print(f"❌ Error loading shard: {e}")
else:
    print("❌ Shard does not exist")
