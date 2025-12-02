"""
K-Grouped Batch Sampler for Variable-K Datasets
================================================

Ensures all samples in a batch have the same K value to avoid collation errors.
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Iterator, List


class KGroupedBatchSampler(Sampler):
    """
    Batch sampler that groups samples by K (number of sources).
    Ensures all samples in a batch have the same K value.
    
    OPTIMIZED: Reads K values directly from shards without loading full samples.
    """
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by K (optimized - read K directly from dataset)
        print("📊 Grouping samples by K (fast mode)...")
        self.k_groups = {}
        
        # Check if dataset is a Subset
        from torch.utils.data import Subset
        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            indices = dataset.indices
        else:
            base_dataset = dataset
            indices = list(range(len(dataset)))
        
        # Fast K extraction from ShardNPZDataset
        if hasattr(base_dataset, 'meta') and hasattr(base_dataset, 'index_map'):
            # ShardNPZDataset - read K directly from cached npz files
            import numpy as np
            for idx in indices:
                si, i = base_dataset.index_map[idx]
                p, n, L = base_dataset.meta[si]
                
                # Load only K array (very fast)
                z = base_dataset._npz_cache.get(p)
                if z is None:
                    z = np.load(p, allow_pickle=False, mmap_mode="r")
                    base_dataset._npz_cache[p] = z
                
                K = int(z["K"][i])
                
                if K not in self.k_groups:
                    self.k_groups[K] = []
                self.k_groups[K].append(idx)
        else:
            # Fallback: iterate through dataset (slower but works for any dataset)
            for idx in indices:
                sample = dataset[idx]
                K = int(sample['K'])
                
                if K not in self.k_groups:
                    self.k_groups[K] = []
                self.k_groups[K].append(idx)
        
        # Print distribution
        for k in sorted(self.k_groups.keys()):
            print(f"  K={k}: {len(self.k_groups[k])} samples")
        
        print(f"✅ K-grouped sampling ready ({len(self.k_groups)} unique K values)")
    
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle within each K group if requested
        if self.shuffle:
            for k in self.k_groups:
                np.random.shuffle(self.k_groups[k])
        
        # Create batches from each K group
        batches = []
        for k, indices in self.k_groups.items():
            # Split into batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) > 0:  # Include partial batches
                    batches.append(batch)
        
        # Shuffle batch order if requested
        if self.shuffle:
            np.random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        # Total number of batches
        n_batches = 0
        for indices in self.k_groups.values():
            n_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return n_batches

