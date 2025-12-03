#!/usr/bin/env python3
"""
Train DCD-MUSIC and NF-SubspaceNet baselines on our L=100 dataset
to fairly compare with our L=16 hybrid model.

DCD-MUSIC: Two-stage training (angle branch → range branch)
NF-SubspaceNet: End-to-end training for near-field localization
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

# Add original AI-Subspace-Methods to path
sys.path.insert(0, str(Path(__file__).parent / "AI-Subspace-Methods-Original"))

from src.models_pack.dcd_music import DCDMUSIC
from src.models_pack.subspacenet import SubspaceNet
from src.system_model import SystemModel, SystemModelParams
from src.metrics import RMSPELoss, CartesianLoss


class KGroupedBatchSampler(Sampler):
    """
    Batch sampler that ensures all samples in a batch have the same K.
    Required for DCD-MUSIC which expects uniform K per batch.
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Efficiently group indices by K using shard metadata
        print("[KGroupedBatchSampler] Grouping dataset by K (efficient method)...")
        self.k_to_indices = {}
        
        # Use dataset's internal structure to avoid loading all samples
        if hasattr(dataset, 'shard_files') and hasattr(dataset, 'cumulative_lengths'):
            # Pre-load K values from shards without loading full samples
            for shard_idx, shard_file in enumerate(dataset.shard_files):
                with np.load(shard_file) as data:
                    K_values = data["K"]  # Load only K values
                    start_idx = dataset.cumulative_lengths[shard_idx]
                    
                    for local_idx, k in enumerate(K_values):
                        global_idx = start_idx + local_idx
                        if global_idx < dataset.total_samples:  # Respect max_samples limit
                            k = int(k)
                            if k not in self.k_to_indices:
                                self.k_to_indices[k] = []
                            self.k_to_indices[k].append(global_idx)
        else:
            # Fallback: load samples one by one (slower)
            print("[KGroupedBatchSampler] Using fallback method (slower)...")
            for idx in range(len(dataset)):
                sample = dataset[idx]
                k = sample["K"]
                if k not in self.k_to_indices:
                    self.k_to_indices[k] = []
                self.k_to_indices[k].append(idx)
        
        print(f"[KGroupedBatchSampler] K distribution:")
        for k in sorted(self.k_to_indices.keys()):
            print(f"  K={k}: {len(self.k_to_indices[k])} samples")
        
        # Pre-compute batches
        self._create_batches()
    
    def _create_batches(self):
        self.batches = []
        
        for k, indices in self.k_to_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            
            # Create batches of size batch_size for this K
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) > 0:  # Allow partial batches
                    self.batches.append(batch)
        
        if self.shuffle:
            random.shuffle(self.batches)
    
    def __iter__(self):
        if self.shuffle:
            self._create_batches()  # Re-shuffle for each epoch
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)


class L100RawDataset(Dataset):
    """
    Memory-efficient dataset for loading L=100 raw snapshots.
    Loads data on-demand to avoid memory issues with large shards.
    DCD-MUSIC and NF-SubspaceNet expect raw snapshots, not preprocessed features.
    """
    def __init__(self, shard_dir: Path, max_samples: int = None):
        self.shard_dir = Path(shard_dir)
        self.shard_files = sorted(list(self.shard_dir.glob("shard_*.npz")))
        
        if len(self.shard_files) == 0:
            raise FileNotFoundError(f"No shards found in {self.shard_dir}")
        
        # Count total samples
        self.shard_lengths = []
        for sf in self.shard_files:
            with np.load(sf) as data:
                self.shard_lengths.append(len(data["K"]))
        self.cumulative_lengths = np.cumsum([0] + self.shard_lengths)
        self.total_samples = self.cumulative_lengths[-1]
        
        # Limit samples for memory efficiency
        if max_samples is not None:
            self.total_samples = min(self.total_samples, max_samples)
        
        print(f"[Dataset] Loaded {len(self.shard_files)} shards, {self.total_samples} samples from {self.shard_dir}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict:
        # Limit to available samples
        if idx >= self.total_samples:
            idx = idx % self.total_samples
        
        # Find which shard
        shard_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[shard_idx]
        
        # Load shard on-demand
        with np.load(self.shard_files[shard_idx]) as data:
            Y = data["Y"][local_idx]  # [L, N] complex raw snapshots
            K = int(data["K"][local_idx])
            ptr = data["ptr"][local_idx]  # [3*K_MAX] padded [phi, theta, r]
        
        # Parse ptr: [φ1..φK_MAX, θ1..θK_MAX, r1..rK_MAX]
        K_MAX = len(ptr) // 3
        phi = ptr[:K_MAX][:K].astype(np.float32)
        theta = ptr[K_MAX:2*K_MAX][:K].astype(np.float32)
        r = ptr[2*K_MAX:][:K].astype(np.float32)
        
        return {
            "Y": Y,  # [L, N] complex raw snapshots  
            "K": K,
            "phi": phi,
            "theta": theta,
            "r": r,
        }


def collate_fn_angle(batch: List[Dict]) -> Dict:
    """
    Collate function for DCD-MUSIC angle branch.
    Returns:
        x: [B, N, T] complex raw snapshots
        sources_num: [B] number of sources per sample
        labels: [B, 2*K] stacked as [angles, ranges]
    """
    B = len(batch)
    # Stack raw snapshots: [B, N, T] where T is number of snapshots (L)
    Y_batch = np.stack([b["Y"].T for b in batch])  # Transpose to [N, L] then stack to [B, N, L]
    Y_batch = torch.from_numpy(Y_batch)  # [B, N, L] complex
    
    K_batch = torch.tensor([b["K"] for b in batch], dtype=torch.long)  # [B]
    
    # Determine K from batch (all samples in batch have same K due to K-grouping)
    K = batch[0]["K"]  # All samples have same K
    
    # For DCD-MUSIC: labels must be [B, 2*K] and split at K gives [angles, ranges]
    # where angles=[B, K] contains azimuth angles and ranges=[B, K] contains ranges
    labels = []
    for b in batch:
        # Stack as [phi_1, ..., phi_K, r_1, ..., r_K]
        phi_vec = b["phi"]  # [K]
        r_vec = b["r"]  # [K]
        labels.append(np.concatenate([phi_vec, r_vec]))  # [2*K]
    
    labels = torch.from_numpy(np.stack(labels))  # [B, 2*K]
    
    return Y_batch, K_batch, labels


def collate_fn_range(batch: List[Dict]) -> Dict:
    """
    Collate function for DCD-MUSIC range branch (same as angle for now).
    Returns:
        x: [B, N, T] complex raw snapshots
        sources_num: [B] number of sources per sample
        labels: [B, 2*K] stacked as [angles, ranges]
    """
    return collate_fn_angle(batch)  # Same format for now


def collate_fn_joint(batch: List[Dict]) -> Dict:
    """
    Collate function for DCD-MUSIC joint training (same as angle for now).
    Returns:
        x: [B, N, T] complex raw snapshots
        sources_num: [B] number of sources per sample
        labels: [B, 2*K] stacked as [angles, ranges]
    """
    return collate_fn_angle(batch)  # Same format for now


def collate_fn(batch: List[Dict]) -> Dict:
    """Default collate function (same as angle for compatibility)"""
    return collate_fn_angle(batch)


def train_dcd_music(
    train_loader: DataLoader,
    val_loader: DataLoader,
    system_model: SystemModel,
    tau: int = 100,
    epochs_angle: int = 30,
    epochs_range: int = 30,
    epochs_joint: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    save_dir: Path = Path("results_baselines_L100_12x12/checkpoints/dcd"),
):
    """
    Train DCD-MUSIC in three stages:
    1. Angle branch (ESPRIT)
    2. Range branch (1D MUSIC given ground-truth angles)
    3. Joint fine-tuning
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Training DCD-MUSIC (L=100)")
    print("="*80)
    
    # Initialize model
    model = DCDMUSIC(
        system_model=system_model,
        tau=tau,
        diff_method=("esprit", "music_1d"),  # ESPRIT for angles, 1D MUSIC for range
        regularization=None,
        variant="small",
        norm_layer=True,
        batch_norm=False,
        psd_epsilon=1e-6,
        load_angle_branch=False,
        load_range_branch=False,
    )
    model.to(device)
    
    # ============ Stage 1: Train Angle Branch ============
    print(f"\n[Stage 1/3] Training angle branch ({epochs_angle} epochs)...")
    model.update_train_mode("angle")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_angle)
    
    # Create new loaders with angle-specific collate function
    train_loader_angle = DataLoader(
        train_loader.dataset,
        batch_sampler=train_loader.batch_sampler,
        num_workers=train_loader.num_workers,
        collate_fn=collate_fn_angle,
        pin_memory=train_loader.pin_memory,
    )
    val_loader_angle = DataLoader(
        val_loader.dataset,
        batch_sampler=val_loader.batch_sampler,
        num_workers=val_loader.num_workers,
        collate_fn=collate_fn_angle,
        pin_memory=val_loader.pin_memory,
    )
    
    best_val_angle = 1e9
    for epoch in range(1, epochs_angle + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch_idx, (x, sources_num, labels) in enumerate(train_loader):
            x = x.to(device)
            sources_num = sources_num.to(device)
            labels = labels.to(device)
            
            loss, acc, _ = model.training_step((x, sources_num, labels), batch_idx)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += acc
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch_idx, (x, sources_num, labels) in enumerate(val_loader):
                x = x.to(device)
                sources_num = sources_num.to(device)
                labels = labels.to(device)
                loss, acc = model.validation_step((x, sources_num, labels), batch_idx)
                val_loss += loss.item()
                val_acc += acc
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        print(f"[Angle] Epoch {epoch:03d}/{epochs_angle} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_angle:
            best_val_angle = val_loss
            torch.save(model.angle_branch.state_dict(), save_dir / "dcd_angle_branch.pt")
            print(f"  → Saved best angle branch (val_loss={val_loss:.4f})")
    
    # ============ Stage 2: Train Range Branch ============
    print(f"\n[Stage 2/3] Training range branch ({epochs_range} epochs)...")
    model.update_train_mode("range")
    optimizer = torch.optim.AdamW(model.range_branch.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_range)
    
    best_val_range = 1e9
    for epoch in range(1, epochs_range + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, sources_num, labels) in enumerate(train_loader):
            x = x.to(device)
            sources_num = sources_num.to(device)
            labels = labels.to(device)
            
            loss, acc, _ = model.training_step((x, sources_num, labels), batch_idx)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.range_branch.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x, sources_num, labels) in enumerate(val_loader):
                x = x.to(device)
                sources_num = sources_num.to(device)
                labels = labels.to(device)
                loss, acc = model.validation_step((x, sources_num, labels), batch_idx)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"[Range] Epoch {epoch:03d}/{epochs_range} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_range:
            best_val_range = val_loss
            torch.save(model.range_branch.state_dict(), save_dir / "dcd_range_branch.pt")
            print(f"  → Saved best range branch (val_loss={val_loss:.4f})")
    
    # ============ Stage 3: Joint Fine-tuning ============
    print(f"\n[Stage 3/3] Joint fine-tuning ({epochs_joint} epochs)...")
    model.update_train_mode("position")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr/10, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_joint)
    
    best_val_joint = 1e9
    for epoch in range(1, epochs_joint + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, sources_num, labels) in enumerate(train_loader):
            x = x.to(device)
            sources_num = sources_num.to(device)
            labels = labels.to(device)
            
            loss, acc, _ = model.training_step((x, sources_num, labels), batch_idx)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x, sources_num, labels) in enumerate(val_loader):
                x = x.to(device)
                sources_num = sources_num.to(device)
                labels = labels.to(device)
                loss, acc = model.validation_step((x, sources_num, labels), batch_idx)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"[Joint] Epoch {epoch:03d}/{epochs_joint} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_joint:
            best_val_joint = val_loss
            torch.save(model.state_dict(), save_dir / "dcd_joint.pt")
            print(f"  → Saved best joint model (val_loss={val_loss:.4f})")
    
    print(f"\n✓ DCD-MUSIC training complete!")
    print(f"  Best angle val: {best_val_angle:.4f}")
    print(f"  Best range val: {best_val_range:.4f}")
    print(f"  Best joint val: {best_val_joint:.4f}")
    print(f"  Saved to: {save_dir}\n")


def train_nf_subspacenet(
    train_loader: DataLoader,
    val_loader: DataLoader,
    system_model: SystemModel,
    tau: int = 100,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    save_dir: Path = Path("results_baselines_L100_12x12/checkpoints/nfssn"),
):
    """
    Train NF-SubspaceNet end-to-end for near-field localization.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Training NF-SubspaceNet (L=100)")
    print("="*80)
    
    # Initialize model
    model = SubspaceNet(
        tau=tau,
        diff_method="music",  # MUSIC for near-field
        train_loss_type="rmspe",
        system_model=system_model,
        field_type="near",
        regularization=None,
        variant="small",
        norm_layer=True,
        batch_norm=False,
        psd_epsilon=1e-6,
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    
    best_val = 1e9
    patience = 10
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch_idx, (x, sources_num, labels) in enumerate(train_loader):
            x = x.to(device)
            sources_num = sources_num.to(device)
            labels = labels.to(device)
            
            loss, acc, _ = model.training_step((x, sources_num, labels), batch_idx)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += acc
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch_idx, (x, sources_num, labels) in enumerate(val_loader):
                x = x.to(device)
                sources_num = sources_num.to(device)
                labels = labels.to(device)
                loss, acc = model.validation_step((x, sources_num, labels), batch_idx)
                val_loss += loss.item()
                val_acc += acc
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        print(f"[NFSSN] Epoch {epoch:03d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "nfssn_best.pt")
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
    
    print(f"\n✓ NF-SubspaceNet training complete!")
    print(f"  Best val: {best_val:.4f}")
    print(f"  Saved to: {save_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Train DCD-MUSIC and NF-SubspaceNet on L=100 dataset")
    parser.add_argument("--model", type=str, choices=["dcd", "nfssn", "both"], default="both",
                        help="Which model to train")
    parser.add_argument("--features_root", type=str, default="results_baselines_L100_12x12/features",
                        help="Root directory for L=100 feature shards")
    parser.add_argument("--tau", type=int, default=100, help="Number of time lags")
    parser.add_argument("--epochs_dcd_angle", type=int, default=30, help="DCD angle branch epochs")
    parser.add_argument("--epochs_dcd_range", type=int, default=30, help="DCD range branch epochs")
    parser.add_argument("--epochs_dcd_joint", type=int, default=20, help="DCD joint fine-tuning epochs")
    parser.add_argument("--epochs_nfssn", type=int, default=50, help="NFSSN epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=1, help="DataLoader workers")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup system model (12x12 UPA, 1 GHz, λ/2 spacing)
    params = SystemModelParams()
    params.M = 5  # K_MAX
    params.N = 144  # 12x12 UPA
    params.T = 100  # L=100 snapshots
    params.wavelength = 0.3  # 1 GHz
    params.field_type = "near"
    params.doa_range = 60  # ±60° FOV
    params.max_range_ratio_to_limit = 0.5  # Range limit
    system_model = SystemModel(system_model_params=params)
    
    # Load datasets from raw data shards (use full size for fair comparison with L=16 model)
    # DCD-MUSIC and NF-SubspaceNet expect raw snapshots, not preprocessed features
    data_root = Path("results_baselines_L100_12x12/data/shards")
    train_dataset = L100RawDataset(data_root / "train", max_samples=80000)  # Match L=16 training size
    val_dataset = L100RawDataset(data_root / "val", max_samples=16000)  # Match L=16 validation size
    
    # DCD-MUSIC requires uniform K per batch, so use K-grouped batch sampler
    if args.model in ["dcd", "both"]:
        print("\n[DataLoader] Using K-grouped batch sampler for DCD-MUSIC...")
        train_sampler = KGroupedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_sampler = KGroupedBatchSampler(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device == "cuda"),
        )
    else:
        # NF-SubspaceNet can handle variable K, so use regular DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device == "cuda"),
        )
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val: {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  System: 12x12 UPA, L={args.tau}, λ=0.3m (1 GHz)\n")
    
    # Train models
    if args.model in ["dcd", "both"]:
        train_dcd_music(
            train_loader=train_loader,
            val_loader=val_loader,
            system_model=system_model,
            tau=args.tau,
            epochs_angle=args.epochs_dcd_angle,
            epochs_range=args.epochs_dcd_range,
            epochs_joint=args.epochs_dcd_joint,
            lr=args.lr,
            device=device,
        )
    
    if args.model in ["nfssn", "both"]:
        train_nf_subspacenet(
            train_loader=train_loader,
            val_loader=val_loader,
            system_model=system_model,
            tau=args.tau,
            epochs=args.epochs_nfssn,
            lr=args.lr,
            device=device,
        )
    
    print("\n" + "="*80)
    print("✓ All training complete!")
    print("="*80)


if __name__ == "__main__":
    main()

