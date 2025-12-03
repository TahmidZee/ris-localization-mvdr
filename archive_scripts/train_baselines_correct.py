#!/usr/bin/env python3
"""
Correct training script for DCD-MUSIC and NF-SubspaceNet baselines.
Follows the exact procedures described in the papers:
- DCD-MUSIC: 3-stage training (angle → range → position)
- NF-SubspaceNet: Single network with spectrum-aware training
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Add original AI-Subspace-Methods to path
sys.path.insert(0, str(Path(__file__).parent / "AI-Subspace-Methods-Original"))

from src.models_pack.dcd_music import DCDMUSIC
from src.models_pack.subspacenet import SubspaceNet
from src.system_model import SystemModel, SystemModelParams
from src.metrics import RMSPELoss, CartesianLoss
from src.training import Trainer, TrainingParamsNew
from src.models import ModelGenerator


class BaselineDataset(Dataset):
    """Dataset for baseline training using generated data."""
    
    def __init__(self, data_dir: Path, method: str, max_samples: int = None):
        self.data_dir = Path(data_dir)
        self.method = method
        self.shard_files = sorted(list(self.data_dir.glob("shard_*.npz")))
        
        if len(self.shard_files) == 0:
            raise FileNotFoundError(f"No shards found in {self.data_dir}")
        
        # Count total samples
        self.shard_lengths = []
        for sf in self.shard_files:
            with np.load(sf) as data:
                self.shard_lengths.append(len(data["K"]))
        self.cumulative_lengths = np.cumsum([0] + self.shard_lengths)
        self.total_samples = self.cumulative_lengths[-1]
        
        if max_samples is not None:
            self.total_samples = min(self.total_samples, max_samples)
        
        print(f"[Dataset] Loaded {len(self.shard_files)} shards, {self.total_samples} samples from {self.data_dir}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict:
        if idx >= self.total_samples:
            idx = idx % self.total_samples
        
        # Find which shard
        shard_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[shard_idx]
        
        # Load data
        with np.load(self.shard_files[shard_idx]) as data:
            K = int(data["K"][local_idx])
            ptr = data["ptr"][local_idx]  # [3*K_MAX] padded [phi, theta, r]
            snr = float(data.get("snr", [0.0])[local_idx])
            
            if self.method == "dcd":
                # DCD-MUSIC: X shape [N=15, L=100, 2]
                X = data["X"][local_idx]  # [15, 100, 2]
                Y = X[:, :, 0] + 1j * X[:, :, 1]  # [15, 100] complex
            else:  # nfssn
                # NF-SubspaceNet: Y shape [N=144, L=100, 2]
                Y_raw = data["Y"][local_idx]  # [144, 100, 2]
                Y = Y_raw[:, :, 0] + 1j * Y_raw[:, :, 1]  # [144, 100] complex
        
        # Parse ptr: [φ1..φK_MAX, θ1..θK_MAX, r1..rK_MAX]
        K_MAX = len(ptr) // 3
        phi = ptr[:K_MAX][:K].astype(np.float32)
        theta = ptr[K_MAX:2*K_MAX][:K].astype(np.float32)
        r = ptr[2*K_MAX:][:K].astype(np.float32)
        
        # Add small regularization to measurements to improve conditioning
        Y = Y + 1e-6 * (np.random.randn(*Y.shape) + 1j * np.random.randn(*Y.shape))
        
        return {
            "Y": Y,  # [N, L] complex measurements
            "K": K,
            "phi": phi,
            "theta": theta,
            "r": r,
            "snr": snr,
        }


class KGroupedBatchSampler(Sampler):
    """
    Batch sampler that groups samples by K (number of sources) to ensure
    all samples in a batch have the same K value.
    """
    def __init__(self, dataset: BaselineDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Group indices by K
        self.k_groups = {}
        for idx in range(len(dataset)):
            # Get K value for this sample
            shard_idx = np.searchsorted(dataset.cumulative_lengths[1:], idx, side='right')
            local_idx = idx - dataset.cumulative_lengths[shard_idx]
            
            with np.load(dataset.shard_files[shard_idx]) as data:
                K = int(data["K"][local_idx])
            
            if K not in self.k_groups:
                self.k_groups[K] = []
            self.k_groups[K].append(idx)
        
        # Create batches
        self.batches = []
        for K, indices in self.k_groups.items():
            # Shuffle indices for this K
            np.random.shuffle(indices)
            
            # Create batches of size batch_size
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]
                if len(batch) == batch_size:  # Only use full batches
                    self.batches.append(batch)
        
        # Shuffle batches
        np.random.shuffle(self.batches)
    
    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for baseline training."""
    B = len(batch)
    
    # Stack measurements: [B, N, L] complex
    Y_batch = np.stack([b["Y"] for b in batch])
    Y_batch = torch.from_numpy(Y_batch)
    
    # Stack K values
    K_batch = torch.tensor([b["K"] for b in batch], dtype=torch.long)
    
    # For DCD-MUSIC: labels should be [angles, ranges] format
    # DCD-MUSIC expects: [B, 2*K] where first K are angles, next K are ranges
    # Since all samples in batch have same K (due to K-grouping), we can use K directly
    K = batch[0]["K"]  # All samples have same K due to K-grouping
    labels = []
    for b in batch:
        phi_vec = b["phi"][:K]
        r_vec = b["r"][:K]
        
        # For DCD-MUSIC: combine phi and r as "angles" and "ranges"
        # Format: [angles, ranges] = [phi1, phi2, ..., phiK, r1, r2, ..., rK]
        angles = phi_vec  # [K] - DCD-MUSIC uses only phi (azimuth) for angles
        ranges = r_vec    # [K]
        
        labels.append(np.concatenate([angles, ranges]))  # [2*K]
    
    labels = torch.from_numpy(np.stack(labels))
    
    return Y_batch, K_batch, labels


def train_dcd_music_correct(train_loader, val_loader, device, epochs_per_stage=30, lr=1e-3, save_dir="checkpoints/dcd"):
    """
    Train DCD-MUSIC following the correct 3-stage procedure from the paper:
    1. Angle training: Train ψ_a with ESPRIT, minimize RMSPE of DoAs
    2. Range training: Train ψ_r with 1D MUSIC, minimize range error
    3. Position fine-tuning: Joint training with Cartesian position loss
    """
    print(f"🎯 Training DCD-MUSIC with correct 3-stage procedure...")
    
    # System model for ULA N=15
    system_params = SystemModelParams()
    system_params.N = 15  # ULA elements
    system_params.M = 5  # Max sources
    system_params.T = 100  # Snapshots
    system_params.wavelength = 0.3  # 1 GHz
    system_params.field_type = "near"  # Near-field
    system_params.signal_type = "narrowband"
    system_params.signal_nature = "non-coherent"
    system_model = SystemModel(system_params)
    
    # DCD-MUSIC model with correct parameters
    model = DCDMUSIC(
        system_model=system_model,
        tau=100,
        diff_method=("esprit", "music_1d"),  # ESPRIT for angles, 1D MUSIC for range
        regularization=None,
        variant="small"
    )
    model = model.to(device)
    
    # Training parameters
    training_params = TrainingParamsNew(
        learning_rate=lr,
        weight_decay=1e-9,
        epochs=epochs_per_stage,
        optimizer="Adam",
        scheduler="ReduceLROnPlateau",
        batch_size=32,  # Add batch_size parameter
        training_objective="angle"  # Add training objective
    )
    
    # Stage 1: Angle Training
    print("📐 Stage 1: Training angle branch (ψ_a) with ESPRIT...")
    model.update_train_mode("angle")
    training_params["training_objective"] = "angle"
    trainer = Trainer(model, training_params, show_plots=False)
    trainer.train(train_loader, val_loader, use_wandb=False, save_final=False)
    
    # Stage 2: Range Training
    print("📏 Stage 2: Training range branch (ψ_r) with 1D MUSIC...")
    model.update_train_mode("range")
    training_params["training_objective"] = "range"
    trainer = Trainer(model, training_params, show_plots=False)
    trainer.train(train_loader, val_loader, use_wandb=False, save_final=False)
    
    # Stage 3: Position Fine-tuning
    print("🎯 Stage 3: Joint position fine-tuning...")
    model.update_train_mode("position")
    training_params["training_objective"] = "angle, range"
    trainer = Trainer(model, training_params, show_plots=False)
    trainer.train(train_loader, val_loader, use_wandb=False, save_final=False)
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/dcd_music_correct.pt")
    print(f"✅ DCD-MUSIC model saved to {save_dir}/dcd_music_correct.pt")


def train_nf_subspacenet_correct(train_loader, val_loader, device, epochs=50, lr=1e-3, save_dir="checkpoints/nfssn"):
    """
    Train NF-SubspaceNet following the correct procedure from the paper:
    - Single network with spectrum-aware training
    - Surrogate covariance: R̂ = K K^H + εI
    - Losses: Spectrum loss + Model-order loss
    """
    print(f"🎯 Training NF-SubspaceNet with correct spectrum-aware procedure...")
    
    # System model for UPA N=144
    system_params = SystemModelParams()
    system_params.N = 144  # UPA elements
    system_params.M = 5  # Max sources
    system_params.T = 100  # Snapshots
    system_params.wavelength = 0.3  # 1 GHz
    system_params.field_type = "near"  # Near-field
    system_params.signal_type = "narrowband"
    system_params.signal_nature = "non-coherent"
    system_model = SystemModel(system_params)
    
    # NF-SubspaceNet model with correct parameters
    model = SubspaceNet(
        tau=100,
        diff_method="music_2D",  # 2D MUSIC for near-field
        train_loss_type="music_spectrum",  # Spectrum-aware loss
        system_model=system_model,
        field_type="near",
        regularization=None,
        variant="small"
    )
    model = model.to(device)
    
    # Training parameters
    training_params = TrainingParamsNew(
        learning_rate=lr,
        weight_decay=1e-9,
        epochs=epochs,
        optimizer="Adam",
        scheduler="ReduceLROnPlateau",
        batch_size=32,  # Add batch_size parameter
        training_objective="angle, range"  # Add training objective
    )
    
    # Single-stage training (spectrum-aware)
    print("🎯 Training NF-SubspaceNet with spectrum-aware loss...")
    trainer = Trainer(model, training_params, show_plots=False)
    trainer.train(train_loader, val_loader, use_wandb=False, save_final=False)
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/nf_subspacenet_correct.pt")
    print(f"✅ NF-SubspaceNet model saved to {save_dir}/nf_subspacenet_correct.pt")


def main():
    parser = argparse.ArgumentParser(description="Train baseline models with correct procedures")
    parser.add_argument("--model", type=str, required=True, choices=["dcd", "nfssn", "both"],
                       help="Which model to train")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Data directory (e.g., data_dcd_benchmark or data_nfssn_benchmark)")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs per stage (DCD) or total (NF-SubspaceNet)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to use (for testing)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.model in ["dcd", "both"]:
        print("🎯 Training DCD-MUSIC with correct 3-stage procedure...")
        
        # Load DCD-MUSIC data
        train_dataset = BaselineDataset(
            Path(args.data_dir) / "train", 
            method="dcd",
            max_samples=args.max_samples
        )
        val_dataset = BaselineDataset(
            Path(args.data_dir) / "val", 
            method="dcd",
            max_samples=args.max_samples
        )
        
        # Use K-grouped batch sampler for DCD-MUSIC
        train_sampler = KGroupedBatchSampler(train_dataset, args.batch_size)
        val_sampler = KGroupedBatchSampler(val_dataset, args.batch_size)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        train_dcd_music_correct(
            train_loader, 
            val_loader, 
            device, 
            epochs_per_stage=args.epochs, 
            lr=args.lr,
            save_dir="checkpoints/dcd"
        )
    
    if args.model in ["nfssn", "both"]:
        print("🎯 Training NF-SubspaceNet with correct spectrum-aware procedure...")
        
        # Load NF-SubspaceNet data
        train_dataset = BaselineDataset(
            Path(args.data_dir) / "train", 
            method="nfssn",
            max_samples=args.max_samples
        )
        val_dataset = BaselineDataset(
            Path(args.data_dir) / "val", 
            method="nfssn",
            max_samples=args.max_samples
        )
        
        # Use K-grouped batch sampler for NF-SubspaceNet too
        train_sampler = KGroupedBatchSampler(train_dataset, args.batch_size)
        val_sampler = KGroupedBatchSampler(val_dataset, args.batch_size)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        train_nf_subspacenet_correct(
            train_loader, 
            val_loader, 
            device, 
            epochs=args.epochs, 
            lr=args.lr,
            save_dir="checkpoints/nfssn"
        )
    
    print("✅ Training completed with correct procedures!")


if __name__ == "__main__":
    main()
