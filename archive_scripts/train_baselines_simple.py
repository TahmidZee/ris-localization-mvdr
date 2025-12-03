#!/usr/bin/env python3
"""
Simple training script for DCD-MUSIC and NF-SubspaceNet baselines.
Works directly with the generated baseline datasets (data_dcd_benchmark, data_nfssn_benchmark).
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Add original AI-Subspace-Methods to path
sys.path.insert(0, str(Path(__file__).parent / "AI-Subspace-Methods-Original"))

from src.models_pack.dcd_music import DCDMUSIC
from src.models_pack.subspacenet import SubspaceNet
from src.system_model import SystemModel, SystemModelParams
from src.metrics import RMSPELoss, CartesianLoss


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
        
        return {
            "Y": Y,  # [N, L] complex measurements
            "K": K,
            "phi": phi,
            "theta": theta,
            "r": r,
            "snr": snr,
        }


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for baseline training."""
    B = len(batch)
    
    # Stack measurements: [B, N, L] complex
    Y_batch = np.stack([b["Y"] for b in batch])
    Y_batch = torch.from_numpy(Y_batch)
    
    # Stack K values
    K_batch = torch.tensor([b["K"] for b in batch], dtype=torch.long)
    
    # Stack labels: [B, 3*K_MAX]
    K = batch[0]["K"]  # All samples have same K due to K-grouping
    labels = []
    for b in batch:
        phi_vec = b["phi"][:K]
        theta_vec = b["theta"][:K]
        r_vec = b["r"][:K]
        labels.append(np.concatenate([phi_vec, theta_vec, r_vec]))
    
    labels = torch.from_numpy(np.stack(labels))
    
    return Y_batch, K_batch, labels


def train_dcd_music(train_loader, val_loader, device, epochs=50, lr=1e-3, save_dir="checkpoints/dcd"):
    """Train DCD-MUSIC model."""
    print(f"🎯 Training DCD-MUSIC for {epochs} epochs...")
    
    # System model for ULA N=15
    system_params = SystemModelParams()
    system_params.N = 15  # ULA elements - set directly
    system_params.M = 5  # Max sources
    system_params.T = 100  # Snapshots
    system_params.wavelength = 0.3  # 1 GHz
    system_params.field_type = "near"  # Near-field
    system_params.signal_type = "narrowband"
    system_params.signal_nature = "non-coherent"
    system_model = SystemModel(system_params)
    
    # DCD-MUSIC model
    model = DCDMUSIC(system_model, tau=100)
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = RMSPELoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (Y, K, labels) in enumerate(train_loader):
            Y = Y.to(device)
            K = K.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            try:
                # DCD-MUSIC expects [B, N, T] format
                output = model(Y, K)
                loss = criterion(output, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/dcd_music.pt")
    print(f"✅ DCD-MUSIC model saved to {save_dir}/dcd_music.pt")


def train_nf_subspacenet(train_loader, val_loader, device, epochs=50, lr=1e-3, save_dir="checkpoints/nfssn"):
    """Train NF-SubspaceNet model."""
    print(f"🎯 Training NF-SubspaceNet for {epochs} epochs...")
    
    # System model for UPA N=144
    system_params = SystemModelParams()
    system_params.N = 144  # UPA elements - set directly
    system_params.M = 5  # Max sources
    system_params.T = 100  # Snapshots
    system_params.wavelength = 0.3  # 1 GHz
    system_params.field_type = "near"  # Near-field
    system_params.signal_type = "narrowband"
    system_params.signal_nature = "non-coherent"
    system_model = SystemModel(system_params)
    
    # NF-SubspaceNet model
    model = SubspaceNet(tau=100, system_model=system_model)
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = RMSPELoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (Y, K, labels) in enumerate(train_loader):
            Y = Y.to(device)
            K = K.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            try:
                # NF-SubspaceNet expects [B, N, T] format
                output = model(Y, K)
                loss = criterion(output, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/nf_subspacenet.pt")
    print(f"✅ NF-SubspaceNet model saved to {save_dir}/nf_subspacenet.pt")


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--model", type=str, required=True, choices=["dcd", "nfssn", "both"],
                       help="Which model to train")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Data directory (e.g., data_dcd_benchmark or data_nfssn_benchmark)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
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
        print("🎯 Training DCD-MUSIC...")
        
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
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=2
        )
        
        train_dcd_music(
            train_loader, 
            val_loader, 
            device, 
            epochs=args.epochs, 
            lr=args.lr,
            save_dir="checkpoints/dcd"
        )
    
    if args.model in ["nfssn", "both"]:
        print("🎯 Training NF-SubspaceNet...")
        
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
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=2
        )
        
        train_nf_subspacenet(
            train_loader, 
            val_loader, 
            device, 
            epochs=args.epochs, 
            lr=args.lr,
            save_dir="checkpoints/nfssn"
        )
    
    print("✅ Training completed!")


if __name__ == "__main__":
    main()
