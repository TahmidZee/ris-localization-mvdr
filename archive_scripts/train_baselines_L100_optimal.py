#!/usr/bin/env python3
"""
Train DCD-MUSIC and NF-SubspaceNet baselines on L=100 dataset with optimal fairness.
Uses the same preprocessing pipeline as our L=16 model for fair comparison.
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


class L100BaselineDataset(Dataset):
    """
    Dataset for L=100 baseline training using raw snapshots.
    Loads raw snapshots directly from data shards (not features).
    """
    def __init__(self, shard_dir: Path, max_samples: int = None):
        # Point to raw data shards, not features
        # shard_dir should be results_baselines_L100_12x12/data/shards/train
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
        
        # Load raw data shard directly
        with np.load(self.shard_files[shard_idx]) as data:
            # Use 'codes' which are the correct raw snapshots [L, N, 2]
            codes = data["codes"][local_idx]  # [L=100, N=144, 2]
            K = int(data["K"][local_idx])
            ptr = data["ptr"][local_idx]  # [3*K_MAX] padded [phi, theta, r]
            snr = float(data.get("snr", [0.0])[local_idx])  # SNR value
        
        # Convert codes to complex: [L, N]
        Y = codes[:, :, 0] + 1j * codes[:, :, 1]  # [L=100, N=144] complex
        
        # Parse ptr: [φ1..φK_MAX, θ1..θK_MAX, r1..rK_MAX]
        K_MAX = len(ptr) // 3
        phi = ptr[:K_MAX][:K].astype(np.float32)
        theta = ptr[K_MAX:2*K_MAX][:K].astype(np.float32)
        r = ptr[2*K_MAX:][:K].astype(np.float32)
        
        # Return raw snapshots in the format expected by baselines
        # DCD-MUSIC expects [N, T] where T is number of snapshots (L)
        # We transpose Y from [L, N] to [N, L]
        
        return {
            "Y": Y.T,  # [N, L=100] complex raw snapshots (transposed)
            "K": K,
            "phi": phi,
            "theta": theta,
            "r": r,
            "snr": snr,
        }


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


def collate_fn_baseline(batch: List[Dict]) -> Dict:
    """
    Collate function for DCD-MUSIC with proper 3D decoupled format.
    Returns:
        x: [B, N, T] complex raw snapshots (approximate)
        sources_num: [B] number of sources per sample
        labels: [B, 3*K] stacked as [phi, theta, r] for 3D decoupled localization
    """
    B = len(batch)
    # Stack raw snapshots: [B, N, T] where T is number of snapshots
    Y_batch = np.stack([b["Y"].T for b in batch])  # Transpose to [N, T] then stack to [B, N, T]
    Y_batch = torch.from_numpy(Y_batch)  # [B, N, T] complex
    
    K_batch = torch.tensor([b["K"] for b in batch], dtype=torch.long)  # [B]
    
    # Determine K from batch (all samples in batch have same K due to K-grouping)
    K = batch[0]["K"]  # All samples have same K
    
    # For DCD-MUSIC 3D decoupled approach:
    # - Angle branch: estimates (phi, theta) + K from learned covariance
    # - Range branch: estimates r via separate MUSIC stage
    # We need [B, 3*K] format: [phi_1, ..., phi_K, theta_1, ..., theta_K, r_1, ..., r_K]
    labels = []
    for b in batch:
        phi_vec = b["phi"]  # [K] azimuth angles
        theta_vec = b["theta"]  # [K] elevation angles  
        r_vec = b["r"]  # [K] ranges
        
        # Stack as [phi_1, ..., phi_K, theta_1, ..., theta_K, r_1, ..., r_K]
        # This gives DCD-MUSIC all 3D coordinates in the right order
        labels.append(np.concatenate([phi_vec, theta_vec, r_vec]))  # [3*K]
    
    labels = torch.from_numpy(np.stack(labels))  # [B, 3*K]
    
    return Y_batch, K_batch, labels


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
    Train DCD-MUSIC in three stages with full 3D coordinate support.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Training DCD-MUSIC (L=100) - Full 3D Coordinates")
    print("="*80)
    
    # Initialize model
    model = DCDMUSIC(
        system_model=system_model,
        tau=tau,
        diff_method=("esprit", "music_1d"),
        regularization=None,
        variant="small",
        norm_layer=True,
        batch_norm=False,
        psd_epsilon=1e-6,
        load_angle_branch=False,
        load_range_branch=False,
    )
    model.to(device)
    
    # Custom training loop that handles 3D coordinates
    print(f"\n[Stage 1/3] Training angle branch ({epochs_angle} epochs)...")
    model.update_train_mode("angle")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_angle)
    
    best_val_angle = 1e9
    for epoch in range(1, epochs_angle + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch_idx, (x, sources_num, labels_3d) in enumerate(train_loader):
            x = x.to(device)
            sources_num = sources_num.to(device)
            labels_3d = labels_3d.to(device)
            
            # Parse 3D labels [B, 3*K] for DCD-MUSIC decoupled approach
            # labels_3d = [phi_1, ..., phi_K, theta_1, ..., theta_K, r_1, ..., r_K]
            B = x.size(0)  # Batch size
            K = sources_num[0].item()  # All samples have same K due to K-grouping
            phi = labels_3d[:, :K]  # [B, K] azimuth angles
            theta = labels_3d[:, K:2*K]  # [B, K] elevation angles  
            r = labels_3d[:, 2*K:]  # [B, K] ranges
            
            # For DCD-MUSIC with UPA: supports full 3D coordinates (φ, θ, r)
            # DCD-MUSIC angle stage: estimates (φ, θ) + K from learned covariance
            # DCD-MUSIC range stage: estimates r via separate MUSIC stage
            
            # For UPA, DCD-MUSIC expects 2D angles (φ, θ) in the angle branch
            # We need to encode 2D angles properly for DCD-MUSIC's angle branch
            # Strategy: Interleave φ and θ as [φ₁, θ₁, φ₂, θ₂, ..., φₖ, θₖ]
            angles_2d = torch.zeros(B, 2*K, dtype=torch.float32, device=device)
            angles_2d[:, ::2] = phi  # φ at even indices
            angles_2d[:, 1::2] = theta  # θ at odd indices
            
            # For DCD-MUSIC with UPA: angle branch gets 2D angles, range branch gets ranges
            # Format: [φ₁, θ₁, ..., φₖ, θₖ, r₁, ..., rₖ] = [2D_angles, ranges]
            labels_3d_upa = torch.cat([angles_2d, r], dim=1)  # [B, 3*K]
            
            # Create batch format for DCD-MUSIC with UPA 3D support
            batch_data = (x, sources_num, labels_3d_upa)
            
            try:
                loss, acc, _ = model.training_step(batch_data, batch_idx)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_acc += acc
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Warning: OOM at batch {batch_idx}, clearing cache...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"  Warning: Training step failed: {e}")
                    continue
            except Exception as e:
                print(f"  Warning: Training step failed: {e}")
                continue
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch_idx, (x, sources_num, labels_3d) in enumerate(val_loader):
                x = x.to(device)
                sources_num = sources_num.to(device)
                labels_3d = labels_3d.to(device)
                
                # Parse 3D labels for DCD-MUSIC with UPA (full 3D support)
                K = sources_num[0].item()
                phi = labels_3d[:, :K]  # [B, K] azimuth angles
                theta = labels_3d[:, K:2*K]  # [B, K] elevation angles  
                r = labels_3d[:, 2*K:]  # [B, K] ranges
                
                # For UPA, DCD-MUSIC expects 2D angles (φ, θ) in angle branch
                # Interleave φ and θ as [φ₁, θ₁, φ₂, θ₂, ..., φₖ, θₖ]
                angles_2d = torch.zeros(B, 2*K, dtype=torch.float32, device=device)
                angles_2d[:, ::2] = phi  # φ at even indices
                angles_2d[:, 1::2] = theta  # θ at odd indices
                
                # Format: [φ₁, θ₁, ..., φₖ, θₖ, r₁, ..., rₖ] = [2D_angles, ranges]
                labels_3d_upa = torch.cat([angles_2d, r], dim=1)  # [B, 3*K]
                batch_data = (x, sources_num, labels_3d_upa)
                
                try:
                    loss, acc = model.validation_step(batch_data, batch_idx)
                    val_loss += loss.item()
                    val_acc += acc
                except Exception as e:
                    print(f"  Warning: Validation step failed: {e}")
                    continue
        
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
    
    # ========================
    # Stage 2: Range Branch Training
    # ========================
    print(f"\n[Stage 2/3] Training range branch ({epochs_range} epochs)...")
    model.update_train_mode("range")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_range)
    
    best_val_range = 1e9
    for epoch in range(1, epochs_range + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch_idx, (x, sources_num, labels_3d) in enumerate(train_loader):
            x = x.to(device)
            sources_num = sources_num.to(device)
            labels_3d = labels_3d.to(device)
            
            # Parse 3D labels
            B = x.size(0)
            K = sources_num[0].item()
            phi = labels_3d[:, :K]
            theta = labels_3d[:, K:2*K]
            r = labels_3d[:, 2*K:]
            
            # For range training, we need to provide ground truth angles
            # The model will use GT angles to train range estimation
            angles_2d = torch.zeros(B, 2*K, dtype=torch.float32, device=device)
            angles_2d[:, ::2] = phi
            angles_2d[:, 1::2] = theta
            
            labels_3d_upa = torch.cat([angles_2d, r], dim=1)  # [B, 3*K]
            batch_data = (x, sources_num, labels_3d_upa)
            
            try:
                # In range mode, the model returns (angles_pred, ranges_pred, sources_est, eigen_reg)
                # and loss includes both angle and range components
                loss, acc, _ = model.training_step(batch_data, batch_idx)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_acc += acc
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Warning: OOM at batch {batch_idx}, clearing cache...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"  Warning: Training step failed: {e}")
                    continue
            except Exception as e:
                print(f"  Warning: Training step failed: {e}")
                continue
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch_idx, (x, sources_num, labels_3d) in enumerate(val_loader):
                x = x.to(device)
                sources_num = sources_num.to(device)
                labels_3d = labels_3d.to(device)
                
                B = x.size(0)
                K = sources_num[0].item()
                phi = labels_3d[:, :K]
                theta = labels_3d[:, K:2*K]
                r = labels_3d[:, 2*K:]
                
                angles_2d = torch.zeros(B, 2*K, dtype=torch.float32, device=device)
                angles_2d[:, ::2] = phi
                angles_2d[:, 1::2] = theta
                
                labels_3d_upa = torch.cat([angles_2d, r], dim=1)
                batch_data = (x, sources_num, labels_3d_upa)
                
                try:
                    loss, acc = model.validation_step(batch_data, batch_idx)
                    val_loss += loss.item()
                    val_acc += acc
                except Exception as e:
                    print(f"  Warning: Validation step failed: {e}")
                    continue
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        print(f"[Range] Epoch {epoch:03d}/{epochs_range} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_range:
            best_val_range = val_loss
            torch.save(model.range_branch.state_dict(), save_dir / "dcd_range_branch.pt")
            print(f"  → Saved best range branch (val_loss={val_loss:.4f})")
    
    # ========================
    # Stage 3: Joint Fine-tuning (Position Mode)
    # ========================
    print(f"\n[Stage 3/3] Joint fine-tuning ({epochs_joint} epochs)...")
    model.update_train_mode("position")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-5)  # Lower LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_joint)
    
    best_val_joint = 1e9
    for epoch in range(1, epochs_joint + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch_idx, (x, sources_num, labels_3d) in enumerate(train_loader):
            x = x.to(device)
            sources_num = sources_num.to(device)
            labels_3d = labels_3d.to(device)
            
            B = x.size(0)
            K = sources_num[0].item()
            phi = labels_3d[:, :K]
            theta = labels_3d[:, K:2*K]
            r = labels_3d[:, 2*K:]
            
            angles_2d = torch.zeros(B, 2*K, dtype=torch.float32, device=device)
            angles_2d[:, ::2] = phi
            angles_2d[:, 1::2] = theta
            
            labels_3d_upa = torch.cat([angles_2d, r], dim=1)  # [B, 3*K]
            batch_data = (x, sources_num, labels_3d_upa)
            
            try:
                loss, acc, _ = model.training_step(batch_data, batch_idx)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_acc += acc
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Warning: OOM at batch {batch_idx}, clearing cache...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"  Warning: Training step failed: {e}")
                    continue
            except Exception as e:
                print(f"  Warning: Training step failed: {e}")
                continue
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch_idx, (x, sources_num, labels_3d) in enumerate(val_loader):
                x = x.to(device)
                sources_num = sources_num.to(device)
                labels_3d = labels_3d.to(device)
                
                B = x.size(0)
                K = sources_num[0].item()
                phi = labels_3d[:, :K]
                theta = labels_3d[:, K:2*K]
                r = labels_3d[:, 2*K:]
                
                angles_2d = torch.zeros(B, 2*K, dtype=torch.float32, device=device)
                angles_2d[:, ::2] = phi
                angles_2d[:, 1::2] = theta
                
                labels_3d_upa = torch.cat([angles_2d, r], dim=1)
                batch_data = (x, sources_num, labels_3d_upa)
                
                try:
                    loss, acc = model.validation_step(batch_data, batch_idx)
                    val_loss += loss.item()
                    val_acc += acc
                except Exception as e:
                    print(f"  Warning: Validation step failed: {e}")
                    continue
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        print(f"[Joint] Epoch {epoch:03d}/{epochs_joint} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_joint:
            best_val_joint = val_loss
            torch.save(model.state_dict(), save_dir / "dcd_music_full.pt")
            print(f"  → Saved best full model (val_loss={val_loss:.4f})")
    
    print(f"\n✓ DCD-MUSIC 3-stage training complete!")
    print(f"  Best angle val: {best_val_angle:.4f}")
    print(f"  Best range val: {best_val_range:.4f}")
    print(f"  Best joint val: {best_val_joint:.4f}")
    print(f"  Note: Using UPA 3D format (φ,θ,r) with decoupled angle/range stages")
    print(f"  Saved to: {save_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Train baselines with optimal fairness")
    parser.add_argument("--model", type=str, choices=["dcd", "nfssn", "both"], default="both",
                        help="Which model to train")
    parser.add_argument("--data_dir", type=str, default="results_baselines_L100_12x12/data/shards",
                        help="Directory for L=100 raw data shards")
    parser.add_argument("--tau", type=int, default=100, help="Number of time lags")
    parser.add_argument("--epochs_dcd_angle", type=int, default=30, help="DCD angle branch epochs")
    parser.add_argument("--epochs_dcd_range", type=int, default=30, help="DCD range branch epochs")
    parser.add_argument("--epochs_dcd_joint", type=int, default=20, help="DCD joint fine-tuning epochs")
    parser.add_argument("--epochs_nfssn", type=int, default=50, help="NFSSN epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (can increase to 128+ on GPU)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=1, help="DataLoader workers")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Memory optimization for GPU
    if device == "cuda":
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable memory efficient attention if available
        torch.backends.cuda.enable_flash_sdp(True)
    
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
    
    # Load datasets (use full size for fair comparison with L=16 model)
    data_root = Path(args.data_dir)
    train_dataset = L100BaselineDataset(data_root / "train", max_samples=80000)
    val_dataset = L100BaselineDataset(data_root / "val", max_samples=16000)
    
    # DCD-MUSIC requires uniform K per batch, so use K-grouped batch sampler
    if args.model in ["dcd", "both"]:
        print("\n[DataLoader] Using K-grouped batch sampler for DCD-MUSIC...")
        train_sampler = KGroupedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_sampler = KGroupedBatchSampler(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn_baseline,
            pin_memory=(device == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn_baseline,
            pin_memory=(device == "cuda"),
        )
    else:
        # NF-SubspaceNet can handle variable K, so use regular DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn_baseline,
            pin_memory=(device == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_baseline,
            pin_memory=(device == "cuda"),
        )
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val: {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  System: 12x12 UPA, L={args.tau}, λ=0.3m (1 GHz)")
    
    # Memory usage estimation
    input_memory_mb = args.batch_size * 144 * 100 * 8 / 1e6  # Complex64 = 8 bytes
    print(f"  Estimated input memory: {input_memory_mb:.1f} MB per batch")
    
    if device == "cuda":
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU memory: {gpu_memory_gb:.1f} GB")
        if gpu_memory_gb >= 16:
            print(f"  💡 Recommendation: Can use batch_size=128+ on this GPU")
        elif gpu_memory_gb >= 8:
            print(f"  💡 Recommendation: Can use batch_size=64-96 on this GPU")
        else:
            print(f"  ⚠️  Warning: Limited GPU memory, consider batch_size=32-48")
    print()
    
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
    
    print("\n" + "="*80)
    print("✓ All training complete!")
    print("="*80)


if __name__ == "__main__":
    main()
