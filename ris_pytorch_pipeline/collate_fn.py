"""
Collate function with padding for variable-K batches
==================================================

Pads ground truth arrays (phi, theta, r) to K_max and adds a mask to handle
variable number of sources per sample.
"""

import torch
from typing import List, Dict, Any


def collate_pad_to_kmax(batch: List[Dict[str, Any]], K_max: int) -> Dict[str, torch.Tensor]:
    """
    Collate function that pads variable-length arrays to K_max.
    
    Args:
        batch: List of sample dictionaries
        K_max: Maximum number of sources to pad to
        
    Returns:
        Dictionary with padded tensors and mask
    """
    out = {}
    
    # Get batch size
    B = len(batch)
    
    # Process each key in the batch
    for key in batch[0].keys():
        if key in ["phi", "theta", "r"]:  # Variable-length arrays [K]
            # Pad to K_max
            padded = []
            for sample in batch:
                x = torch.as_tensor(sample[key], dtype=torch.float32)  # [K]
                pad_size = K_max - x.numel()
                if pad_size > 0:
                    pad = torch.zeros(pad_size, dtype=x.dtype)
                    padded.append(torch.cat([x, pad], dim=0))  # [K_max]
                else:
                    padded.append(x[:K_max])  # Truncate if K > K_max
            out[key] = torch.stack(padded, dim=0)  # [B, K_max]
            
        elif key == "K":
            # K values and mask
            Ks = torch.as_tensor([sample["K"] for sample in batch], dtype=torch.long)  # [B]
            # CRITICAL: enforce 1-based K (1..K_max) to match loss expectations
            if torch.any(Ks < 1) or torch.any(Ks > K_max):
                raise ValueError(f"K out of range in collate: min={Ks.min().item()}, max={Ks.max().item()}, expected [1,{K_max}]")
            out["K"] = Ks
            
            # Create mask: True for valid sources, False for padding
            ar = torch.arange(K_max, device=Ks.device).unsqueeze(0)  # [1, K_max]
            out["k_mask"] = (ar < Ks.unsqueeze(1))  # [B, K_max] bool
            
        else:
            # Other tensors (y, H, codes, snr_db) - stack normally
            out[key] = torch.stack([torch.as_tensor(sample[key]) for sample in batch], dim=0)
    
    return out


def collate_pad_to_kmax_with_snr(batch: List[Dict[str, Any]], K_max: int) -> Dict[str, torch.Tensor]:
    """
    Enhanced collate function that handles SNR loading and padding.
    
    Args:
        batch: List of sample dictionaries
        K_max: Maximum number of sources to pad to
        
    Returns:
        Dictionary with padded tensors, mask, and SNR
    """
    out = {}
    
    # Get batch size
    B = len(batch)
    
    # Process each key in the batch
    for key in batch[0].keys():
        if key in ["phi", "theta", "r"]:  # Variable-length arrays [K]
            # Pad to K_max
            padded = []
            for sample in batch:
                x = torch.as_tensor(sample[key], dtype=torch.float32)  # [K]
                pad_size = K_max - x.numel()
                if pad_size > 0:
                    pad = torch.zeros(pad_size, dtype=x.dtype)
                    padded.append(torch.cat([x, pad], dim=0))  # [K_max]
                else:
                    padded.append(x[:K_max])  # Truncate if K > K_max
            out[key] = torch.stack(padded, dim=0)  # [B, K_max]
            
        elif key == "K":
            # K values and mask
            Ks = torch.as_tensor([sample["K"] for sample in batch], dtype=torch.long)  # [B]
            # CRITICAL: enforce 1-based K (1..K_max) to match loss expectations
            if torch.any(Ks < 1) or torch.any(Ks > K_max):
                raise ValueError(f"K out of range in collate: min={Ks.min().item()}, max={Ks.max().item()}, expected [1,{K_max}]")
            out["K"] = Ks
            
            # Create mask: True for valid sources, False for padding
            ar = torch.arange(K_max, device=Ks.device).unsqueeze(0)  # [1, K_max]
            out["k_mask"] = (ar < Ks.unsqueeze(1))  # [B, K_max] bool
            
        elif key == "snr_db":
            # Handle SNR (check both 'snr' and 'snr_db' keys)
            snr_values = []
            for sample in batch:
                if "snr_db" in sample:
                    snr_values.append(torch.as_tensor(sample["snr_db"], dtype=torch.float32))
                elif "snr" in sample:
                    snr_values.append(torch.as_tensor(sample["snr"], dtype=torch.float32))
                else:
                    snr_values.append(torch.tensor(0.0, dtype=torch.float32))  # Default SNR
            out[key] = torch.stack(snr_values, dim=0)  # [B]
            
        elif key == "snr":
            # Handle SNR from 'snr' key and map to 'snr_db'
            snr_values = []
            for sample in batch:
                if "snr" in sample:
                    snr_values.append(torch.as_tensor(sample["snr"], dtype=torch.float32))
                else:
                    snr_values.append(torch.tensor(0.0, dtype=torch.float32))  # Default SNR
            out["snr_db"] = torch.stack(snr_values, dim=0)  # [B] - map to snr_db
            
        else:
            # Other tensors (y, H, codes) - stack normally
            out[key] = torch.stack([torch.as_tensor(sample[key]) for sample in batch], dim=0)
    
    return out
