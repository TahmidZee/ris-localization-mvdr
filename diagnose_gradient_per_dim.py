#!/usr/bin/env python3
"""
Quick diagnostic: Check gradient flow to phi, theta, r separately.
Run on goose to identify which output dimension has zero gradients.
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, ".")

from ris_pytorch_pipeline.configs import cfg, mdl_cfg
from ris_pytorch_pipeline.model import HybridModel
from ris_pytorch_pipeline.loss import UltimateHybridLoss

# Force deterministic
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create model (no args - reads cfg/mdl_cfg from module globals)
model = HybridModel().to(device)
model.train()

# Create loss (uses default params)
loss_fn = UltimateHybridLoss()
loss_fn.lam_aux = 1.0
loss_fn.lam_cov = 0.0  # Disable cov loss to isolate aux gradient

# Create dummy batch matching model.forward(y, H_full, codes, snr_db=None, R_samp=None)
B = 4
M = cfg.M  # BS antennas
N = cfg.N  # RIS elements
L = cfg.L  # Number of snapshots
Kmax = int(cfg.K_MAX)

print(f"Config: M={M}, N={N}, L={L}, Kmax={Kmax}")

# Fake inputs (RI format = real/imag as last dim with size 2)
y = torch.randn(B, L, M, 2, device=device)  # received signal [B, L, M, 2]
H_full = torch.randn(B, M, N, 2, device=device)  # BS→RIS channel [B, M, N, 2]
codes = torch.randn(B, L, N, 2, device=device)  # RIS codes [B, L, N, 2]
# Normalize codes to unit modulus
codes_complex = torch.complex(codes[..., 0], codes[..., 1])
codes_complex = codes_complex / codes_complex.abs().clamp(min=1e-6)
codes = torch.stack([codes_complex.real, codes_complex.imag], dim=-1)

snr_db = torch.rand(B, device=device) * 20 + 5  # SNR in [5, 25] dB

# Create fake R_true for loss (complex covariance)
R_true = torch.randn(B, N, N, dtype=torch.complex64, device=device)
R_true = R_true @ R_true.conj().transpose(-1, -2)
R_true = (R_true + R_true.conj().transpose(-1, -2)) / 2  # Hermitian

# Fake GT (random angles/ranges)
K_true = torch.randint(1, 3, (B,)).to(device)  # K in [1,2]
phi_t = torch.randn(B, Kmax, device=device) * 0.5  # ~±30°
theta_t = torch.randn(B, Kmax, device=device) * 0.3  # ~±17°
r_t = torch.rand(B, Kmax, device=device) * 5 + 1  # [1, 6] m

# Forward
print("\nRunning model forward...")
out = model(y, H_full, codes, snr_db=snr_db, R_samp=None)

# Extract predictions
phi_p = out["phi_soft"]
theta_p = out["theta_soft"]
r_p = out["r_soft"]

print(f"\n=== Model outputs (first sample) ===")
print(f"phi_p[0]:   {phi_p[0].detach().cpu().numpy()}")
print(f"theta_p[0]: {theta_p[0].detach().cpu().numpy()}")
print(f"r_p[0]:     {r_p[0].detach().cpu().numpy()}")

# Compute separate losses for each dimension
from ris_pytorch_pipeline.loss import _wrapped_huber_loss, _range_huber_loss

mask = torch.arange(Kmax, device=device).unsqueeze(0) < K_true.unsqueeze(1)
mask = mask.float()

phi_loss = (_wrapped_huber_loss(phi_p, phi_t) * mask).sum() / (mask.sum() + 1e-9)
theta_loss = (_wrapped_huber_loss(theta_p, theta_t) * mask).sum() / (mask.sum() + 1e-9)
r_loss = _range_huber_loss(r_p, r_t).mean()

print(f"\n=== Individual loss components ===")
print(f"phi_loss:   {phi_loss.item():.6f}")
print(f"theta_loss: {theta_loss.item():.6f}")
print(f"r_loss:     {r_loss.item():.6f}")

# Zero grads
model.zero_grad()

# Backward each loss separately and check gradients
print(f"\n=== Gradient flow test ===")

# Find the slot_head final linear layer (outputs [phi, theta, r, power, mask])
slot_head_params = list(model.slot_head.parameters())
last_layer_weight = slot_head_params[-2]  # Weight of last Linear
last_layer_bias = slot_head_params[-1]    # Bias of last Linear

print(f"Slot head last layer: weight {last_layer_weight.shape}, bias {last_layer_bias.shape}")

# Test phi gradient
model.zero_grad()
phi_loss.backward(retain_graph=True)
phi_grad_w = last_layer_weight.grad.clone() if last_layer_weight.grad is not None else None
phi_grad_b = last_layer_bias.grad.clone() if last_layer_bias.grad is not None else None

# Test theta gradient
model.zero_grad()
theta_loss.backward(retain_graph=True)
theta_grad_w = last_layer_weight.grad.clone() if last_layer_weight.grad is not None else None
theta_grad_b = last_layer_bias.grad.clone() if last_layer_bias.grad is not None else None

# Test r gradient
model.zero_grad()
r_loss.backward(retain_graph=True)
r_grad_w = last_layer_weight.grad.clone() if last_layer_weight.grad is not None else None
r_grad_b = last_layer_bias.grad.clone() if last_layer_bias.grad is not None else None

print(f"\nGradient norms on slot_head final layer:")
if phi_grad_w is not None:
    # The last layer outputs [phi, theta, r, power, mask] - check gradient per output dim
    print(f"  phi_loss   → ||grad_weight||={phi_grad_w.norm().item():.6f}, ||grad_bias||={phi_grad_b.norm().item():.6f}")
    print(f"             → grad_bias = {phi_grad_b.detach().cpu().numpy()}")
else:
    print(f"  phi_loss   → NO GRADIENT!")

if theta_grad_w is not None:
    print(f"  theta_loss → ||grad_weight||={theta_grad_w.norm().item():.6f}, ||grad_bias||={theta_grad_b.norm().item():.6f}")
    print(f"             → grad_bias = {theta_grad_b.detach().cpu().numpy()}")
else:
    print(f"  theta_loss → NO GRADIENT!")

if r_grad_w is not None:
    print(f"  r_loss     → ||grad_weight||={r_grad_w.norm().item():.6f}, ||grad_bias||={r_grad_b.norm().item():.6f}")
    print(f"             → grad_bias = {r_grad_b.detach().cpu().numpy()}")
else:
    print(f"  r_loss     → NO GRADIENT!")

# Now test full aux loss from loss_fn
print(f"\n=== Full loss function test ===")
# Need fresh forward pass for clean gradients
model.zero_grad()
out2 = model(y, H_full, codes, snr_db=snr_db, R_samp=None)

loss_dict = loss_fn(
    y_pred=out2,
    R_true=R_true,
    phi_theta_r_targets=(phi_t, theta_t, r_t),
    K_true=K_true,
)
total_loss = loss_dict["total"]
print(f"Total loss: {total_loss.item():.6f}")
total_loss.backward()

full_grad_w = last_layer_weight.grad.clone() if last_layer_weight.grad is not None else None
full_grad_b = last_layer_bias.grad.clone() if last_layer_bias.grad is not None else None

if full_grad_w is not None:
    print(f"Full loss → ||grad_weight||={full_grad_w.norm().item():.6f}")
    print(f"          → grad_bias = {full_grad_b.detach().cpu().numpy()}")
    
    # Check per-output-dimension
    # bias is [5]: [phi, theta, r, power, mask]
    print(f"\nPer-output bias gradients:")
    print(f"  phi (dim 0):   {full_grad_b[0].item():.6f}")
    print(f"  theta (dim 1): {full_grad_b[1].item():.6f}")
    print(f"  r (dim 2):     {full_grad_b[2].item():.6f}")
    print(f"  power (dim 3): {full_grad_b[3].item():.6f}")
    print(f"  mask (dim 4):  {full_grad_b[4].item():.6f}")
else:
    print(f"Full loss → NO GRADIENT!")

print("\n=== DIAGNOSIS ===")
if full_grad_b is not None:
    if abs(full_grad_b[0].item()) < 1e-8:
        print("❌ PHI (dim 0) has ZERO gradient!")
    else:
        print("✓ PHI (dim 0) has gradient")
    if abs(full_grad_b[1].item()) < 1e-8:
        print("❌ THETA (dim 1) has ZERO gradient!")
    else:
        print("✓ THETA (dim 1) has gradient")
    if abs(full_grad_b[2].item()) < 1e-8:
        print("❌ R (dim 2) has ZERO gradient!")
    else:
        print("✓ R (dim 2) has gradient")
