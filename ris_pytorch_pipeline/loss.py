
import torch, torch.nn as nn, torch.nn.functional as F
from .configs import cfg, mdl_cfg
from .physics import shrink
from .covariance_utils import trace_norm_torch, shrink_torch, build_effective_cov_torch
import math

def _ri_to_c(x_ri):
    return torch.complex(x_ri[...,0], x_ri[...,1])

def _wrap_angle(x):
    """Wrap angles to [-π, π]"""
    return ((x + math.pi) % (2 * math.pi)) - math.pi

def _wrapped_huber_loss(pred, gt, delta=math.pi/720):  # delta = 0.25° in radians
    """Huber loss that respects angular wrap-around"""
    d = _wrap_angle(pred - gt)
    abs_d = torch.abs(d)
    return torch.where(abs_d <= delta, 
                      0.5 * (d ** 2) / delta, 
                      abs_d - 0.5 * delta)

def _range_huber_loss(pred_r, gt_r, delta=0.2):
    """Huber loss on log-range for better scale handling"""
    pred_r_pos = torch.clamp(pred_r, min=cfg.RANGE_R[0] * 0.9)
    gt_r_pos = torch.clamp(gt_r, min=cfg.RANGE_R[0] * 0.9)
    pr = torch.log(pred_r_pos)
    gr = torch.log(gt_r_pos)
    e = torch.abs(pr - gr)
    return torch.where(e < delta, 0.5 * (e ** 2) / delta, e - 0.5 * delta)

def _vec2c(v):
    v = v.float()
    xr, xi = v[:, ::2], v[:, 1::2]
    A = torch.complex(xr.view(-1, cfg.N, cfg.K_MAX), xi.view(-1, cfg.N, cfg.K_MAX)).to(torch.complex64)

    # Magnitude leash: normalize columns to prevent rare huge factor spikes from overflowing
    # before downstream conditioning/trace-normalization.
    if bool(getattr(cfg, "FACTOR_COLNORM_ENABLE", True)):
        eps = float(getattr(cfg, "FACTOR_COLNORM_EPS", 1e-6))
        max_norm = float(getattr(cfg, "FACTOR_COLNORM_MAX", 1e3))
        col = torch.linalg.norm(A, dim=-2, keepdim=True).clamp_min(eps)  # [B,1,K]
        # Optional upper bound (keeps division from magnifying tiny columns too much)
        if max_norm > 0:
            col = col.clamp(max=max_norm)
        A = A / col
    return A

def _steer_torch(phi, theta, r):
    B, K = phi.shape
    device = phi.device
    h = torch.linspace(-(cfg.N_H - 1)//2, (cfg.N_H - 1)//2, steps=cfg.N_H, device=device) * cfg.d_H
    v = torch.linspace(-(cfg.N_V - 1)//2, (cfg.N_V - 1)//2, steps=cfg.N_V, device=device) * cfg.d_V
    H, V = torch.meshgrid(h, v, indexing="xy")
    hv = torch.stack([H.reshape(-1), V.reshape(-1)], dim=-1)[:cfg.N]
    vh = hv[:,0].view(1,1,cfg.N)
    vv = hv[:,1].view(1,1,cfg.N)
    sin_phi   = torch.sin(phi).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    r_eff = torch.clamp(r, min=1e-6).unsqueeze(-1)
    dist = r.unsqueeze(-1) - vh*sin_phi*cos_theta - vv*sin_theta + (vh**2 + vv**2)/(2.0*r_eff)
    phase = cfg.k0 * dist
    a = torch.exp(1j * phase) / (cfg.N ** 0.5)
    return a.transpose(-1, -2).contiguous()



class UltimateHybridLoss(nn.Module):
    """
    Structured loss for hybrid CNN+Transformer with covariance surrogate
    and geometric auxiliaries.

    Terms (K-free version for MVDR localization):
      • NMSE (diag/off-diag, per-element, size-invariant)
      • Stiefel orthogonality (QR retraction first)
      • Cross-term consistency (light Gram matching)
      • Eigengap hinge (for covariance quality)
      • Aux L2 on (phi, theta, range[log-space]) (lam_aux)
      • Chamfer on angles (radians) (lam_peak)
      • Subspace alignment (training-inference alignment)
      • Heatmap supervision for SpectrumRefiner (optional)
      
    NOTE: K classification removed - using MVDR peak detection instead.

    Weights are set externally with a 3-phase curriculum in Trainer.
    """

    def __init__(
        self,
        lam_cov: float   = 0.10,    # Covariance NMSE weight (primary objective)
        lam_cov_pred: float = 0.02,  # Small auxiliary NMSE on R_pred to prevent hiding
        lam_diag: float = 0.2,
        lam_off: float  = 0.8,
        lam_ortho: float = 1e-3,
        lam_cross: float = 0.0,
        # Eigengap/margin regularizers are disabled in the current system (MVDR+Refiner).
        # They rely on SVD/eigenvalue backprop and were a major source of NaN gradients in HPO.
        lam_gap: float   = 0.0,
        lam_aux: float   = 1.00,    # Primary: angle/range guidance
        lam_peak: float  = 0.20,    # Moderate peak regularizer
        lam_margin: float = 0.0,
        lam_range_factor: float = 0.3,  # weight for range factor in cov computation
        gap_margin: float = 0.03,
        lam_subspace_align: float = 0.50,  # Primary: subspace alignment
        lam_peak_contrast: float = 0.0,   # Will be set from config
        lam_heatmap: float = 0.0,   # SpectrumRefiner heatmap supervision
        heatmap_sigma_phi: float = 2.0,   # Gaussian blob sigma (grid cells)
        heatmap_sigma_theta: float = 2.0,
    ):
        super().__init__()
        self.lam_cov   = lam_cov   # Covariance NMSE weight (CRITICAL!)
        self.lam_cov_pred = lam_cov_pred  # Aux penalty on predicted covariance
        self.lam_diag  = lam_diag
        self.lam_off   = lam_off
        self.lam_ortho = lam_ortho
        self.lam_cross = lam_cross
        self.lam_gap   = lam_gap
        self.lam_aux   = lam_aux
        self.lam_peak  = lam_peak
        self.lam_margin = lam_margin
        self.lam_range_factor = lam_range_factor
        self.gap_margin = gap_margin
        self.lam_subspace_align = lam_subspace_align  # Subspace alignment loss
        self.lam_peak_contrast = lam_peak_contrast     # Peak contrast loss
        
        # SpectrumRefiner heatmap supervision
        self.lam_heatmap = lam_heatmap
        self.heatmap_sigma_phi = heatmap_sigma_phi
        self.heatmap_sigma_theta = heatmap_sigma_theta

    # -------- helpers --------

    def _as_complex(self, R):
        """Accept either complex tensors or (..., 2) real/imag stacks"""
        if torch.is_complex(R): 
            return R
        # last dim is [real, imag]
        return R[..., 0].to(torch.float32) + 1j * R[..., 1].to(torch.float32)

    def _nmse_cov(self, R_hat_c: torch.Tensor, R_true_c: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        NMSE = ||R_hat - R_true||_F^2 / (||R_true||_F^2 + eps)
        Returns 0 when R_hat == R_true, ~1 when R_hat == 0
        """
        H = self._as_complex(R_hat_c)
        T = self._as_complex(R_true_c)
        # DEBUG: Print shapes if mismatch
        if H.shape != T.shape:
            print(f"[NMSE SHAPE ERROR] R_hat_c.shape={R_hat_c.shape}, R_true_c.shape={R_true_c.shape}", flush=True)
            print(f"[NMSE SHAPE ERROR] H.shape={H.shape}, T.shape={T.shape}", flush=True)
        diff = H - T
        num = (diff.conj() * diff).real.sum(dim=(-2, -1))
        den = (T.conj() * T).real.sum(dim=(-2, -1)).clamp_min(eps)
        return num / den  # shape: [B]

    # NOTE: trace_norm_torch and shrink_torch now come from covariance_utils

    def _ortho_penalty(self, A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Penalize deviation from Stiefel manifold for predicted steering columns.
        A: [B,N,Kmax] complex; mask: [B,Kmax] in {0,1}
        """
        AhA = A.conj().transpose(-2, -1) @ A  # [B,Kmax,Kmax]
        B, Kmax, _ = AhA.shape
        I = torch.eye(Kmax, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
        diff = AhA - I
        if mask is not None:
            m = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # [B,Kmax,Kmax]
            diff = diff * m.to(diff.dtype)
            denom = m.sum(dim=(-2, -1)).clamp_min(1e-9)
        else:
            denom = torch.tensor(float(Kmax*Kmax), device=A.device, dtype=diff.real.dtype)
        val = (diff.real**2 + diff.imag**2).sum(dim=(-2, -1)) / denom
        return val.real

    def _subspace_align(self, R_cov, phi_pred, theta_pred, r_pred, K_true) -> torch.Tensor:
        """
        Expert-fixed subspace alignment: Uses SVD + projector (no eigenvector phase issue).
        Aligns predicted steering to signal subspace of R_cov.
        """
        # Build projector onto top-K signal subspace of R_cov via SVD
        B, N, _ = R_cov.shape
        eps = getattr(mdl_cfg, 'EPS_PSD', 1e-4)
        eye = torch.eye(N, device=R_cov.device, dtype=R_cov.dtype)
        R = 0.5*(R_cov + R_cov.conj().transpose(-2,-1)) + eps * eye

        # IMPORTANT: detach SVD outputs to avoid backprop through SVD (numerical instability).
        U, S, Vh = torch.linalg.svd(R, full_matrices=False)  # U: [B,N,N], S desc
        U = U.detach()
        A_pred = _steer_torch(phi_pred[:, :cfg.K_MAX], theta_pred[:, :cfg.K_MAX], r_pred[:, :cfg.K_MAX])  # [B,N,Kmax]

        losses = []
        for b in range(B):
            k = int(K_true[b].item())
            if not (1 <= k < N): 
                continue
            U_sig = U[b, :, :k]                      # [N,k]
            P_sig = U_sig @ U_sig.conj().transpose(-2, -1)  # [N,N]
            A_act = A_pred[b, :, :k]                 # [N,k]
            resid = (torch.eye(N, device=R.device, dtype=R.dtype) - P_sig) @ A_act
            num = (resid.real**2 + resid.imag**2).sum()
            den = (A_act.real**2 + A_act.imag**2).sum().clamp_min(1e-9)
            losses.append((num/den).real)
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=R_cov.device)
    
    def _subspace_alignment_loss(self, R_pred, R_true, K_true, ptr_gt):
        """
        STABLE subspace alignment via projection loss (no EVD backprop).
        Uses GT steering vectors (from GT angles/ranges) as the true signal subspace.
        This creates a physics-aligned objective that matches the classical inference backend.
        
        Args:
            R_pred: Predicted covariance [B, N, N] complex
            R_true: True covariance [B, N, N] complex (unused, for API compatibility)
            K_true: True number of sources [B] int
            ptr_gt: Ground truth parameters [B, K_MAX, 3] (phi, theta, r)
            
        Returns:
            Loss scalar (fraction of energy in wrong subspace)
        """
        from .configs import cfg
        import numpy as np
        
        B, N = R_pred.shape[:2]
        device = R_pred.device
        dtype = R_pred.dtype
        
        # Geometry for steering vectors (same as inference)
        N_H = getattr(cfg, 'N_H', 12)
        N_V = getattr(cfg, 'N_V', 12)
        d_h = getattr(cfg, 'd_H', 0.5)  # wavelengths
        d_v = getattr(cfg, 'd_V', 0.5)  # wavelengths
        lam = getattr(cfg, 'WAVEL', 0.3)  # meters
        k0 = 2.0 * np.pi / lam
        
        # Generate sensor coordinates (centered, in meters)
        h_idx = np.arange(-(N_H - 1)//2, (N_H + 1)//2) * d_h * lam
        v_idx = np.arange(-(N_V - 1)//2, (N_V + 1)//2) * d_v * lam
        x_grid, y_grid = np.meshgrid(h_idx, v_idx, indexing='xy')
        x = torch.from_numpy(x_grid.ravel()).to(device, dtype=torch.float32)
        y = torch.from_numpy(y_grid.ravel()).to(device, dtype=torch.float32)
        
        total_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        # DEBUG: Print once to see what we're working with
        if not hasattr(self, '_subspace_align_internal_logged'):
            print(f"[SUBSPACE DEBUG] B={B}, N={N}, ptr_gt.shape={ptr_gt.shape}")
            print(f"[SUBSPACE DEBUG] K_true={K_true.tolist()}")
            self._subspace_align_internal_logged = True
        
        for b in range(B):
            K = int(K_true[b].item())
            if K <= 0 or K >= N:
                continue
                
            try:
                # Build A_gt from GT angles/ranges using canonical steering (no grad)
                A_cols = []
                for k in range(K):
                    # Expert fix: ptr_gt is already in radians, don't convert again!
                    phi_rad = float(ptr_gt[b, k, 0].item())
                    theta_rad = float(ptr_gt[b, k, 1].item())
                    r_m = ptr_gt[b, k, 2].item()
                    
                    # Build near-field steering vector (same as inference)
                    sin_phi = np.sin(phi_rad)
                    cos_theta = np.cos(theta_rad)
                    sin_theta = np.sin(theta_rad)
                    
                    # Phase (curvature term with correct sign)
                    x_np = x.cpu().numpy()
                    y_np = y.cpu().numpy()
                    dist = r_m - x_np * sin_phi * cos_theta - y_np * sin_theta + (x_np**2 + y_np**2) / (2.0 * r_m)
                    phase = k0 * (r_m - dist)
                    
                    # Steering vector (unit-norm)
                    a = np.exp(1j * phase)
                    a = a / np.sqrt(np.sum(np.abs(a)**2))
                    
                    # Convert to torch (detached, no grad)
                    a_torch = torch.from_numpy(a).to(device, dtype=dtype)
                    A_cols.append(a_torch)
                
                # Stack into A_gt [N, K]
                A_gt = torch.stack(A_cols, dim=1)  # [N, K]
                
                # Build projector onto GT signal subspace (stable solve)
                G = A_gt.conj().T @ A_gt  # [K, K] Gramian
                eye_k = torch.eye(K, dtype=dtype, device=device)
                G_reg = G + 1e-4 * eye_k  # Regularization for stability
                P = A_gt @ torch.linalg.solve(G_reg, A_gt.conj().T)  # [N, N]
                
                # Orthogonal projector
                eye_N = torch.eye(N, dtype=dtype, device=device)
                P_perp = eye_N - P
                
                # Energy in wrong subspace vs total energy
                R_b = R_pred[b]
                num = torch.linalg.norm(P_perp @ R_b @ P_perp, ord='fro') ** 2
                den = torch.linalg.norm(R_b, ord='fro') ** 2 + 1e-12
                
                loss_b = (num / den).real
                total_loss = total_loss + loss_b
                valid_batches += 1
                
            except Exception as e:
                # Skip problematic batches
                continue
        
        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=device)
    
    def _peak_contrast_loss(self, R_pred, phi_gt, theta_gt, K_true):
        """
        NEW: Peak contrast loss for training-inference alignment.
        Local ridge around GT angles using MUSIC pseudospectrum.
        
        Args:
            R_pred: Predicted covariance [B, N, N] complex
            phi_gt: Ground truth azimuth [B, K] 
            theta_gt: Ground truth elevation [B, K]
            K_true: True number of sources [B] int
            
        Returns:
            Loss scalar
        """
        B, N = R_pred.shape[:2]
        device = R_pred.device
        
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(B):
            K = int(K_true[b].item())
            if K <= 0:
                continue
                
            try:
                # Build MUSIC spectrum around GT angles (5x5 stencil)
                phi_center = phi_gt[b, :K]  # [K]
                theta_center = theta_gt[b, :K]  # [K]
                
                # Create 5x5 stencil around each GT angle
                stencil_size = 5
                phi_offset = torch.linspace(-0.1, 0.1, stencil_size, device=device)  # ±0.1 rad ≈ ±6°
                theta_offset = torch.linspace(-0.1, 0.1, stencil_size, device=device)
                
                contrast_loss = 0.0
                for k in range(K):
                    phi_k = phi_center[k]
                    theta_k = theta_center[k]
                    
                    # Build stencil
                    phi_grid = phi_k + phi_offset  # [5]
                    theta_grid = theta_k + theta_offset  # [5]
                    
                    # Compute MUSIC spectrum on stencil
                    spectrum = torch.zeros(stencil_size, stencil_size, device=device)
                    
                    for i, phi in enumerate(phi_grid):
                        for j, theta in enumerate(theta_grid):
                            # Build steering vector
                            a = _steer_torch(phi.unsqueeze(0), theta.unsqueeze(0), 
                                           torch.ones(1, device=device) * 2.0)  # Default range
                            a = a[0, :, 0]  # [N]
                            
                            # MUSIC spectrum: 1 / (a^H G a) where G is noise projector
                            # For simplicity, use R_pred directly (approximation)
                            denom = torch.real(a.conj() @ R_pred[b] @ a)
                            denom = torch.clamp(denom, min=1e-12)
                            spectrum[i, j] = 1.0 / denom
                    
                    # Contrastive loss: GT (center) should be higher than neighbors
                    gt_value = spectrum[stencil_size//2, stencil_size//2]  # Center
                    neighbor_values = spectrum.flatten()
                    neighbor_values = neighbor_values[neighbor_values != gt_value]  # Remove center
                    
                    # Softmax-NLL: GT should outrank neighbors
                    if len(neighbor_values) > 0:
                        values = torch.cat([gt_value.unsqueeze(0), neighbor_values])
                        logits = values / 0.1  # Temperature scaling
                        target = torch.zeros(1, dtype=torch.long, device=device)  # GT is index 0
                        contrast_loss += F.cross_entropy(logits.unsqueeze(0), target)
                
                total_loss += contrast_loss / K  # Average over sources
                valid_batches += 1
                
            except Exception:
                # Skip problematic batches
                continue
        
        if valid_batches > 0:
            return total_loss / valid_batches
        else:
            return torch.tensor(0.0, device=device)

    def _eigengap_hinge(self, R_hat_c: torch.Tensor, K_true: torch.Tensor) -> torch.Tensor:
        """
        Expert-fixed eigengap loss: Batched SVD, no eigenvector phase issue.
        SVD returns singular values in DESCENDING order (no flip needed).
        """
        B, N = R_hat_c.shape[:2]
        eps = getattr(mdl_cfg, 'EPS_PSD', 1e-4)
        eye = torch.eye(N, device=R_hat_c.device, dtype=R_hat_c.dtype)

        # Hermitize + load (batched)
        R = 0.5 * (R_hat_c + R_hat_c.conj().transpose(-2, -1)) + eps * eye

        # Batched SVD (descending singular values)
        # Use computational dtype for safety
        U, S, Vh = torch.linalg.svd(R.to(torch.complex64), full_matrices=False)
        # S is [B, N] in DESCENDING order (no flip needed!)

        gaps = []
        for b in range(B):
            k = int(K_true[b].item())
            if 1 <= k < N:
                lam_k  = S[b, k-1]   # kth largest
                lam_k1 = S[b, k]     # (k+1)th largest
                gap = (lam_k - lam_k1).real
                gaps.append(F.relu(self.gap_margin - gap))
            else:
                gaps.append(torch.zeros((), device=R.device, dtype=R.real.dtype))
        return torch.stack(gaps).mean()

    def _subspace_margin_regularizer(self, R_hat: torch.Tensor, K_true: torch.Tensor, margin_target: float = 0.02) -> torch.Tensor:
        """
        Expert-fixed subspace margin: Uses SVD (batched), avoids eigenvector phase issue.
        Encourages clear gap between signal and noise subspaces.
        """
        B, N, _ = R_hat.shape
        eps = getattr(mdl_cfg, 'EPS_PSD', 1e-4)
        eye = torch.eye(N, device=R_hat.device, dtype=R_hat.dtype)

        R = 0.5 * (R_hat + R_hat.conj().transpose(-2, -1)) + eps * eye
        # Singular values descending
        S = torch.linalg.svdvals(R)  # [B, N], descending

        margins = []
        for b in range(B):
            k = int(K_true[b].item())
            if 1 <= k < N:
                # gap between kth and (k+1)th (descending)
                gap = (S[b, k-1] - S[b, k]).relu()
                margins.append((margin_target - gap).clamp(min=0))
            else:
                margins.append(torch.zeros((), device=R.device, dtype=R.real.dtype))
        return torch.stack(margins).mean()
    
    def _angle_chamfer(self, phi_p, theta_p, phi_t, theta_t, mask):
        """
        Chamfer distance on angles (radians).
        """
        P = torch.stack([phi_p.float(), theta_p.float()], dim=-1)  # [B,K,2]
        T = torch.stack([phi_t.float(), theta_t.float()], dim=-1)  # [B,K,2]
        D = ((P.unsqueeze(2) - T.unsqueeze(1))**2).sum(-1)         # [B,K,K]
        big = torch.tensor(1e6, dtype=D.dtype, device=D.device)
        D_pred = torch.where(mask.unsqueeze(1) > 0, D, big)
        d1 = D_pred.min(dim=-1)[0]
        d1 = (d1 * mask).sum(-1) / (mask.sum(-1) + 1e-9)
        D_gt = torch.where(mask.unsqueeze(-1) > 0, D, big)
        d2 = D_gt.min(dim=-2)[0]
        d2 = (d2 * mask).sum(-1) / (mask.sum(-1) + 1e-9)
        return (d1 + d2).mean()

    def _heatmap_loss(self, refined_spectrum, phi_gt, theta_gt, K_true, 
                      grid_phi, grid_theta):
        """
        Compute loss for SpectrumRefiner heatmap supervision.
        
        Creates Gaussian blobs at GT source locations and computes 
        focal loss between refined spectrum and GT heatmap.
        
        Args:
            refined_spectrum: [B, 1, G_phi, G_theta] from SpectrumRefiner
            phi_gt: [B, K_max] GT azimuth in radians
            theta_gt: [B, K_max] GT elevation in radians
            K_true: [B] number of active sources
            grid_phi: [G_phi] phi grid values in radians
            grid_theta: [G_theta] theta grid values in radians
            
        Returns:
            loss: scalar focal loss value
        """
        B = refined_spectrum.shape[0]
        G_phi = refined_spectrum.shape[2]
        G_theta = refined_spectrum.shape[3]
        device = refined_spectrum.device
        
        # Create meshgrid for distance computation
        phi_mesh = grid_phi.view(1, G_phi, 1).expand(B, -1, G_theta)
        theta_mesh = grid_theta.view(1, 1, G_theta).expand(B, G_phi, -1)
        
        # Grid spacing for sigma conversion
        d_phi = (grid_phi[-1] - grid_phi[0]) / (G_phi - 1)
        d_theta = (grid_theta[-1] - grid_theta[0]) / (G_theta - 1)
        
        sigma_phi_rad = self.heatmap_sigma_phi * d_phi
        sigma_theta_rad = self.heatmap_sigma_theta * d_theta
        
        # Initialize GT heatmap
        gt_heatmap = torch.zeros(B, G_phi, G_theta, device=device)
        
        # Add Gaussian blob for each GT source
        for b in range(B):
            K = int(K_true[b].item())
            for k in range(K):
                phi_k = phi_gt[b, k]
                theta_k = theta_gt[b, k]
                
                # Squared distance normalized by sigma
                d_phi_sq = ((phi_mesh[b] - phi_k) / sigma_phi_rad) ** 2
                d_theta_sq = ((theta_mesh[b] - theta_k) / sigma_theta_rad) ** 2
                
                # Gaussian blob
                blob = torch.exp(-0.5 * (d_phi_sq + d_theta_sq))
                
                # Max-blend (allows overlapping sources)
                gt_heatmap[b] = torch.maximum(gt_heatmap[b], blob)
        
        # Add channel dim: [B, 1, G_phi, G_theta]
        gt_heatmap = gt_heatmap.unsqueeze(1)
        
        # Focal loss for sparse target handling (α=0.25, γ=2.0)
        p = refined_spectrum.clamp(1e-7, 1 - 1e-7)
        alpha = 0.25
        gamma = 2.0
        
        pos_weight = alpha * (1 - p) ** gamma
        neg_weight = (1 - alpha) * p ** gamma
        
        pos_loss = -pos_weight * gt_heatmap * torch.log(p)
        neg_loss = -neg_weight * (1 - gt_heatmap) * torch.log(1 - p)
        
        return (pos_loss + neg_loss).mean()

    # -------- forward & debug --------

    def forward(self, y_pred: dict, y_true: dict) -> torch.Tensor:
        from .physics import shrink  # Import at top to avoid scope issues
        device = next(iter(y_pred.values())).device if len(y_pred) else y_true["K"].device
        B = y_true["K"].shape[0]
        
        # Guard against invalid weights (0 is valid for phase-specific ablations like K-only diagnostics)
        if self.lam_cov < 0:
            raise ValueError(f"lam_cov must be >= 0, got {self.lam_cov}")

        # Fast-path: SpectrumRefiner-only supervision
        # If the user sets all other weights to 0 and provides refined_spectrum, avoid
        # the heavy covariance/aux computations.
        only_heatmap = (
            (self.lam_heatmap > 0.0)
            and ("refined_spectrum" in y_pred)
            and (self.lam_cov == 0.0)
            and (self.lam_cov_pred == 0.0)
            and (self.lam_ortho == 0.0)
            and (self.lam_cross == 0.0)
            and (self.lam_gap == 0.0)
            and (self.lam_margin == 0.0)
            and (self.lam_aux == 0.0)
            and (self.lam_peak == 0.0)
            and (self.lam_subspace_align == 0.0)
            and (self.lam_peak_contrast == 0.0)
        )
        if only_heatmap:
            K_true = y_true["K"].long().to(device)
            ptr_gt = y_true["ptr"].to(device).float()
            phi_t = ptr_gt[:, :cfg.K_MAX]
            theta_t = ptr_gt[:, cfg.K_MAX:2*cfg.K_MAX]

            # Build grids (radians) from config FOV
            phi_range = getattr(cfg, 'RANGE_PHI', (-60, 60))  # degrees
            theta_range = getattr(cfg, 'RANGE_THETA', (-30, 30))  # degrees
            G_phi = y_pred['refined_spectrum'].shape[2]
            G_theta = y_pred['refined_spectrum'].shape[3]
            grid_phi = torch.linspace(phi_range[0] * math.pi / 180, phi_range[1] * math.pi / 180, G_phi, device=device)
            grid_theta = torch.linspace(theta_range[0] * math.pi / 180, theta_range[1] * math.pi / 180, G_theta, device=device)

            return self.lam_heatmap * self._heatmap_loss(y_pred['refined_spectrum'], phi_t, theta_t, K_true, grid_phi, grid_theta)

        # GT unpack
        K_true = y_true["K"].long().to(device)
        mask = (torch.arange(cfg.K_MAX, device=device).unsqueeze(0) < K_true.unsqueeze(1)).float()
        ptr_gt = y_true["ptr"].to(device).float()
        phi_t   = ptr_gt[:, :cfg.K_MAX]
        theta_t = ptr_gt[:, cfg.K_MAX:2*cfg.K_MAX]
        r_t     = ptr_gt[:, 2*cfg.K_MAX:3*cfg.K_MAX]

        # R_true complex, Hermitian, trace-normalized
        R_true_ri = y_true["R_true"].to(device).view(B, cfg.N, cfg.N, 2)
        R_true = _ri_to_c(R_true_ri)
        R_true = 0.5 * (R_true + R_true.conj().transpose(-2, -1))
        trt = torch.diagonal(R_true, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-9)
        R_true = R_true / trt.view(-1, 1, 1)
        
        # NMSE self-test (first forward pass only)
        if not hasattr(self, "_nmse_selftest_done"):
            z_eq = self._nmse_cov(R_true, R_true).mean().item()
            z_0  = self._nmse_cov(torch.zeros_like(R_true), R_true).mean().item()
            print(f"[SELFTEST] nmse(R_true,R_true)={z_eq:.3e} (expect ~0), nmse(0,R_true)={z_0:.3e} (expect ~1)", flush=True)
            self._nmse_selftest_done = True
        
        # Apply SNR-aware shrinkage to R_true (torch-native, per-sample) on trace=N convention
        if "snr_db" in y_true:
            R_true = trace_norm_torch(R_true, target_trace=float(cfg.N))
            R_true = shrink_torch(R_true, y_true["snr_db"])

        # predicted cov factors
        if "cov_fact_angle" in y_pred:
            A_angle = _vec2c(y_pred["cov_fact_angle"]).to(device)
        else:
            A_angle = torch.zeros((B, cfg.N, cfg.K_MAX), device=device, dtype=torch.complex64)
        if "cov_fact_range" in y_pred:
            A_range = _vec2c(y_pred["cov_fact_range"]).to(device)
        else:
            A_range = torch.zeros_like(A_angle)

        # NOTE: QR retraction removed (it was a frequent source of NaN gradients early in training).
        # We rely on the orthogonality penalty directly on A_angle instead.

        # Use R_blend if available (training-inference alignment), otherwise construct R_hat from factors
        if 'R_blend' in y_pred:
            R_hat = y_pred['R_blend']  # Use blended covariance for training-inference alignment
        else:
            # Fallback: construct R_hat from factors (for backward compatibility)
            R_hat = (A_angle @ A_angle.conj().transpose(-2, -1)) + self.lam_range_factor * (A_range @ A_range.conj().transpose(-2, -1))
            R_hat = 0.5 * (R_hat + R_hat.conj().transpose(-2, -1))
            trh = torch.diagonal(R_hat, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-9)
            R_hat = R_hat / trh.view(-1, 1, 1)
        
        # Gradient-path sanity check (optional).
        # IMPORTANT: During SpectrumRefiner-only training (or frozen-backbone phases),
        # it is expected that covariance outputs do NOT require gradients.
        is_train = torch.is_grad_enabled() and self.training
        if is_train and (not R_hat.requires_grad):
            strict = bool(getattr(mdl_cfg, "STRICT_GRAD_ASSERTS", False))
            if strict:
                which = "R_blend" if ('R_blend' in y_pred) else "R_hat"
                raise AssertionError(
                    f"❌ {which} does not require gradients during TRAIN. "
                    f"If you are training only SpectrumRefiner, set lam_cov/aux weights to 0 or disable STRICT_GRAD_ASSERTS."
                )
            else:
                if not hasattr(self, "_gradpath_warned"):
                    which = "R_blend" if ('R_blend' in y_pred) else "R_hat"
                    print(
                        f"[WARN] {which}.requires_grad=False during TRAIN. "
                        "This is OK for frozen-backbone / refiner-only training, "
                        "but covariance-related losses will not backprop into the covariance predictor.",
                        flush=True,
                    )
                    self._gradpath_warned = True

        # aux preds
        phi_p   = y_pred.get("phi_soft",   torch.zeros_like(phi_t))
        theta_p = y_pred.get("theta_soft", torch.zeros_like(theta_t))
        r_p     = y_pred.get("r_soft",     torch.zeros_like(r_t))
        if ("phi_soft" not in y_pred) or ("theta_soft" not in y_pred) or ("r_soft" not in y_pred):
            aux = y_pred.get("phi_theta_r", None)
            if aux is not None and aux.shape[-1] >= 3*cfg.K_MAX:
                aux = aux.to(device).float()
                phi_p   = aux[:, :cfg.K_MAX]
                theta_p = aux[:, cfg.K_MAX:2*cfg.K_MAX]
                r_p     = aux[:, 2*cfg.K_MAX:3*cfg.K_MAX]

        # Build effective predicted covariance to MATCH inference object:
        # hermitize → trace-normalize (to trace=N) → (optional) hybrid → diag load → per-sample shrink
        R_pred_in = y_pred['R_blend'] if 'R_blend' in y_pred else R_hat
        beta = None  # already blended if R_blend present
        R_eff_pred = build_effective_cov_torch(
            R_pred_in,
            snr_db=y_true.get("snr_db", None),
            R_samp=None,
            beta=beta,
            diag_load=True,
            apply_shrink=("snr_db" in y_true),
            target_trace=float(cfg.N),
        )

        # Main NMSE loss on effective covariances (train==eval==infer alignment)
        # DEBUG: Check shapes before NMSE
        if not hasattr(self, "_shape_debug_done"):
            print(f"[LOSS DEBUG] R_eff_pred.shape={R_eff_pred.shape}, R_true.shape={R_true.shape}", flush=True)
            self._shape_debug_done = True
        loss_nmse = self._nmse_cov(R_eff_pred, R_true).mean()
        
        # NEW: small auxiliary NMSE on R_pred (constructed from factors) to prevent hiding
        # Only compute when factor heads are present AND not in pure overfit mode
        loss_nmse_pred = torch.tensor(0.0, device=device)
        lam_cov_pred = 0.0 if getattr(mdl_cfg, "OVERFIT_NMSE_PURE", False) else self.lam_cov_pred
        if lam_cov_pred > 0.0 and ("cov_fact_angle" in y_pred) and ("cov_fact_range" in y_pred):
            # Rebuild R_pred from factors (same as fallback branch)
            R_pred_aux = (A_angle @ A_angle.conj().transpose(-2, -1)) + self.lam_range_factor * (A_range @ A_range.conj().transpose(-2, -1))
            R_pred_aux = build_effective_cov_torch(
                R_pred_aux,
                snr_db=y_true.get("snr_db", None),
                R_samp=None,
                beta=None,
                diag_load=True,
                apply_shrink=("snr_db" in y_true),
                target_trace=float(cfg.N),
            )
            loss_nmse_pred = self._nmse_cov(R_pred_aux, R_true).mean()
        
        # Debug logging (once per run)
        if not hasattr(self, '_loss_debug_printed'):
            print(f"[LOSS] lam_cov={self.lam_cov:.3g}, lam_cov_pred={self.lam_cov_pred:.3g}")
            print(f"[LOSS] loss_nmse={loss_nmse.detach().item():.6f}, loss_nmse_pred={loss_nmse_pred.detach().item():.6f}")
            if 'R_blend' in y_pred:
                R_test = y_pred['R_blend']
            else:
                R_test = R_hat
            print(f"[LOSS] R_test.requires_grad={R_test.requires_grad}")
            print(f"[LOSS] ||R_hat - R_true||_F={torch.linalg.norm(R_hat - R_true, ord='fro', dim=(-2,-1)).mean().item():.3f}")
            self._loss_debug_printed = True
        loss_ortho = self._ortho_penalty(A_angle, mask).mean()

        # light cross-term: Gram(A_angle) ≈ Gram(A_range) on off-diagonals
        if self.lam_cross > 0.0:
            def _col_norm(X):
                n = torch.linalg.norm(X, dim=-2, keepdim=True).clamp_min(1e-9)
                return X / n
            Aa = _col_norm(A_angle); Ar = _col_norm(A_range)
            Ga = Aa.conj().transpose(-2, -1) @ Aa
            Gr = Ar.conj().transpose(-2, -1) @ Ar
            B_, Kmax, _ = Ga.shape
            eye = torch.eye(Kmax, device=Ga.device, dtype=Ga.dtype).unsqueeze(0).expand(B_, -1, -1)
            off = (Ga - eye) - (Gr - eye)
            loss_cross = (off.real**2 + off.imag**2).mean()
        else:
            loss_cross = torch.tensor(0.0, device=device)

        # Eigengap / margin terms are disabled in this system.
        # If someone sets non-zero weights, ignore them (do not run SVD-based backward).
        if (self.lam_gap != 0.0) or (self.lam_margin != 0.0):
            if not hasattr(self, "_gap_margin_disabled_warned"):
                print("[WARN] lam_gap/lam_margin are disabled (ignored). Set them to 0.0 for clarity.", flush=True)
                self._gap_margin_disabled_warned = True
        loss_gap = torch.tensor(0.0, device=device)
        loss_margin = torch.tensor(0.0, device=device)

        # NOTE: K-loss removed - using MVDR peak detection instead (K-free localization)

        # Aux: wrapped Huber on angles + Huber on log-range
        phi_huber = (_wrapped_huber_loss(phi_p, phi_t) * mask).sum() / (mask.sum() + 1e-9)
        theta_huber = (_wrapped_huber_loss(theta_p, theta_t) * mask).sum() / (mask.sum() + 1e-9)
        theta_huber *= mdl_cfg.THETA_LOSS_SCALE  # Emphasize elevation for better θ accuracy
        ang_err = phi_huber + theta_huber
        rng_err_log = (_range_huber_loss(r_p, r_t) * mask).sum() / (mask.sum() + 1e-9)
        aux_l2 = ang_err + rng_err_log

        # Chamfer on (phi,theta)
        peak_l2 = self._angle_chamfer(phi_p, theta_p, phi_t, theta_t, mask)

        # small linear range term normalized by span (optional)
        r_span = (cfg.RANGE_R[1] - cfg.RANGE_R[0] + 1e-9)
        range_raw = (((r_p - r_t) / r_span)**2 * mask).sum() / (mask.sum() + 1e-9)

        loss_align = torch.tensor(0.0, device=device)
        if getattr(mdl_cfg, "LAM_ALIGN", 0.0) > 0.0:
            # Expert fix: Re-enabled subspace alignment with SVD + projector (now safe)
            Rc = y_pred['R_blend'] if 'R_blend' in y_pred else R_hat
            loss_align = self._subspace_align(Rc, phi_p, theta_p, r_p, K_true)

        # Training-inference alignment losses
        loss_subspace_align = 0.0
        loss_peak_contrast = 0.0
        # Expert fix: Re-enabled subspace alignment loss (GT-based, no eigendecomposition)
        if self.lam_subspace_align > 0.0:
            if 'R_blend' in y_pred:
                loss_subspace_align = self._subspace_alignment_loss(y_pred['R_blend'], R_true, K_true, ptr_gt)
            else:
                loss_subspace_align = self._subspace_alignment_loss(R_hat, R_true, K_true, ptr_gt)
            
            if not hasattr(self, '_subspace_align_logged'):
                print(f"[LOSS DEBUG] Subspace align: {loss_subspace_align.item():.6f} @ weight={self.lam_subspace_align}")
                print(f"[LOSS DEBUG] ptr_gt shape: {ptr_gt.shape}, K_true: {K_true.tolist()}")
                self._subspace_align_logged = True
        else:
            if not hasattr(self, '_subspace_align_logged'):
                print(f"[LOSS DEBUG] Subspace align: DISABLED (weight={self.lam_subspace_align})")
                self._subspace_align_logged = True
        if self.lam_peak_contrast > 0.0:
            loss_peak_contrast = self._peak_contrast_loss(R_hat, phi_t, theta_t, K_true)
            # DEBUG: Print once to verify it's being computed
            if not hasattr(self, '_peak_contrast_logged'):
                print(f"[LOSS DEBUG] Peak contrast: {loss_peak_contrast.item():.6f} @ weight={self.lam_peak_contrast}")
                self._peak_contrast_logged = True
        else:
            # DEBUG: Print that it's disabled
            if not hasattr(self, '_peak_contrast_logged'):
                print(f"[LOSS DEBUG] Peak contrast: DISABLED (weight={self.lam_peak_contrast})")
                self._peak_contrast_logged = True

        # SpectrumRefiner heatmap supervision (optional)
        loss_heatmap = torch.tensor(0.0, device=device)
        if self.lam_heatmap > 0.0 and 'refined_spectrum' in y_pred:
            # Need grid coordinates for heatmap loss
            # Use default FOV from config
            phi_range = getattr(cfg, 'RANGE_PHI', (-60, 60))  # degrees
            theta_range = getattr(cfg, 'RANGE_THETA', (-30, 30))  # degrees
            G_phi = y_pred['refined_spectrum'].shape[2]
            G_theta = y_pred['refined_spectrum'].shape[3]
            
            # Create grid in radians
            grid_phi = torch.linspace(
                phi_range[0] * math.pi / 180, 
                phi_range[1] * math.pi / 180, 
                G_phi, device=device
            )
            grid_theta = torch.linspace(
                theta_range[0] * math.pi / 180, 
                theta_range[1] * math.pi / 180, 
                G_theta, device=device
            )
            
            loss_heatmap = self._heatmap_loss(
                y_pred['refined_spectrum'], phi_t, theta_t, K_true,
                grid_phi, grid_theta
            )
            
            if not hasattr(self, '_heatmap_loss_logged'):
                print(f"[LOSS DEBUG] Heatmap loss: {loss_heatmap.item():.6f} @ weight={self.lam_heatmap}")
                self._heatmap_loss_logged = True
        else:
            if not hasattr(self, '_heatmap_loss_logged'):
                has_refined = 'refined_spectrum' in y_pred
                print(f"[LOSS DEBUG] Heatmap loss: DISABLED (weight={self.lam_heatmap}, has_spectrum={has_refined})")
                self._heatmap_loss_logged = True

        total = (
            self.lam_cov * loss_nmse  # Primary: Covariance NMSE
            + self.lam_cov_pred * loss_nmse_pred  # Aux pressure on R_pred
            + self.lam_ortho * loss_ortho
            + self.lam_cross * loss_cross
            + self.lam_aux   * aux_l2
            + self.lam_peak  * peak_l2  # Chamfer/peak angle loss
            + 0.02           * range_raw
            + getattr(mdl_cfg, "LAM_ALIGN", 0.0) * loss_align
            + self.lam_subspace_align * loss_subspace_align  # Subspace alignment loss
            + self.lam_peak_contrast * loss_peak_contrast     # Peak contrast loss
            + self.lam_heatmap * loss_heatmap                 # SpectrumRefiner supervision
        )
        
        # Log loss breakdown (once per run)
        if not hasattr(self, '_loss_breakdown_printed'):
            aux_contrib = self.lam_aux * aux_l2.item() if isinstance(aux_l2, torch.Tensor) else 0
            cov_contrib = self.lam_cov * loss_nmse.item() if isinstance(loss_nmse, torch.Tensor) else 0
            print(f"[LOSS BREAKDOWN] total={total.item():.4f}")
            print(f"  lam_cov*nmse    = {self.lam_cov:.3f} * {loss_nmse.item():.4f} = {cov_contrib:.4f}")
            print(f"  lam_aux*aux_l2  = {self.lam_aux:.3f} * {aux_l2.item():.4f} = {aux_contrib:.4f}")
            self._loss_breakdown_printed = True
        
        return total

    @torch.no_grad()
    def debug_terms(self, y_pred: dict, y_true: dict) -> dict:
        # (lightweight re-computation for logging; mirrors forward)
        device = next(iter(y_pred.values())).device if len(y_pred) else y_true["K"].device
        B = y_true["K"].shape[0]
        K_true = y_true["K"].long().to(device)
        mask = (torch.arange(cfg.K_MAX, device=device).unsqueeze(0) < K_true.unsqueeze(1)).float()
        ptr_gt = y_true["ptr"].to(device).float()
        phi_t   = ptr_gt[:, :cfg.K_MAX]
        theta_t = ptr_gt[:, cfg.K_MAX:2*cfg.K_MAX]
        r_t     = ptr_gt[:, 2*cfg.K_MAX:3*cfg.K_MAX]

        R_true_ri = y_true["R_true"].to(device).view(B, cfg.N, cfg.N, 2)
        R_true = _ri_to_c(R_true_ri); R_true = 0.5*(R_true + R_true.conj().transpose(-2,-1))
        trt = torch.diagonal(R_true, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-9)
        R_true = R_true / trt.view(-1,1,1)
        
        # Apply same shrink as inference if SNR available (per-sample)
        if "snr_db" in y_true:
            from .physics import shrink
            R_true_list = []
            for b in range(B):
                R_b = shrink(R_true[b:b+1].cpu().numpy(),
                           float(y_true["snr_db"][b]),
                           base=mdl_cfg.SHRINK_BASE_ALPHA)
                R_true_list.append(torch.from_numpy(R_b))
            R_true = torch.cat(R_true_list, dim=0).to(R_true.device, R_true.dtype)

        if "cov_fact_angle" in y_pred: A_angle = _vec2c(y_pred["cov_fact_angle"]).to(device)
        else: A_angle = torch.zeros((B, cfg.N, cfg.K_MAX), device=device, dtype=torch.complex64)
        if "cov_fact_range" in y_pred: A_range = _vec2c(y_pred["cov_fact_range"]).to(device)
        else: A_range = torch.zeros_like(A_angle)

        Aq=[]
        for b in range(B):
            k=int(K_true[b].item())
            if k<=0: Aq.append(A_angle[b]); continue
            Q,_=torch.linalg.qr(A_angle[b,:,:k].to(torch.complex64),mode='reduced')
            Qpad=torch.zeros_like(A_angle[b]); Qpad[:,:k]=Q; Aq.append(Qpad)
        A_angle=torch.stack(Aq,dim=0)

        R_hat=(A_angle@A_angle.conj().transpose(-2,-1))+self.lam_range_factor*(A_range@A_range.conj().transpose(-2,-1))
        R_hat=0.5*(R_hat+R_hat.conj().transpose(-2,-1))
        trh=torch.diagonal(R_hat,dim1=-2,dim2=-1).real.sum(-1).clamp_min(1e-9)
        R_hat=R_hat/trh.view(-1,1,1)

        # CRITICAL FIX: Use blended covariance for debug NMSE (consistent with inference)
        if 'R_blend' in y_pred:
            nmse = self._nmse_cov(y_pred['R_blend'], R_true).mean().item()
        else:
            nmse = self._nmse_cov(R_hat, R_true).mean().item()
        ortho=self._ortho_penalty(A_angle,mask).mean().item()

        if self.lam_cross>0.0:
            def _col_norm(X):
                n=torch.linalg.norm(X,dim=-2,keepdim=True).clamp_min(1e-9); return X/n
            Aa=_col_norm(A_angle); Ar=_col_norm(A_range)
            Ga=Aa.conj().transpose(-2,-1)@Aa; Gr=Ar.conj().transpose(-2,-1)@Ar
            B_,Kmax,_=Ga.shape; eye=torch.eye(Kmax,device=Ga.device,dtype=Ga.dtype).unsqueeze(0).expand(B_,-1,-1)
            off=(Ga-eye)-(Gr-eye); cross=(off.real**2+off.imag**2).mean().item()
        else:
            cross=0.0

        gap=(self._eigengap_hinge(R_hat,K_true).item() if self.lam_gap>0.0 else 0.0)

        phi_p=y_pred.get("phi_soft",torch.zeros_like(phi_t))
        theta_p=y_pred.get("theta_soft",torch.zeros_like(theta_t))
        r_p=y_pred.get("r_soft",torch.zeros_like(r_t))
        if ("phi_soft" not in y_pred) or ("theta_soft" not in y_pred) or ("r_soft" not in y_pred):
            aux=y_pred.get("phi_theta_r",None)
            if aux is not None and aux.shape[-1]>=3*cfg.K_MAX:
                aux=aux.to(device).float()
                phi_p=aux[:,:cfg.K_MAX]; theta_p=aux[:,cfg.K_MAX:2*cfg.K_MAX]; r_p=aux[:,2*cfg.K_MAX:3*cfg.K_MAX]

        phi_huber_sum=(_wrapped_huber_loss(phi_p,phi_t)*mask).sum()/(mask.sum()+1e-9)
        theta_huber_sum=(_wrapped_huber_loss(theta_p,theta_t)*mask).sum()/(mask.sum()+1e-9)
        theta_huber_sum *= mdl_cfg.THETA_LOSS_SCALE  # Emphasize elevation for better θ accuracy
        ang_err=phi_huber_sum+theta_huber_sum
        r_p_pos=torch.clamp(r_p,min=1e-6); r_t_pos=torch.clamp(r_t,min=1e-6)
        rng_err_log=(((torch.log(r_p_pos)-torch.log(r_t_pos))**2)*mask).sum()/(mask.sum()+1e-9)
        aux_l2=(ang_err+rng_err_log).item()
        peak=self._angle_chamfer(phi_p,theta_p,phi_t,theta_t,mask).item()
        
        # Expert fix: Re-enabled margin and align for logging (now safe with SVD)
        margin = self._subspace_margin_regularizer(R_true, K_true).item() if self.lam_margin > 0.0 else 0.0
        align = (self._subspace_align(R_true, phi_p, theta_p, r_p, K_true).item() 
                if getattr(mdl_cfg, "LAM_ALIGN", 0.0) > 0.0 else 0.0)
        
        # NEW: Training-inference alignment losses
        loss_subspace_align = 0.0
        loss_peak_contrast = 0.0
        if self.lam_subspace_align > 0.0:
            loss_subspace_align = self._subspace_alignment_loss(R_hat, R_true, K_true, ptr_gt)
        if self.lam_peak_contrast > 0.0:
            loss_peak_contrast = self._peak_contrast_loss(R_hat, phi_t, theta_t, K_true)
        
        r_span=(cfg.RANGE_R[1]-cfg.RANGE_R[0]+1e-9)
        rng=((((r_p-r_t)/r_span)**2)*mask).sum()/(mask.sum()+1e-9)
        
        # Return with keys matching what train.py expects
        return dict(
            loss_nmse=nmse,           # Covariance NMSE
            ortho=ortho,              # Orthogonality penalty
            cross=cross,              # Cross-term
            loss_gap=gap,             # Eigengap hinge
            aux=aux_l2,               # Combined aux loss
            phi_huber=phi_huber_sum.item(),  # Angle error
            rng_err_log=rng_err_log.item(),  # Range log error
            peak=peak,                # Chamfer loss
            margin=margin,            # Margin regularizer
            align=align,              # Subspace alignment
            range=float(rng)          # Range L2 error
        )
