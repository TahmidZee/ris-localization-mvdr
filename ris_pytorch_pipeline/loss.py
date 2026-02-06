
import torch, torch.nn as nn, torch.nn.functional as F
from .configs import cfg, mdl_cfg
from .physics import shrink
from .covariance_utils import trace_norm_torch, shrink_torch, build_effective_cov_torch
import math
import itertools

def _ri_to_c(x_ri):
    return torch.complex(x_ri[...,0], x_ri[...,1])

def _wrap_angle(x):
    """Wrap angles to [-π, π]"""
    return ((x + math.pi) % (2 * math.pi)) - math.pi

def _wrapped_huber_loss(pred, gt, delta=0.175):  # delta = 10° in radians (not 0.25°!)
    """
    Huber loss that respects angular wrap-around.
    
    CRITICAL FIX: delta was π/720 = 0.25° which is way too small!
    With initial errors of ~19°, everything was in linear regime (constant gradient).
    Changed to 10° so that errors < 10° get quadratic (proportional) gradients.
    """
    d = _wrap_angle(pred - gt)
    abs_d = torch.abs(d)
    return torch.where(abs_d <= delta, 
                      0.5 * (d ** 2) / delta, 
                      abs_d - 0.5 * delta)

def _range_huber_loss(pred_r, gt_r, delta=0.2):
    """Huber loss on log-range for better scale handling"""
    # Only clamp to a tiny epsilon for log() safety.
    # Do NOT clamp to RANGE_R[0] (or ~0.9*RANGE_R[0]) because that can zero gradients early
    # and stall the range head; it also makes permutation matching unstable.
    eps_m = float(getattr(cfg, "RANGE_EPS_M", 1e-3))
    pred_r_pos = torch.clamp(pred_r, min=eps_m)
    gt_r_pos = torch.clamp(gt_r, min=eps_m)
    pr = torch.log(pred_r_pos)
    gr = torch.log(gt_r_pos)
    e = torch.abs(pr - gr)
    return torch.where(e < delta, 0.5 * (e ** 2) / delta, e - 0.5 * delta)

def _perm_invariant_aux_loss(phi_p, theta_p, r_p, phi_t, theta_t, r_t, K_true, *,
                             delta_ang=0.175, delta_logr=0.5, mask_p=None,
                             soft: bool = False, soft_tau: float = 0.25):
    """
    Permutation-invariant aux loss for unordered multi-source scenes.
    Matches predicted slots (size K_MAX) to GT slots (size k<=K_MAX) via brute-force
    (K_MAX<=5 => at most 120 perms), then computes wrapped Huber on angles and Huber on log-range.
    Selection is done under no_grad for stability; gradients flow through the chosen pairing.
    
    If mask_p is provided [B, Kmax], also computes BCE on masks using the assignment:
      - matched slots → target = 1
      - unmatched slots → target = 0
    Returns (geom_loss, mask_bce_loss) if mask_p is provided, else just geom_loss.
    """
    device = phi_p.device
    B, Kmax = phi_p.shape
    losses = []
    mask_bce_losses = [] if mask_p is not None else None

    # Precompute permutations of indices for each k in [1..Kmax]
    perms_by_k = {}
    base_idx = list(range(Kmax))
    for k in range(1, Kmax + 1):
        perms = list(itertools.permutations(base_idx, k))
        perms_by_k[k] = torch.tensor(perms, device=device, dtype=torch.long)  # [P,k]

    for b in range(B):
        k = int(K_true[b].item())
        if not (1 <= k <= Kmax):
            continue
        perms = perms_by_k[k]  # [P,k]

        # Vectorized per-permutation losses/costs.
        # Shapes:
        #   pp,tp,rp: [P,k]
        #   gt_*:     [1,k]
        pp = phi_p[b][perms]
        tp = theta_p[b][perms]
        rp = r_p[b][perms]
        gt_phi = phi_t[b, :k].view(1, k)
        gt_th  = theta_t[b, :k].view(1, k)
        gt_r   = r_t[b, :k].view(1, k)

        # Cost for assignment (same structure as loss, but used for weighting/argmin)
        dphi = _wrap_angle(pp - gt_phi).abs()
        dth  = _wrap_angle(tp - gt_th).abs()
        cphi = torch.where(dphi <= delta_ang, 0.5 * (dphi ** 2) / delta_ang, dphi - 0.5 * delta_ang)
        cth  = torch.where(dth  <= delta_ang, 0.5 * (dth  ** 2) / delta_ang, dth  - 0.5 * delta_ang)

        eps_m = float(getattr(cfg, "RANGE_EPS_M", 1e-3))
        rp_pos = torch.clamp(rp, min=eps_m)
        gt_pos = torch.clamp(gt_r, min=eps_m)
        elog = (torch.log(rp_pos) - torch.log(gt_pos)).abs()
        cr = torch.where(elog <= delta_logr, 0.5 * (elog ** 2) / delta_logr, elog - 0.5 * delta_logr)

        cost = (cphi + cth + cr).sum(dim=1)  # [P]

        # True loss per permutation (with wrapped huber + range huber)
        phi_h = _wrapped_huber_loss(pp, gt_phi, delta=delta_ang).mean(dim=1)  # [P]
        th_h  = _wrapped_huber_loss(tp, gt_th,  delta=delta_ang).mean(dim=1)  # [P]
        th_h  = th_h * float(getattr(mdl_cfg, "THETA_LOSS_SCALE", 1.0))
        r_h   = _range_huber_loss(rp, gt_r.expand_as(rp), delta=delta_logr).mean(dim=1)  # [P]
        loss_perm = phi_h + th_h + r_h  # [P]

        if soft:
            # Softmin weighting for smooth optimization early (reduces assignment flips).
            tau = float(max(1e-6, soft_tau))
            w = torch.softmax(-cost / tau, dim=0)  # [P]
            losses.append((w * loss_perm).sum())
            # For mask BCE, keep hard assignment semantics (only used when masks are enabled later).
            if mask_p is not None:
                best_i = int(torch.argmin(cost.detach()).item())
                best_perm = perms[best_i]
                mask_target = torch.zeros(Kmax, device=device)
                mask_target[best_perm] = 1.0
                m = mask_p[b].clamp(min=1e-6, max=1.0 - 1e-6)
                bce = -(mask_target * torch.log(m) + (1.0 - mask_target) * torch.log(1.0 - m))
                mask_bce_losses.append(bce.mean())
        else:
            # Hard assignment (argmin) used once geometry is stable.
            with torch.no_grad():
                best_i = int(torch.argmin(cost).item())
                best_perm = perms[best_i]
            # Selected permutation loss
            losses.append(loss_perm[best_i])
            if mask_p is not None:
                mask_target = torch.zeros(Kmax, device=device)
                mask_target[best_perm] = 1.0
                m = mask_p[b].clamp(min=1e-6, max=1.0 - 1e-6)
                bce = -(mask_target * torch.log(m) + (1.0 - mask_target) * torch.log(1.0 - m))
                mask_bce_losses.append(bce.mean())

    if not losses:
        zero = torch.tensor(0.0, device=device)
        return (zero, zero) if mask_p is not None else zero
    
    geom_loss = torch.stack(losses).mean()
    if mask_p is not None:
        mask_bce = torch.stack(mask_bce_losses).mean() if mask_bce_losses else torch.tensor(0.0, device=device)
        return geom_loss, mask_bce
    return geom_loss

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
    # Match physics.py / dataset.py convention exactly:
    # dist = r - planar + curvature
    # phase = k0 * (r - dist) = k0 * (planar - curvature)
    planar = vh * sin_phi * cos_theta + vv * sin_theta
    curvature = (vh**2 + vv**2) / (2.0 * r_eff)
    phase = cfg.k0 * (planar - curvature)
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
        lam_subspace_align: float = 0.0,  # Default OFF; trainer/phase sets this explicitly
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
        
        # Mask loss scale: controlled by Trainer based on MASK_LOSS_WARMUP_EPOCHS.
        # CRITICAL FIX (2026-02-05): Default to 1.0 (was 0.0).
        # With MASK_LOSS_WARMUP_EPOCHS=0 (no warmup), mask losses should be active
        # from epoch 0 to provide the "no object" gradient for unmatched slots.
        self.mask_loss_scale = 1.0  # Active by default

        # Aux matching mode (perm-invariant geometry loss):
        # Use soft matching early to avoid assignment flips, then switch to hard.
        self.aux_match_soft = False
        self.aux_match_tau = 0.25
    
    def set_mask_loss_scale(self, scale: float):
        """Set the mask loss scale (0.0 = disabled, 1.0 = full weight)"""
        self.mask_loss_scale = float(max(0.0, min(1.0, scale)))

    def set_aux_match(self, soft: bool, tau: float = 0.25):
        """Configure permutation matching for aux geometry loss."""
        self.aux_match_soft = bool(soft)
        self.aux_match_tau = float(max(1e-6, tau))
    
    def _slot_diversity_loss(self, phi_p, theta_p, r_p):
        """
        Penalize slots that predict too-similar values (margin-based).
        
        Uses a margin: when pairwise distance > margin, loss is 0 (slots are 
        already well-separated). When distance < margin, loss = (margin - dist),
        penalizing collapse. This is always non-negative, preventing the negative
        loss bug that was overwhelming other loss terms.
        
        Args:
            phi_p, theta_p: [B, K_MAX] angle predictions (radians)
            r_p: [B, K_MAX] range predictions (meters)
            
        Returns:
            Loss scalar >= 0 (0 when slots are well-separated, positive when collapsed)
        """
        B, K = phi_p.shape
        
        # Pairwise angular distances (with wrapping)
        dphi = phi_p.unsqueeze(2) - phi_p.unsqueeze(1)  # [B, K, K]
        dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))  # Wrap to [-π, π]
        dphi = dphi.abs()  # [B, K, K]
        
        dtheta = theta_p.unsqueeze(2) - theta_p.unsqueeze(1)
        dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))
        dtheta = dtheta.abs()
        
        # Pairwise range distances (normalized by range span)
        r_span = float(getattr(cfg, "RANGE_R", (0.5, 10.0))[1] - getattr(cfg, "RANGE_R", (0.5, 10.0))[0])
        dr = (r_p.unsqueeze(2) - r_p.unsqueeze(1)).abs() / r_span  # [B, K, K]
        
        # Combined distance (exclude diagonal)
        dist = dphi + dtheta + dr  # [B, K, K]
        
        # Mask out diagonal (slot vs itself)
        eye_mask = 1.0 - torch.eye(K, device=dist.device)
        
        # Margin-based: penalize when slots are closer than margin (radians + normalized range)
        # margin ≈ 0.15 rad (~8.6°) is reasonable given FOV of ±60° azimuth
        margin = 0.15
        violation = torch.relu(margin - dist) * eye_mask  # [B, K, K], non-negative
        loss = violation.sum() / (eye_mask.sum() + 1e-9)
        
        return loss

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
        # IMPORTANT: cfg.d_H / cfg.d_V are in **meters** in this repo (see configs.py / dataset.py).
        # Keep steering consistent with physics.nearfield_vec and music_gpu.py.
        B, N = R_pred.shape[:2]
        device = R_pred.device
        dtype = R_pred.dtype
        
        # ptr_gt may arrive as chunked [B, 3*Kmax]; convert to [B, Kmax, 3] for convenience.
        Kmax = int(getattr(cfg, "K_MAX", 5))
        if isinstance(ptr_gt, torch.Tensor) and ptr_gt.dim() == 2 and ptr_gt.shape[1] >= 3 * Kmax:
            phi = ptr_gt[:, :Kmax]
            theta = ptr_gt[:, Kmax:2 * Kmax]
            rr = ptr_gt[:, 2 * Kmax:3 * Kmax]
            ptr_gt = torch.stack([phi, theta, rr], dim=-1)  # [B,Kmax,3]
        if (not isinstance(ptr_gt, torch.Tensor)) or (ptr_gt.dim() != 3) or (ptr_gt.shape[-1] != 3):
            return torch.tensor(0.0, device=device)

        # Geometry (meters)
        N_H = int(getattr(cfg, "N_H", 12))
        N_V = int(getattr(cfg, "N_V", 12))
        d_h_m = float(getattr(cfg, "d_H", 0.15))
        d_v_m = float(getattr(cfg, "d_V", d_h_m))
        k0 = float(getattr(cfg, "k0", 2.0 * math.pi / float(getattr(cfg, "WAVEL", 0.3))))

        # Sensor coordinates (centered UPA), meters
        h_idx = torch.arange(-(N_H - 1) // 2, (N_H + 1) // 2, device=device, dtype=torch.float32) * d_h_m
        v_idx = torch.arange(-(N_V - 1) // 2, (N_V + 1) // 2, device=device, dtype=torch.float32) * d_v_m
        x_grid, y_grid = torch.meshgrid(h_idx, v_idx, indexing="xy")
        x = x_grid.reshape(-1)  # [N]
        y = y_grid.reshape(-1)  # [N]
        hv_sq = (x * x + y * y).view(1, -1)  # [1,N]

        # Optional one-time debug
        if bool(getattr(cfg, "TRAIN_EPOCH_DEBUG", False)) and (not hasattr(self, "_subspace_align_internal_logged")):
            print(f"[SUBSPACE DEBUG] B={B}, N={N}, ptr_gt.shape={tuple(ptr_gt.shape)}", flush=True)
            self._subspace_align_internal_logged = True
        
        total_loss = torch.tensor(0.0, device=device)
        valid_batches = 0

        for b in range(B):
            K = int(K_true[b].item())
            if K <= 0 or K >= N:
                continue
                
            # GT (radians/meters)
            phi_rad = ptr_gt[b, :K, 0].to(torch.float32)
            theta_rad = ptr_gt[b, :K, 1].to(torch.float32)
            r_m = ptr_gt[b, :K, 2].to(torch.float32).clamp_min(1e-6)

            sin_phi = torch.sin(phi_rad)
            cos_theta = torch.cos(theta_rad)
            sin_theta = torch.sin(theta_rad)
                    
            # phase = k0 * (planar - curvature) where planar=x*sinφ*cosθ + y*sinθ
            planar = (sin_phi * cos_theta).unsqueeze(1) * x.view(1, -1) + sin_theta.unsqueeze(1) * y.view(1, -1)  # [K,N]
            curvature = hv_sq / (2.0 * r_m.unsqueeze(1))  # [K,N]
            phase = k0 * (planar - curvature)  # [K,N]
            A_gt = (torch.exp(1j * phase) / math.sqrt(float(N))).to(dtype)  # [K,N] complex
            A_gt = A_gt.transpose(0, 1).contiguous()  # [N,K]
                
            # Projector onto GT signal subspace
            G = A_gt.conj().transpose(-2, -1) @ A_gt  # [K,K]
            eye_k = torch.eye(K, dtype=dtype, device=device)
            G_reg = G + 1e-4 * eye_k
            P = A_gt @ torch.linalg.solve(G_reg, A_gt.conj().transpose(-2, -1))  # [N,N]
                
            eye_N = torch.eye(N, dtype=dtype, device=device)
            P_perp = eye_N - P
                
            R_b = R_pred[b]
            num = torch.linalg.norm(P_perp @ R_b @ P_perp, ord="fro") ** 2
            den = torch.linalg.norm(R_b, ord="fro") ** 2 + 1e-12
            total_loss = total_loss + (num / den).real
            valid_batches += 1
                
        return (total_loss / valid_batches) if valid_batches > 0 else torch.tensor(0.0, device=device)
    
    def _peak_contrast_loss(self, R_pred, phi_gt, theta_gt, r_gt, K_true):
        """
        NEW: Peak contrast loss for training-inference alignment.
        Local ridge around GT angles using MVDR/Capon pseudospectrum.
        
        Args:
            R_pred: Predicted **effective** covariance [B, N, N] complex (should be diag-loaded + shrunk)
            phi_gt: Ground truth azimuth [B, Kmax] radians
            theta_gt: Ground truth elevation [B, Kmax] radians
            r_gt: Ground truth range [B, Kmax] meters
            K_true: True number of sources [B] int
            
        Returns:
            Loss scalar
        """
        # NOTE: We keep this loss cheap by using a small stencil (default 3x3) and doing
        # one Cholesky solve per sample with (Kmax * stencil^2) RHS vectors.
        B, N = R_pred.shape[:2]
        device = R_pred.device
        Kmax = int(getattr(cfg, "K_MAX", 5))

        # Hyperparams (configurable)
        stencil = int(getattr(cfg, "PEAK_CONTRAST_STENCIL", 3))
        stencil = max(3, int(stencil))
        if stencil % 2 == 0:
            stencil += 1
        delta = float(getattr(cfg, "PEAK_CONTRAST_DELTA_RAD", 0.10))
        # tau controls softmax temperature. Higher tau → softer distribution, more stable gradients.
        # Default increased from 0.10 to 0.50 to avoid numerical overflow in cross_entropy.
        tau = float(getattr(cfg, "PEAK_CONTRAST_TAU", 0.50))
        tau = max(tau, 0.1)  # Floor at 0.1 to prevent gradient explosion

        # Mask for valid sources
        mask = (torch.arange(Kmax, device=device).unsqueeze(0) < K_true.unsqueeze(1)).float()  # [B,Kmax]
        if mask.sum() <= 0:
            return torch.tensor(0.0, device=device)

        # Offsets (radians)
        phi_off = torch.linspace(-delta, delta, stencil, device=device, dtype=torch.float32)
        th_off = torch.linspace(-delta, delta, stencil, device=device, dtype=torch.float32)

        # Build per-source stencil points: [B,Kmax,st,st] -> flatten to G=st^2
        phi_c = phi_gt[:, :Kmax].to(device=device, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        th_c = theta_gt[:, :Kmax].to(device=device, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        r_c = r_gt[:, :Kmax].to(device=device, dtype=torch.float32).clamp_min(1e-6).unsqueeze(-1).unsqueeze(-1)

        phi_pts = (phi_c + phi_off.view(1, 1, stencil, 1)).expand(-1, -1, stencil, stencil)
        th_pts = (th_c + th_off.view(1, 1, 1, stencil)).expand(-1, -1, stencil, stencil)
        r_pts = r_c.expand(-1, -1, stencil, stencil)

        G = stencil * stencil
        phi_q = phi_pts.reshape(B, Kmax * G)
        th_q = th_pts.reshape(B, Kmax * G)
        r_q = r_pts.reshape(B, Kmax * G)

        # Steering vectors: [B,N,Q]
        A = _steer_torch(phi_q, th_q, r_q).to(torch.complex64)

        # Robust solve: Cholesky if possible (faster/more stable than LU).
        # Ensure Hermitian + add tiny eps for PSD.
        eps_psd = float(getattr(mdl_cfg, "EPS_PSD", 1e-4))
        I = torch.eye(N, device=device, dtype=R_pred.dtype).unsqueeze(0).expand(B, N, N)
        R = 0.5 * (R_pred + R_pred.conj().transpose(-2, -1)) + eps_psd * I

        L, info = torch.linalg.cholesky_ex(R)
        if info is not None and torch.any(info != 0):
            # Fallback to solve (rare); keep it safe.
            X = torch.linalg.solve(R, A)
        else:
            X = torch.cholesky_solve(A, L)

        denom = (A.conj() * X).sum(dim=1).real  # [B,Q]
        denom = denom.clamp_min(1e-12)
        logP = (-torch.log(denom)).reshape(B * Kmax, G)  # larger is better
                    
        # Target is center of the stencil grid
        center_idx = (stencil // 2) * stencil + (stencil // 2)
        target = torch.full((B * Kmax,), int(center_idx), device=device, dtype=torch.long)

        # Contrast CE per source, masked
        ce = F.cross_entropy(logP / tau, target, reduction="none")  # [B*Kmax]
        m = mask.reshape(B * Kmax)
        return (ce * m).sum() / (m.sum() + 1e-9)

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
        
        # IMPORTANT (train/infer alignment):
        # Inference consumes an *effective* covariance with:
        #   trace-norm → diag-load → (optional) SNR-aware shrink (and trace-norm after diag-load)
        # Previously we applied this pipeline only to R_pred but not to R_true, which
        # creates a systematic NMSE floor and pushes R_pred toward a different target.
        #
        # Build R_eff_true using the same helper used for R_eff_pred.
        R_eff_true = build_effective_cov_torch(
            R_true,
            snr_db=y_true.get("snr_db", None),
            R_samp=None,
            beta=None,
            diag_load=True,
            apply_shrink=("snr_db" in y_true),
            target_trace=float(cfg.N),
        )

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

        # STRUCTURAL FIX: Use R_pred directly if available (geometry-aware covariance)
        # Priority: R_pred (structural) > R_blend (hybrid) > factors (legacy)
        if 'R_pred' in y_pred:
            # NEW: Structural covariance from geometry predictions
            R_hat = y_pred['R_pred']  # [B, N, N] complex, already trace-normalized
            if not hasattr(self, '_structural_R_logged'):
                print(f"[LOSS] Using STRUCTURAL R_pred (geometry-aware covariance)", flush=True)
                self._structural_R_logged = True
        elif 'R_blend' in y_pred:
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
            print(f"[LOSS DEBUG] R_eff_pred.shape={R_eff_pred.shape}, R_eff_true.shape={R_eff_true.shape}", flush=True)
            self._shape_debug_done = True
        loss_nmse = self._nmse_cov(R_eff_pred, R_eff_true).mean()
        
        # Auxiliary NMSE on R_pred to provide a second gradient path through geometry.
        # CRITICAL FIX (2026-02-05): Added structural R mode path.
        # Previously this only worked with legacy factor heads (cov_fact_angle/range),
        # which are absent in structural R mode → loss_nmse_pred was always 0.
        # Now we also compute NMSE directly on R_pred when it's available.
        loss_nmse_pred = torch.tensor(0.0, device=device)
        lam_cov_pred = 0.0 if getattr(mdl_cfg, "OVERFIT_NMSE_PURE", False) else self.lam_cov_pred
        if lam_cov_pred > 0.0:
            if "R_pred" in y_pred:
                # Structural R mode: compute NMSE directly on R_pred (already available)
                # This gives a direct gradient path: NMSE → R_pred → build_structured_R → geometry
                # NOTE: R_blend may also exist (train.py wraps R_pred), but we want the RAW R_pred
                # here for a clean aux gradient, not the trace-normalized R_blend.
                R_pred_eff = build_effective_cov_torch(
                    y_pred['R_pred'],
                    snr_db=y_true.get("snr_db", None),
                    R_samp=None,
                    beta=None,
                    diag_load=True,
                    apply_shrink=("snr_db" in y_true),
                    target_trace=float(cfg.N),
                )
                loss_nmse_pred = self._nmse_cov(R_pred_eff, R_eff_true).mean()
            elif ("cov_fact_angle" in y_pred) and ("cov_fact_range" in y_pred):
                # Legacy factor mode: rebuild R_pred from factors
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
                loss_nmse_pred = self._nmse_cov(R_pred_aux, R_eff_true).mean()
        
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

        # NOTE: K-loss removed - using MVDR peak detection instead (K-free localization)
        # NOTE: Eigengap/margin losses removed - they were disabled globally and caused SVD instability

        # Aux: wrapped Huber on angles + Huber on log-range
        # Also compute permutation-aware mask BCE if masks are available
        # Note: lam_mask_bce is also scaled by mask_loss_scale (warmup)
        loss_mask_bce = torch.tensor(0.0, device=device)
        lam_mask_bce = float(getattr(mdl_cfg, "LAM_AUX_MASK_BCE", 0.0)) * self.mask_loss_scale
        
        if bool(getattr(cfg, "AUX_LOSS_PERM_INVARIANT", True)):
            # Get mask predictions if available (for permutation-aware BCE)
            mask_for_perm = None
            if lam_mask_bce > 0.0 and ("aux_mask" in y_pred or "aux_mask_logit" in y_pred):
                if "aux_mask" in y_pred:
                    mask_for_perm = y_pred["aux_mask"].to(device).float()
                else:
                    mask_for_perm = torch.sigmoid(y_pred["aux_mask_logit"].to(device).float())
            
            # Use Hungarian (optimal) matching for permutation-invariant aux loss.
            # Soft matching is enabled during warmup via set_aux_match() to prevent assignment flips.
            if mask_for_perm is not None:
                aux_l2, loss_mask_bce = _perm_invariant_aux_loss(
                    phi_p, theta_p, r_p,
                    phi_t, theta_t, r_t,
                    K_true,
                    mask_p=mask_for_perm,
                    soft=bool(getattr(self, "aux_match_soft", False)),
                    soft_tau=float(getattr(self, "aux_match_tau", 0.25)),
                )
                if not hasattr(self, "_mask_bce_logged"):
                    print(f"[LOSS DEBUG] Permutation-aware mask BCE: enabled @ weight={lam_mask_bce}", flush=True)
                    self._mask_bce_logged = True
            else:
                aux_l2 = _perm_invariant_aux_loss(
                    phi_p, theta_p, r_p,
                    phi_t, theta_t, r_t,
                    K_true,
                    soft=bool(getattr(self, "aux_match_soft", False)),
                    soft_tau=float(getattr(self, "aux_match_tau", 0.25)),
                )
        else:
            phi_huber = (_wrapped_huber_loss(phi_p, phi_t) * mask).sum() / (mask.sum() + 1e-9)
            theta_huber = (_wrapped_huber_loss(theta_p, theta_t) * mask).sum() / (mask.sum() + 1e-9)
            theta_huber *= mdl_cfg.THETA_LOSS_SCALE  # Emphasize elevation for better θ accuracy
            ang_err = phi_huber + theta_huber
            rng_err_log = (_range_huber_loss(r_p, r_t) * mask).sum() / (mask.sum() + 1e-9)
            aux_l2 = ang_err + rng_err_log

        # ------------------------------------------------------------
        # IMPORTANT: Avoid slot-index-biased auxiliary terms when using
        # permutation-invariant aux supervision.
        #
        # Historically we added:
        #  - a Chamfer term on (phi,theta) (lam_peak)
        #  - a tiny slot-index range MSE term (range_raw)
        #
        # Both of these were implemented with the "first K slots are active"
        # mask (slot-index semantics). When aux is permutation-invariant, this
        # creates contradictory gradients:
        #   - aux_l2 matches *any* slots to GT via optimal assignment
        #   - chamfer/range_raw push *specific slot indices* toward GT
        #
        # In practice this can destabilize training (φ jumps toward random/edge
        # predictions even when gradients exist), exactly the failure mode
        # observed in full training logs.
        #
        # Therefore:
        #  - If AUX_LOSS_PERM_INVARIANT=True, disable these slot-index-biased
        #    auxiliaries by default.
        #  - If you really want them, re-enable only after introducing a
        #    canonical slot ordering (scheduled sorted matching) or rewrite them
        #    to use the same assignment as aux_l2.
        # ------------------------------------------------------------
        use_perm_aux = bool(getattr(cfg, "AUX_LOSS_PERM_INVARIANT", True))
        if use_perm_aux:
            peak_l2 = torch.tensor(0.0, device=device)
            range_raw = torch.tensor(0.0, device=device)
            if (self.lam_peak != 0.0) and (not hasattr(self, "_perm_aux_peak_disabled_logged")):
                print("[LOSS DEBUG] Chamfer/range_raw disabled under perm-invariant aux (avoid slot-index bias).", flush=True)
                self._perm_aux_peak_disabled_logged = True
        else:
            # Chamfer on (phi,theta)
            peak_l2 = self._angle_chamfer(phi_p, theta_p, phi_t, theta_t, mask)

            # small linear range term normalized by span (optional)
            r_span = (cfg.RANGE_R[1] - cfg.RANGE_R[0] + 1e-9)
            range_raw = (((r_p - r_t) / r_span)**2 * mask).sum() / (mask.sum() + 1e-9)

        # CRITICAL FIX (2026-02-05): Slot diversity loss to prevent collapse.
        # Without this, all slots converge to predict identical values (φ std = 0.04°).
        # The permutation-invariant loss doesn't penalize this symmetric solution.
        loss_diversity = torch.tensor(0.0, device=device)
        lam_diversity = float(getattr(mdl_cfg, "LAM_SLOT_DIVERSITY", 0.1))
        if lam_diversity > 0.0 and use_perm_aux:
            loss_diversity = self._slot_diversity_loss(phi_p, theta_p, r_p)
            if not hasattr(self, "_diversity_logged"):
                print(f"[LOSS DEBUG] Slot diversity loss: enabled @ weight={lam_diversity:.3f}", flush=True)
                self._diversity_logged = True
        
        loss_align = torch.tensor(0.0, device=device)
        if getattr(mdl_cfg, "LAM_ALIGN", 0.0) > 0.0:
            # Expert fix: Re-enabled subspace alignment with SVD + projector (now safe)
            Rc = y_pred['R_blend'] if 'R_blend' in y_pred else R_hat
            loss_align = self._subspace_align(Rc, phi_p, theta_p, r_p, K_true)

        # ------------------------------------------------------------
        # Presence / mask supervision (permutation-safe)
        # ------------------------------------------------------------
        # We avoid slot-index BCE targets (slots are unordered). Instead:
        #  - enforce that the *count* of active slots matches K_true
        #  - optionally encourage binarization (push probabilities toward {0,1})
        #
        # CRITICAL: Mask losses are scaled by self.mask_loss_scale (set by Trainer).
        # During early training, geometry is random, so permutation matching is unstable,
        # which makes mask gradients noisy and interferes with geometry learning.
        # We warm up mask losses only after geometry has stabilized.
        loss_mask = torch.tensor(0.0, device=device)
        lam_mask = float(getattr(mdl_cfg, "LAM_AUX_MASK", 0.0)) * self.mask_loss_scale
        lam_mask_bin = float(getattr(mdl_cfg, "LAM_AUX_MASK_BIN", 0.0)) * self.mask_loss_scale
        if lam_mask > 0.0 and ("aux_mask" in y_pred or "aux_mask_logit" in y_pred):
            if "aux_mask" in y_pred:
                m = y_pred["aux_mask"].to(device).float()
            else:
                m = torch.sigmoid(y_pred["aux_mask_logit"].to(device).float())
            # Clamp for safety; keep gradients (avoid hard 0/1 due to numerical extremes)
            m = m.clamp(min=1e-4, max=1.0 - 1e-4)
            kf = K_true.to(device).float()
            # Count loss: encourage sum(m) ≈ K_true (normalized by K_MAX for scale stability)
            count_err = (m.sum(dim=1) - kf) / float(max(1, int(getattr(cfg, "K_MAX", 5))))
            loss_mask = (count_err ** 2).mean()
            if lam_mask_bin > 0.0:
                # Binarization penalty: minimized at 0 or 1, maximized at 0.5
                loss_mask = loss_mask + lam_mask_bin * (m * (1.0 - m)).mean()
            if not hasattr(self, "_mask_loss_logged"):
                print(f"[LOSS DEBUG] Mask count loss: enabled @ weight={lam_mask:.3f} (bin={lam_mask_bin:.3f}, scale={self.mask_loss_scale:.2f})", flush=True)
                self._mask_loss_logged = True
        elif self.mask_loss_scale < 1e-6 and not hasattr(self, "_mask_loss_warmup_logged"):
            print(f"[LOSS DEBUG] Mask losses: WARMUP (scale={self.mask_loss_scale:.2f}, disabled until geometry stabilizes)", flush=True)
            self._mask_loss_warmup_logged = True

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
            # Peak-contrast should use the *effective* covariance (diag-loaded + shrunk) and
            # the near-field GT range to match inference physics.
            loss_peak_contrast = self._peak_contrast_loss(R_eff_pred, phi_t, theta_t, r_t, K_true)
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
            + lam_mask * loss_mask                             # Slot presence/mask count supervision
            + lam_mask_bce * loss_mask_bce                     # Permutation-aware mask BCE (strongest)
            + lam_diversity * loss_diversity                   # CRITICAL: Slot diversity (prevents collapse)
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

        gap=0.0  # Eigengap loss removed (was always disabled)

        phi_p=y_pred.get("phi_soft",torch.zeros_like(phi_t))
        theta_p=y_pred.get("theta_soft",torch.zeros_like(theta_t))
        r_p=y_pred.get("r_soft",torch.zeros_like(r_t))
        if ("phi_soft" not in y_pred) or ("theta_soft" not in y_pred) or ("r_soft" not in y_pred):
            aux=y_pred.get("phi_theta_r",None)
            if aux is not None and aux.shape[-1]>=3*cfg.K_MAX:
                aux=aux.to(device).float()
                phi_p=aux[:,:cfg.K_MAX]; theta_p=aux[:,cfg.K_MAX:2*cfg.K_MAX]; r_p=aux[:,2*cfg.K_MAX:3*cfg.K_MAX]

        # Compute aux metrics (permutation-invariant matching is the default)
        if bool(getattr(cfg, "AUX_LOSS_PERM_INVARIANT", True)):
            aux_l2 = _perm_invariant_aux_loss(phi_p, theta_p, r_p, phi_t, theta_t, r_t, K_true).item()
        else:
            # Index-based fallback (not recommended for multi-source scenes)
            phi_huber_sum=(_wrapped_huber_loss(phi_p,phi_t)*mask).sum()/(mask.sum()+1e-9)
            theta_huber_sum=(_wrapped_huber_loss(theta_p,theta_t)*mask).sum()/(mask.sum()+1e-9)
            theta_huber_sum *= mdl_cfg.THETA_LOSS_SCALE
            ang_err=phi_huber_sum+theta_huber_sum
            r_p_pos=torch.clamp(r_p,min=1e-6); r_t_pos=torch.clamp(r_t,min=1e-6)
            rng_err_log=(((torch.log(r_p_pos)-torch.log(r_t_pos))**2)*mask).sum()/(mask.sum()+1e-9)
            aux_l2=(ang_err+rng_err_log).item()
        
        # Compute per-term scalars for logging (approximate, by-index)
        phi_huber_sum = (_wrapped_huber_loss(phi_p, phi_t) * mask).sum() / (mask.sum() + 1e-9)
        theta_huber_sum = (_wrapped_huber_loss(theta_p, theta_t) * mask).sum() / (mask.sum() + 1e-9)
        theta_huber_sum *= mdl_cfg.THETA_LOSS_SCALE
        r_p_pos = torch.clamp(r_p, min=1e-6); r_t_pos = torch.clamp(r_t, min=1e-6)
        rng_err_log = (((torch.log(r_p_pos) - torch.log(r_t_pos)) ** 2) * mask).sum() / (mask.sum() + 1e-9)
        
        peak=self._angle_chamfer(phi_p,theta_p,phi_t,theta_t,mask).item()
        
        # Margin loss removed (was always disabled)
        margin = 0.0
        align = (self._subspace_align(R_true, phi_p, theta_p, r_p, K_true).item() 
                if getattr(mdl_cfg, "LAM_ALIGN", 0.0) > 0.0 else 0.0)
        
        # NEW: Training-inference alignment losses
        loss_subspace_align = 0.0
        loss_peak_contrast = 0.0
        if self.lam_subspace_align > 0.0:
            loss_subspace_align = self._subspace_alignment_loss(R_hat, R_true, K_true, ptr_gt)
        if self.lam_peak_contrast > 0.0:
            # Use effective covariance (diag-loaded + optional shrink) to match the forward loss path.
            from .covariance_utils import build_effective_cov_torch
            R_pred_in = y_pred['R_blend'] if 'R_blend' in y_pred else R_hat
            R_eff = build_effective_cov_torch(
                R_pred_in,
                snr_db=y_true.get("snr_db", None),
                R_samp=None,
                beta=None,
                diag_load=True,
                apply_shrink=("snr_db" in y_true),
                target_trace=float(cfg.N),
            )
            loss_peak_contrast = self._peak_contrast_loss(R_eff, phi_t, theta_t, r_t, K_true)
        
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

# ==============================================================================
# CRITICAL FIX: Sorted matching to break symmetric equilibrium
# ==============================================================================

