
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from .configs import cfg, mdl_cfg
from .covariance_utils import trace_norm_torch, shrink_torch, build_effective_cov_torch

class AntiDiagPool(nn.Module):
    """
    Anti-diagonal pooling for covariance matrices.
    Sums all elements along each anti-diagonal (2N-1 diagonals total).
    Helps especially at low L where covariance info is limited.
    """
    def __init__(self, N_H, N_V, out_dim=None):
        super().__init__()
        self.N_H = N_H
        self.N_V = N_V 
        self.N = N_H * N_V
        
        # Create anti-diagonal sum mappings
        # For an N×N matrix, there are 2N-1 anti-diagonals
        # Main anti-diagonal is i+j = N-1
        # Other anti-diagonals have i+j = k for k in [0, 2N-2]
        self.num_antidiags = 2 * self.N - 1
        
        # Build indices for each anti-diagonal
        antidiag_indices = [[] for _ in range(self.num_antidiags)]
        for i in range(self.N):
            for j in range(self.N):
                antidiag_idx = i + j  # Sum of indices determines anti-diagonal
                antidiag_indices[antidiag_idx].append(i * self.N + j)
        
        # Store as tensors for efficient gathering
        self.antidiag_groups = []
        for group in antidiag_indices:
            if group:  # Only non-empty groups
                self.register_buffer(f'antidiag_{len(self.antidiag_groups)}', 
                                   torch.tensor(group, dtype=torch.long))
                self.antidiag_groups.append(len(self.antidiag_groups))
        
        # Optional projection layer
        if out_dim is not None:
            self.proj = nn.Linear(2 * self.num_antidiags, out_dim)  # real + imag parts
        else:
            self.proj = None
    
    def forward(self, R_complex):
        """
        R_complex: [B, N, N] complex covariance matrix
        Returns: [B, out_dim] or [B, 2*(2N-1)] anti-diagonal sums
        """
        B = R_complex.shape[0]
        R_flat = R_complex.view(B, -1)  # [B, N*N]
        
        # Sum elements along each anti-diagonal (normalized by count)
        antidiag_sums = []
        for i, group_idx in enumerate(self.antidiag_groups):
            indices = getattr(self, f'antidiag_{group_idx}')
            diagonal_sum = R_flat[:, indices].sum(dim=1)  # [B,]
            # Normalize by element count to remove length bias
            diagonal_mean = diagonal_sum / len(indices)
            antidiag_sums.append(diagonal_mean)
        
        antidiag_tensor = torch.stack(antidiag_sums, dim=1)  # [B, 2N-1]
        
        # Convert complex to real features [real, imag]
        features = torch.cat([antidiag_tensor.real, antidiag_tensor.imag], dim=1)  # [B, 2*(2N-1)]
        
        if self.proj is not None:
            features = self.proj(features)
            
        return features

class SoftArgmax2D(nn.Module):
    def __init__(self, grid_phi, grid_theta, tau=0.15):
        super().__init__()
        self.register_buffer('grid_phi', torch.tensor(grid_phi, dtype=torch.float32))
        self.register_buffer('grid_theta', torch.tensor(grid_theta, dtype=torch.float32))
        self.tau = tau

    def set_tau(self, tau):
        """Update softmax temperature for annealing"""
        self.tau = tau
    
    def forward(self, P):
        # P: [B, Gp, Gt]
        B, Gp, Gt = P.shape
        m = P.view(B, -1).amax(-1, keepdim=True).view(B, 1, 1)
        beta = F.softmax(((P - m) / max(self.tau, 1e-2)).view(B, -1), dim=-1).view(B, Gp, Gt)
        mp = beta.sum(2)   # [B, Gp]
        mt = beta.sum(1)   # [B, Gt]
        phi   = (mp * self.grid_phi).sum(1)
        theta = (mt * self.grid_theta).sum(1)
        return phi, theta

class HybridModel(nn.Module):
    """
    CNN + Transformer encoder on snapshots, plus H/C feature paths,
    with soft-argmax grid head + auxiliary φ/θ/r and K classification.
    """
    def _choose_heads(self, D, pref):
        if pref is None: pref = 6
        if D % pref == 0 and pref >= 1:
            return pref
        for h in [8, 6, 5, 4, 3, 2, 1]:
            if h <= D and D % h == 0:
                return h
        return 1
    
    def _apply_psd_parameterization(self, cf_raw):
        """
        Apply PSD parameterization to covariance factor: R̂ = A Aᴴ + εI
        Input: cf_raw [B, N*K*2] - raw factor output
        Output: cf_psd [B, N*K*2] - factor ensuring R̂ is PSD
        """
        B = cf_raw.shape[0]
        
        # Ensure full precision for complex operations (AMP uses float16 which doesn't support complex batch ops)
        cf_raw_fp32 = cf_raw.float()
        
        # Reshape to [B, 2, N, K] for complex conversion
        cf_reshape = cf_raw_fp32.view(B, 2, cfg.N, cfg.K_MAX)
        A = torch.complex(cf_reshape[:, 0], cf_reshape[:, 1])  # [B, N, K] complex64
        
        # CHEAP & SAFE: Build R̂ = A Aᴴ + εI directly (no eigendecomp)
        # This ensures PSD without backprop through eigenvectors
        eps_psd = getattr(mdl_cfg, 'EPS_PSD', 1e-4)
        
        # Use the original factor directly (already PSD by construction)
        A_new = A  # [B, N, K] - no eigendecomp needed
        
        # Convert back to real/imag format and match input dtype
        cf_psd = torch.stack([A_new.real, A_new.imag], dim=1)  # [B, 2, N, K]
        cf_psd = cf_psd.view(B, 2 * cfg.N * cfg.K_MAX)  # [B, N*K*2]
        
        # Convert back to original dtype for consistency with AMP
        return cf_psd.to(cf_raw.dtype)

    def __init__(self):
        super().__init__()
        D = mdl_cfg.D_MODEL
        
        # Lock the geometry (so HPO can't drift)
        assert cfg.M == cfg.M_BS, f"M_BS mismatch: {cfg.M} != {cfg.M_BS}"
        assert cfg.N == cfg.N_H * cfg.N_V, f"N mismatch: {cfg.N} != {cfg.N_H}*{cfg.N_V}"
        assert cfg.L > 0, f"L must be positive: {cfg.L}"
        assert cfg.M_beams > 0, f"M_beams must be positive: {cfg.M_beams}"
        assert cfg.K_MAX > 0, f"K_MAX must be positive: {cfg.K_MAX}"
        # --- y path (snapshots) ---
        self.y_conv1 = nn.Conv1d(cfg.M * 2, D // 2, 5, padding=2)
        self.y_dw    = nn.Conv1d(D // 2, D // 2, 3, padding=1, groups=D // 2)
        self.y_conv2 = nn.Conv1d(D // 2, D, 1)

        # --- transformer encoder on sequence (length = L) ---
        nheads = self._choose_heads(D, getattr(mdl_cfg, 'NUM_HEADS', 6))
        ff_dim = getattr(mdl_cfg, 'FF_DIM', D * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=nheads,
            dim_feedforward=ff_dim,
            dropout=getattr(mdl_cfg, 'DROPOUT', 0.1),
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, getattr(mdl_cfg, 'N_BLOCKS', 4))

        # --- H and codes paths ---
        self.H_proj    = nn.Linear(cfg.L * cfg.M * 2, D // 2)  # FIXED: L*M (not M*N) to match H.reshape(B,-1)
        self.codes_conv= nn.Conv1d(cfg.N * 2, D // 2, 5, padding=2)

        # --- SNR embedding (lets the backbone adapt features to noise regime) ---
        # NOTE: This changes the model architecture and invalidates old checkpoints (expected).
        self.snr_dim = max(8, D // 8)
        self.snr_embed = nn.Sequential(
            nn.Linear(1, self.snr_dim),
            nn.GELU(),
            nn.Linear(self.snr_dim, self.snr_dim),
        )

        # --- fusion ---
        self.fusion = nn.Linear(D + D // 2 + D // 2 + self.snr_dim, D)

        # --- normalization (stabilize large heads) ---
        self.cov_ln = nn.LayerNorm(D)

        # --- covariance factors (angle & range) ---
        self.cov_fact_angle = nn.Linear(D, cfg.N * cfg.K_MAX * 2)   # real+imag
        self.cov_fact_range = nn.Linear(D, cfg.N * cfg.K_MAX * 2)   # real+imag
        
        # --- AntiDiagPool for enhanced covariance features ---
        self.use_antidiag = getattr(mdl_cfg, 'USE_ANTIDIAG_POOL', True)
        if self.use_antidiag:
            antidiag_dim = D // 4  # Compact representation
            self.antidiag_pool = AntiDiagPool(cfg.N_H, cfg.N_V, out_dim=antidiag_dim)
            self.fusion_with_antidiag = nn.Linear(D + antidiag_dim, D)
        else:
            self.fusion_with_antidiag = nn.Identity()  # Pass-through when disabled

        # NOTE: K-head removed - using MVDR peak detection instead (K-free localization)
        
        # --- soft-argmax grid head for angles ---
        G = getattr(mdl_cfg, 'INFERENCE_GRID_SIZE_COARSE', 61)
        # Asymmetric FOV: φ ±60° (120° total), θ ±30° (60° total)
        phi_grid   = np.linspace(-cfg.ANGLE_RANGE_PHI, cfg.ANGLE_RANGE_PHI, G).astype('float32')     # ±60°
        theta_grid = np.linspace(-cfg.ANGLE_RANGE_THETA, cfg.ANGLE_RANGE_THETA, G).astype('float32') # ±30°
        self.logits_gg   = nn.Linear(D, cfg.K_MAX * G * G)
        self.soft_argmax = SoftArgmax2D(phi_grid, theta_grid, tau=getattr(mdl_cfg, 'SOFTMAX_TAU', 0.15))

        # --- auxiliary φ/θ/r (r via Softplus; loss uses log-range) ---
        self.aux_angles = nn.Linear(D, 2 * cfg.K_MAX)
        self.aux_range  = nn.Sequential(nn.Linear(D, cfg.K_MAX), nn.Softplus())

    def set_dropout(self, p):
        """Update dropout probability for annealing"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = p
    
    def set_tau(self, tau):
        """Update softmax temperature for annealing"""
        self.soft_argmax.set_tau(tau)
    
    # ----------------------------
    # Phase-aware freezing helper
    # ----------------------------
    def set_trainable_for_phase(
        self,
        freeze_backbone: bool = False,
        freeze_aux: bool = False,
        freeze_cov: bool = False,
    ):
        """
        Control which parts of the model are trainable for phase-specific runs.
        - freeze_backbone: freeze feature extractor
        - freeze_aux: freeze auxiliary angle/range heads
        - freeze_cov: freeze covariance factor predictors
        """
        def _set(mods, flag: bool):
            for m in mods:
                for p in m.parameters():
                    p.requires_grad = not flag

        # Backbone: feature stacks
        backbone_modules = [
            m for m in [
                getattr(self, "y_conv1", None),
                getattr(self, "y_dw", None),
                getattr(self, "y_conv2", None),
                getattr(self, "transformer", None),
                getattr(self, "H_proj", None),
                getattr(self, "codes_conv", None),
                getattr(self, "fusion", None),
                getattr(self, "fusion_with_antidiag", None),
                getattr(self, "antidiag_pool", None),
            ] if m is not None
        ]

        # Covariance factor heads
        factor_modules = [
            m for m in [
                getattr(self, "cov_fact_angle", None),
                getattr(self, "cov_fact_range", None),
            ] if m is not None
        ]

        # Aux heads
        aux_modules = [
            m for m in [
                getattr(self, "logits_gg", None),
                getattr(self, "aux_angles", None),
                getattr(self, "aux_range", None),
            ] if m is not None
        ]

        _set(backbone_modules, freeze_backbone)
        _set(factor_modules, freeze_cov)
        _set(aux_modules, freeze_aux)
    
    def _hermitize_trace_norm(self, R):
        """Hermitize and trace-normalize covariance matrix"""
        R = 0.5*(R + R.conj().transpose(-2, -1))
        tr = torch.diagonal(R, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-9)
        return R / tr.view(-1,1,1)
    
    def _shrink_alpha(self, snr_db, base=None):
        """SNR-aware shrinkage coefficient (kept consistent with physics/covariance_utils)."""
        from .physics import alpha_from_snr_db
        base = float(getattr(mdl_cfg, "SHRINK_BASE_ALPHA", 1e-3)) if base is None else float(base)
        alpha0 = float(alpha_from_snr_db(float(snr_db), L=int(getattr(cfg, "L", 16))))
        alpha = alpha0 * (base / 1e-3)
        return float(max(1e-4, min(alpha, 0.25)))
    
    def _whiten_covariance(self, R):
        """Whiten covariance matrix: R_w = D^(-1/2) R D^(-1/2) where D = diag(R)"""
        B, N, _ = R.shape
        device = R.device
        dtype = R.dtype
        
        # Get diagonal elements
        diag = torch.diagonal(R, dim1=-2, dim2=-1)  # [B, N]
        diag_real = diag.real
        
        # Avoid division by zero
        diag_sqrt = torch.sqrt(torch.clamp(diag_real, min=1e-8))  # [B, N]
        
        # Create D^(-1/2) matrix (complex)
        D_inv_sqrt = torch.diag_embed(1.0 / diag_sqrt).to(dtype)  # [B, N, N] - keep complex dtype
        
        # Whiten: R_w = D^(-1/2) R D^(-1/2)
        R_whitened = D_inv_sqrt @ R @ D_inv_sqrt
        
        return R_whitened

    def forward(self, y, H, codes, snr_db=None, R_samp=None):
        """
        y:     [B, L, M, 2]   (ri)
        H:     [B, L, M, 2]   (ri) - FIXED: L first, not M
        codes: [B, L, N, 2]   (ri) - FIXED: L first, not Lc
        snr_db: [B] optional SNR values for shrinkage
        R_samp: Optional sample covariance for hybrid blending (supports complex [B,N,N] or RI [B,N,N,2])
        """
        B, L = y.shape[0], y.shape[1]
        
        # CRITICAL: Assert shapes right before use
        assert y.shape == (B, cfg.L, cfg.M, 2), f"y shape mismatch: {y.shape} != ({B}, {cfg.L}, {cfg.M}, 2)"
        assert H.shape == (B, cfg.L, cfg.M, 2), f"H shape mismatch: {H.shape} != ({B}, {cfg.L}, {cfg.M}, 2)"
        assert codes.shape == (B, cfg.L, cfg.N, 2), f"codes shape mismatch: {codes.shape} != ({B}, {cfg.L}, {cfg.N}, 2)"

        # --- y sequence features ---
        y_seq = y.reshape(B, L, cfg.M * 2).permute(0, 2, 1)               # [B, 2M, L]
        x = F.gelu(self.y_conv1(y_seq))
        x = self.y_dw(x)
        x = F.gelu(self.y_conv2(x))                                       # [B, D, L]
        x = x.permute(0, 2, 1)                                            # [B, L, D]
        x = self.transformer(x).mean(1)                                   # [B, D]

        # --- H features ---
        # Enforce the shape and match H_proj
        B, Lh, Mh, Ch = H.shape
        assert (Lh, Mh, Ch) == (cfg.L, cfg.M, 2), f"Expected H[B,{cfg.L},{cfg.M},2], got {H.shape}"
        H_feat = F.gelu(self.H_proj(H.reshape(B, -1)))                    # [B, D/2]

        # --- codes features ---
        Lc = codes.shape[1]
        C_seq = codes.reshape(B, Lc, cfg.N * 2).permute(0, 2, 1)          # [B, 2N, Lc]
        c_feat = F.gelu(self.codes_conv(C_seq)).mean(2)                   # [B, D/2]

        # --- fuse ---
        # Provide SNR to the backbone as a learnable conditioning feature.
        if snr_db is None:
            snr_t = torch.zeros((B, 1), device=x.device, dtype=torch.float32)
        else:
            # Accept scalar or [B]; cast to float32 for stability.
            snr_t = snr_db
            if not torch.is_tensor(snr_t):
                snr_t = torch.tensor(snr_t, device=x.device)
            snr_t = snr_t.to(device=x.device, dtype=torch.float32).view(B, 1)
        snr_feat = self.snr_embed(snr_t)  # [B, snr_dim]
        feats = F.gelu(self.fusion(torch.cat([x, H_feat, c_feat, snr_feat], dim=1)))  # [B, D]

        # --- factor heads (simplified - disable PSD parameterization for AMP compatibility) ---
        feats_cov = self.cov_ln(feats)
        cf_ang = self.cov_fact_angle(feats_cov)                          # [B, N*K*2]
        cf_rng = self.cov_fact_range(feats_cov)                          # [B, N*K*2]
        
        # Note: PSD parameterization re-enabled in spectral branch under autocast(False)
        # The main factor heads use direct linear layers for speed, while K-head uses PSD covariance
        
        # --- AntiDiagPool features from learned covariance (if enabled) ---
        if self.use_antidiag:
            # Build covariance from angle factors for anti-diagonal extraction
            def _vec2c(v):
                v = v.float()
                xr, xi = v[:, ::2], v[:, 1::2]
                return torch.complex(xr.view(-1, cfg.N, cfg.K_MAX), xi.view(-1, cfg.N, cfg.K_MAX))
            
            A_angle = _vec2c(cf_ang)  # [B, N, K]
            # Apply the same magnitude leash used elsewhere before whitening.
            if bool(getattr(cfg, "FACTOR_COLNORM_ENABLE", True)):
                eps = float(getattr(cfg, "FACTOR_COLNORM_EPS", 1e-6))
                max_norm = float(getattr(cfg, "FACTOR_COLNORM_MAX", 1e3))
                col = torch.linalg.norm(A_angle, dim=-2, keepdim=True).clamp_min(eps)  # [B,1,K]
                if max_norm > 0:
                    col = col.clamp(max=max_norm)
                A_angle = A_angle / col
            R_learned = A_angle @ A_angle.conj().transpose(-2, -1)  # [B, N, N]
            
            # CRITICAL FIX: Whiten covariance before AntiDiagPool for better angle separability
            R_whitened = self._whiten_covariance(R_learned)  # [B, N, N]
            
            # Extract anti-diagonal features and enhance main features
            antidiag_feat = self.antidiag_pool(R_whitened)  # [B, antidiag_dim]
            feats_enhanced = F.gelu(self.fusion_with_antidiag(torch.cat([feats, antidiag_feat], dim=1)))  # [B, D]
        else:
            feats_enhanced = feats  # Use original features when AntiDiagPool is disabled

        # --- soft-argmax grid head (angles) with enhanced features ---
        G = getattr(mdl_cfg, 'INFERENCE_GRID_SIZE_COARSE', 61)
        logits = self.logits_gg(feats_enhanced).view(B, cfg.K_MAX, G, G)   # [B, K, G, G]
        phi_list, theta_list = [], []
        for k in range(cfg.K_MAX):
            phi_k, theta_k = self.soft_argmax(logits[:, k])
            phi_list.append(phi_k); theta_list.append(theta_k)
        phi_soft   = torch.stack(phi_list,   dim=1)                        # [B, K]
        theta_soft = torch.stack(theta_list, dim=1)                        # [B, K]

        # --- auxiliary ptr (angles + positive range) with enhanced features ---
        aux_angles = self.aux_angles(feats_enhanced)                       # [B, 2K]
        aux_range  = self.aux_range(feats_enhanced)                        # [B, K]  (positive)
        aux_ptr    = torch.cat([aux_angles, aux_range], dim=1)             # [B, 3K]

        # NOTE: K-head removed - using MVDR peak detection instead (K-free localization)
        # The number of sources is determined by peak detection on MVDR spectrum

        return {
            "cov_fact_angle": cf_ang,
            "cov_fact_range": cf_rng,
            "phi_theta_r":    aux_ptr,
            "phi_soft":       phi_soft,
            "theta_soft":     theta_soft,
        }


# =============================================================================
# SpectrumRefiner CNN - Learned Denoising/Sharpening of Physics-Based Spectrum
# =============================================================================

class SpectrumRefiner(nn.Module):
    """
    Small U-Net-like CNN that takes a physics-based MVDR spectrum and learns
    to sharpen peaks and suppress noise. Outputs a probability heatmap where
    peaks indicate source locations.
    
    Architecture:
        - Input: [B, 1, H, W] MVDR spectrum (2D slice at a range plane)
        - Encoder: 3 conv blocks with downsampling
        - Bottleneck: dilated convolutions for large receptive field
        - Decoder: 3 conv blocks with upsampling + skip connections
        - Output: [B, 1, H, W] refined probability map (sigmoid activation)
    
    Training:
        - Supervised with Gaussian blobs centered at GT source (φ, θ) locations
        - BCE loss between refined map and GT heatmap
    
    Inference:
        - Apply to each range plane of MVDR spectrum
        - Extract peaks via NMS on refined maps
    """
    
    def __init__(self, in_channels: int = 1, base_filters: int = 32, 
                 use_skip: bool = True, use_batch_norm: bool = True):
        super().__init__()
        self.use_skip = use_skip
        self.use_batch_norm = use_batch_norm
        
        F = base_filters  # 32 base filters
        
        # ========== Encoder ==========
        # Block 1: [B, 1, H, W] -> [B, F, H/2, W/2]
        self.enc1 = self._conv_block(in_channels, F, stride=2)
        
        # Block 2: [B, F, H/2, W/2] -> [B, 2F, H/4, W/4]
        self.enc2 = self._conv_block(F, F*2, stride=2)
        
        # Block 3: [B, 2F, H/4, W/4] -> [B, 4F, H/8, W/8]
        self.enc3 = self._conv_block(F*2, F*4, stride=2)
        
        # ========== Bottleneck with Dilated Convolutions ==========
        # Large receptive field without downsampling
        self.bottleneck = nn.Sequential(
            nn.Conv2d(F*4, F*4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(F*4) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(F*4, F*4, 3, padding=4, dilation=4),
            nn.BatchNorm2d(F*4) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        
        # ========== Decoder with Skip Connections ==========
        # Block 3: [B, 4F, H/8, W/8] -> [B, 2F, H/4, W/4]
        self.dec3_up = nn.ConvTranspose2d(F*4, F*2, kernel_size=2, stride=2)
        self.dec3_conv = self._conv_block(F*4 if use_skip else F*2, F*2, stride=1)
        
        # Block 2: [B, 2F, H/4, W/4] -> [B, F, H/2, W/2]
        self.dec2_up = nn.ConvTranspose2d(F*2, F, kernel_size=2, stride=2)
        self.dec2_conv = self._conv_block(F*2 if use_skip else F, F, stride=1)
        
        # Block 1: [B, F, H/2, W/2] -> [B, F/2, H, W]
        self.dec1_up = nn.ConvTranspose2d(F, F//2, kernel_size=2, stride=2)
        self.dec1_conv = self._conv_block(F//2 + in_channels if use_skip else F//2, F//2, stride=1)
        
        # ========== Output Head ==========
        self.output_conv = nn.Conv2d(F//2, 1, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _conv_block(self, in_ch, out_ch, stride=1):
        """Double convolution block with optional downsampling"""
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    @staticmethod
    def _match_hw(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Match spatial dims (H,W) of `x` to `ref` by center-cropping or symmetric padding.
        This makes the refiner robust to odd input sizes (e.g. 181×361).
        """
        Hr, Wr = ref.shape[-2], ref.shape[-1]
        H, W = x.shape[-2], x.shape[-1]

        # Center-crop if too large
        if H > Hr:
            dh = H - Hr
            top = dh // 2
            x = x[:, :, top:top + Hr, :]
        if W > Wr:
            dw = W - Wr
            left = dw // 2
            x = x[:, :, :, left:left + Wr]

        # Symmetric pad if too small
        H2, W2 = x.shape[-2], x.shape[-1]
        if (H2 < Hr) or (W2 < Wr):
            pad_h = max(0, Hr - H2)
            pad_w = max(0, Wr - W2)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        return x

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W] - Physics-based spectrum (MVDR or MUSIC)
               Values should be normalized to [0, 1] or log-scaled
        
        Returns:
            out: [B, 1, H, W] - Refined probability map (sigmoid output)
        """
        # Encoder
        e1 = self.enc1(x)      # [B, F, H/2, W/2]
        e2 = self.enc2(e1)     # [B, 2F, H/4, W/4]
        e3 = self.enc3(e2)     # [B, 4F, H/8, W/8]
        
        # Bottleneck
        b = self.bottleneck(e3)  # [B, 4F, H/8, W/8]
        
        # Decoder with skip connections
        d3 = self.dec3_up(b)   # [B, 2F, H/4, W/4]
        d3 = self._match_hw(d3, e2)
        if self.use_skip:
            d3 = torch.cat([d3, e2], dim=1)  # [B, 4F, H/4, W/4]
        d3 = self.dec3_conv(d3)  # [B, 2F, H/4, W/4]
        
        d2 = self.dec2_up(d3)  # [B, F, H/2, W/2]
        d2 = self._match_hw(d2, e1)
        if self.use_skip:
            d2 = torch.cat([d2, e1], dim=1)  # [B, 2F, H/2, W/2]
        d2 = self.dec2_conv(d2)  # [B, F, H/2, W/2]
        
        d1 = self.dec1_up(d2)  # [B, F/2, H, W]
        d1 = self._match_hw(d1, x)
        if self.use_skip:
            d1 = torch.cat([d1, x], dim=1)  # [B, F/2+1, H, W]
        d1 = self.dec1_conv(d1)  # [B, F/2, H, W]
        
        # Output with sigmoid for probability
        out = torch.sigmoid(self.output_conv(d1))  # [B, 1, H, W]
        
        return out


class SpectrumRefinerLoss(nn.Module):
    """
    Loss function for training SpectrumRefiner.
    
    Creates Gaussian blob heatmaps at GT source locations and computes
    BCE loss between refined spectrum and GT heatmap.
    """
    
    def __init__(self, sigma_phi: float = 2.0, sigma_theta: float = 2.0,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 use_focal: bool = True):
        """
        Args:
            sigma_phi: Gaussian blob std in phi direction (grid cells)
            sigma_theta: Gaussian blob std in theta direction (grid cells)
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            use_focal: Use focal loss instead of BCE (better for sparse targets)
        """
        super().__init__()
        self.sigma_phi = sigma_phi
        self.sigma_theta = sigma_theta
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal
    
    def _create_gaussian_heatmap(self, phi_gt, theta_gt, K_true, 
                                  grid_phi, grid_theta, device):
        """
        Create Gaussian blob heatmap at GT source locations.
        
        Args:
            phi_gt: [B, K_max] GT azimuth in radians
            theta_gt: [B, K_max] GT elevation in radians
            K_true: [B] number of active sources
            grid_phi: [G_phi] phi grid values in radians
            grid_theta: [G_theta] theta grid values in radians
            device: torch device
            
        Returns:
            heatmap: [B, 1, G_phi, G_theta] with Gaussian blobs at GT locations
        """
        B = phi_gt.shape[0]
        G_phi = len(grid_phi)
        G_theta = len(grid_theta)
        
        # Create meshgrid for distance computation
        phi_mesh = grid_phi.view(1, G_phi, 1).expand(B, -1, G_theta)
        theta_mesh = grid_theta.view(1, 1, G_theta).expand(B, G_phi, -1)
        
        # Initialize heatmap
        heatmap = torch.zeros(B, G_phi, G_theta, device=device)
        
        # Grid spacing for sigma conversion
        d_phi = (grid_phi[-1] - grid_phi[0]) / (G_phi - 1)
        d_theta = (grid_theta[-1] - grid_theta[0]) / (G_theta - 1)
        
        sigma_phi_rad = self.sigma_phi * d_phi
        sigma_theta_rad = self.sigma_theta * d_theta
        
        # Add Gaussian blob for each GT source
        for b in range(B):
            K = int(K_true[b].item())
            for k in range(K):
                phi_k = phi_gt[b, k]
                theta_k = theta_gt[b, k]
                
                # Compute squared distance normalized by sigma
                d_phi_sq = ((phi_mesh[b] - phi_k) / sigma_phi_rad) ** 2
                d_theta_sq = ((theta_mesh[b] - theta_k) / sigma_theta_rad) ** 2
                
                # Gaussian blob
                blob = torch.exp(-0.5 * (d_phi_sq + d_theta_sq))
                
                # Max-blend (allows overlapping sources)
                heatmap[b] = torch.maximum(heatmap[b], blob)
        
        # Add channel dimension
        return heatmap.unsqueeze(1)  # [B, 1, G_phi, G_theta]
    
    def forward(self, pred_heatmap, phi_gt, theta_gt, K_true, grid_phi, grid_theta):
        """
        Compute loss between predicted heatmap and GT Gaussian blobs.
        
        Args:
            pred_heatmap: [B, 1, G_phi, G_theta] from SpectrumRefiner
            phi_gt: [B, K_max] GT azimuth in radians
            theta_gt: [B, K_max] GT elevation in radians  
            K_true: [B] number of active sources
            grid_phi: [G_phi] phi grid values in radians
            grid_theta: [G_theta] theta grid values in radians
            
        Returns:
            loss: scalar loss value
        """
        device = pred_heatmap.device
        
        # Create GT heatmap
        gt_heatmap = self._create_gaussian_heatmap(
            phi_gt, theta_gt, K_true, grid_phi, grid_theta, device
        )
        
        if self.use_focal:
            # Focal loss for sparse target handling
            # FL = -α(1-p)^γ log(p) for positive, -α p^γ log(1-p) for negative
            p = pred_heatmap.clamp(1e-7, 1 - 1e-7)
            
            pos_weight = self.focal_alpha * (1 - p) ** self.focal_gamma
            neg_weight = (1 - self.focal_alpha) * p ** self.focal_gamma
            
            pos_loss = -pos_weight * gt_heatmap * torch.log(p)
            neg_loss = -neg_weight * (1 - gt_heatmap) * torch.log(1 - p)
            
            loss = (pos_loss + neg_loss).mean()
        else:
            # Standard BCE
            loss = F.binary_cross_entropy(pred_heatmap, gt_heatmap, reduction='mean')
        
        return loss


def create_spectrum_refiner_with_pretrained_backbone(backbone_model, 
                                                      freeze_backbone: bool = True):
    """
    Factory function to create a combined model with:
    1. Pretrained backbone (HybridModel) for covariance prediction
    2. SpectrumRefiner for spectrum sharpening
    
    Args:
        backbone_model: Pretrained HybridModel
        freeze_backbone: Whether to freeze backbone weights
        
    Returns:
        Combined model that takes raw inputs and outputs refined spectrum
    """
    if freeze_backbone:
        for param in backbone_model.parameters():
            param.requires_grad = False
    
    refiner = SpectrumRefiner(in_channels=1, base_filters=32)
    
    class CombinedModel(nn.Module):
        def __init__(self, backbone, refiner):
            super().__init__()
            self.backbone = backbone
            self.refiner = refiner
        
        def forward(self, y, H, codes, mvdr_spectrum, snr_db=None, R_samp=None):
            """
            Args:
                y, H, codes: Raw inputs for backbone
                mvdr_spectrum: [B, 1, G_phi, G_theta] precomputed MVDR spectrum
                snr_db, R_samp: Optional backbone inputs
                
            Returns:
                dict with backbone outputs + refined_spectrum
            """
            # Run backbone
            backbone_out = self.backbone(y, H, codes, snr_db=snr_db, R_samp=R_samp)
            
            # Refine spectrum
            refined = self.refiner(mvdr_spectrum)
            
            backbone_out['refined_spectrum'] = refined
            return backbone_out
    
    return CombinedModel(backbone_model, refiner)
