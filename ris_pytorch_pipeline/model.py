
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

        # --- fusion ---
        self.fusion = nn.Linear(D + D // 2 + D // 2, D)

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

        # --- K classification head with spectral features ---
        # K ranges from 1 to K_MAX (1,2,3,4,5), so we need K_MAX classes (5)
        # We'll map K → K-1 for 0-indexed class labels
        
        # Spectral feature projection (MUST be in __init__ for optimizer!)
        # Features: 6 top eigenvalues + 1 tail mean + 5 eigenvalue gaps = 12 dims
        spec_dim = 12
        self.k_spec_proj = nn.Sequential(
            nn.Linear(spec_dim, D // 4),
            nn.GELU(),
            nn.Linear(D // 4, D // 4)
        )
        self.k_fuse = nn.Linear(D + D // 4, D)
        
        # K classification head with enhanced spectral features
        # Features: [gaps(K_MAX), ratios(K_MAX), log_slopes(K_MAX), mdl(K_MAX), mdl_onehot(K_MAX), λ1(1), λN(1)] = 5*K_MAX + 2
        k_feat_dim = 5*cfg.K_MAX + 2
        self.k_mlp = nn.Sequential(
            nn.LayerNorm(k_feat_dim),
            nn.Linear(k_feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, cfg.K_MAX)
        )

        # Ordinal K head (recommended): logits for P(K > t), t=1..K_MAX-1
        # Shape: [B, K_MAX-1] (for K_MAX=5 → 4 logits)
        self.k_ord_mlp = nn.Sequential(
            nn.LayerNorm(k_feat_dim),
            nn.Linear(k_feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, cfg.K_MAX - 1)
        )
        
        # SOTA FIX: Direct K-MLP bypass from main features (ensemble with spectral)
        # This gives K-head a direct gradient path that works from step 1
        # Ensemble: 0.5 * spectral_path + 0.5 * direct_path
        self.k_direct_mlp = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, cfg.K_MAX)
        )

        self.k_direct_ord_mlp = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, cfg.K_MAX - 1)
        )
        
        # Learnable logit scale for K-head
        self.k_logit_scale = nn.Parameter(torch.ones(1))
        
        # Initialize K-head with larger weights for better learning
        nn.init.xavier_uniform_(self.k_mlp[1].weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_mlp[3].weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_direct_mlp[1].weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_direct_mlp[3].weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_ord_mlp[1].weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_ord_mlp[3].weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_direct_ord_mlp[1].weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_direct_ord_mlp[3].weight, gain=1.0)
        # FIXED: Initialize bias with small positive values for class priors
        nn.init.constant_(self.k_mlp[1].bias, 0.1)
        nn.init.constant_(self.k_mlp[3].bias, 0.1)
        nn.init.constant_(self.k_direct_mlp[1].bias, 0.1)
        nn.init.constant_(self.k_direct_mlp[3].bias, 0.1)
        nn.init.constant_(self.k_ord_mlp[1].bias, 0.1)
        nn.init.constant_(self.k_ord_mlp[3].bias, 0.1)
        nn.init.constant_(self.k_direct_ord_mlp[1].bias, 0.1)
        nn.init.constant_(self.k_direct_ord_mlp[3].bias, 0.1)

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
        freeze_k: bool = False,
    ):
        """
        Control which parts of the model are trainable for phase-specific runs.
        - freeze_backbone: freeze feature extractor + covariance predictors
        - freeze_aux: freeze auxiliary angle/range heads
        - freeze_k: freeze K classification head
        """
        def _set(mods, flag: bool):
            for m in mods:
                for p in m.parameters():
                    p.requires_grad = not flag

        # Backbone: feature stacks and covariance factor predictors
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

        # Covariance factor heads (should remain trainable even if backbone is frozen in K-only phase)
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

        # K head
        k_modules = [
            m for m in [
                getattr(self, "k_spec_proj", None),
                getattr(self, "k_fuse", None),
                getattr(self, "k_mlp", None),
                getattr(self, "k_direct_mlp", None),
            ] if m is not None
        ]

        _set(backbone_modules, freeze_backbone)
        # Keep factor modules tied to k freeze flag (trainable in K-only phase)
        _set(factor_modules, freeze_k)
        _set(aux_modules, freeze_aux)
        _set(k_modules, freeze_k)
        # k_logit_scale is a parameter tensor (not in a module list)
        if hasattr(self, "k_logit_scale"):
            self.k_logit_scale.requires_grad = not freeze_k
    
    def _hermitize_trace_norm(self, R):
        """Hermitize and trace-normalize covariance matrix"""
        R = 0.5*(R + R.conj().transpose(-2, -1))
        tr = torch.diagonal(R, dim1=-2, dim2=-1).real.sum(-1).clamp_min(1e-9)
        return R / tr.view(-1,1,1)
    
    def _shrink_alpha(self, snr_db, base=None):
        """SNR-aware shrinkage coefficient"""
        base = float(getattr(mdl_cfg, "SHRINK_BASE_ALPHA", 1e-3)) if base is None else float(base)
        alpha = base * (10.0 ** (-snr_db / 20.0))
        return float(max(1e-4, min(alpha, 5e-2)))
    
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
        feats = F.gelu(self.fusion(torch.cat([x, H_feat, c_feat], dim=1)))  # [B, D]

        # --- factor heads (simplified - disable PSD parameterization for AMP compatibility) ---
        cf_ang = self.cov_fact_angle(feats)                              # [B, N*K*2]
        cf_rng = self.cov_fact_range(feats)                              # [B, N*K*2]
        
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

        # --- Enhanced K-head with stable spectral features ---
        # =========================
        # Measurement-domain K features (MxM) — more statistically stable at small L
        # =========================
        # y: [B,L,M,2] RI → complex [B,L,M]
        y_c = torch.complex(y[..., 0].float(), y[..., 1].float())  # [B, L, M]
        # Ryy = (1/L) y^H y  (M×M)
        # NOTE: y_c is [B,L,M] so y^H is [B,M,L]
        # Use actual snapshot count from the batch (safer if L changes later).
        L_eff = float(y_c.shape[1])
        Ryy = (y_c.conj().transpose(-2, -1) @ y_c) / max(L_eff, 1.0)  # [B, M, M]
        Ryy = 0.5 * (Ryy + Ryy.conj().transpose(-2, -1))

        # =========================
        # RIS-domain K features (NxN) — kept as auxiliary, but NOT used for MDL decisions at L=16
        # =========================
        def _vec2c_k(v):
            v = v.float()
            xr, xi = v[:, ::2], v[:, 1::2]
            return torch.complex(xr.view(-1, cfg.N, cfg.K_MAX), xi.view(-1, cfg.N, cfg.K_MAX))

        A_for_k = _vec2c_k(cf_ang)  # [B, N, K]
        R_pred_k = A_for_k @ A_for_k.conj().transpose(-2, -1)  # [B, N, N]
        
        # Enhanced eigenvalue features for K-head (stable vectorized approach)
        with torch.amp.autocast('cuda', enabled=False):
            # --- eigenfeatures from MEASUREMENT DOMAIN (M×M) ---
            evals_yy = torch.linalg.eigvalsh(Ryy.detach()).real  # ascending
            evals_yy = torch.flip(evals_yy, dims=[-1])           # descending [B,M]
            # Pad to length K_MAX+1 indexing convenience (M is 16 so this is fine)
            gaps_yy = evals_yy[:, :cfg.K_MAX] - evals_yy[:, 1:cfg.K_MAX+1]
            csum_yy = torch.cumsum(evals_yy, dim=-1)
            total_yy = csum_yy[:, -1].unsqueeze(-1).clamp_min(1e-9)
            ratios_yy = csum_yy[:, :cfg.K_MAX] / total_yy
            log_evals_yy = torch.log(torch.clamp(evals_yy, min=1e-12))
            log_slopes_yy = log_evals_yy[:, :cfg.K_MAX] - log_evals_yy[:, 1:cfg.K_MAX+1]

            # MDL scores on M×M covariance (more stable at small L)
            Mdim = evals_yy.shape[-1]
            T = int(cfg.L)
            mdl_yy = torch.zeros(evals_yy.shape[0], cfg.K_MAX, device=evals_yy.device)
            logT = torch.log(torch.tensor(float(max(T, 2)), device=evals_yy.device))
            for k in range(1, cfg.K_MAX + 1):
                k_idx = k - 1
                noise = evals_yy[:, k:].clamp_min(1e-12)
                gm = torch.exp(torch.mean(torch.log(noise), dim=-1))
                am = torch.mean(noise, dim=-1)
                Lk = (T * (Mdim - k)) * torch.log(am / gm + 1e-12)
                mdl_yy[:, k_idx] = Lk + 0.5 * k * (2 * Mdim - k) * logT

            k_mdl_idx_yy = torch.argmin(mdl_yy, dim=-1)  # 0-indexed
            k_mdl_onehot_yy = F.one_hot(k_mdl_idx_yy, num_classes=cfg.K_MAX).float()

            # Feature vector: measurement-domain spectrum stats + mdl info
            k_feats = torch.cat(
                [gaps_yy, ratios_yy, log_slopes_yy, mdl_yy, k_mdl_onehot_yy, evals_yy[:, :1], evals_yy[:, -1:]],
                dim=-1
            )
            
            # PHASE 1 FIX: Removed per-sample normalization (was killing gradients when std≈0)
            # The k_mlp has LayerNorm internally, so this normalization is redundant AND harmful.
        
        # SOTA FIX: Ensemble spectral path with direct path for robust K estimation
        # Spectral path: eigenvalue-based features (good when R̂ is clean)
        # Direct path: from main feature vector (works even when eigvals are messy)
        k_logits_spectral = self.k_mlp(k_feats)
        k_logits_direct = self.k_direct_mlp(feats_enhanced)

        # Ordinal K logits (recommended decoding)
        k_ord_logits_spectral = self.k_ord_mlp(k_feats)
        k_ord_logits_direct = self.k_direct_ord_mlp(feats_enhanced)
        
        # Ensemble: average both paths (could also be learned weights)
        k_logits = (0.5 * k_logits_spectral + 0.5 * k_logits_direct) * self.k_logit_scale
        k_ord_logits = (0.5 * k_ord_logits_spectral + 0.5 * k_ord_logits_direct) * self.k_logit_scale

        return {
            "cov_fact_angle": cf_ang,
            "cov_fact_range": cf_rng,
            "phi_theta_r":    aux_ptr,
            "phi_soft":       phi_soft,
            "theta_soft":     theta_soft,
            "k_logits":       k_logits,
            "k_ord_logits":   k_ord_logits,
        }
