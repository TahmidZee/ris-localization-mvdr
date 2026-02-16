"""
V2 CovariancePredictor — predicts low-rank covariance factors from raw inputs.

Design:  y-encoder (Conv1D + Transformer) + H/code projections
         → fusion → factor head → A [B, N, R] complex
         → R̂ = A Aᴴ + ε I   (guaranteed PSD)

No angle/range heads.  Geometry comes from MUSIC on R̂.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..covariance_utils import hermitize_torch, trace_norm_torch
from .config_v2 import v2_cfg, v2_mdl


class FreqPool(nn.Module):
    """Attention pooling over subcarrier features [B, F, D] -> [B, D]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model)

    def forward(self, x_f: torch.Tensor, return_attn: bool = False):
        # x_f: [B, F, D]
        keys = self.key_proj(x_f)  # [B, F, D]
        logits = (keys * self.query.view(1, 1, -1)).sum(-1) / math.sqrt(float(keys.shape[-1]))
        attn = torch.softmax(logits, dim=1).unsqueeze(-1)  # [B, F, 1]
        pooled = (attn * x_f).sum(1)  # [B, D]
        if return_attn:
            return pooled, attn.squeeze(-1)  # [B, D], [B, F]
        return pooled


class CovariancePredictor(nn.Module):
    """
    Predict a PSD covariance matrix from received signal + channel/code features.

    Inputs
    ------
    y      : [B, L, (F,) M, 2]  received signal (RI)
    H      : [B, L, M, 2]       direct channel (RI)
    codes  : [B, L, N, 2]       RIS codebook (RI)

    Output dict
    -----------
    R_pred : [B, N, N] complex   predicted covariance (Hermitian, PSD, trace-normalised)
    A_pred : [B, N, R] complex   low-rank factor (for diagnostics)
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _choose_heads(D: int, pref: int = 8) -> int:
        if D % pref == 0 and pref >= 1:
            return pref
        for h in [8, 6, 4, 2, 1]:
            if h <= D and D % h == 0:
                return h
        return 1

    @staticmethod
    def _ri_to_complex_torch(x: torch.Tensor | None) -> torch.Tensor | None:
        """Convert RI tensor (...,2) to complex tensor, pass-through if already complex."""
        if x is None:
            return None
        if torch.is_complex(x):
            return x
        if x.shape[-1] == 2:
            return torch.complex(x[..., 0].float(), x[..., 1].float())
        raise ValueError(f"Expected complex or RI(...,2), got shape={tuple(x.shape)}")

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def __init__(
        self,
        M: int | None = None,
        N: int | None = None,
        L: int | None = None,
        D: int | None = None,
        rank: int | None = None,
        n_heads: int | None = None,
        n_layers: int | None = None,
        ff_dim: int | None = None,
        dropout: float | None = None,
        eps_psd: float | None = None,
    ):
        super().__init__()
        M = M or v2_cfg.M
        N = N or v2_cfg.N
        L = L or v2_cfg.L
        D = D or v2_mdl.D_MODEL
        rank = rank or v2_mdl.FACTOR_RANK
        n_heads = n_heads or v2_mdl.NUM_HEADS
        n_layers = n_layers or v2_mdl.N_LAYERS
        ff_dim = ff_dim or v2_mdl.FF_DIM
        dropout = dropout if dropout is not None else v2_mdl.DROPOUT
        self.eps_psd = eps_psd if eps_psd is not None else v2_mdl.EPS_PSD

        self.M = M
        self.N = N
        self.L = L
        self.D = D
        self.rank = rank
        self.use_tone_factor_head = bool(getattr(v2_mdl, "USE_TONE_FACTOR_HEAD", True))
        self.tap_count = max(1, int(getattr(v2_cfg, "D_TAPS", 8)))

        # ── y path: Conv1D stem + Transformer encoder ──
        self.y_conv1 = nn.Conv1d(M * 2, D // 2, kernel_size=5, padding=2)
        self.y_dw = nn.Conv1d(D // 2, D // 2, kernel_size=3, padding=1, groups=D // 2)
        self.y_conv2 = nn.Conv1d(D // 2, D, kernel_size=1)

        nheads = self._choose_heads(D, n_heads)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=nheads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.freq_pool = FreqPool(D)

        # ── H path ──
        self.H_proj = nn.Linear(L * M * 2, D // 2)
        # Tap encoder uses pooled per-tap descriptors [M + N + power] rather than
        # flattening [P, M, N, 2], which is prohibitively large at 64x16x256.
        self.H_tap_feat_dim = M + N + 1
        self.H_tap_proj = nn.Linear(self.tap_count * self.H_tap_feat_dim, D // 2)
        self.Hf_token_proj = nn.Linear(M + N + 1, D)
        self.Hop_global_proj = nn.Linear(D, D // 2)
        self.freq_op_fuse = nn.Linear(2 * D, D)

        # ── codes path ──
        self.codes_conv = nn.Conv1d(N * 2, D // 2, kernel_size=5, padding=2)

        # ── fusion ──
        self.fusion = nn.Linear(D + D // 2 + D // 2, D)

        # ── factor head: outputs [B, N * rank * 2] (real/imag interleaved) ──
        self.factor_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, N * rank * 2),
        )
        self.tone_factor_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, N * rank * 2),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        """Conservative init for factor head to keep initial R̂ ≈ small."""
        def _init_head(head: nn.Sequential):
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            last_linear = list(head.modules())[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.normal_(last_linear.weight, std=0.01)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)

        # Last layer small std keeps initial R close to eps*I and stabilizes early training.
        _init_head(self.factor_head)
        _init_head(self.tone_factor_head)

        for layer in [self.H_tap_proj, self.Hf_token_proj, self.Hop_global_proj, self.freq_op_fuse]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    def _factor_to_cov(self, factor_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw factor vector to PSD covariance.

        factor_vec: [B, N * rank * 2]  (real/imag interleaved)
        Returns:
            R_pred: [B, N, N] complex  (Hermitian, PSD, trace-normalised)
            A:      [B, N, rank] complex
        """
        B = factor_vec.shape[0]
        # Full precision for complex construction
        fv = factor_vec.float()
        fv = fv.view(B, 2, self.N, self.rank)
        A = torch.complex(fv[:, 0], fv[:, 1])  # [B, N, rank] complex64

        # R̂ = A Aᴴ + ε I  (guaranteed PSD)
        R = A @ A.conj().transpose(-2, -1)  # [B, N, N]
        R = R + self.eps_psd * torch.eye(self.N, device=R.device, dtype=R.dtype).unsqueeze(0)

        # Reuse v1 covariance helpers for canonical post-processing.
        R = hermitize_torch(R)
        R = trace_norm_torch(R, target_trace=float(self.N))

        return R, A

    # ------------------------------------------------------------------
    def _factor_to_cov_f(self, factor_vec_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert per-tone factor vectors to PSD per-tone covariance.

        factor_vec_f: [B, F, N*rank*2]
        Returns:
            R_f_pred: [B, F, N, N] complex
            A_f_pred: [B, F, N, rank] complex
        """
        B, F_sub = factor_vec_f.shape[:2]
        fv = factor_vec_f.float().view(B, F_sub, 2, self.N, self.rank)
        A_f = torch.complex(fv[:, :, 0], fv[:, :, 1])  # [B, F, N, rank]

        R_f = A_f @ A_f.conj().transpose(-2, -1)  # [B, F, N, N]
        eye = torch.eye(self.N, device=R_f.device, dtype=R_f.dtype).view(1, 1, self.N, self.N)
        R_f = R_f + self.eps_psd * eye

        # Reuse v1 helpers by flattening B*F.
        R_f_flat = R_f.reshape(B * F_sub, self.N, self.N)
        R_f_flat = hermitize_torch(R_f_flat)
        R_f_flat = trace_norm_torch(R_f_flat, target_trace=float(self.N))
        R_f = R_f_flat.reshape(B, F_sub, self.N, self.N)
        return R_f, A_f

    # ------------------------------------------------------------------
    def _encode_h(self, H: torch.Tensor) -> torch.Tensor:
        """
        Encode H feature tensor.

        Supports:
          - [B, L, M, 2]
          - [B, L, F, M, 2] (frequency pooled over F)
        """
        if H.dim() == 5:
            H = H.mean(dim=2)  # [B, L, M, 2]
        if H.dim() != 4:
            raise ValueError(f"Unsupported H shape {tuple(H.shape)}")
        B = H.shape[0]
        return F.gelu(self.H_proj(H.reshape(B, -1)))  # [B, D/2]

    # ------------------------------------------------------------------
    def _encode_h_taps(self, H_taps: dict[str, torch.Tensor] | None) -> torch.Tensor | None:
        """
        Encode tap-domain channel features when available.
        Expected primary key: H_taps_ri [B, P, M, N, 2].
        """
        if not H_taps:
            return None
        h_tap_key = str(getattr(v2_cfg, "WIDEBAND_H_TAPS_KEY", "H_taps_ri"))
        h_tap_ri = H_taps.get(h_tap_key, None)
        if h_tap_ri is None:
            h_tap_ri = H_taps.get("H_taps_ri", None)
        if h_tap_ri is None:
            return None

        if torch.is_complex(h_tap_ri):
            h_tap_ri = torch.view_as_real(h_tap_ri)
        if h_tap_ri.dim() != 5 or h_tap_ri.shape[-1] != 2:
            raise ValueError(f"Expected H_taps_ri shape [B,P,M,N,2], got {tuple(h_tap_ri.shape)}")
        if h_tap_ri.shape[2] != self.M or h_tap_ri.shape[3] != self.N:
            raise ValueError(
                f"Expected H_taps_ri spatial dims [M,N]=[{self.M},{self.N}], got {tuple(h_tap_ri.shape[2:4])}"
            )

        B, P = h_tap_ri.shape[0], h_tap_ri.shape[1]
        if P < self.tap_count:
            pad = torch.zeros(
                (B, self.tap_count - P, self.M, self.N, 2),
                dtype=h_tap_ri.dtype,
                device=h_tap_ri.device,
            )
            h_tap_ri = torch.cat([h_tap_ri, pad], dim=1)
        elif P > self.tap_count:
            h_tap_ri = h_tap_ri[:, : self.tap_count]

        # Compact per-tap descriptor: BS-profile + RIS-profile + tap power.
        mag = torch.sqrt((h_tap_ri[..., 0] ** 2 + h_tap_ri[..., 1] ** 2).clamp_min(1e-12))  # [B,P,M,N]
        bs_profile = mag.mean(dim=-1)  # [B,P,M]
        ris_profile = mag.mean(dim=-2)  # [B,P,N]
        tap_power = (mag * mag).mean(dim=(-2, -1), keepdim=True)  # [B,P,1]
        tap_feat = torch.cat([bs_profile, ris_profile, tap_power], dim=-1)  # [B,P,M+N+1]
        feat = torch.log1p(tap_feat.float()).reshape(B, -1)
        return F.gelu(self.H_tap_proj(feat))  # [B, D/2]

    # ------------------------------------------------------------------
    def _build_hf_from_taps(
        self,
        H_taps: dict[str, torch.Tensor] | None,
        n_freq: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """
        Build per-tone operator H(f) from tap-domain tensors.
        Expected H_taps_ri shape: [B, P, M, N, 2] (or complex [B, P, M, N]).
        """
        if not H_taps:
            return None
        h_taps_ri = H_taps.get(str(getattr(v2_cfg, "WIDEBAND_H_TAPS_KEY", "H_taps_ri")), None)
        if h_taps_ri is None:
            h_taps_ri = H_taps.get("H_taps_ri", None)
        if h_taps_ri is None:
            return None

        Htap = self._ri_to_complex_torch(h_taps_ri)
        if Htap.dim() != 4:
            raise ValueError(f"Expected H_taps shape [B,P,M,N], got {tuple(Htap.shape)}")
        B, P, _, _ = Htap.shape
        Htap = Htap.to(device=device, dtype=torch.complex64)

        path_mask = H_taps.get("path_mask", None)
        if path_mask is not None:
            if path_mask.dim() == 1:
                path_mask = path_mask.unsqueeze(0)
            path_mask = path_mask.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            Htap = Htap * path_mask.unsqueeze(-1).unsqueeze(-1)

        taus = H_taps.get("taus_s", None)
        if taus is not None:
            if taus.dim() == 1:
                taus = taus.unsqueeze(0)
            taus = taus.to(device=device, dtype=torch.float32)
            bw_hz = float(getattr(v2_cfg, "BW_HZ", float(getattr(v2_cfg, "OFDM_ACTIVE_SC", 1596)) * float(getattr(v2_cfg, "SUBCARRIER_SPACING_HZ", 30e3))))
            freq_offsets = torch.linspace(-0.5 * bw_hz, 0.5 * bw_hz, n_freq, device=device, dtype=torch.float32)
            phase = torch.exp(-1j * (2.0 * math.pi) * taus.unsqueeze(-1) * freq_offsets.view(1, 1, n_freq))  # [B,P,F]
            H_f = torch.einsum("bpmn,bpf->bfmn", Htap, phase)
        else:
            # Fallback: DFT-style synthesis when explicit delays are unavailable.
            tap_idx = torch.arange(P, device=device, dtype=torch.float32).view(P, 1)
            tone_idx = torch.arange(n_freq, device=device, dtype=torch.float32).view(1, n_freq)
            phase = torch.exp(-1j * (2.0 * math.pi) * tap_idx * tone_idx / float(max(n_freq, 1)))  # [P,F]
            H_f = torch.einsum("bpmn,pf->bfmn", Htap, phase)
        return H_f  # [B,F,M,N]

    # ------------------------------------------------------------------
    def _encode_hf_tokens(self, H_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode per-tone operator tokens.
        Input: H_f [B, F, M, N] complex
        Returns:
          tone_tokens [B, F, D]
          global_token [B, D/2]
        """
        mag = torch.abs(H_f)  # [B,F,M,N]
        bs_profile = mag.mean(dim=-1)  # [B,F,M]
        ris_profile = mag.mean(dim=-2)  # [B,F,N]
        power = (mag * mag).mean(dim=(-2, -1)).unsqueeze(-1)  # [B,F,1]
        token_raw = torch.cat([bs_profile, ris_profile, power], dim=-1)  # [B,F,M+N+1]
        token_raw = torch.log1p(token_raw.float())

        tone_tokens = F.gelu(self.Hf_token_proj(token_raw))  # [B,F,D]
        global_token = F.gelu(self.Hop_global_proj(tone_tokens.mean(dim=1)))  # [B,D/2]
        return tone_tokens, global_token

    # ------------------------------------------------------------------
    def _encode_y_single(self, y_single: torch.Tensor) -> torch.Tensor:
        """
        Encode narrowband y input.

        y_single: [B, L, M, 2] -> [B, D]
        """
        bsz, snapshots = y_single.shape[0], y_single.shape[1]
        y_flat = y_single.reshape(bsz, snapshots, self.M * 2).permute(0, 2, 1)  # [B, 2M, L]
        x = F.gelu(self.y_conv1(y_flat))
        x = self.y_dw(x)
        x = F.gelu(self.y_conv2(x))  # [B, D, L]
        x = x.permute(0, 2, 1)  # [B, L, D]
        return self.transformer(x).mean(1)  # [B, D]

    # ------------------------------------------------------------------
    def _encode_y(self, y: torch.Tensor):
        """
        Encode y for both narrowband and wideband paths.

        Supported:
          - [B, L, M, 2]
          - [B, L, F, M, 2] (frequency pooled)

        Returns:
          x_global: [B, D]
          x_f: [B, F, D] or None
          f_attn: [B, F] or None
        """
        if y.dim() == 4:
            x = self._encode_y_single(y)
            return x, None, None
        if y.dim() == 5:
            bsz, snapshots, n_freq, n_ant, _ = y.shape
            if n_ant != self.M:
                raise ValueError(f"Expected M={self.M}, got y.shape[-2]={n_ant}")
            # Flatten (B, F) and reuse narrowband encoder, then pool over F.
            y_bf = y.permute(0, 2, 1, 3, 4).reshape(bsz * n_freq, snapshots, n_ant, 2)
            x_f = self._encode_y_single(y_bf).reshape(bsz, n_freq, self.D)  # [B, F, D]
            x_global, f_attn = self.freq_pool(x_f, return_attn=True)
            return x_global, x_f, f_attn
        raise ValueError(f"Unsupported y shape {tuple(y.shape)}; expected [B,L,M,2] or [B,L,F,M,2]")

    # ------------------------------------------------------------------
    def forward(
        self,
        y: torch.Tensor,
        H: torch.Tensor,
        codes: torch.Tensor,
        snr_db: torch.Tensor | None = None,
        R_samp: torch.Tensor | None = None,
        H_taps: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        B = y.shape[0]

        # ── y features ──
        x_global, x_f, f_attn = self._encode_y(y)  # [B, D], optional per-tone features

        H_f = None
        hop_applied = False
        if (
            x_f is not None
            and bool(getattr(v2_mdl, "USE_OPERATOR_TAP_CONDITIONING", True))
            and H_taps is not None
        ):
            try:
                H_f = self._build_hf_from_taps(H_taps, n_freq=x_f.shape[1], device=x_f.device)
            except Exception:
                H_f = None
            if H_f is not None:
                h_tokens, h_global = self._encode_hf_tokens(H_f)
                x_f = F.gelu(self.freq_op_fuse(torch.cat([x_f, h_tokens], dim=-1)))  # [B,F,D]
                x_global, f_attn = self.freq_pool(x_f, return_attn=True)
                H_feat = h_global
                hop_applied = True
            else:
                H_tap_feat = self._encode_h_taps(H_taps)
                H_feat = H_tap_feat if H_tap_feat is not None else self._encode_h(H)
        else:
            H_tap_feat = self._encode_h_taps(H_taps)
            H_feat = H_tap_feat if H_tap_feat is not None else self._encode_h(H)

        # ── codes features ──
        Lc = codes.shape[1]
        c_seq = codes.reshape(B, Lc, self.N * 2).permute(0, 2, 1)  # [B, 2N, Lc]
        c_feat = F.gelu(self.codes_conv(c_seq)).mean(2)  # [B, D/2]

        # ── fuse ──
        feats = F.gelu(self.fusion(torch.cat([x_global, H_feat, c_feat], dim=1)))  # [B, D]

        # ── global covariance head ──
        factor_vec = self.factor_head(feats)  # [B, N*rank*2]
        R_pred, A_pred = self._factor_to_cov(factor_vec)

        out = {
            "R_pred": R_pred,      # [B, N, N] complex
            "A_pred": A_pred,      # [B, N, rank] complex
            "factor_vec": factor_vec,
            "operator_tap_conditioned": torch.tensor(1.0 if hop_applied else 0.0, device=R_pred.device),
        }

        # ── per-tone covariance head (wideband supervision path) ──
        if self.use_tone_factor_head and (x_f is not None):
            Bf, F_sub, _ = x_f.shape
            factor_vec_f = self.tone_factor_head(x_f.reshape(Bf * F_sub, self.D)).reshape(Bf, F_sub, -1)
            R_f_pred, A_f_pred = self._factor_to_cov_f(factor_vec_f)
            out["R_f_pred"] = R_f_pred                # [B, F, N, N]
            out["A_f_pred"] = A_f_pred                # [B, F, N, rank]
            out["R_f_mean_pred"] = R_f_pred.mean(dim=1)  # [B, N, N]
            out["factor_vec_f"] = factor_vec_f
            if f_attn is not None:
                out["freq_attn"] = f_attn             # [B, F]
        return out
