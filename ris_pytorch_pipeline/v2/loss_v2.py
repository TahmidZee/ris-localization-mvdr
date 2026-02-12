"""
V2 Covariance Loss — NMSE primary, optional structure terms.

No permutation matching.  No slot diversity.  No direct geometry loss.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from ..physics import nearfield_vec
from .config_v2 import v2_cfg, v2_mdl


def _as_complex(R: torch.Tensor) -> torch.Tensor:
    """Accept complex or (..., 2) real/imag format."""
    if torch.is_complex(R):
        return R
    return R[..., 0].to(torch.float32) + 1j * R[..., 1].to(torch.float32)


class V2CovarianceLoss(nn.Module):
    """
    Primary: NMSE on covariance.
    Optional: subspace alignment (GT-steering based), peak contrast (MVDR).

    Usage
    -----
    loss_fn = V2CovarianceLoss(lam_subspace=0.05, lam_peak=0.02)
    loss, info = loss_fn(R_pred, R_true, K_true=K, ptr_gt=ptr)
    """

    def __init__(
        self,
        lam_cov_main: float | None = None,
        lam_cov_f: float | None = None,
        lam_cov_consistency: float | None = None,
        lam_subspace: float | None = None,
        lam_peak: float | None = None,
    ):
        super().__init__()
        self.lam_cov_main = lam_cov_main if lam_cov_main is not None else float(getattr(v2_mdl, "LAM_COV_MAIN", 1.0))
        self.lam_cov_f = lam_cov_f if lam_cov_f is not None else float(getattr(v2_mdl, "LAM_COV_F", 0.0))
        self.lam_cov_consistency = (
            lam_cov_consistency
            if lam_cov_consistency is not None
            else float(getattr(v2_mdl, "LAM_COV_CONSIST", 0.0))
        )
        self.lam_subspace = lam_subspace if lam_subspace is not None else v2_mdl.LAM_SUBSPACE
        self.lam_peak = lam_peak if lam_peak is not None else v2_mdl.LAM_PEAK

    # ------------------------------------------------------------------ #
    # Core NMSE
    # ------------------------------------------------------------------ #
    def _nmse(self, R_hat: torch.Tensor, R_true: torch.Tensor) -> torch.Tensor:
        """
        NMSE = || R̂ - R_true ||²_F  /  || R_true ||²_F

        Returns: [B] per-sample NMSE.
        """
        H = _as_complex(R_hat)
        T = _as_complex(R_true)
        diff = H - T
        num = (diff.conj() * diff).real.sum(dim=(-2, -1))
        den = (T.conj() * T).real.sum(dim=(-2, -1)).clamp_min(1e-12)
        return num / den

    # ------------------------------------------------------------------ #
    # Reused physics helper
    # ------------------------------------------------------------------ #
    def _build_gt_steering(
        self,
        phi_b: torch.Tensor,
        theta_b: torch.Tensor,
        r_b: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build GT steering matrix A_gt [K, N] using v1 near-field physics.
        """
        rows = []
        for i in range(phi_b.numel()):
            a_np = nearfield_vec(
                v2_cfg,
                float(phi_b[i].item()),
                float(theta_b[i].item()),
                float(r_b[i].item()),
            )
            rows.append(torch.from_numpy(a_np))
        if not rows:
            return torch.empty(0, int(v2_cfg.N), device=device, dtype=dtype)
        return torch.stack(rows, dim=0).to(device=device, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Subspace alignment (reuse v1 idea, simplified)
    # ------------------------------------------------------------------ #
    def _subspace_alignment(
        self,
        R_pred: torch.Tensor,
        K_true: torch.Tensor,
        ptr_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stable subspace alignment using GT steering vectors.

        R_pred  : [B, N, N] complex (predicted covariance)
        K_true  : [B] int (number of true sources)
        ptr_gt  : [B, K_MAX, 3] or [B, 3*K_MAX]  (phi, theta, r in radians/meters)
        """
        R_pred_c = _as_complex(R_pred)
        B, N = R_pred_c.shape[:2]
        device = R_pred_c.device
        dtype = R_pred_c.dtype
        K_MAX = int(v2_cfg.K_MAX)

        # Unpack ptr_gt
        if ptr_gt.dim() == 2 and ptr_gt.shape[1] >= 3 * K_MAX:
            phi = ptr_gt[:, :K_MAX]
            theta = ptr_gt[:, K_MAX : 2 * K_MAX]
            rr = ptr_gt[:, 2 * K_MAX : 3 * K_MAX]
        elif ptr_gt.dim() == 3 and ptr_gt.shape[-1] == 3:
            phi = ptr_gt[:, :, 0]
            theta = ptr_gt[:, :, 1]
            rr = ptr_gt[:, :, 2]
        else:
            return torch.tensor(0.0, device=device)

        # SVD of R_pred for projector
        eps = float(v2_mdl.EPS_PSD)
        eye = torch.eye(N, device=device, dtype=dtype)
        R_sym = 0.5 * (R_pred_c + R_pred_c.conj().transpose(-2, -1)) + eps * eye
        U, _, _ = torch.linalg.svd(R_sym, full_matrices=False)

        losses = []
        for b in range(B):
            K = int(K_true[b].item())
            if K <= 0 or K >= N:
                continue

            # GT steering vectors
            phi_b = phi[b, :K].float()
            theta_b = theta[b, :K].float()
            r_b = rr[b, :K].float().clamp_min(1e-6)

            A_gt = self._build_gt_steering(phi_b, theta_b, r_b, device=device, dtype=dtype)

            # Signal-subspace projector
            U_sig = U[b, :, :K]  # [N, K]
            P_sig = U_sig @ U_sig.conj().T  # [N, N]

            # Residual energy outside signal subspace
            A_gt_T = A_gt.T  # [N, K]
            resid = (eye - P_sig) @ A_gt_T
            num = (resid.real ** 2 + resid.imag ** 2).sum()
            den = (A_gt_T.real ** 2 + A_gt_T.imag ** 2).sum().clamp_min(1e-9)
            losses.append((num / den).real)

        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------ #
    # Peak contrast (MVDR Capon at GT angles)
    # ------------------------------------------------------------------ #
    def _peak_contrast(
        self,
        R_pred: torch.Tensor,
        K_true: torch.Tensor,
        ptr_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage MVDR pseudo-spectrum peaks at GT source angles.
        Loss = 1 - mean( a(φ_gt)ᴴ R⁻¹ a(φ_gt) normalised ).
        """
        R_pred_c = _as_complex(R_pred)
        B, N = R_pred_c.shape[:2]
        device = R_pred_c.device
        dtype = R_pred_c.dtype
        K_MAX = int(v2_cfg.K_MAX)

        if ptr_gt.dim() == 2 and ptr_gt.shape[1] >= 3 * K_MAX:
            phi = ptr_gt[:, :K_MAX]
            theta = ptr_gt[:, K_MAX : 2 * K_MAX]
            rr = ptr_gt[:, 2 * K_MAX : 3 * K_MAX]
        elif ptr_gt.dim() == 3 and ptr_gt.shape[-1] == 3:
            phi = ptr_gt[:, :, 0]
            theta = ptr_gt[:, :, 1]
            rr = ptr_gt[:, :, 2]
        else:
            return torch.tensor(0.0, device=device)

        eps = 1e-3
        eye = torch.eye(N, device=device, dtype=dtype)
        R_reg = R_pred_c + eps * eye

        try:
            R_inv = torch.linalg.inv(R_reg)
        except Exception:
            return torch.tensor(0.0, device=device)

        total, count = 0.0, 0
        for b in range(B):
            K = int(K_true[b].item())
            if K <= 0:
                continue
            for k in range(K):
                phi_k = phi[b, k].float()
                theta_k = theta[b, k].float()
                r_k = rr[b, k].float().clamp_min(1e-6)

                a = self._build_gt_steering(
                    phi_k.unsqueeze(0),
                    theta_k.unsqueeze(0),
                    r_k.unsqueeze(0),
                    device=device,
                    dtype=dtype,
                )[0]

                # Capon: P = 1 / (aᴴ R⁻¹ a)  — we want this to be large
                denom = (a.conj() @ R_inv[b] @ a).real.clamp_min(1e-12)
                total += denom  # higher = worse for source → we minimise this
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)
        # Mean inverse-Capon power (lower = sharper peaks)
        return total / count

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        R_pred: torch.Tensor,
        R_true: torch.Tensor,
        R_f_pred: torch.Tensor | None = None,
        R_f_true: torch.Tensor | None = None,
        R_f_mean_pred: torch.Tensor | None = None,
        K_true: torch.Tensor | None = None,
        ptr_gt: torch.Tensor | None = None,
        snr_db: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute total loss.

        Returns
        -------
        loss : scalar
        info : dict of component values (for logging)
        """
        nmse_main = self._nmse(R_pred, R_true).mean()
        loss = self.lam_cov_main * nmse_main
        info: dict = {
            "nmse": nmse_main.item(),
            "nmse_main": nmse_main.item(),
            "lam_cov_main": float(self.lam_cov_main),
            "lam_cov_f": float(self.lam_cov_f),
            "lam_cov_consistency": float(self.lam_cov_consistency),
        }

        has_rf = (R_f_pred is not None) and (R_f_true is not None) and (R_f_true.numel() > 0)
        if self.lam_cov_f > 0.0 and has_rf:
            nmse_f = self._nmse(R_f_pred, R_f_true).mean()
            loss = loss + self.lam_cov_f * nmse_f
            info["nmse_f"] = nmse_f.item()

        if self.lam_cov_consistency > 0.0 and (R_f_pred is not None):
            if R_f_mean_pred is None:
                R_f_mean_pred = _as_complex(R_f_pred).mean(dim=1)
            nmse_cons = self._nmse(R_f_mean_pred, R_pred).mean()
            loss = loss + self.lam_cov_consistency * nmse_cons
            info["nmse_consistency"] = nmse_cons.item()

        R_struct = R_f_mean_pred if (R_f_mean_pred is not None) else R_pred

        if self.lam_subspace > 0 and K_true is not None and ptr_gt is not None:
            L_sub = self._subspace_alignment(R_struct, K_true, ptr_gt)
            loss = loss + self.lam_subspace * L_sub
            info["subspace"] = L_sub.item()

        if self.lam_peak > 0 and K_true is not None and ptr_gt is not None:
            L_peak = self._peak_contrast(R_struct, K_true, ptr_gt)
            loss = loss + self.lam_peak * L_peak
            info["peak"] = L_peak.item()

        info["total"] = loss.item()
        return loss, info
