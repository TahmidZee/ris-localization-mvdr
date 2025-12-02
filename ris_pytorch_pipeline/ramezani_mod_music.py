# -*- coding: utf-8 -*-
"""
Ramezani et al. (2024) Modified-MUSIC for RIS-assisted near-field localization.
Implements:
  1) LS back-projection A~s(t) = (H Φ_t)^† y(t)  (paper Sec. III; "Setting G=HΦ")
  2) Anti-diagonal vector \bar{y} from R = E[A~s A~s^H] (eqs (6), (14))
  3) Spatial smoothing with overlapping sub-RISes DH x DV -> R_q (Fig. 1, Sec. III)
  4) 2D MUSIC over (phi, theta) using b(α,β) with α,β from eq after (6)
     α = 2π/λ * d_H * sinφ cosθ,   β = 2π/λ * d_V * sinθ
  5) 1D MUSIC over distance r for each detected AoA (eq. (24))
Author: pipeline add-on (PyTorch/Numpy)
"""

import numpy as np
from numpy.linalg import pinv, eigh
from typing import Tuple, Optional, List
import warnings

# -------------------------
# Geometry / steering utils
# -------------------------
def nearfield_vec(N_H:int, N_V:int, d_H:float, d_V:float, k0:float,
                  phi:float, theta:float, r:float) -> np.ndarray:
    """Near-field steering vector a(phi,theta,r) over an N_H x N_V RIS (flatten row-major)."""
    # centered coordinates
    h_idx = (np.arange(N_H) - (N_H-1)/2.0) * d_H
    v_idx = (np.arange(N_V) - (N_V-1)/2.0) * d_V
    H, V = np.meshgrid(h_idx, v_idx, indexing="xy")  # (N_V, N_H)
    # spherical wave approximation (2nd-order Taylor as in near-field literature)
    # distance to each element
    sin_phi, cos_theta, sin_theta = np.sin(phi), np.cos(theta), np.sin(theta)
    # Effective distance offset per element
    dist = r - H*sin_phi*cos_theta - V*sin_theta + (H**2 + V**2)/(2.0*max(r, 1e-9))
    a = np.exp(1j*k0*(r - dist))  # (N_V, N_H)
    return (a.reshape(-1) / np.sqrt(N_H*N_V)).astype(np.complex64)

def alpha_beta(phi:float, theta:float, d_H:float, d_V:float, lamb:float) -> Tuple[float,float]:
    """α, β from the paper (right below (6)): α=2π/λ d_H sinφ cosθ, β=2π/λ d_V sinθ."""
    alpha = 2*np.pi/lamb * d_H * np.sin(phi) * np.cos(theta)
    beta  = 2*np.pi/lamb * d_V * np.sin(theta)
    return alpha, beta

# ------------------------------------------
# Anti-diagonal, spatial smoothing, spectrum
# ------------------------------------------
def _flat_to_centered(n:int, N_H:int, N_V:int) -> Tuple[int,int]:
    nH = n % N_H
    nV = n // N_H
    nHc = int(nH - (N_H-1)//2)
    nVc = int(nV - (N_V-1)//2)
    return nHc, nVc

def _centered_to_flat(nHc:int, nVc:int, N_H:int, N_V:int) -> int:
    nH = nHc + (N_H-1)//2
    nV = nVc + (N_V-1)//2
    return int(nV*N_H + nH)

def anti_diagonal_vector(R:np.ndarray, N_H:int, N_V:int) -> np.ndarray:
    """
    Build \bar{y} by collecting anti-diagonal entries R[n, mirror(n)]
    where mirror(n) is the element at (-nH, -nV) in centered coordinates.
    Result length N = N_H*N_V ordered row-major on centered grid.
    """
    N = N_H*N_V
    ybar = np.empty(N, dtype=np.complex64)
    for n in range(N):
        nHc, nVc = _flat_to_centered(n, N_H, N_V)
        m = _centered_to_flat(-nHc, -nVc, N_H, N_V)
        ybar[n] = R[n, m]
    return ybar

def spatial_smoothing_ybar(ybar:np.ndarray, N_H:int, N_V:int, DH:int, DV:int) -> np.ndarray:
    """
    Split the RIS into overlapping sub-RISes of size DH x DV (Fig. 1).
    For each sub-RIS, extract the corresponding block from ybar (reshaped N_V x N_H),
    flatten to q_j (length D=DH*DV), and average R_q = (1/J) sum q_j q_j^H.
    """
    Y2 = ybar.reshape(N_V, N_H)  # row-major (v,h)
    JH = N_H - DH + 1
    JV = N_V - DV + 1
    if JH <= 0 or JV <= 0:
        raise ValueError(f"Invalid sub-RIS size DHxDV={DH}x{DV} for N_HxN_V={N_H}x{N_V}")
    D = DH*DV
    Rq = np.zeros((D, D), dtype=np.complex64)
    J = 0
    for v0 in range(JV):
        for h0 in range(JH):
            block = Y2[v0:v0+DV, h0:h0+DH]  # (DV, DH)
            q = block.reshape(-1)  # (D,)
            Rq += np.outer(q, q.conj())
            J += 1
    return (Rq / max(J, 1)).astype(np.complex64)

def steering_b(alpha:float, beta:float, DH:int, DV:int) -> np.ndarray:
    """
    b(α,β) over a DH x DV sub-RIS for the anti-diagonal domain.
    Entries are exp(j*2*(i*α + j*β)) with centered indices i,j (factor 2 per paper’s derivation).
    Returned as vector of length D=DH*DV (flatten row-major).
    """
    h_idx = (np.arange(DH) - (DH-1)/2.0)
    v_idx = (np.arange(DV) - (DV-1)/2.0)
    H, V = np.meshgrid(h_idx, v_idx, indexing="xy")
    B = np.exp(1j * 2.0 * (H*alpha + V*beta))
    return B.reshape(-1).astype(np.complex64)

def music_spectrum_angles(Rq:np.ndarray, K:int, grid_phi:np.ndarray, grid_theta:np.ndarray,
                          DH:int, DV:int, d_H:float, d_V:float, lamb:float) -> np.ndarray:
    """
    Compute f(φ,θ) = 1 / (b(α,β)^H U_n U_n^H b(α,β)) on a coarse grid.
    """
    # Eigendecomp of Rq
    evals, evecs = eigh(Rq)  # ascending
    # noise subspace of size D-K
    D = Rq.shape[0]
    if K is None or K <= 0 or K >= D:
        # Blind-K via MDL on Rq if needed (simple fallback)
        # Use last half as "noise" if no K provided; avoids crash.
        K = max(1, min(D-1, D//4))
    Un = evecs[:, :D-K]  # smallest eigenvectors
    P = np.zeros((len(grid_phi), len(grid_theta)), dtype=np.float32)
    for i, phi in enumerate(grid_phi):
        for j, theta in enumerate(grid_theta):
            a, b = alpha_beta(phi, theta, d_H, d_V, lamb)
            bv = steering_b(a, b, DH, DV).astype(np.complex64)
            den = np.real(np.conj(bv) @ (Un @ (Un.conj().T @ bv)) ) + 1e-9
            P[i, j] = 1.0 / den
    return P

def pick_topK_peaks_2d(P:np.ndarray, K:int, sep:int=2) -> List[Tuple[int,int]]:
    """Greedy non-maximum-suppression on 2D grid."""
    H, W = P.shape
    flat_idx = np.argpartition(P.ravel(), -max(K*4, K))[-max(K*4, K):]
    cand = sorted([(int(idx//W), int(idx%W), P.ravel()[idx]) for idx in flat_idx],
                  key=lambda x: -x[2])
    picked = []
    for (i, j, _) in cand:
        good = True
        for (pi, pj) in picked:
            if abs(pi-i) <= sep and abs(pj-j) <= sep:
                good = False; break
        if good:
            picked.append((i, j))
        if len(picked) == K:
            break
    return picked

# --------------------------
# LS back-projection (Sec.III)
# --------------------------
def ls_cov_from_y(H:np.ndarray, codes:np.ndarray, y:np.ndarray) -> np.ndarray:
    """
    Build R = (1/L) sum_t x_t x_t^H with x_t = (H diag(c_t))^† y_t.
    Shapes:
      H: (M,N) complex
      codes: (L,N) complex  (row l is c_l)
      y: (L,M) complex      (row l is y_l)
    """
    L, M = y.shape
    N = H.shape[1]
    R = np.zeros((N, N), dtype=np.complex64)
    for t in range(L):
        Ct = np.diag(codes[t].astype(np.complex64))
        Gt = H @ Ct  # (M,N)
        # Moore-Penrose (small M,N, L=4) -> cheap & stable
        Xt = pinv(Gt) @ y[t].astype(np.complex64)  # (N,)
        R += np.outer(Xt, Xt.conj())
    return (R / max(L,1)).astype(np.complex64)

# --------------------------
# Range MUSIC (eq. 24)
# --------------------------
def distance_sweep(R:np.ndarray, K:int, phi_hat:np.ndarray, theta_hat:np.ndarray,
                   N_H:int, N_V:int, d_H:float, d_V:float, k0:float,
                   r_min:float, r_max:float, n_grid:int) -> np.ndarray:
    """For each (phi,theta), 1D MUSIC over r; noise subspace from R."""
    # eig of R
    evals, evecs = eigh(R)  # ascending
    N = R.shape[0]
    Un = evecs[:, :max(N-K,1)]
    r_grid = np.linspace(r_min, r_max, n_grid)
    out = []
    for (phi, theta) in zip(phi_hat, theta_hat):
        best_r, best_val = r_grid[0], -np.inf
        for r in r_grid:
            a = nearfield_vec(N_H, N_V, d_H, d_V, k0, phi, theta, float(r))
            den = np.real(np.conj(a) @ (Un @ (Un.conj().T @ a))) + 1e-9
            val = 1.0 / den
            if val > best_val:
                best_val, best_r = val, float(r)
        out.append(best_r)
    return np.asarray(out, dtype=np.float32)

# --------------------------
# Public entry point
# --------------------------
def ramezani_modified_music(y:Optional[np.ndarray],
                            H:Optional[np.ndarray],
                            codes:Optional[np.ndarray],
                            R:Optional[np.ndarray],
                            *,
                            N_H:int, N_V:int,
                            d_H:float, d_V:float,
                            lamb:float, k0:float,
                            K:Optional[int],
                            DH:int=3, DV:int=3,
                            grid_phi:Optional[np.ndarray]=None,
                            grid_theta:Optional[np.ndarray]=None,
                            range_grid:int=61,
                            r_min:float=1.0, r_max:float=100.0
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    If R is None, we reconstruct R via LS back-projection from (y,H,codes).
    Returns (phi_hat [K], theta_hat [K], r_hat [K]) in radians & meters.
    """
    if R is None:
        if y is None or H is None or codes is None:
            raise ValueError("Provide either R (NxN) or (y,H,codes).")
        R = ls_cov_from_y(H, codes, y)  # NxN

    # anti-diagonal vector
    ybar = anti_diagonal_vector(R, N_H, N_V)  # (N,)
    # spatial smoothing over DH x DV
    Rq = spatial_smoothing_ybar(ybar, N_H, N_V, DH, DV)  # (D,D)

    # grids
    # Match main model FOV: φ ±60°, θ ±30°
    if grid_phi is None:   grid_phi   = np.linspace(-np.pi/3.0, np.pi/3.0, 41).astype(np.float32)   # ±60°
    if grid_theta is None: grid_theta = np.linspace(-np.pi/6.0, np.pi/6.0, 41).astype(np.float32)   # ±30°

    # 2D MUSIC over angles
    P = music_spectrum_angles(Rq, K, grid_phi, grid_theta, DH, DV, d_H, d_V, lamb)
    if K is None or K <= 0:
        # crude K from MDL on R (fallback) to size
        evals, _ = eigh(R)
        M = R.shape[0]
        T = 16  # fictitious snapshots; K estimator is not very sensitive here
        ks = np.arange(0, min(8, M-1))
        crit = []
        lam = np.sort(np.clip(np.real(evals), 1e-12, None))[::-1]
        for k in ks:
            noise = lam[k:] if k < len(lam) else lam[-1:]
            gm = np.exp(np.mean(np.log(noise)))
            am = np.mean(noise)
            crit.append(-T*(M-k)*np.log(gm/am) + 0.5*k*(2*M-k)*np.log(T))
        K_eff = int(ks[int(np.argmin(crit))]) if len(ks) else 1
        K = max(1, min(4, K_eff))
    peaks = pick_topK_peaks_2d(P, K, sep=2)
    # map peaks -> angles
    phi_hat = np.array([grid_phi[i] for (i, j) in peaks], dtype=np.float32)
    theta_hat = np.array([grid_theta[j] for (i, j) in peaks], dtype=np.float32)

    # 1D range sweep per (phi,theta)
    r_hat = distance_sweep(R, K, phi_hat, theta_hat, N_H, N_V, d_H, d_V, k0, r_min, r_max, range_grid)
    return phi_hat, theta_hat, r_hat
