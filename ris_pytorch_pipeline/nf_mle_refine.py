"""
Near-Field Maximum Likelihood Estimation (NF-MLE) refinement.

This module provides the short NF-MLE polish that runs after joint NF-Newton
to achieve true low-SNR robustness by fitting actual snapshots instead of
relying on noise projector estimates.

Key benefits:
- Reduces edge-pegging at low SNR
- Handles weak eigengap scenarios
- Better range-angle coupling
- Statistically correct objective under Gaussian noise
"""

import numpy as np
import torch
from typing import Tuple, Optional


def steering_nf(phi_rad: float, theta_rad: float, r: float, 
                x_coords: np.ndarray, y_coords: np.ndarray, k: float) -> np.ndarray:
    """
    Near-field steering vector with unit normalization.
    
    Args:
        phi_rad: Azimuth angle in radians
        theta_rad: Elevation angle in radians  
        r: Range in meters
        x_coords: Sensor x-coordinates [M] in meters
        y_coords: Sensor y-coordinates [M] in meters
        k: Wavenumber (2π/λ)
    
    Returns:
        a: Unit-normalized steering vector [M] complex64
    """
    # Near-field phase: k*(x*sin(φ)*cos(θ) + y*sin(θ) - (x²+y²)/(2r))
    sin_phi = np.sin(phi_rad)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    # Phase calculation
    phase = k * (x_coords * sin_phi * cos_theta + y_coords * sin_theta - 
                 (x_coords**2 + y_coords**2) / (2 * r))
    
    # Steering vector
    a = np.exp(1j * phase).astype(np.complex64)
    
    # Unit normalization (critical for amplitude estimation)
    a = a / np.linalg.norm(a)
    
    return a


def steering_nf_jacobians(phi_rad: float, theta_rad: float, r: float,
                         x_coords: np.ndarray, y_coords: np.ndarray, k: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Jacobians of near-field steering vector w.r.t. (φ, θ, r).
    
    Args:
        phi_rad: Azimuth angle in radians
        theta_rad: Elevation angle in radians
        r: Range in meters
        x_coords: Sensor x-coordinates [M] in meters
        y_coords: Sensor y-coordinates [M] in meters
        k: Wavenumber (2π/λ)
    
    Returns:
        da_dphi: Jacobian w.r.t. φ [M] complex64
        da_dtheta: Jacobian w.r.t. θ [M] complex64  
        da_dr: Jacobian w.r.t. r [M] complex64
    """
    # Get steering vector
    a = steering_nf(phi_rad, theta_rad, r, x_coords, y_coords, k)
    
    # Phase derivatives
    sin_phi = np.sin(phi_rad)
    cos_phi = np.cos(phi_rad)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    # ∂phase/∂φ = k * x * cos(φ) * cos(θ)
    dphase_dphi = k * x_coords * cos_phi * cos_theta
    
    # ∂phase/∂θ = k * (x * sin(φ) * (-sin(θ)) + y * cos(θ))
    dphase_dtheta = k * (x_coords * sin_phi * (-sin_theta) + y_coords * cos_theta)
    
    # ∂phase/∂r = k * (x² + y²) / (2r²)
    dphase_dr = k * (x_coords**2 + y_coords**2) / (2 * r**2)
    
    # Jacobians: ∂a/∂q = j * (∂phase/∂q) ⊙ a
    da_dphi = 1j * dphase_dphi * a
    da_dtheta = 1j * dphase_dtheta * a
    da_dr = 1j * dphase_dr * a
    
    return da_dphi, da_dtheta, da_dr


def negloglike(Y: np.ndarray, A: np.ndarray, lambda_reg: float = 1e-6) -> float:
    """
    Negative log-likelihood under white Gaussian noise.
    
    Args:
        Y: Snapshots [M, L] complex64
        A: Steering matrix [M, K] complex64
        lambda_reg: Tikhonov regularization parameter
    
    Returns:
        Negative log-likelihood (lower is better)
    """
    M, L = Y.shape
    K = A.shape[1]
    
    # Solve amplitudes with Tikhonov regularization: Γ̂ = (A^H A + λI)^(-1) A^H Y
    AHA = A.conj().T @ A  # [K, K]
    AHA_reg = AHA + lambda_reg * np.eye(K, dtype=AHA.dtype)
    
    try:
        Gamma = np.linalg.solve(AHA_reg, A.conj().T @ Y)  # [K, L]
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        Gamma = np.linalg.pinv(A) @ Y  # [K, L]
    
    # Compute residual: E = Y - A @ Γ
    E = Y - A @ Gamma  # [M, L]
    
    # Negative log-likelihood: ||E||_F^2 (proportional to -log L under white Gaussian noise)
    nll = np.linalg.norm(E, 'fro')**2
    
    return nll


def refine_nf_mle(Y: np.ndarray, phi0: float, theta0: float, r0: float,
                  x_coords: np.ndarray, y_coords: np.ndarray, k: float,
                  iters: int = 3, step_size: float = 0.7, 
                  r_min: float = 0.4, r_max: float = 12.0,
                  phi_min: float = -np.pi/3, phi_max: float = np.pi/3,
                  theta_min: float = -np.pi/6, theta_max: float = np.pi/6) -> Tuple[float, float, float]:
    """
    Short NF-MLE polish: fit actual snapshots by re-estimating amplitudes
    and optimizing (φ, θ, r) to minimize residual energy.
    
    Args:
        Y: Snapshots [M, L] complex64
        phi0, theta0, r0: Initial angles (rad) and range (m)
        x_coords, y_coords: Sensor coordinates [M] in meters
        k: Wavenumber (2π/λ)
        iters: Number of MLE iterations (default: 3)
        step_size: Step size scaling (default: 0.7)
        r_min, r_max: Range bounds in meters
        phi_min, phi_max: Azimuth bounds in radians
        theta_min, theta_max: Elevation bounds in radians
    
    Returns:
        phi_refined, theta_refined, r_refined: Refined parameters
    """
    M, L = Y.shape
    phi, theta, r = phi0, theta0, r0
    
    # Pre-compute coordinate arrays
    x_coords = np.asarray(x_coords, dtype=np.float32)
    y_coords = np.asarray(y_coords, dtype=np.float32)
    
    # Initialize best result (monotonic ML acceptance)
    best_phi, best_theta, best_r = phi, theta, r
    a_best = steering_nf(best_phi, best_theta, best_r, x_coords, y_coords, k)
    best_nll = negloglike(Y, a_best.reshape(-1, 1))  # Single source: [M, 1]
    
    for iteration in range(iters):
        # Build steering vector
        a = steering_nf(phi, theta, r, x_coords, y_coords, k)
        
        # Solve amplitudes in closed form: γ̂_ℓ = a^H y_ℓ
        gamma_hat = np.conj(a) @ Y  # [L] complex64
        
        # Compute residual: R = Y - a * γ̂^T
        residual = Y - np.outer(a, gamma_hat)  # [M, L]
        
        # Current objective: ||R||_F^2
        current_obj = np.linalg.norm(residual, 'fro')**2
        
        # Get Jacobians
        da_dphi, da_dtheta, da_dr = steering_nf_jacobians(phi, theta, r, x_coords, y_coords, k)
        
        # Compute gradients w.r.t. (φ, θ, r)
        # ∂f/∂φ = 2 * Re(tr(R^H * ∂R/∂φ)) = 2 * Re(tr(R^H * (-a_φ * γ̂^T)))
        grad_phi = -2 * np.real(np.trace(residual.conj().T @ np.outer(da_dphi, gamma_hat)))
        grad_theta = -2 * np.real(np.trace(residual.conj().T @ np.outer(da_dtheta, gamma_hat)))
        grad_r = -2 * np.real(np.trace(residual.conj().T @ np.outer(da_dr, gamma_hat)))
        
        # Gradient step with backtracking
        step_phi = step_size * grad_phi
        step_theta = step_size * grad_theta  
        step_r = step_size * grad_r
        
        # Apply step
        phi_new = phi - step_phi
        theta_new = theta - step_theta
        r_new = r - step_r
        
        # Clamp to bounds
        phi_new = np.clip(phi_new, phi_min, phi_max)
        theta_new = np.clip(theta_new, theta_min, theta_max)
        r_new = np.clip(r_new, r_min, r_max)
        
        # Monotonic ML acceptance: only accept if NLL improves
        a_candidate = steering_nf(phi_new, theta_new, r_new, x_coords, y_coords, k)
        candidate_nll = negloglike(Y, a_candidate.reshape(-1, 1))
        
        if candidate_nll < best_nll:
            # Accept the step
            best_phi, best_theta, best_r = phi_new, theta_new, r_new
            best_nll = candidate_nll
            phi, theta, r = phi_new, theta_new, r_new
        else:
            # Reject: backtrack or stop
            step_size *= 0.5
            if step_size < 0.1:
                break  # Converged or step too small
    
    return best_phi, best_theta, best_r


def refine_nf_mle_multi(Y: np.ndarray, phi_init: np.ndarray, theta_init: np.ndarray, r_init: np.ndarray,
                        x_coords: np.ndarray, y_coords: np.ndarray, k: float,
                        iters: int = 3, step_size: float = 0.7,
                        r_min: float = 0.4, r_max: float = 12.0,
                        phi_min: float = -np.pi/3, phi_max: float = np.pi/3,
                        theta_min: float = -np.pi/6, theta_max: float = np.pi/6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-source NF-MLE polish using cyclic coordinate descent.
    
    Args:
        Y: Snapshots [M, L] complex64
        phi_init, theta_init, r_init: Initial parameters [K] (rad/m)
        x_coords, y_coords: Sensor coordinates [M] in meters
        k: Wavenumber (2π/λ)
        iters: Number of MLE iterations per source
        step_size: Step size scaling
        r_min, r_max: Range bounds in meters
        phi_min, phi_max: Azimuth bounds in radians
        theta_min, theta_max: Elevation bounds in radians
    
    Returns:
        phi_refined, theta_refined, r_refined: Refined parameters [K]
    """
    K = len(phi_init)
    phi, theta, r = phi_init.copy(), theta_init.copy(), r_init.copy()
    
    # Initialize best result (monotonic ML acceptance)
    best_phi, best_theta, best_r = phi.copy(), theta.copy(), r.copy()
    A_best = np.zeros((len(x_coords), K), dtype=np.complex64)
    for i in range(K):
        A_best[:, i] = steering_nf(best_phi[i], best_theta[i], best_r[i], x_coords, y_coords, k)
    best_nll = negloglike(Y, A_best)
    
    for iteration in range(iters):
        for k_idx in range(K):
            # Build steering matrix for all sources
            A = np.zeros((len(x_coords), K), dtype=np.complex64)
            for i in range(K):
                A[:, i] = steering_nf(phi[i], theta[i], r[i], x_coords, y_coords, k)
            
            # Solve amplitudes with Tikhonov regularization: Γ̂ = (A^H A + λI)^(-1) A^H Y
            # λ ≈ σ̂² for stability when columns of A are coherent
            try:
                AHA = A.conj().T @ A  # [K, K]
                # Estimate noise level for Tikhonov regularization
                noise_estimate = np.mean(np.linalg.eigvalsh(AHA)) * 1e-3  # Conservative estimate
                lambda_reg = max(noise_estimate, 1e-6)  # Ensure positive
                AHA_reg = AHA + lambda_reg * np.eye(K, dtype=AHA.dtype)
                gamma_hat = np.linalg.solve(AHA_reg, A.conj().T @ Y)  # [K, L]
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse
                gamma_hat = np.linalg.pinv(A) @ Y  # [K, L]
            
            # Compute residual
            residual = Y - A @ gamma_hat  # [M, L]
            current_obj = np.linalg.norm(residual, 'fro')**2
            
            # Update source k_idx only
            phi_k, theta_k, r_k = refine_nf_mle(
                Y, phi[k_idx], theta[k_idx], r[k_idx],
                x_coords, y_coords, k,
                iters=1, step_size=step_size,
                r_min=r_min, r_max=r_max,
                phi_min=phi_min, phi_max=phi_max,
                theta_min=theta_min, theta_max=theta_max
            )
            
            # Monotonic ML acceptance: only accept if NLL improves
            A_candidate = A.copy()
            A_candidate[:, k_idx] = steering_nf(phi_k, theta_k, r_k, x_coords, y_coords, k)
            candidate_nll = negloglike(Y, A_candidate)
            
            if candidate_nll < best_nll:
                # Accept the update
                phi[k_idx] = phi_k
                theta[k_idx] = theta_k
                r[k_idx] = r_k
                best_phi, best_theta, best_r = phi.copy(), theta.copy(), r.copy()
                best_nll = candidate_nll
    
    return phi, theta, r


def nf_mle_polish_after_newton(phi_newton: np.ndarray, theta_newton: np.ndarray, r_newton: np.ndarray,
                               Y: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, k: float,
                               cfg, snr_est: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply short NF-MLE polish after joint NF-Newton refinement.
    
    This is the final step in our inference pipeline that provides
    true low-SNR robustness by fitting actual snapshots.
    
    Args:
        phi_newton, theta_newton, r_newton: Newton-refined parameters [K]
        Y: Snapshots [M, L] complex64
        x_coords, y_coords: Sensor coordinates [M] in meters
        k: Wavenumber (2π/λ)
        cfg: Config object
        snr_est: Estimated SNR in dB (gate MLE on low SNR)
    
    Returns:
        phi_final, theta_final, r_final: MLE-polished parameters [K]
    """
    # Gate MLE polish on SNR (only run at low/medium SNR)
    SNR_THRESHOLD = getattr(cfg, 'NF_MLE_SNR_THRESHOLD', 8.0)  # dB
    if snr_est > SNR_THRESHOLD:
        # High SNR: Newton is sufficient
        return phi_newton, theta_newton, r_newton
    
    K = len(phi_newton)
    if K == 0:
        return phi_newton, theta_newton, r_newton
    
    # Extract bounds from config
    r_min = getattr(cfg, 'RANGE_R', (0.5, 10.0))[0]
    r_max = getattr(cfg, 'RANGE_R', (0.5, 10.0))[1]
    phi_min = np.deg2rad(getattr(cfg, 'PHI_MIN_DEG', -60.0))
    phi_max = np.deg2rad(getattr(cfg, 'PHI_MAX_DEG', 60.0))
    theta_min = np.deg2rad(getattr(cfg, 'THETA_MIN_DEG', -30.0))
    theta_max = np.deg2rad(getattr(cfg, 'THETA_MAX_DEG', 30.0))
    
    # Convert initial parameters to radians
    phi_rad = np.deg2rad(phi_newton)
    theta_rad = np.deg2rad(theta_newton)
    
    # Run NF-MLE polish
    if K == 1:
        # Single source: direct optimization
        phi_final, theta_final, r_final = refine_nf_mle(
            Y, phi_rad[0], theta_rad[0], r_newton[0],
            x_coords, y_coords, k,
            iters=3, step_size=0.7,
            r_min=r_min, r_max=r_max,
            phi_min=phi_min, phi_max=phi_max,
            theta_min=theta_min, theta_max=theta_max
        )
        phi_final = np.array([phi_final])
        theta_final = np.array([theta_final])
        r_final = np.array([r_final])
    else:
        # Multi-source: cyclic coordinate descent
        phi_final, theta_final, r_final = refine_nf_mle_multi(
            Y, phi_rad, theta_rad, r_newton,
            x_coords, y_coords, k,
            iters=3, step_size=0.7,
            r_min=r_min, r_max=r_max,
            phi_min=phi_min, phi_max=phi_max,
            theta_min=theta_min, theta_max=theta_max
        )
    
    # Convert back to degrees
    phi_final_deg = np.rad2deg(phi_final)
    theta_final_deg = np.rad2deg(theta_final)
    
    return phi_final_deg, theta_final_deg, r_final
