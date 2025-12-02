# SPDX-License-Identifier: MIT
# Build Rx[τ] stacks from complex snapshots Y (L,T)
import numpy as np

def autocorr_stack_from_y(y: np.ndarray, tau_max: int = 12) -> np.ndarray:
    """
    y: complex ndarray, shape (L, T)  [L sensors, T snapshots]
    returns: RI stack, shape (2*(tau_max+1), L, L), dtype=float32
      order: [Re R0, Im R0, Re R1, Im R1, ..., Re R_tau, Im R_tau]
    """
    assert np.iscomplexobj(y) and y.ndim == 2, "y must be complex (L,T)"
    L, T = y.shape
    out = np.empty((2*(tau_max+1), L, L), dtype=np.float32)
    for tau in range(tau_max+1):
        X0 = y[:, :T-tau]
        X1 = y[:, tau:]
        R = (X0 @ X1.conj().T) / max(1, (T - tau))
        out[2*tau  ] = R.real.astype(np.float32)
        out[2*tau+1] = R.imag.astype(np.float32)
    return out
