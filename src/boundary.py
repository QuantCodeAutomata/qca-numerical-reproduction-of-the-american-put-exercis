"""
Module for extracting the free boundary b(tau,y) from the PDE solution.
"""

import numpy as np
from typing import Tuple, Optional, Dict


def extract_boundary_at_slice(
    w_slice: np.ndarray,
    x_grid: np.ndarray,
    eps: float = 1e-10,
    K: float = 1.0,
) -> float:
    """
    Extract beta=log(b(tau,y)) from a single (tau,y) slice of the gap variable w.

    Method (quadratic-contact rule from paper):
    - w=0 in contact region, w>0 in continuation region
    - Near boundary: sqrt(w) is linear in x-beta
    - Find first two continuation nodes (w > eps), fit line through (x, sqrt(w))
    - Extrapolate line to zero -> beta; return exp(beta) as the boundary level b

    Robust fallbacks (documented per paper §Implementation Notes):
    - Negative w values from roundoff are zeroed before processing.
    - If dsw <= 0 (non-monotone w profile at y=0 degenerate axis), fall back to
      the first continuation node.
    - b is capped at K (American put boundary satisfies b <= K).

    Falls back to x at last contact node if fewer than 2 continuation nodes found.
    """
    # Zero negative values caused by roundoff in the contact region
    w_slice = np.maximum(w_slice, 0.0)

    cont_mask = w_slice > eps
    cont_indices = np.where(cont_mask)[0]

    if len(cont_indices) < 2:
        contact_indices = np.where(~cont_mask)[0]
        if len(contact_indices) > 0:
            return float(np.exp(x_grid[contact_indices[-1]]))
        return float(np.exp(x_grid[0]))

    j1, j2 = cont_indices[0], cont_indices[1]
    sw1 = np.sqrt(w_slice[j1])
    sw2 = np.sqrt(w_slice[j2])
    x1, x2 = x_grid[j1], x_grid[j2]

    denom = x2 - x1
    dsw = sw2 - sw1

    # dsw must be positive; if non-monotone, fall back to first continuation node
    if abs(denom) < 1e-15 or dsw <= 0:
        return float(min(np.exp(x1), K))

    m = dsw / denom          # guaranteed m > 0
    beta = x1 - sw1 / m     # linear zero-crossing
    beta = min(beta, np.log(K))  # enforce b <= K
    return float(np.exp(beta))


def extract_boundary_row(
    U_2d: np.ndarray,
    x_grid: np.ndarray,
    K: float,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Extract boundary b(y) from a 2D solution array U of shape (Nx, Ny) at a single tau.

    Returns array of shape (Ny,) with boundary levels b(y_k).
    """
    Nx, Ny = U_2d.shape
    s_grid = np.exp(x_grid)  # s = e^x
    b_row = np.zeros(Ny)
    for k in range(Ny):
        w_slice = U_2d[:, k] - K + s_grid   # w = U - K + e^x
        b_row[k] = extract_boundary_at_slice(w_slice, x_grid, eps=eps, K=K)
    return b_row


def estimate_boundary_slope(
    b_row: np.ndarray,
    y_grid: np.ndarray,
    y_target: float,
    window: float = 0.1
) -> float:
    """
    Estimate db/dy at y_target using local quadratic polynomial fit.

    Fits b(y) ~ a0 + a1*(y - y_target) + a2*(y - y_target)^2 over the window
    [y_target - window, y_target + window] using OLS. Returns a1 = db/dy.

    The degenerate axis y=0 is always excluded from the fit because the transport
    equation there produces boundary values that behave differently from the
    interior diffusive region; including y=0 would corrupt slope estimates at
    small positive y values (paper §Methodology step 4).
    """
    # Exclude y=0 (degenerate axis) to avoid contaminating the quadratic fit
    interior = y_grid > 0
    y_int = y_grid[interior]
    b_int = b_row[interior]

    mask = (y_int >= y_target - window) & (y_int <= y_target + window)
    y_sel = y_int[mask]
    b_sel = b_int[mask]

    if len(y_sel) < 3:
        # Fallback to finite difference using interior points only
        idx = np.argmin(np.abs(y_int - y_target))
        if idx == 0:
            return (b_int[1] - b_int[0]) / (y_int[1] - y_int[0])
        if idx == len(y_int) - 1:
            return (b_int[-1] - b_int[-2]) / (y_int[-1] - y_int[-2])
        return (b_int[idx+1] - b_int[idx-1]) / (y_int[idx+1] - y_int[idx-1])

    # Fit quadratic: [1, y-y0, (y-y0)^2] * [a0, a1, a2]^T = b
    y_centered = y_sel - y_target
    A = np.column_stack([np.ones_like(y_centered), y_centered, y_centered**2])
    coeffs, _, _, _ = np.linalg.lstsq(A, b_sel, rcond=None)
    return float(coeffs[1])  # a1 = db/dy


def verify_monotonicity(
    b_surface: np.ndarray,
    axis: int = 0,
    tol: float = 1e-6
) -> Tuple[bool, int]:
    """
    Check if b_surface is non-increasing along the given axis.

    b_surface shape: (Ntau, Ny)
    axis=0: non-increasing in tau (row direction) - boundary decreases with time-to-expiry
    axis=1: non-increasing in y (column direction) - boundary decreases with variance

    Returns (is_monotone, num_violations)
    """
    diff = np.diff(b_surface, axis=axis)
    violations = np.sum(diff > tol)
    return violations == 0, int(violations)


def boundary_sanity_check(
    b_surface: np.ndarray,
    K: float,
    tol: float = 1e-3
) -> dict:
    """
    Sanity check on extracted boundary surface.
    b_surface shape: (Ntau, Ny) where b_surface[0] is at tau=0.

    Checks:
    1. b in (0, K) for tau > 0
    2. b(0, y) approaches K (tau=0 row)
    3. b is finite everywhere
    """
    results = {}
    results['all_positive'] = bool(np.all(b_surface > 0))
    results['all_below_K'] = bool(np.all(b_surface <= K + tol))
    results['finite'] = bool(np.all(np.isfinite(b_surface)))
    # At tau=0 (n=0), b should approach K
    b_at_tau0 = b_surface[0, :]
    results['tau0_near_K'] = bool(np.all(np.abs(b_at_tau0 - K) < 0.05))  # within 5%
    results['tau0_mean'] = float(np.mean(b_at_tau0))
    return results
