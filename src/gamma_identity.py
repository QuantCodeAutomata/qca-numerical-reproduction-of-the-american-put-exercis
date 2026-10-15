"""
Module for verifying the boundary gamma identity.

Gamma identity: P_ss(tau,b+,y) = 2*(r*K - q*b) / [y*(b^2 - 2*rho*sigma*b*b_y + sigma^2*b_y^2)]
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def estimate_wxx_at_boundary(
    w_slice: np.ndarray,
    x_grid: np.ndarray,
    beta: float,
    dx: float,
    n_skip: int = 2,
    n_use: int = 4,
    eps: float = 1e-10
) -> float:
    """
    Estimate w_xx^+ at the boundary by linear extrapolation from continuation nodes.

    Steps:
    1. Find continuation nodes (w > eps)
    2. Skip first n_skip nodes, use next n_use nodes
    3. Compute second differences w_xx_j = (w_{j+1} - 2*w_j + w_{j-1}) / dx^2
    4. Fit linear function of x through these n_use values
    5. Extrapolate to x=beta

    Returns w_xx^+ at the boundary.
    """
    cont_mask = w_slice > eps
    cont_indices = np.where(cont_mask)[0]

    if len(cont_indices) < n_skip + n_use + 1:
        # Not enough continuation nodes
        return np.nan

    # Use nodes at positions n_skip, n_skip+1, ..., n_skip+n_use-1 in continuation region
    target_indices = cont_indices[n_skip:n_skip + n_use]

    # Need one neighbor on each side for second difference
    # Check that all target nodes have valid neighbors
    valid = []
    wxx_vals = []
    x_vals = []

    for j in target_indices:
        if j > 0 and j < len(w_slice) - 1:
            wxx = (w_slice[j+1] - 2*w_slice[j] + w_slice[j-1]) / (dx**2)
            wxx_vals.append(wxx)
            x_vals.append(x_grid[j])

    if len(wxx_vals) < 2:
        return np.nan

    x_arr = np.array(x_vals)
    wxx_arr = np.array(wxx_vals)

    # Fit linear function: wxx = a + b*(x - beta)
    A = np.column_stack([np.ones(len(x_arr)), x_arr - beta])
    coeffs, _, _, _ = np.linalg.lstsq(A, wxx_arr, rcond=None)
    # Evaluate at x=beta: just the intercept
    return float(coeffs[0])


def gamma_formula(
    b: float,
    b_y: float,
    y: float,
    r: float,
    q: float,
    K: float,
    rho: float,
    sigma: float
) -> float:
    """
    Evaluate the analytical boundary gamma formula.

    Formula: P_ss = 2*(r*K - q*b) / [y*(b^2 - 2*rho*sigma*b*b_y + sigma^2*b_y^2)]

    Returns NaN if denominator is non-positive (undefined/degenerate).
    """
    numerator = 2.0 * (r * K - q * b)
    denominator = y * (b**2 - 2.0*rho*sigma*b*b_y + sigma**2 * b_y**2)
    if denominator <= 0 or y <= 0:
        return np.nan
    return numerator / denominator


def compute_gamma_table(
    w_checkpoints: Dict[float, np.ndarray],
    b_checkpoints: Dict[float, np.ndarray],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    tau_targets: List[float],
    y_targets: List[float],
    r: float,
    q: float,
    K: float,
    rho: float,
    sigma: float,
    dx: float,
    window: float = 0.1
) -> pd.DataFrame:
    """
    Compute Table 2-style comparison at selected (tau, y) points.

    w_checkpoints: {tau_val: w_array of shape (Nx, Ny)}
    b_checkpoints: {tau_val: b_array of shape (Ny,)}

    Returns DataFrame with columns:
    ['tau', 'y', 'b_over_K', 'by_over_K', 'Pss_fd', 'Pss_formula', 'pct_diff']
    """
    from src.boundary import estimate_boundary_slope

    rows = []
    for tau in tau_targets:
        # Find closest available tau checkpoint
        available_taus = list(w_checkpoints.keys())
        closest_tau = min(available_taus, key=lambda t: abs(t - tau))

        w_2d = w_checkpoints[closest_tau]  # shape (Nx, Ny)
        b_row = b_checkpoints[closest_tau]  # shape (Ny,)

        for y_t in y_targets:
            # Find closest y grid point
            k_idx = int(np.argmin(np.abs(y_grid - y_t)))
            y_actual = y_grid[k_idx]

            # Get boundary level
            b_val = b_row[k_idx]
            beta = np.log(b_val)

            # Estimate b_y using local quadratic fit
            b_y = estimate_boundary_slope(b_row, y_grid, y_t, window=window)

            # Estimate w_xx^+ from continuation side
            w_slice = w_2d[:, k_idx]  # shape (Nx,)
            wxx_plus = estimate_wxx_at_boundary(w_slice, x_grid, beta, dx)

            if np.isnan(wxx_plus) or b_val <= 0:
                Pss_fd = np.nan
            else:
                # P_ss_fd = w_xx^+ / b^2
                Pss_fd = wxx_plus / (b_val**2)

            # Formula gamma
            Pss_form = gamma_formula(b_val, b_y, y_actual, r, q, K, rho, sigma)

            # Percent difference
            if not np.isnan(Pss_fd) and not np.isnan(Pss_form) and abs(Pss_form) > 1e-12:
                pct_diff = (Pss_fd / Pss_form - 1.0) * 100.0
            else:
                pct_diff = np.nan

            rows.append({
                'tau': tau,
                'y': y_actual,
                'b_over_K': b_val / K,
                'by_over_K': b_y / K,
                'Pss_fd': Pss_fd,
                'Pss_formula': Pss_form,
                'pct_diff': pct_diff
            })

    return pd.DataFrame(rows)


def validate_gamma_positivity(
    table: pd.DataFrame
) -> dict:
    """
    Check theoretical properties of the gamma table.
    Returns dict of validation results.
    """
    results = {}
    results['all_Pss_fd_positive'] = bool(table['Pss_fd'].dropna().gt(0).all())
    results['all_Pss_formula_positive'] = bool(table['Pss_formula'].dropna().gt(0).all())
    results['all_by_negative'] = bool(table['by_over_K'].dropna().lt(0).all())
    results['all_b_below_K'] = bool(table['b_over_K'].dropna().lt(1).all())
    max_pct = table['pct_diff'].abs().dropna().max()
    results['max_abs_pct_diff'] = float(max_pct)
    return results
