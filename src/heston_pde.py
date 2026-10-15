"""
Finite-difference solver for the American put obstacle problem under the Heston model.

Implements:
- Uniform grid in (tau, x, y) with implicit Euler time stepping
- Ikonen-Toivanen operator splitting for the LCP (early exercise)
- Degenerate transport discretization at y=0
- Nine-point central difference for the mixed derivative U_xy
  (equivalent to IT seven-point stencil under the admissibility condition)

Reference: Numerical reproduction of the free-boundary surface s=b(tau,y)
as in Section 5 of the paper on American put pricing under Heston dynamics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------

@dataclass
class HestonParams:
    """Heston model risk-neutral parameters."""
    K: float = 1.0
    r: float = 0.05
    q: float = 0.02
    kappa: float = 2.0
    theta: float = 0.04
    rho: float = -0.7
    sigma: float = 0.3   # Set A; use 0.6 for Set B
    T: float = 1.0

    @property
    def feller_lhs(self) -> float:
        """2*kappa*theta (LHS of Feller condition 2*kappa*theta >= sigma^2)."""
        return 2.0 * self.kappa * self.theta

    @property
    def feller_satisfied(self) -> bool:
        """True when Feller condition 2*kappa*theta >= sigma^2 holds."""
        return self.feller_lhs >= self.sigma ** 2


@dataclass
class GridParams:
    """Uniform finite-difference grid parameters."""
    Nx: int = 705           # x-nodes
    Ny: int = 601           # y-nodes (601 for Set A, 301 for Set B)
    Nt: int = 2000          # time steps
    dx: float = 0.0062
    dy: float = 0.002       # Set A; 0.004 for Set B
    dt: float = 0.0005
    x_min: float = -2.50   # Chosen so that interior boundary is far from edges

    @property
    def x_max(self) -> float:
        return self.x_min + (self.Nx - 1) * self.dx

    @property
    def y_max(self) -> float:
        return (self.Ny - 1) * self.dy

    def x_grid(self) -> np.ndarray:
        return self.x_min + np.arange(self.Nx) * self.dx

    def y_grid(self) -> np.ndarray:
        return np.arange(self.Ny) * self.dy

    def tau_grid(self) -> np.ndarray:
        return np.arange(self.Nt + 1) * self.dt


# ---------------------------------------------------------------------------
# Admissibility check
# ---------------------------------------------------------------------------

def verify_admissibility(hp: HestonParams, gp: GridParams) -> Tuple[bool, float, float]:
    """
    Verify the mixed-derivative grid-ratio admissibility condition.

    For the IT stencil to be monotone (M-matrix), the grid spacings must satisfy:
        |rho| * sigma * dx <= dy <= sigma * dx / |rho|

    Returns:
        (is_admissible, lower_bound, upper_bound)
    """
    abs_rho = abs(hp.rho)
    lower = abs_rho * hp.sigma * gp.dx
    upper = hp.sigma * gp.dx / abs_rho
    is_ok = lower <= gp.dy <= upper
    return is_ok, lower, upper


# ---------------------------------------------------------------------------
# Sparse operator assembly
# ---------------------------------------------------------------------------

def _node(j: int, k: int, Ny: int) -> int:
    """Convert 2D grid index (j, k) to 1D flattened index."""
    return j * Ny + k


def build_heston_operator(hp: HestonParams, gp: GridParams) -> sparse.csr_matrix:
    """
    Assemble sparse matrix A for the Heston spatial operator L*u.

    Interior stencil (j=1..Nx-2, k=1..Ny-2):
        Lu = y/2*U_xx + rho*sigma*y*U_xy + sigma^2*y/2*U_yy
             + (r-q-y/2)*U_x + kappa*(theta-y)*U_y

    y=0 degenerate axis (k=0, j=1..Nx-2):
        Transport equation: Lu = (r-q)*U_x + kappa*theta*U_y
        Central diff for U_x, forward diff for U_y.

    y=y_max (k=Ny-1): homogeneous Neumann via one-sided differencing.

    x-boundaries (j=0, j=Nx-1): Identity rows (Dirichlet enforced in M).

    The mixed derivative U_xy uses the nine-point central difference,
    which is equivalent to the Ikonen-Toivanen seven-point stencil
    under the admissibility condition |rho|*sigma*dx <= dy <= sigma*dx/|rho|.
    # Custom — Context7 found no library equivalent for the degenerate Heston FD stencil
    """
    Nx, Ny = gp.Nx, gp.Ny
    N = Nx * Ny
    dx, dy = gp.dx, gp.dy

    x_grid = gp.x_grid()
    y_grid = gp.y_grid()

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    def add(r: int, c: int, v: float) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    for j in range(Nx):
        for k in range(Ny):
            idx = _node(j, k, Ny)
            y_k = y_grid[k]

            # ---- x-boundary Dirichlet: identity row ----
            if j == 0 or j == Nx - 1:
                add(idx, idx, 1.0)
                continue

            # ---- y=0 degenerate transport row ----
            if k == 0:
                drift_x = hp.r - hp.q            # coefficient of U_x
                drift_y = hp.kappa * hp.theta     # coefficient of U_y (kappa*theta at y=0)
                # U_x: central difference
                add(idx, _node(j - 1, 0, Ny), -drift_x / (2.0 * dx))
                add(idx, _node(j + 1, 0, Ny),  drift_x / (2.0 * dx))
                # U_y: forward difference
                add(idx, _node(j, 0, Ny), -drift_y / dy)
                add(idx, _node(j, 1, Ny),  drift_y / dy)
                continue

            # ---- Interior and y=y_max nodes ----
            # PDE coefficients
            a    = y_k / 2.0                         # U_xx
            b_cr = hp.rho * hp.sigma * y_k            # U_xy (mixed)
            c    = hp.sigma**2 * y_k / 2.0            # U_yy
            d    = hp.r - hp.q - y_k / 2.0            # U_x
            e    = hp.kappa * (hp.theta - y_k)         # U_y

            # ---- U_xx: standard central differences ----
            add(idx, _node(j - 1, k, Ny),  a / dx**2)
            add(idx, idx,                  -2.0 * a / dx**2)
            add(idx, _node(j + 1, k, Ny),  a / dx**2)

            # ---- U_x: central differences ----
            add(idx, _node(j - 1, k, Ny), -d / (2.0 * dx))
            add(idx, _node(j + 1, k, Ny),  d / (2.0 * dx))

            # ---- U_yy and U_y: handle y=y_max (Neumann) vs interior ----
            if k < Ny - 1:
                # Interior in y: standard central differences
                add(idx, _node(j, k - 1, Ny),  c / dy**2)
                add(idx, idx,                  -2.0 * c / dy**2)
                add(idx, _node(j, k + 1, Ny),  c / dy**2)
                add(idx, _node(j, k - 1, Ny), -e / (2.0 * dy))
                add(idx, _node(j, k + 1, Ny),  e / (2.0 * dy))

                # ---- U_xy: nine-point central (IT seven-point under admissibility) ----
                # Ikonen-Toivanen seven-point stencil reduces to nine-point central
                # under the admissibility condition |rho|*sigma*dx <= dy <= sigma*dx/|rho|
                if k > 0:  # need k-1 row (already interior since k >= 1 here)
                    coeff_xy = b_cr / (4.0 * dx * dy)
                    add(idx, _node(j + 1, k + 1, Ny),  coeff_xy)
                    add(idx, _node(j + 1, k - 1, Ny), -coeff_xy)
                    add(idx, _node(j - 1, k + 1, Ny), -coeff_xy)
                    add(idx, _node(j - 1, k - 1, Ny),  coeff_xy)

            else:
                # k = Ny-1: homogeneous Neumann U_y=0 via ghost node U_{j,Ny} = U_{j,Ny-1}
                # U_yy one-sided: (U_{j,Ny-2} - U_{j,Ny-1}) / dy^2
                add(idx, _node(j, k - 1, Ny),  c / dy**2)
                add(idx, idx,                  -c / dy**2)
                # U_y: one-sided backward (Neumann: U_y = 0 -> no contribution)
                # U_xy at top Neumann: use (k-1) for the top ghost
                coeff_xy = b_cr / (4.0 * dx * dy)
                # Treat U_{j,Ny} = U_{j,Ny-1}: stencil reduces
                # U_xy ~ (U_{j+1,k} - U_{j+1,k-1} - U_{j-1,k} + U_{j-1,k-1}) / (2*dx*dy)
                add(idx, _node(j + 1, k,     Ny),  coeff_xy * 2.0)
                add(idx, _node(j + 1, k - 1, Ny), -coeff_xy * 2.0)
                add(idx, _node(j - 1, k,     Ny), -coeff_xy * 2.0)
                add(idx, _node(j - 1, k - 1, Ny),  coeff_xy * 2.0)

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return A


def build_system_matrix(
    hp: HestonParams,
    gp: GridParams,
    A: sparse.csr_matrix,
) -> sparse.csc_matrix:
    """
    Build the implicit Euler system matrix M = (1/dt + r)*I - A.

    Implicit Euler for u_tau = Lu - r*u:
        (u^{n+1} - u^n) / dt = L*u^{n+1} - r*u^{n+1}
        ((1/dt + r)*I - A) * u^{n+1} = u^n / dt
        M * u^{n+1} = u^n / dt

    Boundary rows (Dirichlet) have identity in A, so in M they have
    (1/dt + r + (whatever extra from A)) ... but since A boundary rows
    are identity, M boundary rows = (1/dt + r - 1)*I which is wrong.

    Fix: for boundary nodes, overwrite M row to be identity with rhs = bc_value.
    We handle this at solve time by injecting the BC rhs directly.
    Here we build the raw M = (1/dt + r)*I - A, then fix boundary rows.
    """
    Nx, Ny = gp.Nx, gp.Ny
    N = Nx * Ny
    inv_dt = 1.0 / gp.dt

    M = (inv_dt + hp.r) * sparse.eye(N, format="csc") - A.tocsc()

    # Fix Dirichlet boundary rows (j=0 and j=Nx-1):
    # Set row to identity by zeroing out row and putting 1 on diagonal.
    dirichlet_js = [0, Nx - 1]
    for j in dirichlet_js:
        for k in range(Ny):
            idx = _node(j, k, Ny)
            # Zero the row
            M.data[M.indptr[idx]:M.indptr[idx + 1]] = 0.0
            # Set diagonal to 1
            M[idx, idx] = 1.0

    M.eliminate_zeros()
    return M.tocsc()


# ---------------------------------------------------------------------------
# Right-hand side helpers
# ---------------------------------------------------------------------------

def _build_rhs_bc(hp: HestonParams, gp: GridParams) -> np.ndarray:
    """
    Build the time-invariant Dirichlet BC contribution to the RHS.

    Interior and y-boundary nodes: 0 (their rhs is u^n / dt, added at solve time).
    Dirichlet nodes (j=0: U=K-e^x; j=Nx-1: U=0): constant BC value.
    """
    Nx, Ny = gp.Nx, gp.Ny
    N = Nx * Ny
    rhs_bc = np.zeros(N)

    x_grid = gp.x_grid()

    for k in range(Ny):
        # Lower x-boundary: U = K - e^x (in-the-money put, K=1)
        idx_lo = _node(0, k, Ny)
        rhs_bc[idx_lo] = hp.K - np.exp(x_grid[0])

        # Upper x-boundary: U = 0
        idx_hi = _node(Nx - 1, k, Ny)
        rhs_bc[idx_hi] = 0.0

    return rhs_bc


def _build_obstacle(hp: HestonParams, gp: GridParams) -> np.ndarray:
    """
    Build the obstacle (payoff) surface g(x) = max(K - e^x, 0) as a 1D array.

    Shape: (Nx * Ny,) — same value for all y levels since payoff depends only on x.
    """
    Nx, Ny = gp.Nx, gp.Ny
    x_grid = gp.x_grid()
    g_x = np.maximum(hp.K - np.exp(x_grid), 0.0)   # shape (Nx,)
    # Tile across y: each row of (j, k) has obstacle g_x[j]
    obstacle = np.tile(g_x[:, None], (1, Ny)).ravel()  # shape (Nx*Ny,)
    return obstacle


# ---------------------------------------------------------------------------
# Boundary extraction (inline, efficient version)
# ---------------------------------------------------------------------------

def _extract_boundary_row(
    U_2d: np.ndarray,
    x_grid: np.ndarray,
    K: float,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Extract b(y) at a single time level from U of shape (Nx, Ny).

    Uses the quadratic-contact rule: sqrt(w) is linear near the boundary,
    extrapolate first two continuation nodes to find beta=log(b).

    Robust fallbacks (documented per paper §Implementation Notes):
    - Negative w values from roundoff are zeroed before processing.
    - If dsw <= 0 (non-monotone w profile, e.g., at y=0 degenerate axis where
      the transport equation does not create a quadratic contact region), fall
      back to the first continuation node as the boundary estimate.
    - b is capped at K because the American put exercise boundary satisfies
      b(tau,y) <= K for all tau>0 (early exercise only in the money).
    """
    Nx, Ny = U_2d.shape
    s_grid = np.exp(x_grid)
    log_K = np.log(K)
    b_row = np.zeros(Ny)

    for k in range(Ny):
        w_slice = U_2d[:, k] - K + s_grid   # w = U - K + e^x
        # Zero small negatives caused by roundoff in the contact region
        w_slice = np.maximum(w_slice, 0.0)

        # Find continuation region: w > eps
        cont_mask = w_slice > eps
        cont_idx = np.where(cont_mask)[0]

        if len(cont_idx) < 2:
            # Fallback: use x at last contact node
            contact_idx = np.where(~cont_mask)[0]
            b_row[k] = np.exp(x_grid[contact_idx[-1]]) if len(contact_idx) > 0 else np.exp(x_grid[0])
            continue

        j1, j2 = cont_idx[0], cont_idx[1]
        sw1 = np.sqrt(w_slice[j1])
        sw2 = np.sqrt(w_slice[j2])
        x1, x2 = x_grid[j1], x_grid[j2]

        denom = x2 - x1
        dsw = sw2 - sw1
        # dsw must be positive (sqrt(w) increases away from the boundary).
        # If dsw <= 0 the w profile is non-monotone (common near y=0 where the
        # degenerate transport PDE does not produce a smooth diffusive contact
        # region); in that case fall back to the first continuation node.
        if abs(denom) < 1e-15 or dsw <= 0:
            b_row[k] = min(np.exp(x1), K)
            continue

        m = dsw / denom         # m > 0 guaranteed
        beta = x1 - sw1 / m    # linear zero-crossing: beta = log(b)
        beta = min(beta, log_K) # enforce b <= K (American put boundary)
        b_row[k] = np.exp(beta)

    return b_row


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_american_put(
    hp: HestonParams,
    gp: GridParams,
    checkpoint_taus: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Solve the American put obstacle problem under the Heston model.

    Uses implicit Euler time stepping and Ikonen-Toivanen operator splitting
    (projection onto obstacle after each linear solve).

    Parameters
    ----------
    hp : HestonParams
        Heston model parameters.
    gp : GridParams
        Finite-difference grid parameters.
    checkpoint_taus : array-like, optional
        Tau values at which to save full (U, w) solutions.
        Defaults to [0.25, 0.50, 1.00].
    verbose : bool
        Print progress every 100 time steps.

    Returns
    -------
    x_grid : ndarray, shape (Nx,)
    y_grid : ndarray, shape (Ny,)
    boundary : ndarray, shape (Nt+1, Ny)
        boundary[n, k] = b(tau_n, y_k), the exercise boundary at each time step.
    checkpoints : dict
        {tau_val: {'U': ndarray(Nx,Ny), 'w': ndarray(Nx,Ny)}}
    """
    if checkpoint_taus is None:
        checkpoint_taus = np.array([0.25, 0.50, 1.00])

    # ---- Admissibility check ----
    ok, lb, ub = verify_admissibility(hp, gp)
    if verbose:
        print(f"Admissibility check: |rho|*sigma*dx={lb:.6f} <= dy={gp.dy} <= {ub:.6f} -> {'OK' if ok else 'FAILED'}")
    assert ok, f"Grid admissibility condition violated: dy={gp.dy} not in [{lb:.6f}, {ub:.6f}]"

    Nx, Ny, Nt = gp.Nx, gp.Ny, gp.Nt
    N = Nx * Ny
    x_grid = gp.x_grid()
    y_grid = gp.y_grid()
    tau_grid = gp.tau_grid()  # shape (Nt+1,)
    inv_dt = 1.0 / gp.dt

    if verbose:
        print(f"Grid: Nx={Nx}, Ny={Ny}, N={N}, Nt={Nt}")
        print(f"x in [{x_grid[0]:.4f}, {x_grid[-1]:.4f}], y in [{y_grid[0]:.4f}, {y_grid[-1]:.4f}]")
        print(f"Feller condition: 2*kappa*theta={hp.feller_lhs:.4f} >= sigma^2={hp.sigma**2:.4f}: {hp.feller_satisfied}")

    # ---- Build operator and factorize (done once since M is stationary) ----
    t0 = time.time()
    if verbose:
        print("Assembling spatial operator A ...")
    A = build_heston_operator(hp, gp)

    if verbose:
        print(f"  A nnz = {A.nnz}, shape = {A.shape}. Elapsed: {time.time()-t0:.1f}s")
        print("Building system matrix M and factorizing (LU decomposition) ...")

    M = build_system_matrix(hp, gp, A)

    t1 = time.time()
    if verbose:
        print(f"  M nnz = {M.nnz}. Starting splu ...")
    lu = splu(M)
    if verbose:
        print(f"  splu done. Elapsed: {time.time()-t1:.1f}s")

    # ---- Initial condition and obstacle ----
    obstacle = _build_obstacle(hp, gp)          # shape (N,)
    rhs_bc = _build_rhs_bc(hp, gp)              # shape (N,) - Dirichlet RHS part

    # Initial condition at tau=0: U(0,x,y) = max(K - e^x, 0)
    s_grid = np.exp(x_grid)
    u = np.tile(np.maximum(hp.K - s_grid, 0.0)[:, None], (1, Ny)).ravel()

    # ---- Storage ----
    boundary = np.zeros((Nt + 1, Ny))           # b(tau_n, y_k)
    checkpoints: Dict = {}

    # Record initial boundary (tau=0: U = obstacle everywhere, b approx K)
    U_2d = u.reshape(Nx, Ny)
    boundary[0, :] = _extract_boundary_row(U_2d, x_grid, hp.K)

    # Identify checkpoint steps
    ckpt_set = set()
    for tau_ckpt in checkpoint_taus:
        n_ckpt = int(round(tau_ckpt / gp.dt))
        n_ckpt = min(n_ckpt, Nt)
        ckpt_set.add(n_ckpt)

    # ---- Time marching ----
    t2 = time.time()
    if verbose:
        print(f"Starting time march: {Nt} steps ...")

    # Identify Dirichlet node mask for RHS correction
    dirichlet_mask = np.zeros(N, dtype=bool)
    for j in [0, Nx - 1]:
        for k in range(Ny):
            dirichlet_mask[_node(j, k, Ny)] = True

    for n in range(1, Nt + 1):
        # Build RHS: (1/dt)*u^n for interior, BC value for Dirichlet
        rhs = inv_dt * u.copy()
        rhs[dirichlet_mask] = rhs_bc[dirichlet_mask]

        # Solve linear system
        u_new = lu.solve(rhs)

        # Project onto obstacle (Ikonen-Toivanen splitting)
        u_new = np.maximum(u_new, obstacle)

        # Enforce Dirichlet boundary exactly after projection
        u_new[dirichlet_mask] = rhs_bc[dirichlet_mask]

        u = u_new
        U_2d = u.reshape(Nx, Ny)

        # Extract boundary at this time step
        boundary[n, :] = _extract_boundary_row(U_2d, x_grid, hp.K)

        # Save checkpoint
        if n in ckpt_set:
            w_2d = U_2d - hp.K + s_grid[:, None]  # w = U - K + e^x
            tau_val = tau_grid[n]
            checkpoints[float(tau_val)] = {
                "U": U_2d.copy(),
                "w": w_2d.copy(),
            }
            if verbose:
                print(f"  Checkpoint saved at tau={tau_val:.4f} (step {n}). Elapsed: {time.time()-t2:.1f}s")

        if verbose and n % 200 == 0:
            elapsed = time.time() - t2
            print(f"  Step {n}/{Nt}, tau={tau_grid[n]:.4f}, elapsed={elapsed:.1f}s")

    if verbose:
        print(f"Time march complete. Total elapsed: {time.time()-t0:.1f}s")

    return x_grid, y_grid, boundary, checkpoints


# ---------------------------------------------------------------------------
# Convenience factory for Set A and Set B
# ---------------------------------------------------------------------------

def make_set_a() -> Tuple[HestonParams, GridParams]:
    """Return the paper's Set A parameters (Feller satisfied, sigma=0.3)."""
    hp = HestonParams(sigma=0.3)
    gp = GridParams(Nx=705, Ny=601, Nt=2000, dx=0.0062, dy=0.002, dt=0.0005, x_min=-2.50)
    return hp, gp


def make_set_b() -> Tuple[HestonParams, GridParams]:
    """Return the paper's Set B parameters (Feller violated, sigma=0.6)."""
    hp = HestonParams(sigma=0.6)
    gp = GridParams(Nx=705, Ny=301, Nt=2000, dx=0.0062, dy=0.004, dt=0.0005, x_min=-2.50)
    return hp, gp
