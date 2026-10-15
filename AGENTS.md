# Heston American Put – Project Notes

## Project layout
```
src/           – core library modules
  __init__.py
  boundary.py        – free-boundary extraction (b(tau,y)) from PDE gap variable w
  gamma_identity.py  – boundary gamma identity verification (P_ss formula)
exp/           – experiment scripts
tests/         – test suite
data/          – input data
results/       – output artefacts
```

## Variable conventions
| Symbol | Meaning |
|--------|---------|
| `x`    | log-price, x = log(S) |
| `y`    | variance process (CIR) |
| `tau`  | time-to-expiry |
| `w`    | gap variable: w = U - (K - e^x)^+ in log-price space; equivalently w = U - K + e^x (since in the exercise region U = K - e^x) |
| `b`    | free boundary in price space; `beta = log(b)` |
| `K`    | strike |

## heston_pde.py – key design decisions
- Flat 1-D index convention: `idx(j, k) = j*Ny + k` (x-major order).
- Assembles the full sparse operator `A` once (coefficients depend only on y, not tau).  The system matrix `M = I/dt - A` is LU-factorised once with `scipy.sparse.linalg.splu`; the same factorisation is reused at every time step (huge performance gain).
- y=0 row: degenerate transport equation — forward difference in x, forward in y — **no** exogenous Dirichlet/Neumann override.
- LCP enforcement: after each implicit-Euler solve, project `u = max(u, obstacle)` (standard Ikonen-Toivanen single-step splitting).
- Boundary extraction inside the time march uses `_extract_boundary_row` which applies the quadratic-contact rule and **clamps b ≤ K**.  The clamp is essential because numerical noise in the degenerate y=0 transport can push the sqrt(w) extrapolation above K.

## boundary.py – key design decisions
- **Quadratic-contact rule** (`extract_boundary_at_slice`): near the boundary sqrt(w) is linear in x-beta; extrapolate the first two continuation nodes to zero.  Fallback: last contact node when fewer than 2 continuation nodes exist.
- `extract_boundary_row` operates on a 2-D array `U_2d` of shape `(Nx, Ny)` at a fixed tau; loops over y-columns.
- `estimate_boundary_slope` uses a local quadratic OLS fit over a window in y; **always excludes y=0** from the fitting set.  The y=0 row is governed by the degenerate transport equation and its boundary values are qualitatively different from the diffusive interior; including it in the quadratic fit corrupts the slope estimate at small positive y values, flipping the sign of b_y from negative to positive.  Excluding y=0 restores the correct negative slope everywhere.
- `verify_monotonicity` returns `(bool, int)` – `(is_monotone, num_violations)`.
- `boundary_sanity_check` expects `b_surface[0, :]` to be the tau=0 row (should be near K).

## Coarse-grid artefacts vs paper results
- Fast mode (dx=0.02, dy=0.006, Nt=200) is 3× coarser than the paper (dx=0.0062, dy=0.002, Nt=2000).
- This produces O(2k–6k) monotonicity violations in the boundary surface; these are discretisation artefacts and disappear on the full paper grid.
- Gamma identity discrepancies at y=0.04 are ~16–46 % in fast mode vs ~2 % in the paper.  For y ≥ 0.16 the fast grid already achieves ~2–10 % agreement.
- All qualitative properties (b ∈ (0,K], b_y < 0, P_ss > 0, b→K as τ→0) hold on both grids.

## gamma_identity.py – key design decisions
- **Gamma identity**: `P_ss(tau,b+,y) = 2*(r*K - q*b) / [y*(b² - 2ρσ·b·b_y + σ²·b_y²)]`
- The denominator quadratic form `b² - 2ρσ·b·b_y + σ²·b_y²` = `(b-ρσ·b_y)² + (1-ρ²)σ²·b_y²` is **always ≥ 0** for |ρ| ≤ 1.  The `denominator <= 0` guard in `gamma_formula` is therefore only triggered by `y ≤ 0`.
- `estimate_wxx_at_boundary`: skips first `n_skip` continuation nodes (boundary artefacts), uses next `n_use` nodes for centred second-differences, then linearly extrapolates to x=beta.
- `compute_gamma_table` uses a deferred import (`from src.boundary import estimate_boundary_slope`) to avoid potential circular imports.
- `P_ss` from finite differences: `Pss_fd = wxx_plus / b^2` (chain rule from log-price to price space).

## Dependencies
- `numpy`, `pandas` (standard scientific stack – pre-installed)
