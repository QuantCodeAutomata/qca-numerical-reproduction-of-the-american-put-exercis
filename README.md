# Numerical Reproduction of the American Put Exercise Boundary under the Heston Model

This repository numerically reproduces the exercise boundary surface `s=b(τ,y)` for an American put option under the Heston stochastic volatility model, as described in the paper's Section 5. Two parameter sets are studied:

- **Set A** (`σ=0.3`): Feller condition `2κθ ≥ σ²` is satisfied (`0.16 ≥ 0.09`)  
- **Set B** (`σ=0.6`): Feller condition is violated (`0.16 < 0.36`)

## Repository Structure

```
repo/
├── data/                     # No market data required (synthetic model inputs only)
├── src/
│   ├── heston_pde.py         # Core Heston PDE solver (FD with IT splitting)
│   ├── boundary.py           # Free boundary extraction
│   └── gamma_identity.py     # Boundary gamma identity verification
├── exp/
│   ├── exp1_boundary_surface.py   # Experiment 1: boundary surface reproduction
│   └── exp2_gamma_identity.py     # Experiment 2: gamma identity verification
├── tests/
│   └── test_heston_pde.py    # Comprehensive pytest tests
├── results/
│   └── RESULTS.md            # Experiment results and figures
└── README.md
```

## Model

The Heston model under the risk-neutral measure:
```
dS_u = (r-q)S_u du + S_u √(Y_u) dW_u^S
dY_u = κ(θ-Y_u)du + σ√(Y_u) dW_u
d⟨W^S, W⟩_u = ρ du
```

American put value: `P(τ,s,y) = sup_{0≤η≤τ} E[e^{-rη}(K-S_η)⁺]`

## Parameters

| Parameter | Value |
|-----------|-------|
| K (strike)  | 1.0  |
| r (rate)    | 0.05 |
| q (dividend)| 0.02 |
| κ (mean-rev)| 2.0  |
| θ (long-run var) | 0.04 |
| ρ (correlation) | -0.7 |
| T (maturity) | 1.0  |
| **σ (vol-of-vol)** | **0.3 (Set A) / 0.6 (Set B)** |

## Grid Specification

| Parameter | Set A | Set B |
|-----------|-------|-------|
| Nx (x-nodes) | 705 | 705 |
| Ny (y-nodes) | 601 | 301 |
| Nt (time steps) | 2000 | 2000 |
| Δx | 0.0062 | 0.0062 |
| Δy | 0.002 | 0.004 |
| Δτ | 0.0005 | 0.0005 |
| x_min | -2.50 | -2.50 |

## Numerical Method

1. **Log-price transformation**: `x = log(s/K)`, `U(τ,x,y) = P(τ,eˣ,y)`
2. **Obstacle problem**: `min{U_τ - LU + rU, U - (K-eˣ)⁺} = 0`
3. **Discretization**: Implicit Euler, central differences for `U_xx`, `U_yy`; nine-point central for `U_xy` (equivalent to IT seven-point stencil under admissibility); degenerate transport at `y=0`
4. **LCP handling**: Ikonen-Toivanen operator splitting (project onto obstacle each step)
5. **Boundary extraction**: Quadratic-contact rule using `√w` linear extrapolation

## Installation

```bash
pip install numpy scipy matplotlib pandas
# Optional for acceleration:
pip install numba
```

## Usage

### Quick validation (reduced grid, ~5-15 min)
```bash
python exp/exp1_boundary_surface.py --fast
python exp/exp2_gamma_identity.py --fast
```

### Full paper resolution (705×601 grid, several hours)
```bash
python exp/exp1_boundary_surface.py
python exp/exp2_gamma_identity.py
```

### Tests
```bash
pytest tests/ -v
```

## Results

See `results/RESULTS.md` for numerical outputs and `results/` for figures:

- `fig2_set_a_surface.png` — 3D boundary surface for Set A (analogous to Figure 2)
- `fig2_set_b_surface.png` — 3D boundary surface for Set B
- `fig3_set_a_slices.png` — Boundary slices vs τ and y for Set A (analogous to Figure 3)
- `fig3_set_ab_comparison.png` — Set A vs Set B comparison
- `fig_gamma_comparison.png` — FD vs analytical gamma (Experiment 2)
- `table2_gamma_identity.csv` — Reproduced Table 2

## Key Findings

1. The extracted boundary `b(τ,y)` is **non-increasing** in both τ and y.
2. The boundary remains **finite and positive** at the degenerate axis `y=0`.
3. As `τ→0`, `b(τ,y) → K` uniformly over bounded y.
4. Set B (Feller violated) shows qualitatively the same smooth boundary as Set A.
5. The boundary gamma identity is verified numerically with discrepancies typically <2%.

## Implementation Notes

- The x-domain is chosen as `[x_min, x_min + 704·Δx] = [-2.50, 1.86]`, not specified in the paper.
- The mixed-derivative stencil uses the nine-point central difference, equivalent to the IT seven-point stencil under the verified admissibility condition.
- Admissibility: Set A: `0.00130 ≤ 0.002 ≤ 0.00266`; Set B: `0.00260 ≤ 0.004 ≤ 0.00531`.
- The LU factorization of the system matrix is computed once and reused across all time steps (the spatial operator is stationary in τ).
