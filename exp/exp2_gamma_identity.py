"""
Experiment 2: Numerical verification of the continuation-side boundary gamma
identity for the American put exercise boundary under the Heston model.

Reproduces a Table 2-style comparison at selected (tau, y) points for the
Set A parameter configuration (sigma=0.3, Feller condition satisfied).

Usage:
    python exp/exp2_gamma_identity.py [--fast]

    --fast  Use a reduced grid for quick validation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.heston_pde import HestonParams, GridParams, make_set_a, solve_american_put
from src.boundary import extract_boundary_row, estimate_boundary_slope, verify_monotonicity
from src.gamma_identity import (
    compute_gamma_table,
    validate_gamma_positivity,
    gamma_formula,
    estimate_wxx_at_boundary,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# Paper Table 2 reference values (Set A, for comparison)
TABLE2_REFERENCE = {
    (0.25, 0.04): {"b_K": 0.8783, "by_K": -1.730, "Pss_fd": 4.125, "Pss_form": 4.028},
    (0.25, 0.09): {"b_K": 0.8098, "by_K": -1.179, "Pss_fd": 1.955, "Pss_form": 1.977},
    (0.25, 0.16): {"b_K": 0.7411, "by_K": -0.852, "Pss_fd": 1.238, "Pss_form": 1.259},
    (0.25, 0.36): {"b_K": 0.6135, "by_K": -0.496, "Pss_fd": 0.761, "Pss_form": 0.774},
    (0.25, 0.64): {"b_K": 0.5047, "by_K": -0.310, "Pss_fd": 0.627, "Pss_form": 0.631},
    (0.50, 0.04): {"b_K": 0.8520, "by_K": -1.785, "Pss_fd": 4.549, "Pss_form": 4.407},
    (0.50, 0.09): {"b_K": 0.7797, "by_K": -1.256, "Pss_fd": 2.264, "Pss_form": 2.258},
    (0.50, 0.16): {"b_K": 0.7057, "by_K": -0.917, "Pss_fd": 1.481, "Pss_form": 1.485},
    (0.50, 0.36): {"b_K": 0.5689, "by_K": -0.527, "Pss_fd": 0.958, "Pss_form": 0.964},
    (0.50, 0.64): {"b_K": 0.4539, "by_K": -0.323, "Pss_fd": 0.829, "Pss_form": 0.831},
    (1.00, 0.04): {"b_K": 0.8200, "by_K": -1.714, "Pss_fd": 4.977, "Pss_form": 4.848},
    (1.00, 0.09): {"b_K": 0.7496, "by_K": -1.233, "Pss_fd": 2.535, "Pss_form": 2.505},
    (1.00, 0.16): {"b_K": 0.6766, "by_K": -0.915, "Pss_fd": 1.676, "Pss_form": 1.669},
    (1.00, 0.36): {"b_K": 0.5390, "by_K": -0.533, "Pss_fd": 1.112, "Pss_form": 1.115},
    (1.00, 0.64): {"b_K": 0.4234, "by_K": -0.324, "Pss_fd": 0.992, "Pss_form": 0.990},
}

TAU_TARGETS = [0.25, 0.50, 1.00]
Y_TARGETS   = [0.04, 0.09, 0.16, 0.36, 0.64]


def make_set_a_fast() -> tuple[HestonParams, GridParams]:
    """
    Reduced-resolution Set A for quick validation.

    Uses dx=0.02, dy=0.006 which satisfies the admissibility condition
        |rho|*sigma*dx = 0.7*0.3*0.02 = 0.0042 <= 0.006 <= 0.3*0.02/0.7 = 0.00857
    Grid: Nx=200 (x in [-2.5, 1.48]), Ny=201 (y in [0, 1.2]), Nt=200 (tau in [0,1]).

    y-target proximity with dy=0.006:
        y=0.04 -> k=7  (y_grid=0.042)
        y=0.09 -> k=15 (y_grid=0.090)
        y=0.16 -> k=27 (y_grid=0.162)
        y=0.36 -> k=60 (y_grid=0.360)
        y=0.64 -> k=107 (y_grid=0.642)
    All within [0, 1.2].
    """
    hp = HestonParams(sigma=0.3)
    gp = GridParams(Nx=200, Ny=201, Nt=200, dx=0.02, dy=0.006, dt=0.005, x_min=-2.50)
    return hp, gp


# ---------------------------------------------------------------------------
# Sensitivity checks
# ---------------------------------------------------------------------------

def sensitivity_x_domain(
    hp: HestonParams,
    gp: GridParams,
    tau_targets: list[float],
    y_targets:   list[float],
    n_variations: int = 2,
) -> pd.DataFrame:
    """
    Vary x_min by ±0.5 to check sensitivity of interior boundary.

    Returns DataFrame showing b/K across x_min variations.
    """
    records = []
    for delta in np.linspace(-0.5, 0.5, 2 * n_variations + 1):
        gp_var = GridParams(
            Nx=gp.Nx, Ny=gp.Ny, Nt=gp.Nt,
            dx=gp.dx, dy=gp.dy, dt=gp.dt,
            x_min=gp.x_min + delta,
        )
        _, _, boundary, _ = solve_american_put(
            hp, gp_var,
            checkpoint_taus=np.array(tau_targets),
            verbose=False,
        )
        y_grid = gp_var.y_grid()
        tau_grid = gp_var.tau_grid()
        for tau_v in tau_targets:
            n_idx = int(np.argmin(np.abs(tau_grid - tau_v)))
            for y_v in y_targets:
                k_idx = int(np.argmin(np.abs(y_grid - y_v)))
                records.append({
                    "x_min_delta": delta,
                    "tau": tau_v,
                    "y": y_v,
                    "b_over_K": boundary[n_idx, k_idx] / hp.K,
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Table plotting
# ---------------------------------------------------------------------------

def plot_gamma_table(
    table: pd.DataFrame,
    filename: str,
) -> None:
    """Visualize Table 2 comparisons."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, tau_v in zip(axes, TAU_TARGETS):
        sub = table[table["tau"] == tau_v].copy()
        y_vals = sub["y"].values
        ax.plot(y_vals, sub["Pss_fd"].values,      "bo-", label="P_ss FD",      markersize=6)
        ax.plot(y_vals, sub["Pss_formula"].values,  "r^--", label="P_ss formula", markersize=6)
        ax.set_xlabel("Variance y")
        ax.set_ylabel("P_ss")
        ax.set_title(f"τ = {tau_v:.2f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Boundary gamma identity: FD vs analytical formula (Set A)", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_pct_diff(table: pd.DataFrame, filename: str) -> None:
    """Plot signed percent difference across (tau, y) pairs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for tau_v in TAU_TARGETS:
        sub = table[table["tau"] == tau_v].copy()
        ax.plot(sub["y"].values, sub["pct_diff"].values, "o-", label=f"τ={tau_v:.2f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Variance y")
    ax.set_ylabel("Signed percent difference (%)")
    ax.set_title("Gamma identity: (P_ss^fd / P_ss^formula − 1) × 100%")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_boundary_comparison(
    table: pd.DataFrame,
    filename: str,
) -> None:
    """Plot our b/K vs paper reference at the table points."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"0.25": "blue", "0.50": "green", "1.00": "red"}
    for tau_v in TAU_TARGETS:
        sub   = table[table["tau"] == tau_v].copy()
        y_arr = sub["y"].values
        b_arr = sub["b_over_K"].values
        ref_b = [TABLE2_REFERENCE.get((tau_v, y_v), {}).get("b_K", np.nan) for y_v in y_arr]
        c = colors.get(f"{tau_v:.2f}", "black")
        ax.plot(y_arr, b_arr,  color=c, ls="-",  marker="o", label=f"τ={tau_v:.2f} (ours)")
        ax.plot(y_arr, ref_b,  color=c, ls="--", marker="x", label=f"τ={tau_v:.2f} (paper)")
    ax.set_xlabel("Variance y")
    ax.set_ylabel("b / K")
    ax.set_title("Boundary b/K: ours vs paper Table 2 (Set A)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(fast: bool = False) -> dict:
    """
    Run Experiment 2: verify the boundary gamma identity for Set A.

    Returns dict with the gamma table DataFrame and validation results.
    """
    print("=" * 70)
    print("Experiment 2: Boundary gamma identity verification (Set A)")
    print("=" * 70)
    print(f"Mode: {'FAST (reduced grid)' if fast else 'FULL (paper grid)'}\n")

    hp, gp = (make_set_a_fast() if fast else make_set_a())
    print(f"Heston Set A: K={hp.K}, r={hp.r}, q={hp.q}, kappa={hp.kappa}, "
          f"theta={hp.theta}, rho={hp.rho}, sigma={hp.sigma}, T={hp.T}")
    print(f"Grid: Nx={gp.Nx}, Ny={gp.Ny}, Nt={gp.Nt}, "
          f"dx={gp.dx}, dy={gp.dy}, dt={gp.dt}, x_min={gp.x_min}\n")

    # ----- Solve PDE -----
    t_start = time.time()
    x_grid, y_grid, boundary, checkpoints = solve_american_put(
        hp, gp,
        checkpoint_taus=np.array(TAU_TARGETS),
        verbose=True,
    )
    solve_time = time.time() - t_start
    print(f"\nPDE solved in {solve_time:.1f}s")
    print(f"Checkpoints available at tau = {sorted(checkpoints.keys())}")

    # ----- Build b_checkpoints (boundary at each checkpoint tau) -----
    w_checkpoints: dict[float, np.ndarray] = {}
    b_checkpoints: dict[float, np.ndarray] = {}
    tau_grid = gp.tau_grid()

    for tau_v in TAU_TARGETS:
        # Find closest checkpoint
        ck_tau = min(checkpoints.keys(), key=lambda t: abs(t - tau_v))
        ck_data = checkpoints[ck_tau]
        w_checkpoints[tau_v] = ck_data["w"]              # shape (Nx, Ny)
        # Extract boundary from solution at this tau
        n_idx = int(np.argmin(np.abs(tau_grid - tau_v)))
        b_checkpoints[tau_v] = boundary[n_idx, :]         # shape (Ny,)

    # ----- Compute gamma table -----
    print("\nComputing gamma identity table ...")
    table = compute_gamma_table(
        w_checkpoints=w_checkpoints,
        b_checkpoints=b_checkpoints,
        x_grid=x_grid,
        y_grid=y_grid,
        tau_targets=TAU_TARGETS,
        y_targets=Y_TARGETS,
        r=hp.r, q=hp.q, K=hp.K, rho=hp.rho, sigma=hp.sigma,
        dx=gp.dx,
        window=0.1,
    )

    # ----- Validate theoretical properties -----
    validation = validate_gamma_positivity(table)
    print("\nTheoretical validation:")
    for k, v in validation.items():
        print(f"  {k}: {v}")

    # ----- Print table -----
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print("\n--- Reproduced Table 2-style comparison ---")
    print(table.to_string(index=False))

    # ----- Compare against paper reference -----
    print("\n--- Comparison with paper Table 2 ---")
    print(f"{'τ':>5} {'y':>5} | {'b/K ours':>10} {'b/K ref':>10} | "
          f"{'Pss_fd':>8} {'Pss_ref':>8} | {'form':>8} {'form_ref':>8} | {'%diff':>7}")
    print("-" * 90)
    for _, row in table.iterrows():
        tau_v = float(row["tau"])
        y_v   = float(row["y"])
        ref   = TABLE2_REFERENCE.get((round(tau_v, 2), round(y_v, 2)), {})
        ref_b = ref.get("b_K",       np.nan)
        ref_fd = ref.get("Pss_fd",   np.nan)
        ref_fo = ref.get("Pss_form", np.nan)
        print(f"{tau_v:5.2f} {y_v:5.2f} | {row['b_over_K']:10.4f} {ref_b:10.4f} | "
              f"{row['Pss_fd']:8.3f} {ref_fd:8.3f} | "
              f"{row['Pss_formula']:8.3f} {ref_fo:8.3f} | "
              f"{row['pct_diff']:7.2f}%")

    # ----- Save outputs -----
    table.to_csv(RESULTS_DIR / "table2_gamma_identity.csv", index=False)
    print(f"\nTable saved to results/table2_gamma_identity.csv")

    plot_gamma_table(table, str(RESULTS_DIR / "fig_gamma_comparison.png"))
    plot_pct_diff(table, str(RESULTS_DIR / "fig_gamma_pct_diff.png"))
    plot_boundary_comparison(table, str(RESULTS_DIR / "fig_boundary_vs_paper.png"))

    # ----- Monotonicity of boundary -----
    is_mono_tau, viol_tau = verify_monotonicity(boundary, axis=0)
    is_mono_y,   viol_y   = verify_monotonicity(boundary, axis=1)
    print(f"\nMonotonicity (τ): {is_mono_tau} (violations={viol_tau})")
    print(f"Monotonicity (y): {is_mono_y}   (violations={viol_y})")

    return {
        "table": table,
        "validation": validation,
        "solve_time": solve_time,
        "mono_tau": (is_mono_tau, viol_tau),
        "mono_y":   (is_mono_y,   viol_y),
    }


def _write_markdown_summary(results: dict, fast: bool) -> None:
    """Append Experiment 2 summary to RESULTS.md."""
    md_path = RESULTS_DIR / "RESULTS.md"
    table: pd.DataFrame = results.get("table", pd.DataFrame())
    val: dict = results.get("validation", {})
    mode = "fast (reduced grid)" if fast else "full (paper grid)"

    # Build table string
    if not table.empty:
        tbl_md = table.to_markdown(index=False, floatfmt=".4f")
    else:
        tbl_md = "_No data_"

    section = f"""
## Experiment 2: Boundary Gamma Identity Verification (Set A)
**Mode:** {mode}  
**Solve time:** {results.get('solve_time', 0):.1f}s

### Reproduced Table 2

{tbl_md}

### Validation
| Check | Result |
|---|---|
| All P_ss^fd > 0 | {val.get('all_Pss_fd_positive', 'N/A')} |
| All P_ss^formula > 0 | {val.get('all_Pss_formula_positive', 'N/A')} |
| All b_y < 0 | {val.get('all_by_negative', 'N/A')} |
| All b < K | {val.get('all_b_below_K', 'N/A')} |
| Max |%diff| | {val.get('max_abs_pct_diff', float('nan')):.2f}% |
| Monotone in τ | {results.get('mono_tau', ('N/A', 0))[0]} (violations={results.get('mono_tau', (False, 0))[1]}) |
| Monotone in y | {results.get('mono_y', ('N/A', 0))[0]} (violations={results.get('mono_y', (False, 0))[1]}) |

### Output files
- `results/table2_gamma_identity.csv` — Full table
- `results/fig_gamma_comparison.png` — FD vs formula across y
- `results/fig_gamma_pct_diff.png` — Signed percent differences
- `results/fig_boundary_vs_paper.png` — Our b/K vs paper reference

"""
    with open(md_path, "a") as f:
        f.write(section)
    print(f"Summary appended to {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Use reduced grid for quick validation")
    args = parser.parse_args()

    results = run_experiment(fast=args.fast)
    _write_markdown_summary(results, fast=args.fast)
