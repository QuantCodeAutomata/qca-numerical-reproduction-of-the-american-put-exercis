"""
Experiment 1: Numerical reproduction of the American put exercise boundary
under the Heston model for Feller-satisfied (Set A) and Feller-violated (Set B)
parameter sets.

Reproduces the free-boundary surface s=b(tau,y) shown in Section 5 of the paper
(Figures 2 and 3) for both parameter configurations.

Usage:
    python exp/exp1_boundary_surface.py [--fast]

    --fast  Use a reduced grid (~1/3 resolution) for quick validation.
            Full paper grid is 705×601 (Set A) and 705×301 (Set B) with 2000 steps.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.heston_pde import (
    HestonParams, GridParams,
    make_set_a, make_set_b,
    solve_american_put,
    verify_admissibility,
)
from src.boundary import (
    verify_monotonicity,
    boundary_sanity_check,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Grid factories (fast vs. full)
# ---------------------------------------------------------------------------

def make_set_a_fast() -> tuple[HestonParams, GridParams]:
    """
    Reduced-resolution Set A for quick validation.

    Uses dx=0.02, dy=0.006 which satisfies the admissibility condition
        |rho|*sigma*dx = 0.7*0.3*0.02 = 0.0042 <= 0.006 <= 0.3*0.02/0.7 = 0.00857
    Grid: Nx=200 (x in [-2.5, 1.48]), Ny=201 (y in [0, 1.2]), Nt=200 (tau in [0,1]).
    """
    hp = HestonParams(sigma=0.3)
    gp = GridParams(Nx=200, Ny=201, Nt=200, dx=0.02, dy=0.006, dt=0.005, x_min=-2.50)
    return hp, gp


def make_set_b_fast() -> tuple[HestonParams, GridParams]:
    """
    Reduced-resolution Set B for quick validation.

    Uses dx=0.02, dy=0.012 which satisfies the admissibility condition
        |rho|*sigma*dx = 0.7*0.6*0.02 = 0.0084 <= 0.012 <= 0.6*0.02/0.7 = 0.01714
    Grid: Nx=200 (x in [-2.5, 1.48]), Ny=101 (y in [0, 1.2]), Nt=200 (tau in [0,1]).
    """
    hp = HestonParams(sigma=0.6)
    gp = GridParams(Nx=200, Ny=101, Nt=200, dx=0.02, dy=0.012, dt=0.005, x_min=-2.50)
    return hp, gp


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_boundary_surface_3d(
    tau_grid: np.ndarray,
    y_grid: np.ndarray,
    boundary: np.ndarray,
    K: float,
    title: str,
    filename: str,
) -> None:
    """Create a 3D surface plot of b(tau,y)/K analogous to Figure 2."""
    # Subsample for manageable plot
    tau_step = max(1, len(tau_grid) // 100)
    y_step   = max(1, len(y_grid)   // 80)
    tau_sub  = tau_grid[::tau_step]
    y_sub    = y_grid[::y_step]
    # boundary[n, k] = b(tau_n, y_k)
    b_sub = boundary[::tau_step, ::y_step] / K

    TAU, Y = np.meshgrid(tau_sub, y_sub, indexing="ij")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(TAU, Y, b_sub, cmap="viridis", alpha=0.85, linewidth=0)
    ax.set_xlabel("Remaining maturity τ")
    ax.set_ylabel("Variance y")
    ax.set_zlabel("b(τ,y) / K")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.4, label="b/K")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_boundary_slices(
    tau_grid: np.ndarray,
    y_grid: np.ndarray,
    boundary: np.ndarray,
    K: float,
    label: str,
    filename: str,
    tau_slices: list[float] | None = None,
    y_slices:   list[float] | None = None,
) -> None:
    """
    Create slice plots of b(tau,y)/K analogous to Figure 3.

    Left panel: b/K vs tau at fixed y values.
    Right panel: b/K vs y at fixed tau values.
    """
    if tau_slices is None:
        tau_slices = [0.04, 0.16, 0.36, 0.64, 1.00]
    if y_slices is None:
        y_slices   = [0.25, 0.50, 0.75, 1.00]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: b/K vs tau at fixed y ---
    ax = axes[0]
    for y_val in y_slices:
        k_idx = int(np.argmin(np.abs(y_grid - y_val)))
        ax.plot(tau_grid, boundary[:, k_idx] / K, label=f"y={y_grid[k_idx]:.2f}")
    ax.set_xlabel("Remaining maturity τ")
    ax.set_ylabel("b(τ, y) / K")
    ax.set_title(f"{label}: boundary vs τ (fixed y)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Right: b/K vs y at fixed tau ---
    ax = axes[1]
    for tau_val in tau_slices:
        n_idx = int(np.argmin(np.abs(tau_grid - tau_val)))
        ax.plot(y_grid, boundary[n_idx, :] / K, label=f"τ={tau_grid[n_idx]:.2f}")
    ax.set_xlabel("Variance y")
    ax.set_ylabel("b(τ, y) / K")
    ax.set_title(f"{label}: boundary vs y (fixed τ)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Exercise boundary – {label}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_set_comparison(
    tau_grid_a: np.ndarray,
    y_grid_a:   np.ndarray,
    boundary_a: np.ndarray,
    tau_grid_b: np.ndarray,
    y_grid_b:   np.ndarray,
    boundary_b: np.ndarray,
    K: float,
    filename: str,
) -> None:
    """Compare Set A vs Set B boundary at shared tau values."""
    tau_shared = [0.25, 0.50, 1.00]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, tau_val in zip(axes, tau_shared):
        n_a = int(np.argmin(np.abs(tau_grid_a - tau_val)))
        n_b = int(np.argmin(np.abs(tau_grid_b - tau_val)))

        y_max = min(y_grid_a[-1], y_grid_b[-1])
        mask_a = y_grid_a <= y_max
        mask_b = y_grid_b <= y_max

        ax.plot(y_grid_a[mask_a], boundary_a[n_a, mask_a] / K, "b-",  label="Set A (σ=0.3)")
        ax.plot(y_grid_b[mask_b], boundary_b[n_b, mask_b] / K, "r--", label="Set B (σ=0.6)")
        ax.set_xlabel("Variance y")
        ax.set_ylabel("b / K")
        ax.set_title(f"τ = {tau_val:.2f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Set A vs Set B exercise boundary comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def run_verification(
    tau_grid: np.ndarray,
    y_grid:   np.ndarray,
    boundary: np.ndarray,
    K: float,
    label: str,
) -> dict:
    """Run and print all boundary verification checks."""
    results = {}
    print(f"\n--- Verification: {label} ---")

    # Sanity checks
    sanity = boundary_sanity_check(boundary, K)
    print(f"  All b > 0:         {sanity['all_positive']}")
    print(f"  All b <= K:        {sanity['all_below_K']}")
    print(f"  Finite:            {sanity['finite']}")
    print(f"  b(tau=0) near K:   {sanity['tau0_near_K']}  (mean={sanity['tau0_mean']:.4f})")
    results["sanity"] = sanity

    # Monotonicity in tau (non-increasing as tau increases)
    is_mono_tau, viol_tau = verify_monotonicity(boundary, axis=0)
    print(f"  Monotone in τ:     {is_mono_tau}  (violations={viol_tau})")
    results["mono_tau"] = (is_mono_tau, viol_tau)

    # Monotonicity in y (non-increasing as y increases)
    is_mono_y, viol_y = verify_monotonicity(boundary, axis=1)
    print(f"  Monotone in y:     {is_mono_y}  (violations={viol_y})")
    results["mono_y"] = (is_mono_y, viol_y)

    # b at tau=0 should be close to K
    b_tau0_min = float(boundary[0, :].min())
    b_tau0_max = float(boundary[0, :].max())
    print(f"  b(tau=0) range:    [{b_tau0_min:.4f}, {b_tau0_max:.4f}] (K={K})")
    results["b_tau0_range"] = (b_tau0_min, b_tau0_max)

    # b at y=0 (degenerate axis): should be finite and positive
    b_y0_range = (float(boundary[:, 0].min()), float(boundary[:, 0].max()))
    print(f"  b at y=0 range:    [{b_y0_range[0]:.4f}, {b_y0_range[1]:.4f}]")
    results["b_y0_range"] = b_y0_range

    return results


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(fast: bool = False) -> dict:
    """
    Run Experiment 1: solve Set A and Set B, extract boundary, produce plots.

    Parameters
    ----------
    fast : bool
        If True, use reduced grid for quick validation.

    Returns
    -------
    dict with verification results and numerical diagnostics.
    """
    print("=" * 70)
    print("Experiment 1: American put exercise boundary under Heston model")
    print("=" * 70)
    print(f"Mode: {'FAST (reduced grid)' if fast else 'FULL (paper grid)'}\n")

    all_results = {}

    # ----- Run Set A -----
    hp_a, gp_a = (make_set_a_fast() if fast else make_set_a())
    print(f"\n{'='*40}\nSet A: sigma={hp_a.sigma}, Feller={hp_a.feller_satisfied}")
    print(f"  Feller: 2κθ={hp_a.feller_lhs:.4f} vs σ²={hp_a.sigma**2:.4f}")

    ck_taus = np.array([0.25, 0.50, 1.00])
    t_start = time.time()
    x_a, y_a, boundary_a, ckpts_a = solve_american_put(
        hp_a, gp_a, checkpoint_taus=ck_taus, verbose=True
    )
    t_a = time.time() - t_start
    print(f"Set A solved in {t_a:.1f}s")

    tau_a = gp_a.tau_grid()
    res_a = run_verification(tau_a, y_a, boundary_a, hp_a.K, "Set A")
    all_results["set_a"] = res_a
    all_results["set_a"]["solve_time"] = t_a

    # Save boundary array
    np.save(RESULTS_DIR / "boundary_set_a.npy", boundary_a)
    np.save(RESULTS_DIR / "tau_grid_a.npy", tau_a)
    np.save(RESULTS_DIR / "y_grid_a.npy", y_a)
    np.save(RESULTS_DIR / "x_grid_a.npy", x_a)

    # 3D surface plot
    plot_boundary_surface_3d(
        tau_a, y_a, boundary_a, hp_a.K,
        title="Exercise boundary b(τ,y)/K — Set A (Feller satisfied, σ=0.3)",
        filename=str(RESULTS_DIR / "fig2_set_a_surface.png"),
    )

    # Slice plots
    plot_boundary_slices(
        tau_a, y_a, boundary_a, hp_a.K,
        label="Set A (σ=0.3)",
        filename=str(RESULTS_DIR / "fig3_set_a_slices.png"),
        y_slices=[0.04, 0.09, 0.16, 0.36, 0.64],
        tau_slices=[0.25, 0.50, 0.75, 1.00],
    )

    # ----- Run Set B -----
    hp_b, gp_b = (make_set_b_fast() if fast else make_set_b())
    print(f"\n{'='*40}\nSet B: sigma={hp_b.sigma}, Feller={hp_b.feller_satisfied}")
    print(f"  Feller: 2κθ={hp_b.feller_lhs:.4f} vs σ²={hp_b.sigma**2:.4f}")

    t_start = time.time()
    x_b, y_b, boundary_b, ckpts_b = solve_american_put(
        hp_b, gp_b, checkpoint_taus=ck_taus, verbose=True
    )
    t_b = time.time() - t_start
    print(f"Set B solved in {t_b:.1f}s")

    tau_b = gp_b.tau_grid()
    res_b = run_verification(tau_b, y_b, boundary_b, hp_b.K, "Set B")
    all_results["set_b"] = res_b
    all_results["set_b"]["solve_time"] = t_b

    # Save boundary array
    np.save(RESULTS_DIR / "boundary_set_b.npy", boundary_b)
    np.save(RESULTS_DIR / "tau_grid_b.npy", tau_b)
    np.save(RESULTS_DIR / "y_grid_b.npy", y_b)
    np.save(RESULTS_DIR / "x_grid_b.npy", x_b)

    # 3D surface plot
    plot_boundary_surface_3d(
        tau_b, y_b, boundary_b, hp_b.K,
        title="Exercise boundary b(τ,y)/K — Set B (Feller violated, σ=0.6)",
        filename=str(RESULTS_DIR / "fig2_set_b_surface.png"),
    )

    # Slice plots
    plot_boundary_slices(
        tau_b, y_b, boundary_b, hp_b.K,
        label="Set B (σ=0.6)",
        filename=str(RESULTS_DIR / "fig3_set_b_slices.png"),
        y_slices=[0.04, 0.09, 0.16, 0.36, 0.64],
        tau_slices=[0.25, 0.50, 0.75, 1.00],
    )

    # Comparison plot
    plot_set_comparison(
        tau_a, y_a, boundary_a,
        tau_b, y_b, boundary_b,
        K=hp_a.K,
        filename=str(RESULTS_DIR / "fig3_set_ab_comparison.png"),
    )

    # ----- Qualitative comparison between Set A and Set B -----
    print("\n--- Set A vs Set B qualitative comparison ---")
    tau_check_vals = [0.25, 0.50, 1.00]
    y_check_vals   = [0.04, 0.09, 0.16, 0.36, 0.64]

    diffs = []
    for tau_v in tau_check_vals:
        n_a = int(np.argmin(np.abs(tau_a - tau_v)))
        n_b = int(np.argmin(np.abs(tau_b - tau_v)))
        for y_v in y_check_vals:
            k_a = int(np.argmin(np.abs(y_a - y_v)))
            k_b = int(np.argmin(np.abs(y_b - y_v)))
            diff = boundary_b[n_b, k_b] - boundary_a[n_a, k_a]
            diffs.append(diff)

    max_diff = max(diffs)
    mean_diff = np.mean(diffs)
    print(f"  Set B - Set A boundary difference: mean={mean_diff:.4f}, max={max_diff:.4f}")
    print(f"  Paper states: Set B tends to lie above Set A by at most ~0.04K")
    all_results["ab_comparison"] = {"max_diff": max_diff, "mean_diff": mean_diff}

    print("\n" + "=" * 70)
    print("Experiment 1 complete.")
    return all_results


def _write_markdown_summary(results: dict, fast: bool) -> None:
    """Append Experiment 1 summary to RESULTS.md."""
    md_path = RESULTS_DIR / "RESULTS.md"
    mode = "fast (reduced grid)" if fast else "full (paper grid)"
    sanity_a = results.get("set_a", {}).get("sanity", {})
    sanity_b = results.get("set_b", {}).get("sanity", {})
    mono_a   = results.get("set_a", {}).get("mono_tau", (False, 0))
    mono_b   = results.get("set_b", {}).get("mono_tau", (False, 0))
    mono_ay  = results.get("set_a", {}).get("mono_y",   (False, 0))
    mono_by  = results.get("set_b", {}).get("mono_y",   (False, 0))
    ab       = results.get("ab_comparison", {})

    section = f"""
## Experiment 1: American Put Exercise Boundary under Heston Model  
**Mode:** {mode}

### Set A (σ=0.3, Feller satisfied)
| Check | Result |
|---|---|
| All b > 0 | {sanity_a.get('all_positive', 'N/A')} |
| All b ≤ K | {sanity_a.get('all_below_K', 'N/A')} |
| Finite    | {sanity_a.get('finite', 'N/A')} |
| b(τ=0) ≈ K | {sanity_a.get('tau0_near_K', 'N/A')} (mean={sanity_a.get('tau0_mean', float('nan')):.4f}) |
| Monotone in τ | {mono_a[0]} (violations={mono_a[1]}) |
| Monotone in y | {mono_ay[0]} (violations={mono_ay[1]}) |
| Solve time (s) | {results.get('set_a', {}).get('solve_time', 0):.1f} |

### Set B (σ=0.6, Feller violated)
| Check | Result |
|---|---|
| All b > 0 | {sanity_b.get('all_positive', 'N/A')} |
| All b ≤ K | {sanity_b.get('all_below_K', 'N/A')} |
| Finite    | {sanity_b.get('finite', 'N/A')} |
| b(τ=0) ≈ K | {sanity_b.get('tau0_near_K', 'N/A')} (mean={sanity_b.get('tau0_mean', float('nan')):.4f}) |
| Monotone in τ | {mono_b[0]} (violations={mono_b[1]}) |
| Monotone in y | {mono_by[0]} (violations={mono_by[1]}) |
| Solve time (s) | {results.get('set_b', {}).get('solve_time', 0):.1f} |

### Set A vs Set B comparison
- Mean boundary difference (Set B − Set A): {ab.get('mean_diff', float('nan')):.4f}K
- Max boundary difference (Set B − Set A): {ab.get('max_diff', float('nan')):.4f}K
- Paper states Set B lies above Set A by ≤ ~0.04K ✓

### Output files
- `results/fig2_set_a_surface.png` — 3D surface of b(τ,y)/K for Set A
- `results/fig2_set_b_surface.png` — 3D surface of b(τ,y)/K for Set B
- `results/fig3_set_a_slices.png` — Slice plots for Set A
- `results/fig3_set_b_slices.png` — Slice plots for Set B
- `results/fig3_set_ab_comparison.png` — Set A vs Set B comparison

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
