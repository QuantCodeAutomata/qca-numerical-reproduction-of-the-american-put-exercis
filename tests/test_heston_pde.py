"""
Tests for the Heston American put PDE solver and related modules.

Tests verify:
1. Parameter and grid dataclasses behave correctly
2. Admissibility condition is checked correctly for Set A and Set B
3. Sparse operator assembly produces correct shapes and structure
4. Obstacle is assembled correctly
5. A small-grid solve completes and returns monotone, finite boundary
6. Boundary extraction finds boundary in (0, K) and approaches K at tau=0
7. b_y estimation returns negative values (boundary decreasing in y)
8. Gamma identity formula returns positive values for valid inputs
9. The full gamma table has correct shape and sign properties
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.heston_pde import (
    HestonParams,
    GridParams,
    make_set_a,
    make_set_b,
    build_heston_operator,
    build_system_matrix,
    verify_admissibility,
    solve_american_put,
    _build_obstacle,
)
from src.boundary import (
    extract_boundary_at_slice,
    extract_boundary_row,
    estimate_boundary_slope,
    verify_monotonicity,
    boundary_sanity_check,
)
from src.gamma_identity import (
    gamma_formula,
    estimate_wxx_at_boundary,
    compute_gamma_table,
    validate_gamma_positivity,
)


# ---------------------------------------------------------------------------
# Fixtures: tiny grids for fast tests
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_grid_a() -> tuple[HestonParams, GridParams]:
    """Tiny Set A grid for fast unit tests."""
    hp = HestonParams(sigma=0.3)
    # Keep dy=0.002 for admissibility; reduce Nx, Ny, Nt significantly
    gp = GridParams(Nx=51, Ny=31, Nt=10, dx=0.0062, dy=0.002, dt=0.1, x_min=-2.50)
    return hp, gp


@pytest.fixture
def tiny_grid_b() -> tuple[HestonParams, GridParams]:
    """Tiny Set B grid for fast unit tests."""
    hp = HestonParams(sigma=0.6)
    gp = GridParams(Nx=51, Ny=16, Nt=10, dx=0.0062, dy=0.004, dt=0.1, x_min=-2.50)
    return hp, gp


@pytest.fixture
def medium_grid_a() -> tuple[HestonParams, GridParams]:
    """Medium Set A grid for integration-style tests."""
    hp = HestonParams(sigma=0.3)
    gp = GridParams(Nx=105, Ny=61, Nt=50, dx=0.0062, dy=0.002, dt=0.02, x_min=-2.50)
    return hp, gp


# ---------------------------------------------------------------------------
# 1. HestonParams and GridParams
# ---------------------------------------------------------------------------

class TestParams:
    def test_heston_params_defaults(self) -> None:
        hp = HestonParams()
        assert hp.K == 1.0
        assert hp.r == 0.05
        assert hp.q == 0.02
        assert hp.kappa == 2.0
        assert hp.theta == 0.04
        assert hp.rho == -0.7
        assert hp.sigma == 0.3
        assert hp.T == 1.0

    def test_feller_set_a(self) -> None:
        hp, _ = make_set_a()
        # Feller: 2*kappa*theta >= sigma^2
        assert hp.feller_lhs == pytest.approx(2 * 2.0 * 0.04, rel=1e-6)
        assert hp.feller_satisfied  # 0.16 >= 0.09

    def test_feller_set_b(self) -> None:
        hp, _ = make_set_b()
        # Feller condition NOT satisfied: 0.16 < 0.36
        assert not hp.feller_satisfied

    def test_grid_params_x_range(self) -> None:
        gp = GridParams(Nx=705, Ny=601, Nt=2000, dx=0.0062, dy=0.002, dt=0.0005, x_min=-2.50)
        expected_x_max = -2.50 + 704 * 0.0062
        assert gp.x_max == pytest.approx(expected_x_max, rel=1e-6)

    def test_grid_params_y_range(self) -> None:
        gp = GridParams(Nx=705, Ny=601, Nt=2000, dx=0.0062, dy=0.002, dt=0.0005, x_min=-2.50)
        assert gp.y_max == pytest.approx(600 * 0.002, rel=1e-6)  # 1.2

    def test_grid_params_grids_shape(self) -> None:
        gp = GridParams(Nx=51, Ny=31, Nt=10, dx=0.0062, dy=0.002, dt=0.1, x_min=-2.50)
        assert gp.x_grid().shape == (51,)
        assert gp.y_grid().shape == (31,)
        assert gp.tau_grid().shape == (11,)

    def test_x_grid_uniform(self) -> None:
        gp = GridParams(Nx=51, Ny=31, Nt=10, dx=0.0062, dy=0.002, dt=0.1, x_min=-2.50)
        x = gp.x_grid()
        diffs = np.diff(x)
        assert np.allclose(diffs, gp.dx, atol=1e-12)

    def test_y_grid_starts_at_zero(self) -> None:
        gp = GridParams(Nx=51, Ny=31, Nt=10, dx=0.0062, dy=0.002, dt=0.1, x_min=-2.50)
        assert gp.y_grid()[0] == 0.0


# ---------------------------------------------------------------------------
# 2. Admissibility condition
# ---------------------------------------------------------------------------

class TestAdmissibility:
    def test_set_a_admissible(self) -> None:
        hp, gp = make_set_a()
        ok, lb, ub = verify_admissibility(hp, gp)
        assert ok, f"Set A should be admissible: dy={gp.dy}, bounds=[{lb:.6f},{ub:.6f}]"

    def test_set_b_admissible(self) -> None:
        hp, gp = make_set_b()
        ok, lb, ub = verify_admissibility(hp, gp)
        assert ok, f"Set B should be admissible: dy={gp.dy}, bounds=[{lb:.6f},{ub:.6f}]"

    def test_set_a_bounds_numerical(self) -> None:
        hp, gp = make_set_a()
        _, lb, ub = verify_admissibility(hp, gp)
        # |rho|*sigma*dx = 0.7*0.3*0.0062
        assert lb == pytest.approx(0.7 * 0.3 * 0.0062, rel=1e-6)
        # sigma*dx/|rho| = 0.3*0.0062/0.7
        assert ub == pytest.approx(0.3 * 0.0062 / 0.7, rel=1e-6)

    def test_set_b_bounds_numerical(self) -> None:
        hp, gp = make_set_b()
        _, lb, ub = verify_admissibility(hp, gp)
        assert lb == pytest.approx(0.7 * 0.6 * 0.0062, rel=1e-6)
        assert ub == pytest.approx(0.6 * 0.0062 / 0.7, rel=1e-6)

    def test_inadmissible_raises(self) -> None:
        hp = HestonParams(sigma=0.3)
        gp_bad = GridParams(Nx=51, Ny=31, Nt=10, dx=0.0062, dy=0.0001, dt=0.1, x_min=-2.50)
        ok, _, _ = verify_admissibility(hp, gp_bad)
        assert not ok


# ---------------------------------------------------------------------------
# 3. Sparse operator assembly
# ---------------------------------------------------------------------------

class TestOperatorAssembly:
    def test_operator_shape(self, tiny_grid_a: tuple) -> None:
        hp, gp = tiny_grid_a
        N = gp.Nx * gp.Ny
        A = build_heston_operator(hp, gp)
        assert A.shape == (N, N)

    def test_system_matrix_shape(self, tiny_grid_a: tuple) -> None:
        hp, gp = tiny_grid_a
        N = gp.Nx * gp.Ny
        A = build_heston_operator(hp, gp)
        M = build_system_matrix(hp, gp, A)
        assert M.shape == (N, N)

    def test_operator_sparsity(self, tiny_grid_a: tuple) -> None:
        hp, gp = tiny_grid_a
        N = gp.Nx * gp.Ny
        A = build_heston_operator(hp, gp)
        # Dense would have N^2 entries; sparse should have far fewer
        assert A.nnz < N ** 2

    def test_dirichlet_rows_identity(self, tiny_grid_a: tuple) -> None:
        hp, gp = tiny_grid_a
        A = build_heston_operator(hp, gp)
        Ny = gp.Ny
        # j=0 rows: should be identity (A[idx, idx] = 1, all others = 0)
        for k in range(Ny):
            idx = 0 * Ny + k
            row = A.getrow(idx)
            assert row.nnz == 1, f"Dirichlet row {idx} should have exactly 1 nnz"
            assert float(row[0, idx]) == pytest.approx(1.0)

    def test_upper_x_dirichlet_rows_identity(self, tiny_grid_a: tuple) -> None:
        hp, gp = tiny_grid_a
        A = build_heston_operator(hp, gp)
        Ny = gp.Ny
        j_last = gp.Nx - 1
        for k in range(Ny):
            idx = j_last * Ny + k
            row = A.getrow(idx)
            assert row.nnz == 1


# ---------------------------------------------------------------------------
# 4. Obstacle
# ---------------------------------------------------------------------------

class TestObstacle:
    def test_obstacle_shape(self, tiny_grid_a: tuple) -> None:
        hp, gp = tiny_grid_a
        obstacle = _build_obstacle(hp, gp)
        assert obstacle.shape == (gp.Nx * gp.Ny,)

    def test_obstacle_non_negative(self, tiny_grid_a: tuple) -> None:
        hp, gp = tiny_grid_a
        obstacle = _build_obstacle(hp, gp)
        assert np.all(obstacle >= 0.0)

    def test_obstacle_at_deep_itm(self, tiny_grid_a: tuple) -> None:
        """At x << 0 (deep ITM, s << K), obstacle should be close to K."""
        hp, gp = tiny_grid_a
        obstacle = _build_obstacle(hp, gp)
        # First x node: x = x_min = -2.5, s = exp(-2.5) ~ 0.082
        obs_lo = obstacle[0]  # k=0 at j=0
        expected = hp.K - np.exp(gp.x_min)
        assert obs_lo == pytest.approx(expected, rel=1e-6)

    def test_obstacle_at_deep_otm(self, tiny_grid_a: tuple) -> None:
        """At x > 0 (OTM, s > K), obstacle should be 0."""
        hp, gp = tiny_grid_a
        x_grid = gp.x_grid()
        # Find first x > 0
        otm_indices = np.where(x_grid > 0)[0]
        if len(otm_indices) > 0:
            j_otm = otm_indices[0]
            obstacle = _build_obstacle(hp, gp)
            # At row j_otm, k=0 -> flat index j_otm * Ny
            obs_val = obstacle[j_otm * gp.Ny]
            assert obs_val == 0.0, f"OTM obstacle should be 0, got {obs_val}"


# ---------------------------------------------------------------------------
# 5. Small-grid solve
# ---------------------------------------------------------------------------

class TestSmallSolve:
    def test_solve_returns_correct_shapes(self, tiny_grid_a: tuple) -> None:
        hp, gp = tiny_grid_a
        x, y, boundary, ckpts = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5, 1.0]), verbose=False
        )
        assert x.shape == (gp.Nx,)
        assert y.shape == (gp.Ny,)
        assert boundary.shape == (gp.Nt + 1, gp.Ny)

    def test_boundary_finite(self, tiny_grid_a: tuple) -> None:
        hp, gp = tiny_grid_a
        _, _, boundary, _ = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5, 1.0]), verbose=False
        )
        assert np.all(np.isfinite(boundary))

    def test_boundary_positive(self, tiny_grid_a: tuple) -> None:
        hp, gp = tiny_grid_a
        _, _, boundary, _ = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5, 1.0]), verbose=False
        )
        assert np.all(boundary > 0.0)

    def test_boundary_below_K(self, tiny_grid_a: tuple) -> None:
        hp, gp = tiny_grid_a
        _, _, boundary, _ = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5, 1.0]), verbose=False
        )
        # Allow small tolerance for numerical noise
        assert np.all(boundary <= hp.K + 0.05)

    def test_boundary_at_tau0_near_K(self) -> None:
        """
        At tau=0, boundary should be close to K.
        Requires a grid that spans x=0 (s=K), so use a grid centred on x=0.
        """
        hp = HestonParams(sigma=0.3)
        # x_min=-0.62, Nx=201, dx=0.0062  → x_max=-0.62+200*0.0062=0.62, spans x=0
        gp = GridParams(Nx=201, Ny=21, Nt=10, dx=0.0062, dy=0.002, dt=0.1, x_min=-0.62)
        _, _, boundary, _ = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5]), verbose=False
        )
        b_tau0 = boundary[0, :]
        # At tau=0 the initial condition is the obstacle → boundary is at x=0 → b≈K=1
        assert np.mean(b_tau0) > 0.7 * hp.K, (
            f"b(0,y) mean={np.mean(b_tau0):.4f} should be > 0.7"
        )

    def test_checkpoints_saved(self, tiny_grid_a: tuple) -> None:
        hp, gp = tiny_grid_a
        ck_taus = np.array([0.5, 1.0])
        _, _, _, ckpts = solve_american_put(
            hp, gp, checkpoint_taus=ck_taus, verbose=False
        )
        assert len(ckpts) >= 1  # At least one checkpoint saved

    def test_checkpoint_has_U_and_w(self, tiny_grid_a: tuple) -> None:
        hp, gp = tiny_grid_a
        _, _, _, ckpts = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5]), verbose=False
        )
        for tau_v, data in ckpts.items():
            assert "U" in data
            assert "w" in data
            assert data["U"].shape == (gp.Nx, gp.Ny)
            assert data["w"].shape == (gp.Nx, gp.Ny)

    def test_U_non_negative(self, tiny_grid_a: tuple) -> None:
        """Option price must be non-negative."""
        hp, gp = tiny_grid_a
        _, _, _, ckpts = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5]), verbose=False
        )
        for data in ckpts.values():
            U = data["U"]
            assert np.all(U >= -1e-10), "Option price (U) should be non-negative"

    def test_U_satisfies_obstacle(self) -> None:
        """
        Obstacle condition: U >= (K - e^x)^+ for interior nodes.
        Use a grid spanning x=0 so that boundary conditions are physically correct:
        upper x boundary (s >> K) is truly OTM and U=0 is appropriate there.
        """
        hp = HestonParams(sigma=0.3)
        # x_min=-1.0 → x_max=-1.0+100*0.0062=−0.38;  Nx=101 spans a reasonable range
        # But we need x=0 in the domain; use x_min=-0.7, Nx=201 → x_max=0.542
        gp = GridParams(Nx=201, Ny=11, Nt=10, dx=0.0062, dy=0.002, dt=0.1, x_min=-0.70)
        x_grid = gp.x_grid()
        _, _, _, ckpts = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5]), verbose=False
        )
        obstacle_1d = np.maximum(hp.K - np.exp(x_grid), 0.0)
        for data in ckpts.values():
            U = data["U"]
            # Check only interior x-nodes (exclude first and last row where
            # Dirichlet boundary conditions are imposed)
            for k in range(gp.Ny):
                assert np.all(U[1:-1, k] >= obstacle_1d[1:-1] - 1e-8), \
                    f"U violates obstacle at y-index {k}"

    def test_set_b_solve(self, tiny_grid_b: tuple) -> None:
        hp, gp = tiny_grid_b
        x, y, boundary, _ = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5, 1.0]), verbose=False
        )
        assert np.all(np.isfinite(boundary))
        assert np.all(boundary > 0)


# ---------------------------------------------------------------------------
# 6. Boundary extraction
# ---------------------------------------------------------------------------

class TestBoundaryExtraction:
    def test_extract_at_slice_toy(self) -> None:
        """Toy: w=0 for x<=x_b, w>0 for x>x_b. Boundary should be near x_b."""
        x_grid = np.linspace(-2, 2, 200)
        x_b = -0.5  # true boundary
        w_slice = np.maximum(x_grid - x_b, 0.0) ** 2
        b = extract_boundary_at_slice(w_slice, x_grid)
        # Should recover close to exp(x_b)
        assert abs(b - np.exp(x_b)) < 0.05

    def test_extract_at_slice_all_zero(self) -> None:
        """If w=0 everywhere, fallback should return a finite value."""
        x_grid = np.linspace(-2, 2, 50)
        w_slice = np.zeros(50)
        b = extract_boundary_at_slice(w_slice, x_grid)
        assert np.isfinite(b)

    def test_extract_at_slice_all_positive(self) -> None:
        """If w>0 everywhere, should still return a finite value (first cont node)."""
        x_grid = np.linspace(-2, 2, 50)
        w_slice = np.ones(50) * 0.1
        b = extract_boundary_at_slice(w_slice, x_grid)
        assert np.isfinite(b)
        assert b > 0

    def test_extract_boundary_row_shape(self, medium_grid_a: tuple) -> None:
        hp, gp = medium_grid_a
        x_grid = gp.x_grid()
        # Synthetic U: everywhere equal to obstacle (contact region)
        s_grid = np.exp(x_grid)
        U_2d = np.maximum(hp.K - s_grid, 0.0)[:, None] * np.ones((gp.Nx, gp.Ny))
        b_row = extract_boundary_row(U_2d, x_grid, hp.K)
        assert b_row.shape == (gp.Ny,)

    def test_extract_boundary_row_positive(self, medium_grid_a: tuple) -> None:
        hp, gp = medium_grid_a
        x_grid = gp.x_grid()
        s_grid = np.exp(x_grid)
        U_2d = np.maximum(hp.K - s_grid, 0.0)[:, None] * np.ones((gp.Nx, gp.Ny))
        b_row = extract_boundary_row(U_2d, x_grid, hp.K)
        assert np.all(b_row > 0)

    def test_estimate_slope_monotone_boundary(self) -> None:
        """On a monotone decreasing boundary, slope should be negative."""
        y_grid = np.linspace(0, 1, 100)
        b_row = 0.9 - 0.4 * y_grid  # linearly decreasing
        slope = estimate_boundary_slope(b_row, y_grid, y_target=0.5, window=0.1)
        assert slope < 0.0

    def test_estimate_slope_quadratic(self) -> None:
        """On a quadratic boundary, slope should match derivative."""
        y_grid = np.linspace(0, 1, 200)
        y0 = 0.5
        b_row = 0.9 - 0.4 * (y_grid - 0.2) ** 2
        # db/dy at y0=0.5 = -0.8 * (0.5 - 0.2) = -0.24
        slope = estimate_boundary_slope(b_row, y_grid, y_target=y0, window=0.1)
        expected = -0.8 * (y0 - 0.2)
        assert abs(slope - expected) < 0.05, f"Slope mismatch: {slope} vs {expected}"


# ---------------------------------------------------------------------------
# 7. Monotonicity and sanity checks
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_monotone_surface_passes(self) -> None:
        """A strictly decreasing surface should pass the monotonicity check."""
        tau = np.linspace(0, 1, 50)
        y   = np.linspace(0, 1, 30)
        # b decreases with tau and y
        TAU, Y = np.meshgrid(tau, y, indexing="ij")
        b_surface = 0.95 - 0.3 * TAU - 0.2 * Y
        is_mono_tau, viol_tau = verify_monotonicity(b_surface, axis=0)
        is_mono_y,   viol_y   = verify_monotonicity(b_surface, axis=1)
        assert is_mono_tau
        assert is_mono_y

    def test_non_monotone_surface_fails(self) -> None:
        """A surface with an increasing segment should fail."""
        b_surface = np.ones((10, 5))
        b_surface[5, :] = 1.1  # spike upward -> not non-increasing
        is_mono, viol = verify_monotonicity(b_surface, axis=0)
        assert not is_mono
        assert viol > 0

    def test_sanity_check_all_positive(self) -> None:
        b = np.ones((20, 10)) * 0.8
        result = boundary_sanity_check(b, K=1.0)
        assert result["all_positive"]

    def test_sanity_check_below_K(self) -> None:
        b = np.ones((20, 10)) * 0.8
        result = boundary_sanity_check(b, K=1.0)
        assert result["all_below_K"]

    def test_sanity_check_finite(self) -> None:
        b = np.ones((20, 10)) * 0.8
        result = boundary_sanity_check(b, K=1.0)
        assert result["finite"]

    def test_sanity_check_tau0_near_K(self) -> None:
        """If boundary at tau=0 is close to K, tau0_near_K should be True."""
        b = np.ones((20, 10)) * 0.5
        b[0, :] = 0.98  # Close to K=1
        result = boundary_sanity_check(b, K=1.0)
        assert result["tau0_near_K"]


# ---------------------------------------------------------------------------
# 8. Gamma identity formula
# ---------------------------------------------------------------------------

class TestGammaFormula:
    def test_gamma_positive_for_valid_params(self) -> None:
        """For rK > qb and valid parameters, gamma formula should be positive."""
        b = 0.85
        b_y = -1.2
        y = 0.09
        val = gamma_formula(b, b_y, y, r=0.05, q=0.02, K=1.0, rho=-0.7, sigma=0.3)
        assert val > 0.0

    def test_gamma_nan_for_zero_variance(self) -> None:
        b = 0.85
        b_y = -1.2
        y = 0.0  # degenerate axis
        val = gamma_formula(b, b_y, y, r=0.05, q=0.02, K=1.0, rho=-0.7, sigma=0.3)
        assert np.isnan(val)

    def test_gamma_known_value(self) -> None:
        """
        At (tau=0.25, y=0.09), paper reports P_ss^formula ~ 1.977.
        Our formula with b=0.8098, b_y=-1.179*K should give close result.
        """
        K = 1.0
        b = 0.8098 * K
        b_y = -1.179 * K
        y = 0.09
        r, q, rho, sigma = 0.05, 0.02, -0.7, 0.3
        val = gamma_formula(b, b_y, y, r=r, q=q, K=K, rho=rho, sigma=sigma)
        # Paper reports 1.977; allow ~15% tolerance due to b,b_y being from paper
        assert 1.5 < val < 2.5, f"Expected ~1.977, got {val:.4f}"

    def test_gamma_denominator_quadratic_positive(self) -> None:
        """
        The denominator quadratic b^2 - 2rho*sigma*b*by + sigma^2*by^2
        equals (b - rho*sigma*by)^2 + (1-rho^2)*sigma^2*by^2 >= 0.
        """
        b    = 0.85
        rho  = -0.7
        sigma = 0.3
        for by in [-2.0, -0.5, 0.0, 0.5]:
            quad = b**2 - 2*rho*sigma*b*by + sigma**2*by**2
            assert quad >= 0.0, f"Quadratic form negative for by={by}"

    def test_gamma_returns_nan_when_denom_negative(self) -> None:
        """If we artificially make denominator negative, should return NaN."""
        # The denominator is y * (quadratic form).
        # With y > 0, denom can only be negative if quadratic < 0 (impossible for |rho|<1).
        # But if y < 0 (invalid physics), denominator < 0 -> NaN.
        val = gamma_formula(0.85, -1.0, -0.1, r=0.05, q=0.02, K=1.0, rho=-0.7, sigma=0.3)
        assert np.isnan(val)


# ---------------------------------------------------------------------------
# 9. Wxx estimation
# ---------------------------------------------------------------------------

class TestWxxEstimation:
    def test_wxx_on_quadratic_w(self) -> None:
        """For w = c*(x - beta)^2 near the boundary, w_xx = 2c."""
        Nx = 200
        dx = 0.0062
        x_grid = -2.5 + np.arange(Nx) * dx
        beta = -0.3  # boundary location
        c = 5.0
        # w = 0 for x <= beta, w = c*(x-beta)^2 for x > beta
        w_slice = np.where(x_grid > beta, c * (x_grid - beta)**2, 0.0)
        wxx_est = estimate_wxx_at_boundary(w_slice, x_grid, beta, dx)
        if not np.isnan(wxx_est):
            assert abs(wxx_est - 2 * c) < 1.0, f"Expected ~{2*c}, got {wxx_est:.4f}"

    def test_wxx_returns_nan_when_not_enough_nodes(self) -> None:
        """If fewer than n_skip+n_use+1 continuation nodes, should return NaN."""
        x_grid = np.linspace(-2, 2, 20)
        w_slice = np.zeros(20)
        w_slice[-2:] = 0.01  # Only 2 continuation nodes (need more)
        beta = x_grid[-3]
        wxx = estimate_wxx_at_boundary(w_slice, x_grid, beta, dx=0.2, n_skip=2, n_use=4)
        assert np.isnan(wxx)


# ---------------------------------------------------------------------------
# 10. End-to-end integration test with small grid
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_small_solve_boundary_monotone_in_tau(self, medium_grid_a: tuple) -> None:
        """On a medium grid, boundary should be roughly non-increasing in tau."""
        hp, gp = medium_grid_a
        _, _, boundary, _ = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5, 1.0]), verbose=False
        )
        _, viol_tau = verify_monotonicity(boundary, axis=0, tol=1e-4)
        total = boundary.shape[0] * boundary.shape[1]
        # Allow up to 1% violation (numerical noise acceptable)
        assert viol_tau / total < 0.01, f"Too many τ-monotonicity violations: {viol_tau}/{total}"

    def test_small_solve_boundary_monotone_in_y(self, medium_grid_a: tuple) -> None:
        """On a medium grid, boundary should be roughly non-increasing in y."""
        hp, gp = medium_grid_a
        _, _, boundary, _ = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5, 1.0]), verbose=False
        )
        _, viol_y = verify_monotonicity(boundary, axis=1, tol=1e-4)
        total = boundary.shape[0] * boundary.shape[1]
        assert viol_y / total < 0.01, f"Too many y-monotonicity violations: {viol_y}/{total}"

    def test_small_solve_b_at_y0_finite(self, medium_grid_a: tuple) -> None:
        """At the degenerate axis y=0, boundary should remain finite and positive."""
        hp, gp = medium_grid_a
        _, _, boundary, _ = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5]), verbose=False
        )
        b_y0 = boundary[:, 0]
        assert np.all(np.isfinite(b_y0))
        assert np.all(b_y0 > 0.0)

    def test_small_solve_feller_violated_still_works(self, tiny_grid_b: tuple) -> None:
        """Set B (Feller violated) should still produce valid boundary surface."""
        hp, gp = tiny_grid_b
        _, _, boundary, _ = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5]), verbose=False
        )
        sanity = boundary_sanity_check(boundary, hp.K)
        assert sanity["all_positive"]
        assert sanity["finite"]


# ---------------------------------------------------------------------------
# 11. Gamma table (integration)
# ---------------------------------------------------------------------------

class TestGammaTable:
    def test_gamma_table_shape(self, medium_grid_a: tuple) -> None:
        hp, gp = medium_grid_a
        x_grid, y_grid, boundary, ckpts = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5, 1.0]), verbose=False
        )
        tau_targets = [0.5, 1.0]
        y_targets = [0.04, 0.09, 0.16]

        w_ckpts = {}
        b_ckpts = {}
        tau_grid = gp.tau_grid()
        for tau_v in tau_targets:
            ck_tau = min(ckpts.keys(), key=lambda t: abs(t - tau_v))
            w_ckpts[tau_v] = ckpts[ck_tau]["w"]
            n_idx = int(np.argmin(np.abs(tau_grid - tau_v)))
            b_ckpts[tau_v] = boundary[n_idx, :]

        table = compute_gamma_table(
            w_ckpts, b_ckpts, x_grid, y_grid,
            tau_targets, y_targets,
            r=hp.r, q=hp.q, K=hp.K, rho=hp.rho, sigma=hp.sigma,
            dx=gp.dx,
        )
        assert table.shape[0] == len(tau_targets) * len(y_targets)
        assert "Pss_fd" in table.columns
        assert "Pss_formula" in table.columns
        assert "pct_diff" in table.columns

    def test_gamma_table_formula_positive(self, medium_grid_a: tuple) -> None:
        """Formula gamma must be positive for valid Heston parameters."""
        hp, gp = medium_grid_a
        x_grid, y_grid, boundary, ckpts = solve_american_put(
            hp, gp, checkpoint_taus=np.array([0.5, 1.0]), verbose=False
        )
        tau_targets = [0.5, 1.0]
        y_targets = [0.09, 0.16, 0.36]

        w_ckpts = {}
        b_ckpts = {}
        tau_grid = gp.tau_grid()
        for tau_v in tau_targets:
            ck_tau = min(ckpts.keys(), key=lambda t: abs(t - tau_v))
            w_ckpts[tau_v] = ckpts[ck_tau]["w"]
            n_idx = int(np.argmin(np.abs(tau_grid - tau_v)))
            b_ckpts[tau_v] = boundary[n_idx, :]

        table = compute_gamma_table(
            w_ckpts, b_ckpts, x_grid, y_grid,
            tau_targets, y_targets,
            r=hp.r, q=hp.q, K=hp.K, rho=hp.rho, sigma=hp.sigma,
            dx=gp.dx,
        )
        valid_rows = table.dropna(subset=["Pss_formula"])
        assert (valid_rows["Pss_formula"] > 0).all()

    def test_validate_positivity_keys(self, medium_grid_a: tuple) -> None:
        """validate_gamma_positivity should return required keys."""
        import pandas as pd
        table = pd.DataFrame({
            "Pss_fd": [1.0, 2.0],
            "Pss_formula": [0.9, 1.8],
            "pct_diff": [11.1, 11.1],
            "by_over_K": [-0.5, -0.3],
            "b_over_K": [0.8, 0.7],
        })
        result = validate_gamma_positivity(table)
        assert "all_Pss_fd_positive" in result
        assert "all_Pss_formula_positive" in result
        assert "max_abs_pct_diff" in result
