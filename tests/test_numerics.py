"""V2 mathematical checks, with independent exact-mode/exponential references."""
import numpy as np
import pytest
from scipy.sparse.linalg import expm_multiply

from backend.app.diffusion import Grid, diffusion_step, solve_initial
from backend.app.numerics import (advection_matrix, advection_rate, clear_factor_cache,
                                  diffusion_matrix, diffusion_rate, factor_cache_info, solve)


@pytest.mark.parametrize("boundary", ["zero_flux", "periodic", "open"])
@pytest.mark.parametrize("shape", [(2, 2), (7, 4)])
def test_conservative_operator_structure(boundary, shape):
    grid = Grid((0, -2, 3, 4), *shape)
    matrix = diffusion_matrix(grid, 0.7, boundary)
    dense = matrix.toarray()
    z = np.random.default_rng(7).normal(size=(grid.ny, grid.nx))
    np.testing.assert_allclose(dense, dense.T, atol=1e-14)
    np.testing.assert_allclose(dense.sum(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(dense.sum(axis=1), 0, atol=1e-14)
    assert z.ravel() @ matrix @ z.ravel() <= 1e-13
    np.testing.assert_allclose(matrix @ z.ravel(), diffusion_rate(z, grid, .7, boundary).ravel(), atol=1e-13)
    if boundary != "periodic":
        np.testing.assert_allclose((diffusion_step(z, grid, .7, .001) - z) / .001,
                                   diffusion_rate(z, grid, .7, boundary), atol=3e-13)


def test_fe_legacy_parity_and_input_unchanged():
    grid = Grid((0, 0, 40, 60), 8, 6)
    initial = np.random.default_rng(1).uniform(size=(6, 8))
    copy = initial.copy()
    expected, _ = solve_initial(grid, initial, .2, [0, .17, .73, 2], dt=.1)
    actual, diagnostics = solve(grid, initial, .2, [0, .17, .73, 2], dt=.1)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(initial, copy)
    assert diagnostics["relative_mass_balance_error"] < 1e-13


@pytest.mark.parametrize("method,backend", [("explicit_euler", "numpy"), ("backward_euler", "scipy"),
                                            ("crank_nicolson", "scipy")])
def test_constants_mass_energy_and_positivity(method, backend):
    grid = Grid((0, 0, 3, 2), 12, 9)
    constant, d = solve(grid, np.ones((9, 12)), .1, [0, .1, .7], method=method, backend=backend, dt=.03)
    np.testing.assert_allclose(constant, 1, atol=3e-14)
    initial = np.random.default_rng(9).uniform(size=(9, 12))
    frames, d = solve(grid, initial, .1, np.linspace(0, .6, 7), method=method, backend=backend, dt=.03)
    assert d["relative_mass_drift"] < 1e-12
    assert d["energy_max_increase"] < 1e-13
    assert frames.min() >= initial.min() - 1e-13
    assert frames.max() <= initial.max() + 1e-13
    assert d["linear_residual_max"] < 1e-12


def test_factor_cache_uses_actual_short_steps_and_is_bounded():
    clear_factor_cache()
    grid = Grid((0, 0, 1, 1), 8, 6)
    initial = np.random.default_rng(2).uniform(size=(6, 8))
    frames, d = solve(grid, initial, .1, [.125, .3125], method="backward_euler", backend="scipy", dt=.125)
    matrix = diffusion_matrix(grid, .1).toarray()
    eye = np.eye(48)
    reference = initial.ravel()
    for dt in [.125, .125, .0625]:
        reference = np.linalg.solve(eye - dt * matrix, reference)
    np.testing.assert_allclose(frames[-1].ravel(), reference, atol=2e-15)
    assert d["factorization_count"] == 2
    assert d["factorization_cache_hits"] == 1
    for dt in np.linspace(.001, .02, 12):
        solve(grid, initial, .1, [dt], method="backward_euler", backend="scipy", dt=dt)
    info = factor_cache_info()
    assert info["entries"] <= info["max_entries"]
    assert info["bytes"] <= info["max_bytes"]


@pytest.mark.parametrize("method,backend,expected", [("explicit_euler", "numpy", 1),
                                                       ("backward_euler", "scipy", 1),
                                                       ("crank_nicolson", "scipy", 2)])
def test_time_order_against_exact_semidiscrete_mode(method, backend, expected):
    grid = Grid((0, 0, 2, 1.5), 20, 16)
    xx, yy = np.meshgrid(grid.x, grid.y)
    mode = np.cos(np.pi * xx / 2) * np.cos(np.pi * yy / 1.5)
    lam = -4 * .07 * (np.sin(np.pi * grid.dx / 4) ** 2 / grid.dx ** 2
                     + np.sin(np.pi * grid.dy / 3) ** 2 / grid.dy ** 2)
    reference = 1 + .5 * np.exp(lam * .5) * mode
    errors = []
    for count in [32, 64, 128, 256]:
        frames, _ = solve(grid, 1 + .5 * mode, .07, [.5], dt=.5 / count, method=method, backend=backend)
        errors.append(np.sqrt(np.mean((frames[-1] - reference) ** 2)))
    orders = np.log2(np.asarray(errors[:-1]) / errors[1:])
    assert np.all(np.abs(orders - expected) < .12), (errors, orders)


def test_cn_oscillation_and_rannacher_damping():
    grid = Grid((0, 0, 1, 1), 20, 20)
    initial = np.zeros((20, 20))
    initial[10, 10] = 1
    plain, d = solve(grid, initial, 1, [.1], method="crank_nicolson", backend="scipy", dt=.1)
    started, ds = solve(grid, initial, 1, [.1], method="crank_nicolson", backend="scipy", dt=.1, startup="rannacher")
    backward, _ = solve(grid, initial, 1, [.1], method="backward_euler", backend="scipy", dt=.1)
    assert plain.min() < -.5
    assert started.min() > 0 and backward.min() > 0
    assert ds["internal_steps"] == 2
    assert d["energy_max_increase"] == 0
    with pytest.raises(ValueError, match="CFL"):
        solve(grid, initial, 1, [.1], dt=.1)


@pytest.mark.parametrize("wind", [(1, 0), (-1, 0), (.8, -.4), (-.5, .7)])
@pytest.mark.parametrize("boundary", ["periodic", "open"])
def test_upwind_operator_flux_and_open_balance(wind, boundary):
    grid = Grid((0, 0, 3, 2), 9, 5)
    initial = np.random.default_rng(10).uniform(size=(5, 9))
    rate, outward = advection_rate(initial, grid, wind, boundary)
    np.testing.assert_allclose(advection_matrix(grid, wind, boundary) @ initial.ravel(), rate.ravel(), atol=1e-14)
    assert abs(rate.sum() * grid.dx * grid.dy + outward) < 1e-13
    for method, backend in [("advection_explicit", "numpy"), ("imex_euler", "numpy_scipy")]:
        frames, d = solve(grid, initial, .03, [0, .123, .61], method=method, backend=backend,
                          boundary=boundary, velocity=wind, dt=.017)
        assert d["relative_mass_balance_error"] < 1e-12
        assert frames.min() >= 0
        if boundary == "open":
            assert d["final_mass"] < d["initial_mass"]
            assert d["cumulative_outward_flux"] > 0
        else:
            assert d["relative_mass_drift"] < 1e-12


def test_one_dimensional_outflow_sign_and_zero_inflow():
    # Two identical rows produce an effectively 1D advection problem.
    grid = Grid((0, 0, 4, 2), 4, 2)
    initial = np.array([[1., 2, 3, 4], [1., 2, 3, 4]])
    rate, out = advection_rate(initial, grid, (2., 0.), "open")
    np.testing.assert_array_equal(rate, [[-2, -2, -2, -2], [-2, -2, -2, -2]])
    assert out == 16
    reverse, out_reverse = advection_rate(initial, grid, (-2., 0.), "open")
    np.testing.assert_array_equal(reverse, [[2, 2, 2, -8], [2, 2, 2, -8]])
    assert out_reverse == 4


def test_imex_zero_wind_and_zero_diffusion_reductions():
    grid = Grid((0, 0, 2, 3), 8, 7)
    initial = np.random.default_rng(3).uniform(size=(7, 8))
    be, _ = solve(grid, initial, .2, [.4], method="backward_euler", backend="scipy", dt=.02)
    imex, _ = solve(grid, initial, .2, [.4], method="imex_euler", backend="numpy_scipy", dt=.02)
    np.testing.assert_array_equal(be, imex)
    adv, _ = solve(grid, initial, 0, [.4], method="advection_explicit", velocity=(.2, -.3), boundary="periodic", dt=.02)
    imex, _ = solve(grid, initial, 0, [.4], method="imex_euler", backend="numpy_scipy", velocity=(.2, -.3), boundary="periodic", dt=.02)
    np.testing.assert_array_equal(adv, imex)


@pytest.mark.parametrize("method,backend", [("advection_explicit", "numpy"), ("imex_euler", "numpy_scipy")])
def test_transport_time_order_with_independent_matrix_exponential(method, backend):
    grid = Grid((0, 0, 2, 3), 12, 10)
    xx, yy = np.meshgrid(grid.x, grid.y)
    initial = 1 + .3 * np.cos(np.pi * xx) * np.cos(2 * np.pi * yy / 3)
    operator = diffusion_matrix(grid, .03, "periodic") + advection_matrix(grid, (.3, -.2), "periodic")
    reference = expm_multiply(.3 * operator, initial.ravel()).reshape(initial.shape)
    errors = []
    for n in [16, 32, 64, 128]:
        frames, _ = solve(grid, initial, .03, [.3], method=method, backend=backend,
                          boundary="periodic", velocity=(.3, -.2), dt=.3 / n)
        errors.append(np.sqrt(np.mean((frames[-1] - reference) ** 2)))
    assert np.all(np.abs(np.log2(np.asarray(errors[:-1]) / errors[1:]) - 1) < .12)


def test_degenerate_duration_mass_and_invalid_arguments():
    grid = Grid((0, 0, 1, 2), 4, 3)
    initial = np.zeros((3, 4))
    frames, d = solve(grid, initial, 0, [0])
    np.testing.assert_array_equal(frames[0], initial)
    assert d["internal_steps"] == 0 and d["relative_mass_drift"] == 0
    for kwargs in [{"dt": 0}, {"dt": np.nan}, {"kappa": -1}, {"velocity": (1, np.inf)},
                   {"boundary": "wrong"}, {"method": "wrong"}, {"method": "backward_euler"},
                   {"backend": "scipy"}, {"startup": "rannacher"}, {"velocity": (1, 0)}]:
        params = dict(kappa=.1, output_times_s=[0, .1])
        params.update(kwargs)
        with pytest.raises(ValueError):
            solve(grid, initial, **params)
    for times in [[0, 0], [-1], [np.nan], []]:
        with pytest.raises(ValueError):
            solve(grid, initial, .1, times)
    for bad in [np.zeros((4, 3)), np.full((3, 4), np.inf)]:
        with pytest.raises(ValueError):
            solve(grid, bad, .1, [0])


def test_time_units_do_not_erase_short_but_physical_evolution():
    grid = Grid((0, 0, 1, 1), 5, 4)
    initial = np.random.default_rng(12).uniform(size=(4, 5))
    ordinary, _ = solve(grid, initial, 1, [.02], dt=.001)
    scaled, d = solve(grid, initial, 1e20, [2e-22], dt=1e-23)
    assert d["internal_steps"] == 20
    np.testing.assert_allclose(scaled, ordinary, atol=1e-14, rtol=1e-14)


def test_validation_production_reports_both_norms_and_continuous_transport():
    from backend.app.numerics.validation import run_validation
    report = run_validation(quick=True)
    assert report["summary"]["all_order_checks_pass"]
    assert report["summary"]["transport_balance_pass"]
    assert report["summary"]["cn_negative_example_observed"]
    for row in report["transport"]["spatial_diffusion"][1:]:
        assert .8 <= row["order_l2"] <= 1.2
        assert .8 <= row["order_linf"] <= 1.2


def test_short_final_output_does_not_reduce_implicit_default_macro_step():
    grid = Grid((0, 0, 4, 3), 4, 3)
    _, d = solve(grid, np.ones((3, 4)), .1, [0, 30, 60, 60.0001],
                 method="backward_euler", backend="scipy")
    assert d["dt_max_s"] == 30
    assert d["internal_steps"] == 3
    assert d["actual_dt_min_s"] == pytest.approx(.0001)
