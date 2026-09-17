"""Native tests skip only for the documented pure-Python installation.

FLOW_REQUIRE_NATIVE=1 turns absence/stale builds into failures in native CI.
"""
from concurrent.futures import ThreadPoolExecutor
import importlib
import os

import numpy as np
import pytest

from backend.app.diffusion import Grid, diffusion_step, gaussian_initial, solve_initial, stable_timestep
from backend.app.numerics import native


@pytest.fixture
def cpp():
    info = native.status()
    if not info["available"]:
        if os.environ.get("FLOW_REQUIRE_NATIVE") == "1":
            pytest.fail(info["reason"])
        pytest.skip("Optional native extension is unavailable in the pure Python installation.")
    return importlib.import_module("flow_cpp")


@pytest.mark.parametrize("shape", [(2, 2), (9, 13), (32, 32)])
def test_single_and_multistep_parity(cpp, shape):
    ny, nx = shape
    grid = Grid((0, 0, 100, 75), nx, ny)
    initial = np.random.default_rng(42).uniform(0, 1, shape)
    original = initial.copy()
    dt = stable_timestep(grid, 2.5)
    np.testing.assert_allclose(cpp.diffusion_step(initial, grid.dx, grid.dy, 2.5, dt), diffusion_step(initial, grid, 2.5, dt), rtol=1e-11, atol=1e-12)
    times = np.array([0, dt * 0.3, dt * 7.1, dt * 15.37])
    expected, py_diag = solve_initial(grid, initial, 2.5, times, dt)
    actual, diag = native.solve_native(grid, initial, 2.5, times, dt)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(initial, original)
    assert diag["internal_steps"] == py_diag["internal_steps"]
    assert sum(diag["actual_step_sizes_s"]) == pytest.approx(times[-1])
    assert diag["relative_mass_drift"] < 1e-12
    assert diag["min_concentration"] >= 0
    assert all(a >= b - 1e-10 for a, b in zip(diag["energy_by_output"], diag["energy_by_output"][1:]))


def test_default_gaussian_and_readonly_input(cpp):
    grid = Grid((0, 0, 4000, 4000))
    initial = gaussian_initial(grid, (2000, 2000))
    initial.flags.writeable = False
    times = np.linspace(0, 1800, 61)
    expected, _ = solve_initial(grid, initial, 20, times)
    actual, diagnostics = native.solve_native(grid, initial, 20, times)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    assert diagnostics["native_build"]["threads"] == 1
    assert not diagnostics["native_build"]["fast_math"]


@pytest.mark.parametrize("layout", ["float32", "transposed", "strided", "unaligned"])
def test_public_conversion_is_explicit_and_measured(cpp, layout):
    grid = Grid((0, 0, 10, 10), 8, 8)
    values = np.arange(64.0).reshape(8, 8)
    unaligned = np.ndarray(values.shape, dtype=float, buffer=bytearray(values.nbytes + 1), offset=1)
    unaligned[:] = values
    initial = {"float32": values.astype(np.float32), "transposed": values.T, "strided": np.repeat(values, 2, axis=1)[:, ::2], "unaligned": unaligned}[layout]
    original = initial.copy()
    result, diag = native.solve_native(grid, initial, 1, [0, 0.3])
    expected, _ = solve_initial(grid, initial, 1, [0, 0.3])
    np.testing.assert_allclose(result, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_array_equal(initial, original)
    assert diag["conversion_ms"] >= 0
    assert diag["solve_ms"] >= diag["conversion_ms"]


def test_strict_binding_rejects_implicit_copies(cpp):
    field = np.ones((4, 4))
    unaligned = np.ndarray(field.shape, dtype=float, buffer=bytearray(field.nbytes + 1), offset=1)
    for invalid in (field.astype(np.float32), field.astype(">f8"), field.T, field[:, ::2], field.tolist(), unaligned):
        with pytest.raises((ValueError, TypeError)):
            cpp.solve(invalid, np.array([0, 1.0]), 1, 1, 1, 0.2)


@pytest.mark.parametrize("times", [[], [1, 1], [2, 1], [-1, 1], [0, np.nan], [[0, 1]]])
def test_binding_invalid_times(cpp, times):
    with pytest.raises(ValueError):
        cpp.solve(np.ones((4, 4)), np.array(times, dtype=float), 1, 1, 1, 0.2)


@pytest.mark.parametrize("dx,dy,kappa,dt", [(0, 1, 1, 0.1), (1, -1, 1, 0.1), (1, 1, -1, 0.1), (1, 1, float("nan"), 0.1), (1, 1, 1, 0), (1, 1, 1, -0.1), (1, 1, 1, 0.251)])
def test_binding_invalid_parameters(cpp, dx, dy, kappa, dt):
    with pytest.raises(ValueError):
        cpp.solve(np.ones((4, 4)), np.array([1.0]), dx, dy, kappa, dt)


def test_bad_fields_and_grid_mismatch(cpp):
    with pytest.raises(ValueError):
        cpp.solve(np.full((3, 3), np.nan), np.array([1.0]), 1, 1, 1, 0.2)
    with pytest.raises(ValueError):
        cpp.solve(np.ones((1, 3)), np.array([1.0]), 1, 1, 1, 0.2)
    with pytest.raises(ValueError):
        native.solve_native(Grid((0, 0, 1, 1), 4, 4), np.ones((3, 3)), 1, [0])


def test_zero_time_and_independent_concurrent_workspaces(cpp):
    grid = Grid((0, 0, 1, 1), 17, 12)
    initial = np.random.default_rng(7).random((12, 17))
    result, diag = native.solve_native(grid, initial, 1, [0])
    np.testing.assert_array_equal(result[0], initial)
    assert diag["internal_steps"] == 0
    inputs = [initial * factor for factor in (1, 2, 3, 4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda item: native.solve_native(grid, item, 1, [0, 0.2])[0], inputs))
    for item, actual in zip(inputs, results):
        expected, _ = solve_initial(grid, item, 1, [0, 0.2])
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    assert not np.shares_memory(results[0], results[1])


def test_missing_extension_has_actionable_error(monkeypatch):
    def unavailable(name):
        raise ImportError("test deliberately blocks flow_cpp")
    monkeypatch.setattr(native, "import_module", unavailable)
    assert not native.status()["available"]
    with pytest.raises(RuntimeError, match="pip install ./cpp"):
        native.solve_native(Grid((0, 0, 1, 1), 2, 2), np.ones((2, 2)), 1, [0])


def test_stale_extension_is_rejected(cpp, monkeypatch):
    monkeypatch.setattr(native, "source_hash", lambda: "different source")
    assert not native.status()["available"]
    with pytest.raises(RuntimeError, match="different source"):
        native.solve_native(Grid((0, 0, 1, 1), 2, 2), np.ones((2, 2)), 1, [0])


def test_nonbinary_timestep_matches_public_numpy_solver(cpp):
    from backend.app.numerics.solver import solve
    grid = Grid((0, 0, 10, 10), 7, 5)
    initial = np.random.default_rng(19).random((5, 7))
    times = [0, 0.3, 0.7, 1.03, 3.13]
    expected, numpy_diag = solve(grid, initial, 1, times, method="explicit_euler", backend="numpy", dt=0.1)
    actual, cpp_diag = solve(grid, initial, 1, times, method="explicit_euler", backend="cpp", dt=0.1)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    assert cpp_diag["internal_steps"] == numpy_diag["internal_steps"]
    for key in ("mass_history", "energy_history", "min_history", "max_history"):
        np.testing.assert_allclose(cpp_diag[key], numpy_diag[key], rtol=1e-11, atol=1e-12)
    assert cpp_diag["relative_mass_balance_error"] < 1e-12


def test_zero_diffusion_is_exact_and_does_not_fallback(cpp):
    from backend.app.numerics.solver import solve
    grid = Grid((0, 0, 10, 10), 7, 5)
    initial = np.random.default_rng(20).random((5, 7))
    for dt in (None, 0.1):
        frames, diagnostics = native.solve_native(grid, initial, 0, [0, 0.3, 2.33], dt)
        np.testing.assert_array_equal(frames, np.stack([initial] * 3))
        assert diagnostics["backend"] == "cpp"
        assert diagnostics["cfl_limit_s"] is None
    frames, diagnostics = solve(grid, initial, 0, [0, 0.3, 2.33], backend="cpp")
    np.testing.assert_array_equal(frames, np.stack([initial] * 3))
    assert diagnostics["backend"] == "cpp"


def test_time_alignment_respects_physical_scale(cpp):
    from backend.app.numerics.solver import solve
    grid = Grid((0, 0, 2, 2), 2, 2)
    initial = np.array([[1.0, 0], [0, 0]])
    expected, numpy_diag = solve(grid, initial, 1e20, [1e-20], backend="numpy")
    actual, cpp_diag = native.solve_native(grid, initial, 1e20, [1e-20])
    assert cpp_diag["internal_steps"] > 0
    assert cpp_diag["internal_steps"] == numpy_diag["internal_steps"]
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
