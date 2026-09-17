import numpy as np
import pytest

from backend.app.diffusion import (Grid, bilinear_interpolate, diffusion_step,
                                   gaussian_initial, simulate, solve_initial,
                                   stable_timestep)
from backend.app.validation import validate_numerics


def test_constant_field_is_unchanged_with_zero_flux_walls():
    grid = Grid((0, 0, 100, 80), 10, 8)
    initial = np.full((8, 10), 0.37)
    frames, diagnostic = solve_initial(grid, initial, 20, [0, 30, 120])
    np.testing.assert_array_equal(frames, np.broadcast_to(initial, frames.shape))
    assert diagnostic["relative_mass_drift"] == 0


def test_reflecting_boundary_conserves_mass_without_clipping():
    grid = Grid((0, 0, 4000, 4000), 80, 80)
    run = simulate(grid, (0, 0), kappa=50)
    assert run.diagnostics["relative_mass_drift"] < 1e-10
    assert run.frames.min() >= 0
    assert run.frames.max() <= gaussian_initial(grid, (0, 0)).max()
    assert run.frames.dtype == np.float64
    np.testing.assert_array_equal(run.times_s, np.arange(0, 1801, 30))


def test_output_times_and_cfl_follow_plan():
    grid = Grid((0, 0, 4000, 4000))
    assert grid.h == 25
    assert stable_timestep(grid, 20) == pytest.approx(0.9 * 25 ** 2 / (4 * 20))
    run = simulate(grid, (2000, 2000), duration_s=31, frame_interval_s=30)
    assert run.frames.shape == (3, 160, 160)
    np.testing.assert_array_equal(run.times_s, [0, 30, 31])
    assert simulate(grid, (2000, 2000), duration_s=0).frames.shape == (1, 160, 160)


def test_cell_centre_interpolation_is_exact_for_affine_field_inside_centres():
    grid = Grid((100, 200, 200, 280), 10, 8)
    xx, yy = np.meshgrid(grid.x, grid.y)
    points = np.array([[110, 210], [128.3, 230.8], [180, 266]])
    sampled = bilinear_interpolate(2 * xx + 3 * yy, grid, points)
    np.testing.assert_allclose(sampled, 2 * points[:, 0] + 3 * points[:, 1])
    edges = bilinear_interpolate(np.ones((8, 10)), grid, np.array([[100, 200], [200, 280]]))
    np.testing.assert_array_equal(edges, [1, 1])
    with pytest.raises(ValueError, match="outside"):
        bilinear_interpolate(xx, grid, np.array([[99, 250]]))


def test_invalid_numerical_inputs_are_rejected():
    grid = Grid((0, 0, 100, 100), 10, 10)
    with pytest.raises(ValueError, match="stability"):
        diffusion_step(np.ones((10, 10)), grid, 20, 2)
    for bad in (-1, 0, np.nan):
        with pytest.raises(ValueError):
            stable_timestep(grid, bad)
    with pytest.raises(ValueError, match="inside"):
        gaussian_initial(grid, (-1, 10))
    with pytest.raises(ValueError, match="strictly increasing"):
        solve_initial(grid, np.ones((10, 10)), 20, [0, 0])


def test_cosine_analytic_solution_has_second_order_joint_convergence():
    report = validate_numerics()
    assert report["passed"], report
    assert [row["n"] for row in report["convergence"]] == [32, 64, 128]
    assert all(1.8 < row["observed_order"] < 2.2 for row in report["convergence"][1:])
