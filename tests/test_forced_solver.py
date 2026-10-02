"""Independent source quadrature, flux ledger, clock and dispatch checks."""
import numpy as np
import pytest

from backend.app.diffusion import Grid
from backend.app.numerics import PrescribedFields, solve, timestep_limit, clear_factor_cache


def source_fields(grid, times, values, boundary="zero_flux"):
    return PrescribedFields(grid, times, source=np.asarray(values)[:, None, None]
                            * np.ones((len(times), grid.ny, grid.nx)), boundary=boundary)


@pytest.mark.parametrize("method,backend,expected", [
    ("explicit_euler", "numpy", 3.125), ("backward_euler", "scipy", 3.875),
    ("crank_nicolson", "scipy", 3.5), ("advection_explicit", "numpy", 3.125),
    ("imex_euler", "numpy_scipy", 3.125)])
def test_source_quadrature_and_actual_injection_ledger(method, backend, expected):
    grid = Grid((0, 0, 2, 3), 4, 3)
    forcing = source_fields(grid, [0, 1], [2, 5])
    frames, d = solve(grid, np.zeros((3, 4)), 0, [0, 1], method=method,
                      backend=backend, dt=.25, forcing=forcing)
    np.testing.assert_allclose(frames[-1], expected, atol=2e-15)
    assert d["cumulative_source_mass"] == pytest.approx(6 * expected)
    assert d["mass_balance_error"] == pytest.approx(0, abs=1e-14)
    assert d["max_internal_mass_balance_error"] < 1e-14
    assert d["relative_mass_drift"] > 1  # Raw drift is expected with a source.


@pytest.mark.parametrize("method,backend,expected", [
    ("explicit_euler", "numpy", 2), ("backward_euler", "scipy", .8),
    ("crank_nicolson", "scipy", 1.4)])
def test_triangle_knots_split_steps_even_without_output(method, backend, expected):
    grid = Grid((0, 0, 1, 1), 2, 2)
    forcing = source_fields(grid, [0, .2, .7, 1], [0, 4, 0, 0])
    f, d = solve(grid, np.zeros((2, 2)), 0, [1], method=method, backend=backend,
                 dt=2, forcing=forcing)
    np.testing.assert_allclose(f[-1], expected)
    assert d["internal_steps"] == 3
    assert d["cumulative_source_mass"] == pytest.approx(expected)


def test_rannacher_uses_two_distinct_right_endpoint_source_times():
    grid = Grid((0, 0, 1, 1), 2, 2)
    forcing = source_fields(grid, [0, 1], [0, 1])
    f, d = solve(grid, np.zeros((2, 2)), 0, [.4, 1], method="crank_nicolson", backend="scipy",
                 dt=.4, forcing=forcing, startup="rannacher")
    np.testing.assert_allclose(f[0], .12, atol=2e-16)
    np.testing.assert_allclose(f[-1], .54, atol=2e-16)
    assert d["internal_steps"] == 4
    assert d["mass_balance_error"] == pytest.approx(0, abs=1e-15)


@pytest.mark.parametrize("method,backend", [("advection_explicit", "numpy"), ("imex_euler", "numpy_scipy")])
@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_uniform_face_wind_reduces_to_legacy(method, backend, boundary):
    grid = Grid((0, 0, 2, 1), 8, 5)
    forcing = PrescribedFields(grid, [0, .3],
                               velocity_x=np.full((2, 5, 9), .3),
                               velocity_y=np.full((2, 6, 8), -.2), boundary=boundary)
    initial = np.random.default_rng(73).uniform(size=(5, 8))
    args = dict(method=method, backend=backend, dt=.003, boundary=boundary)
    old, od = solve(grid, initial, .01, [0, .1, .3], velocity=(.3, -.2), **args)
    new, nd = solve(grid, initial, .01, [0, .1, .3], forcing=forcing, **args)
    np.testing.assert_allclose(new, old, atol=5e-15, rtol=5e-15)
    assert nd["cumulative_outward_flux"] == pytest.approx(od["cumulative_outward_flux"], abs=1e-14)
    assert nd["max_internal_mass_balance_error"] < 2e-14


@pytest.mark.parametrize("method,backend", [("advection_explicit", "numpy"), ("imex_euler", "numpy_scipy")])
def test_open_affine_wind_manufactured_solution_and_separate_flux_integrals(method, backend):
    grid = Grid((0, 0, 2, 3), 8, 3)
    times = np.array([0., 1.])
    x = np.linspace(0, 2, 9)
    u0, a, b = .4, .3, .2
    forcing = PrescribedFields(grid, times, velocity_x=np.broadcast_to(u0+a*x, (2, 3, 9)),
        source=np.broadcast_to((b+a*(1+b*times))[:, None, None], (2, 3, 8)),
        inflow={"left": np.broadcast_to((1+b*times)[:, None], (2, 3))}, boundary="open")
    _, d = solve(grid, np.ones((3, 8)), .02, [0, .5, 1], method=method,
                 backend=backend, dt=.05, forcing=forcing, boundary="open")
    assert d["final_mass"] == pytest.approx(7.2, abs=1e-13)
    assert d["cumulative_boundary_inward_mass"] > 0
    assert d["cumulative_boundary_outward_mass"] > 0
    assert d["max_internal_mass_balance_error"] < 1e-13
    # Analytic integrals are distinct from the Euler quadrature actually used.
    assert abs(d["cumulative_source_mass"] - 6*(b+a*(1+b/2))) > .005
    assert abs(d["cumulative_boundary_inward_mass"] - 3*u0*(1+b/2)) > .005


def test_compressive_wind_changes_constant_concentration_without_mass_creation():
    grid = Grid((0, 0, 1, 1), 16, 4)
    ux = np.sin(2*np.pi*np.linspace(0, 1, 17))
    ux[-1] = ux[0]
    forcing = PrescribedFields(grid, [0, .1], velocity_x=np.broadcast_to(ux, (2, 4, 17)), boundary="periodic")
    f, d = solve(grid, np.ones((4, 16)), 0, [0, .1], method="advection_explicit", dt=.002,
                 boundary="periodic", forcing=forcing)
    assert f[-1].max() > 1.3
    assert f[-1].min() < .8
    assert d["energy_max_increase"] > 0
    assert d["max_internal_mass_balance_error"] < 2e-15


def test_future_velocity_knots_constrain_step_even_when_initial_wind_zero():
    grid = Grid((0, 0, 1, 1), 4, 4)
    vx = np.zeros((2, 4, 5)); vx[1] = 10
    forcing = PrescribedFields(grid, [0, 1], velocity_x=vx, boundary="periodic")
    assert timestep_limit(grid, 0, "advection_explicit", (0, 0), "periodic", forcing=forcing) == pytest.approx(.9/40)
    with pytest.raises(ValueError, match="CFL"):
        solve(grid, np.ones((4, 4)), 0, [1], method="advection_explicit", dt=.1,
              forcing=forcing, boundary="periodic")


def test_signed_source_is_not_clipped_and_diffusion_factors_ignore_rhs_changes():
    grid = Grid((0, 0, 1, 1), 3, 3)
    clear_factor_cache()
    d1 = None
    for strength in [1, -2]:
        forcing = source_fields(grid, [0, 1], [strength, strength])
        f, d = solve(grid, np.zeros((3, 3)), .1, [1], method="backward_euler", backend="scipy",
                     dt=.25, forcing=forcing)
        np.testing.assert_allclose(f[-1], strength, atol=4e-15)
        if d1 is None:
            d1 = d
        else:
            assert d["factorization_count"] == 0
            assert d["factorization_cache_hits"] == 4
            assert d["min_concentration"] < 0


@pytest.mark.parametrize("kappa", [.03, np.full((3, 4), .03)])
def test_source_works_with_variable_diffusion_and_auto_keeps_numpy(kappa):
    grid = Grid((0, 0, 1, 1), 4, 3)
    forcing = source_fields(grid, [0, 1], [2, 2])
    f, d = solve(grid, np.zeros((3, 4)), kappa, [1], dt=.01, backend="auto", forcing=forcing)
    assert d["backend"] == "numpy"
    np.testing.assert_allclose(f[-1], 2, atol=4e-15)


@pytest.mark.parametrize("boundary", ["zero_flux", "periodic"])
@pytest.mark.parametrize("method,backend,order", [("explicit_euler", "numpy", 1),
    ("backward_euler", "scipy", 1), ("crank_nicolson", "scipy", 2)])
def test_variable_material_and_nonuniform_time_source_against_independent_augmented_system(boundary, method, backend, order):
    from scipy.linalg import expm
    grid = Grid((0, 0, 1, 1), 2, 2)
    materials = np.array([[.02, .2], [.1, .03]])
    matrix = np.zeros((6, 6))
    flat = materials.ravel()
    for a, b in [(0, 1), (2, 3), (0, 2), (1, 3)]:
        rate = 2/(1/flat[a]+1/flat[b])/.5**2
        if boundary == "periodic":
            rate *= 2  # A second physical face joins each pair on a 2-cell axis.
        matrix[a, a] -= rate; matrix[b, b] -= rate
        matrix[a, b] += rate; matrix[b, a] += rate
    q0 = np.array([.1, .2, .3, .4])
    q1 = np.array([-.2, .1, .3, -.1])
    matrix[:4, 4], matrix[:4, 5], matrix[4, 5] = q1, q0, 1
    initial = np.array([[1., .4], [.2, .8]])
    end = .2
    expected = (expm(end * matrix) @ np.r_[initial.ravel(), 0, 1])[:4]
    forcing = PrescribedFields(grid, [0, end], source=np.array([q0, q0+end*q1]).reshape(2, 2, 2), boundary=boundary)
    errors = []
    for count in [16, 32]:
        frames, d = solve(grid, initial, materials, [end], method=method, backend=backend,
                          boundary=boundary, dt=end/count, forcing=forcing)
        errors.append(np.linalg.norm(frames[-1].ravel()-expected))
        assert d["max_internal_mass_balance_error"] < 3e-15
    assert np.log2(errors[0]/errors[1]) == pytest.approx(order, abs=.03)


@pytest.mark.parametrize("options,match", [({"backend": "cpp"}, "C\\+\\+"),
    ({"velocity": (1, 0)}, "zero pair"), ({"startup": "rannacher"}, "Rannacher"),
    ({"boundary": "periodic"}, "match"), ({"dt": -.1}, "positive")])
def test_unsupported_forcing_contracts_rejected(options, match):
    grid = Grid((0, 0, 1, 1), 2, 2)
    with pytest.raises(ValueError, match=match):
        solve(grid, np.ones((2, 2)), .01, [1], forcing=source_fields(grid, [0, 1], [0, 1]), **options)


def test_input_time_coverage_and_no_output_initial_step():
    grid = Grid((0, 0, 1, 1), 2, 2)
    forcing = source_fields(grid, [0, 1], [2, 2])
    f, d = solve(grid, np.ones((2, 2)), 0, [0], forcing=forcing)
    assert d["internal_steps"] == 0
    np.testing.assert_array_equal(f[0], 1)
    with pytest.raises(ValueError):
        solve(grid, np.ones((2, 2)), 0, [1.01], forcing=forcing)
