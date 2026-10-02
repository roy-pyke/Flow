"""Variable-material finite-volume invariants and solver integration."""
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy.linalg import expm

from backend.app.diffusion import Grid
from backend.app.numerics import (
    DiffusionCoefficients, clear_factor_cache, diffusion_matrix, diffusion_rate,
    prepare_diffusivity, solve, timestep_limit,
)
from backend.app.numerics.coefficients import harmonic_mean


def independent_faces(grid, cells, boundary):
    """Plain Python face enumeration, independent of production vectorization."""
    faces = []
    for y in range(grid.ny):
        for x in range(grid.nx - 1):
            rate = 2 / (1 / cells[y, x] + 1 / cells[y, x+1]) / grid.dx**2
            faces.append((y*grid.nx+x, y*grid.nx+x+1, rate))
    for y in range(grid.ny - 1):
        for x in range(grid.nx):
            rate = 2 / (1 / cells[y, x] + 1 / cells[y+1, x]) / grid.dy**2
            faces.append((y*grid.nx+x, (y+1)*grid.nx+x, rate))
    if boundary == "periodic":
        for y in range(grid.ny):
            rate = 2 / (1 / cells[y, -1] + 1 / cells[y, 0]) / grid.dx**2
            faces.append((y*grid.nx+grid.nx-1, y*grid.nx, rate))
        for x in range(grid.nx):
            rate = 2 / (1 / cells[-1, x] + 1 / cells[0, x]) / grid.dy**2
            faces.append(((grid.ny-1)*grid.nx+x, x, rate))
    return faces


def independent_matrix(grid, cells, boundary):
    result = np.zeros((cells.size, cells.size))
    for a, b, rate in independent_faces(grid, cells, boundary):
        result[a, b] += rate
        result[b, a] += rate
        result[a, a] -= rate
        result[b, b] -= rate
    return result


@pytest.mark.parametrize("boundary", ["zero_flux", "periodic"])
@pytest.mark.parametrize("shape", [(2, 2), (2, 5), (7, 2), (5, 4)])
def test_operator_matches_independent_dense_face_pairs_and_energy_identity(boundary, shape):
    ny, nx = shape
    grid = Grid((-2, 1, 5, 4), nx, ny)
    values = np.exp(np.random.default_rng(22).uniform(-3, 3, size=shape))
    expected = independent_matrix(grid, values, boundary)
    prepared = prepare_diffusivity(grid, values, boundary)
    matrix = diffusion_matrix(grid, prepared, boundary)
    assert matrix.format == "csc"
    np.testing.assert_allclose(matrix.toarray(), expected, rtol=4e-15, atol=4e-15)
    np.testing.assert_allclose(matrix.toarray(), matrix.T.toarray(), atol=0, rtol=0)
    np.testing.assert_allclose(matrix.sum(axis=0), 0, atol=2e-14)
    np.testing.assert_allclose(matrix.sum(axis=1), 0, atol=2e-14)
    np.testing.assert_allclose(-matrix.diagonal(), prepared.outgoing_rates.ravel(), rtol=3e-15)
    assert prepared.max_outgoing_rate == float(prepared.outgoing_rates.max())
    assert np.linalg.eigvalsh(expected).max() < 2e-14
    field = np.random.default_rng(31).normal(size=shape)
    rate = diffusion_rate(field, grid, prepared, boundary)
    np.testing.assert_allclose(rate.ravel(), expected @ field.ravel(), rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(diffusion_rate(np.ones(shape), grid, prepared, boundary), 0)
    energy_loss = sum(conductance * (field.ravel()[a] - field.ravel()[b])**2
                      for a, b, conductance in independent_faces(grid, values, boundary))
    assert float(field.ravel() @ rate.ravel()) == pytest.approx(-energy_loss, rel=3e-15)
    assert float(rate.sum()) == pytest.approx(0, abs=4e-14)


def test_two_cell_periodic_axis_keeps_both_physical_faces():
    grid = Grid((0, 0, 2, 2), 2, 2)
    cells = np.array([[1.0, 4.0], [2.0, 8.0]])
    closed = diffusion_matrix(grid, cells).toarray()
    periodic = diffusion_matrix(grid, cells, "periodic").toarray()
    np.testing.assert_allclose(periodic, 2 * closed, atol=0, rtol=2e-16)
    assert periodic[0, 1] == pytest.approx(3.2)


@pytest.mark.parametrize("boundary", ["zero_flux", "periodic"])
def test_uniform_array_reduces_to_scalar_operator(boundary):
    grid = Grid((0, 0, 5, 3), 7, 4)
    cells = np.full((4, 7), 0.7)
    field = np.random.default_rng(8).normal(size=cells.shape)
    np.testing.assert_allclose(diffusion_matrix(grid, cells, boundary).toarray(),
                               diffusion_matrix(grid, 0.7, boundary).toarray(), rtol=3e-16, atol=3e-16)
    np.testing.assert_allclose(diffusion_rate(field, grid, cells, boundary),
                               diffusion_rate(field, grid, 0.7, boundary), rtol=2e-15, atol=2e-15)


def test_material_snapshot_hash_immutable_backing_and_original_input_isolation():
    grid = Grid((0, 0, 4, 3), 4, 3)
    cells = np.arange(1.0, 13.0).reshape(3, 4)
    prepared = prepare_diffusivity(grid, cells)
    assert isinstance(prepared, DiffusionCoefficients)
    assert prepare_diffusivity(grid, prepared) is prepared
    expected_hash = prepare_diffusivity(grid, np.asfortranarray(cells)).coefficient_sha256
    assert prepared.coefficient_sha256 == expected_hash
    assert prepare_diffusivity(grid, cells.astype(">f8")).coefficient_sha256 == expected_hash
    cells[0, 0] = 777
    assert prepared.values[0, 0] == 1
    assert prepare_diffusivity(grid, cells).coefficient_sha256 != expected_hash
    for array in (prepared.values, prepared.x_rates, prepared.y_rates, prepared.outgoing_rates):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 3
        if isinstance(array.base, np.ndarray):
            with pytest.raises(ValueError):
                array.base.setflags(write=True)
    with pytest.raises(ValueError, match="match"):
        prepare_diffusivity(Grid((0, 0, 8, 3), 4, 3), prepared)
    with pytest.raises(ValueError, match="match"):
        diffusion_matrix(grid, prepared, "periodic")
    periodic = prepare_diffusivity(grid, prepared.values, "periodic")
    # Material content is identical; the operator's grid/boundary remain part
    # of the solver cache key separately.
    assert periodic.cache_key == prepared.cache_key
    for array in (periodic.x_periodic_rates, periodic.y_periodic_rates):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    metadata = periodic.metadata()
    arrays = [periodic.values, periodic.x_rates, periodic.y_rates, periodic.outgoing_rates,
              periodic.x_periodic_rates, periodic.y_periodic_rates]
    assert metadata["resident_bytes"] == sum(a.nbytes for a in arrays)
    assert metadata["matrix_free_temporary_bytes"] == 8 * (12 + max(9, 8, 3, 4))
    assert "not process RSS" in metadata["memory_accounting"]
    assert metadata["min"] == 1 and metadata["max"] == 12
    metadata["min"] = -5
    assert periodic.metadata()["min"] == 1


@pytest.mark.parametrize("a,b", [(1e-300, 1e300), (1e300, 1e300),
                                 (np.finfo(float).max, np.finfo(float).max),
                                 (np.nextafter(0.0, 1.0), 1e300), (1e-200, 3e-200),
                                 (1e200, 3e200)])
def test_stable_harmonic_mean_against_high_precision(a, b):
    with localcontext() as context:
        context.prec = 800
        da, db = Decimal.from_float(float(a)), Decimal.from_float(float(b))
        expected = float(2 * da * db / (da + db))
    actual = float(harmonic_mean(a, b))
    assert np.isfinite(actual) and actual > 0
    assert actual == pytest.approx(expected, rel=4e-16, abs=np.nextafter(0.0, 1.0))
    assert float(harmonic_mean(b, a)) == actual


def test_rate_scaling_avoids_unnecessary_squared_spacing_overflow_and_underflow():
    small = prepare_diffusivity(Grid((0, 0, 2e-200, 2e-200), 2, 2), np.full((2, 2), 1e-300))
    large = prepare_diffusivity(Grid((0, 0, 2e200, 2e200), 2, 2), np.full((2, 2), 1e300))
    np.testing.assert_allclose(small.x_rates, 1e100, rtol=3e-16)
    np.testing.assert_allclose(large.x_rates, 1e-100, rtol=3e-16)


@pytest.mark.parametrize("bad", [np.ones((3, 2)), np.ones((2, 2), dtype=complex),
                                 np.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=object),
                                 np.zeros((2, 2)), np.full((2, 2), -1),
                                 np.full((2, 2), np.inf), np.full((2, 2), np.nan)])
def test_invalid_material_arrays_rejected(bad):
    with pytest.raises(ValueError):
        prepare_diffusivity(Grid((0, 0, 2, 2), 2, 2), bad)


def test_unrepresentable_rates_and_outgoing_sums_rejected():
    with pytest.raises(ValueError, match="face rates"):
        prepare_diffusivity(Grid((0, 0, 2e-200, 2e-200), 2, 2), np.ones((2, 2)))
    with pytest.raises(ValueError, match="face rates"):
        prepare_diffusivity(Grid((0, 0, 2e200, 2e200), 2, 2), np.full((2, 2), 1e-300))
    with pytest.raises(ValueError, match="outgoing"):
        prepare_diffusivity(Grid((0, 0, 2, 2), 2, 2), np.full((2, 2), np.finfo(float).max))
    with pytest.raises(ValueError, match="boundaries"):
        diffusion_matrix(Grid((0, 0, 2, 2), 2, 2), np.ones((2, 2)), "open")
    with pytest.raises(ValueError, match="finite"):
        diffusion_rate(np.full((2, 2), np.nan), Grid((0, 0, 2, 2), 2, 2), np.ones((2, 2)))
    assert prepare_diffusivity(Grid((0, 0, 2, 2), 2, 2), 0) == 0.0


def test_layered_steady_series_resistance_reference_with_external_boundary_terms():
    """Validation-only imposed endpoints; production supports closed/periodic."""
    nx, ny = 10, 2
    grid = Grid((0, 0, 2, 1), nx, ny)
    line = np.r_[np.full(4, 0.03), np.full(6, 3.0)]
    cells = np.repeat(line[None], ny, axis=0)
    matrix = diffusion_matrix(grid, cells).toarray()
    c_left, c_right = 2.0, 0.2
    rhs = np.zeros((ny, nx))
    # Add exterior half-cell resistances only to this independent validation
    # system, leaving the production closed-domain operator untouched.
    for y in range(ny):
        left, right = y*nx, y*nx+nx-1
        left_rate, right_rate = 2 * line[0] / grid.dx**2, 2 * line[-1] / grid.dx**2
        matrix[left, left] -= left_rate
        matrix[right, right] -= right_rate
        rhs[y, 0] = -left_rate*c_left
        rhs[y, -1] = -right_rate*c_right
    solution = np.linalg.solve(matrix, rhs.ravel()).reshape(ny, nx)
    resistance = np.r_[grid.dx/(2*line[0]), grid.dx/(2*line[:-1]) + grid.dx/(2*line[1:]),
                       grid.dx/(2*line[-1])]
    flux = (c_left-c_right)/resistance.sum()
    expected = c_left - flux*np.cumsum(resistance[:-1])
    np.testing.assert_allclose(solution, np.broadcast_to(expected, solution.shape), rtol=2e-13, atol=2e-14)
    numerical_flux = 2/(1/line[:-1]+1/line[1:]) * (solution[0, :-1]-solution[0, 1:]) / grid.dx
    np.testing.assert_allclose(numerical_flux, flux, rtol=2e-13, atol=1e-14)


@pytest.mark.parametrize("boundary", ["zero_flux", "periodic"])
def test_solver_fe_limit_is_exact_maximum_outgoing_rate_and_preserves_convexity(boundary):
    grid = Grid((0, 0, 3, 2), 5, 4)
    cells = np.exp(np.random.default_rng(17).uniform(-2, 2, (4, 5)))
    material = prepare_diffusivity(grid, cells, boundary)
    limit = timestep_limit(grid, material, "explicit_euler", (0, 0), boundary=boundary)
    assert limit == 1/material.max_outgoing_rate
    field = np.zeros((4, 5))
    field.flat[int(np.argmax(material.outgoing_rates))] = 1
    frames, diagnostics = solve(grid, field, material, [0, limit], method="explicit_euler",
                                dt=limit, boundary=boundary)
    assert frames[-1].min() >= -3e-16
    assert frames[-1].max() <= 1
    assert frames[-1].sum() == pytest.approx(1, rel=2e-15)
    assert diagnostics["cfl_limit_s"] == limit
    bad_step = field + 1.01*limit*diffusion_rate(field, grid, material, boundary)
    assert bad_step.min() < -0.009
    with pytest.raises(ValueError, match="CFL"):
        solve(grid, field, material, [limit], method="explicit_euler", dt=1.01*limit, boundary=boundary)


@pytest.mark.parametrize("method,backend,order", [("explicit_euler", "numpy", 1),
                                                ("backward_euler", "scipy", 1),
                                                ("crank_nicolson", "scipy", 2)])
@pytest.mark.parametrize("boundary", ["zero_flux", "periodic"])
def test_solver_variable_time_convergence_against_independent_dense_exponential(method, backend, order, boundary):
    grid = Grid((0, 0, 3, 2), 4, 3)
    cells = np.array([[0.2, 0.4, 0.3, 0.6], [0.7, 0.5, 0.2, 0.4], [0.3, 0.8, 0.6, 0.4]])
    field = np.random.default_rng(4).random(cells.shape)
    reference = (expm(0.1*independent_matrix(grid, cells, boundary)) @ field.ravel()).reshape(field.shape)
    errors = []
    for step in (0.025, 0.0125, 0.00625):
        frames, diagnostics = solve(grid, field, cells, [0.1], method=method, backend=backend,
                                    dt=step, boundary=boundary)
        errors.append(np.linalg.norm(frames[-1] - reference))
        assert frames[-1].sum() == pytest.approx(field.sum(), rel=3e-14)
        assert frames[-1].min() >= 0
        assert diagnostics["energy_max_increase"] < 1e-13
    assert errors[0]/errors[1] > 2**order*0.9
    assert errors[1]/errors[2] > 2**order*0.9


def test_solver_material_cache_keys_are_content_based_and_isolated_from_input_mutation():
    clear_factor_cache()
    grid = Grid((0, 0, 4, 3), 4, 3)
    cells = np.ones((3, 4))
    field = np.random.default_rng(13).random(cells.shape)
    prepared = prepare_diffusivity(grid, cells)
    a, cold = solve(grid, field, prepared, [0.2], method="backward_euler", backend="scipy", dt=0.1)
    b, warm = solve(grid, field, cells.copy(), [0.2], method="backward_euler", backend="scipy", dt=0.1)
    np.testing.assert_array_equal(a, b)
    assert cold["factorization_count"] > 0
    assert warm["factorization_count"] == 0
    assert warm["factorization_cache_hits"] > 0
    cells[1, 1] = 8
    c, changed = solve(grid, field, cells, [0.2], method="backward_euler", backend="scipy", dt=0.1)
    assert changed["factorization_count"] > 0
    assert np.max(np.abs(c-a)) > 1e-3
    dense_system = np.eye(field.size) - 0.1 * independent_matrix(grid, cells, "zero_flux")
    expected = np.linalg.solve(dense_system, np.linalg.solve(dense_system, field.ravel()))
    np.testing.assert_allclose(c[-1].ravel(), expected, rtol=3e-15, atol=3e-15)
    repeated, _ = solve(grid, field, prepared, [0.2], method="backward_euler", backend="scipy", dt=0.1)
    np.testing.assert_array_equal(repeated, a)


def test_solver_rejects_unsupported_variable_combinations_without_native_fallback():
    grid = Grid((0, 0, 2, 2), 2, 2)
    cells = np.ones((2, 2))
    field = np.array([[0.0, 1.0], [2.0, 3.0]])
    for kwargs in ({"backend": "cpp"}, {"boundary": "open"},
                   {"method": "advection_explicit", "backend": "numpy", "boundary": "periodic"},
                   {"method": "imex_euler", "backend": "numpy_scipy", "boundary": "periodic"}):
        with pytest.raises(ValueError):
            solve(grid, field, cells, [0.01], **kwargs)
    frames, diagnostics = solve(grid, field, cells, [0.01], backend="auto")
    assert diagnostics["backend"] == "numpy"
    assert np.isfinite(frames).all()


@pytest.mark.parametrize("bad_field", [np.full((2, 2), 1+2j), np.ones((2, 2), dtype=bool),
                                      np.ones((2, 2), dtype=object)])
def test_matrix_free_rejects_nonreal_and_nonnumeric_fields_without_silent_cast(bad_field):
    grid = Grid((0, 0, 2, 2), 2, 2)
    with pytest.raises(ValueError, match="real numeric"):
        diffusion_rate(bad_field, grid, np.ones((2, 2)))
    integer_rate = diffusion_rate(np.array([[0, 1], [2, 3]]), grid, np.ones((2, 2)))
    np.testing.assert_array_equal(integer_rate, [[3, 1], [-1, -3]])


@pytest.mark.parametrize("method,backend", [("explicit_euler", "numpy"),
                                          ("backward_euler", "scipy"),
                                          ("crank_nicolson", "scipy")])
@pytest.mark.parametrize("boundary", ["zero_flux", "periodic"])
def test_solver_constant_state_preserved_and_uniform_array_matches_scalar(method, backend, boundary):
    grid = Grid((0, 0, 3, 2), 5, 4)
    cells = np.exp(np.random.default_rng(18).uniform(-2, 1, (4, 5)))
    constant = np.full((4, 5), 2.5)
    frames, diagnostics = solve(grid, constant, cells, [0, 0.03, 0.1], method=method,
                                backend=backend, boundary=boundary, dt=0.01)
    np.testing.assert_allclose(frames, 2.5, rtol=3e-15, atol=3e-15)
    assert diagnostics["relative_mass_drift"] < 3e-15
    assert diagnostics["energy_max_increase"] < 1e-25
    initial = np.random.default_rng(13).random(cells.shape)
    arguments = {"method": method, "backend": backend, "boundary": boundary, "dt": 0.01}
    uniform, _ = solve(grid, initial, np.full_like(cells, 0.7), [0, 0.03, 0.1], **arguments)
    scalar, _ = solve(grid, initial, 0.7, [0, 0.03, 0.1], **arguments)
    np.testing.assert_allclose(uniform, scalar, rtol=3e-15, atol=3e-15)


def test_solver_variable_cn_energy_stability_does_not_imply_positivity():
    grid = Grid((0, 0, 1, 1), 8, 8)
    cells = np.random.default_rng(42).uniform(0.1, 1.0, (8, 8))
    initial = np.zeros((8, 8))
    initial[3, 4] = 1
    cn, cn_diag = solve(grid, initial, cells, [10.0], method="crank_nicolson", backend="scipy", dt=10)
    be, be_diag = solve(grid, initial, cells, [10.0], method="backward_euler", backend="scipy", dt=10)
    assert cn[-1].min() < -0.9
    assert cn_diag["energy_max_increase"] < 1e-13
    assert be[-1].min() > 0
    assert be_diag["relative_mass_drift"] < 1e-12
    assert cn_diag["relative_mass_drift"] < 1e-12


@pytest.mark.parametrize("dtype", [complex, object, bool, "U2"])
@pytest.mark.parametrize("variable", [False, True])
def test_solver_rejects_nonreal_initial_without_discarding_information(dtype, variable):
    grid = Grid((0, 0, 1, 1), 2, 2)
    initial = np.ones((2, 2), dtype=dtype)
    if dtype is complex:
        initial += 3j
    with pytest.raises(ValueError, match="real numeric"):
        solve(grid, initial, np.ones((2, 2)) if variable else 1.0, [0])
