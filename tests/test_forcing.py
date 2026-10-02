"""Independent flux, clock, boundary and immutability checks for prescribed fields."""
from dataclasses import replace

import numpy as np
import pytest

from backend.app.diffusion import Grid
from backend.app.numerics.forcing import ForcingSnapshot, PrescribedFields, face_advection_rate
from backend.app.numerics.operators import advection_rate


def field_series(grid, boundary, seed=17):
    rng = np.random.default_rng(seed)
    times = np.array([0., .3, 1.7, 3.])
    vx = rng.uniform(-2, 2, (4, grid.ny, grid.nx+1))
    vy = rng.uniform(-2, 2, (4, grid.ny+1, grid.nx))
    if boundary == "zero_flux":
        vx[:, :, [0, -1]] = 0
        vy[:, [0, -1], :] = 0
    if boundary == "periodic":
        vx[:, :, -1] = vx[:, :, 0]
        vy[:, -1, :] = vy[:, 0, :]
    inflow = {name: rng.uniform(0, 2, (4, size)) if boundary == "open" else np.zeros((4, size))
              for name, size in (("left", grid.ny), ("right", grid.ny), ("bottom", grid.nx), ("top", grid.nx))}
    return PrescribedFields(grid, times, vx, vy, rng.normal(size=(4, grid.ny, grid.nx)), inflow, boundary)


def independent_advection(grid, snapshot):
    """Build transfers face by face without production differencing or upwind arrays."""
    matrix, incoming = np.zeros((grid.nx*grid.ny,)*2), np.zeros(grid.nx*grid.ny)
    def face(left, right, velocity, spacing):
        donor, recipient = (left, right) if velocity >= 0 else (right, left)
        rate = abs(velocity)/spacing
        matrix[donor, donor] -= rate
        matrix[recipient, donor] += rate
    for y in range(grid.ny):
        for x in range(1, grid.nx):
            face(y*grid.nx+x-1, y*grid.nx+x, snapshot.velocity_x[y, x], grid.dx)
        if snapshot.boundary == "periodic":
            face(y*grid.nx+grid.nx-1, y*grid.nx, snapshot.velocity_x[y, 0], grid.dx)
    for x in range(grid.nx):
        for y in range(1, grid.ny):
            face((y-1)*grid.nx+x, y*grid.nx+x, snapshot.velocity_y[y, x], grid.dy)
        if snapshot.boundary == "periodic":
            face((grid.ny-1)*grid.nx+x, x, snapshot.velocity_y[0, x], grid.dy)
    if snapshot.boundary == "open":
        boundary_faces = []
        for y in range(grid.ny):
            boundary_faces += [(y*grid.nx, -snapshot.velocity_x[y, 0], snapshot.inflow["left"][y], grid.dx),
                               (y*grid.nx+grid.nx-1, snapshot.velocity_x[y, -1], snapshot.inflow["right"][y], grid.dx)]
        for x in range(grid.nx):
            boundary_faces += [(x, -snapshot.velocity_y[0, x], snapshot.inflow["bottom"][x], grid.dy),
                               ((grid.ny-1)*grid.nx+x, snapshot.velocity_y[-1, x], snapshot.inflow["top"][x], grid.dy)]
        for cell, outward_velocity, exterior, spacing in boundary_faces:
            if outward_velocity >= 0:
                matrix[cell, cell] -= outward_velocity/spacing
            else:
                incoming[cell] += -outward_velocity*exterior/spacing
    return matrix, incoming


@pytest.mark.parametrize("boundary", ["zero_flux", "periodic", "open"])
@pytest.mark.parametrize("shape", [(2, 2), (3, 5)])
def test_face_flux_matches_independent_dense_transfers_and_mass_ledger(boundary, shape):
    ny, nx = shape
    grid = Grid((-2, 1, 5, 4), nx, ny)
    forcing = field_series(grid, boundary)
    values = np.random.default_rng(72).normal(size=shape)
    for time in (0, .17, .3, 1, 2.1, 3):
        snapshot = forcing.at(time)
        matrix, incoming = independent_advection(grid, snapshot)
        rate, ledger = face_advection_rate(values, grid, snapshot, boundary)
        np.testing.assert_allclose(rate.ravel(), matrix@values.ravel()+incoming, rtol=3e-15, atol=3e-15)
        assert grid.dx*grid.dy*rate.sum() == pytest.approx(-ledger["net_outward"], abs=4e-14)
        assert ledger["net_outward"] == ledger["outward"]-ledger["inward"]
        if boundary != "open":
            assert ledger == {"outward": 0, "inward": 0, "net_outward": 0}


@pytest.mark.parametrize("boundary", ["periodic", "open"])
@pytest.mark.parametrize("velocity", [(1.2, -.7), (-.4, .8), (0., 0.)])
def test_constant_faces_reduce_to_legacy_constant_velocity_operator(boundary, velocity):
    grid = Grid((0, 0, 3, 2), 5, 4)
    forcing = PrescribedFields(grid, [0, 1], np.full((2, 4, 6), velocity[0]),
                               np.full((2, 5, 5), velocity[1]), boundary=boundary)
    values = np.random.default_rng(8).random((4, 5))
    rate, ledger = face_advection_rate(values, grid, forcing.at(.5), boundary)
    old, outward = advection_rate(values, grid, velocity, boundary)
    np.testing.assert_array_equal(rate, old)
    assert ledger["net_outward"] == pytest.approx(outward, abs=2e-15)


def test_spatial_divergence_changes_constant_concentration_but_preserves_closed_mass():
    grid = Grid((0, 0, 3, 2), 3, 2)
    vx = np.broadcast_to([0., 1., -1., 0.], (2, 2, 4))
    forcing = PrescribedFields(grid, [0, 1], vx)
    snapshot = forcing.at(.4)
    rate, ledger = face_advection_rate(np.ones((2, 3)), grid, snapshot, "zero_flux")
    np.testing.assert_array_equal(rate, [[-1, 2, -1], [-1, 2, -1]])
    divergence = np.diff(snapshot.velocity_x, axis=1)/grid.dx + np.diff(snapshot.velocity_y, axis=0)/grid.dy
    np.testing.assert_array_equal(rate, -divergence)
    assert rate.sum() == 0 and ledger["net_outward"] == 0
    advanced = 1 + .9/forcing.max_outgoing_rate*rate
    assert advanced.min() >= 0 and advanced.max() > 1
    assert forcing.metadata()["max_abs_discrete_divergence"] == 2


def test_vertex_streamfunction_gives_discrete_solenoidal_closed_circulation():
    grid = Grid((0, 0, 2, 1), 8, 6)
    xx, yy = np.meshgrid(np.linspace(0, 2, 9), np.linspace(0, 1, 7))
    psi = np.sin(np.pi*xx/2)**2 * np.sin(np.pi*yy)**2
    psi[:, [0, -1]] = 0
    psi[[0, -1], :] = 0
    vx, vy = np.diff(psi, axis=0)/grid.dy, -np.diff(psi, axis=1)/grid.dx
    forcing = PrescribedFields(grid, [0, 1], np.repeat(vx[None], 2, axis=0), np.repeat(vy[None], 2, axis=0))
    rate, ledger = face_advection_rate(np.ones((6, 8)), grid, forcing.at(.2), "zero_flux")
    assert forcing.has_velocity
    np.testing.assert_allclose(rate, 0, atol=7e-15)
    assert ledger["net_outward"] == 0


def test_global_outgoing_bound_covers_interpolation_and_ensures_fe_positivity():
    grid = Grid((0, 0, 3, 2), 5, 4)
    forcing = field_series(grid, "open")
    largest = 0
    for time in np.linspace(0, 3, 61):
        snapshot = forcing.at(time)
        matrix, incoming = independent_advection(grid, snapshot)
        rate = -np.diag(matrix)
        assert float(rate.max()) <= forcing.max_outgoing_rate*(1+3e-16)
        if time in forcing.times_s:
            largest = max(largest, float(rate.max()))
        values = np.random.default_rng(5).random((4, 5))
        increment, _ = face_advection_rate(values, grid, snapshot, "open")
        advanced = values+increment/forcing.max_outgoing_rate
        assert advanced.min() >= -3e-16
    # Check all actual frame endpoints, not merely a uniform time sample.
    expected = max(float(np.max(-np.diag(independent_advection(grid, forcing.at(t))[0]))) for t in forcing.times_s)
    assert forcing.max_outgoing_rate == pytest.approx(expected, rel=3e-16)


def test_numerical_triangle_source_and_nonuniform_clock_are_not_silently_clamped():
    grid = Grid((0, 0, 2, 1), 4, 2)
    times = np.array([0., .2, 1.1, 2.])
    source = np.broadcast_to(np.array([0., 3., 0., -2.])[:, None, None], (4, 2, 4))
    forcing = PrescribedFields(grid, times, source=source)
    np.testing.assert_allclose(forcing.at(.1).source, 1.5)
    np.testing.assert_allclose(forcing.at(.65).source, 1.5)
    np.testing.assert_allclose(forcing.at(1.55).source, -1)
    np.testing.assert_array_equal(forcing.at(2).source, -2)
    assert forcing.next_knot(0) == .2
    assert forcing.next_knot(.2) == 1.1
    assert forcing.next_knot(2) is None
    assert forcing.max_outgoing_rate == 0
    assert forcing.has_source and not forcing.has_velocity and not forcing.has_inflow
    for query in (-1e-20, np.nextafter(2., np.inf), np.nan, 1+2j, [1.]):
        with pytest.raises(ValueError):
            forcing.at(query)
    for start, end in ((0, 2.000000001), (-1e-20, 1), (1, .9)):
        with pytest.raises(ValueError, match="covered"):
            forcing.validate_interval(start, end)
    forcing.validate_interval(.2, 1.1)


def test_open_upwind_inflow_is_ignored_on_outgoing_faces_and_ledger_orients_signed_values():
    grid = Grid((0, 0, 2, 1), 2, 2)
    vx, vy = np.zeros((2, 2, 3)), np.zeros((2, 3, 2))
    vx[:, :, 0] = 2
    vx[:, :, -1] = 3
    forcing = PrescribedFields(grid, [0, 1], vx, vy,
                               inflow={"left": np.full((2, 2), 5.), "right": np.full((2, 2), 1e20)},
                               boundary="open")
    rate, ledger = face_advection_rate(np.full((2, 2), -4.), grid, forcing.at(.5), "open")
    assert ledger["outward"] == -12
    assert ledger["inward"] == 10
    assert ledger["net_outward"] == -22
    assert rate.sum()*grid.dx*grid.dy == 22


def test_forcing_and_snapshots_own_immutable_data_and_hash_exact_inputs():
    grid = Grid((0, 0, 2, 1), 4, 2)
    times = np.array([0., 1., 2.])
    source = np.arange(24.).reshape(3, 2, 4)
    forcing = PrescribedFields(grid, times, source=source)
    same = PrescribedFields(grid, times.astype(">f8"), source=np.asfortranarray(source))
    assert forcing.content_sha256 == same.content_sha256
    arrays = [forcing.times_s, forcing.velocity_x, forcing.velocity_y, forcing.source, *forcing.inflow.values()]
    for time in (0, .3):
        snapshot = forcing.at(time)
        arrays += [snapshot.velocity_x, snapshot.velocity_y, snapshot.source, *snapshot.inflow.values()]
        with pytest.raises(TypeError):
            snapshot.inflow["left"] = np.zeros(2)
    for values in arrays:
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)
        if isinstance(values.base, np.ndarray):
            with pytest.raises(ValueError):
                values.base.setflags(write=True)
    source[0, 0, 0] = -42
    times[1] = .7
    assert forcing.source[0, 0, 0] == 0 and forcing.times_s[1] == 1
    assert forcing.content_sha256 != PrescribedFields(grid, times, source=source).content_sha256
    metadata = forcing.metadata()
    expected = sum(a.nbytes for a in [forcing.times_s, forcing.velocity_x, forcing.velocity_y, forcing.source, *forcing.inflow.values()])
    assert metadata["resident_bytes"] == expected
    assert "not process RSS" in metadata["memory_accounting"]


@pytest.mark.parametrize("times", [[0], [1, 2], [0, 0], [0, 1, .5], [0, np.inf], [0, np.nan], [False, True]])
def test_bad_shared_clocks_rejected(times):
    with pytest.raises(ValueError):
        PrescribedFields(Grid((0, 0, 1, 1), 2, 2), times)


@pytest.mark.parametrize("bad", [np.ones((2, 2, 2)), np.ones((2, 2, 3), dtype=bool),
                                 np.ones((2, 2, 3), dtype=complex), np.ones((2, 2, 3), dtype=object),
                                 np.full((2, 2, 3), np.nan)])
def test_bad_velocity_shapes_types_and_values_rejected(bad):
    with pytest.raises(ValueError):
        PrescribedFields(Grid((0, 0, 1, 1), 2, 2), [0, 1], bad)


def test_boundary_policies_reject_mismatch_instead_of_modifying_wind_or_inflow():
    grid = Grid((0, 0, 2, 1), 2, 2)
    vx, vy = np.zeros((2, 2, 3)), np.zeros((2, 3, 2))
    vx[1, 0, 0] = 1
    with pytest.raises(ValueError, match="zero normal"):
        PrescribedFields(grid, [0, 1], vx)
    with pytest.raises(ValueError, match="match exactly"):
        PrescribedFields(grid, [0, 1], vx, boundary="periodic")
    vx.fill(0)
    vy[1, -1, 0] = np.nextafter(0., 1.)
    with pytest.raises(ValueError, match="match exactly"):
        PrescribedFields(grid, [0, 1], vx, vy, boundary="periodic")
    for boundary in ("zero_flux", "periodic"):
        with pytest.raises(ValueError, match="require open"):
            PrescribedFields(grid, [0, 1], inflow={"left": np.ones((2, 2))}, boundary=boundary)
    with pytest.raises(ValueError, match="nonnegative"):
        PrescribedFields(grid, [0, 1], inflow={"left": -np.ones((2, 2))}, boundary="open")
    with pytest.raises(ValueError, match="mapping"):
        PrescribedFields(grid, [0, 1], inflow={"unknown": np.ones((2, 2))})
    with pytest.raises(ValueError, match="boundary"):
        PrescribedFields(grid, [0, 1], boundary="none")


def test_snapshots_validate_direct_construction_and_operator_compatibility():
    grid = Grid((0, 0, 2, 1), 2, 2)
    snapshot = PrescribedFields(grid, [0, 1]).at(0)
    with pytest.raises(ValueError):
        replace(snapshot, velocity_x=np.ones((2, 3), dtype=complex))
    with pytest.raises(ValueError):
        replace(snapshot, velocity_x=np.ones((2, 3)))
    with pytest.raises(ValueError):
        face_advection_rate(np.ones((2, 2)), grid, snapshot, "open")
    with pytest.raises(ValueError):
        face_advection_rate(np.ones((2, 2)), Grid((0, 0, 4, 1), 2, 2), snapshot, "zero_flux")
    with pytest.raises(ValueError):
        face_advection_rate(np.ones((2, 2), dtype=complex), grid, snapshot, "zero_flux")


def test_unrepresentable_outgoing_rate_and_field_flux_fail_explicitly():
    with pytest.raises(ValueError, match="outgoing rates"):
        PrescribedFields(Grid((0, 0, 2e-300, 2), 2, 2), [0, 1],
                         np.full((2, 2, 3), 1e300), boundary="open")
    grid = Grid((0, 0, 2, 2), 2, 2)
    forcing = PrescribedFields(grid, [0, 1], np.full((2, 2, 3), 1e300), boundary="open")
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        face_advection_rate(np.full((2, 2), 1e300), grid, forcing.at(.5), "open")
