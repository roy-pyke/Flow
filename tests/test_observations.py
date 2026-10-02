"""Independent invariants for the frozen-field observation contract."""

import math

import numpy as np
import pytest

from backend.app.diffusion import Grid, bilinear_interpolate, bilinear_weights
from backend.app.observations import build_edge_observer, build_path_trajectory
from backend.app.routing import RoadNetwork


def line_network(polylines):
    nodes, edges = [], []
    node_index = {}
    for polyline in polylines:
        ids = []
        for point in (polyline[0], polyline[-1]):
            point = tuple(point)
            if point not in node_index:
                name = str(len(nodes))
                node_index[point] = name
                nodes.append({"id": name, "x": point[0], "y": point[1], "lon": point[0], "lat": point[1]})
            ids.append(node_index[point])
        edges.append({"u": ids[0], "v": ids[1], "key": len(edges), "coordinates": polyline})
    return RoadNetwork({"nodes": nodes, "edges": edges, "crs": "EPSG:4326"})


@pytest.fixture
def small_network():
    return line_network([[[0, 0], [4, 3]], [[4, 3], [8, 3], [8, 6]],
                         [[4, 3], [0, 0]], [[0, 0], [0, 6], [8, 6]]])


@pytest.mark.parametrize("boundary", ["zero_flux", "open", "periodic"])
@pytest.mark.parametrize("quadrature", ["trapezoid", "gauss2_grid"])
def test_observer_is_nonnegative_and_constant_integral_is_travel_time(small_network, boundary, quadrature):
    grid = Grid((0, 0, 8, 6), 7, 4)
    observer = build_edge_observer(small_network, grid, boundary, 2.3, quadrature)
    expected = np.array([edge["length_m"] / 2.3 for edge in small_network.edges])
    assert observer.matrix.format == "csr"
    assert np.all(observer.matrix.data >= 0)
    np.testing.assert_allclose(observer.apply(np.ones((4, 7))), expected, rtol=2e-15)
    np.testing.assert_allclose(observer.matrix.sum(axis=1).A1, expected, rtol=2e-15)
    np.testing.assert_allclose(observer.apply(np.full((4, 7), 0.37)), 0.37 * expected, rtol=2e-15)
    assert observer.metadata["edge_ids"] == [[edge["u"], edge["v"], edge["key"]] for edge in small_network.edges]
    assert observer.metadata["csr_bytes"] == sum(a.nbytes for a in (observer.matrix.data, observer.matrix.indices, observer.matrix.indptr))


@pytest.mark.parametrize("boundary", ["zero_flux", "open", "periodic"])
def test_default_sparse_operator_matches_independent_legacy_sampling(small_network, boundary):
    grid = Grid((0, 0, 8, 6), 11, 8)
    field = np.random.default_rng(7).normal(size=(8, 11))
    points, weights, edge_indices = small_network._sampling(grid.h / 2)
    old_values = bilinear_interpolate(field, grid, points, boundary)
    expected = np.bincount(edge_indices, weights=old_values * weights / 1.4,
                           minlength=len(small_network.edges))
    observer = build_edge_observer(small_network, grid, boundary)
    np.testing.assert_allclose(observer.apply(field), expected, atol=2e-15)


def test_affine_and_bilinear_line_integrals_against_analytic_integral():
    grid = Grid((0, 0, 8, 6), 8, 6)
    start, end = np.array([0.7, 1.1]), np.array([6.8, 5.1])
    network = line_network([[start, [2.0, 4.0], end]])
    xx, yy = np.meshgrid(grid.x, grid.y)
    for xy_coefficient in (0.0, 1.7):
        field = 2 + 3 * xx - 0.2 * yy + xy_coefficient * xx * yy
        expected = 0.0
        for a, b in zip(network.edges[0]["coordinates"][:-1], network.edges[0]["coordinates"][1:]):
            a, b = np.asarray(a), np.asarray(b)
            dx, dy = b - a
            mean_xy = a[0] * a[1] + (a[0] * dy + a[1] * dx) / 2 + dx * dy / 3
            mean_value = 2 + 3 * (a[0] + b[0]) / 2 - 0.2 * (a[1] + b[1]) / 2 + xy_coefficient * mean_xy
            expected += np.linalg.norm(b - a) * mean_value / 1.4
        result = build_edge_observer(network, grid, quadrature="gauss2_grid").apply(field)[0]
        assert result == pytest.approx(expected, rel=3e-15)
        if xy_coefficient == 0:
            assert build_edge_observer(network, grid).apply(field)[0] == pytest.approx(expected, rel=3e-15)


def test_grid_split_gauss_integrates_piecewise_bilinear_corner_extensions():
    grid = Grid((0, 0, 4, 4), 4, 4)
    network = line_network([[[0, 0], [4, 4]]])
    xx, yy = np.meshgrid(grid.x, grid.y)
    exact = math.sqrt(2) * (0.5 * 0.5**2 + (3.5**3 - 0.5**3) / 3 + 0.5 * 3.5**2) / 1.4
    gauss = build_edge_observer(network, grid, quadrature="gauss2_grid")
    assert gauss.apply(xx * yy)[0] == pytest.approx(exact, rel=2e-15)
    errors = [abs(build_edge_observer(network, grid, sample_spacing_m=h).apply(xx * yy)[0] - exact)
              for h in (0.7, 0.07, 0.007)]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-5


@pytest.mark.parametrize("boundary", ["zero_flux", "open", "periodic"])
def test_grid_split_gauss_matches_refined_quadrature_for_arbitrary_field(small_network, boundary):
    grid = Grid((0, 0, 8, 6), 9, 5)
    field = np.random.default_rng(18).random((5, 9))
    reference = build_edge_observer(small_network, grid, boundary, quadrature="gauss2_grid").apply(field)
    fine = build_edge_observer(small_network, grid, boundary, sample_spacing_m=0.0005).apply(field)
    np.testing.assert_allclose(fine, reference, rtol=0, atol=5e-8)


def test_batch_chunking_signed_fields_and_linear_map(small_network):
    grid = Grid((0, 0, 8, 6), 7, 4)
    observer = build_edge_observer(small_network, grid)
    fields = np.random.default_rng(38).normal(size=(9, 4, 7))
    expected = np.array([observer.apply(field) for field in fields])
    np.testing.assert_allclose(observer.apply_batch(fields), expected, atol=2e-15)
    np.testing.assert_allclose(observer.apply_batch(fields, chunk_size=2), expected, atol=2e-15)
    np.testing.assert_allclose(observer.apply(2 * fields[0] - 3 * fields[1]),
                               2 * expected[0] - 3 * expected[1], atol=3e-15)
    assert observer.apply_batch(np.empty((0, 4, 7))).shape == (0, len(small_network.edges))
    for value in (0, -1, True, 1.2):
        with pytest.raises(ValueError, match="Chunk"):
            observer.apply_batch(fields, chunk_size=value)


def test_stable_identity_covers_geometry_grid_speed_and_sampling(small_network):
    grid = Grid((0, 0, 8, 6), 7, 4)
    baseline = build_edge_observer(small_network, grid)
    equivalent_network = line_network([edge["coordinates"] for edge in small_network.edges])
    assert baseline.metadata["operator_sha256"] == build_edge_observer(equivalent_network, grid).metadata["operator_sha256"]
    changed = [build_edge_observer(small_network, Grid((0, 0, 8, 6), 8, 4)),
               build_edge_observer(small_network, grid, boundary="periodic"),
               build_edge_observer(small_network, grid, speed_mps=2),
               build_edge_observer(small_network, grid, sample_spacing_m=0.1),
               build_edge_observer(small_network, grid, quadrature="gauss2_grid")]
    identities = [baseline.metadata["operator_sha256"], *(observer.metadata["operator_sha256"] for observer in changed)]
    assert len(set(identities)) == len(identities)
    bent = line_network([[[0, 0], [2, 4], [4, 3]]])
    straight = line_network([[[0, 0], [4, 3]]])
    assert build_edge_observer(bent, grid).metadata["geometry_sha256"] != build_edge_observer(straight, grid).metadata["geometry_sha256"]


def test_route_boundary_rejects_every_negative_concentration_and_nonfinite_field(small_network):
    grid = Grid((0, 0, 8, 6), 7, 4)
    for value in (-1, -1e-15, np.nan, np.inf):
        field = np.ones((4, 7))
        field[0, 0] = value
        with pytest.raises(ValueError):
            small_network.edge_exposures(field, grid)
    for shape in ((4, 6), (0,), (4, 7, 1)):
        with pytest.raises(ValueError, match="shape"):
            small_network.edge_exposures(np.ones(shape), grid)


def test_observer_rejects_invalid_inputs_including_outside_gauss_endpoints(small_network):
    grid = Grid((0, 0, 8, 6), 7, 4)
    for kwargs in ({"speed_mps": 0}, {"speed_mps": np.inf}, {"boundary": "silent_clip"},
                   {"quadrature": "midpoint"}, {"sample_spacing_m": 0},
                   {"sample_spacing_m": np.nan}, {"sample_spacing_m": 1, "quadrature": "gauss2_grid"}):
        with pytest.raises(ValueError):
            build_edge_observer(small_network, grid, **kwargs)
    outside = line_network([[[-1e-3, 1], [3, 1]]])
    for rule in ("trapezoid", "gauss2_grid"):
        with pytest.raises(ValueError, match="outside"):
            build_edge_observer(outside, grid, quadrature=rule)
    observer = build_edge_observer(small_network, grid)
    for field in (np.ones((7, 4)), np.full((4, 7), np.nan)):
        with pytest.raises(ValueError):
            observer.apply(field)
    with pytest.raises(ValueError):
        observer.apply_batch(np.ones((4, 7)))


def test_empty_edge_set_produces_empty_observation():
    network = RoadNetwork({"nodes": [{"id": 0, "x": 0, "y": 0}], "edges": []})
    grid = Grid((0, 0, 1, 1), 2, 2)
    for rule in ("trapezoid", "gauss2_grid"):
        observer = build_edge_observer(network, grid, quadrature=rule)
        assert observer.matrix.shape == (0, 4)
        assert observer.apply(np.ones((2, 2))).shape == (0,)
        assert observer.apply_batch(np.ones((3, 2, 2))).shape == (3, 0)


def test_routing_cache_separates_boundary_speed_and_grid(small_network):
    grid = Grid((0, 0, 8, 6), 7, 4)
    field = np.arange(28).reshape(4, 7)
    zero = small_network.edge_exposures(field, grid)
    original_id = small_network.last_observer_metadata["operator_sha256"]
    small_network.edge_exposures(field + 1, grid)
    assert len(small_network._observer_cache) == 1
    periodic = small_network.edge_exposures(field, grid, boundary="periodic")
    assert not np.allclose(periodic, zero)
    assert original_id != small_network.last_observer_metadata["operator_sha256"]
    assert len(small_network._observer_cache) == 2
    np.testing.assert_allclose(small_network.edge_exposures(field, grid, speed_mps=2.8), zero / 2)
    for speed in range(1, 15):
        small_network.edge_exposures(field, grid, speed_mps=speed)
    assert len(small_network._observer_cache) == 8


def test_retained_cache_bytes_are_bounded_and_oversized_objects_are_not_cached(small_network):
    grid = Grid((0, 0, 8, 6), 7, 4)
    field = np.ones((4, 7))
    expected = small_network.edge_exposures(field, grid)
    operator_size = small_network._observer_cache_bytes
    small_network._observer_cache_max_bytes = operator_size + 1
    small_network.edge_exposures(field, grid, speed_mps=2)
    assert len(small_network._observer_cache) == 1
    assert small_network._observer_cache_bytes <= small_network._observer_cache_max_bytes
    sampling_size = small_network._sampling_cache_bytes
    small_network._sampling_cache_max_bytes = sampling_size + 1
    small_network._sampling(grid.h)
    assert small_network._sampling_cache_bytes <= small_network._sampling_cache_max_bytes
    fresh = line_network([edge["coordinates"] for edge in small_network.edges])
    fresh._observer_cache_max_bytes = 1
    fresh._sampling_cache_max_bytes = 1
    np.testing.assert_allclose(fresh.edge_exposures(field, grid), expected)
    assert fresh._observer_cache_bytes == fresh._sampling_cache_bytes == 0
    assert len(fresh._observer_cache) == len(fresh._sampling_cache) == 0


def test_observation_rejects_unrepresentable_weights_and_results(small_network):
    grid = Grid((0, 0, 8, 6), 7, 4)
    with pytest.raises(ValueError, match="finitely"):
        build_edge_observer(small_network, grid, speed_mps=np.nextafter(0.0, 1.0))
    observer = build_edge_observer(small_network, grid)
    huge = np.full((4, 7), np.finfo(float).max)
    with pytest.raises(ValueError, match="nonfinite"):
        observer.apply(huge)
    with pytest.raises(ValueError, match="nonfinite"):
        observer.apply_batch(huge[None])


def test_prescribed_trajectory_preserves_polyline_timing_and_repeated_edges(small_network):
    trajectory = build_path_trajectory(small_network, [0, 2, 0, 1], speed_mps=2, departure_time_s=17)
    assert trajectory.arrival_time_s == pytest.approx(28)
    np.testing.assert_allclose(trajectory.vertex_times_s, [17, 19.5, 22, 24.5, 26.5, 28])
    np.testing.assert_allclose(trajectory.positions(np.array([17, 18.25, 21, 25.5, 28])),
                               [[0, 0], [2, 1.5], [1.6, 1.2], [6, 3], [8, 6]])
    np.testing.assert_allclose(trajectory.positions(17), [0, 0])
    assert trajectory.positions(np.empty((0, 2))).shape == (0, 2, 2)
    assert trajectory.edge_indices == (0, 2, 0, 1)
    assert trajectory.metadata["connector_policy"] == "graph_nodes_only"
    for time in (16.99, 28.01, np.nan):
        with pytest.raises(ValueError, match="interval"):
            trajectory.positions(time)
    for indices in ([], [0, 0], [-1], [True], [0.1], [100]):
        with pytest.raises(ValueError):
            build_path_trajectory(small_network, indices)
    with pytest.raises(ValueError, match="finite"):
        build_path_trajectory(small_network, [0], departure_time_s=np.inf)


def test_trajectory_retains_corners_and_skips_zero_length_vertices():
    network = line_network([[[0, 0], [0, 0], [1, 0], [1, 0], [1, 1]]])
    trajectory = build_path_trajectory(network, [0], speed_mps=1)
    np.testing.assert_array_equal(trajectory.vertex_times_s, [0, 1, 2])
    np.testing.assert_allclose(trajectory.positions([0.5, 1.5]), [[0.5, 0], [1, 0.5]])


@pytest.mark.parametrize("boundary", ["zero_flux", "open", "periodic"])
def test_bilinear_weights_partition_unity_on_rectangular_grid(boundary):
    grid = Grid((-3, 5, 13, 29), 7, 11)
    points = np.vstack((np.random.default_rng(42).uniform([-3, 5], [13, 29], size=(100, 2)),
                        [[-3, 5], [-3, 29], [13, 5], [13, 29]]))
    indices, weights = bilinear_weights(grid, points, boundary)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(axis=1), 1, rtol=0, atol=2e-16)
    assert indices.min() >= 0 and indices.max() < grid.nx * grid.ny
    np.testing.assert_allclose(bilinear_interpolate(np.full((11, 7), 3.7), grid, points, boundary), 3.7)


def test_periodic_fourier_seam_edges_and_corners_use_last_first_centres():
    grid = Grid((0, 0, 8, 6), 8, 6)
    xx, yy = np.meshgrid(grid.x, grid.y)
    field = 2 + np.cos(2 * np.pi * xx / 8) + 0.7 * np.sin(2 * np.pi * yy / 6)
    # A sampled Fourier mode is reconstructed linearly, not evaluated as the
    # analytic cosine between cells. At the seam it averages adjacent centres.
    corners = np.array([[0, 0], [8, 0], [0, 6], [8, 6]])
    expected_corner = np.mean(field[np.ix_([0, -1], [0, -1])])
    np.testing.assert_allclose(bilinear_interpolate(field, grid, corners, "periodic"), expected_corner)
    points = np.array([[0, 2.5], [8, 2.5], [3.5, 0], [3.5, 6]])
    expected = [(field[2, -1] + field[2, 0]) / 2] * 2 + [(field[-1, 3] + field[0, 3]) / 2] * 2
    np.testing.assert_allclose(bilinear_interpolate(field, grid, points, "periodic"), expected)
    near = np.array([[1e-9, 2.3], [8 - 1e-9, 2.3]])
    assert np.diff(bilinear_interpolate(field, grid, near, "periodic"))[0] == pytest.approx(0, abs=3e-9)


@pytest.mark.parametrize("boundary", ["zero_flux", "open", "periodic"])
def test_sampling_rejects_truly_outside_and_bad_points(boundary):
    grid = Grid((0, 0, 8, 6), 8, 6)
    for points in ([[-1e-8, 3]], [[8 + 1e-8, 3]], [[2, -1e-8]], [[2, 6 + 1e-8]], [[np.nan, 2]], [1, 2]):
        with pytest.raises(ValueError):
            bilinear_interpolate(np.ones((6, 8)), grid, points, boundary)
    assert bilinear_interpolate(np.ones((6, 8)), grid, np.empty((0, 2)), boundary).shape == (0,)


@pytest.mark.parametrize("boundary", ["zero_flux", "open"])
def test_nonperiodic_wall_sampling_is_explicit_constant_extension(boundary):
    grid = Grid((0, 0, 8, 6), 8, 6)
    field = np.arange(48).reshape(6, 8)
    points = np.array([[0, 0], [8, 6], [0, 2.5], [3.5, 6]])
    np.testing.assert_array_equal(bilinear_interpolate(field, grid, points, boundary),
                                  [field[0, 0], field[-1, -1], field[2, 0], field[-1, 3]])
