"""Independent temporal-observation contracts and integration invariants."""

import math

import numpy as np
from numpy.polynomial import Polynomial
import pytest
from scipy.integrate import quad

from backend.app.diffusion import Grid, bilinear_interpolate
from backend.app.observations import (
    PathTrajectory, build_edge_observer, build_path_trajectory, build_stationary_trajectory,
)
from backend.app.routing import RoadNetwork
from backend.app.temporal_observations import build_trajectory_observer, sample_time_series


def network():
    return RoadNetwork({"nodes": [{"id": "a", "x": 0.5, "y": 0.5},
                                  {"id": "b", "x": 1.5, "y": 0.5}],
                        "edges": [{"u": "a", "v": "b", "key": 1,
                                   "coordinates": [[0.5, 0.5], [1.0, 1.2], [1.5, 0.5]]},
                                  {"u": "b", "v": "a", "key": 2,
                                   "coordinates": [[1.5, 0.5], [0.5, 0.5]]}],
                        "crs": "EPSG:32610"})


def curve():
    return PathTrajectory([[0.2, 0.2], [1.6, 1.3], [1.6, 1.3], [0.3, 1.5]],
                          [0.2, 1.4, 2.5, 4.8], (), {})


def analytic_frames(grid, times):
    x, y = np.meshgrid(grid.x, grid.y)
    return np.array([1 + 0.2 * x + 0.3 * y + 0.1 * t + 0.02 * x * y * t for t in times])


def exact_polynomial_integral(trajectory):
    result = 0.0
    for a, b, start, end in zip(trajectory.vertex_xy[:-1], trajectory.vertex_xy[1:],
                               trajectory.vertex_times_s[:-1], trajectory.vertex_times_s[1:]):
        x, y = [Polynomial([a[axis], b[axis] - a[axis]]) for axis in (0, 1)]
        time = Polynomial([start, end - start])
        polynomial = 1 + 0.2 * x + 0.3 * y + 0.1 * time + 0.02 * x * y * time
        primitive = polynomial.integ()
        result += (end - start) * (primitive(1) - primitive(0))
    return result


@pytest.mark.parametrize("boundary", ["zero_flux", "open", "periodic"])
@pytest.mark.parametrize("quadrature", ["gauss2_split", "trapezoid"])
def test_nonnegative_weights_constant_integral_includes_waits(boundary, quadrature):
    grid = Grid((0, 0, 2, 2), 7, 5)
    times = np.array([0, 0.6, 1.7, 2.7, 5.0])
    trajectory = curve()
    observer = build_trajectory_observer(grid, times, trajectory, boundary, quadrature)
    assert observer.matrix.format == "csr"
    assert observer.matrix.shape == (1, len(times) * 35)
    assert np.all(observer.matrix.data >= 0)
    assert observer.matrix.sum() == pytest.approx(4.6, rel=2e-15)
    assert observer.apply(np.ones((5, 5, 7))) == pytest.approx(4.6, rel=2e-15)
    assert observer.apply(np.full((5, 5, 7), -3.1)) == pytest.approx(-3.1 * 4.6)
    assert observer.metadata["sample_count"] > 0
    assert observer.metadata["includes_waiting"]
    assert observer.metadata["field_order"] == "C:frame,y,x"
    assert not observer.times_s.flags.writeable
    for array in (observer.matrix.data, observer.matrix.indices, observer.matrix.indptr):
        assert not array.flags.writeable


def test_gauss_is_exact_for_cubic_space_time_composition_with_nonuniform_frames():
    grid = Grid((0, 0, 2, 2), 14, 17)
    times = np.array([0, 0.6, 1.7, 2.7, 5.0])
    trajectory = curve()
    observer = build_trajectory_observer(grid, times, trajectory)
    expected = exact_polynomial_integral(trajectory)
    assert observer.apply(analytic_frames(grid, times)) == pytest.approx(expected, rel=3e-15)
    # A much sparser temporal representation reconstructs this same affine-in-
    # time field, and must produce the same exact integral.
    sparse_times = np.array([0.0, 5.0])
    sparse = build_trajectory_observer(grid, sparse_times, trajectory)
    assert sparse.apply(analytic_frames(grid, sparse_times)) == pytest.approx(expected, rel=3e-15)


def test_sampling_matches_analytic_values_including_frame_endpoints_and_scalar():
    grid = Grid((0, 0, 2, 2), 10, 12)
    times = np.array([-1, 0.0, 1.7, 5.0])
    queries = np.array([[-1, 0.0], [0.9, 5.0]])
    positions = np.array([[[0.2, 0.4], [0.8, 0.6]], [[1.5, 1.7], [0.3, 1.2]]])
    x, y = positions[..., 0], positions[..., 1]
    expected = 1 + 0.2*x + 0.3*y + 0.1*queries + 0.02*x*y*queries
    frames = analytic_frames(grid, times)
    np.testing.assert_allclose(sample_time_series(grid, times, frames, positions, queries), expected, rtol=3e-15)
    scalar = sample_time_series(grid, times, frames, [0.2, 0.4], -1.0)
    assert scalar.shape == ()
    assert scalar == pytest.approx(expected[0, 0])
    assert sample_time_series(grid, times, frames, np.empty((0, 2)), np.empty(0)).shape == (0,)


def test_affine_time_integral_on_stationary_path_and_nonuniform_times():
    grid = Grid((0, 0, 2, 2), 4, 6)
    times = np.array([-3, -0.5, 1, 8])
    frames = np.broadcast_to((2 + 3*times)[:, None, None], (4, 6, 4))
    trajectory = build_stationary_trajectory([0, 2], -2, 7)
    expected = 2 * 9 + 1.5 * (7**2 - (-2)**2)
    for rule in ("gauss2_split", "trapezoid"):
        observer = build_trajectory_observer(grid, times, trajectory, quadrature=rule)
        assert observer.apply(frames) == pytest.approx(expected, rel=3e-15)


@pytest.mark.parametrize("boundary", ["zero_flux", "open", "periodic"])
def test_frozen_frames_match_static_polyline_integral(boundary):
    road = network()
    grid = Grid((0, 0, 2, 2), 9, 13)
    trajectory = build_path_trajectory(road, [0, 1, 0], speed_mps=1.4, departure_time_s=0.3)
    times = np.array([0, 0.4, 1.7, trajectory.arrival_time_s, trajectory.arrival_time_s + 1])
    field = np.random.default_rng(7).normal(size=(13, 9))
    frames = np.repeat(field[None], len(times), axis=0)
    temporal = build_trajectory_observer(grid, times, trajectory, boundary).apply(frames)
    static = build_edge_observer(road, grid, boundary, quadrature="gauss2_grid").apply(field)
    assert temporal == pytest.approx(2 * static[0] + static[1], abs=3e-15)


@pytest.mark.parametrize("boundary", ["zero_flux", "open", "periodic"])
def test_boundary_sampling_matches_spatial_policy_and_exact_wait_integral(boundary):
    grid = Grid((0, 0, 2, 2), 4, 4)
    times = np.array([0.0, 0.7, 3.0])
    frames = np.random.default_rng(18).normal(size=(3, 4, 4))
    points = np.array([[0, 0], [2, 2], [0, 2]])
    expected = bilinear_interpolate(frames[1], grid, points, boundary)
    np.testing.assert_allclose(sample_time_series(grid, times, frames, points, np.full(3, 0.7), boundary), expected)
    trajectory = build_stationary_trajectory(points[0], 0, 3)
    values = np.array([bilinear_interpolate(f, grid, points[:1], boundary)[0] for f in frames])
    expected_integral = np.sum(np.diff(times) * (values[:-1] + values[1:]) / 2)
    assert build_trajectory_observer(grid, times, trajectory, boundary).apply(frames) == pytest.approx(expected_integral)


def test_piecewise_spatial_and_temporal_breaks_match_independent_adaptive_quadrature():
    grid = Grid((0, 0, 2, 2), 4, 5)
    trajectory = PathTrajectory([[0, 0], [2, 2]], [0.1, 2.9], (), {})
    times = np.array([0, 0.8, 1.1, 3.0])
    frames = np.random.default_rng(71).normal(size=(4, 5, 4))
    # Independent evaluation calls the established spatial interpolator and
    # directly interpolates two frame values, rather than applying the CSR.
    cuts = sorted(set([0.1, 2.9, 0.8, 1.1] + [0.1 + 1.4*x for x in grid.x]
                      + [0.1 + 1.4*y for y in grid.y]))
    def integrand(t):
        left = min(np.searchsorted(times, t, side="right") - 1, len(times) - 2)
        fraction = (t - times[left]) / (times[left+1] - times[left])
        point = np.array([[(t-0.1)/1.4, (t-0.1)/1.4]])
        values = [bilinear_interpolate(frames[i], grid, point, "periodic")[0] for i in (left, left+1)]
        return (1-fraction)*values[0] + fraction*values[1]
    expected = sum(quad(integrand, a, b, epsabs=1e-12)[0] for a, b in zip(cuts[:-1], cuts[1:]))
    observer = build_trajectory_observer(grid, times, trajectory, "periodic")
    assert observer.apply(frames) == pytest.approx(expected, rel=3e-14, abs=3e-14)


def test_waits_before_every_edge_occurrence_and_at_destination():
    road = network()
    waits = [0.25, 0.5, 0.75, 1.0]
    trajectory = build_path_trajectory(road, [0, 1, 0], speed_mps=2, departure_time_s=4, waits_s=waits)
    travel = (2*road.edges[0]["length_m"] + road.edges[1]["length_m"]) / 2
    assert trajectory.arrival_time_s == pytest.approx(4 + travel + sum(waits))
    assert trajectory.metadata["waiting_time_s"] == 2.5
    assert trajectory.metadata["waiting"]
    assert trajectory.edge_indices == (0, 1, 0)
    np.testing.assert_allclose(trajectory.positions([4, 4.2]), [[0.5, 0.5], [0.5, 0.5]])
    for edge, leave, arrive in zip([0, 1, 0], trajectory.metadata["edge_departure_times_s"],
                                  trajectory.metadata["edge_arrival_times_s"]):
        assert arrive - leave == pytest.approx(road.edges[edge]["length_m"] / 2)
    np.testing.assert_allclose(trajectory.positions(trajectory.arrival_time_s - 0.5), [1.5, 0.5])
    for bad in ([0], [-1, 0], [math.nan, 0], [0, math.inf], [[0, 0]]):
        with pytest.raises(ValueError, match="waits_s"):
            build_path_trajectory(road, [0], waits_s=bad)


@pytest.mark.parametrize("xy,times", [([], []), ([[0, 0]], [0, 1]), ([[0, 0], [1, 1]], [1, 1]),
                                       ([[0, 0], [1, 1]], [1, 0]), ([[0, 0], [1, 1]], [0, np.nan]),
                                       ([[0, np.inf]], [0]), ([[0]], [0])])
def test_malformed_direct_trajectory_is_rejected(xy, times):
    with pytest.raises(ValueError):
        PathTrajectory(xy, times, (), {})


def test_direct_trajectory_arrays_are_owned_readonly_copies_and_zero_duration_is_valid():
    xy, times = np.array([[0.3, 0.4], [0.5, 0.6]]), np.array([0.0, 1.0])
    trajectory = PathTrajectory(xy, times, (), {})
    xy[0] = 9
    times[0] = 9
    np.testing.assert_array_equal(trajectory.vertex_xy[0], [0.3, 0.4])
    assert trajectory.departure_time_s == 0
    assert not trajectory.vertex_xy.flags.writeable
    assert not trajectory.vertex_times_s.flags.writeable
    grid = Grid((0, 0, 1, 1), 4, 4)
    instant = build_stationary_trajectory([0.3, 0.4], 0.5, 0.5)
    observer = build_trajectory_observer(grid, [0, 1], instant)
    assert observer.metadata["sample_count"] == 0
    assert observer.matrix.nnz == 0
    assert observer.apply(np.ones((2, 4, 4))) == 0


def test_trapezoid_refines_independent_space_and_time_steps_with_mandatory_breaks():
    grid = Grid((0, 0, 2, 2), 6, 7)
    times = np.array([0, 0.6, 1.7, 2.7, 5.0])
    trajectory = curve()
    frames = analytic_frames(grid, times)
    exact = exact_polynomial_integral(trajectory)
    for keyword in ("spatial_step_m", "time_step_s"):
        observers = [build_trajectory_observer(grid, times, trajectory, quadrature="trapezoid", **{keyword: h})
                     for h in (0.15, 0.075, 0.0375)]
        errors = [abs(obs.apply(frames) - exact) for obs in observers]
        assert errors[2] < errors[1] < errors[0]
        assert errors[2] < errors[0] / 4
        assert len({obs.metadata["mandatory_panel_count"] for obs in observers}) == 1
        assert observers[2].metadata["sample_count"] > observers[0].metadata["sample_count"]
    # Spatial refinement alone cannot subdivide a stationary wait. Temporal
    # refinement can; these are independently controllable evaluation steps.
    wait = build_stationary_trajectory([0.5, 0.5], 0.0, 5.0)
    coarse = build_trajectory_observer(grid, times, wait, quadrature="trapezoid", spatial_step_m=1e-8)
    fine = build_trajectory_observer(grid, times, wait, quadrature="trapezoid", time_step_s=0.01)
    assert coarse.metadata["sample_count"] == 2 * (len(times) - 1)
    assert fine.metadata["sample_count"] > 500


@pytest.mark.parametrize("bad_times", [[0], [0, 0], [1, 0], [0, np.inf], [[0, 1]]])
def test_invalid_frame_times_rejected(bad_times):
    with pytest.raises(ValueError, match="Frame times"):
        build_trajectory_observer(Grid((0, 0, 1, 1), 2, 2), bad_times,
                                  build_stationary_trajectory([0.5, 0.5], 0, 0))


def test_temporal_coverage_is_strict_and_spatial_excursions_rejected():
    grid = Grid((0, 0, 2, 2), 4, 5)
    frames = np.ones((2, 5, 4))
    for time in (-1e-20, np.nextafter(1.0, 2.0), np.nan):
        with pytest.raises(ValueError, match="covered"):
            sample_time_series(grid, [0, 1], frames, [0.5, 0.5], time)
    for start, end in ((-1e-20, 0.5), (0.5, 1.0000000001)):
        with pytest.raises(ValueError, match="covered"):
            build_trajectory_observer(grid, [0, 1], build_stationary_trajectory([0.5, 0.5], start, end))
    for point in ([2.01, 1], [1, -0.01]):
        with pytest.raises(ValueError, match="outside"):
            build_trajectory_observer(grid, [0, 1], build_stationary_trajectory(point, 0, 1))


def test_invalid_shapes_boundaries_steps_and_evaluation_are_rejected():
    grid = Grid((0, 0, 2, 2), 4, 5)
    trajectory = build_stationary_trajectory([0.5, 0.5], 0, 1)
    for options in ({"boundary": "unknown"}, {"quadrature": "simpson"}, {"time_step_s": 0.1},
                    {"spatial_step_m": 0.1}, {"quadrature": "trapezoid", "time_step_s": 0},
                    {"quadrature": "trapezoid", "spatial_step_m": np.nan}):
        with pytest.raises(ValueError):
            build_trajectory_observer(grid, [0, 1], trajectory, **options)
    observer = build_trajectory_observer(grid, [0, 1], trajectory)
    for frames in (np.ones((2, 4, 5)), np.ones((5, 4)), np.full((2, 5, 4), np.inf)):
        with pytest.raises(ValueError, match="Frames"):
            observer.apply(frames)
    with pytest.raises(ValueError, match="Positions"):
        sample_time_series(grid, [0, 1], np.ones((2, 5, 4)), [[0.5, 0.5]], 0.5)
    with pytest.raises(ValueError, match="nonfinite"):
        long_wait = build_trajectory_observer(grid, [0, 5], build_stationary_trajectory([0.5, 0.5], 0, 5))
        long_wait.apply(np.full((2, 5, 4), np.finfo(float).max))


def test_sample_budget_rejects_before_large_grid_or_tiny_step_allocation():
    trajectory = PathTrajectory([[0, 0], [1, 1]], [0, 1], (), {})
    for grid, options in ((Grid((0, 0, 1, 1), 10**9, 2), {}),
                          (Grid((0, 0, 1, 1), 2, 2), {"quadrature": "trapezoid", "time_step_s": 1e-300}),
                          (Grid((0, 0, 1, 1), 2, 2), {"quadrature": "trapezoid", "spatial_step_m": 1e-300})):
        with pytest.raises(ValueError, match="max_samples"):
            build_trajectory_observer(grid, [0, 1], trajectory, max_samples=100, **options)
    many_vertices = PathTrajectory(np.zeros((101, 2)), np.linspace(0, 1, 101), (), {})
    with pytest.raises(ValueError, match="max_samples"):
        build_trajectory_observer(Grid((0, 0, 1, 1), 2, 2), [0, 1], many_vertices, max_samples=100)
    grid = Grid((0, 0, 1, 1), 2, 2)
    for bad in (0, -1, True, 1.2):
        with pytest.raises(ValueError, match="max_samples"):
            build_trajectory_observer(grid, [0, 1], trajectory, max_samples=bad)
    observer = build_trajectory_observer(grid, [0, 1], trajectory)
    count = observer.metadata["sample_count"]
    assert build_trajectory_observer(grid, [0, 1], trajectory, max_samples=count).metadata["sample_count"] == count
    with pytest.raises(ValueError, match="max_samples"):
        build_trajectory_observer(grid, [0, 1], trajectory, max_samples=count-1)


@pytest.mark.parametrize("departure", [0.0, 0.123456789, 17.3])
@pytest.mark.parametrize("first_wait", [0.0, 0.17])
def test_trajectory_edge_clock_matches_search_exactly_at_strict_horizon(departure, first_wait):
    points = [[0.1, 0.1], [0.31529526772962263, 0.7067003751060119],
              [0.5493078171597499, 0.5681219753894384], [0.9, 0.9]]
    road = RoadNetwork({"nodes": [{"id": "a", "x": 0.1, "y": 0.1},
                                  {"id": "b", "x": 0.9, "y": 0.9}],
                        "edges": [{"u": "a", "v": "b", "coordinates": points},
                                  {"u": "b", "v": "a", "coordinates": points[::-1]}]})
    waits = [first_wait, 0.29, 0.0, 0.11]
    trajectory = build_path_trajectory(road, [0, 1, 0], speed_mps=1.4,
                                       departure_time_s=departure, waits_s=waits)
    clock = departure
    for occurrence, edge in enumerate([0, 1, 0]):
        clock += waits[occurrence]
        assert trajectory.metadata["edge_departure_times_s"][occurrence] == clock
        clock += road.edges[edge]["length_m"] / 1.4
        assert trajectory.metadata["edge_arrival_times_s"][occurrence] == clock
    clock += waits[-1]
    assert trajectory.arrival_time_s == clock
    assert np.all(np.diff(trajectory.vertex_times_s) > 0)
    observer = build_trajectory_observer(Grid((0, 0, 1, 1), 6, 5), [departure, clock], trajectory)
    assert observer.apply(np.ones((2, 5, 6))) == pytest.approx(clock - departure)
    # Preserve the review's exact one-edge endpoint reproducer too.
    one_edge = build_path_trajectory(road, [0], speed_mps=1.4, departure_time_s=departure,
                                     waits_s=[first_wait, 0.0])
    end = (departure + first_wait) + road.edges[0]["length_m"] / 1.4
    assert one_edge.arrival_time_s == end
    build_trajectory_observer(Grid((0, 0, 1, 1), 6, 5), [departure, end], one_edge)
