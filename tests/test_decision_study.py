"""Scientific invariants, independent small-graph answers and bundle integrity."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.diffusion import Grid, bilinear_interpolate
from backend.app.research.decision import (candidate_bounds, corridor_network, enumerate_simple_paths,
    integral_error_bound, minimizers, path_costs, static_graph_interval_bound)
from backend.app.routing import RoadNetwork, shortest_path
from scripts.run_decision_experiments import (DEFAULT_CONFIG, analytic_quadrature_bounds,
    exact_edge_exposures, run_study, validate_config, write_bundle)


def tiny_graph() -> RoadNetwork:
    points = {"s": [0, 0], "a": [1, 1], "t": [2, 0], "z": [3, 3]}
    pairs = [("s", "a", 0), ("s", "a", 1), ("a", "t", 0),
             ("s", "t", 0), ("a", "s", 0), ("a", "a", 0)]
    return RoadNetwork({"nodes": [{"id": n, "x": p[0], "y": p[1]} for n, p in points.items()],
        "edges": [{"u": u, "v": v, "key": key,
                   "coordinates": [points[u], [1.5, 1.5], points[v]] if u == v else [points[u], points[v]]}
                  for u, v, key in pairs]})


def test_simple_path_oracle_retains_parallel_edges_excludes_cycles_and_reports_unreachable():
    graph = tiny_graph()
    candidates = enumerate_simple_paths(graph, "s", "t")
    assert candidates.paths == ((0, 2), (1, 2), (3,))
    assert enumerate_simple_paths(graph, "s", "z").paths == ()
    assert enumerate_simple_paths(graph, "s", "s").paths == ((),)
    costs = path_costs(candidates, [1, 2, 2, 3, 0, 0])
    np.testing.assert_equal(costs, [3, 4, 3])
    assert minimizers(costs) == [0, 2]
    assert candidates.to_dict(graph)["complete"] is True
    assert len(candidates.graph_hash) == len(candidates.candidate_hash) == 64
    # Increasing the second parallel edge's cost never collapses its identity.
    changed = enumerate_simple_paths(graph, "s", "t")
    assert changed.candidate_hash == candidates.candidate_hash


def test_oracle_limits_never_return_incomplete_claims():
    with pytest.raises(ValueError, match="path budget"):
        enumerate_simple_paths(tiny_graph(), "s", "t", max_paths=2)
    with pytest.raises(ValueError, match="node budget"):
        enumerate_simple_paths(tiny_graph(), "s", "t", max_nodes=3)
    with pytest.raises(ValueError, match="positive integers"):
        enumerate_simple_paths(tiny_graph(), "s", "t", max_nodes=True)
    with pytest.raises(ValueError, match="At least one"):
        minimizers([])


def test_production_search_matches_independent_complete_oracle():
    graph = tiny_graph()
    candidates = enumerate_simple_paths(graph, "s", "t")
    speed = 1.4
    travel_times = np.array([edge["length_m"] / speed for edge in graph.edges])
    rng = np.random.default_rng(617)
    for weight in (0, .3, 2):
        exposures = rng.uniform(0, 10, len(graph.edges))
        exact_costs = path_costs(candidates, travel_times + weight * exposures)
        for algorithm in ("astar", "dijkstra"):
            result = shortest_path(graph, "s", "t", exposures, weight, algorithm, speed)
            assert result["objective"] == pytest.approx(exact_costs.min(), rel=1e-14)
            assert tuple(result["edge_indices"]) in [candidates.paths[i] for i in minimizers(exact_costs)]


def test_uniform_rule_heterogeneous_intervals_and_shared_error_cancellation():
    # Reference costs 10,11; approximations share a +5 bias. Marginal bounds
    # alone overlap, while the exactly preserved difference certifies ordering.
    certificate = candidate_bounds([15, 16], [5, 5], 0,
                                    difference_error_bounds=[0, 0], evidence="assumed")
    assert certificate["two_epsilon_plus_eta"] == 10
    assert not certificate["strict_interval_separation"]
    assert certificate["direct_difference"]["strict_separation"]
    assert certificate["direct_difference"]["regret_bound"] == 0
    separated = candidate_bounds([1, 5, 8], [.2, .4, .1], 0)
    assert separated["strict_interval_separation"]
    # eta is measured against the full finite approximate set.
    approximate_search = candidate_bounds([3, 1], [1, 1], 0, search_gap=2)
    assert approximate_search["two_epsilon_plus_eta"] == 4
    assert candidate_bounds([1, 2], [.1, .1], 0, evidence="empirical")["bound_status"] == "empirical_only"


@pytest.mark.parametrize("costs,errors,selected,kwargs", [
    ([1, 2], [.1], 0, {}),  # selected-path-only error cannot certify the set
    ([3, 1], [.1, .1], 0, {"search_gap": 1}),
    ([1, 2], [-.1, .1], 0, {}),
    ([1, 2], [.1, .1], 1, {"difference_error_bounds": [0, 1]}),
    ([1, float("nan")], [.1, .1], 0, {}),
])
def test_invalid_certificate_premises_rejected(costs, errors, selected, kwargs):
    with pytest.raises(ValueError):
        candidate_bounds(costs, errors, selected, **kwargs)


def test_candidate_bounds_cover_reference_regret_for_many_full_finite_sets():
    rng = np.random.default_rng(1987)
    for _ in range(40):
        exact = rng.uniform(0, 30, 10)
        error = rng.uniform(.01, 2, 10)
        approximate = exact + rng.uniform(-1, 1, 10) * error
        selected = int(np.argmin(approximate))
        result = candidate_bounds(approximate, error, selected)
        regret = exact[selected] - exact.min()
        assert regret <= result["two_epsilon_plus_eta"] + 1e-12
        assert regret <= result["interval_regret_bound"] + 1e-12


def test_graph_interval_bound_matches_exhaustive_oracle_and_uses_nonnegative_prior():
    network = tiny_graph()
    lower = [-1, 1, 2, 4, 0, 0]
    upper = [2, 3, 3, 7, 2, 1]
    result = static_graph_interval_bound(network, "s", "t", lower, upper, [3])
    paths = enumerate_simple_paths(network, "s", "t")
    assert result["lower_graph_optimum"] == min(path_costs(paths, np.maximum(lower, 0))) == 2
    assert result["selected_upper"] == 7
    assert result["regret_bound"] == 5
    rng = np.random.default_rng(42)
    for _ in range(25):
        actual = np.maximum(lower, 0) + rng.random(6) * (np.array(upper) - np.maximum(lower, 0))
        actual_regret = actual[3] - min(path_costs(paths, actual))
        assert actual_regret <= result["regret_bound"] + 1e-12
    with pytest.raises(ValueError, match="Negative lower"):
        static_graph_interval_bound(network, "s", "t", lower, upper, [3], nonnegative_cost_prior=False)
    with pytest.raises(ValueError, match="disconnected"):
        static_graph_interval_bound(network, "s", "t", lower, upper, [2])


def test_fixed_path_sup_bound_scales_with_time_and_preference():
    bound = integral_error_bound([2, 8], .05, lambda_weight=3)
    np.testing.assert_allclose(bound["exposure"], [.1, .4])
    np.testing.assert_allclose(bound["objective"], [.3, 1.2])
    with pytest.raises(ValueError):
        integral_error_bound(1, -.01)


def test_analytic_polyline_integral_and_trapezoid_remainder_against_independent_quadrature():
    from scipy.integrate import quad
    network, _ = corridor_network((0, 0, 1000, 1000))
    frequency = np.pi / 1000
    exact = exact_edge_exposures(network, wave_number=frequency, amplitude=.6,
                                 mean=1, ymin=0, speed_mps=1.4)
    independently_integrated = []
    for edge in network.edges:
        a, b = np.asarray(edge["coordinates"])
        value, _ = quad(lambda f: 1 + .6 * np.cos(frequency * (a[1] + f * (b[1] - a[1]))), 0, 1,
                        epsabs=1e-12, epsrel=1e-12)
        independently_integrated.append(edge["length_m"] / 1.4 * value)
    np.testing.assert_allclose(exact, independently_integrated, rtol=1e-14, atol=1e-12)
    grid = Grid((0, 0, 1000, 1000), 8, 8)
    points, weights, owners = network._sampling(grid.h / 2)
    numerical_quadrature = np.bincount(owners, weights=weights / 1.4 * (1 + .6 * np.cos(frequency * points[:, 1])))
    remainder = analytic_quadrature_bounds(network, grid, wave_number=frequency, amplitude=.6, speed_mps=1.4)
    assert np.all(abs(exact - numerical_quadrature) <= remainder + 1e-12)


@pytest.fixture(scope="module")
def study():
    config = json.loads(DEFAULT_CONFIG.read_text())
    return config, run_study(config)


def test_actual_pde_counterexamples_ties_and_bounds(study):
    config, (report, raw, _) = study
    cases = {case["id"]: case for case in report["cases"]}
    large = cases["large_error_same_route"]
    small = cases["small_error_wrong_route"]
    assert large["selected_is_reference_optimal_within_tolerance"]
    assert not small["selected_is_reference_optimal_within_tolerance"]
    assert large["field_error"]["nodal_linf"] > 100 * small["field_error"]["nodal_linf"]
    assert small["reference_regret_s"] > 0
    assert cases["reference_tie"]["reference_minimizers"] == [0, 1]
    assert cases["reference_tie"]["reference_regret_s"] < 1e-9
    assert 1e-9 < cases["reference_near_tie"]["reference_cost_gap_s"] < 1e-4
    assert cases["positive_regret"]["reference_regret_s"] > 10
    for case in cases.values():
        assert case["input_and_output_fields_unmodified"]
        assert case["reference_regret_s"] <= case["static_graph_certificate"]["regret_bound"] + 1e-9
        assert case["reference_regret_s"] <= case["candidate_certificate"]["two_epsilon_plus_eta"] + 1e-9
        for route in case["paths"]:
            assert abs(route["approximate_cost_s"] - route["reference_cost_s"]) <= route["objective_error_bound_s"] + 1e-9
        exact = raw[f"{case['id']}__continuous_exact_nodes"]
        assert exact.min() > 0
        # Independent dense samples include both physical walls, where cell-
        # centre constant extension contributes reconstruction error.
        grid = Grid(tuple(case["grid"]["bounds"]), case["grid"]["nx"], case["grid"]["ny"])
        yy = np.linspace(grid.bounds[1], grid.bounds[3], 1001)
        points = np.column_stack((np.full_like(yy, grid.bounds[0]), yy))
        reconstructed = bilinear_interpolate(raw[f"{case['id']}__numerical"], grid, points)
        frequency = config["mode_y"] * np.pi / (grid.bounds[3] - grid.bounds[1])
        continuous = config["mean_concentration"] + config["mode_amplitude"] * np.exp(
            -config["kappa_m2_s"] * frequency**2 * case["snapshot_s"]) * np.cos(frequency * (yy - grid.bounds[1]))
        assert np.max(abs(reconstructed - continuous)) <= case["field_error"]["reconstructed_numerical_vs_continuous_sup_bound"] + 1e-12


def test_configuration_rejects_unknown_keys_and_unresolved_modes(study):
    config, _ = study
    bad = copy.deepcopy(config)
    bad["hidden_policy"] = True
    with pytest.raises(ValueError, match="top-level"):
        validate_config(bad)
    bad = copy.deepcopy(config)
    bad["mode_y"] = 8
    with pytest.raises(ValueError, match="resolved"):
        validate_config(bad)
    bad = copy.deepcopy(config)
    bad["cases"][0]["id"] = "../escape"
    with pytest.raises(ValueError, match="safe"):
        validate_config(bad)
    bad = copy.deepcopy(config)
    bad["cases"][0]["lambda_weight"] = True
    with pytest.raises(ValueError, match="not a boolean"):
        validate_config(bad)
    bad = copy.deepcopy(config)
    bad["kappa_m2_s"] = "20"
    with pytest.raises(ValueError, match="finite real"):
        validate_config(bad)


def test_saved_bundle_integrity_raw_replay_and_no_overwrite(tmp_path: Path, study):
    config, _ = study
    output = tmp_path / "decision"
    report = write_bundle(config, output)
    manifest = json.loads((output / "manifest.json").read_text())
    for relative, info in manifest["files"].items():
        data = (output / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == info["sha256"]
        assert len(data) == info["bytes"]
    assert (output / "decision_evidence.png").stat().st_size > 1000
    assert (output / "source_snapshot/scripts/run_decision_experiments.py").exists()
    with np.load(output / "raw_fields.npz", allow_pickle=False) as arrays:
        for case in report["cases"]:
            np.testing.assert_allclose(arrays[f"{case['id']}__continuous_exact_edge_exposure"],
                                        case["edge_error_bounds"]["exact_exposure"], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        write_bundle(config, output)
    assert not list(tmp_path.glob(".*.tmp-*"))
