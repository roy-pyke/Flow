"""Independent finite-graph checks for time-dependent reference routing."""
from __future__ import annotations

import math

import numpy as np
import pytest

from backend.app.research.dynamic_routing import solve_time_expanded
from backend.app.routing import RoadNetwork


def graph(points, pairs):
    return RoadNetwork({"nodes": [{"id": node, "x": xy[0], "y": xy[1]}
                                   for node, xy in points.items()],
                        "edges": [{"u": u, "v": v, "key": key,
                                   "coordinates": [points[u], points[v]]}
                                  for u, v, key in pairs]})


def line():
    return graph({"s": (0, 0), "a": (.5, 0), "t": (1, 0)},
                 [("s", "a", 0), ("a", "t", 0)])


def solve(network, **kwargs):
    defaults = dict(departure_time_s=0, horizon_time_s=4, time_step_s=1,
                    speed_mps=1, edge_exposure=lambda i, time: 0,
                    waiting_exposure=lambda node, start, end: 0)
    defaults.update(kwargs)
    return solve_time_expanded(network, "s", "t", **defaults)


def brute_force(network, departure, horizon, step, exposure, waiting, weight, allow_wait):
    """Enumerate the full action tree, retaining every history without labels.

    This independently constructs lattice times by multiplication and compares
    physical arrivals against the list, without using the solver's rounding or
    relaxation implementation. Test instances keep this exponential tree tiny.
    """
    lattice = [i * step for i in range(100) if departure <= i * step <= horizon]
    terminal_costs = []

    def visit(node, time, integral):
        if node == "t":
            terminal_costs.append(time - departure + weight * integral)
            return
        for i in network.adjacency[node]:
            arrival = time + network.edges[i]["length_m"]
            if arrival > horizon:
                continue
            neighbor = network.edges[i]["v"]
            rounded = arrival if neighbor == "t" else next((t for t in lattice if t >= arrival), None)
            if rounded is None:
                continue
            wait = waiting(neighbor, arrival, rounded) if rounded > arrival else 0
            visit(neighbor, rounded, integral + exposure(i, time) + wait)
        if allow_wait:
            later = next((t for t in lattice if t > time), None)
            if later is not None:
                visit(node, later, integral + waiting(node, time, later))

    visit("s", departure, 0)
    return min(terminal_costs) if terminal_costs else None


@pytest.mark.parametrize("allow_wait", [False, True])
@pytest.mark.parametrize("departure", [0, .25])
@pytest.mark.parametrize("weight", [0, .7, 3])
def test_matches_independent_exhaustive_action_tree(allow_wait, departure, weight):
    network = graph({"s": (0, 0), "a": (.5, 0), "t": (1, 0)},
                    [("s", "a", 0), ("s", "a", 1), ("a", "t", 0),
                     ("s", "t", 0), ("a", "s", 0)])
    exposure = lambda i, time: .1 + (i + 1) * (1 + math.cos(time * 1.7))
    waiting = lambda node, begin, end: (end - begin) * {"s": 1, "a": 2, "t": 3}[node]
    expected = brute_force(network, departure, 3, .5, exposure, waiting, weight, allow_wait)
    result = solve(network, departure_time_s=departure, horizon_time_s=3, time_step_s=.5,
                   edge_exposure=exposure, waiting_exposure=waiting,
                   lambda_weight=weight, allow_wait=allow_wait)
    assert result["objective"] == pytest.approx(expected, rel=2e-15)
    assert result["scope"] == "optimal_on_this_declared_rounded_finite_graph"
    assert result["complete"] and not result["continuous_time_optimum"]
    assert result["visited_states"] == result["discovered_states"]
    assert result["arrival_time_s"] <= 3
    assert result["elapsed_time_s"] == pytest.approx(result["travel_time_s"] + result["wait_time_s"])
    assert result["E"] == pytest.approx(sum(action["exposure"] for action in result["actions"]))
    assert result["J"] == pytest.approx(result["elapsed_time_s"] + weight * result["E"])
    assert result["J"] == pytest.approx(sum(action["objective"] for action in result["actions"]))


def test_rounding_wait_is_explicit_charged_and_destination_is_not_rounded():
    result = solve(line(), allow_wait=False, lambda_weight=1,
                   edge_exposure=lambda i, time: .5,
                   waiting_exposure=lambda node, begin, end: 2 * (end - begin))
    assert result["edge_indices"] == [0, 1]
    assert result["arrival_time_s"] == 1.5
    assert result["travel_time_s"] == 1
    assert result["rounding_wait_time_s"] == .5
    assert result["voluntary_wait_time_s"] == 0
    assert result["exposure"] == 2
    assert result["objective"] == 3.5
    assert [(a["type"], a.get("reason")) for a in result["actions"]] == [
        ("travel", None), ("wait", "rounding"), ("travel", None)]
    assert result["edges"][0]["actual_arrival_time_s"] == .5
    assert result["edges"][0]["rounded_arrival_time_s"] == 1
    assert result["edges"][1]["rounded_arrival_time_s"] == 1.5


def test_voluntary_waiting_can_change_optimum_and_is_costed():
    network = graph({"s": (0, 0), "t": (1, 0)}, [("s", "t", 0)])
    callback = lambda i, time: 100 if time < 2 else 0
    result = solve(network, lambda_weight=1, edge_exposure=callback,
                   waiting_exposure=lambda node, begin, end: 2 * (end - begin))
    assert result["arrival_time_s"] == 3
    assert result["voluntary_wait_time_s"] == 2
    assert result["edges"][0]["voluntary_wait_before_s"] == 2
    assert result["E"] == 4
    assert result["J"] == 7
    assert solve(network, lambda_weight=1, edge_exposure=callback, allow_wait=False)["J"] == 101
    # Waiting remains optional: a high stationary concentration can outweigh it.
    expensive = solve(network, lambda_weight=1, edge_exposure=callback,
                      waiting_exposure=lambda node, begin, end: 100 * (end - begin))
    assert expensive["arrival_time_s"] == 1
    assert expensive["J"] == 101


def test_one_spatial_label_is_invalid_even_with_constant_fifo_travel_times():
    network = graph({"s": (0, 0), "a": (1, 0), "b": (0, 1), "t": (2, 0)},
                    [("s", "a", 0), ("s", "b", 0), ("b", "a", 0), ("a", "t", 0)])
    exposure = lambda i, departure: 100 if i == 3 and departure < 3 else 0
    result = solve(network, lambda_weight=1, edge_exposure=exposure, allow_wait=False)
    # The locally worse arrival at a at time 3 must survive alongside time 1.
    # A single lowest-cost label at a would discard it and return cost 102.
    assert result["edge_indices"] == [1, 2, 3]
    assert result["J"] == 4
    assert result["edges"][-1]["departure_time_s"] == 3
    assert result["rounding_wait_time_s"] == pytest.approx(2 - math.sqrt(2))


def test_off_lattice_departure_and_strict_nonterminal_ceiling():
    result = solve(line(), departure_time_s=.25, allow_wait=False)
    assert result["edges"][0]["departure_time_s"] == .25
    assert result["edges"][0]["actual_arrival_time_s"] == .75
    assert result["edges"][0]["rounded_arrival_time_s"] == 1
    assert result["elapsed_time_s"] == 1.25
    assert result["time_grid"]["origin_s"] == 0
    assert not result["time_grid"]["initial_departure_on_lattice"]
    # A representably later arrival cannot be 'isclose'-rounded backward.
    x = np.nextafter(1.0, np.inf)
    network = graph({"s": (0, 0), "a": (x, 0), "t": (x + .5, 0)},
                    [("s", "a", 0), ("a", "t", 0)])
    result = solve(network, allow_wait=False)
    assert result["edges"][0]["actual_arrival_time_s"] == x
    assert result["edges"][0]["rounded_arrival_time_s"] == 2
    assert result["arrival_time_s"] == 2.5


def test_destination_horizon_is_exact_and_unreachable_result_has_no_cost_claim():
    feasible = solve(line(), horizon_time_s=1.5, allow_wait=False)
    assert feasible["status"] == "optimal" and feasible["arrival_time_s"] == 1.5
    infeasible = solve(line(), horizon_time_s=np.nextafter(1.5, -np.inf), allow_wait=False)
    assert infeasible["status"] == "unreachable" and infeasible["J"] is None
    assert infeasible["complete"]
    disconnected = graph({"s": (0, 0), "t": (1, 0)}, [])
    result = solve(disconnected)
    assert result["status"] == "unreachable" and result["actions"] == []


def test_identical_endpoints_return_empty_route_without_invoking_callbacks():
    def forbidden(*args):
        raise AssertionError("An empty route must not invoke integrals.")
    result = solve_time_expanded(line(), "s", "s", departure_time_s=.25,
                                horizon_time_s=100, time_step_s=1, edge_exposure=forbidden,
                                waiting_exposure=forbidden, max_states=1)
    assert result["status"] == "optimal" and result["J"] == 0
    assert result["arrival_time_s"] == .25 and result["nodes"] == ["s"]


def test_parallel_edges_preserve_key_types_and_ties_are_deterministic():
    network = graph({"s": (0, 0), "t": (1, 0)}, [("s", "t", "1"), ("s", "t", 1)])
    result = solve(network, lambda_weight=1, edge_exposure=lambda i, time: [10, 2][i])
    assert result["edge_indices"] == [1]
    assert result["edge_ids"] == [{"u": "s", "v": "t", "key": 1, "key_type": "int"}]
    tied = [solve(network)["edge_ids"] for _ in range(3)]
    assert tied[0] == tied[1] == tied[2]
    # Sorting by typed identity is stable when file edge order changes.
    reverse = graph({"s": (0, 0), "t": (1, 0)}, [("s", "t", 1), ("s", "t", "1")])
    assert solve(reverse)["edge_ids"] == tied[0]


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_rejects_invalid_exposure_even_when_lambda_is_zero(value):
    with pytest.raises(ValueError, match="Edge exposure"):
        solve(line(), edge_exposure=lambda i, time: value)
    with pytest.raises(ValueError, match="Waiting exposure"):
        solve(line(), waiting_exposure=lambda node, begin, end: value)


@pytest.mark.parametrize("kwargs", [
    {"departure_time_s": -1}, {"horizon_time_s": -1}, {"time_step_s": 0},
    {"time_step_s": math.nan}, {"speed_mps": 0}, {"lambda_weight": -1},
    {"lambda_weight": math.inf}, {"allow_wait": 1}, {"max_states": 0},
    {"max_transitions": True}, {"departure_time_s": 5}, {"time_step_s": 1e-300},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        solve(line(), **kwargs)


def test_state_preflight_and_transition_exhaustion_never_return_partial_answers():
    def forbidden(*args):
        raise AssertionError("Preflight must run before exposure callbacks.")
    with pytest.raises(ValueError, match="state budget exceeded in preflight"):
        solve(line(), max_states=2, edge_exposure=forbidden, waiting_exposure=forbidden)
    with pytest.raises(ValueError, match="transition budget exhausted"):
        solve(line(), max_transitions=1)
    network = graph({"s": (0, 0), "t": (1, 0)}, [("s", "t", 0), ("s", "t", 1)])
    # Even finding a target before budget exhaustion cannot justify completion.
    with pytest.raises(ValueError, match="transition budget exhausted"):
        solve(network, max_transitions=1)


def test_duplicate_typed_edge_identity_is_rejected():
    network = graph({"s": (0, 0), "t": (1, 0)}, [("s", "t", 0), ("s", "t", 0)])
    with pytest.raises(ValueError, match="identities must be unique"):
        solve(network)


def test_accumulated_objective_overflow_is_rejected():
    with pytest.raises(ValueError, match="overflowed"):
        solve(line(), lambda_weight=1e308, edge_exposure=lambda i, time: 1e308)
    with pytest.raises(ValueError, match="overflowed"):
        solve(line(), edge_exposure=lambda i, time: 1e308,
              waiting_exposure=lambda node, begin, end: 1e308)
