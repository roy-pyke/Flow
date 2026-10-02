"""Exact reference search on a declared, finite, rounded time-expanded graph.

This is a small-instance research oracle, not a continuous-time optimum. The
global time lattice has origin zero. A nonterminal traversal arrives at its
physical arrival time, then waits until the next lattice time; that wait is
explicit and incurs both elapsed time and exposure. Destination arrivals are
never rounded. An off-lattice initial departure is a single additional state.
"""
from __future__ import annotations

import heapq
import json
import math
from typing import Callable

from ..routing import RoadNetwork


def _nonnegative(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number.")
    return number


def _identity(network: RoadNetwork, index: int) -> dict:
    edge = network.edges[index]
    key = edge["key"]
    # JSON's bool/int and string/int ambiguities must not merge parallel roads.
    if type(key) not in (str, int, float, bool, type(None)):
        raise ValueError("Edge keys must be JSON scalar values with stable types.")
    if isinstance(key, float) and not math.isfinite(key):
        raise ValueError("Edge keys must be finite JSON values.")
    return {"u": edge["u"], "v": edge["v"], "key": key, "key_type": type(key).__name__}


def solve_time_expanded(
    network: RoadNetwork, start: str, end: str, *, departure_time_s: float,
    horizon_time_s: float, time_step_s: float, speed_mps: float = 1.4,
    lambda_weight: float = 0.0, edge_exposure: Callable[[int, float], float],
    waiting_exposure: Callable[[str, float, float], float], allow_wait: bool = True,
    max_states: int = 100000, max_transitions: int = 1000000,
) -> dict:
    """Minimize elapsed time + lambda * exposure on the stated finite graph.

    Edge durations are their complete polyline lengths / constant speed. The
    callbacks must be deterministic integrals over the complete edge traversal
    and stationary interval, respectively, in compatible concentration-seconds
    units. Only feasible reachable transitions invoke them. Voluntary waiting
    advances one lattice step; mandatory rounding waits remain when
    ``allow_wait=False``. Time-state labels, not one label per spatial node, are
    retained. Strict binary64 feasibility comparisons use no ``isclose`` slack.

    State capacity is checked conservatively before callback evaluation. The
    transition budget counts feasible evaluated transitions and raises on
    exhaustion: an incomplete search never returns a partial optimality claim.
    Costs are evaluated in ordinary binary64, not certified interval arithmetic.
    """
    start, end = str(start), str(end)
    if start not in network.nodes or end not in network.nodes:
        raise ValueError("Start and destination must be graph nodes.")
    departure = _nonnegative(departure_time_s, "Departure time")
    horizon = _nonnegative(horizon_time_s, "Horizon time")
    step = _nonnegative(time_step_s, "Time step", positive=True)
    speed = _nonnegative(speed_mps, "Speed", positive=True)
    weight = _nonnegative(lambda_weight, "Lambda")
    if horizon < departure:
        raise ValueError("Horizon time must not precede departure time.")
    if type(allow_wait) is not bool:
        raise ValueError("allow_wait must be a boolean.")
    if any(type(value) is not int or value < 1 for value in (max_states, max_transitions)):
        raise ValueError("State and transition budgets must be positive integers.")
    if not callable(edge_exposure) or not callable(waiting_exposure):
        raise ValueError("Both exposure integrals must be callable.")
    if not math.isfinite(horizon / step) or horizon / step > 2**52:
        raise ValueError("Time lattice exceeds the supported binary64 index resolution.")

    def ceil_index(time: float) -> int:
        index = math.ceil(time / step)
        # Division can round onto an integer; do not move a nextafter-later
        # arrival back in time. No approximate-equality tolerance is applied.
        while index * step < time:
            index += 1
        while index > 0 and (index - 1) * step >= time:
            index -= 1
        return index

    first_index = ceil_index(departure)
    last_index = math.floor(horizon / step)
    while last_index >= 0 and last_index * step > horizon:
        last_index -= 1
    while (last_index + 1) * step <= horizon:
        last_index += 1
    source_index = first_index if first_index * step == departure else -1
    grid_count = max(0, last_index - first_index + 1)
    # Destination is an exact-time absorbing terminal rather than a grid node.
    capacity = (len(network.nodes) - 1) * grid_count + int(source_index == -1)
    capacity = 1 if start == end else max(1, capacity)
    if capacity > max_states:
        raise ValueError(f"Time-expanded state budget exceeded in preflight: {capacity} > {max_states}.")
    identities = [_identity(network, index) for index in range(len(network.edges))]
    identity_strings = [json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
                        for identity in identities]
    if len(set(identity_strings)) != len(identity_strings):
        raise ValueError("Typed directed edge identities must be unique.")
    adjacency = {node: sorted(network.adjacency[node], key=identity_strings.__getitem__)
                 for node in network.nodes}
    durations = [_nonnegative(edge["length_m"] / speed, "Edge travel time", positive=True)
                 for edge in network.edges]
    base = {
        "scope": "optimal_on_this_declared_rounded_finite_graph",
        "continuous_time_optimum": False, "start": start, "end": end,
        "departure_time_s": departure, "horizon_time_s": horizon,
        "speed_mps": speed, "lambda_weight": weight,
        "time_grid": {"origin_s": 0.0, "time_step_s": step,
                      "first_index": first_index, "last_index": last_index,
                      "count": grid_count, "initial_departure_on_lattice": source_index != -1,
                      "nonterminal_node_ids": sorted(node for node in network.nodes if node != end),
                      "absorbing_destination": end},
        "waiting_policy": {"voluntary_wait_allowed": allow_wait,
                           "nonterminal_rounding_wait_required": True,
                           "destination_arrival_rounded": False,
                           "all_wait_exposure_charged": True},
        "numerical_policy": "strict_binary64_feasibility_no_tolerance; ordinary_float_cost_comparisons",
        "tie_policy": "strict_improvement_only; chronological_states_then_node_id; typed_edge_identity_order_before_wait",
        "assumptions": ["Deterministic nonnegative finite exposure callbacks.",
                        "Each callback integrates the declared traversal or stationary interval.",
                        "No uncertainty or continuous-time approximation error certificate is inferred."],
        "budgets": {"max_states": max_states, "max_transitions": max_transitions,
                    "preflight_state_capacity": capacity},
    }
    if start == end:
        return dict(base, status="optimal", complete=True, actions=[], edges=[], edge_indices=[],
                    edge_ids=[], nodes=[start], arrival_time_s=departure, travel_time_s=0.0,
                    wait_time_s=0.0, rounding_wait_time_s=0.0, voluntary_wait_time_s=0.0,
                    elapsed_time_s=0.0, exposure=0.0, objective=0.0, E=0.0, J=0.0,
                    visited_states=1, discovered_states=1, evaluated_transitions=0)

    # The strictly increasing time coordinate makes the expanded graph a DAG.
    # Process all labels in chronological order so every predecessor is final
    # before its state is expanded, without imposing invalid spatial dominance.
    source = (start, source_index)
    costs = {source: 0.0}
    exposures = {source: 0.0}
    parents: dict[tuple[str, int], tuple[tuple[str, int], list[dict]]] = {}
    queue = [(departure, start, source_index)]
    visited = transitions = 0
    terminal: tuple[float, float, float, tuple[str, int], list[dict]] | None = None

    def count_transition() -> None:
        nonlocal transitions
        if transitions >= max_transitions:
            raise ValueError("Time-expanded transition budget exhausted; no optimality claim is returned.")
        transitions += 1

    def wait_action(node: str, begin: float, finish: float, reason: str) -> dict:
        value = _nonnegative(waiting_exposure(node, begin, finish), "Waiting exposure")
        return {"type": "wait", "reason": reason, "node": node, "start_time_s": begin,
                "end_time_s": finish, "duration_s": finish - begin, "exposure": value}

    def candidate_cost(time: float, exposure: float) -> float:
        value = (time - departure) + weight * exposure
        if not math.isfinite(value) or not math.isfinite(exposure):
            raise ValueError("Accumulated exposure or objective overflowed.")
        return value

    def relax(previous: tuple[str, int], node: str, index: int, actions: list[dict]) -> None:
        key = (node, index)
        time = index * step
        try:
            exposure = exposures[previous] + math.fsum(action["exposure"] for action in actions)
        except OverflowError as error:
            raise ValueError("Accumulated exposure or objective overflowed.") from error
        cost = candidate_cost(time, exposure)
        if key not in costs:
            if len(costs) >= max_states:
                raise ValueError("Time-expanded state budget exhausted; no optimality claim is returned.")
            heapq.heappush(queue, (time, node, index))
        if key not in costs or cost < costs[key]:
            costs[key], exposures[key], parents[key] = cost, exposure, (previous, actions)

    while queue:
        time, node, index = heapq.heappop(queue)
        state = (node, index)
        visited += 1
        for edge_index in adjacency[node]:
            arrival = time + durations[edge_index]
            if not math.isfinite(arrival) or arrival > horizon:
                continue
            if arrival <= time:
                raise ValueError("An edge duration is below the supported clock resolution.")
            neighbor = network.edges[edge_index]["v"]
            next_index = ceil_index(arrival) if neighbor != end else None
            rounded = next_index * step if next_index is not None else arrival
            if rounded > horizon:
                continue
            count_transition()
            value = _nonnegative(edge_exposure(edge_index, time), "Edge exposure")
            actions = [{"type": "travel", "edge_index": edge_index,
                        "edge_id": identities[edge_index], "u": node, "v": neighbor,
                        "start_time_s": time, "end_time_s": arrival,
                        "duration_s": durations[edge_index], "exposure": value,
                        "rounded_arrival_time_s": rounded}]
            if rounded > arrival:
                actions.append(wait_action(neighbor, arrival, rounded, "rounding"))
            if neighbor == end:
                exposure = exposures[state] + value
                cost = candidate_cost(arrival, exposure)
                if terminal is None or cost < terminal[0]:
                    terminal = cost, arrival, exposure, state, actions
            else:
                relax(state, neighbor, next_index, actions)
        next_index = ceil_index(time)
        if next_index * step <= time:
            next_index += 1
        next_time = next_index * step
        if allow_wait and next_time <= horizon:
            count_transition()
            relax(state, node, next_index, [wait_action(node, time, next_time, "voluntary")])

    counts = {"visited_states": visited, "discovered_states": len(costs),
              "evaluated_transitions": transitions}
    if terminal is None:
        return dict(base, **counts, status="unreachable", complete=True, actions=[], edges=[],
                    edge_indices=[], edge_ids=[], nodes=[], arrival_time_s=None, travel_time_s=None,
                    wait_time_s=None, rounding_wait_time_s=None, voluntary_wait_time_s=None,
                    elapsed_time_s=None, exposure=None, objective=None, E=None, J=None)
    cost, arrival, exposure, previous, final_actions = terminal
    chunks = [final_actions]
    while previous != source:
        previous, actions = parents[previous]
        chunks.append(actions)
    actions = [action for chunk in reversed(chunks) for action in chunk]
    for action in actions:
        action["objective"] = action["duration_s"] + weight * action["exposure"]
    edges, nodes = [], [start]
    voluntary_before = 0.0
    for action in actions:
        if action["type"] == "travel":
            edges.append({"edge_index": action["edge_index"], "edge_id": action["edge_id"],
                          "departure_time_s": action["start_time_s"],
                          "actual_arrival_time_s": action["end_time_s"],
                          "rounded_arrival_time_s": action["rounded_arrival_time_s"],
                          "rounding_wait_s": action["rounded_arrival_time_s"] - action["end_time_s"],
                          "voluntary_wait_before_s": voluntary_before,
                          "travel_time_s": action["duration_s"], "exposure": action["exposure"]})
            nodes.append(action["v"])
            voluntary_before = 0.0
        elif action["reason"] == "voluntary":
            voluntary_before += action["duration_s"]
    travel = math.fsum(action["duration_s"] for action in actions if action["type"] == "travel")
    rounding = math.fsum(action["duration_s"] for action in actions
                         if action["type"] == "wait" and action["reason"] == "rounding")
    voluntary = math.fsum(action["duration_s"] for action in actions
                          if action["type"] == "wait" and action["reason"] == "voluntary")
    return dict(base, **counts, status="optimal", complete=True, actions=actions, edges=edges,
                edge_indices=[edge["edge_index"] for edge in edges],
                edge_ids=[edge["edge_id"] for edge in edges], nodes=nodes,
                arrival_time_s=arrival, travel_time_s=travel, wait_time_s=rounding + voluntary,
                rounding_wait_time_s=rounding, voluntary_wait_time_s=voluntary,
                elapsed_time_s=arrival - departure, exposure=exposure, objective=cost, E=exposure, J=cost)
