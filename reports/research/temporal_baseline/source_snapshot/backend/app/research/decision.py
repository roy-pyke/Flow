"""Auditable decisions on finite candidates and static directed tiny graphs.

The bounds in this module are conditional on the supplied intervals being valid
for every compared alternative. Empirical estimates are deliberately labelled
as such. No field L2 error is treated as a line-integral bound, and no finite
candidate certificate is promoted to a continuous or dynamic routing claim.
"""
from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..routing import RoadNetwork


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def edge_identity(network: RoadNetwork, index: int) -> list:
    edge = network.edges[index]
    return [edge["u"], edge["v"], edge["key"]]


@dataclass(frozen=True)
class SimplePathSet:
    """A complete enumeration; budget exhaustion raises instead of truncating."""
    paths: tuple[tuple[int, ...], ...]
    start: str
    end: str
    candidate_hash: str
    graph_hash: str

    def to_dict(self, network: RoadNetwork) -> dict:
        return {"scope": "all_simple_directed_paths_in_this_finite_graph",
                "complete": True, "start": self.start, "end": self.end,
                "graph_hash": self.graph_hash, "candidate_hash": self.candidate_hash,
                "count": len(self.paths),
                "paths": [{"edge_indices": list(path),
                           "edge_identities": [edge_identity(network, i) for i in path]}
                          for path in self.paths]}


def enumerate_simple_paths(network: RoadNetwork, start: str, end: str, *,
                           max_nodes: int = 12, max_paths: int = 10000) -> SimplePathSet:
    """Enumerate vertex-simple paths, retaining distinct directed parallel edges.

    Self loops and repeated vertices are excluded. The start=end result is the
    empty path. An unreachable target has an empty, complete candidate set.
    Size limits are hard guards, not an approximate candidate-generation policy.
    """
    start, end = str(start), str(end)
    if start not in network.nodes or end not in network.nodes:
        raise ValueError("Start and end must be graph nodes.")
    if any(type(value) is not int or value < 1 for value in (max_nodes, max_paths)):
        raise ValueError("Enumeration limits must be positive integers.")
    if len(network.nodes) > max_nodes:
        raise ValueError("Graph exceeds the exact-oracle node budget.")
    identities = [edge_identity(network, i) for i in range(len(network.edges))]
    if len({canonical_hash(identity) for identity in identities}) != len(identities):
        raise ValueError("Stable (u,v,key) identities must be unique.")
    paths: list[tuple[int, ...]] = []

    def visit(node: str, visited: frozenset, path: tuple[int, ...]) -> None:
        if node == end:
            if len(paths) >= max_paths:
                raise ValueError("Exact-oracle path budget exhausted; no completeness claim is returned.")
            paths.append(path)
            return
        for index in network.adjacency[node]:
            neighbor = network.edges[index]["v"]
            if neighbor not in visited:
                visit(neighbor, visited | {neighbor}, (*path, index))

    visit(start, frozenset({start}), ())
    graph_data = {"crs": network.crs, "bounds": list(network.bounds),
                  "nodes": [{"id": node, "x": network.nodes[node]["x"], "y": network.nodes[node]["y"]}
                            for node in sorted(network.nodes)], "edges": [
        {"identity": edge_identity(network, i), "coordinates": edge["coordinates"],
         "length_m": edge["length_m"]} for i, edge in enumerate(network.edges)]}
    graph_hash = canonical_hash(graph_data)
    return SimplePathSet(tuple(paths), start, end,
                         canonical_hash({"graph_hash": graph_hash, "start": start, "end": end,
                                         "paths": [[identities[i] for i in path] for path in paths]}),
                         graph_hash)


def path_costs(paths: SimplePathSet, edge_costs: Sequence[float]) -> np.ndarray:
    costs = np.asarray(edge_costs, dtype=float)
    if costs.ndim != 1 or not np.isfinite(costs).all():
        raise ValueError("Edge costs must be a finite vector.")
    if any(i >= len(costs) for path in paths.paths for i in path):
        raise ValueError("Edge cost vector does not cover every path.")
    return np.array([sum(costs[i] for i in path) for path in paths.paths], dtype=float)


def minimizers(costs: Sequence[float], *, atol: float = 1e-10) -> list[int]:
    values = np.asarray(costs, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("At least one finite candidate cost is required.")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("Tie tolerance must be finite and nonnegative.")
    return np.flatnonzero(values <= values.min() + atol).tolist()


def integral_error_bound(travel_time_s: Sequence[float] | float,
                         concentration_linf: float, *, lambda_weight: float = 1.0) -> dict:
    """|E-Ehat| <= T epsilon, |J-Jhat| <= lambda T epsilon.

    epsilon must bound the concentration error everywhere along the same fixed
    path and observation times. A norm measured only at grid nodes does not by
    itself meet that premise. This expression excludes quadrature error.
    """
    times = np.asarray(travel_time_s, dtype=float)
    if (not np.isfinite(times).all() or np.any(times < 0)
            or not np.isfinite([concentration_linf, lambda_weight]).all()
            or min(concentration_linf, lambda_weight) < 0):
        raise ValueError("Times, sup error and lambda must be finite and nonnegative.")
    exposure = times * concentration_linf
    return {"exposure": exposure, "objective": lambda_weight * exposure,
            "scope": "fixed_path_same_time_parameterization; quadrature_excluded"}


def candidate_bounds(approximate_costs: Sequence[float], error_bounds: Sequence[float],
                     selected: int, *, search_gap: float = 0.0,
                     evidence: str = "assumed", difference_error_bounds: Sequence[float] | None = None,
                     arithmetic_tolerance: float = 1e-10) -> dict:
    """Conditional 2 epsilon + eta and heterogeneous interval regret bounds.

    Every candidate needs an absolute cost-error bound. eta bounds selected
    approximate cost minus minimum approximate cost. Direct difference bounds,
    when supplied, bound (J_selected-J_i)-(Jhat_selected-Jhat_i); they can retain
    shared-error cancellation that independent marginal intervals lose.
    """
    approximate = np.asarray(approximate_costs, dtype=float)
    errors = np.asarray(error_bounds, dtype=float)
    if (approximate.ndim != 1 or not len(approximate) or errors.shape != approximate.shape
            or not np.isfinite(approximate).all() or not np.isfinite(errors).all()
            or np.any(errors < 0)):
        raise ValueError("A finite nonnegative error bound is required for EVERY candidate.")
    if type(selected) is not int or not 0 <= selected < len(approximate):
        raise ValueError("Selected index is not a candidate.")
    if (not np.isfinite([search_gap, arithmetic_tolerance]).all()
            or min(search_gap, arithmetic_tolerance) < 0):
        raise ValueError("Search gap and arithmetic tolerance must be finite and nonnegative.")
    measured_gap = float(approximate[selected] - approximate.min())
    if measured_gap > search_gap + arithmetic_tolerance:
        raise ValueError("Claimed search gap is smaller than the observed finite-set search gap.")
    if evidence not in {"assumed", "analytic_manufactured", "empirical"}:
        raise ValueError("Unknown error-bound evidence type.")
    lower, upper = approximate - errors, approximate + errors
    competitors = np.arange(len(approximate)) != selected
    result = {"scope": "this_finite_candidate_set_only",
              "bound_status": "empirical_only" if evidence == "empirical" else "conditional_on_all_supplied_bounds",
              "evidence": evidence, "selected": selected, "candidate_count": len(approximate),
              "all_candidate_bounds_supplied": True, "search_gap": float(search_gap),
              "observed_approximate_search_gap": measured_gap,
              "uniform_epsilon": float(errors.max()),
              "two_epsilon_plus_eta": float(2 * errors.max() + search_gap),
              "lower": lower.tolist(), "upper": upper.tolist(),
              "interval_regret_bound": float(max(0.0, upper[selected] - lower.min())),
              "strict_interval_separation": bool(np.all(upper[selected] < lower[competitors])),
              "assumptions": ["Every supplied bound covers the same reference functional.",
                              "Candidates and model inputs are identical between approximate and reference costs.",
                              "Floating-point comparisons use the reported arithmetic tolerance; interval arithmetic is not implemented."],
              "arithmetic_tolerance": arithmetic_tolerance}
    if difference_error_bounds is not None:
        difference = np.asarray(difference_error_bounds, dtype=float)
        if (difference.shape != approximate.shape or not np.isfinite(difference).all()
                or np.any(difference < 0) or difference[selected] != 0):
            raise ValueError("Supply a nonnegative difference bound for every competitor, with zero for self.")
        upper_difference = approximate[selected] - approximate + difference
        result["direct_difference"] = {"error_bounds": difference.tolist(),
            "regret_bound": float(max(0.0, upper_difference.max())),
            "strict_separation": bool(np.all(upper_difference[competitors] < 0)),
            "premise": "Direct joint difference bounds must be independently justified; correlation is not inferred."}
    return result


def static_graph_interval_bound(network: RoadNetwork, start: str, end: str,
                                edge_lower: Sequence[float], edge_upper: Sequence[float],
                                selected_edges: Sequence[int], *, nonnegative_cost_prior: bool = True,
                                evidence: str = "assumed") -> dict:
    """Upper(selected) - shortest(lower graph) bounds static graph regret.

    With a valid nonnegative-cost prior, negative interval lower endpoints can
    be tightened to zero. Otherwise negative lower edges are rejected, never
    passed silently to Dijkstra. Nonnegative static costs admit a simple optimum
    even when the feasible set allows walks. This does not cover time dependence.
    """
    start, end = str(start), str(end)
    lower, upper = np.asarray(edge_lower, dtype=float), np.asarray(edge_upper, dtype=float)
    if (lower.shape != (len(network.edges),) or upper.shape != lower.shape
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower > upper)):
        raise ValueError("Each graph edge requires a finite ordered interval.")
    if start not in network.nodes or end not in network.nodes:
        raise ValueError("Start and end must be graph nodes.")
    if evidence not in {"assumed", "analytic_manufactured", "empirical"}:
        raise ValueError("Unknown error-bound evidence type.")
    if nonnegative_cost_prior:
        if np.any(upper < 0):
            raise ValueError("Upper bounds contradict the nonnegative-cost prior.")
        lower = np.maximum(lower, 0)
    elif np.any(lower < 0):
        raise ValueError("Negative lower edges require a different shortest-path algorithm.")
    current = start
    selected = list(selected_edges)
    for index in selected:
        if type(index) is not int or not 0 <= index < len(network.edges):
            raise ValueError("Selected path contains an invalid edge index.")
        if network.edges[index]["u"] != current:
            raise ValueError("Selected edge sequence is disconnected.")
        current = network.edges[index]["v"]
    if current != end:
        raise ValueError("Selected path does not end at the destination.")
    distance = {start: 0.0}
    previous: dict[str, tuple[str, int]] = {}
    queue = [(0.0, start)]
    while queue:
        value, node = heapq.heappop(queue)
        if value > distance[node]:
            continue
        if node == end:
            break
        for index in network.adjacency[node]:
            neighbor = network.edges[index]["v"]
            candidate = value + lower[index]
            if candidate < distance.get(neighbor, float("inf")):
                distance[neighbor] = candidate
                previous[neighbor] = (node, index)
                heapq.heappush(queue, (float(candidate), neighbor))
    if end not in distance:
        raise ValueError("No directed route connects the selected endpoints.")
    lower_path, current = [], end
    while current != start:
        current, index = previous[current]
        lower_path.append(index)
    lower_path.reverse()
    selected_upper = float(sum(upper[index] for index in selected))
    return {"scope": "all_static_directed_walks_in_this_graph_with_nonnegative_costs",
            "bound_status": "empirical_only" if evidence == "empirical" else "conditional_on_all_edge_intervals",
            "evidence": evidence, "nonnegative_cost_prior": nonnegative_cost_prior,
            "tightened_edge_lower": lower.tolist(), "selected_upper": selected_upper,
            "lower_graph_optimum": float(distance[end]),
            "regret_bound": max(0.0, selected_upper - float(distance[end])),
            "lower_graph_path": lower_path,
            "lower_graph_edge_identities": [edge_identity(network, i) for i in lower_path]}


def corridor_network(bounds: Sequence[float], *, upper_fraction: float = 0.8,
                      lower_fraction: float = 0.35) -> tuple[RoadNetwork, dict]:
    """Two directed polygonal corridors with a known, exhaustive two-path set."""
    xmin, ymin, xmax, ymax = map(float, bounds)
    if not np.isfinite([xmin, ymin, xmax, ymax, upper_fraction, lower_fraction]).all():
        raise ValueError("Corridor geometry must be finite.")
    if not (xmax > xmin and ymax > ymin and 0 < lower_fraction < .5 < upper_fraction < 1):
        raise ValueError("Corridor fractions must satisfy 0 < lower < .5 < upper < 1.")
    x0, x1 = xmin + .1 * (xmax - xmin), xmin + .9 * (xmax - xmin)
    middle = ymin + .5 * (ymax - ymin)
    positions = {"s": (x0, middle), "u0": (x0, ymin + upper_fraction * (ymax - ymin)),
                 "u1": (x1, ymin + upper_fraction * (ymax - ymin)), "t": (x1, middle),
                 "l0": (x0, ymin + lower_fraction * (ymax - ymin)),
                 "l1": (x1, ymin + lower_fraction * (ymax - ymin))}
    pairs = [("s", "u0"), ("u0", "u1"), ("u1", "t"),
             ("s", "l0"), ("l0", "l1"), ("l1", "t")]
    data = {"bounds": list(bounds), "crs": "LOCAL_SYNTHETIC_METRES",
            "metadata": {"provenance": "manufactured_two_corridor_graph; not OSM", "units": "metres"},
            "nodes": [{"id": name, "x": xy[0], "y": xy[1]} for name, xy in positions.items()],
            "edges": [{"u": u, "v": v, "key": 0, "coordinates": [positions[u], positions[v]]}
                      for u, v in pairs]}
    return RoadNetwork(data), data
