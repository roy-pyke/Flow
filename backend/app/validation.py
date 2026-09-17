"""Reproducible numerical and algorithm validation used by the report API."""

from __future__ import annotations

import math
from time import perf_counter

import networkx as nx
import numpy as np

from .diffusion import Grid, gaussian_initial, solve_initial
from .routing import RoadNetwork, shortest_path


def validate_numerics() -> dict:
    started = perf_counter()
    convergence = []
    # Cosine cell-centre samples satisfy homogeneous Neumann boundaries.
    # The temporal error is O(dt), spatial error O(h²), and dt=0.1h².
    final_time = 0.02
    for size in (32, 64, 128):
        grid = Grid((0, 0, 1, 1), size, size)
        xx, yy = np.meshgrid(grid.x, grid.y)
        mode = np.cos(np.pi * xx) * np.cos(np.pi * yy)
        initial = 1 + 0.5 * mode
        frames, diagnostic = solve_initial(grid, initial, 1, [final_time], dt=0.1 * grid.h ** 2)
        exact = 1 + 0.5 * np.exp(-2 * np.pi ** 2 * final_time) * mode
        error = frames[-1] - exact
        l2 = float(np.sqrt(np.mean(error ** 2)))
        convergence.append({"n": size, "h": grid.h, "dt": 0.1 * grid.h ** 2,
                            "l2_error": l2, "linf_error": float(np.max(np.abs(error))),
                            "observed_order": None if not convergence else math.log(convergence[-1]["l2_error"] / l2, 2),
                            "solve_ms": diagnostic["solve_ms"],
                            "relative_mass_drift": diagnostic["relative_mass_drift"]})
    grid = Grid((0, 0, 4000, 4000))
    constant, _ = solve_initial(grid, np.full((160, 160), 0.37), 20, [0, 1800])
    constant_error = float(np.max(np.abs(constant[-1] - 0.37)))
    # The off-centre release reaches two walls and exercises reflection.
    initial = gaussian_initial(grid, (250, 250))
    _, conservation = solve_initial(grid, initial, 50, np.arange(0, 1801, 30))
    checks = {"constant_field": constant_error < 1e-14,
              "mass_conservation": conservation["relative_mass_drift"] < 1e-10,
              "nonnegative": conservation["min_concentration"] >= -1e-12,
              "second_order_joint_convergence": all(1.8 < row["observed_order"] < 2.2 for row in convergence[1:])}
    return {"passed": all(checks.values()), "checks": checks,
            "constant_max_error": constant_error, "conservation": conservation,
            "convergence": convergence, "analytic_solution": "1 + 0.5 exp(-2 pi² t) cos(pi x) cos(pi y)",
            "analytic_domain": [0, 0, 1, 1], "analytic_final_time": final_time,
            "convergence_note": "Euler is first order in time; dt=0.1 h² makes the combined spatial/time refinement approximately second order.",
            "total_ms": (perf_counter() - started) * 1000}


def validate_routing(network: RoadNetwork, grid: Grid, field: np.ndarray,
                     start_id: str, end_id: str,
                     lambdas: tuple = (0, 0.5, 1, 2, 5, 10)) -> dict:
    exposures = network.edge_exposures(field, grid)
    integration_ms = network.last_integration_ms
    constant_exposures = network.edge_exposures(np.full((grid.ny, grid.nx), 0.37), grid)
    expected = np.array([edge["length_m"] / 1.4 * 0.37 for edge in network.edges])
    constant_error = float(np.max(np.abs(constant_exposures - expected))) if len(expected) else 0.0
    comparisons = []
    for weight in lambdas:
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(network.nodes)
        for index, edge in enumerate(network.edges):
            graph.add_edge(edge["u"], edge["v"], key=index,
                           weight=edge["length_m"] / 1.4 + weight * exposures[index])
        reference_started = perf_counter()
        reference = nx.shortest_path_length(graph, str(start_id), str(end_id), weight="weight")
        reference_ms = (perf_counter() - reference_started) * 1000
        for algorithm in ("dijkstra", "astar"):
            route = shortest_path(network, start_id, end_id, exposures, weight, algorithm)
            comparisons.append({"lambda_weight": weight, "algorithm": algorithm,
                                "objective": route["objective"], "reference_objective": reference,
                                "absolute_cost_error": abs(route["objective"] - reference),
                                "search_ms": route["search_ms"], "reference_ms": reference_ms,
                                "visited_nodes": route["visited_nodes"]})
    zero = shortest_path(network, start_id, end_id, np.zeros(len(network.edges)), 10)
    shortest = shortest_path(network, start_id, end_id, exposures, 0)
    checks = {"networkx_cost_agreement": all(row["absolute_cost_error"] < 1e-8 * max(1, row["reference_objective"]) for row in comparisons),
              "constant_field_integral": constant_error < 1e-8,
              "zero_field_shortest_route": abs(zero["distance_m"] - shortest["distance_m"]) < 1e-8}
    return {"passed": all(checks.values()), "checks": checks, "comparisons": comparisons,
            "constant_field_max_error": constant_error, "integration_ms": integration_ms,
            "start_id": str(start_id), "end_id": str(end_id)}
