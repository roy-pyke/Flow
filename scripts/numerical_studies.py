"""Work/precision and frozen-route sensitivity experiments (no file writes)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
from pyproj import Transformer
from scipy.sparse.linalg import expm_multiply

from backend.app.diffusion import Grid, bilinear_interpolate, gaussian_initial
from backend.app.numerics import (advection_matrix, clear_factor_cache,
                                  diffusion_matrix, solve)
from backend.app.numerics.validation import errors
from backend.app.routing import RoadNetwork, shortest_path

ROOT = Path(__file__).resolve().parents[1]


def run_work_precision(quick: bool = False) -> dict:
    """Cold and reusable-factor costs at measured errors against fixed-grid truth.

    Raw wall timings include conversion, assembly, cold LU, output and
    diagnostics. A warm call still assembles the operator, but reuses LU.
    Timings exclude import/startup and generation of the reference solution.
    """
    repeats = 3 if quick else 10
    grid = Grid((0, 0, 2, 1.5), 20, 16)
    kappa, duration = .07, .5
    xx, yy = np.meshgrid(grid.x, grid.y)
    mode = np.cos(np.pi * xx / 2) * np.cos(np.pi * yy / 1.5)
    diffusion_initial = 1 + .5 * mode
    lam_h = -4 * kappa * (np.sin(np.pi * grid.dx / 4) ** 2 / grid.dx ** 2
                          + np.sin(np.pi * grid.dy / 3) ** 2 / grid.dy ** 2)
    diffusion_reference = 1 + .5 * np.exp(lam_h * duration) * mode
    transport_initial = 1 + .3 * np.cos(np.pi * xx) * np.cos(2 * np.pi * yy / 1.5)
    wind = (.3, -.2)
    operator = diffusion_matrix(grid, kappa, "periodic") + advection_matrix(grid, wind, "periodic")
    transport_reference = expm_multiply(duration * operator, transport_initial.ravel()).reshape(mode.shape)
    specifications = [
        ("diffusion", "explicit_euler", "numpy", "zero_flux", (0., 0.), diffusion_initial, diffusion_reference, [32, 64, 128, 256]),
        ("diffusion", "backward_euler", "scipy", "zero_flux", (0., 0.), diffusion_initial, diffusion_reference, [4, 8, 16, 32, 64, 128]),
        ("diffusion", "crank_nicolson", "scipy", "zero_flux", (0., 0.), diffusion_initial, diffusion_reference, [4, 8, 16, 32, 64, 128]),
        ("transport", "advection_explicit", "numpy", "periodic", wind, transport_initial, transport_reference, [32, 64, 128, 256]),
        ("transport", "imex_euler", "numpy_scipy", "periodic", wind, transport_initial, transport_reference, [16, 32, 64, 128, 256]),
    ]
    rows = []
    for problem, method, backend, boundary, velocity, initial, reference, counts in specifications:
        if quick:
            counts = counts[:3]
        for count in counts:
            dt = duration / count
            # Prime Python allocation/code paths; it is not a recorded sample.
            for _ in range(1 if quick else 3):
                solve(grid, initial, kappa, [duration], method=method, backend=backend,
                      boundary=boundary, velocity=velocity, dt=dt)
            for cache in ["cold", "warm"]:
                samples = []
                if cache == "warm":
                    clear_factor_cache()
                    solve(grid, initial, kappa, [duration], method=method, backend=backend,
                          boundary=boundary, velocity=velocity, dt=dt)
                for _ in range(repeats):
                    if cache == "cold":
                        clear_factor_cache()
                    started = perf_counter()
                    frames, d = solve(grid, initial, kappa, [duration], method=method, backend=backend,
                                      boundary=boundary, velocity=velocity, dt=dt)
                    wall_ms = (perf_counter() - started) * 1000
                    samples.append({"wall_ms": wall_ms, "timings_ms": d["timings_ms"],
                                    "factorizations": d["factorization_count"],
                                    "cache_hits": d["factorization_cache_hits"],
                                    "cache_bytes": d["factorization_cache_bytes"],
                                    "internal_steps": d["internal_steps"],
                                    "mass_balance_relative": d["relative_mass_balance_error"],
                                    "linear_residual": d["linear_residual_max"],
                                    "minimum": d["min_concentration"], "maximum": d["max_concentration"]})
                wall = [sample["wall_ms"] for sample in samples]
                rows.append({"problem": problem, "method": method, "backend": backend, "cache": cache,
                             "dt_s": dt, "output_strategy": "final_only", "grid": [grid.nx, grid.ny],
                             "kappa": kappa, "velocity": list(velocity), "boundary": boundary,
                             "samples": samples, "median_total_ms": float(np.median(wall)),
                             "p90_total_ms": float(np.percentile(wall, 90)),
                             **errors(frames[-1], reference)})
    attainment = []
    for problem in ["diffusion", "transport"]:
        for cache in ["cold", "warm"]:
            for tolerance in [1e-3, 1e-4, 1e-5]:
                for method in dict.fromkeys(row["method"] for row in rows if row["problem"] == problem):
                    eligible = [row for row in rows if row["problem"] == problem and row["cache"] == cache
                                and row["method"] == method and row["l2"] <= tolerance]
                    fastest = min(eligible, key=lambda row: row["median_total_ms"]) if eligible else None
                    attainment.append({"problem": problem, "cache": cache, "l2_tolerance": tolerance,
                                       "method": method, "attained": bool(fastest),
                                       "median_total_ms": fastest["median_total_ms"] if fastest else None,
                                       "dt_s": fastest["dt_s"] if fastest else None,
                                       "measured_l2": fastest["l2"] if fastest else None})
    return {"schema_version": 2, "quick": quick,
            "machine": {"platform": platform.platform(), "python": platform.python_version(),
                        "threads": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]}},
            "protocol": {"repetitions": repeats, "warmups": 1 if quick else 3, "output_times_s": [duration],
                         "timed_scope": "whole solve call; initial conversions, operator assembly, LU, advance, outputs, diagnostics",
                         "cold": "clear LU cache before each sample", "warm": "prime identical actual-dt LU factors once; still reconstruct spatial operator",
                         "error": "RMS concentration error against exact fixed-grid reference; same physical input and output work for all methods per problem",
                         "caution": "small-grid method study; no claim of asymptotic or large-scale LU performance; unmet thresholds remain unmeasured, not failures"},
            "problems": {"diffusion": "exact semidiscrete Neumann cosine, no spatial error in target",
                         "transport": "independent expm_multiply reference for periodic upwind+diffusion operator"},
            "rows": rows, "tolerance_attainment": attainment}


def run_routing_sensitivity(quick: bool = False) -> dict:
    """Separate graph optimality from the effect of a PDE field approximation.

    All grids use exactly the same finest-grid polyline samples. The numerical
    reference is additionally refined and is never represented as measured truth.
    """
    path = ROOT / "data/demo/network.json"
    network_bytes = path.read_bytes()
    network = RoadNetwork.from_json(json.loads(network_bytes))
    bounds = tuple(network.bounds)
    transformer = Transformer.from_crs("EPSG:4326", network.crs, always_xy=True)
    source_lonlat = [-122.4149, 37.7599]
    start_lonlat, end_lonlat = [-122.428, 37.7515], [-122.403, 37.7675]
    source = transformer.transform(*source_lonlat)
    start = network.nearest_node(*transformer.transform(*start_lonlat))
    end = network.nearest_node(*transformer.transform(*end_lonlat))
    kappa, tstar, speed, lam = 20., 600., 1.4, 5.
    reference_n, refined_n = (160, 320) if quick else (320, 640)
    candidate_n = 80 if quick else 160
    fine_grid = Grid(bounds, refined_n, refined_n)
    points, weights, owners = network._sampling(fine_grid.h / 2)

    def exposures(field: np.ndarray, grid: Grid) -> np.ndarray:
        if field.min() < 0:
            raise ValueError("Signed diagnostic fields cannot be used for exposure routing.")
        values = bilinear_interpolate(field, grid, points)
        return np.bincount(owners, weights=values * weights / speed, minlength=len(network.edges))

    def costs_check(edge_exposures: np.ndarray) -> tuple[dict, float]:
        own = shortest_path(network, start, end, edge_exposures, lam, "astar", speed)
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(network.nodes)
        for index, edge in enumerate(network.edges):
            graph.add_edge(edge["u"], edge["v"], key=index,
                           weight=edge["length_m"] / speed + lam * edge_exposures[index])
        nx_cost = float(nx.shortest_path_length(graph, start, end, weight="weight"))
        if abs(own["objective"] - nx_cost) > 1e-7:
            raise AssertionError("Custom route search does not agree with NetworkX.")
        return own, nx_cost

    models = []
    for model, velocity, boundary in [("diffusion", (0., 0.), "zero_flux"),
                                      ("advection_diffusion", (.35, -.15), "open")]:
        reference_fields = []
        reference_metadata = []
        # Explicit reference avoids sparse LU fill at 640²; time step shrinks
        # with h², while the reference-refinement check quantifies remaining error.
        for n in [reference_n, refined_n]:
            grid = Grid(bounds, n, n)
            method = "explicit_euler" if model == "diffusion" else "advection_explicit"
            frames, diagnostics = solve(grid, gaussian_initial(grid, source), kappa, [tstar],
                                        method=method, boundary=boundary, velocity=velocity)
            reference_fields.append((grid, frames[-1], exposures(frames[-1], grid)))
            reference_metadata.append({"n": n, "method": method, "dt_s": diagnostics["dt_max_s"],
                                       "steps": diagnostics["internal_steps"], "solve_ms": diagnostics["solve_ms"],
                                       "mass_balance_relative": diagnostics["relative_mass_balance_error"]})
        coarse_grid, coarse_field, coarse_exposure = reference_fields[0]
        ref_grid, ref_field, ref_exposure = reference_fields[1]
        reference_route, reference_nx = costs_check(ref_exposure)
        coarse_route, _ = costs_check(coarse_exposure)
        coarse_nodes = np.stack(np.meshgrid(coarse_grid.x, coarse_grid.y), axis=-1).reshape(-1, 2)
        sampled_fine = bilinear_interpolate(ref_field, ref_grid, coarse_nodes).reshape(coarse_field.shape)
        refinement = {"coarse_n": reference_n, "fine_n": refined_n,
                      "field_l2": errors(coarse_field, sampled_fine)["l2"],
                      "edge_exposure_l2": float(np.sqrt(np.mean((coarse_exposure - ref_exposure) ** 2))),
                      "edge_exposure_max": float(np.max(np.abs(coarse_exposure - ref_exposure))),
                      "optimal_cost_change_s": abs(reference_route["objective"] - coarse_route["objective"]),
                      "cost_relative_change": abs(reference_route["objective"] - coarse_route["objective"]) / reference_route["objective"],
                      "same_edge_path": coarse_route["edge_indices"] == reference_route["edge_indices"],
                      "interpretation": "remaining numerical-reference uncertainty; no extrapolation or truth claim"}
        configurations = ([("explicit_euler", "numpy", None), ("backward_euler", "scipy", 30.),
                            ("crank_nicolson", "scipy", 30.)] if model == "diffusion"
                           else [("advection_explicit", "numpy", None), ("imex_euler", "numpy_scipy", 30.)])
        candidates = []
        for method, backend, dt in configurations:
            grid = Grid(bounds, candidate_n, candidate_n)
            frames, d = solve(grid, gaussian_initial(grid, source), kappa, [tstar], method=method,
                              backend=backend, dt=dt, boundary=boundary, velocity=velocity)
            field = frames[-1]
            if field.min() < 0:
                candidates.append({"method": method, "backend": backend, "eligible": False,
                                   "minimum": float(field.min()), "reason": "negative field excluded without clipping"})
                continue
            edge_exposures = exposures(field, grid)
            route, nx_cost = costs_check(edge_exposures)
            evaluated_cost = route["time_s"] + lam * float(ref_exposure[route["edge_indices"]].sum())
            grid_nodes = np.stack(np.meshgrid(grid.x, grid.y), axis=-1).reshape(-1, 2)
            sampled_reference = bilinear_interpolate(ref_field, ref_grid, grid_nodes).reshape(field.shape)
            candidates.append({"method": method, "backend": backend, "eligible": True,
                               "grid_n": candidate_n, "dt_s": d["dt_max_s"], "steps": d["internal_steps"],
                               "minimum": float(field.min()), "solve_ms": d["solve_ms"],
                               "field_l2": errors(field, sampled_reference)["l2"],
                               "field_linf": errors(field, sampled_reference)["linf"],
                               "edge_exposure_l2": float(np.sqrt(np.mean((edge_exposures - ref_exposure) ** 2))),
                               "edge_exposure_max": float(np.max(np.abs(edge_exposures - ref_exposure))),
                               "candidate_optimal_cost_s": route["objective"], "networkx_cost_s": nx_cost,
                               "networkx_absolute_error": abs(route["objective"] - nx_cost),
                               "route_cost_in_reference_s": evaluated_cost,
                               "reference_regret_s": evaluated_cost - reference_route["objective"],
                               "relative_reference_regret": (evaluated_cost - reference_route["objective"]) / reference_route["objective"],
                               "same_edge_path_as_reference": route["edge_indices"] == reference_route["edge_indices"],
                               "route_edge_indices": route["edge_indices"],
                               "mass_balance_relative": d["relative_mass_balance_error"]})
        models.append({"model": model, "velocity": list(velocity), "boundary": boundary,
                       "reference": {"resolutions": reference_metadata, "optimal_cost_s": reference_route["objective"],
                                     "networkx_cost_s": reference_nx, "route_edge_indices": reference_route["edge_indices"]},
                       "reference_refinement": refinement, "candidates": candidates})
    return {"schema_version": 2, "quick": quick, "data_version": hashlib.sha256(network_bytes).hexdigest(),
            "protocol": {"source": source_lonlat, "start": start_lonlat, "end": end_lonlat,
                         "kappa": kappa, "frozen_time_s": tstar, "walk_speed_mps": speed, "lambda_weight": lam,
                         "sigma_m": 250., "same_graph": True, "same_polyline_samples": True,
                         "integration_max_spacing_m": fine_grid.h / 2,
                         "reference": "finer explicit numerical field, with one further spatial/time refinement",
                         "regret": "candidate route reevaluated in finest reference field minus that field's graph-optimal cost",
                         "distinction": "NetworkX checks each frozen graph's algorithm correctness; field/edge errors and regret measure PDE approximation effects",
                         "limits": "one graph, source, endpoint pair and frozen time; a zero regret can coexist with field error; near ties do not imply algorithm failure"},
            "models": models}
