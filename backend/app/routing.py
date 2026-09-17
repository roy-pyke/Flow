"""Directed multigraph routing and frozen-field exposure integration."""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .diffusion import Grid, bilinear_interpolate


class RoadNetwork:
    def __init__(self, data: dict):
        self.metadata = data.get("metadata", {})
        self.crs = data.get("crs", self.metadata.get("crs", "EPSG:32610"))
        self.nodes = {str(node["id"]): dict(node, id=str(node["id"])) for node in data["nodes"]}
        if not self.nodes:
            raise ValueError("Road network has no nodes.")
        self.node_ids = list(self.nodes)
        self.node_xy = np.array([[self.nodes[node]["x"], self.nodes[node]["y"]] for node in self.node_ids], dtype=float)
        if not np.isfinite(self.node_xy).all():
            raise ValueError("Road coordinates must be finite.")
        self.bounds = tuple(data.get("bounds", self.metadata.get("bounds", [*self.node_xy.min(axis=0), *self.node_xy.max(axis=0)])))
        self.edges: list[dict] = []
        self.adjacency: dict[str, list[int]] = {node: [] for node in self.nodes}
        segment_starts, segment_vectors, segment_lengths, segment_owners = [], [], [], []
        for raw in data["edges"]:
            edge = dict(raw, u=str(raw["u"]), v=str(raw["v"]))
            if edge["u"] not in self.nodes or edge["v"] not in self.nodes:
                raise ValueError("Road edge refers to an unknown node.")
            coords = np.asarray(edge["coordinates"], dtype=float)
            if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
                raise ValueError("Every road needs a finite complete projected polyline.")
            u_xy = np.array([self.nodes[edge["u"]]["x"], self.nodes[edge["u"]]["y"]])
            v_xy = np.array([self.nodes[edge["v"]]["x"], self.nodes[edge["v"]]["y"]])
            if np.linalg.norm(coords[0] - u_xy) > np.linalg.norm(coords[-1] - u_xy):
                coords = coords[::-1].copy()
                for key in ("geometry_lonlat", "coordinates_lonlat"):
                    if key in edge:
                        edge[key] = list(reversed(edge[key]))
            if max(np.linalg.norm(coords[0] - u_xy), np.linalg.norm(coords[-1] - v_xy)) > 0.1:
                raise ValueError("Road polyline endpoints do not match its graph nodes.")
            vectors = np.diff(coords, axis=0)
            lengths = np.linalg.norm(vectors, axis=1)
            length = float(lengths.sum())
            if length <= 0:
                continue
            edge.update(coordinates=coords.tolist(), length_m=length, key=raw.get("key", 0))
            valid = lengths > 0
            segment_starts.append(coords[:-1][valid])
            segment_vectors.append(vectors[valid])
            segment_lengths.append(lengths[valid])
            segment_owners.append(np.full(int(valid.sum()), len(self.edges), dtype=int))
            self.adjacency[edge["u"]].append(len(self.edges))
            self.edges.append(edge)
        self._segment_starts = np.concatenate(segment_starts) if segment_starts else np.empty((0, 2))
        self._segment_vectors = np.concatenate(segment_vectors) if segment_vectors else np.empty((0, 2))
        self._segment_lengths = np.concatenate(segment_lengths) if segment_lengths else np.empty(0)
        self._segment_owners = np.concatenate(segment_owners) if segment_owners else np.empty(0, dtype=int)
        self._sampling_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.last_integration_ms = 0.0
        self._transformer = None

    @classmethod
    def from_json(cls, value: dict | str | Path) -> "RoadNetwork":
        if not isinstance(value, dict):
            with Path(value).open(encoding="utf-8") as handle:
                value = json.load(handle)
        return cls(value)

    def nearest_node(self, x: float, y: float, max_distance_m: float | None = None) -> str:
        if not np.isfinite([x, y]).all():
            raise ValueError("Click coordinates must be finite.")
        if not (self.bounds[0] <= x <= self.bounds[2] and self.bounds[1] <= y <= self.bounds[3]):
            raise ValueError("Click lies outside the imported road region.")
        distances = np.linalg.norm(self.node_xy - [x, y], axis=1)
        index = int(np.argmin(distances))
        if max_distance_m is not None and distances[index] > max_distance_m:
            raise ValueError("No road node is close enough to the selected point.")
        return self.node_ids[index]

    def _sampling(self, max_step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if max_step in self._sampling_cache:
            return self._sampling_cache[max_step]
        if not np.isfinite(max_step) or max_step <= 0:
            raise ValueError("Sampling spacing must be positive and finite.")
        # Integrate every original polyline segment, preserving all vertices.
        # Shared vertices appear twice with the appropriate half weight from
        # each adjacent segment. Vectorization keeps even the first request fast.
        counts = np.maximum(1, np.ceil(self._segment_lengths / max_step).astype(int))
        sample_counts = counts + 1
        owners = np.repeat(np.arange(len(counts)), sample_counts)
        offsets = np.cumsum(sample_counts) - sample_counts
        positions = np.arange(len(owners)) - np.repeat(offsets, sample_counts)
        fractions = positions / counts[owners]
        points = self._segment_starts[owners] + fractions[:, None] * self._segment_vectors[owners]
        weights = self._segment_lengths[owners] / counts[owners]
        endpoint = (positions == 0) | (positions == counts[owners])
        weights[endpoint] *= 0.5
        result = points, weights, self._segment_owners[owners]
        self._sampling_cache[max_step] = result
        return result

    def edge_exposures(self, field: np.ndarray, grid: Grid, speed_mps: float = 1.4) -> np.ndarray:
        if not np.isfinite(speed_mps) or speed_mps <= 0:
            raise ValueError("Walking speed must be positive and finite.")
        if np.min(field) < -1e-12:
            raise ValueError("Exposure routing requires a nonnegative concentration field.")
        started = perf_counter()
        points, weights, edge_indices = self._sampling(grid.h / 2)
        concentration = bilinear_interpolate(field, grid, points)
        result = np.bincount(edge_indices, weights=concentration * weights / speed_mps, minlength=len(self.edges))
        self.last_integration_ms = (perf_counter() - started) * 1000
        return result

    def edge_lonlat(self, index: int) -> list[list[float]]:
        edge = self.edges[index]
        if "geometry_lonlat" in edge:
            return edge["geometry_lonlat"]
        if "coordinates_lonlat" in edge:
            return edge["coordinates_lonlat"]
        if self._transformer is None:
            from pyproj import Transformer
            self._transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        points = np.asarray(edge["coordinates"])
        lon, lat = self._transformer.transform(points[:, 0], points[:, 1])
        edge["geometry_lonlat"] = np.column_stack([lon, lat]).tolist()
        return edge["geometry_lonlat"]


def shortest_path(network: RoadNetwork, start_id: Any, end_id: Any,
                  exposures: np.ndarray | None = None, lambda_weight: float = 0,
                  algorithm: str = "astar", speed_mps: float = 1.4) -> dict:
    """Own heap-based Dijkstra/A*, retaining each directed parallel edge.

Euclidean distance / speed is admissible and consistent because edge lengths
are measured along projected polylines and all exposure penalties are >= 0.
"""
    start_id, end_id = str(start_id), str(end_id)
    if start_id not in network.nodes or end_id not in network.nodes:
        raise ValueError("Start or destination is not part of the road network.")
    if not np.isfinite(lambda_weight) or lambda_weight < 0:
        raise ValueError("Route preference lambda must be nonnegative and finite.")
    if not np.isfinite(speed_mps) or speed_mps <= 0:
        raise ValueError("Walking speed must be positive and finite.")
    algorithm = algorithm.lower()
    if algorithm not in ("dijkstra", "astar"):
        raise ValueError("Algorithm must be 'dijkstra' or 'astar'.")
    exposures = np.zeros(len(network.edges)) if exposures is None else np.asarray(exposures, dtype=float)
    if exposures.shape != (len(network.edges),) or not np.isfinite(exposures).all() or np.any(exposures < 0):
        raise ValueError("One finite nonnegative exposure is required per road edge.")
    started = perf_counter()
    target = network.nodes[end_id]

    def heuristic(node_id: str) -> float:
        if algorithm == "dijkstra":
            return 0.0
        node = network.nodes[node_id]
        return math.hypot(node["x"] - target["x"], node["y"] - target["y"]) / speed_mps

    distances = {start_id: 0.0}
    previous: dict[str, tuple[str, int]] = {}
    queue = [(heuristic(start_id), 0.0, start_id)]
    settled: set[str] = set()
    while queue:
        _, distance, current = heapq.heappop(queue)
        if current in settled or distance > distances.get(current, math.inf):
            continue
        settled.add(current)
        if current == end_id:
            break
        for index in network.adjacency[current]:
            edge = network.edges[index]
            neighbor = edge["v"]
            candidate = distance + edge["length_m"] / speed_mps + lambda_weight * exposures[index]
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                previous[neighbor] = current, index
                heapq.heappush(queue, (candidate + heuristic(neighbor), candidate, neighbor))
    search_ms = (perf_counter() - started) * 1000
    if end_id not in settled:
        raise ValueError("No directed walking route connects these two nodes.")
    node_path, edge_path = [end_id], []
    while node_path[-1] != start_id:
        parent, edge_index = previous[node_path[-1]]
        node_path.append(parent)
        edge_path.append(edge_index)
    node_path.reverse()
    edge_path.reverse()
    distance_m = sum(network.edges[index]["length_m"] for index in edge_path)
    exposure = float(sum(exposures[index] for index in edge_path))
    coordinates: list = []
    for index in edge_path:
        points = network.edge_lonlat(index)
        coordinates.extend(points if not coordinates else points[1:])
    if not coordinates:
        node = network.nodes[start_id]
        if "lon" in node and "lat" in node:
            point = [node["lon"], node["lat"]]
        else:
            from pyproj import Transformer
            point = list(Transformer.from_crs(network.crs, "EPSG:4326", always_xy=True).transform(node["x"], node["y"]))
        coordinates = [point, point]
    return {"nodes": node_path, "edge_indices": edge_path, "distance_m": float(distance_m),
            "time_s": float(distance_m / speed_mps), "exposure": exposure,
            "objective": float(distance_m / speed_mps + lambda_weight * exposure),
            "lambda_weight": float(lambda_weight), "algorithm": algorithm,
            "search_ms": search_ms, "visited_nodes": len(settled),
            "geometry": {"type": "LineString", "coordinates": coordinates}}
