"""Nonnegative time-series field adapter for prescribed and searched journeys.

The observer remains a linear mathematical operation on signed arrays. This
adapter owns the stricter routing contract and immutable input copies. Its
scalar-result cache limits retained entries, not total process or solver memory.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from numbers import Integral

import numpy as np

from ..diffusion import Grid, bilinear_weights
from ..observations import PathTrajectory, build_path_trajectory, build_stationary_trajectory
from ..routing import RoadNetwork
from ..temporal_observations import build_trajectory_observer


class TimeDependentExposure:
    def __init__(self, network: RoadNetwork, grid: Grid, times_s, frames, *,
                 speed_mps: float = 1.4, boundary: str = "zero_flux",
                 max_cache_entries: int = 2048, max_field_bytes: int = 128*1024**2,
                 max_samples: int = 1_000_000):
        for value, name in [(max_cache_entries, "cache entries"), (max_field_bytes, "field bytes"),
                            (max_samples, "sample budget")]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0 or (name != "cache entries" and value == 0):
                raise ValueError(f"Invalid {name} limit.")
        speed = float(speed_mps)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Speed must be finite and positive.")
        times = np.asarray(times_s, dtype=np.float64)
        values = np.asarray(frames)
        with np.errstate(over="ignore", invalid="ignore"):
            intervals = np.diff(times) if times.ndim == 1 else np.array([np.nan])
        if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or not np.isfinite(intervals).all() or np.any(intervals <= 0):
            raise ValueError("At least two finite increasing frame times are required.")
        if values.shape != (len(times), grid.ny, grid.nx) or values.dtype.kind not in "fiu":
            raise ValueError("Fields must be real numeric arrays with shape (time,ny,nx).")
        if values.size*8 > max_field_bytes:
            raise ValueError("Copied float64 fields exceed the field byte budget.")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Routing requires finite nonnegative concentration fields.")
        # A graph with slightly separated edge endpoints would otherwise move a
        # traveler between samples and waiting nodes without charging a connector.
        for edge in network.edges:
            for endpoint, node in ((edge["coordinates"][0], edge["u"]),
                                   (edge["coordinates"][-1], edge["v"])):
                node_xy = [network.nodes[node]["x"], network.nodes[node]["y"]]
                if not np.array_equal(endpoint, node_xy):
                    raise ValueError("Timed routing requires exact geometry joins at graph nodes.")
        vertices = np.concatenate((network.node_xy, network._segment_starts,
                                   network._segment_starts+network._segment_vectors))
        bilinear_weights(grid, vertices, boundary)
        self.network, self.grid = network, grid
        self.times_s = np.array(times, copy=True)
        self.frames = np.array(values, dtype=np.float64, order="C", copy=True)
        self.times_s.flags.writeable = self.frames.flags.writeable = False
        self.speed_mps, self.boundary = speed, boundary
        self.max_cache_entries, self.max_samples = int(max_cache_entries), int(max_samples)
        self._cache: OrderedDict[tuple, float] = OrderedDict()
        self.cache_hits = self.cache_misses = 0
        descriptor = {"grid": grid.to_dict(), "times_s": self.times_s.tolist(), "boundary": boundary,
                      "speed_mps": speed, "frames_sha256": hashlib.sha256(self.frames.astype("<f8", copy=False).tobytes()).hexdigest(),
                      "nodes": [{"id": k, "x": v["x"], "y": v["y"]} for k,v in network.nodes.items()],
                      "edges": [{"identity": [e["u"], e["v"], e["key"]], "coordinates": e["coordinates"]} for e in network.edges]}
        self.identity = hashlib.sha256(json.dumps(descriptor, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()

    def _coverage(self, start: float, end: float):
        if not np.isfinite([start, end]).all() or end < start or start < self.times_s[0] or end > self.times_s[-1]:
            raise ValueError("Entire journey or waiting interval must lie within saved frame coverage.")

    def _cached(self, key: tuple, compute) -> float:
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.cache_misses += 1
        value = float(compute())
        if not np.isfinite(value) or value < 0:
            raise ValueError("Routing observation must be finite and nonnegative.")
        if self.max_cache_entries:
            self._cache[key] = value
            while len(self._cache) > self.max_cache_entries:
                self._cache.popitem(last=False)
        return value

    def trajectory_exposure(self, trajectory) -> float:
        self._coverage(trajectory.departure_time_s, trajectory.arrival_time_s)
        return build_trajectory_observer(self.grid, self.times_s, trajectory, boundary=self.boundary,
                                         max_samples=self.max_samples).apply(self.frames)

    def edge_exposure(self, edge_index: int, departure_s: float) -> float:
        if isinstance(edge_index, bool) or not isinstance(edge_index, Integral) or not 0 <= edge_index < len(self.network.edges):
            raise ValueError("Edge index is invalid.")
        departure = float(departure_s)
        trajectory = build_path_trajectory(self.network, [int(edge_index)], self.speed_mps, departure)
        self._coverage(departure, trajectory.arrival_time_s)
        return self._cached(("edge", int(edge_index), departure), lambda: self.trajectory_exposure(trajectory))

    def waiting_exposure(self, node_id: str, start_s: float, end_s: float) -> float:
        node_id, start, end = str(node_id), float(start_s), float(end_s)
        if node_id not in self.network.nodes:
            raise ValueError("Waiting node is unknown.")
        self._coverage(start, end)
        if start == end:
            return 0.0
        node = self.network.nodes[node_id]
        return self._cached(("wait", node_id, start, end), lambda: self.trajectory_exposure(
            build_stationary_trajectory([node["x"], node["y"]], start, end)))

    def metadata(self) -> dict:
        return {"field_graph_identity": self.identity, "boundary": self.boundary,
                "field_bytes": self.frames.nbytes, "frame_count": len(self.times_s),
                "coverage_s": [float(self.times_s[0]), float(self.times_s[-1])],
                "speed_mps": self.speed_mps, "cache_entries": len(self._cache),
                "max_cache_entries": self.max_cache_entries, "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses, "quadrature": "gauss2_split",
                "interpretation": "piecewise bilinear space and linear time reconstruction; no physical calibration",
                "network_ownership": "do not mutate network geometry during this adapter lifetime"}


def trajectory_from_schedule(network: RoadNetwork, result: dict) -> PathTrajectory:
    """Reconstruct one full journey from an optimal result's explicit actions.

    Action clocks are authoritative. No connector, wait, or time gap is inserted
    to repair an inconsistent result. Geometry is parameterized at constant
    speed within every recorded travel action and preserves all polyline bends.
    """
    if result.get("status") != "optimal":
        raise ValueError("Only an optimal schedule has a journey to reconstruct.")
    node_id = str(result["start"])
    node = network.nodes[node_id]
    vertices = [np.array([node["x"], node["y"]], dtype=float)]
    times = [float(result["departure_time_s"])]
    edges = []
    for action in result["actions"]:
        start, end = float(action["start_time_s"]), float(action["end_time_s"])
        if not np.isfinite([start, end]).all() or start != times[-1] or end <= start:
            raise ValueError("Schedule actions must form a finite uninterrupted time sequence.")
        if action["type"] == "wait":
            if str(action["node"]) != node_id:
                raise ValueError("Schedule waiting location differs from the current node.")
            vertices.append(vertices[-1]);times.append(end)
        elif action["type"] == "travel":
            index=action["edge_index"]
            if isinstance(index,bool) or not isinstance(index,Integral) or not 0<=index<len(network.edges):
                raise ValueError("Schedule edge index is invalid.")
            edge=network.edges[index];points=np.asarray(edge["coordinates"],dtype=float)
            if edge["u"]!=node_id or not np.array_equal(points[0],vertices[-1]):
                raise ValueError("Schedule edge geometry is disconnected.")
            expected=start+edge["length_m"]/result["speed_mps"]
            if end!=expected:
                raise ValueError("Schedule travel time differs from its edge and speed.")
            lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
            cumulative=np.cumsum(lengths)
            for point,distance,length in zip(points[1:],cumulative,lengths):
                if length>0:
                    vertices.append(point);times.append(start+(end-start)*(distance/edge["length_m"]))
            times[-1]=end
            node_id=edge["v"];node=network.nodes[node_id]
            if not np.array_equal(vertices[-1],[node["x"],node["y"]]):
                raise ValueError("Schedule geometry does not meet its graph node exactly.")
            edges.append(int(index))
        else:
            raise ValueError("Unknown schedule action type.")
    if node_id!=str(result["end"]) or times[-1]!=result["arrival_time_s"]:
        raise ValueError("Schedule final node or time is inconsistent.")
    if edges!=result["edge_indices"]:
        raise ValueError("Schedule edge list differs from its actions.")
    return PathTrajectory(np.asarray(vertices),np.asarray(times),tuple(edges),
        {"schema_version":1,"source":"explicit_time_expanded_actions","waiting":any(a["type"]=="wait" for a in result["actions"]),
         "connector_policy":"none; exact geometry joins required","departure_time_s":times[0],"arrival_time_s":times[-1]})
