"""Linear spatial observations and prescribed trajectories, independent of search.

The sparse map integrates a *specified reconstruction* of a frozen grid field.
Its entries are nonnegative and each row sums to the corresponding travel time.
This describes neither an exact PDE solution nor a physical exposure model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from time import perf_counter
from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .diffusion import Grid, bilinear_weights

if TYPE_CHECKING:
    from .routing import RoadNetwork


def _positive_speed(speed_mps: float) -> float:
    speed = float(speed_mps)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Walking speed must be positive and finite.")
    return speed


def _edge_id(edge: dict) -> list:
    # Keep the original key type; integer 1 and string "1" are distinct edges.
    return [edge["u"], edge["v"], edge["key"]]


def _geometry_identity(network: RoadNetwork) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"crs": network.crs, "edges": [_edge_id(e) for e in network.edges]},
                             sort_keys=True, separators=(",", ":")).encode())
    for edge in network.edges:
        coordinates = np.asarray(edge["coordinates"], dtype="<f8")
        digest.update(np.asarray(coordinates.shape, dtype="<i8").tobytes())
        digest.update(coordinates.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class EdgeObserver:
    """Reusable ``H`` with row order matching ``network.edges`` at construction.

    Signed finite fields are supported for linear-algebra and adjoint work.
    Routing must enforce its separate nonnegative-concentration assumption.
    ``apply_batch`` accepts exactly ``(batch,ny,nx)`` and returns ``(batch,edge)``.
    The matrix buffers are read-only; do not replace them or mutate metadata.
    """

    grid: Grid
    matrix: csr_matrix
    metadata: dict

    def apply(self, field: np.ndarray) -> np.ndarray:
        field = np.asarray(field, dtype=np.float64)
        if field.shape != (self.grid.ny, self.grid.nx) or not np.isfinite(field).all():
            raise ValueError("Field must be finite and have grid shape (ny,nx).")
        result = np.asarray(self.matrix @ field.ravel())
        if not np.isfinite(result).all():
            raise ValueError("Observation evaluation produced nonfinite values.")
        return result

    def apply_batch(self, fields: np.ndarray, chunk_size: int | None = None) -> np.ndarray:
        fields = np.asarray(fields, dtype=np.float64)
        if fields.ndim != 3 or fields.shape[1:] != (self.grid.ny, self.grid.nx) or not np.isfinite(fields).all():
            raise ValueError("Batch fields must be finite with shape (batch,ny,nx).")
        if chunk_size is not None and (isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size <= 0):
            raise ValueError("Chunk size must be a positive integer.")
        size = len(fields) if chunk_size is None else int(chunk_size)
        result = np.empty((len(fields), self.matrix.shape[0]), dtype=np.float64)
        for start in range(0, len(fields), max(size, 1)):
            end = min(start + size, len(fields))
            flat = fields[start:end].reshape(end - start, self.grid.nx * self.grid.ny)
            result[start:end] = (self.matrix @ flat.T).T
        if not np.isfinite(result).all():
            raise ValueError("Observation evaluation produced nonfinite values.")
        return result


def _gauss2_grid_sampling(network: RoadNetwork, grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split at reconstruction knots, then integrate quadratics along a line.

    A bilinear polynomial restricted to a straight line is degree at most two.
    Two-point Gauss integrates it exactly up to roundoff on each split interval.
    This is a reference for this reconstruction, not for the continuous PDE.
    """
    points, weights, owners = [], [], []
    for start, vector, length, owner in zip(network._segment_starts, network._segment_vectors,
                                          network._segment_lengths, network._segment_owners):
        cuts = [np.array([0.0, 1.0])]
        for axis, centres in enumerate((grid.x, grid.y)):
            if vector[axis] != 0:
                fractions = (centres - start[axis]) / vector[axis]
                cuts.append(fractions[(fractions > 0) & (fractions < 1)])
        breaks = np.unique(np.concatenate(cuts))
        halfwidth = np.diff(breaks) / 2
        midpoints = (breaks[:-1] + breaks[1:]) / 2
        fractions = np.column_stack((midpoints - halfwidth / np.sqrt(3),
                                     midpoints + halfwidth / np.sqrt(3))).ravel()
        points.append(start + fractions[:, None] * vector)
        weights.append(np.repeat(halfwidth * length, 2))
        owners.append(np.full(len(fractions), owner, dtype=np.int64))
    if not points:
        return np.empty((0, 2)), np.empty(0), np.empty(0, dtype=np.int64)
    return np.concatenate(points), np.concatenate(weights), np.concatenate(owners)


def build_edge_observer(network: RoadNetwork, grid: Grid, boundary: str = "zero_flux",
                        speed_mps: float = 1.4, quadrature: str = "trapezoid", *,
                        sample_spacing_m: float | None = None) -> EdgeObserver:
    """Construct CSR exposure weights using the complete directed polylines.

    ``trapezoid`` defaults to the legacy spacing ``grid.h/2`` and preserves every
    original vertex. ``gauss2_grid`` splits at cell-centre reconstruction knots;
    its integration error is roundoff-level for this bilinear reconstruction.
    No extrapolation outside the rectangle or periodic geometric teleportation
    is performed. A seam-crossing edge must have an explicit in-domain geometry.
    """
    started = perf_counter()
    speed = _positive_speed(speed_mps)
    if quadrature not in ("trapezoid", "gauss2_grid"):
        raise ValueError("Quadrature must be 'trapezoid' or 'gauss2_grid'.")
    if quadrature == "gauss2_grid" and sample_spacing_m is not None:
        raise ValueError("sample_spacing_m applies only to trapezoid quadrature.")
    spacing = grid.h / 2 if sample_spacing_m is None else float(sample_spacing_m)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Sampling spacing must be positive and finite.")
    # Gauss nodes exclude segment endpoints: explicitly validate all vertices so
    # a short excursion outside the domain cannot disappear between Gauss nodes.
    vertices = np.concatenate((network._segment_starts,
                               network._segment_starts + network._segment_vectors))
    bilinear_weights(grid, vertices, boundary)
    if quadrature == "trapezoid":
        points, distance_weights, edge_indices = network._sampling(spacing)
    else:
        points, distance_weights, edge_indices = _gauss2_grid_sampling(network, grid)
    columns, reconstruction_weights = bilinear_weights(grid, points, boundary)
    rows = np.repeat(edge_indices, 4)
    with np.errstate(over="ignore", invalid="ignore"):
        values = (reconstruction_weights * (distance_weights / speed)[:, None]).ravel()
    if not np.isfinite(values).all():
        raise ValueError("Edge travel times cannot be represented finitely at this speed.")
    matrix = coo_matrix((values, (rows, columns.ravel())),
                        shape=(len(network.edges), grid.nx * grid.ny)).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all():
        raise ValueError("Edge observation weights cannot be represented finitely.")
    config = {"schema_version": 1, "geometry_sha256": _geometry_identity(network),
              "grid": grid.to_dict(), "boundary": boundary, "speed_mps": speed,
              "quadrature": quadrature,
              "sample_spacing_m": spacing if quadrature == "trapezoid" else None,
              "reconstruction": "cell_centred_piecewise_bilinear_v1"}
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata = {**config, "operator_sha256": identity,
                "edge_ids": [_edge_id(edge) for edge in network.edges],
                "field_order": "C:y,x", "shape": list(matrix.shape), "nnz": int(matrix.nnz),
                "csr_bytes": matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
                "sample_count": len(points), "build_ms": (perf_counter() - started) * 1000,
                "outside_domain": "reject_except_coordinate_roundoff",
                "wall_sampling": "periodic_seam" if boundary == "periodic" else "constant_extension",
                "interpretation": "integral_of_reconstruction_not_exact_PDE_or_physical_exposure"}
    for buffer in (matrix.data, matrix.indices, matrix.indptr):
        buffer.flags.writeable = False
    return EdgeObserver(grid, matrix, metadata)


@dataclass(frozen=True)
class PathTrajectory:
    """Prescribed piecewise-linear positions on an absolute seconds clock.

    Equal consecutive positions encode waiting. Times are strictly increasing;
    a single vertex represents a zero-duration trajectory. Construction copies
    and freezes the arrays, including when this dataclass is used directly.
    """

    vertex_xy: np.ndarray
    vertex_times_s: np.ndarray
    edge_indices: tuple[int, ...]
    metadata: dict

    def __post_init__(self) -> None:
        xy = np.array(self.vertex_xy, dtype=np.float64, copy=True)
        times = np.array(self.vertex_times_s, dtype=np.float64, copy=True)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0 or not np.isfinite(xy).all():
            raise ValueError("Trajectory vertices must be finite with shape (n,2), n >= 1.")
        if times.ndim != 1 or len(times) != len(xy) or not np.isfinite(times).all():
            raise ValueError("Trajectory vertex times must be a finite vector matching the vertices.")
        with np.errstate(over="ignore", invalid="ignore"):
            durations = np.diff(times)
            total_duration = times[-1] - times[0]
        if not np.isfinite(total_duration) or not np.isfinite(durations).all() or np.any(durations <= 0):
            raise ValueError("Trajectory vertex times must be strictly increasing with finite durations.")
        indices = tuple(self.edge_indices)
        if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in indices):
            raise ValueError("Trajectory edge indices must be nonnegative integers.")
        if not isinstance(self.metadata, dict):
            raise ValueError("Trajectory metadata must be a dictionary.")
        xy.flags.writeable = times.flags.writeable = False
        object.__setattr__(self, "vertex_xy", xy)
        object.__setattr__(self, "vertex_times_s", times)
        object.__setattr__(self, "edge_indices", tuple(int(i) for i in indices))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def departure_time_s(self) -> float:
        return float(self.vertex_times_s[0])

    @property
    def arrival_time_s(self) -> float:
        return float(self.vertex_times_s[-1])

    def positions(self, times_s: np.ndarray) -> np.ndarray:
        times = np.asarray(times_s, dtype=np.float64)
        if not np.isfinite(times).all() or np.any(times < self.departure_time_s) or np.any(times > self.arrival_time_s):
            raise ValueError("Trajectory query times must lie within the departure/arrival interval.")
        xy = np.column_stack([np.interp(times.ravel(), self.vertex_times_s, self.vertex_xy[:, axis])
                              for axis in (0, 1)])
        return xy.reshape(times.shape + (2,))


def build_stationary_trajectory(point: Sequence[float], start: float, end: float) -> PathTrajectory:
    """A prescribed wait, including the valid zero-duration case ``start=end``."""
    xy = np.asarray(point, dtype=np.float64)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Stationary point must contain two finite coordinates.")
    start, end = float(start), float(end)
    if not np.isfinite([start, end]).all() or end < start:
        raise ValueError("Stationary times must be finite and end must not precede start.")
    times = np.array([start]) if start == end else np.array([start, end])
    return PathTrajectory(np.repeat(xy[None], len(times), axis=0), times, (),
                          {"schema_version": 1, "departure_time_s": start,
                           "arrival_time_s": end, "waiting": end > start,
                           "waiting_time_s": end - start, "distance_m": 0.0,
                           "connector_policy": "prescribed_point"})


def build_path_trajectory(network: RoadNetwork, edge_indices: Sequence[int],
                          speed_mps: float = 1.4, departure_time_s: float = 0.0, *,
                          waits_s: Sequence[float] | None = None) -> PathTrajectory:
    """Turn a connected directed edge sequence into timed positions and waits.

    ``waits_s`` has ``len(edge_indices)+1`` entries: wait before each edge
    occurrence and once at the destination. Repeated edges retain independent
    waits and full travel time. The start is the graph node; click connectors
    are not part of this trajectory. Arrival includes the final prescribed wait.
    """
    speed = _positive_speed(speed_mps)
    departure = float(departure_time_s)
    if not np.isfinite(departure):
        raise ValueError("Departure time must be finite.")
    selected = tuple(edge_indices)
    if not selected:
        raise ValueError("A trajectory requires at least one directed edge.")
    if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 or i >= len(network.edges) for i in selected):
        raise ValueError("Trajectory edge indices must be valid integers.")
    selected = tuple(int(i) for i in selected)
    waits = np.zeros(len(selected) + 1) if waits_s is None else np.asarray(waits_s, dtype=np.float64)
    if waits.shape != (len(selected) + 1,) or not np.isfinite(waits).all() or np.any(waits < 0):
        raise ValueError("waits_s must have len(edges)+1 finite nonnegative durations.")
    for previous, following in zip(selected, selected[1:]):
        if network.edges[previous]["v"] != network.edges[following]["u"]:
            raise ValueError("Trajectory edges must form a connected directed path.")
    first = np.asarray(network.edges[selected[0]]["coordinates"], dtype=np.float64)
    vertices, times = [first[0]], [departure]
    edge_departures, edge_arrivals = [], []
    distance = 0.0

    def append_time(point: np.ndarray, time: float) -> None:
        if not np.isfinite(time) or time <= times[-1]:
            raise ValueError("Trajectory times cannot resolve segment durations at this departure time.")
        vertices.append(point)
        times.append(time)

    for occurrence, index in enumerate(selected):
        points = np.asarray(network.edges[index]["coordinates"], dtype=np.float64)
        # RoadNetwork allows small endpoint discrepancies. Never insert an
        # unbilled connector when composing a prescribed trajectory.
        if not np.array_equal(vertices[-1], points[0]):
            raise ValueError("Trajectory edge geometry must meet exactly at shared nodes.")
        if waits[occurrence] > 0:
            append_time(points[0], times[-1] + float(waits[occurrence]))
        edge_start = times[-1]
        edge_departures.append(edge_start)
        edge_length = float(network.edges[index]["length_m"])
        edge_end = edge_start + edge_length / speed
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        valid = lengths > 0
        segment_ends = points[1:][valid]
        cumulative_lengths = np.cumsum(lengths[valid])
        # Search transitions use this authoritative edge clock. Accumulating
        # length/speed per segment instead can overshoot a strict frame horizon
        # by one ULP. Preserve every interior corner while anchoring the last
        # vertex exactly to edge_start + stored_length/speed.
        for occurrence_in_edge, (point, arc_length) in enumerate(zip(segment_ends, cumulative_lengths)):
            time = (edge_end if occurrence_in_edge == len(segment_ends) - 1
                    else edge_start + float(arc_length) / speed)
            append_time(point, time)
        distance += edge_length
        edge_arrivals.append(times[-1])
    if waits[-1] > 0:
        append_time(vertices[-1], times[-1] + float(waits[-1]))
    metadata = {"schema_version": 1, "geometry_sha256": _geometry_identity(network),
                "edge_ids": [_edge_id(network.edges[i]) for i in selected],
                "speed_mps": speed, "departure_time_s": departure,
                "arrival_time_s": times[-1], "connector_policy": "graph_nodes_only",
                "waiting": bool(np.any(waits > 0)), "waits_s": waits.tolist(),
                "waiting_time_s": float(waits.sum()), "distance_m": distance,
                "edge_departure_times_s": edge_departures, "edge_arrival_times_s": edge_arrivals}
    return PathTrajectory(np.asarray(vertices), np.asarray(times), selected, metadata)
