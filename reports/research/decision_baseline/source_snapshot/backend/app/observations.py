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
    """Prescribed constant-speed path; times are seconds on the caller's clock."""

    vertex_xy: np.ndarray
    vertex_times_s: np.ndarray
    edge_indices: tuple[int, ...]
    metadata: dict

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


def build_path_trajectory(network: RoadNetwork, edge_indices: Sequence[int],
                          speed_mps: float = 1.4, departure_time_s: float = 0.0) -> PathTrajectory:
    """Turn an explicitly connected directed edge sequence into timed positions.

    The path starts at its graph node: click-to-node connectors and waiting are
    not included. Repeated edges and loops retain their full travel time.
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
    for previous, following in zip(selected, selected[1:]):
        if network.edges[previous]["v"] != network.edges[following]["u"]:
            raise ValueError("Trajectory edges must form a connected directed path.")
    coordinates = [np.asarray(network.edges[selected[0]]["coordinates"], dtype=np.float64)]
    for index in selected[1:]:
        points = np.asarray(network.edges[index]["coordinates"], dtype=np.float64)
        # RoadNetwork permits sub-decimetre endpoint disagreement, but silently
        # joining it would change the travel time relative to sum(edge lengths).
        if not np.array_equal(coordinates[-1][-1], points[0]):
            raise ValueError("Trajectory edge geometry must meet exactly at shared nodes.")
        coordinates.append(points[1:])
    xy = np.concatenate(coordinates)
    distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.concatenate(([True], distances > 0))
    xy = xy[keep]
    times = departure + np.concatenate(([0.0], np.cumsum(distances[distances > 0]) / speed))
    if np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory times cannot resolve segment durations at this departure time.")
    metadata = {"schema_version": 1, "geometry_sha256": _geometry_identity(network),
                "edge_ids": [_edge_id(network.edges[i]) for i in selected],
                "speed_mps": speed, "departure_time_s": departure,
                "arrival_time_s": float(times[-1]), "connector_policy": "graph_nodes_only",
                "waiting": False, "distance_m": float(distances.sum())}
    xy.flags.writeable = False
    times.flags.writeable = False
    return PathTrajectory(xy, times, selected, metadata)
