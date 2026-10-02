"""Linear observations of explicitly reconstructed space-time fields.

Stored frames define a cell-centred piecewise bilinear spatial reconstruction
and linear interpolation in time. These observations integrate that declared
reconstruction, not the continuous PDE or real-world pollutant exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .diffusion import Grid, bilinear_weights
from .observations import PathTrajectory


def _frame_times(times_s: np.ndarray) -> np.ndarray:
    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("Frame times must be a finite vector with at least two entries.")
    with np.errstate(over="ignore", invalid="ignore"):
        durations = np.diff(times)
    if not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("Frame times must be strictly increasing with finite intervals.")
    return times


def _validated_frames(grid: Grid, times: np.ndarray, frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float64)
    if frames.shape != (len(times), grid.ny, grid.nx) or not np.isfinite(frames).all():
        raise ValueError("Frames must be finite with shape (frame,ny,nx).")
    return frames


def _time_weights(times: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(queries).all() or np.any(queries < times[0]) or np.any(queries > times[-1]):
        raise ValueError("Query times must be covered by stored frame times; extrapolation is forbidden.")
    # The final frame belongs to the last interval with alpha=1. This only
    # selects a valid interval; out-of-range times were rejected above.
    left = np.minimum(np.searchsorted(times, queries, side="right") - 1, len(times) - 2)
    alpha = (queries - times[left]) / (times[left + 1] - times[left])
    return np.column_stack((left, left + 1)), np.column_stack((1 - alpha, alpha))


def sample_time_series(grid: Grid, times_s: np.ndarray, frames: np.ndarray,
                       points_xy: np.ndarray, query_times_s: np.ndarray,
                       boundary: str = "zero_flux") -> np.ndarray:
    """Sample matching positions/times; output has ``query_times_s.shape``.

    Points have shape ``query_times_s.shape + (2,)``. Signed finite fields are
    supported. Every query must be inside the physical rectangle and temporal
    coverage; spatial boundary handling matches ``bilinear_weights`` exactly.
    """
    times = _frame_times(times_s)
    frames = _validated_frames(grid, times, frames)
    queries = np.asarray(query_times_s, dtype=np.float64)
    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape != queries.shape + (2,):
        raise ValueError("Positions must have query_times_s.shape + (2,).")
    temporal_indices, temporal_weights = _time_weights(times, queries.ravel())
    spatial_indices, spatial_weights = bilinear_weights(grid, points.reshape(-1, 2), boundary)
    flat_frames = frames.reshape(len(times), -1)
    result = np.zeros(queries.size)
    with np.errstate(over="ignore", invalid="ignore"):
        for side in (0, 1):
            spatial_values = np.sum(flat_frames[temporal_indices[:, side, None], spatial_indices]
                                    * spatial_weights, axis=1)
            result += temporal_weights[:, side] * spatial_values
    if not np.isfinite(result).all():
        raise ValueError("Space-time sampling produced nonfinite values.")
    return result.reshape(queries.shape)


@dataclass(frozen=True)
class TrajectoryObserver:
    """CSR row maps flattened ``(frame,y,x)`` arrays to a time integral.

    Matrix buffers and frame times are read-only. Apply accepts signed fields;
    algorithms that require nonnegative costs must check that separately.
    """

    grid: Grid
    times_s: np.ndarray
    matrix: csr_matrix
    metadata: dict

    def apply(self, frames: np.ndarray) -> float:
        frames = _validated_frames(self.grid, self.times_s, frames)
        with np.errstate(over="ignore", invalid="ignore"):
            value = float((self.matrix @ frames.ravel())[0])
        if not np.isfinite(value):
            raise ValueError("Trajectory observation produced a nonfinite value.")
        return value


def _positive_step(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def _centre_crossings(start: float, delta: float, lower: float, spacing: float,
                      count: int, budget: int) -> np.ndarray:
    if delta == 0:
        return np.empty(0)
    low, high = sorted((start, start + delta))
    # Include possible endpoint centres then apply strict fraction filtering.
    # Compute index bounds before allocating, avoiding a full huge grid.x/y.
    first = max(0, int(np.ceil((low - lower) / spacing - 0.5)))
    last = min(count - 1, int(np.floor((high - lower) / spacing - 0.5)))
    number = max(0, last - first + 1)
    if number > budget + 2:
        raise ValueError("Trajectory quadrature exceeds max_samples before allocation.")
    centres = lower + (np.arange(first, last + 1, dtype=np.float64) + 0.5) * spacing
    fractions = (centres - start) / delta
    return fractions[(fractions > 0) & (fractions < 1)]


def _segment_breaks(grid: Grid, times: np.ndarray, a: np.ndarray, b: np.ndarray,
                    start: float, end: float, budget: int) -> np.ndarray:
    first = int(np.searchsorted(times, start, side="right"))
    last = int(np.searchsorted(times, end, side="left"))
    if last - first > budget:
        raise ValueError("Trajectory quadrature exceeds max_samples before allocation.")
    cuts = [np.array([0.0, 1.0]), (times[first:last] - start) / (end - start)]
    for axis, (spacing, count) in enumerate(((grid.dx, grid.nx), (grid.dy, grid.ny))):
        cuts.append(_centre_crossings(a[axis], b[axis] - a[axis], grid.bounds[axis],
                                      spacing, count, budget))
    return np.unique(np.concatenate(cuts))


def build_trajectory_observer(grid: Grid, times_s: np.ndarray, trajectory: PathTrajectory,
                              boundary: str = "zero_flux", quadrature: str = "gauss2_split", *,
                              spatial_step_m: float | None = None,
                              time_step_s: float | None = None,
                              max_samples: int = 1_000_000) -> TrajectoryObserver:
    """Integrate along moving segments and waits without temporal clamping.

    Both rules preserve every trajectory vertex, crossed spatial reconstruction
    knot, and frame time. ``gauss2_split`` uses two Gauss nodes per split panel:
    bilinear space times linear time is at most cubic along a straight segment,
    so this integrates the declared reconstruction exactly up to roundoff.

    ``trapezoid`` additionally subdivides each panel to meet the independent
    optional spatial/time maximum steps. An omitted bound adds no refinement;
    with both omitted there is one trapezoid per mandatory panel. Gauss rejects
    either step argument rather than silently ignoring it. ``max_samples`` caps
    quadrature evaluations, checked before sample and sparse-matrix allocation.
    Its bound is not a total process-memory limit.
    """
    started = perf_counter()
    times = _frame_times(times_s)
    if not isinstance(trajectory, PathTrajectory):
        raise ValueError("trajectory must be a validated PathTrajectory.")
    if isinstance(max_samples, bool) or not isinstance(max_samples, Integral) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer.")
    max_samples = int(max_samples)
    if quadrature not in ("gauss2_split", "trapezoid"):
        raise ValueError("Quadrature must be 'gauss2_split' or 'trapezoid'.")
    spatial_step = _positive_step(spatial_step_m, "spatial_step_m")
    time_step = _positive_step(time_step_s, "time_step_s")
    if quadrature == "gauss2_split" and (spatial_step is not None or time_step is not None):
        raise ValueError("Spatial/time step arguments apply only to trapezoid quadrature.")
    _time_weights(times, np.array([trajectory.departure_time_s, trajectory.arrival_time_s]))
    # Each nonzero-duration segment needs at least two evaluations. Reject
    # obviously over-budget trajectories before allocating vertex weights too.
    if 2 * (len(trajectory.vertex_times_s) - 1) > max_samples:
        raise ValueError("Trajectory quadrature exceeds max_samples before allocation.")
    # Validate endpoints too; Gauss nodes alone could miss a short excursion.
    bilinear_weights(grid, trajectory.vertex_xy, boundary)
    total_columns = len(times) * grid.nx * grid.ny
    if total_columns > np.iinfo(np.int64).max:
        raise ValueError("Space-time grid exceeds sparse index capacity.")

    plans, sample_count, panel_count = [], 0, 0
    for a, b, start, end in zip(trajectory.vertex_xy[:-1], trajectory.vertex_xy[1:],
                               trajectory.vertex_times_s[:-1], trajectory.vertex_times_s[1:]):
        breaks = _segment_breaks(grid, times, a, b, start, end, max_samples)
        widths = np.diff(breaks)
        if quadrature == "gauss2_split":
            counts = None
            additions = 2 * len(widths)
        else:
            ratios = np.ones(len(widths))
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if spatial_step is not None:
                    ratios = np.maximum(ratios, widths * np.linalg.norm(b - a) / spatial_step)
                if time_step is not None:
                    ratios = np.maximum(ratios, widths * (end - start) / time_step)
            if not np.isfinite(ratios).all() or np.any(ratios > max_samples):
                raise ValueError("Trajectory quadrature exceeds max_samples before allocation.")
            counts = np.ceil(ratios).astype(np.int64)
            additions = sum(int(n) + 1 for n in counts)
        sample_count += additions
        panel_count += len(widths)
        if sample_count > max_samples:
            raise ValueError("Trajectory quadrature exceeds max_samples before allocation.")
        plans.append((a, b, start, end, breaks, counts))

    points = np.empty((sample_count, 2))
    queries = np.empty(sample_count)
    weights = np.empty(sample_count)
    offset = 0
    for a, b, start, end, breaks, counts in plans:
        duration = end - start
        if quadrature == "gauss2_split":
            halfwidth = np.diff(breaks) / 2
            midpoint = (breaks[:-1] + breaks[1:]) / 2
            fractions = np.column_stack((midpoint - halfwidth / np.sqrt(3),
                                         midpoint + halfwidth / np.sqrt(3))).ravel()
            time_weights = np.repeat(halfwidth * duration, 2)
        else:
            parts, weight_parts = [], []
            for left, right, count in zip(breaks[:-1], breaks[1:], counts):
                parts.append(np.linspace(left, right, int(count) + 1))
                local_weights = np.full(int(count) + 1, (right - left) * duration / int(count))
                local_weights[[0, -1]] *= 0.5
                weight_parts.append(local_weights)
            fractions, time_weights = np.concatenate(parts), np.concatenate(weight_parts)
        size = len(fractions)
        points[offset:offset + size] = a + fractions[:, None] * (b - a)
        # Evaluate from the nearer endpoint: a weighted sum of two large
        # clock values can round outside their interval. This computes the
        # prescribed Gauss/trapezoid time without clamping any user query.
        queries[offset:offset + size] = np.where(fractions <= 0.5,
                                                 start + fractions * duration,
                                                 end - (1 - fractions) * duration)
        weights[offset:offset + size] = time_weights
        offset += size
    temporal_indices, temporal_weights = _time_weights(times, queries)
    spatial_indices, spatial_weights = bilinear_weights(grid, points, boundary)
    columns = (temporal_indices[:, :, None] * (grid.nx * grid.ny)
               + spatial_indices[:, None, :]).ravel()
    values = (weights[:, None, None] * temporal_weights[:, :, None]
              * spatial_weights[:, None, :]).ravel()
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Trajectory weights must be finite and nonnegative.")
    matrix = coo_matrix((values, (np.zeros(len(columns), dtype=np.int64), columns)),
                        shape=(1, total_columns)).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all():
        raise ValueError("Trajectory weights cannot be represented finitely.")
    for array in (matrix.data, matrix.indices, matrix.indptr):
        array.flags.writeable = False
    times = np.array(times, copy=True)
    times.flags.writeable = False
    descriptor = {"schema_version": 1, "grid": grid.to_dict(), "frame_times_s": times.tolist(),
                  "vertex_xy": trajectory.vertex_xy.tolist(),
                  "vertex_times_s": trajectory.vertex_times_s.tolist(), "boundary": boundary,
                  "quadrature": quadrature, "spatial_step_m": spatial_step, "time_step_s": time_step,
                  "reconstruction": "cell_centred_piecewise_bilinear_space_linear_time_v1"}
    identity = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata = {**descriptor, "operator_sha256": identity, "field_order": "C:frame,y,x",
                "shape": list(matrix.shape), "nnz": int(matrix.nnz),
                "csr_bytes": sum(a.nbytes for a in (matrix.data, matrix.indices, matrix.indptr)),
                "sample_count": sample_count, "max_samples": max_samples,
                "mandatory_panel_count": panel_count,
                "departure_time_s": trajectory.departure_time_s,
                "arrival_time_s": trajectory.arrival_time_s,
                "elapsed_time_s": trajectory.arrival_time_s - trajectory.departure_time_s,
                "includes_waiting": True, "temporal_extrapolation": "reject",
                "build_ms": (perf_counter() - started) * 1000,
                "interpretation": "integral_of_reconstruction_not_exact_PDE_or_physical_exposure"}
    return TrajectoryObserver(grid, times, matrix, metadata)
