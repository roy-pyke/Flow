"""Prescribed face winds, additive cell sources and advective inflow data.

All stored fields share one strictly covered clock and piecewise-linear time
reconstruction. Transport is conservative ``-div(u*c)``; spatially divergent
winds may change a constant concentration even on a closed domain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..diffusion import Grid

SIDES = ("left", "right", "bottom", "top")
BOUNDARIES = {"zero_flux", "periodic", "open"}


def _freeze(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="<f8", order="C")
    return np.frombuffer(values.tobytes(order="C"), dtype="<f8").reshape(values.shape)


def _array(values: np.ndarray | None, shape: tuple[int, ...], name: str) -> np.ndarray:
    if values is None:
        return _freeze(np.zeros(shape, dtype=np.float64))
    raw = np.asarray(values)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array of shape {shape}; bool, complex and object are forbidden.")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain finite float64 values.")
    return _freeze(converted)


def _validate_boundary(velocity_x: np.ndarray, velocity_y: np.ndarray,
                       inflow: Mapping[str, np.ndarray], boundary: str) -> None:
    if boundary not in BOUNDARIES:
        raise ValueError("Forcing boundary must be zero_flux, periodic or open.")
    # Ellipsis permits both a full time series and an instantaneous snapshot.
    if boundary == "zero_flux":
        if (np.any(velocity_x[..., 0] != 0) or np.any(velocity_x[..., -1] != 0)
                or np.any(velocity_y[..., 0, :] != 0) or np.any(velocity_y[..., -1, :] != 0)):
            raise ValueError("Closed forcing requires zero normal velocity on every exterior face.")
    elif boundary == "periodic":
        if (not np.array_equal(velocity_x[..., 0], velocity_x[..., -1])
                or not np.array_equal(velocity_y[..., 0, :], velocity_y[..., -1, :])):
            raise ValueError("Periodic duplicate face velocities must match exactly; seam averaging is forbidden.")
    if any(np.any(values < 0) for values in inflow.values()):
        raise ValueError("Prescribed inflow concentrations must be nonnegative.")
    if boundary != "open" and any(np.any(values != 0) for values in inflow.values()):
        raise ValueError("Nonzero inflow concentrations require open boundaries.")


def _outgoing(grid: Grid, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = (np.maximum(vx[:, 1:], 0) / grid.dx + np.maximum(-vx[:, :-1], 0) / grid.dx
                  + np.maximum(vy[1:, :], 0) / grid.dy + np.maximum(-vy[:-1, :], 0) / grid.dy)
    if not np.isfinite(result).all():
        raise ValueError("Advective outgoing rates cannot be represented finitely on this grid.")
    return result


def _clock(value: float) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "fiu":
        raise ValueError("Query time must be a finite real scalar.")
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError("Query time must be a finite real scalar.")
    return result


@dataclass(frozen=True)
class ForcingSnapshot:
    """Instantaneous read-only arrays returned by ``PrescribedFields.at``."""
    grid: Grid
    boundary: str
    time_s: float
    velocity_x: np.ndarray = field(repr=False)
    velocity_y: np.ndarray = field(repr=False)
    source: np.ndarray = field(repr=False)
    inflow: Mapping[str, np.ndarray] = field(repr=False)

    def __post_init__(self) -> None:
        grid = self.grid
        if not isinstance(grid, Grid) or not np.isfinite([grid.dx, grid.dy]).all() or min(grid.dx, grid.dy) <= 0:
            raise ValueError("A forcing snapshot requires a valid Grid.")
        vx = _array(self.velocity_x, (grid.ny, grid.nx+1), "snapshot velocity_x")
        vy = _array(self.velocity_y, (grid.ny+1, grid.nx), "snapshot velocity_y")
        source = _array(self.source, (grid.ny, grid.nx), "snapshot source")
        if not isinstance(self.inflow, Mapping) or set(self.inflow) != set(SIDES):
            raise ValueError("A forcing snapshot requires all four inflow sides.")
        incoming = {side: _array(self.inflow[side],
                                (grid.ny if side in ("left", "right") else grid.nx,), "snapshot inflow_"+side)
                    for side in SIDES}
        _validate_boundary(vx, vy, incoming, self.boundary)
        for name, value in {"velocity_x": vx, "velocity_y": vy, "source": source,
                            "inflow": MappingProxyType(incoming), "time_s": _clock(self.time_s)}.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, init=False)
class PrescribedFields:
    """An immutable time series of face winds, cell sources and boundary data.

    ``times_s`` contains at least two strictly increasing finite entries and
    starts exactly at zero. Shapes are ``(nt,ny,nx+1)`` for x faces,
    ``(nt,ny+1,nx)`` for y faces, ``(nt,ny,nx)`` for the additive source,
    ``(nt,ny)`` for left/right inflow and ``(nt,nx)`` for bottom/top inflow.
    None and omitted inflow sides mean zero arrays; scalar broadcasting is not
    implicit. Source values may be signed. Inflow concentrations are nonnegative
    and apply only on geometrically incoming faces of an open boundary.

    All arrays are copied to immutable bytes owners. Preparation does not cap
    process RSS; callers loading external arrays must apply input budgets first.
    """
    grid: Grid
    boundary: str
    times_s: np.ndarray = field(repr=False)
    velocity_x: np.ndarray = field(repr=False)
    velocity_y: np.ndarray = field(repr=False)
    source: np.ndarray = field(repr=False)
    inflow: Mapping[str, np.ndarray] = field(repr=False)
    max_outgoing_rate: float
    has_velocity: bool
    has_source: bool
    has_inflow: bool
    content_sha256: str
    _max_abs_divergence: float

    def __init__(self, grid: Grid, times_s: np.ndarray, velocity_x: np.ndarray | None = None,
                 velocity_y: np.ndarray | None = None, source: np.ndarray | None = None,
                 inflow: Mapping[str, np.ndarray] | None = None,
                 boundary: str = "zero_flux") -> None:
        if not isinstance(grid, Grid) or not np.isfinite([grid.dx, grid.dy]).all() or min(grid.dx, grid.dy) <= 0:
            raise ValueError("Forcing requires a valid Grid with finite positive spacings.")
        raw_times = np.asarray(times_s)
        if raw_times.ndim != 1 or len(raw_times) < 2 or raw_times.dtype.kind not in "fiu":
            raise ValueError("Forcing times must be a real vector with at least two entries.")
        times = _array(raw_times, raw_times.shape, "times_s")
        if times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError("Forcing times must start at zero and be strictly increasing.")
        nt = len(times)
        vx = _array(velocity_x, (nt, grid.ny, grid.nx + 1), "velocity_x")
        vy = _array(velocity_y, (nt, grid.ny + 1, grid.nx), "velocity_y")
        sources = _array(source, (nt, grid.ny, grid.nx), "source")
        if inflow is not None and (not isinstance(inflow, Mapping) or not set(inflow) <= set(SIDES)):
            raise ValueError("Inflow must be a mapping with only left, right, bottom and top keys.")
        incoming = {side: _array(None if inflow is None else inflow.get(side),
                                (nt, grid.ny if side in ("left", "right") else grid.nx), "inflow_" + side)
                    for side in SIDES}
        _validate_boundary(vx, vy, incoming, boundary)
        maximum, max_divergence = 0.0, 0.0
        for x_faces, y_faces in zip(vx, vy):
            maximum = max(maximum, float(_outgoing(grid, x_faces, y_faces).max()))
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                divergence = np.diff(x_faces, axis=1) / grid.dx + np.diff(y_faces, axis=0) / grid.dy
            if not np.isfinite(divergence).all():
                raise ValueError("Discrete face-wind divergence cannot be represented finitely.")
            max_divergence = max(max_divergence, float(np.max(np.abs(divergence))))
        descriptor = {"schema_version": 1, "grid": grid.to_dict(), "boundary": boundary,
                      "time_reconstruction": "piecewise_linear", "unit_velocity": "m/s",
                      "unit_source": "relative_concentration/s", "unit_inflow": "relative_concentration"}
        digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode())
        arrays = {"times_s": times, "velocity_x": vx, "velocity_y": vy, "source": sources,
                  **{"inflow_" + side: incoming[side] for side in SIDES}}
        for name, values in arrays.items():
            digest.update(name.encode() + b"\0")
            digest.update(values.tobytes(order="C"))
        for name, value in {"grid": grid, "boundary": boundary, "times_s": times,
                            "velocity_x": vx, "velocity_y": vy, "source": sources,
                            "inflow": MappingProxyType(incoming), "max_outgoing_rate": maximum,
                            "has_velocity": bool(np.any(vx != 0) or np.any(vy != 0)),
                            "has_source": bool(np.any(sources != 0)),
                            "has_inflow": any(np.any(values != 0) for values in incoming.values()),
                            "content_sha256": digest.hexdigest(), "_max_abs_divergence": max_divergence}.items():
            object.__setattr__(self, name, value)

    def validate_interval(self, start: float, end: float) -> None:
        start, end = _clock(start), _clock(end)
        if start < self.times_s[0] or end > self.times_s[-1] or end < start:
            raise ValueError("Requested interval must be covered by forcing times; extrapolation is forbidden.")

    def next_knot(self, time_s: float) -> float | None:
        time_s = _clock(time_s)
        self.validate_interval(time_s, time_s)
        index = int(np.searchsorted(self.times_s, time_s, side="right"))
        return float(self.times_s[index]) if index < len(self.times_s) else None

    def at(self, time_s: float) -> ForcingSnapshot:
        time_s = _clock(time_s)
        self.validate_interval(time_s, time_s)
        index = int(np.searchsorted(self.times_s, time_s, side="left"))
        exact = self.times_s[index] == time_s
        if not exact:
            lower = index - 1
            alpha = (time_s-self.times_s[lower]) / (self.times_s[index]-self.times_s[lower])
        def interpolate(values: np.ndarray) -> np.ndarray:
            if exact:
                return values[index]
            with np.errstate(over="ignore", invalid="ignore"):
                result = (1-alpha)*values[lower] + alpha*values[index]
            if not np.isfinite(result).all():
                raise ValueError("Time-interpolated forcing cannot be represented finitely.")
            return result
        return ForcingSnapshot(self.grid, self.boundary, time_s, interpolate(self.velocity_x),
                               interpolate(self.velocity_y), interpolate(self.source),
                               MappingProxyType({side: interpolate(values) for side, values in self.inflow.items()}))

    def metadata(self) -> dict:
        arrays = [self.times_s, self.velocity_x, self.velocity_y, self.source, *self.inflow.values()]
        return {"schema_version": 1, "type": "prescribed_space_time_fields",
                "content_sha256": self.content_sha256, "boundary": self.boundary,
                "frame_count": len(self.times_s), "time_coverage_s": [float(self.times_s[0]), float(self.times_s[-1])],
                "time_reconstruction": "piecewise_linear", "temporal_extrapolation": "reject",
                "source_interpretation": "signed_additive_cell_rate", "inflow_interpretation": "nonnegative_advective_incoming_concentration",
                "has_velocity": self.has_velocity, "has_source": self.has_source, "has_inflow": self.has_inflow,
                "max_outgoing_rate": self.max_outgoing_rate,
                "outgoing_bound_scope": "global maximum over all cells and time knots; valid between knots by convexity of positive-part face rates",
                "max_abs_discrete_divergence": self._max_abs_divergence,
                "resident_bytes": sum(values.nbytes for values in arrays),
                "snapshot_array_bytes": sum(values[0].nbytes for values in arrays[1:]),
                "memory_accounting": "retained field-array bytes and one materialized snapshot; excludes validation/interpolation temporaries, Python/NumPy overhead and allocator workspace; not process RSS",
                "units": {"velocity": "m/s", "source": "relative_concentration/s", "inflow": "relative_concentration"}}


def face_advection_rate(field: np.ndarray, grid: Grid, snapshot: ForcingSnapshot,
                         boundary: str) -> tuple[np.ndarray, dict[str, float]]:
    """Conservative upwind ``-div(u*c)`` with an oriented boundary ledger.

    Ledger ``outward`` and ``inward`` are geometric outgoing/incoming
    contributions, in relative concentration*m²/s, and ``net_outward`` is their
    difference. Outward can be negative for a signed interior field; incoming
    data remain nonnegative. Diffusion and cell-source contributions are not
    included. An open inflow value specifies its advective incoming flux, not
    a production Dirichlet concentration boundary for diffusion.
    """
    if not isinstance(snapshot, ForcingSnapshot) or snapshot.grid != grid or snapshot.boundary != boundary:
        raise ValueError("Forcing snapshot grid and boundary must match the operator.")
    raw = np.asarray(field)
    if raw.dtype.kind not in "fiu" or raw.shape != (grid.ny, grid.nx):
        raise ValueError("Transport field must be a real numeric array of grid shape.")
    values = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Transport field must be finite.")
    vx, vy = snapshot.velocity_x, snapshot.velocity_y
    if (vx.shape != (grid.ny, grid.nx+1) or vy.shape != (grid.ny+1, grid.nx)
            or not np.isfinite(vx).all() or not np.isfinite(vy).all()
            or set(snapshot.inflow) != set(SIDES)):
        raise ValueError("Invalid face-wind snapshot arrays.")
    for side, incoming in snapshot.inflow.items():
        if incoming.shape != (grid.ny if side in ("left", "right") else grid.nx,) or not np.isfinite(incoming).all():
            raise ValueError("Invalid inflow snapshot arrays.")
    _validate_boundary(vx, vy, snapshot.inflow, boundary)
    fx, fy = np.zeros_like(vx), np.zeros_like(vy)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        fx[:, 1:-1] = vx[:, 1:-1] * np.where(vx[:, 1:-1] >= 0, values[:, :-1], values[:, 1:])
        fy[1:-1, :] = vy[1:-1, :] * np.where(vy[1:-1, :] >= 0, values[:-1, :], values[1:, :])
        if boundary == "periodic":
            fx[:, 0] = vx[:, 0] * np.where(vx[:, 0] >= 0, values[:, -1], values[:, 0])
            fx[:, -1] = fx[:, 0]
            fy[0, :] = vy[0, :] * np.where(vy[0, :] >= 0, values[-1, :], values[0, :])
            fy[-1, :] = fy[0, :]
        elif boundary == "open":
            fx[:, 0] = vx[:, 0] * np.where(vx[:, 0] >= 0, snapshot.inflow["left"], values[:, 0])
            fx[:, -1] = vx[:, -1] * np.where(vx[:, -1] >= 0, values[:, -1], snapshot.inflow["right"])
            fy[0, :] = vy[0, :] * np.where(vy[0, :] >= 0, snapshot.inflow["bottom"], values[0, :])
            fy[-1, :] = vy[-1, :] * np.where(vy[-1, :] >= 0, values[-1, :], snapshot.inflow["top"])
        rate = -np.diff(fx, axis=1)/grid.dx - np.diff(fy, axis=0)/grid.dy
        if boundary == "open":
            outward = grid.dy*(np.where(vx[:, 0] < 0, -fx[:, 0], 0).sum()
                               + np.where(vx[:, -1] > 0, fx[:, -1], 0).sum())
            outward += grid.dx*(np.where(vy[0, :] < 0, -fy[0, :], 0).sum()
                                + np.where(vy[-1, :] > 0, fy[-1, :], 0).sum())
            inward = grid.dy*(np.where(vx[:, 0] > 0, fx[:, 0], 0).sum()
                              + np.where(vx[:, -1] < 0, -fx[:, -1], 0).sum())
            inward += grid.dx*(np.where(vy[0, :] > 0, fy[0, :], 0).sum()
                               + np.where(vy[-1, :] < 0, -fy[-1, :], 0).sum())
        else:
            outward = inward = 0.0
        net = float(outward-inward)
    if not np.isfinite(rate).all() or not np.isfinite([outward, inward, net]).all():
        raise FloatingPointError("Nonfinite conservative transport rate or boundary ledger.")
    return rate, {"outward": float(outward), "inward": float(inward), "net_outward": net}
