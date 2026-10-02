"""Immutable static isotropic material fields and conservative face rates.

A material value belongs to a cell. Each interior face represents two equal
half-cell resistances in series, hence its harmonic rather than arithmetic
mean. This module does not add Dirichlet or variable-advection support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

from ..diffusion import Grid


VARIABLE_BOUNDARIES = {"zero_flux", "periodic"}


def _immutable(array: np.ndarray) -> np.ndarray:
    """Canonical float64 C array whose immutable bytes owner cannot be reopened."""
    array = np.asarray(array, dtype="<f8", order="C")
    return np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)


def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Positive harmonic means without overflowing ``a*b`` or ``a+b``.

    The result remains finite even at the largest representable coefficient.
    Extreme ratios may round to zero inside ``min/max``; the resulting limit
    ``2*min`` is representable whenever its rounded harmonic mean is.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind not in "fiu" or b.dtype.kind not in "fiu":
        raise ValueError("Harmonic means require real numeric coefficients.")
    left, right = np.broadcast_arrays(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    if (not np.isfinite(left).all() or not np.isfinite(right).all()
            or np.any(left <= 0) or np.any(right <= 0)):
        raise ValueError("Harmonic means require finite strictly positive coefficients.")
    low, high = np.minimum(left, right), np.maximum(left, right)
    with np.errstate(under="ignore"):
        return low * (2.0 / (1.0 + low / high))


def _face_rates(a: np.ndarray, b: np.ndarray, spacing: float) -> np.ndarray:
    # Sequential division avoids forming spacing**2, which can overflow or
    # underflow while the final physical rate is still representable.
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        rates = harmonic_mean(a, b) / spacing / spacing
    if not np.isfinite(rates).all() or np.any(rates <= 0):
        raise ValueError("Diffusion face rates must be finite and positive; coefficient/grid scale overflows or underflows.")
    return rates


@dataclass(frozen=True, init=False)
class DiffusionCoefficients:
    """Validated, immutable coefficient snapshot prepared once per solve.

    All arrays have immutable bytes owners, not merely a reversible writable
    flag. The content hash identifies values and face policy; solver cache keys
    must additionally include grid, boundary, method, and the exact step size.
    """

    grid: Grid
    boundary: str
    values: np.ndarray = field(repr=False)
    x_rates: np.ndarray = field(repr=False)
    y_rates: np.ndarray = field(repr=False)
    x_periodic_rates: np.ndarray | None = field(repr=False)
    y_periodic_rates: np.ndarray | None = field(repr=False)
    outgoing_rates: np.ndarray = field(repr=False)
    max_outgoing_rate: float
    coefficient_sha256: str
    cache_key: tuple[str, str]

    def __init__(self, grid: Grid, values: np.ndarray, boundary: str = "zero_flux") -> None:
        if not isinstance(grid, Grid):
            raise ValueError("A material field requires a Grid.")
        if boundary not in VARIABLE_BOUNDARIES:
            raise ValueError("Variable diffusivity supports zero_flux or periodic boundaries only.")
        if not np.isfinite([grid.dx, grid.dy]).all() or min(grid.dx, grid.dy) <= 0:
            raise ValueError("Grid spacing must be positive and finite.")
        raw = np.asarray(values)
        if raw.shape != (grid.ny, grid.nx) or raw.dtype.kind not in "fiu":
            raise ValueError("Diffusivity must be a real numeric array of shape (ny,nx); complex, bool and object arrays are forbidden.")
        with np.errstate(over="ignore", invalid="ignore"):
            cells = np.array(raw, dtype=np.float64, order="C", copy=True)
        if not np.isfinite(cells).all() or np.any(cells <= 0):
            raise ValueError("Cell diffusivity must be finite and strictly positive.")
        x_rates = _face_rates(cells[:, :-1], cells[:, 1:], grid.dx)
        y_rates = _face_rates(cells[:-1, :], cells[1:, :], grid.dy)
        x_periodic = y_periodic = None
        outgoing = np.zeros_like(cells)
        with np.errstate(over="ignore", invalid="ignore"):
            outgoing[:, :-1] += x_rates
            outgoing[:, 1:] += x_rates
            outgoing[:-1, :] += y_rates
            outgoing[1:, :] += y_rates
            if boundary == "periodic":
                x_periodic = _face_rates(cells[:, -1], cells[:, 0], grid.dx)
                y_periodic = _face_rates(cells[-1, :], cells[0, :], grid.dy)
                outgoing[:, -1] += x_periodic
                outgoing[:, 0] += x_periodic
                outgoing[-1, :] += y_periodic
                outgoing[0, :] += y_periodic
        if not np.isfinite(outgoing).all() or np.any(outgoing <= 0):
            raise ValueError("Total cell outgoing diffusion rates must be finite and positive.")
        frozen_cells = _immutable(cells)
        descriptor = {"schema_version": 1, "shape": list(cells.shape),
                      "dtype": "float64-little-endian", "axis_order": "y,x",
                      "unit": "m2/s", "interpretation": "cell_material",
                      "face_policy": "harmonic_two_point_transmissibility"}
        digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode())
        digest.update(frozen_cells.tobytes(order="C"))
        identity = digest.hexdigest()
        for name, value in {"grid": grid, "boundary": boundary, "values": frozen_cells,
                            "x_rates": _immutable(x_rates), "y_rates": _immutable(y_rates),
                            "x_periodic_rates": None if x_periodic is None else _immutable(x_periodic),
                            "y_periodic_rates": None if y_periodic is None else _immutable(y_periodic),
                            "outgoing_rates": _immutable(outgoing),
                            "max_outgoing_rate": float(outgoing.max()),
                            "coefficient_sha256": identity,
                            "cache_key": ("cell_centred_diffusivity_v1", identity)}.items():
            object.__setattr__(self, name, value)

    def metadata(self) -> dict:
        arrays = [self.values, self.x_rates, self.y_rates, self.outgoing_rates]
        if self.x_periodic_rates is not None:
            arrays.extend([self.x_periodic_rates, self.y_periodic_rates])
        largest_flux = max(array.size for array in arrays[1:3])
        if self.x_periodic_rates is not None:
            largest_flux = max(largest_flux, self.x_periodic_rates.size, self.y_periodic_rates.size)
        return {"type": "cell_centred_field", "coefficient_sha256": self.coefficient_sha256,
                "min": float(self.values.min()), "max": float(self.values.max()),
                "shape": list(self.values.shape), "unit": "m2/s", "interpretation": "cell_material",
                "face_policy": "harmonic_two_point_transmissibility", "boundary": self.boundary,
                "max_outgoing_rate": self.max_outgoing_rate,
                "resident_bytes": sum(array.nbytes for array in arrays),
                "matrix_free_temporary_bytes": 8 * (self.values.size + largest_flux),
                "memory_accounting": "resident: coefficient and face/outgoing arrays; matrix-free temporary: result plus largest face-flux buffer; excludes caller input conversion, validation, Python/NumPy overhead and allocator workspace; not process RSS"}


def prepare_diffusivity(grid: Grid, kappa: float | np.ndarray | DiffusionCoefficients,
                        boundary: str = "zero_flux") -> float | DiffusionCoefficients:
    """Canonical scalar or immutable array snapshot; never resample a material.

    Existing prepared snapshots may be reused only with their exact Grid and
    boundary. Scalar values retain the historical finite nonnegative contract.
    Array values must be strictly positive and use the supported closed domain.
    """
    if isinstance(kappa, DiffusionCoefficients):
        if kappa.grid != grid or kappa.boundary != boundary:
            raise ValueError("Prepared diffusivity grid and boundary must match exactly.")
        return kappa
    if np.isscalar(kappa):
        raw = np.asarray(kappa)
        if raw.dtype.kind not in "fiub":
            raise ValueError("Scalar kappa must be finite, real and nonnegative.")
        value = float(kappa)
        if not np.isfinite(value) or value < 0:
            raise ValueError("Scalar kappa must be finite, real and nonnegative.")
        return value
    return DiffusionCoefficients(grid, kappa, boundary)
