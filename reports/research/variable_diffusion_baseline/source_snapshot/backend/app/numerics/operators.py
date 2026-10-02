"""Conservative uniform-grid operators (C-order: y, x).

At open inflow the *total* prescribed flux is u.n*c_in=0. Diffusion
therefore adds no exterior contribution; outlet transport is upwind.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from ..diffusion import Grid
from .coefficients import DiffusionCoefficients, prepare_diffusivity

BOUNDARIES = {"zero_flux", "periodic", "open"}


def diffusion_matrix(grid: Grid, kappa: float | np.ndarray | DiffusionCoefficients,
                     boundary: str = "zero_flux") -> sparse.csc_matrix:
    """Return L including kappa, assembled from equal/opposite face fluxes."""
    if not np.isscalar(kappa):
        material = prepare_diffusivity(grid, kappa, boundary)
        return _material_matrix(material)
    if boundary not in BOUNDARIES:
        raise ValueError("Unknown boundary condition.")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative.")

    def axis(n: int, spacing: float) -> sparse.csc_matrix:
        left = np.arange(n - 1)
        right = left + 1
        if boundary == "periodic":
            left = np.append(left, n - 1)
            right = np.append(right, 0)
        rows = np.concatenate((left, right, left, right))
        cols = np.concatenate((right, left, left, right))
        scale = kappa / spacing ** 2
        values = np.concatenate((np.full(2 * len(left), scale), np.full(2 * len(left), -scale)))
        return sparse.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsc()

    return (sparse.kron(sparse.eye(grid.ny), axis(grid.nx, grid.dx), format="csc")
            + sparse.kron(axis(grid.ny, grid.dy), sparse.eye(grid.nx), format="csc")).tocsc()


def diffusion_rate(field: np.ndarray, grid: Grid, kappa: float | np.ndarray | DiffusionCoefficients,
                   boundary: str = "zero_flux") -> np.ndarray:
    """Vectorized face divergence; exactly the same spatial operator as L."""
    if not np.isscalar(kappa):
        material = prepare_diffusivity(grid, kappa, boundary)
        return _material_rate(field, material)
    if boundary not in BOUNDARIES:
        raise ValueError("Unknown boundary condition.")
    result = np.zeros_like(field, dtype=np.float64)
    fx = (kappa / grid.dx ** 2) * np.diff(field, axis=1)
    fy = (kappa / grid.dy ** 2) * np.diff(field, axis=0)
    result[:, :-1] += fx
    result[:, 1:] -= fx
    result[:-1, :] += fy
    result[1:, :] -= fy
    if boundary == "periodic":
        fx = (kappa / grid.dx ** 2) * (field[:, 0] - field[:, -1])
        fy = (kappa / grid.dy ** 2) * (field[0, :] - field[-1, :])
        result[:, -1] += fx
        result[:, 0] -= fx
        result[-1, :] += fy
        result[0, :] -= fy
    return result



def _material_matrix(material: DiffusionCoefficients) -> sparse.csc_matrix:
    """One symmetric graph-Laplacian contribution for every physical face."""
    grid = material.grid
    cells = np.arange(grid.nx * grid.ny, dtype=np.int64).reshape(grid.ny, grid.nx)
    left = [cells[:, :-1].ravel(), cells[:-1, :].ravel()]
    right = [cells[:, 1:].ravel(), cells[1:, :].ravel()]
    conductances = [material.x_rates.ravel(), material.y_rates.ravel()]
    if material.boundary == "periodic":
        # For a two-cell periodic axis these are a second physical face, not
        # an accidental duplicate. COO summation retains both conductances.
        left.extend([cells[:, -1], cells[-1, :]])
        right.extend([cells[:, 0], cells[0, :]])
        conductances.extend([material.x_periodic_rates, material.y_periodic_rates])
    left, right, rates = np.concatenate(left), np.concatenate(right), np.concatenate(conductances)
    rows = np.concatenate((left, right, left, right))
    cols = np.concatenate((right, left, left, right))
    values = np.concatenate((rates, rates, -rates, -rates))
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(cells.size, cells.size)).tocsc()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _material_rate(field: np.ndarray, material: DiffusionCoefficients) -> np.ndarray:
    """Flux differences, retaining only one directional temporary at a time."""
    raw = np.asarray(field)
    if raw.dtype.kind not in "fiu":
        raise ValueError("Field must be real numeric; complex, bool and object arrays are forbidden.")
    field = np.asarray(raw, dtype=np.float64)
    if field.shape != material.values.shape or not np.isfinite(field).all():
        raise ValueError("Field must be finite and have grid shape (ny,nx).")
    result = np.zeros_like(field)
    flux = np.subtract(field[:, 1:], field[:, :-1])
    flux *= material.x_rates
    result[:, :-1] += flux
    result[:, 1:] -= flux
    del flux
    flux = np.subtract(field[1:, :], field[:-1, :])
    flux *= material.y_rates
    result[:-1, :] += flux
    result[1:, :] -= flux
    del flux
    if material.boundary == "periodic":
        flux = np.subtract(field[:, 0], field[:, -1])
        flux *= material.x_periodic_rates
        result[:, -1] += flux
        result[:, 0] -= flux
        del flux
        flux = np.subtract(field[0, :], field[-1, :])
        flux *= material.y_periodic_rates
        result[-1, :] += flux
        result[0, :] -= flux
    return result


def advection_rate(field: np.ndarray, grid: Grid, velocity: tuple[float, float],
                   boundary: str) -> tuple[np.ndarray, float]:
    """Return conservative upwind -div(u*c), plus signed outward mass/s.

    The outward flux includes physical face length. For nonnegative c and
    zero-concentration inflow, this is nonnegative; signed input is permitted
    for diagnostics and is not clipped.
    """
    if boundary not in BOUNDARIES:
        raise ValueError("Unknown boundary condition.")
    ux, uy = velocity
    if boundary == "zero_flux" and (ux != 0 or uy != 0):
        raise ValueError("Nonzero wind requires periodic or open boundaries.")
    fx = np.zeros((grid.ny, grid.nx + 1), dtype=np.float64)
    fy = np.zeros((grid.ny + 1, grid.nx), dtype=np.float64)
    fx[:, 1:-1] = ux * (field[:, :-1] if ux >= 0 else field[:, 1:])
    fy[1:-1, :] = uy * (field[:-1, :] if uy >= 0 else field[1:, :])
    if boundary == "periodic":
        fx[:, 0] = ux * (field[:, -1] if ux >= 0 else field[:, 0])
        fx[:, -1] = fx[:, 0]
        fy[0, :] = uy * (field[-1, :] if uy >= 0 else field[0, :])
        fy[-1, :] = fy[0, :]
    elif boundary == "open":
        if ux >= 0:
            fx[:, -1] = ux * field[:, -1]
        else:
            fx[:, 0] = ux * field[:, 0]
        if uy >= 0:
            fy[-1, :] = uy * field[-1, :]
        else:
            fy[0, :] = uy * field[0, :]
    rate = -np.diff(fx, axis=1) / grid.dx - np.diff(fy, axis=0) / grid.dy
    outflow = float(grid.dy * (fx[:, -1].sum() - fx[:, 0].sum())
                    + grid.dx * (fy[-1, :].sum() - fy[0, :].sum()))
    return rate, outflow


def advection_matrix(grid: Grid, velocity: tuple[float, float], boundary: str) -> sparse.csc_matrix:
    """Sparse independent upwind operator for small-grid exponential references."""
    if boundary not in BOUNDARIES:
        raise ValueError("Unknown boundary condition.")
    if not np.isfinite(velocity).all() or len(velocity) != 2:
        raise ValueError("Velocity must contain two finite components.")
    if boundary == "zero_flux" and any(velocity):
        raise ValueError("Nonzero wind requires periodic or open boundaries.")

    def axis(n: int, spacing: float, speed: float) -> sparse.csc_matrix:
        if speed == 0:
            return sparse.csc_matrix((n, n))
        rows = np.arange(n)
        upstream = rows - (1 if speed > 0 else -1)
        valid = (upstream >= 0) & (upstream < n)
        if boundary == "periodic":
            upstream %= n
            valid[:] = True
        rate = abs(speed) / spacing
        return sparse.coo_matrix((np.r_[np.full(n, -rate), np.full(valid.sum(), rate)],
                                  (np.r_[rows, rows[valid]], np.r_[rows, upstream[valid]])),
                                 shape=(n, n)).tocsc()

    return (sparse.kron(sparse.eye(grid.ny), axis(grid.nx, grid.dx, velocity[0]), format="csc")
            + sparse.kron(axis(grid.ny, grid.dy, velocity[1]), sparse.eye(grid.nx), format="csc")).tocsc()
