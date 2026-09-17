"""Conservative cell-centred diffusion with reflecting (zero flux) walls.

Concentration is dimensionless and is never rescaled or clipped. Spatial
coordinates and Gaussian width are metres; time is seconds and kappa is m²/s.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Grid:
    bounds: tuple[float, float, float, float]
    nx: int = 160
    ny: int = 160

    def __post_init__(self) -> None:
        bounds = tuple(float(value) for value in self.bounds)
        if len(bounds) != 4 or not np.isfinite(bounds).all():
            raise ValueError("Grid bounds must contain four finite coordinates.")
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError("Grid bounds must have positive width and height.")
        if not isinstance(self.nx, int) or not isinstance(self.ny, int) or min(self.nx, self.ny) < 2:
            raise ValueError("Each grid dimension must be an integer of at least two.")
        object.__setattr__(self, "bounds", bounds)

    @property
    def dx(self) -> float:
        return (self.bounds[2] - self.bounds[0]) / self.nx

    @property
    def dy(self) -> float:
        return (self.bounds[3] - self.bounds[1]) / self.ny

    @property
    def h(self) -> float:
        return min(self.dx, self.dy)

    @property
    def x(self) -> np.ndarray:
        return self.bounds[0] + (np.arange(self.nx) + 0.5) * self.dx

    @property
    def y(self) -> np.ndarray:
        return self.bounds[1] + (np.arange(self.ny) + 0.5) * self.dy

    def contains(self, x: float, y: float) -> bool:
        return bool(np.isfinite([x, y]).all() and self.bounds[0] <= x <= self.bounds[2] and self.bounds[1] <= y <= self.bounds[3])

    def to_dict(self) -> dict:
        return {"bounds": list(self.bounds), "nx": self.nx, "ny": self.ny,
                "dx_m": self.dx, "dy_m": self.dy, "location": "cell_centres",
                "axis_order": "frame,y,x", "y_direction": "south_to_north"}


@dataclass
class Simulation:
    grid: Grid
    times_s: np.ndarray
    frames: np.ndarray
    parameters: dict
    diagnostics: dict


def stable_timestep(grid: Grid, kappa: float, safety: float = 0.9) -> float:
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("Diffusion coefficient must be positive and finite.")
    if not 0 < safety <= 1:
        raise ValueError("CFL safety factor must lie in (0, 1].")
    return safety / (2.0 * kappa * (grid.dx ** -2 + grid.dy ** -2))


def gaussian_initial(grid: Grid, source: Sequence[float], sigma_m: float = 250.0) -> np.ndarray:
    if len(source) != 2 or not grid.contains(*source):
        raise ValueError("Release position must lie inside the simulation region.")
    if not np.isfinite(sigma_m) or sigma_m <= 0:
        raise ValueError("Gaussian width must be positive and finite.")
    xx, yy = np.meshgrid(grid.x, grid.y)
    return np.exp(-((xx - source[0]) ** 2 + (yy - source[1]) ** 2) / (2 * sigma_m ** 2))


def diffusion_step(field: np.ndarray, grid: Grid, kappa: float, dt: float) -> np.ndarray:
    """One Euler step; equal/opposite face fluxes enforce mass conservation.

No flux is added at exterior faces. The resulting boundary stencil is the
cell-centred ghost-cell rule c_ghost=c_boundary, not a one-sided derivative.
"""
    field = np.asarray(field, dtype=np.float64)
    if field.shape != (grid.ny, grid.nx):
        raise ValueError("Concentration shape does not match the grid.")
    limit = stable_timestep(grid, kappa, safety=1.0)
    if not np.isfinite(dt) or dt < 0 or dt > limit * (1 + 1e-12):
        raise ValueError("Time step violates the explicit diffusion stability bound.")
    result = field.copy()
    flux_x = (kappa * dt / grid.dx ** 2) * np.diff(field, axis=1)
    flux_y = (kappa * dt / grid.dy ** 2) * np.diff(field, axis=0)
    result[:, :-1] += flux_x
    result[:, 1:] -= flux_x
    result[:-1, :] += flux_y
    result[1:, :] -= flux_y
    return result


def solve_initial(grid: Grid, initial: np.ndarray, kappa: float,
                  output_times_s: Sequence[float], dt: float | None = None) -> tuple[np.ndarray, dict]:
    """Evolve an arbitrary initial field, landing exactly on each output time."""
    times = np.asarray(output_times_s, dtype=np.float64)
    field = np.asarray(initial, dtype=np.float64).copy()
    if field.shape != (grid.ny, grid.nx) or not np.isfinite(field).all():
        raise ValueError("Initial field must be finite and have grid shape (ny,nx).")
    if times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Output times must be finite, nonnegative and strictly increasing.")
    automatic_dt = stable_timestep(grid, kappa)
    chosen_dt = automatic_dt if dt is None else float(dt)
    if not np.isfinite(chosen_dt) or chosen_dt <= 0 or chosen_dt > stable_timestep(grid, kappa, 1.0) * (1 + 1e-12):
        raise ValueError("Time step must be positive and satisfy the stability bound.")
    started = perf_counter()
    frames = np.empty((len(times), grid.ny, grid.nx), dtype=np.float64)
    initial_mass = float(field.sum() * grid.dx * grid.dy)
    max_mass_drift = 0.0
    minimum = float(field.min())
    maximum = float(field.max())
    current_time = 0.0
    steps = 0
    for index, target in enumerate(times):
        while current_time < target:
            actual_dt = min(chosen_dt, float(target - current_time))
            field = diffusion_step(field, grid, kappa, actual_dt)
            current_time = min(float(target), current_time + actual_dt)
            steps += 1
        frames[index] = field
        mass = float(field.sum() * grid.dx * grid.dy)
        denominator = abs(initial_mass) if initial_mass != 0 else 1.0
        max_mass_drift = max(max_mass_drift, abs(mass - initial_mass) / denominator)
        minimum = min(minimum, float(field.min()))
        maximum = max(maximum, float(field.max()))
    diagnostics = {"solve_ms": (perf_counter() - started) * 1000,
                   "internal_steps": steps, "dt_max_s": chosen_dt,
                   "cfl_limit_s": stable_timestep(grid, kappa, 1.0),
                   "initial_mass": initial_mass,
                   "final_mass": float(field.sum() * grid.dx * grid.dy),
                   "relative_mass_drift": max_mass_drift,
                   "min_concentration": minimum, "max_concentration": maximum,
                   "dtype": "float64", "boundary": "zero_flux"}
    return frames, diagnostics


def simulate(grid: Grid, source: Sequence[float], kappa: float = 20.0,
             duration_s: float = 1800.0, frame_interval_s: float = 30.0,
             sigma_m: float = 250.0) -> Simulation:
    if not np.isfinite(duration_s) or duration_s < 0:
        raise ValueError("Duration must be finite and nonnegative.")
    if not np.isfinite(frame_interval_s) or frame_interval_s <= 0:
        raise ValueError("Frame interval must be positive and finite.")
    times = np.arange(0, duration_s, frame_interval_s, dtype=np.float64)
    times = np.append(times, duration_s)
    initial = gaussian_initial(grid, source, sigma_m)
    frames, diagnostics = solve_initial(grid, initial, kappa, times)
    parameters = {"source_xy": [float(v) for v in source], "kappa": float(kappa),
                  "sigma_m": float(sigma_m), "duration_s": float(duration_s),
                  "frame_interval_s": float(frame_interval_s), "initial_peak": 1.0,
                  "method": "conservative_five_point_explicit_euler", "cfl_safety": 0.9}
    return Simulation(grid, times, frames, parameters, diagnostics)


def bilinear_interpolate(field: np.ndarray, grid: Grid, points: np.ndarray) -> np.ndarray:
    """Interpolate cell centres; constant extension to walls models zero flux.

Coordinates outside the physical rectangle are rejected rather than wrapped.
"""
    field = np.asarray(field, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if field.shape != (grid.ny, grid.nx) or not np.isfinite(field).all():
        raise ValueError("Field must be finite and have grid shape.")
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Points must be a finite array of shape (n,2).")
    xmin, ymin, xmax, ymax = grid.bounds
    tolerance = 1e-7
    if np.any(points[:, 0] < xmin - tolerance) or np.any(points[:, 0] > xmax + tolerance) or np.any(points[:, 1] < ymin - tolerance) or np.any(points[:, 1] > ymax + tolerance):
        raise ValueError("Sampling points lie outside the simulation region.")
    fx = np.clip((points[:, 0] - xmin) / grid.dx - 0.5, 0, grid.nx - 1)
    fy = np.clip((points[:, 1] - ymin) / grid.dy - 0.5, 0, grid.ny - 1)
    ix = np.minimum(np.floor(fx).astype(int), grid.nx - 2)
    iy = np.minimum(np.floor(fy).astype(int), grid.ny - 2)
    tx, ty = fx - ix, fy - iy
    return ((1 - tx) * (1 - ty) * field[iy, ix] + tx * (1 - ty) * field[iy, ix + 1]
            + (1 - tx) * ty * field[iy + 1, ix] + tx * ty * field[iy + 1, ix + 1])
