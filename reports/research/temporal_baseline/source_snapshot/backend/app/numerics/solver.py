"""Verified FE, theta diffusion, and first-order upwind IMEX integration."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from time import perf_counter
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ..diffusion import Grid, diffusion_step
from .operators import BOUNDARIES, advection_rate, diffusion_matrix, diffusion_rate

METHOD_BACKENDS = {
    "explicit_euler": {"numpy", "cpp"},
    "backward_euler": {"scipy"},
    "crank_nicolson": {"scipy"},
    "advection_explicit": {"numpy"},
    "imex_euler": {"numpy_scipy"},
}
CACHE_MAX_ENTRIES = 8
CACHE_MAX_BYTES = 256 * 1024 ** 2
_CACHE: OrderedDict[tuple, object] = OrderedDict()
_CACHE_LOCK = RLock()


@dataclass
class Factorization:
    matrix: sparse.csc_matrix
    lu: object
    nbytes: int
    lock: RLock


def _sparse_bytes(matrix: sparse.csc_matrix) -> int:
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def clear_factor_cache() -> None:
    """Clear LU factors so reports can measure an explicitly cold invocation."""
    with _CACHE_LOCK:
        _CACHE.clear()


def factor_cache_info() -> dict:
    with _CACHE_LOCK:
        return {"entries": len(_CACHE), "bytes": sum(item.nbytes for item in _CACHE.values()),
                "max_entries": CACHE_MAX_ENTRIES, "max_bytes": CACHE_MAX_BYTES,
                "memory_accounting": "CSC system and SuperLU L/U arrays; excludes allocator/workspace overhead"}


def _factor(key: tuple, operator: sparse.csc_matrix, theta_dt: float) -> tuple[Factorization, bool, float]:
    with _CACHE_LOCK:
        if key in _CACHE:
            entry = _CACHE.pop(key)
            _CACHE[key] = entry
            return entry, True, 0.0
        begin = perf_counter()
        matrix = (sparse.eye(operator.shape[0], format="csc") - theta_dt * operator).tocsc()
        lu = splu(matrix)
        elapsed = (perf_counter() - begin) * 1000
        nbytes = _sparse_bytes(matrix) + _sparse_bytes(lu.L) + _sparse_bytes(lu.U)
        entry = Factorization(matrix, lu, nbytes, RLock())
        # An individual oversized LU is used once without retaining it in the cache.
        if nbytes <= CACHE_MAX_BYTES:
            while _CACHE and (len(_CACHE) >= CACHE_MAX_ENTRIES
                             or sum(item.nbytes for item in _CACHE.values()) + nbytes > CACHE_MAX_BYTES):
                _CACHE.popitem(last=False)
            _CACHE[key] = entry
        return entry, False, elapsed


def resolve_backend(method: str, backend: str, boundary: str = "zero_flux") -> str:
    if method not in METHOD_BACKENDS:
        raise ValueError(f"Unknown numerical method: {method}.")
    if backend == "auto":
        if method == "explicit_euler":
            if boundary == "zero_flux":
                from .native import status
                return "cpp" if status()["available"] else "numpy"
            return "numpy"
        return next(iter(METHOD_BACKENDS[method]))
    if backend not in METHOD_BACKENDS[method]:
        raise ValueError(f"Method {method} does not support backend {backend}.")
    if backend == "cpp" and boundary != "zero_flux":
        raise ValueError("The C++ kernel currently supports zero_flux boundaries only.")
    return backend


def timestep_limit(grid: Grid, kappa: float, method: str,
                   velocity: tuple[float, float]) -> float:
    """Production limits; diffusion FE retains V1's exact CFL bound."""
    diffusion = 2 * kappa * (grid.dx ** -2 + grid.dy ** -2)
    advection = abs(velocity[0]) / grid.dx + abs(velocity[1]) / grid.dy
    if method == "explicit_euler":
        return 1 / diffusion if diffusion else np.inf
    if method == "advection_explicit":
        return 0.9 / (diffusion + advection) if diffusion + advection else np.inf
    if method == "imex_euler":
        return 0.9 / advection if advection else np.inf
    return np.inf


def solve(grid: Grid, initial: np.ndarray, kappa: float,
          output_times_s: Sequence[float], *, method: str = "explicit_euler",
          backend: str = "numpy", dt: float | None = None,
          boundary: str = "zero_flux", velocity: tuple[float, float] = (0.0, 0.0),
          startup: str = "none") -> tuple[np.ndarray, dict]:
    """Integrate without clipping or renormalization; land on requested times.

    Rannacher replaces the first actual CN macro step by two BE half steps.
    Diagnostics of mass, energy and extrema are sampled at output times (plus
    the initial state); linear residuals are checked at every implicit solve.
    """
    total_begin = perf_counter()
    conversion_begin = perf_counter()
    if not np.isfinite([grid.dx, grid.dy]).all() or min(grid.dx, grid.dy) <= 0:
        raise ValueError("Grid spacing must be positive and finite.")
    field = np.array(initial, dtype=np.float64, order="C", copy=True)
    times = np.asarray(output_times_s, dtype=np.float64)
    if field.shape != (grid.ny, grid.nx) or not np.isfinite(field).all():
        raise ValueError("Initial field must be finite and have shape (ny,nx).")
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or times[0] < 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("Output times must be finite, nonnegative and strictly increasing.")
    if not np.isscalar(kappa) or not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative.")
    kappa = float(kappa)
    wind = np.asarray(velocity, dtype=np.float64)
    if wind.shape != (2,) or not np.isfinite(wind).all():
        raise ValueError("Velocity must contain two finite components.")
    velocity = (float(wind[0]), float(wind[1]))
    if boundary not in BOUNDARIES:
        raise ValueError("Unknown boundary condition.")
    selected_backend = resolve_backend(method, backend, boundary)
    if method in {"explicit_euler", "backward_euler", "crank_nicolson"} and any(velocity):
        raise ValueError("Pure diffusion methods require zero velocity.")
    if boundary == "zero_flux" and any(velocity):
        raise ValueError("Nonzero wind requires periodic or open boundaries.")
    if startup not in {"none", "rannacher"} or (startup != "none" and method != "crank_nicolson"):
        raise ValueError("Rannacher startup is supported only for Crank-Nicolson.")
    limit = timestep_limit(grid, kappa, method, velocity)
    if dt is None:
        if np.isfinite(limit):
            chosen_dt = limit * (0.9 if method == "explicit_euler" else 1.0)
        else:
            # Output alignment shortens individual steps; a tiny final output
            # interval must not shrink every preceding macro step.
            chosen_dt = min(30.0, float(times[-1])) if times[-1] > 0 else 1.0
    else:
        chosen_dt = float(dt)
    if not np.isfinite(chosen_dt) or chosen_dt <= 0 or chosen_dt > limit * (1 + 1e-12):
        raise ValueError("Time step must be positive, finite, and satisfy the method's CFL bound.")
    conversion_ms = (perf_counter() - conversion_begin) * 1000
    if selected_backend == "cpp":
        from .native import solve_native
        frames, diagnostics = solve_native(grid, field, kappa, times, chosen_dt)
        diagnostic_begin = perf_counter()
        area = grid.dx * grid.dy
        mass0 = float(field.sum() * area)
        mass_scale = max(abs(mass0), float(np.abs(field).sum() * area), 1.0)
        masses = [float(value.sum() * area) for value in frames]
        energies = [float(area * np.sum((value - value.mean()) ** 2)) for value in frames]
        energy0 = float(area * np.sum((field - field.mean()) ** 2))
        minima = [float(value.min()) for value in frames]
        maxima = [float(value.max()) for value in frames]
        balance = [value - mass0 for value in masses]
        native_steps = diagnostics.get("actual_step_sizes_s", [])
        actual_steps = np.asarray(native_steps, dtype=np.float64)
        actual_histogram: dict[str, int] = {}
        for actual in actual_steps:
            key = repr(float(actual))
            actual_histogram[key] = actual_histogram.get(key, 0) + 1
        diagnostics.update({"method": method, "backend": "cpp", "requested_backend": backend,
                            "requested_dt_s": dt, "startup": startup, "velocity": list(velocity),
                            "linear_solver": "none", "diagnostic_scope": "initial_and_output_frames",
                            "schema_version": 2, "boundary": boundary, "dtype": "float64",
                            "actual_dt_min_s": float(actual_steps.min()) if actual_steps.size else 0.0,
                            "actual_dt_max_s": float(actual_steps.max()) if actual_steps.size else 0.0,
                            "actual_dt_histogram": actual_histogram,
                            "cfl_limit_s": float(limit) if np.isfinite(limit) else None,
                            "mass_scale": mass_scale, "mass_scale_definition": "max(abs(M0), integral(abs(c0)), 1)",
                            "initial_mass": mass0, "final_mass": masses[-1],
                            "min_concentration": min(float(field.min()), min(minima)),
                            "max_concentration": max(float(field.max()), max(maxima)),
                            "relative_mass_drift": max(abs(value) for value in balance) / mass_scale,
                            "mass_balance_error": balance[-1], "max_mass_balance_error": max(abs(value) for value in balance),
                            "relative_mass_balance_error": max(abs(value) for value in balance) / mass_scale,
                            "cumulative_outward_flux": 0.0, "mass_history": masses, "energy_history": energies,
                            "min_history": minima, "max_history": maxima, "mass_balance_history": balance,
                            "cumulative_outflow_history": [0.0] * len(times), "initial_energy": energy0,
                            "final_energy": energies[-1], "energy_max_increase": max(0.0, float(np.max(np.diff([energy0] + energies)))),
                            "output_times_s": times.tolist(), "linear_residual_max": 0.0,
                            "factorization_count": 0, "factorization_cache_hits": 0,
                            "factorization_cache_entries": 0, "factorization_cache_bytes": 0,
                            "output_bytes": frames.nbytes,
                            "timestep_policy": "automatic" if dt is None else "requested_with_exact_output_alignment"})
        diagnostics["timings_ms"] = {
            "matrix": 0.0, "factorization": 0.0, "advance": diagnostics.get("kernel_ms", 0.0),
            "conversion": conversion_ms + diagnostics.get("conversion_ms", 0.0),
            "output": diagnostics.get("output_allocation_ms", 0.0),
            "dispatch_diagnostics": (perf_counter() - diagnostic_begin) * 1000,
            "total": (perf_counter() - total_begin) * 1000,
        }
        diagnostics["solve_ms"] = diagnostics["timings_ms"]["total"]
        return frames, diagnostics

    area = grid.dx * grid.dy
    mass0 = float(field.sum() * area)
    mass_scale = max(abs(mass0), float(np.abs(field).sum() * area), 1.0)
    energy0 = float(area * np.sum((field - field.mean()) ** 2))
    min0, max0 = float(field.min()), float(field.max())
    output_begin = perf_counter()
    frames = np.empty((len(times), grid.ny, grid.nx), dtype=np.float64)
    output_ms = (perf_counter() - output_begin) * 1000
    matrix_ms = factor_ms = advance_ms = 0.0
    implicit = method in {"backward_euler", "crank_nicolson", "imex_euler"}
    operator = None
    if implicit and kappa:
        matrix_begin = perf_counter()
        operator = diffusion_matrix(grid, kappa, boundary)
        matrix_ms = (perf_counter() - matrix_begin) * 1000
    masses, energies, minima, maxima, outflows, balances = [], [], [], [], [], []
    steps = factor_count = cache_hits = 0
    min_step = np.inf
    max_step = cumulative_outflow = residual_max = 0.0
    current_time = 0.0
    startup_done = False
    step_histogram: dict[str, int] = {}

    def advance(actual_dt: float, step_method: str) -> None:
        nonlocal field, steps, min_step, max_step, cumulative_outflow, factor_count
        nonlocal cache_hits, factor_ms, advance_ms, residual_max
        step_begin = perf_counter()
        advection = np.zeros_like(field) if step_method == "imex_euler" else None
        if step_method in {"advection_explicit", "imex_euler"}:
            advection, outflow = advection_rate(field, grid, velocity, boundary)
            cumulative_outflow += actual_dt * outflow
        if step_method == "explicit_euler":
            if kappa and boundary != "periodic":
                field = diffusion_step(field, grid, kappa, actual_dt)
            elif kappa:
                field = field + actual_dt * diffusion_rate(field, grid, kappa, boundary)
        elif step_method == "advection_explicit":
            field = field + actual_dt * (advection + diffusion_rate(field, grid, kappa, boundary))
        elif not kappa:
            if step_method == "imex_euler":
                field = field + actual_dt * advection
        else:
            theta = 0.5 if step_method == "crank_nicolson" else 1.0
            # Hex encoding retains the actual short step exactly; no unsafe dt rounding.
            key = (grid.bounds, grid.nx, grid.ny, kappa, boundary, step_method, actual_dt.hex())
            entry, hit, elapsed = _factor(key, operator, theta * actual_dt)
            factor_ms += elapsed
            factor_count += int(not hit)
            cache_hits += int(hit)
            rhs = field.ravel()
            if theta != 1:
                rhs = rhs + (1 - theta) * actual_dt * (operator @ rhs)
            if step_method == "imex_euler":
                rhs = rhs + actual_dt * advection.ravel()
            with entry.lock:
                solution = entry.lu.solve(rhs)
            residual = np.linalg.norm(entry.matrix @ solution - rhs) / max(float(np.linalg.norm(rhs)), 1.0)
            residual_max = max(residual_max, float(residual))
            field = solution.reshape((grid.ny, grid.nx))
        if not np.isfinite(field).all():
            raise FloatingPointError("Nonfinite result during numerical integration.")
        steps += 1
        min_step = min(min_step, actual_dt)
        max_step = max(max_step, actual_dt)
        key_dt = repr(actual_dt)
        step_histogram[key_dt] = step_histogram.get(key_dt, 0) + 1
        advance_ms += (perf_counter() - step_begin) * 1000

    for index, target in enumerate(times):
        # Compute time from an interval origin rather than repeatedly summing dt.
        origin = current_time
        segment_step = 0
        tolerance = 8 * np.finfo(float).eps * max(np.finfo(float).tiny, abs(float(target)))
        while float(target) - current_time > tolerance:
            actual_dt = float(min(chosen_dt, float(target) - current_time))
            if method == "crank_nicolson" and startup == "rannacher" and not startup_done:
                advance(actual_dt / 2, "backward_euler")
                advance(actual_dt / 2, "backward_euler")
                startup_done = True
            else:
                advance(actual_dt, method)
            segment_step += 1
            current_time = min(float(target), origin + segment_step * chosen_dt)
        current_time = float(target)
        output_begin = perf_counter()
        frames[index] = field
        masses.append(float(area * field.sum()))
        energies.append(float(area * np.sum((field - field.mean()) ** 2)))
        minima.append(float(field.min()))
        maxima.append(float(field.max()))
        outflows.append(float(cumulative_outflow))
        balances.append(float(masses[-1] - mass0 + cumulative_outflow))
        output_ms += (perf_counter() - output_begin) * 1000

    cache = factor_cache_info()
    total_ms = (perf_counter() - total_begin) * 1000
    diagnostics = {
        "schema_version": 2, "solve_ms": total_ms, "internal_steps": steps,
        "dt_max_s": chosen_dt, "requested_dt_s": dt,
        "actual_dt_min_s": float(min_step) if steps else 0.0, "actual_dt_max_s": max_step,
        "actual_dt_histogram": step_histogram, "cfl_limit_s": float(limit) if np.isfinite(limit) else None,
        "method": method, "backend": selected_backend, "requested_backend": backend,
        "boundary": boundary, "velocity": list(velocity), "startup": startup,
        "timestep_policy": "automatic" if dt is None else "requested_with_exact_output_alignment",
        "linear_solver": "scipy.sparse.linalg.splu" if implicit and kappa else "none",
        "linear_residual_max": residual_max, "linear_residual_normalization": "norm(Ax-b)/max(norm(b),1)",
        "initial_mass": mass0, "final_mass": masses[-1],
        "mass_scale": mass_scale, "mass_scale_definition": "max(abs(M0), integral(abs(c0)), 1)",
        "relative_mass_drift": float(np.max(np.abs(np.asarray(masses) - mass0)) / mass_scale),
        "mass_balance_error": balances[-1], "max_mass_balance_error": max(abs(x) for x in balances),
        "relative_mass_balance_error": max(abs(x) for x in balances) / mass_scale,
        "cumulative_outward_flux": cumulative_outflow,
        "initial_energy": energy0, "final_energy": energies[-1],
        "energy_max_increase": max(0.0, float(np.max(np.diff([energy0] + energies)))),
        "min_concentration": min(min0, min(minima)), "max_concentration": max(max0, max(maxima)),
        "mass_history": masses, "energy_history": energies, "min_history": minima, "max_history": maxima,
        "cumulative_outflow_history": outflows, "mass_balance_history": balances,
        "diagnostic_scope": "initial_and_output_frames", "output_times_s": times.tolist(),
        "factorization_count": factor_count, "factorization_cache_hits": cache_hits,
        "factorization_cache_entries": cache["entries"], "factorization_cache_bytes": cache["bytes"],
        "factorization_cache": cache, "output_bytes": frames.nbytes,
        "timings_ms": {"matrix": matrix_ms, "factorization": factor_ms,
                       "advance": max(0.0, advance_ms - factor_ms), "conversion": conversion_ms,
                       "output": output_ms, "total": total_ms},
        "dtype": "float64",
    }
    return frames, diagnostics
