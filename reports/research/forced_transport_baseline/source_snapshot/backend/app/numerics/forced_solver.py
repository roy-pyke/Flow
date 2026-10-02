"""Time-dependent prescribed fields with the same finite-volume mass ledger.

This separate dispatcher leaves legacy, unforced array arithmetic unchanged.
Only diffusion is implicit; time-dependent transport uses first-order upwind.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from ..diffusion import Grid
from .coefficients import DiffusionCoefficients, prepare_diffusivity
from .forcing import PrescribedFields, face_advection_rate
from .operators import diffusion_matrix, diffusion_rate


def _contract(grid, method, velocity, boundary, forcing):
    from .solver import METHOD_BACKENDS
    if not isinstance(forcing, PrescribedFields):
        raise ValueError("forcing must be a validated PrescribedFields object.")
    if (grid.bounds, grid.nx, grid.ny, boundary) != (
            forcing.grid.bounds, forcing.grid.nx, forcing.grid.ny, forcing.boundary):
        raise ValueError("Prescribed fields must match the solver grid and boundary.")
    wind = np.asarray(velocity)
    if wind.shape != (2,) or wind.dtype.kind not in "fiu" or not np.isfinite(wind).all() or np.any(wind):
        raise ValueError("With prescribed fields, the legacy velocity must be the zero pair.")
    if method not in METHOD_BACKENDS:
        raise ValueError("Unknown numerical method.")
    if method in {"explicit_euler", "backward_euler", "crank_nicolson"}:
        if forcing.has_velocity or forcing.has_inflow or boundary == "open":
            raise ValueError("Pure diffusion requires zero prescribed wind/inflow and closed or periodic boundaries.")


def prescribed_timestep_limit(grid, kappa, method, velocity, boundary, forcing):
    """A global bound valid at every time of the piecewise-linear input.

    Each positive-part outgoing face rate is convex in time on a knot interval,
    so its sum, and the cell maximum, are bounded by the stored endpoints.
    """
    from .solver import timestep_limit
    _contract(grid, method, velocity, boundary, forcing)
    kappa = prepare_diffusivity(grid, kappa, boundary)
    if method in {"explicit_euler", "backward_euler", "crank_nicolson"}:
        return timestep_limit(grid, kappa, method, (0.0, 0.0), boundary)
    if isinstance(kappa, DiffusionCoefficients):
        raise ValueError("Prescribed transport currently requires scalar diffusivity.")
    rate = forcing.max_outgoing_rate
    if method == "advection_explicit":
        rate += 2 * kappa * (grid.dx ** -2 + grid.dy ** -2)
    if not np.isfinite(rate):
        raise ValueError("Combined outgoing rate must be finite.")
    return 0.9 / rate if rate else np.inf


def solve_prescribed(grid: Grid, initial, kappa, output_times_s, *, method, backend,
                     dt, boundary, velocity, startup, forcing: PrescribedFields):
    from .solver import _factor, factor_cache_info, resolve_backend
    begin = perf_counter()
    _contract(grid, method, velocity, boundary, forcing)
    raw = np.asarray(initial)
    if raw.shape != (grid.ny, grid.nx) or raw.dtype.kind not in "fiu":
        raise ValueError("Initial field must be a real numeric array of shape (ny,nx).")
    with np.errstate(over="ignore", invalid="ignore"):
        field = np.array(raw, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(field).all():
        raise ValueError("Initial field must be finite.")
    raw_times = np.asarray(output_times_s)
    if raw_times.dtype.kind not in "fiu":
        raise ValueError("Output times must be real numeric values.")
    times = np.asarray(raw_times, dtype=np.float64)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or times[0] < 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("Output times must be finite, nonnegative and strictly increasing.")
    forcing.validate_interval(0.0, float(times[-1]))
    kappa = prepare_diffusivity(grid, kappa, boundary)
    variable = isinstance(kappa, DiffusionCoefficients)
    if backend == "cpp":
        raise ValueError("The C++ kernel does not support prescribed fields or sources.")
    selected_backend = resolve_backend(method, "numpy" if backend == "auto" and method == "explicit_euler"
                                       else backend, boundary, variable_diffusivity=variable)
    if startup not in {"none", "rannacher"} or startup != "none" and method != "crank_nicolson":
        raise ValueError("Rannacher startup requires Crank-Nicolson.")
    limit = prescribed_timestep_limit(grid, kappa, method, velocity, boundary, forcing)
    chosen_dt = (limit * (0.9 if method == "explicit_euler" else 1.0) if np.isfinite(limit)
                 else min(30.0, float(times[-1])) if times[-1] > 0 else 1.0) if dt is None else float(dt)
    if not np.isfinite(chosen_dt) or chosen_dt <= 0 or chosen_dt > limit * (1 + 1e-12):
        raise ValueError("Time step must be positive, finite, and satisfy the method's CFL bound.")
    conversion_ms = (perf_counter() - begin) * 1000
    diffusion_active = variable or bool(kappa)
    implicit = method in {"backward_euler", "crank_nicolson", "imex_euler"}
    coefficient_identity = kappa.cache_key if variable else kappa
    matrix_begin = perf_counter()
    operator = diffusion_matrix(grid, kappa, boundary) if implicit and diffusion_active else None
    matrix_ms = (perf_counter() - matrix_begin) * 1000
    coefficient_metadata = kappa.metadata() if variable else {
        "kind": "constant_scalar", "value_m2_s": kappa, "face_policy": "legacy_constant", "unit": "m2/s"}
    area = grid.dx * grid.dy
    mass0 = float(field.sum() * area)
    mass_scale = max(abs(mass0), float(np.abs(field).sum() * area), 1.0)
    energy0 = float(area * np.sum((field - field.mean()) ** 2))
    min0, max0 = float(field.min()), float(field.max())
    frames = np.empty((len(times), grid.ny, grid.nx), dtype=np.float64)
    history = {key: [] for key in ("mass", "energy", "min", "max", "source", "outflow", "outward", "inward", "balance")}
    source_mass = outward_mass = inward_mass = residual_max = step_balance_max = 0.0
    steps = factor_count = cache_hits = 0
    factor_ms = advance_ms = output_ms = 0.0
    min_step, max_step = np.inf, 0.0
    step_histogram = {}
    transport = method in {"advection_explicit", "imex_euler"}

    def advance(t0, t1, step_method):
        nonlocal field, source_mass, outward_mass, inward_mass, residual_max, step_balance_max
        nonlocal steps, factor_count, cache_hits, factor_ms, advance_ms, min_step, max_step
        tick = perf_counter()
        step = float(t1 - t0)
        if step <= 0:
            raise ValueError("Time step cannot advance the floating-point clock.")
        left = forcing.at(t0)
        if step_method == "backward_euler":
            source = forcing.at(t1).source
        elif step_method == "crank_nicolson":
            source = 0.5 * left.source + 0.5 * forcing.at(t1).source
        else:
            source = left.source
        rate = source
        if transport:
            adv, ledger = face_advection_rate(field, grid, left, boundary)
            rate = rate + adv
            outward_mass += step * ledger["outward"]
            inward_mass += step * ledger["inward"]
        source_mass += step * float(area * source.sum())
        if step_method in {"explicit_euler", "advection_explicit"}:
            field = field + step * (rate + diffusion_rate(field, grid, kappa, boundary))
        elif operator is None:
            field = field + step * rate
        else:
            theta = 0.5 if step_method == "crank_nicolson" else 1.0
            key = (grid.bounds, grid.nx, grid.ny, coefficient_identity, boundary, step_method, step.hex())
            entry, hit, elapsed = _factor(key, operator, theta * step)
            factor_count += int(not hit)
            cache_hits += int(hit)
            factor_ms += elapsed
            rhs = field.ravel()
            if theta != 1:
                rhs = rhs + (1 - theta) * step * (operator @ rhs)
            rhs = rhs + step * rate.ravel()
            with entry.lock:
                solution = entry.lu.solve(rhs)
            residual = np.linalg.norm(entry.matrix @ solution - rhs) / max(float(np.linalg.norm(rhs)), 1.0)
            residual_max = max(residual_max, float(residual))
            field = solution.reshape((grid.ny, grid.nx))
        if not np.isfinite(field).all() or not np.isfinite([source_mass, outward_mass, inward_mass]).all():
            raise FloatingPointError("Nonfinite field or mass ledger during prescribed integration.")
        balance = float(area * field.sum() - mass0 + outward_mass - inward_mass - source_mass)
        step_balance_max = max(step_balance_max, abs(balance))
        steps += 1
        min_step, max_step = min(min_step, step), max(max_step, step)
        key = repr(step)
        step_histogram[key] = step_histogram.get(key, 0) + 1
        advance_ms += (perf_counter() - tick) * 1000

    current = 0.0
    startup_done = False
    for index, target in enumerate(times):
        # Split at input knots, even when no output was requested there. Within
        # each segment reconstruct time from its origin to avoid dt drift.
        ends = list(forcing.times_s[(forcing.times_s > current) & (forcing.times_s < target)]) + [float(target)]
        for end in ends:
            origin, segment_step = current, 0
            while current < end:
                next_time = min(end, origin + (segment_step + 1) * chosen_dt)
                if next_time <= current:
                    raise ValueError("Time step cannot advance the floating-point clock.")
                if method == "crank_nicolson" and startup == "rannacher" and not startup_done:
                    middle = current + (next_time - current) / 2
                    advance(current, middle, "backward_euler")
                    advance(middle, next_time, "backward_euler")
                    startup_done = True
                else:
                    advance(current, next_time, method)
                current = next_time
                segment_step += 1
        tick = perf_counter()
        frames[index] = field
        mass = float(area * field.sum())
        for key, value in {"mass": mass, "energy": float(area * np.sum((field-field.mean())**2)),
                           "min": float(field.min()), "max": float(field.max()), "source": source_mass,
                           "outflow": outward_mass-inward_mass, "outward": outward_mass, "inward": inward_mass,
                           "balance": mass-mass0+outward_mass-inward_mass-source_mass}.items():
            history[key].append(value)
        output_ms += (perf_counter() - tick) * 1000
    cache = factor_cache_info()
    total_ms = (perf_counter() - begin) * 1000
    max_balance = max(abs(value) for value in history["balance"])
    diagnostics = {
        "schema_version": 3, "solve_ms": total_ms, "internal_steps": steps,
        "method": method, "backend": selected_backend, "requested_backend": backend,
        "backend_selection": "prescribed_fields_capability", "boundary": boundary,
        "velocity": list(velocity), "startup": startup, "forcing": forcing.metadata(),
        "diffusivity": coefficient_metadata, "dtype": "float64", "output_bytes": frames.nbytes,
        "dt_max_s": chosen_dt, "requested_dt_s": dt,
        "actual_dt_min_s": float(min_step) if steps else 0.0, "actual_dt_max_s": max_step,
        "actual_dt_histogram": step_histogram,
        "cfl_limit_s": float(limit) if np.isfinite(limit) else None,
        "timestep_policy": "global_knot_envelope_cfl; exact_output_and_forcing_knot_alignment",
        "source_time_rule": {"explicit_euler": "left_endpoint", "advection_explicit": "left_endpoint",
                             "imex_euler": "left_endpoint", "backward_euler": "right_endpoint",
                             "crank_nicolson": "endpoint_trapezoid; Rannacher uses right endpoint of each half step"}[method],
        "transport_time_rule": "left_endpoint" if transport else "none",
        "mass_ledger_definition": "M-M0+outward-inward-injected; source uses actual method quadrature",
        "mass_scale": mass_scale, "mass_scale_definition": "max(abs(M0), integral(abs(c0)), 1)",
        "initial_mass": mass0, "final_mass": history["mass"][-1],
        "relative_mass_drift": max(abs(value-mass0) for value in history["mass"])/mass_scale,
        "cumulative_source_mass": source_mass, "cumulative_outward_flux": outward_mass-inward_mass,
        "cumulative_boundary_outward_mass": outward_mass, "cumulative_boundary_inward_mass": inward_mass,
        "mass_balance_error": history["balance"][-1], "max_mass_balance_error": max_balance,
        "relative_mass_balance_error": max_balance/mass_scale,
        "max_internal_mass_balance_error": step_balance_max,
        "initial_energy": energy0, "final_energy": history["energy"][-1],
        "energy_max_increase": max(0.0, float(np.max(np.diff([energy0]+history["energy"])))),
        "min_concentration": min(min0, min(history["min"])), "max_concentration": max(max0, max(history["max"])),
        "mass_history": history["mass"], "energy_history": history["energy"],
        "min_history": history["min"], "max_history": history["max"],
        "mass_balance_history": history["balance"], "cumulative_source_history": history["source"],
        "cumulative_outflow_history": history["outflow"],
        "cumulative_boundary_outward_history": history["outward"], "cumulative_boundary_inward_history": history["inward"],
        "diagnostic_scope": "initial_and_output_frames; mass_balance_also_every_internal_step",
        "invariant_applicability": {
            "mass_conservation": "only when net boundary flux and integrated source vanish",
            "energy_nonincrease": "not asserted with sources, open boundaries or general divergent wind",
            "positivity": "nonnegative initial/source/inflow with FE CFL or BE/IMEX; not unconditional for CN",
            "maximum_principle": "not asserted for compressive winds or positive sources"},
        "output_times_s": times.tolist(), "linear_solver": "scipy.sparse.linalg.splu" if operator is not None else "none",
        "linear_residual_max": residual_max, "linear_residual_normalization": "norm(Ax-b)/max(norm(b),1)",
        "factorization_count": factor_count, "factorization_cache_hits": cache_hits,
        "factorization_cache_entries": cache["entries"], "factorization_cache_bytes": cache["bytes"],
        "factorization_cache": cache,
        "timings_ms": {"matrix": matrix_ms, "factorization": factor_ms, "advance": max(0.0, advance_ms-factor_ms),
                       "conversion": conversion_ms, "output": output_ms, "total": total_ms},
    }
    return frames, diagnostics
