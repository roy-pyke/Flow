"""Reproducible numerical experiments used by tests and public report scripts.

L2 below means sqrt(sum(error**2)*cell_area/domain_area), equal to RMS on
these uniform grids. Linf is max(abs(error)); no result is renormalized.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply

from ..diffusion import Grid
from .operators import advection_matrix, diffusion_matrix
from .solver import solve


def errors(actual: np.ndarray, expected: np.ndarray) -> dict:
    difference = actual - expected
    return {"l2": float(np.sqrt(np.mean(difference ** 2))), "linf": float(np.max(np.abs(difference)))}


def _orders(rows: list[dict], spacing: str) -> list[dict]:
    for previous, current in zip(rows, rows[1:]):
        factor = previous[spacing] / current[spacing]
        for norm in ["l2", "linf"]:
            current[f"order_{norm}"] = float(np.log(previous[norm] / current[norm]) / np.log(factor))
    return rows


def _cosine(grid: Grid, kappa: float, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lx, ly = grid.bounds[2] - grid.bounds[0], grid.bounds[3] - grid.bounds[1]
    xx, yy = np.meshgrid(grid.x - grid.bounds[0], grid.y - grid.bounds[1])
    mode = np.cos(np.pi * xx / lx) * np.cos(np.pi * yy / ly)
    lam_h = -4 * kappa * (np.sin(np.pi * grid.dx / (2 * lx)) ** 2 / grid.dx ** 2
                          + np.sin(np.pi * grid.dy / (2 * ly)) ** 2 / grid.dy ** 2)
    lam = -kappa * ((np.pi / lx) ** 2 + (np.pi / ly) ** 2)
    return (1 + .5 * mode, 1 + .5 * np.exp(lam_h * time) * mode,
            1 + .5 * np.exp(lam * time) * mode, float(lam_h))


def temporal_convergence() -> dict:
    grid = Grid((0, 0, 2, 1.5), 20, 16)
    kappa, duration = .07, .5
    initial, semidiscrete, _, eigenvalue = _cosine(grid, kappa, duration)
    methods = {}
    for method, backend in [("explicit_euler", "numpy"), ("backward_euler", "scipy"),
                            ("crank_nicolson", "scipy")]:
        rows = []
        for count in [32, 64, 128, 256]:
            dt = duration / count
            frames, diagnostics = solve(grid, initial, kappa, [duration], method=method, backend=backend, dt=dt)
            rows.append({"dt_s": dt, "internal_steps": diagnostics["internal_steps"],
                         "linear_residual": diagnostics["linear_residual_max"], **errors(frames[-1], semidiscrete)})
        methods[method] = _orders(rows, "dt_s")
    return {"grid": grid.to_dict(), "kappa": kappa, "duration_s": duration,
            "reference": "exact discrete Neumann cosine eigenmode; spatial error removed",
            "lambda_h": eigenvalue, "methods": methods,
            "asymptotic_intervals": "all three adjacent refinements, declared before measurement",
            "order_windows": {"explicit_euler": [0.8, 1.2], "backward_euler": [0.8, 1.2], "crank_nicolson": [1.8, 2.2]}}


def spatial_convergence(quick: bool = False) -> dict:
    kappa = .05
    semi_rows = []
    for n in [32, 64, 128, 256]:
        grid = Grid((0, 0, 1, 1), n, n)
        _, semi, continuous, _ = _cosine(grid, kappa, .2)
        semi_rows.append({"n": n, "h_m": grid.h, **errors(semi, continuous)})
    production = {}
    # Short physical time keeps the strictly time-resolved LU study affordable.
    # The measured temporal error must be <=3% of spatial error at every level.
    duration = .001
    sizes = [16, 32, 64] if quick else [32, 64, 128]
    for method, backend in [("explicit_euler", "numpy"), ("backward_euler", "scipy"),
                            ("crank_nicolson", "scipy")]:
        rows = []
        for n in sizes:
            grid = Grid((0, 0, 1, 1), n, n)
            initial, semi, continuous, _ = _cosine(grid, kappa, duration)
            steps = int(np.ceil(duration / (.002 * grid.h ** 2 / kappa)))
            dt = duration / steps
            frames, d = solve(grid, initial, kappa, [duration], method=method, backend=backend, dt=dt)
            time_error = errors(frames[-1], semi)["l2"]
            space_error = errors(semi, continuous)["l2"]
            rows.append({"n": n, "h_m": grid.h, "dt_s": dt,
                         "temporal_l2": time_error, "semidiscrete_l2": space_error,
                         "temporal_to_spatial_ratio": time_error / space_error,
                         "internal_steps": d["internal_steps"], **errors(frames[-1], continuous)})
        production[method] = _orders(rows, "h_m")
    return {"kappa": kappa, "semidiscrete_duration_s": .2, "semidiscrete": _orders(semi_rows, "h_m"),
            "production_duration_s": duration, "production": production,
            "time_error_control": "measured temporal L2 / semidiscrete spatial L2 <= 0.03 at every level",
            "asymptotic_intervals": "all adjacent refinements", "order_window": [1.8, 2.2]}


def positivity_experiment() -> dict:
    grid = Grid((0, 0, 1, 1), 20, 20)
    initial = np.zeros((20, 20))
    initial[10, 10] = 1
    rows = []
    final_fields = {}
    for method, backend, dt, startup in [("explicit_euler", "numpy", .0005, "none"),
                                         ("backward_euler", "scipy", .1, "none"),
                                         ("crank_nicolson", "scipy", .1, "none"),
                                         ("crank_nicolson", "scipy", .1, "rannacher")]:
        frames, d = solve(grid, initial, 1, [0, .1, .2, .3], method=method, backend=backend, dt=dt, startup=startup)
        key = method + ("_rannacher" if startup == "rannacher" else "")
        rows.append({"label": key, "dt_s": dt, "minimum": d["min_concentration"],
                     "mass_drift": d["relative_mass_drift"], "energy_history": d["energy_history"],
                     "min_history": d["min_history"], "max_history": d["max_history"],
                     "linear_residual": d["linear_residual_max"], "steps": d["internal_steps"]})
        final_fields[key] = frames[1].tolist()
    z = np.linspace(-20, 0, 201)
    return {"grid": grid.to_dict(), "output_times_s": [0, .1, .2, .3], "cases": rows,
            "first_output_fields": final_fields,
            "amplification": {"z": z.tolist(), "explicit_euler": (1 + z).tolist(),
                              "backward_euler": (1 / (1 - z)).tolist(),
                              "crank_nicolson": ((1 + z / 2) / (1 - z / 2)).tolist()},
            "interpretation": "CN is linearly stable but its stiff mode amplification is negative; Rannacher dampens initial stiff modes, not a universal positivity guarantee."}


def transport_validation(quick: bool = False) -> dict:
    grid = Grid((0, 0, 2, 3), 12, 10)
    kappa, duration, wind = .03, .3, (.3, -.2)
    xx, yy = np.meshgrid(grid.x, grid.y)
    initial = 1 + .3 * np.cos(np.pi * xx) * np.cos(2 * np.pi * yy / 3)
    operator = diffusion_matrix(grid, kappa, "periodic") + advection_matrix(grid, wind, "periodic")
    reference = expm_multiply(duration * operator, initial.ravel()).reshape(initial.shape)
    temporal = {}
    for method, backend in [("advection_explicit", "numpy"), ("imex_euler", "numpy_scipy")]:
        rows = []
        for n in [16, 32, 64, 128]:
            frames, d = solve(grid, initial, kappa, [duration], method=method, backend=backend,
                              dt=duration / n, boundary="periodic", velocity=wind)
            rows.append({"dt_s": duration / n, "mass_balance_relative": d["relative_mass_balance_error"],
                         **errors(frames[-1], reference)})
        temporal[method] = _orders(rows, "dt_s")
    # Pure advection isolates first-order upwind spatial dissipation; positive,
    # negative and diagonal winds are separately checked in the scenario table.
    spatial = []
    spatial_diffusion = []
    transport_wind = (.7, -.4)
    transport_t = .2
    for n in ([16, 32, 64] if quick else [16, 32, 64, 128]):
        mesh = Grid((0, 0, 2, 3), n, n)
        xx, yy = np.meshgrid(mesh.x, mesh.y)
        start = 1 + .3 * np.cos(np.pi * xx) * np.cos(2 * np.pi * yy / 3)
        analytic = 1 + .3 * np.cos(np.pi * (xx - transport_wind[0] * transport_t)) * np.cos(2 * np.pi * (yy - transport_wind[1] * transport_t) / 3)
        # O(h^2) time steps make temporal error smaller than the O(h) upwind term.
        dt = .2 * mesh.h ** 2
        frames, d = solve(mesh, start, 0, [transport_t], method="advection_explicit", dt=dt,
                          boundary="periodic", velocity=transport_wind)
        spatial.append({"n": n, "h_m": mesh.h, "dt_s": dt, "internal_steps": d["internal_steps"], **errors(frames[-1], analytic)})
        diffusive_kappa = .03
        damped = 1 + (analytic - 1) * np.exp(-diffusive_kappa * (np.pi ** 2 + (2 * np.pi / 3) ** 2) * transport_t)
        mixed, dm = solve(mesh, start, diffusive_kappa, [transport_t], method="advection_explicit", dt=dt,
                          boundary="periodic", velocity=transport_wind)
        spatial_diffusion.append({"n": n, "h_m": mesh.h, "dt_s": dt, "internal_steps": dm["internal_steps"],
                                  "kappa": diffusive_kappa, **errors(mixed[-1], damped)})
    scenarios = []
    test_grid = Grid((0, 0, 3, 2), 12, 7)
    xx, yy = np.meshgrid(test_grid.x, test_grid.y)
    pulse = np.exp(-((xx - 1.5) ** 2 + (yy - 1) ** 2) / .3)
    for boundary in ["periodic", "open"]:
        for case_wind in [(1., 0.), (-1., 0.), (.7, -.4), (-.7, .4)]:
            for case_kappa in [0., .05]:
                for method, backend in [("advection_explicit", "numpy"), ("imex_euler", "numpy_scipy")]:
                    _, d = solve(test_grid, pulse, case_kappa, [0, .125, .6], method=method, backend=backend,
                                 boundary=boundary, velocity=case_wind, dt=.0125)
                    scenarios.append({"boundary": boundary, "velocity": list(case_wind), "kappa": case_kappa,
                                      "method": method, "initial_mass": d["initial_mass"], "final_mass": d["final_mass"],
                                      "cumulative_outward_flux": d["cumulative_outward_flux"],
                                      "mass_balance_absolute": d["max_mass_balance_error"],
                                      "mass_balance_relative": d["relative_mass_balance_error"],
                                      "minimum": d["min_concentration"],
                                      "peclet_x": abs(case_wind[0]) * test_grid.dx / case_kappa if case_kappa else None,
                                      "peclet_y": abs(case_wind[1]) * test_grid.dy / case_kappa if case_kappa else None,
                                      "regime": "pure_advection" if not case_kappa else "advection_diffusion"})
    return {"temporal_reference": "scipy.sparse.linalg.expm_multiply of independently assembled periodic upwind+diffusion matrix",
            "temporal": temporal, "spatial": _orders(spatial, "h_m"),
            "spatial_diffusion": _orders(spatial_diffusion, "h_m"), "scenarios": scenarios,
            "spatial_problem": {"kappa": 0, "velocity": list(transport_wind), "duration_s": transport_t,
                                "reference": "continuous periodic translated Fourier mode", "order_window": [0.8, 1.2]},
            "spatial_diffusion_problem": {"kappa": .03, "velocity": list(transport_wind), "duration_s": transport_t,
                                          "reference": "continuous periodic translated AND diffusively damped Fourier mode",
                                          "order_window": [0.8, 1.2]},
            "boundary_rule": "inflow prescribed total flux u.n*c_in=0; outflow interior upwind and no exterior diffusive flux",
            "mass_balance": "M(t)-M(0)+sum(actual_dt*outward_flux_at_explicit_time_layer)=0"}


def run_validation(quick: bool = False) -> dict:
    temporal = temporal_convergence()
    spatial = spatial_convergence(quick)
    positivity = positivity_experiment()
    transport = transport_validation(quick)
    order_checks = []
    for method, rows in temporal["methods"].items():
        lo, hi = temporal["order_windows"][method]
        order_checks.extend(lo <= row[f"order_{norm}"] <= hi for row in rows[1:] for norm in ["l2", "linf"])
    order_checks.extend(1.8 <= row[f"order_{norm}"] <= 2.2 for row in spatial["semidiscrete"][1:] for norm in ["l2", "linf"])
    for rows in spatial["production"].values():
        order_checks.extend(1.8 <= row[f"order_{norm}"] <= 2.2 for row in rows[1:] for norm in ["l2", "linf"])
        order_checks.extend(row["temporal_to_spatial_ratio"] <= .03 for row in rows)
    for rows in transport["temporal"].values():
        order_checks.extend(.8 <= row[f"order_{norm}"] <= 1.2 for row in rows[1:] for norm in ["l2", "linf"])
    for rows in [transport["spatial"], transport["spatial_diffusion"]]:
        order_checks.extend(.8 <= row[f"order_{norm}"] <= 1.2 for row in rows[1:] for norm in ["l2", "linf"])
    balance = max(row["mass_balance_relative"] for row in transport["scenarios"])
    cn = next(row for row in positivity["cases"] if row["label"] == "crank_nicolson")
    return {"schema_version": 2, "quick": quick, "error_definition": "L2=sqrt(cell_area/domain_area*sum(error^2))=RMS; Linf=max(abs(error))",
            "temporal": temporal, "spatial": spatial, "stability": positivity, "transport": transport,
            "summary": {"all_order_checks_pass": bool(all(order_checks)), "order_checks": len(order_checks),
                        "transport_max_relative_balance_error": balance, "transport_balance_pass": balance <= 1e-10,
                        "cn_negative_example_observed": cn["minimum"] < 0}}
