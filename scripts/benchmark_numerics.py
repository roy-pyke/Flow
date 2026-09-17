"""Fair, reproducible NumPy/C++ FE timings with separate process RSS.

Run from the repository root: .venv/bin/python -m scripts.benchmark_numerics
The native backend must be installed. Each backend/case has a 120 second wall
budget and a 512 MiB measured peak-RSS budget. The 640² case declares 3 warmups
and 5 samples in advance; all other cases use 3 warmups and 10 samples.
No unavailable or timed-out point is reported as a successful measurement.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter

# Set before importing NumPy/SciPy; record the actual thread pools below.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for variable in THREAD_VARIABLES:
    os.environ[variable] = "1"

import numpy as np
from backend.app.diffusion import Grid, diffusion_step, gaussian_initial, stable_timestep
from backend.app.numerics import native

ROOT = Path(__file__).resolve().parents[1]


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def numpy_profiled(grid, initial, kappa, times, dt):
    """Original vectorized face-flux stencil, matching native output diagnostics.

    This preserves the V1 NumPy implementation (including its temporary flux
    arrays), not a slow Python per-cell replacement. Allocation and conversion
    are timed separately; kernel timing includes time stepping, output copies,
    step-size recording, mass/extrema/energy diagnostics on each output frame.
    """
    started = perf_counter()
    stamp = perf_counter()
    source = np.require(initial, dtype=np.float64, requirements=["C", "A"])
    times = np.require(times, dtype=np.float64, requirements=["C", "A"])
    conversion_ms = (perf_counter() - stamp) * 1000
    if source.shape != (grid.ny, grid.nx) or not np.isfinite(source).all():
        raise ValueError("Invalid initial field")
    if len(times) == 0 or not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid times")
    limit = stable_timestep(grid, kappa, 1.0)
    if dt <= 0 or dt > limit * (1 + 1e-12):
        raise ValueError("Invalid step")
    stamp = perf_counter()
    frames = np.empty((len(times), grid.ny, grid.nx))
    output_allocation_ms = (perf_counter() - stamp) * 1000
    stamp = perf_counter()
    field = source.copy()
    workspace_allocation_ms = (perf_counter() - stamp) * 1000
    stamp = perf_counter()
    area = grid.dx * grid.dy
    initial_mass = float(field.sum() * area)
    minimum, maximum = float(field.min()), float(field.max())
    drift, time, steps = 0.0, 0.0, []
    masses, minima, maxima, energies = [], [], [], []
    for index, target in enumerate(times):
        origin, segment_steps = time, 0
        tolerance = 8 * np.finfo(float).eps * max(np.finfo(float).tiny, abs(float(target)))
        while float(target) - time > tolerance:
            actual = min(dt, float(target - time))
            field = diffusion_step(field, grid, kappa, actual)
            segment_steps += 1
            time = min(float(target), origin + segment_steps * dt)
            steps.append(actual)
        time = float(target)
        frames[index] = field
        mass, low, high = float(field.sum() * area), float(field.min()), float(field.max())
        masses.append(mass); minima.append(low); maxima.append(high)
        energies.append(float(0.5 * np.square(field).sum() * area))
        minimum, maximum = min(minimum, low), max(maximum, high)
        drift = max(drift, abs(mass - initial_mass) / (abs(initial_mass) or 1))
    kernel_ms = (perf_counter() - stamp) * 1000
    return frames, {"conversion_ms": conversion_ms, "output_allocation_ms": output_allocation_ms,
                    "workspace_allocation_ms": workspace_allocation_ms, "kernel_ms": kernel_ms,
                    "solve_ms": (perf_counter() - started) * 1000, "internal_steps": len(steps),
                    "actual_step_sizes_s": steps, "dt_max_s": dt, "cfl_limit_s": limit,
                    "initial_mass": initial_mass, "final_mass": mass, "relative_mass_drift": drift,
                    "min_concentration": minimum, "max_concentration": maximum,
                    "mass_by_output": masses, "min_by_output": minima, "max_by_output": maxima,
                    "energy_by_output": energies, "output_bytes": frames.nbytes,
                    "workspace_bytes": 3 * source.nbytes + (grid.nx * (grid.ny - 1) + grid.ny * (grid.nx - 1)) * 8}


def exact_reflecting_gaussian(grid: Grid, kappa: float, time: float) -> np.ndarray:
    """Neumann mirror-image Gaussian (omitted tails far below double precision)."""
    variance = 250.0 ** 2 + 2 * kappa * time
    factors = []
    for coords in (grid.x, grid.y):
        factor = np.zeros_like(coords)
        for image in range(-3, 4):
            factor += np.exp(-np.square(coords - 2000 + 8000 * image) / (2 * variance))
            factor += np.exp(-np.square(coords + 2000 + 8000 * image) / (2 * variance))
        factors.append(factor)
    return 250.0 ** 2 / variance * np.outer(factors[1], factors[0])


def worker(case: dict, backend: str) -> dict:
    n, kappa = case["n"], 20.0
    grid = Grid((0, 0, 4000, 4000), n, n)
    initial = gaussian_initial(grid, (2000, 2000))
    if case.get("layout") == "fortran_float32":
        initial = np.asfortranarray(initial.astype(np.float32))
    times = np.linspace(0, case["duration_s"], 61) if case["output"] == "61_frames" else np.array([float(case["duration_s"])])
    dt = stable_timestep(grid, kappa)
    solver = native.solve_native if backend == "cpp" else numpy_profiled
    rss_before = peak_rss_bytes()
    for _ in range(case["warmups"]):
        frames, diagnostics = solver(grid, initial, kappa, times, dt)
        del frames
    records = []
    for index in range(case["samples"]):
        start = perf_counter()
        frames, diagnostics = solver(grid, initial, kappa, times, dt)
        total_ms = (perf_counter() - start) * 1000
        records.append({"sample": index, "total_ms": total_ms,
                        **{key: float(diagnostics[key]) for key in ("conversion_ms", "kernel_ms", "output_allocation_ms", "workspace_allocation_ms")}})
        if peak_rss_bytes() > case["rss_budget_bytes"]:
            return {"status": "over_memory_budget", "raw_samples": records, "peak_rss_bytes": peak_rss_bytes()}
        # Release each run's output before the next run (including full history).
        if index < case["samples"] - 1:
            del frames
    measured_rss = peak_rss_bytes()
    reference = exact_reflecting_gaussian(grid, kappa, case["duration_s"])
    error = frames[-1] - reference
    from backend.app.diffusion import solve_initial
    baseline, baseline_diag = solve_initial(grid, initial, kappa, times, dt)
    parity_error = float(np.max(np.abs(frames - baseline)))
    if not np.allclose(frames, baseline, rtol=1e-11, atol=1e-12) or diagnostics["internal_steps"] != baseline_diag["internal_steps"]:
        raise RuntimeError("Benchmark work did not match the original NumPy FE solution.")
    return {"status": "complete", "raw_samples": records,
            "median_total_ms": float(np.median([r["total_ms"] for r in records])),
            "p90_total_ms": float(np.percentile([r["total_ms"] for r in records], 90)),
            "median_kernel_ms": float(np.median([r["kernel_ms"] for r in records])),
            "peak_rss_bytes": measured_rss, "peak_rss_before_solve_bytes": rss_before,
            "peak_rss_increment_bytes": max(0, measured_rss - rss_before),
            "output_bytes": int(diagnostics["output_bytes"]), "estimated_peak_workspace_bytes": int(diagnostics["workspace_bytes"]),
            "internal_steps": diagnostics["internal_steps"], "dt_max_s": dt,
            "actual_step_sizes_s": diagnostics["actual_step_sizes_s"],
            "parity_max_abs_error": parity_error,
            "gaussian_image_Linf_error": float(np.max(np.abs(error))),
            "gaussian_image_L2_error": float(np.sqrt(np.sum(error**2) * grid.dx * grid.dy)),
            "initial_mass": diagnostics["initial_mass"], "final_mass": diagnostics["final_mass"],
            "relative_mass_drift": diagnostics["relative_mass_drift"],
            "minimum": diagnostics["min_concentration"], "maximum": diagnostics["max_concentration"],
            "method": "explicit_euler", "backend": backend, "kappa": kappa,
            "wind_m_per_s": [0, 0], "boundary": "zero_flux", "dtype": "float64",
            "matrix_assembly_ms": 0.0, "factorization_ms": 0.0,
            "native_build": native.status()["build_info"] if backend == "cpp" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=["numpy", "cpp"])
    parser.add_argument("--output-dir", default="reports/numerics")
    args = parser.parse_args()
    if args.worker_case:
        print(json.dumps(worker(json.loads(args.worker_case), args.backend)))
        return
    if not native.status()["available"]:
        raise SystemExit(native.status()["reason"])
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    base = {"warmups": 3, "samples": 10, "duration_s": 1800, "layout": "contiguous_float64",
            "wall_budget_s": 120, "rss_budget_bytes": 512 * 1024 ** 2}
    cases = [{**base, "name": f"final_{n}", "n": n, "output": "final_only", "samples": 5 if n == 640 else 10} for n in (80, 160, 320, 640)]
    cases += [{**base, "name": "default_160_61", "n": 160, "output": "61_frames"},
              {**base, "name": "conversion_160_61", "n": 160, "output": "61_frames", "layout": "fortran_float32"},
              {**base, "name": "zero_duration_overhead", "n": 80, "output": "final_only", "duration_s": 0}]
    print("Declared budgets and repeat counts:", json.dumps(cases), flush=True)
    configuration = io.StringIO()
    from contextlib import redirect_stdout
    with redirect_stdout(configuration):
        np.show_config()
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "schema_version": 1,
              "purpose": "Same numerical method, vectorized NumPy FE versus native C++ FE; no algorithm-speedup claim.",
              "machine": {"system": platform.system(), "release": platform.release(), "architecture": platform.machine(),
                          "python": platform.python_version(), "numpy": np.__version__, "processor": platform.processor(),
                          "cpu_count": os.cpu_count(), "numpy_configuration": configuration.getvalue()},
              "threads": {name: os.environ[name] for name in THREAD_VARIABLES},
              "source_sha256": {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in ("scripts/benchmark_numerics.py", "backend/app/diffusion.py", "backend/app/numerics/native.py")},
              "native_build": native.version(), "cases_declared_before_measurement": cases,
              "measurement_notes": [
                  "Each backend/case runs in a fresh subprocess; repeated calls are warm within that process. Import time is excluded from call timings but included in absolute peak RSS.",
                  "Kernel timing includes output copies, internal step-size recording and per-output mass/extrema/energy diagnostics; C++ uses compensated diagnostic sums and NumPy uses vectorized reductions.",
                  "Output allocation measures allocator calls, not physical page first-touch; first-touch output writes are included in kernel_ms.",
                  "C++ total call time includes source/build provenance verification and explicit input conversion. Both backends preserve inputs and land at identical output times.",
                  "NumPy reuses the existing vectorized diffusion_step; both backends use the same requested output history and numerical work. NumPy per-step temporary allocation remains included.",
                  "Peak RSS is sampled from independent process high-water marks; incremental high-water mark is not an exact allocation accounting. Post-timing analytic/parity reference computations are excluded from reported RSS.",
                  "640² declares five measured samples before execution to fit the per-backend 120s budget; this point has lower statistical precision.",
                  "Analytic error uses a reflecting Gaussian mirror-image solution. Its initial difference from the sampled central Gaussian is below 1e-13; float32 conversion cases also include input quantization.",
                  "Benchmark timings are local observations, not general speed guarantees. Run without other heavy workloads for comparison."],
              "results": [], "comparisons": []}
    for case in cases:
        for backend in ("numpy", "cpp"):
            print(f"Measuring {case['name']} / {backend} ...", flush=True)
            command = [sys.executable, "-m", "scripts.benchmark_numerics", "--worker-case", json.dumps(case), "--backend", backend]
            try:
                process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=case["wall_budget_s"], check=False)
                result = json.loads(process.stdout) if process.returncode == 0 else {"status": "failed", "error": process.stderr[-4000:]}
            except subprocess.TimeoutExpired:
                result = {"status": "time_budget_exceeded", "budget_s": case["wall_budget_s"]}
            report["results"].append({"case": case, "backend": backend, **result})
            (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    for case in cases:
        matching = {r["backend"]: r for r in report["results"] if r["case"]["name"] == case["name"]}
        if all(matching[b]["status"] == "complete" for b in ("numpy", "cpp")):
            report["comparisons"].append({"case": case["name"],
                                          "numpy_median_total_ms": matching["numpy"]["median_total_ms"],
                                          "cpp_median_total_ms": matching["cpp"]["median_total_ms"],
                                          "total_speedup_numpy_over_cpp": matching["numpy"]["median_total_ms"] / matching["cpp"]["median_total_ms"]})
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "benchmark.csv").open("w", newline="") as stream:
        columns = ["case", "n", "backend", "status", "sample", "total_ms", "kernel_ms", "conversion_ms", "output_allocation_ms", "workspace_allocation_ms", "peak_rss_bytes", "output_bytes", "internal_steps", "dt_max_s", "parity_max_abs_error", "gaussian_image_Linf_error", "relative_mass_drift"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in report["results"]:
            for sample in result.get("raw_samples", [{}]):
                writer.writerow({"case": result["case"]["name"], "n": result["case"]["n"], "backend": result["backend"],
                                 "status": result["status"], **sample,
                                 **{key: result.get(key) for key in columns[10:]}})
    print(json.dumps(report["comparisons"], indent=2))
    if any(row["status"] != "complete" for row in report["results"]):
        raise SystemExit("Some benchmarks were incomplete; see recorded statuses.")


if __name__ == "__main__":
    main()
