"""Measure prepared variable diffusion apply against CSC on identical arrays.

Raw setup/apply repetitions, tracked allocations and byte accounting are saved.
This measures a spatial operator only, not a PDE solve or end-to-end application.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from time import perf_counter_ns
import tracemalloc

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy

from backend.app.diffusion import Grid
from backend.app.numerics import diffusion_matrix, diffusion_rate
from backend.app.numerics.coefficients import prepare_diffusivity


def measure(nx=256, ny=160, repeats=21, boundary="periodic"):
    if (not isinstance(nx, int) or not isinstance(ny, int) or min(nx, ny) < 2
            or nx*ny > 1_000_000 or not isinstance(repeats, int) or not 3 <= repeats <= 101):
        raise ValueError("Require 2+ cells per axis, at most 1,000,000 cells and 3..101 repeats.")
    if boundary not in {"zero_flux", "periodic"}:
        raise ValueError("Unsupported variable diffusion boundary.")
    grid = Grid((0, 0, 4, 3), nx, ny)
    rng = np.random.default_rng(73)
    values = rng.uniform(.05, .2, (ny, nx))
    field = rng.normal(size=(ny, nx))
    setup = {"prepare": [], "assemble_from_prepared": []}
    for _ in range(repeats):
        start = perf_counter_ns()
        coefficient = prepare_diffusivity(grid, values, boundary)
        setup["prepare"].append((perf_counter_ns()-start)/1e6)
        start = perf_counter_ns()
        matrix = diffusion_matrix(grid, coefficient, boundary)
        setup["assemble_from_prepared"].append((perf_counter_ns()-start)/1e6)
    functions = {"prepared_matrix_free": lambda: diffusion_rate(field, grid, coefficient, boundary).ravel(),
                 "csc_matvec": lambda: matrix @ field.ravel()}
    actual, expected = (fn() for fn in functions.values())
    error = float(np.max(np.abs(actual-expected)))
    scale = max(1., float(np.max(np.abs(expected))))
    if error > 1e-12*scale:
        raise AssertionError("Operator parity failed before measurement.")
    samples = {name: [] for name in functions}
    # Interleave and reverse order to reduce systematic drift between variants.
    for i in range(repeats):
        for name in list(functions)[::1 if i % 2 == 0 else -1]:
            start = perf_counter_ns()
            functions[name]()
            samples[name].append((perf_counter_ns()-start)/1e6)
    peaks = {}
    for name, fn in functions.items():
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            result = fn()
            current, peak = tracemalloc.get_traced_memory()
            peaks[name] = {"peak_bytes": peak, "retained_result_and_tracer_bytes": current,
                           "output_bytes": result.nbytes}
        finally:
            tracemalloc.stop()
    csc_bytes = sum(x.nbytes for x in (matrix.data, matrix.indices, matrix.indptr))
    return {"schema_version": 1, "scope": "spatial operator only; no PDE, LU, routing, HTTP or UI timing",
            "grid": grid.to_dict(), "boundary": boundary, "seed": 73, "repeats": repeats,
            "coefficient": coefficient.metadata(), "csc_bytes": csc_bytes, "csc_nnz": matrix.nnz,
            "setup_ms": setup, "warm_apply_ms": samples,
            "median_setup_ms": {name: float(np.median(x)) for name, x in setup.items()},
            "median_warm_apply_ms": {name: float(np.median(x)) for name, x in samples.items()},
            "max_absolute_difference": error, "reference_max_abs": scale,
            "tracked_allocation": peaks,
            "allocation_scope": "tracemalloc-observed heap for one warm call including result; not process RSS or all native allocator/workspace memory",
            "comparison_scope": "prepared diffusion_rate includes its input validation; scipy CSC @ is its ordinary production call; no speedup assumed",
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "numpy": np.__version__, "scipy": scipy.__version__}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=160)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--boundary", choices=["zero_flux", "periodic"], default="periodic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; select a new directory.")
    report = measure(args.nx, args.ny, args.repeats, args.boundary)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        root = Path(__file__).resolve().parents[1]
        files = sorted(path.relative_to(root) for path in (root / "backend").rglob("*.py"))
        files += [Path(__file__).relative_to(root), Path("scripts/__init__.py"), Path("requirements.lock.txt")]
        report["source_sha256"] = {}
        for path in files:
            target = stage / "measurement_source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / path, target)
            report["source_sha256"][str(path)] = hashlib.sha256(target.read_bytes()).hexdigest()
        (stage / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        artifacts = {str(path.relative_to(stage)): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                   "size_bytes": path.stat().st_size}
                     for path in sorted(stage.rglob("*")) if path.is_file()}
        (stage / "manifest.json").write_text(json.dumps({"schema_version": 1, "artifacts": artifacts}, indent=2)+"\n")
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(json.dumps({"output": str(output), "warm_apply_ms": report["median_warm_apply_ms"],
                      "parity_linf": report["max_absolute_difference"]}, indent=2))


if __name__ == "__main__":
    main()
