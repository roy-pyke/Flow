"""Explicit conversion and provenance checks for the optional native backend."""
from __future__ import annotations

from hashlib import sha256
from importlib import import_module
from pathlib import Path
from time import perf_counter

import numpy as np

from ..diffusion import Grid, stable_timestep


def source_hash() -> str:
    root = Path(__file__).resolve().parents[3] / "cpp"
    paths = sorted([root / "CMakeLists.txt", root / "pyproject.toml", root / "source_hash.py", *root.glob("src/*")])
    payload = "".join(f"{path.relative_to(root).as_posix()}:{sha256(path.read_bytes()).hexdigest()}\n" for path in paths if path.is_file())
    return sha256(payload.encode()).hexdigest()


def status() -> dict:
    """An installed but stale binary is unavailable until explicitly rebuilt."""
    expected = source_hash()
    try:
        module = import_module("flow_cpp")
    except ImportError as exc:
        return {"available": False, "reason": f"Native extension is not installed or loadable: {exc}", "source_hash": expected, "build_info": None}
    info = dict(module.build_info())
    matches = info.get("source_hash") == expected
    return {"available": matches, "reason": None if matches else "Native extension was built from different source; reinstall ./cpp.", "source_hash": expected, "build_info": info}


def version() -> dict:
    info = status()
    return {**info, "wrapper_hash": sha256(Path(__file__).read_bytes()).hexdigest()}


native_version = version


def _require_native():
    info = status()
    if not info["available"]:
        raise RuntimeError(f"{info['reason']} Build with .venv/bin/python -m pip install ./cpp")
    return import_module("flow_cpp"), info


def solve_native(grid: Grid, initial, kappa: float, times, dt: float | None = None) -> tuple[np.ndarray, dict]:
    started = perf_counter()
    module, info = _require_native()
    conversion_started = perf_counter()
    field = np.require(initial, dtype=np.float64, requirements=["C", "A"])
    output_times = np.require(times, dtype=np.float64, requirements=["C", "A"])
    conversion_ms = (perf_counter() - conversion_started) * 1000
    if field.shape != (grid.ny, grid.nx):
        raise ValueError("Initial field must have grid shape (ny,nx).")
    if dt is None and kappa == 0:
        positive_gaps = np.diff(np.r_[0.0, output_times]) if output_times.ndim == 1 else np.array([])
        positive_gaps = positive_gaps[positive_gaps > 0]
        chosen_dt = min(30.0, float(positive_gaps.min())) if len(positive_gaps) else 1.0
    else:
        chosen_dt = stable_timestep(grid, kappa) if dt is None else float(dt)
    frames, diag = module.solve(field, output_times, grid.dx, grid.dy, kappa, chosen_dt)
    diag.update({"conversion_ms": conversion_ms, "solve_ms": (perf_counter() - started) * 1000,
                 "backend": "cpp", "method": "explicit_euler", "native_build": info["build_info"]})
    return frames, diag
