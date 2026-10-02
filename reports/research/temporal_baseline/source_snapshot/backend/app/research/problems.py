"""Versioned research problems; separate physical inputs from discretization.

Analytic initial conditions can be projected to finite-volume cell averages.
Point sampling remains explicit, including the historical V2 Gaussian choice.
Only the currently verified constant-coefficient equations are admitted here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal
import zipfile

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.special import erf

from ..diffusion import Grid
from ..numerics import METHOD_BACKENDS, timestep_limit


class StrictSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True, frozen=True)


class ConstantInitial(StrictSpec):
    kind: Literal["constant"] = "constant"
    value: float = 1.0


class GaussianInitial(StrictSpec):
    kind: Literal["gaussian"] = "gaussian"
    center: tuple[float, float]
    sigma_m: float = Field(gt=0)
    amplitude: float = 1.0
    background: float = 0.0


class CosineInitial(StrictSpec):
    kind: Literal["cosine"] = "cosine"
    modes: tuple[int, int] = (1, 1)
    amplitude: float = 0.5
    background: float = 1.0

    @model_validator(mode="after")
    def check_modes(self):
        if any(m < 0 or m > 1024 for m in self.modes):
            raise ValueError("Cosine modes must be integers between 0 and 1024.")
        return self


class ArrayInitial(StrictSpec):
    kind: Literal["npz"] = "npz"
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bounds: tuple[float, float, float, float]
    nx: int = Field(ge=2, le=4096)
    ny: int = Field(ge=2, le=4096)
    interpretation: Literal["cell_average", "point_sample"] = "cell_average"
    axis_order: Literal["y,x"] = "y,x"
    y_direction: Literal["south_to_north"] = "south_to_north"
    unit: Literal["relative_concentration"] = "relative_concentration"


Initial = Annotated[ConstantInitial | GaussianInitial | CosineInitial | ArrayInitial, Field(discriminator="kind")]


class ProblemSpec(StrictSpec):
    bounds: tuple[float, float, float, float]
    model: Literal["diffusion", "advection_diffusion"] = "diffusion"
    kappa: float = Field(default=0.05, ge=0)
    velocity: tuple[float, float] = (0.0, 0.0)
    boundary: Literal["zero_flux", "periodic", "open"] = "zero_flux"
    initial: Initial
    coordinate_system: Literal["cartesian_metres"] = "cartesian_metres"
    concentration_unit: Literal["relative_concentration"] = "relative_concentration"
    time_unit: Literal["seconds"] = "seconds"

    @model_validator(mode="after")
    def equation_contract(self):
        Grid(self.bounds, 2, 2)
        if self.model == "diffusion" and (any(self.velocity) or self.boundary == "open"):
            raise ValueError("Diffusion uses zero velocity and zero_flux or periodic boundaries.")
        if self.model == "advection_diffusion" and self.boundary == "zero_flux":
            raise ValueError("Transport uses periodic or open boundaries.")
        if isinstance(self.initial, CosineInitial) and self.boundary == "periodic":
            if any(m % 2 for m in self.initial.modes):
                raise ValueError("Periodic cosine initial data require even half-wave mode counts.")
        return self


class NumericalSpec(StrictSpec):
    nx: int = Field(default=32, ge=2, le=4096)
    ny: int = Field(default=32, ge=2, le=4096)
    output_times_s: tuple[float, ...] = (0.0, 0.1, 0.2)
    method: Literal["explicit_euler", "backward_euler", "crank_nicolson", "advection_explicit", "imex_euler"] = "explicit_euler"
    backend: Literal["numpy", "cpp", "scipy", "numpy_scipy", "auto"] = "numpy"
    dt: float | None = Field(default=None, gt=0)
    startup: Literal["none", "rannacher"] = "none"
    initialization: Literal["cell_average", "point_sample"] = "cell_average"

    @model_validator(mode="after")
    def time_contract(self):
        times = self.output_times_s
        if not times or times[0] < 0 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Output times must be nonnegative and strictly increasing.")
        return self


class BudgetSpec(StrictSpec):
    max_output_bytes: int = Field(default=128 * 1024**2, ge=1024, le=1024**3)
    max_cell_updates: int = Field(default=500_000_000, ge=1, le=20_000_000_000)


class ExperimentSpec(StrictSpec):
    schema_version: Literal[1] = 1
    problem: ProblemSpec
    numerical: NumericalSpec = NumericalSpec()
    budget: BudgetSpec = BudgetSpec()

    @model_validator(mode="after")
    def compatibility(self):
        p, n = self.problem, self.numerical
        diffusion_method = n.method in {"explicit_euler", "backward_euler", "crank_nicolson"}
        if diffusion_method != (p.model == "diffusion"):
            raise ValueError("The method must belong to the selected physical model.")
        if n.backend != "auto" and n.backend not in METHOD_BACKENDS[n.method]:
            raise ValueError("Unsupported numerical method/backend combination.")
        if n.backend == "cpp" and p.boundary != "zero_flux":
            raise ValueError("C++ currently supports zero_flux FE only.")
        if n.startup != "none" and n.method != "crank_nicolson":
            raise ValueError("Rannacher startup requires Crank-Nicolson.")
        if isinstance(p.initial, ArrayInitial):
            data = p.initial
            if (data.bounds, data.nx, data.ny, data.interpretation) != (p.bounds, n.nx, n.ny, n.initialization):
                raise ValueError("Imported array grid, domain and interpretation must exactly match; implicit resampling is forbidden.")
        return self

    def grid(self) -> Grid:
        return Grid(self.problem.bounds, self.numerical.nx, self.numerical.ny)

    def physical_descriptor(self) -> dict:
        result = self.problem.model_dump(mode="json")
        if result["initial"]["kind"] == "npz":
            result["initial"].pop("path")  # Content/geometry, not a machine-local filename.
        return result

    def identities(self) -> dict:
        physical = self.physical_descriptor()
        numerical = self.numerical.model_dump(mode="json")
        return {"physical_id": content_id(physical), "numerical_id": content_id(numerical),
                "configuration_id": content_id({"schema_version": self.schema_version,
                                                "physical": physical, "numerical": numerical})}


def content_id(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_spec(path: Path | str) -> ExperimentSpec:
    return ExperimentSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))


def initial_field(spec: ExperimentSpec, base_dir: Path | str = ".") -> np.ndarray:
    grid, initial = spec.grid(), spec.problem.initial
    average = spec.numerical.initialization == "cell_average"
    if isinstance(initial, ConstantInitial):
        return np.full((grid.ny, grid.nx), initial.value, dtype=np.float64)
    if isinstance(initial, ArrayInitial):
        path = Path(base_dir) / initial.path
        if path.stat().st_size > spec.budget.max_output_bytes:
            raise ValueError("Initial archive exceeds input byte budget.")
        raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if raw_hash != initial.sha256:
            raise ValueError("Initial archive SHA-256 mismatch.")
        with zipfile.ZipFile(path) as archive:
            if sum(info.file_size for info in archive.infolist()) > spec.budget.max_output_bytes:
                raise ValueError("Expanded initial archive exceeds input byte budget.")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"values"}:
                raise ValueError("Initial NPZ must contain exactly the values array.")
            values = archive["values"]
            if values.shape != (grid.ny, grid.nx) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
                raise ValueError("Initial array must be a finite real (ny,nx) array.")
            return np.array(values, dtype=np.float64, order="C", copy=True)
    if isinstance(initial, GaussianInitial):
        def profile(points, center, spacing):
            if not average:
                return np.exp(-0.5 * ((points-center)/initial.sigma_m)**2)
            scale = np.sqrt(2.0) * initial.sigma_m
            return initial.sigma_m * np.sqrt(np.pi/2) / spacing * (
                erf((points+spacing/2-center)/scale)-erf((points-spacing/2-center)/scale))
        xx = profile(grid.x, initial.center[0], grid.dx)
        yy = profile(grid.y, initial.center[1], grid.dy)
    else:
        lx, ly = grid.bounds[2]-grid.bounds[0], grid.bounds[3]-grid.bounds[1]
        kx, ky = np.pi*initial.modes[0]/lx, np.pi*initial.modes[1]/ly
        xx = np.cos(kx*(grid.x-grid.bounds[0]))
        yy = np.cos(ky*(grid.y-grid.bounds[1]))
        if average:
            xx *= np.sinc(kx*grid.dx/(2*np.pi))
            yy *= np.sinc(ky*grid.dy/(2*np.pi))
    return np.ascontiguousarray(initial.background + initial.amplitude * yy[:, None]*xx[None, :])


def check_budget(spec: ExperimentSpec) -> dict:
    grid, n, p = spec.grid(), spec.numerical, spec.problem
    output_bytes = grid.nx * grid.ny * len(n.output_times_s) * 8
    if output_bytes > spec.budget.max_output_bytes:
        raise ValueError("Requested output exceeds the research byte budget.")
    limit = timestep_limit(grid, p.kappa, n.method, p.velocity)
    if n.dt is not None:
        dt = n.dt
    elif np.isfinite(limit):
        dt = limit * (0.9 if n.method == "explicit_euler" else 1.0)
    else:
        dt = min(30.0, n.output_times_s[-1]) if n.output_times_s[-1] > 0 else 1.0
    # The budget is a conservative admission estimate, not an RSS or CPU cap.
    steps_budget = spec.budget.max_cell_updates // (grid.nx * grid.ny)
    aligned_reserve = len(n.output_times_s) + (1 if n.startup == "rannacher" else 0)
    if n.output_times_s[-1] > max(0, steps_budget-aligned_reserve)*dt:
        raise ValueError("Estimated cell updates exceed the research work budget.")
    return {"output_bytes": output_bytes, "estimated_steps_upper": int(np.ceil(n.output_times_s[-1]/dt))+aligned_reserve,
            "budget_kind": "preflight estimate; does not cap sparse-factor peak RSS or wall time"}


def capabilities() -> dict:
    combinations = []
    for method, backends in METHOD_BACKENDS.items():
        model = "diffusion" if method in {"explicit_euler", "backward_euler", "crank_nicolson"} else "advection_diffusion"
        for backend in sorted(backends | {"auto"}):
            for boundary in ("zero_flux", "periodic", "open"):
                for startup in ("none", "rannacher"):
                    try:
                        ExperimentSpec.model_validate({
                            "problem": {"bounds": [0, 0, 1, 1], "model": model,
                                        "boundary": boundary, "initial": {"kind": "constant"}},
                            "numerical": {"method": method, "backend": backend, "startup": startup}})
                    except ValueError:
                        continue
                    combinations.append({"model": model, "method": method, "backend": backend,
                                         "boundary": boundary, "startup": startup})
    return {"schema_version": 1, "models": ["diffusion", "advection_diffusion"],
            "methods": {k: sorted(v) for k, v in METHOD_BACKENDS.items()},
            "supported_combinations": combinations,
            "initial_conditions": ["constant", "gaussian", "cosine", "npz"],
            "initialization": ["cell_average", "point_sample"],
            "coefficient_scope": "constant scalar kappa and constant velocity",
            "native_scope": "explicit_euler, zero_flux only",
            "combination_validation": "ExperimentSpec + solver validate model, boundary, startup, backend and CFL",
            "physical_units": {"length": "m", "time": "s", "concentration": "relative"}}
