"""Versioned research problems; separate physical inputs from discretization.

Analytic initial conditions can be projected to finite-volume cell averages.
Point sampling remains explicit, including the historical V2 Gaussian choice.
Diffusion admits constant scalars or positive cell material diffusivities;
advection-diffusion retains its constant-coefficient contract.
"""
from __future__ import annotations

import hashlib
import json
import math
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


class LayeredDiffusivity(StrictSpec):
    """Two positive materials meeting at one axis-aligned physical grid face."""
    kind: Literal["layered"] = "layered"
    axis: Literal["x", "y"] = "x"
    interface_m: float
    left: float = Field(gt=0)
    right: float = Field(gt=0)


class ArrayDiffusivity(StrictSpec):
    """Positive cell material samples; no implicit resampling or averaging."""
    kind: Literal["npz"] = "npz"
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bounds: tuple[float, float, float, float]
    nx: int = Field(ge=2, le=4096)
    ny: int = Field(ge=2, le=4096)
    interpretation: Literal["cell_material"] = "cell_material"
    axis_order: Literal["y,x"] = "y,x"
    y_direction: Literal["south_to_north"] = "south_to_north"
    unit: Literal["m2/s"] = "m2/s"


VariableDiffusivity = Annotated[LayeredDiffusivity | ArrayDiffusivity, Field(discriminator="kind")]
Diffusivity = Annotated[float, Field(ge=0)] | VariableDiffusivity


class UniformWind(StrictSpec):
    kind: Literal["uniform"] = "uniform"
    value: tuple[float, float] = (0.0, 0.0)


class ShearWind(StrictSpec):
    kind: Literal["shear"] = "shear"
    axis: Literal["x", "y"] = "x"
    mean_speed_m_s: float = 0.0
    gradient_s_inv: float = 0.0


class RotationWind(StrictSpec):
    kind: Literal["rotation"] = "rotation"
    angular_rate_s_inv: float
    center: tuple[float, float] | None = None


class ConstantSource(StrictSpec):
    kind: Literal["constant"] = "constant"
    value: float = 0.0


class GaussianSource(StrictSpec):
    kind: Literal["gaussian"] = "gaussian"
    center: tuple[float, float]
    sigma_m: float = Field(gt=0)
    amplitude: float = 1.0


class InflowValues(StrictSpec):
    left: float = Field(default=0.0, ge=0)
    right: float = Field(default=0.0, ge=0)
    bottom: float = Field(default=0.0, ge=0)
    top: float = Field(default=0.0, ge=0)


class ForcingContract(StrictSpec):
    times_s: tuple[float, ...] = Field(min_length=2, max_length=4096)
    temporal_interpolation: Literal["linear"] = "linear"
    temporal_extrapolation: Literal["reject"] = "reject"
    velocity_unit: Literal["m/s"] = "m/s"
    source_unit: Literal["relative_concentration/s"] = "relative_concentration/s"
    inflow_unit: Literal["relative_concentration"] = "relative_concentration"

    @model_validator(mode="after")
    def validate_clock(self):
        if self.times_s[0] != 0 or any(b <= a or not np.isfinite(b-a) for a,b in zip(self.times_s,self.times_s[1:])):
            raise ValueError("Forcing knots must start at zero and have strictly increasing finite intervals.")
        return self


class AnalyticForcing(ForcingContract):
    kind: Literal["analytic"] = "analytic"
    generator_version: Literal[1] = 1
    wind: Annotated[UniformWind | ShearWind | RotationWind, Field(discriminator="kind")] = UniformWind()
    wind_scale: tuple[float, ...] | None = None
    source: Annotated[ConstantSource | GaussianSource, Field(discriminator="kind")] = ConstantSource()
    source_scale: tuple[float, ...] | None = None
    inflow: InflowValues = InflowValues()
    inflow_scale: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def validate_scales(self):
        for name in ("wind_scale", "source_scale", "inflow_scale"):
            values = getattr(self, name)
            if values is not None and len(values) != len(self.times_s):
                raise ValueError(f"{name} must have one value per forcing knot.")
        if self.inflow_scale is not None and any(value < 0 for value in self.inflow_scale):
            raise ValueError("Inflow multipliers must be nonnegative.")
        return self


class ArrayForcing(ForcingContract):
    kind: Literal["npz"] = "npz"
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bounds: tuple[float, float, float, float]
    nx: int = Field(ge=2, le=4096)
    ny: int = Field(ge=2, le=4096)
    axis_order: Literal["time,y,x"] = "time,y,x"
    y_direction: Literal["south_to_north"] = "south_to_north"
    representation: Literal["face_velocity_cell_average_source_boundary_face_inflow"] = "face_velocity_cell_average_source_boundary_face_inflow"


Forcing = Annotated[AnalyticForcing | ArrayForcing, Field(discriminator="kind")]


def _layer_face(grid: Grid, data: LayeredDiffusivity) -> int:
    axis = 0 if data.axis == "x" else 1
    lower, upper = grid.bounds[axis], grid.bounds[axis + 2]
    count = grid.nx if axis == 0 else grid.ny
    if not lower < data.interface_m < upper:
        raise ValueError("Layer interface must be strictly inside the physical domain.")
    spacing = (upper - lower) / count
    coordinate_ulp = max(math.ulp(lower), math.ulp(upper), math.ulp(data.interface_m))
    # Projected coordinates can be large relative to the grid spacing. The
    # interface-minus-origin subtraction therefore loses index-space digits.
    # Compare reconstructed physical faces instead; only two coordinate ULPs
    # of representation error are allowed. At least 32 ULPs per cell keep that
    # allowance far smaller than a cell and prevent neighboring-face overlap.
    if not np.isfinite(spacing) or spacing <= 32 * coordinate_ulp:
        raise ValueError("Layer grid spacing has insufficient physical-coordinate precision to resolve grid faces.")
    index = (data.interface_m - lower) / spacing
    nearest = round(index)
    face = lower + nearest * spacing
    tolerance = 2 * max(coordinate_ulp, math.ulp(face))
    if (not 0 < nearest < count or abs(data.interface_m - face) > tolerance):
        raise ValueError("Layer interface must align with a grid face; subcell interfaces are unsupported.")
    return nearest


class ProblemSpec(StrictSpec):
    bounds: tuple[float, float, float, float]
    model: Literal["diffusion", "advection_diffusion"] = "diffusion"
    kappa: Diffusivity = 0.05
    velocity: tuple[float, float] = (0.0, 0.0)
    forcing: Forcing | None = None
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
        if self.model == "advection_diffusion" and self.boundary == "zero_flux" and self.forcing is None:
            raise ValueError("Transport uses periodic or open boundaries.")
        if not isinstance(self.kappa, float) and self.model != "diffusion":
            raise ValueError("Variable diffusivity supports pure diffusion only.")
        if self.forcing is not None and any(self.velocity):
            raise ValueError("Prescribed forcing owns the wind; legacy velocity must be zero.")
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
    max_forcing_bytes: int = Field(default=128 * 1024**2, ge=1024, le=1024**3)


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
        if not isinstance(p.kappa, float) and n.backend == "cpp":
            raise ValueError("C++ does not support variable diffusivity; use NumPy FE or SciPy BE/CN.")
        if n.startup != "none" and n.method != "crank_nicolson":
            raise ValueError("Rannacher startup requires Crank-Nicolson.")
        if isinstance(p.initial, ArrayInitial):
            data = p.initial
            if (data.bounds, data.nx, data.ny, data.interpretation) != (p.bounds, n.nx, n.ny, n.initialization):
                raise ValueError("Imported array grid, domain and interpretation must exactly match; implicit resampling is forbidden.")
        if isinstance(p.kappa, LayeredDiffusivity):
            _layer_face(self.grid(), p.kappa)
        elif isinstance(p.kappa, ArrayDiffusivity):
            if (p.kappa.bounds, p.kappa.nx, p.kappa.ny) != (p.bounds, n.nx, n.ny):
                raise ValueError("Imported diffusivity grid and domain must exactly match; implicit resampling is forbidden.")
        if p.forcing is not None:
            if n.backend == "cpp":
                raise ValueError("C++ does not support prescribed forcing.")
            if p.forcing.times_s[-1] < n.output_times_s[-1]:
                raise ValueError("Forcing knots must cover the complete solve interval from zero.")
            if isinstance(p.forcing, ArrayForcing) and (p.forcing.bounds, p.forcing.nx, p.forcing.ny) != (p.bounds, n.nx, n.ny):
                raise ValueError("Imported forcing grid and domain must exactly match; implicit resampling is forbidden.")
        return self

    def grid(self) -> Grid:
        return Grid(self.problem.bounds, self.numerical.nx, self.numerical.ny)

    def physical_descriptor(self) -> dict:
        result = self.problem.model_dump(mode="json")
        if result["initial"]["kind"] == "npz":
            result["initial"].pop("path")  # Content/geometry, not a machine-local filename.
        if isinstance(self.problem.kappa, ArrayDiffusivity):
            result["kappa"].pop("path")
        if self.problem.forcing is None:
            result.pop("forcing")  # Preserve every pre-forcing physical identity.
        elif isinstance(self.problem.forcing, ArrayForcing):
            result["forcing"].pop("path")
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


def check_output_budget(spec: ExperimentSpec) -> dict:
    """Reject oversized requested output before allocating or reading inputs."""
    grid, n = spec.grid(), spec.numerical
    output_bytes = grid.nx * grid.ny * len(n.output_times_s) * 8
    if output_bytes > spec.budget.max_output_bytes:
        raise ValueError("Requested output exceeds the research byte budget.")
    return {"output_bytes": output_bytes}


def inspect_npz_array(path: Path | str, key: str, shape: tuple[int, ...], *,
                      max_bytes: int, float64: bool = False) -> None:
    """Check the NPY header before NumPy can allocate an attacker-sized shape.

    Only numeric NPY v1/v2 members are admitted. ZIP expansion checks alone do
    not establish an array allocation bound when its shape header is forged.
    """
    with zipfile.ZipFile(path) as archive:
        entries = [item for item in archive.infolist() if item.filename == key + ".npy"]
        if len(entries) != 1:
            raise ValueError("Each NPZ array must have exactly one named NPY member.")
        entry = entries[0]
        with archive.open(entry) as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                actual_shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                actual_shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError("Only numeric NPY format versions 1 and 2 are supported.")
            if actual_shape != shape:
                raise ValueError("NPZ array header shape differs from the configured grid: " + key)
            if dtype.hasobject or dtype.kind not in "fiu" or (float64 and (dtype.kind != "f" or dtype.itemsize != 8)):
                raise ValueError("NPZ array header must describe finite-real-compatible numeric values: " + key)
            size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if size > max_bytes or handle.tell() + size != entry.file_size:
                raise ValueError("NPZ array header size disagrees with its payload or exceeds byte budget: " + key)


def forcing_shapes(spec: ExperimentSpec) -> dict[str, tuple[int, ...]]:
    if spec.problem.forcing is None:
        return {}
    nt, nx, ny = len(spec.problem.forcing.times_s), spec.numerical.nx, spec.numerical.ny
    return {"times_s": (nt,), "velocity_x": (nt, ny, nx+1), "velocity_y": (nt, ny+1, nx),
            "source": (nt, ny, nx), "inflow_left": (nt, ny), "inflow_right": (nt, ny),
            "inflow_bottom": (nt, nx), "inflow_top": (nt, nx)}


def check_forcing_budget(spec: ExperimentSpec) -> dict:
    check_output_budget(spec)
    size = sum(math.prod(shape)*8 for shape in forcing_shapes(spec).values())
    if size > spec.budget.max_forcing_bytes:
        raise ValueError("Resolved forcing arrays exceed the forcing byte budget.")
    return {"forcing_bytes": size}


def forcing_array_payload(forcing) -> dict[str, np.ndarray]:
    return {"times_s": forcing.times_s, "velocity_x": forcing.velocity_x,
            "velocity_y": forcing.velocity_y, "source": forcing.source,
            **{"inflow_"+side: forcing.inflow[side] for side in ("left", "right", "bottom", "top")}}


def imported_forcing_arrays(spec: ExperimentSpec, base_dir: Path | str = ".") -> dict[str, np.ndarray]:
    """Read exact resolved NPZ values with shape admission before allocation."""
    check_forcing_budget(spec)
    data = spec.problem.forcing
    if not isinstance(data, ArrayForcing):
        raise ValueError("Imported forcing arrays require an NPZ forcing specification.")
    path = Path(base_dir) / data.path
    allowance = spec.budget.max_forcing_bytes + 1_000_000
    if path.stat().st_size > allowance:
        raise ValueError("Forcing input archive exceeds its byte budget.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != data.sha256:
        raise ValueError("Forcing input archive SHA-256 mismatch.")
    shapes = forcing_shapes(spec)
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if (len(entries) != len(shapes) or {item.filename for item in entries} != {key+".npy" for key in shapes}):
            raise ValueError("Forcing NPZ must contain exactly its eight declared arrays.")
        if sum(item.file_size for item in entries) > allowance:
            raise ValueError("Expanded forcing input exceeds its byte budget.")
    for key, shape in shapes.items():
        inspect_npz_array(path, key, shape, max_bytes=spec.budget.max_forcing_bytes)
    with np.load(path, allow_pickle=False) as archive:
        with np.errstate(over="ignore", invalid="ignore"):
            arrays = {key: np.array(archive[key], dtype=np.float64, order="C", copy=True) for key in shapes}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Forcing input values must remain finite in float64.")
    if not np.array_equal(arrays["times_s"], data.times_s):
        raise ValueError("Forcing input knot times differ from the physical configuration.")
    return arrays


def forcing_fields(spec: ExperimentSpec, base_dir: Path | str = "."):
    """Resolve and freeze shared-knot face winds, cell sources and inflows."""
    check_forcing_budget(spec)
    data = spec.problem.forcing
    if data is None:
        return None
    from ..numerics.forcing import PrescribedFields
    grid = spec.grid()
    if isinstance(data, ArrayForcing):
        arrays = imported_forcing_arrays(spec, base_dir)
    else:
        nt = len(data.times_s)
        ux = np.zeros((grid.ny, grid.nx+1))
        uy = np.zeros((grid.ny+1, grid.nx))
        wind = data.wind
        with np.errstate(over="ignore", invalid="ignore"):
            if isinstance(wind, UniformWind):
                ux.fill(wind.value[0]); uy.fill(wind.value[1])
            elif isinstance(wind, ShearWind):
                if wind.axis == "x":
                    ux[:] = (wind.mean_speed_m_s + wind.gradient_s_inv*(grid.y-(grid.bounds[1]+grid.bounds[3])/2))[:,None]
                else:
                    uy[:] = wind.mean_speed_m_s + wind.gradient_s_inv*(grid.x-(grid.bounds[0]+grid.bounds[2])/2)
            else:
                center = wind.center or ((grid.bounds[0]+grid.bounds[2])/2,(grid.bounds[1]+grid.bounds[3])/2)
                ux[:] = (-wind.angular_rate_s_inv*(grid.y-center[1]))[:,None]
                uy[:] = wind.angular_rate_s_inv*(grid.x-center[0])
            source = data.source
            if isinstance(source, ConstantSource):
                density = np.full((grid.ny,grid.nx),source.value)
            else:
                def average(points, center, spacing):
                    scale = np.sqrt(2.0)*source.sigma_m
                    return source.sigma_m*np.sqrt(np.pi/2)/spacing*(
                        erf((points+spacing/2-center)/scale)-erf((points-spacing/2-center)/scale))
                density = source.amplitude * average(grid.y,source.center[1],grid.dy)[:,None] * average(grid.x,source.center[0],grid.dx)[None,:]
            def scale(values):
                return np.ones(nt) if values is None else np.asarray(values,dtype=float)
            ws, ss, ins = scale(data.wind_scale), scale(data.source_scale), scale(data.inflow_scale)
            arrays = {"times_s": np.asarray(data.times_s), "velocity_x": ws[:,None,None]*ux,
                      "velocity_y": ws[:,None,None]*uy, "source": ss[:,None,None]*density,
                      **{"inflow_"+side: np.broadcast_to((ins*getattr(data.inflow,side))[:,None],
                          (nt,grid.ny if side in {"left","right"} else grid.nx)).copy()
                         for side in ("left","right","bottom","top")}}
    fields = PrescribedFields(grid,arrays["times_s"],velocity_x=arrays["velocity_x"],
                              velocity_y=arrays["velocity_y"],source=arrays["source"],
                              inflow={side:arrays["inflow_"+side] for side in ("left","right","bottom","top")},
                              boundary=spec.problem.boundary)
    if spec.problem.model == "diffusion" and fields.has_velocity:
        raise ValueError("Pure diffusion forcing may contain volume sources but no velocity field.")
    return fields


def diffusivity_field(spec: ExperimentSpec, base_dir: Path | str = ".") -> float | np.ndarray:
    """Resolve positive cell material values, retaining the legacy scalar form."""
    check_output_budget(spec)
    grid, data = spec.grid(), spec.problem.kappa
    if isinstance(data, float):
        return data
    if isinstance(data, LayeredDiffusivity):
        face = _layer_face(grid, data)
        values = np.full((grid.ny, grid.nx), data.right, dtype=np.float64)
        if data.axis == "x":
            values[:, :face] = data.left
        else:
            values[:face, :] = data.left
        return values
    path = Path(base_dir) / data.path
    if path.stat().st_size > spec.budget.max_output_bytes:
        raise ValueError("Diffusivity archive exceeds input byte budget.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != data.sha256:
        raise ValueError("Diffusivity archive SHA-256 mismatch.")
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > spec.budget.max_output_bytes:
            raise ValueError("Expanded diffusivity archive exceeds input byte budget.")
        if len(archive.infolist()) != 1 or archive.infolist()[0].filename != "values.npy":
            raise ValueError("Diffusivity NPZ must contain exactly the values array.")
    inspect_npz_array(path, "values", (grid.ny, grid.nx), max_bytes=spec.budget.max_output_bytes)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"values"} or len(archive.files) != 1:
            raise ValueError("Diffusivity NPZ must contain exactly the values array.")
        values = archive["values"]
        if (values.shape != (grid.ny, grid.nx) or values.dtype.kind not in "fiu"
                or not np.isfinite(values).all() or np.any(values <= 0)):
            raise ValueError("Diffusivity must be a finite strictly positive real (ny,nx) array.")
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.array(values, dtype=np.float64, order="C", copy=True)
        if not np.isfinite(converted).all() or np.any(converted <= 0):
            raise ValueError("Diffusivity must remain finite and strictly positive in float64.")
        return converted


def check_budget(spec: ExperimentSpec, *, kappa=None, forcing=None, base_dir: Path | str = ".") -> dict:
    """Use the resolved face-based CFL; never estimate a field by a scalar guess."""
    output = check_output_budget(spec)
    forcing_budget = check_forcing_budget(spec)
    from ..numerics.coefficients import prepare_diffusivity
    grid, n, p = spec.grid(), spec.numerical, spec.problem
    coefficient = diffusivity_field(spec, base_dir) if kappa is None else kappa
    coefficient = prepare_diffusivity(grid, coefficient, p.boundary)
    driving = forcing_fields(spec, base_dir) if forcing is None and p.forcing is not None else forcing
    limit = timestep_limit(grid, coefficient, n.method, p.velocity, boundary=p.boundary, forcing=driving)
    if n.dt is not None:
        dt = n.dt
    elif np.isfinite(limit):
        dt = limit * (0.9 if n.method == "explicit_euler" else 1.0)
    else:
        dt = min(30.0, n.output_times_s[-1]) if n.output_times_s[-1] > 0 else 1.0
    if not np.isfinite(dt) or dt <= 0 or dt > limit * (1 + 1e-12):
        raise ValueError("Requested time step violates the research method's CFL bound.")
    # The budget is a conservative admission estimate, not an RSS or CPU cap.
    steps_budget = spec.budget.max_cell_updates // (grid.nx * grid.ny)
    knot_reserve = sum(0 < time < n.output_times_s[-1] for time in p.forcing.times_s) if p.forcing is not None else 0
    aligned_reserve = len(n.output_times_s) + knot_reserve + (1 if n.startup == "rannacher" else 0)
    if n.output_times_s[-1] > max(0, steps_budget-aligned_reserve)*dt:
        raise ValueError("Estimated cell updates exceed the research work budget.")
    return {**output, **forcing_budget, "forcing_knot_step_reserve": knot_reserve,
            "estimated_steps_upper": int(np.ceil(n.output_times_s[-1]/dt))+aligned_reserve,
            "admission_dt_s": float(dt), "cfl_limit_s": float(limit) if np.isfinite(limit) else None,
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
            "diffusivity_inputs": ["nonnegative scalar", "positive layered", "positive cell-material npz"],
            "coefficient_scope": "pure diffusion: nonnegative scalar or positive cell material diffusivity; transport: scalar kappa with legacy constant or prescribed face wind",
            "variable_diffusivity_combinations": [{"model": "diffusion", "method": method, "backend": backend,
                                                   "boundary": boundary, "startup": startup}
                for method, supported in [("explicit_euler", ["numpy", "auto"]),
                                          ("backward_euler", ["scipy", "auto"]),
                                          ("crank_nicolson", ["scipy", "auto"])]
                for backend in supported for boundary in ["zero_flux", "periodic"]
                for startup in (["none", "rannacher"] if method == "crank_nicolson" else ["none"])],
            "variable_auto_backend": "explicit_euler uses numpy; backward_euler/crank_nicolson use scipy",
            "prescribed_forcing": {"schema_version": 1, "inputs": ["analytic_v1", "complete_npz"],
                "wind": ["uniform_with_knot_reversal", "linear_shear_face_averages", "linear_rotation_face_averages"],
                "source": ["constant", "gaussian_cell_average_with_knot_scale", "signed_finite_npz"],
                "temporal_rule": "shared knots, linear interpolation, no extrapolation",
                "source_only_methods": ["explicit_euler", "backward_euler", "crank_nicolson"],
                "transport_methods": ["advection_explicit", "imex_euler"],
                "supported_combinations": [
                    {"model": "diffusion", "scope": "source_only", "diffusivity": "scalar_or_positive_cell_material",
                     "method": method, "backend": backend, "boundary": boundary, "startup": startup}
                    for method,supported in [("explicit_euler",["numpy","auto"]),
                        ("backward_euler",["scipy","auto"]),("crank_nicolson",["scipy","auto"])]
                    for backend in supported for boundary in ["zero_flux","periodic"]
                    for startup in (["none","rannacher"] if method=="crank_nicolson" else ["none"])] + [
                    {"model": "advection_diffusion", "scope": "wind_source_inflow", "diffusivity": "scalar",
                     "method": method, "backend": backend, "boundary": boundary, "startup": "none"}
                    for method,supported in [("advection_explicit",["numpy","auto"]),("imex_euler",["numpy_scipy","auto"])]
                    for backend in supported for boundary in ["zero_flux","periodic","open"]],
                "native_supported": False,
                "boundary_rule": "zero_flux requires zero normal boundary wind; periodic seams must match; inflow only open"},
            "native_scope": "constant scalar explicit_euler, zero_flux only",
            "combination_validation": "ExperimentSpec + solver validate model, boundary, startup, backend and CFL",
            "physical_units": {"length": "m", "time": "s", "concentration": "relative"}}
