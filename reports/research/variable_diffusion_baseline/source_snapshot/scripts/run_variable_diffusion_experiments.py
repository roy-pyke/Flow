"""Reproducible variable-diffusivity spatial, temporal and interface studies.

The smooth mode is a closed-boundary homogeneous PDE solution, not a forced
solution. Half-cell Dirichlet terms occur only in the steady validation system;
the production boundary contract remains zero_flux or periodic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Literal
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from pydantic import Field, model_validator
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply, spsolve

from backend.app.diffusion import Grid
from backend.app.numerics import solve
from backend.app.numerics.operators import diffusion_matrix
from backend.app.research.experiments import provenance
from backend.app.research.problems import StrictSpec, content_id, inspect_npz_array

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/research/variable_diffusion_study.json"
METHODS = ("explicit_euler", "backward_euler", "crank_nicolson")


class SpatialSpec(StrictSpec):
    length_x_m: float = Field(default=1, gt=0)
    length_y_m: float = Field(default=.25, gt=0)
    nx_values: tuple[int, ...] = (16, 32, 64, 128)
    ny: int = Field(default=4, ge=2, le=8)
    base_diffusivity: float = Field(default=.15, gt=0, le=100)
    shape_b: float = .06
    mean: float = Field(default=1, gt=0)
    amplitude: float = Field(default=.3, gt=0)
    duration_s: float = Field(default=.2, gt=0)
    dt_s: float = Field(default=.0005, gt=0)

    @model_validator(mode="after")
    def protocol(self):
        if (not 2 <= len(self.nx_values) <= 6 or any(not 4 <= n <= 256 for n in self.nx_values)
                or any(b <= a for a, b in zip(self.nx_values, self.nx_values[1:]))):
            raise ValueError("Spatial sizes must be 2..6 strictly increasing integers in 4..256.")
        if not 0 < abs(self.shape_b) < 1/9:
            raise ValueError("Smooth variable coefficient requires 0 < abs(shape_b) < 1/9.")
        if self.mean <= self.amplitude*(1+abs(self.shape_b)):
            raise ValueError("Mean must exceed the smooth mode's maximum amplitude.")
        if not np.isfinite(self.duration_s/self.dt_s) or self.duration_s/self.dt_s > 100000:
            raise ValueError("Spatial run exceeds its maximum step count.")
        return self


class TemporalSpec(StrictSpec):
    length_x_m: float = Field(default=1, gt=0)
    length_y_m: float = Field(default=.7, gt=0)
    nx: int = Field(default=10, ge=4, le=24)
    ny: int = Field(default=6, ge=3, le=16)
    base_diffusivity: float = Field(default=.15, gt=0, le=100)
    variation: float = Field(default=.55, gt=0, lt=1)
    mean: float = Field(default=1, gt=0)
    amplitude: float = Field(default=.25, gt=0)
    duration_cfl_multiple: float = Field(default=4, gt=0, le=32)
    step_counts: tuple[int, ...] = (8, 16, 32, 64)
    boundaries: tuple[Literal["zero_flux", "periodic"], ...] = ("zero_flux", "periodic")

    @model_validator(mode="after")
    def protocol(self):
        if self.nx*self.ny > 256:
            raise ValueError("Independent dense temporal reference is limited to 256 cells.")
        if (not 3 <= len(self.step_counts) <= 8 or any(not 2 <= n <= 4096 for n in self.step_counts)
                or any(b <= a for a,b in zip(self.step_counts,self.step_counts[1:]))):
            raise ValueError("Temporal step counts must be 3..8 strictly increasing integers in 2..4096.")
        if self.duration_cfl_multiple > min(self.step_counts):
            raise ValueError("The coarsest FE step would violate the independent positivity bound.")
        if not self.boundaries or len(set(self.boundaries)) != len(self.boundaries):
            raise ValueError("Temporal boundaries must be nonempty and unique.")
        if self.mean <= self.amplitude:
            raise ValueError("Temporal initial concentration must remain positive.")
        return self


class LayeredSpec(StrictSpec):
    length_x_m: float = Field(default=1, gt=0)
    length_y_m: float = Field(default=.25, gt=0)
    nx: int = Field(default=32, ge=4, le=128)
    ny: int = Field(default=4, ge=2, le=8)
    interface_fraction: float = Field(default=.5, gt=0, lt=1)
    kappa_left: float = Field(default=1, gt=0, le=1e6)
    kappa_right: float = Field(default=20, gt=0, le=1e6)
    concentration_left: float = 1
    concentration_right: float = 0

    @model_validator(mode="after")
    def protocol(self):
        location = self.nx*self.interface_fraction
        if self.nx*self.ny > 512 or abs(location-round(location)) > 1e-12:
            raise ValueError("Layer interface must coincide with a grid face; at most 512 cells.")
        if self.kappa_left == self.kappa_right or self.concentration_left == self.concentration_right:
            raise ValueError("The interface counterexample requires unequal coefficients and boundary values.")
        return self


class StudyBudget(StrictSpec):
    max_array_bytes: int = Field(default=128*1024**2, ge=1024, le=1024**3)
    max_cell_updates: int = Field(default=100_000_000, ge=1, le=2_000_000_000)
    max_reference_scaled_rate: float = Field(default=10000, gt=0, le=100000)


class Study(StrictSpec):
    schema_version: Literal[1] = 1
    spatial: SpatialSpec = SpatialSpec()
    temporal: TemporalSpec = TemporalSpec()
    layered: LayeredSpec = LayeredSpec()
    budget: StudyBudget = StudyBudget()

    @model_validator(mode="after")
    def admission(self):
        check_budget(self)
        return self


def expected_raw_shapes(spec: Study) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for n in spec.spatial.nx_values:
        for suffix in ("kappa", "initial", "numerical", "exact_cell_average", "semidiscrete"):
            shapes[f"spatial_n{n}__{suffix}"] = (spec.spatial.ny,n)
    t=spec.temporal; cells=t.nx*t.ny
    for boundary in t.boundaries:
        prefix=f"temporal_{boundary}__"
        for key in ("kappa", "initial", "semidiscrete"):
            shapes[prefix+key]=(t.ny,t.nx)
        shapes[prefix+"independent_operator"]=(cells,cells)
        shapes[prefix+"eigenvalues"]=(cells,)
        for method in METHODS:
            for count in t.step_counts:
                shapes[prefix+f"{method}_steps{count}"]=(t.ny,t.nx)
    l=spec.layered;cells=l.nx*l.ny
    for key in ("kappa","exact_cell_average","harmonic","arithmetic","rhs"):
        shapes["layered__"+key]=(l.ny,l.nx)
    for key in ("production_closed_operator","harmonic_dirichlet_operator","arithmetic_dirichlet_operator"):
        shapes["layered__"+key]=(cells,cells)
    for average in ("harmonic","arithmetic"):
        shapes[f"layered__{average}_flux_x"]=(l.ny,l.nx+1)
        shapes[f"layered__{average}_flux_y"]=(l.ny+1,l.nx)
    return shapes


def check_budget(spec: Study) -> dict:
    shapes=expected_raw_shapes(spec)
    raw_bytes=sum(int(np.prod(shape))*8 for shape in shapes.values())
    s,t=spec.spatial,spec.temporal
    updates=sum(n*s.ny*(int(np.ceil(s.duration_s/s.dt_s))+2) for n in s.nx_values)
    updates+=len(t.boundaries)*len(METHODS)*t.nx*t.ny*sum(n+2 for n in t.step_counts)
    if raw_bytes>spec.budget.max_array_bytes or updates>spec.budget.max_cell_updates:
        raise ValueError("Study exceeds aggregate raw-array or estimated cell-update budget.")
    return {"raw_array_bytes":raw_bytes,"raw_array_count":len(shapes),
            "estimated_cell_updates_upper":updates,
            "max_reference_scaled_rate":spec.budget.max_reference_scaled_rate,
            "scope":"array/update admission plus spatial exponential T*maxOutgoing cap; excludes sparse factor fill, dense eigensolver workspace, peak RSS and wall time"}


def independent_dense_operator(grid: Grid, coefficient: np.ndarray, boundary: str = "zero_flux", *,
                               face_average: str = "harmonic") -> np.ndarray:
    """Independent explicit face-loop reference; never calls production operators."""
    k=np.asarray(coefficient,dtype=float)
    if k.shape!=(grid.ny,grid.nx) or not np.isfinite(k).all() or np.any(k<=0):
        raise ValueError("Reference coefficient must be a finite positive grid array.")
    if boundary not in {"zero_flux","periodic"} or face_average not in {"harmonic","arithmetic"}:
        raise ValueError("Unsupported reference face or boundary policy.")
    size=grid.nx*grid.ny;matrix=np.zeros((size,size))
    def face(a,b,h):
        ka,kb=k.ravel()[a],k.ravel()[b]
        lo,hi=min(ka,kb),max(ka,kb)
        value=(lo/(.5+.5*(lo/hi)) if face_average=="harmonic" else ka/2+kb/2)/h**2
        if not np.isfinite(value):raise ValueError("Reference face rate is not representable finitely.")
        matrix[a,a]-=value;matrix[b,b]-=value;matrix[a,b]+=value;matrix[b,a]+=value
    for y in range(grid.ny):
        for x in range(grid.nx-1):face(y*grid.nx+x,y*grid.nx+x+1,grid.dx)
        if boundary=="periodic":face(y*grid.nx+grid.nx-1,y*grid.nx,grid.dx)
    for x in range(grid.nx):
        for y in range(grid.ny-1):face(y*grid.nx+x,(y+1)*grid.nx+x,grid.dy)
        if boundary=="periodic":face((grid.ny-1)*grid.nx+x,x,grid.dy)
    if not np.isfinite(matrix).all():raise ValueError("Reference matrix is not representable finitely.")
    return matrix


def smooth_coefficient(grid: Grid, base_diffusivity: float, shape_b: float) -> np.ndarray:
    theta=np.pi*(grid.x-grid.bounds[0])/(grid.bounds[2]-grid.bounds[0])
    ratio=1+2*np.cos(2*theta)
    row=base_diffusivity*(1+shape_b/3*ratio)/(1+3*shape_b*ratio)
    return np.broadcast_to(row,(grid.ny,grid.nx)).copy()


def smooth_exact_cell_average(grid: Grid, time_s: float, spec: SpatialSpec) -> np.ndarray:
    wave=np.pi/spec.length_x_m
    theta=wave*(grid.x-grid.bounds[0])
    row=(np.sinc(wave*grid.dx/(2*np.pi))*np.cos(theta)
         +spec.shape_b*np.sinc(3*wave*grid.dx/(2*np.pi))*np.cos(3*theta))
    return np.broadcast_to(spec.mean+spec.amplitude*np.exp(-spec.base_diffusivity*wave**2*time_s)*row,
                           (grid.ny,grid.nx)).copy()


def errors(value,reference) -> dict:
    difference=np.asarray(value)-np.asarray(reference)
    return {"linf":float(np.max(np.abs(difference))),"rms":float(np.sqrt(np.mean(difference**2)))}


def _add_orders(rows: list[dict], step_key: str, error_key: str) -> None:
    for previous,current in zip(rows,rows[1:]):
        a,b=previous[error_key]["rms"],current[error_key]["rms"]
        current["observed_rms_order"]=(float(np.log(a/b)/np.log(previous[step_key]/current[step_key]))
                                        if min(a,b)>0 else None)
    rows[0]["observed_rms_order"]=None


def spatial_study(spec: SpatialSpec, *, reference_scaled_rate_limit:float=10000) -> tuple[dict,dict]:
    rows=[];raw={}
    for n in spec.nx_values:
        grid=Grid((0,0,spec.length_x_m,spec.length_y_m),n,spec.ny)
        kappa=smooth_coefficient(grid,spec.base_diffusivity,spec.shape_b)
        initial=smooth_exact_cell_average(grid,0,spec)
        operator=diffusion_matrix(grid,kappa,"zero_flux")
        scaled_rate=spec.duration_s*float(np.max(-operator.diagonal()))
        if not np.isfinite(scaled_rate) or scaled_rate>reference_scaled_rate_limit:
            raise ValueError("Spatial matrix-exponential reference exceeds the T*maxOutgoing work cap.")
        initial_copy=initial.copy();coefficient_copy=kappa.copy()
        fields,diagnostics=solve(grid,initial,kappa,[spec.duration_s],method="crank_nicolson",
                                 backend="scipy",dt=spec.dt_s,boundary="zero_flux")
        exact=smooth_exact_cell_average(grid,spec.duration_s,spec)
        semidiscrete=expm_multiply(spec.duration_s*operator,initial.ravel()).reshape(initial.shape)
        if not np.array_equal(initial,initial_copy) or not np.array_equal(kappa,coefficient_copy):
            raise AssertionError("Production solve modified study inputs.")
        total=errors(fields[-1],exact);time=errors(fields[-1],semidiscrete);space=errors(semidiscrete,exact)
        ratio=time["rms"]/space["rms"] if space["rms"]>0 else None
        rows.append({"nx":n,"ny":spec.ny,"grid":grid.to_dict(),"h_x_m":grid.dx,
                     "dt_s":spec.dt_s,"duration_s":spec.duration_s,"total_error":total,
                     "temporal_error":time,"spatial_error":space,"temporal_to_spatial_rms_ratio":ratio,
                     "time_error_below_ten_percent_of_spatial":ratio is not None and ratio<.1,
                     "kappa_min":float(kappa.min()),"kappa_max":float(kappa.max()),
                     "reference_admission":{"duration_times_max_outgoing":scaled_rate,
                         "limit":reference_scaled_rate_limit,"scope":"matrix-exponential stiffness/work proxy, not a wall-time guarantee"},
                     "inputs_unmodified":True,"diagnostics":diagnostics})
        for name,value in {"kappa":kappa,"initial":initial,"numerical":fields[-1],
                           "exact_cell_average":exact,"semidiscrete":semidiscrete}.items():
            raw[f"spatial_n{n}__{name}"]=value
    _add_orders(rows,"h_x_m","spatial_error")
    return {"cases":rows,"boundary":"zero_flux","method":"crank_nicolson",
            "coefficient_sampling":"analytic positive scalar coefficient sampled at cell centres; harmonic face rule",
            "initialization":"exact finite-volume cell averages, including cosine sinc factors",
            "continuous_solution":"mean+A*exp(-D*pi^2*t/Lx^2)*(cos(theta)+b*cos(3*theta)), theta=pi*x/Lx",
            "coefficient":"D*(1+(b/3)*(1+2*cos(2*theta)))/(1+3*b*(1+2*cos(2*theta))), 0<|b|<1/9",
            "identity":"kappa*u_x=-A*exp(-D*pi^2*t/Lx^2)*(D*pi/Lx)*(sin(theta)+(b/3)*sin(3*theta))",
            "reference_hierarchy":"independent continuous analytic cell averages; expm_multiply of fixed production spatial matrix isolates time error",
            "reference_limitation":"matrix exponential is a floating-point high-accuracy reference, not interval-certified; spatial correctness is judged separately against the exact PDE",
            "scope":"x refinement only, fixed ny; y-constant solution and coefficient; no general 2-D spatial-order claim"},raw


def temporal_study(spec: TemporalSpec) -> tuple[dict,dict]:
    grid=Grid((0,0,spec.length_x_m,spec.length_y_m),spec.nx,spec.ny)
    xx,yy=np.meshgrid(grid.x/spec.length_x_m,grid.y/spec.length_y_m)
    kappa=spec.base_diffusivity*(1+spec.variation*np.sin(2*np.pi*xx)*np.cos(2*np.pi*yy))
    raw={};boundaries=[]
    for boundary in spec.boundaries:
        mode=1 if boundary=="zero_flux" else 2
        wave_x=mode*np.pi/spec.length_x_m;wave_y=mode*np.pi/spec.length_y_m
        initial=spec.mean+spec.amplitude*np.cos(mode*np.pi*xx)*np.cos(mode*np.pi*yy)*np.sinc(wave_x*grid.dx/(2*np.pi))*np.sinc(wave_y*grid.dy/(2*np.pi))
        reference_operator=independent_dense_operator(grid,kappa,boundary)
        production=diffusion_matrix(grid,kappa,boundary).toarray()
        difference=float(np.max(abs(reference_operator-production)))
        tolerance=1e-12*max(1,float(np.max(abs(reference_operator))))
        if difference>tolerance:raise AssertionError("Production operator differs from independently assembled face graph.")
        outgoing=float(np.max(-np.diag(reference_operator)))
        if outgoing<=0 or not np.isfinite(outgoing):raise ValueError("Independent temporal reference rate must be finite and positive.")
        cfl=1/outgoing
        duration=spec.duration_cfl_multiple*cfl
        if not np.isfinite([cfl,duration]).all():raise ValueError("Independent temporal reference time is not representable finitely.")
        eigenvalues,eigenvectors=eigh(reference_operator)
        reference=(eigenvectors@(np.exp(duration*eigenvalues)*(eigenvectors.T@initial.ravel()))).reshape(initial.shape)
        prefix=f"temporal_{boundary}__"
        for key,value in {"kappa":kappa,"initial":initial,"semidiscrete":reference,
                          "independent_operator":reference_operator,"eigenvalues":eigenvalues}.items():raw[prefix+key]=value
        methods={}
        for method in METHODS:
            rows=[]
            for steps in spec.step_counts:
                dt=duration/steps
                fields,diagnostics=solve(grid,initial,kappa,[duration],method=method,
                    backend="numpy" if method=="explicit_euler" else "scipy",dt=dt,boundary=boundary)
                row={"requested_steps":steps,"dt_s":dt,"error":errors(fields[-1],reference),"diagnostics":diagnostics}
                rows.append(row);raw[prefix+f"{method}_steps{steps}"]=fields[-1]
            _add_orders(rows,"dt_s","error");methods[method]=rows
        boundaries.append({"boundary":boundary,"grid":grid.to_dict(),"duration_s":duration,
            "independent_fe_positivity_limit_s":cfl,"operator_max_absolute_difference":difference,
            "independent_matrix_symmetry_residual":float(np.max(abs(reference_operator-reference_operator.T))),
            "independent_matrix_row_sum_residual":float(np.max(abs(reference_operator.sum(axis=1)))),
            "largest_eigenvalue":float(eigenvalues[-1]),"methods":methods})
    return {"boundaries":boundaries,
            "reference":"independent dense face-loop assembly + symmetric eigh exponential; not production matrix assembly",
            "scope":"fixed grid and initial cell averages; temporal error only, no continuous PDE accuracy claim",
            "coefficient":"D*(1+variation*sin(2*pi*x/Lx)*cos(2*pi*y/Ly)) sampled at cell centres",
            "periodic_two_cell_rule":"each periodic face is added separately, including two distinct faces for a two-cell axis"},raw


def _dirichlet_validation_system(matrix,grid:Grid,kappa,left,right):
    result=np.asarray(matrix,dtype=float).copy();rhs=np.zeros((grid.ny,grid.nx))
    for y in range(grid.ny):
        for x,value in ((0,left),(grid.nx-1,right)):
            weight=2*kappa[y,x]/grid.dx**2;index=y*grid.nx+x
            result[index,index]-=weight;rhs[y,x]+=weight*value
    return result,rhs


def layered_fluxes(values,grid,kappa,left,right,average="harmonic"):
    if average not in {"harmonic","arithmetic"}:raise ValueError("Unknown validation face average.")
    x=np.zeros((grid.ny,grid.nx+1));y=np.zeros((grid.ny+1,grid.nx))
    x[:,0]=-kappa[:,0]*(values[:,0]-left)/(grid.dx/2)
    x[:,-1]=-kappa[:,-1]*(right-values[:,-1])/(grid.dx/2)
    if average=="harmonic":
        kx=2/(1/kappa[:,:-1]+1/kappa[:,1:]);ky=2/(1/kappa[:-1,:]+1/kappa[1:,:])
    else:kx=(kappa[:,:-1]+kappa[:,1:])/2;ky=(kappa[:-1,:]+kappa[1:,:])/2
    x[:,1:-1]=-kx*np.diff(values,axis=1)/grid.dx
    y[1:-1,:]=-ky*np.diff(values,axis=0)/grid.dy
    return x,y


def layered_study(spec:LayeredSpec)->tuple[dict,dict]:
    grid=Grid((0,0,spec.length_x_m,spec.length_y_m),spec.nx,spec.ny)
    interface=spec.interface_fraction*spec.length_x_m
    coefficient=np.broadcast_to(np.where(grid.x<interface,spec.kappa_left,spec.kappa_right),(spec.ny,spec.nx)).copy()
    resistance=interface/spec.kappa_left+(spec.length_x_m-interface)/spec.kappa_right
    flux=(spec.concentration_left-spec.concentration_right)/resistance
    x_resistance=np.minimum(grid.x,interface)/spec.kappa_left+np.maximum(grid.x-interface,0)/spec.kappa_right
    exact=np.broadcast_to(spec.concentration_left-flux*x_resistance,(spec.ny,spec.nx)).copy()
    production=diffusion_matrix(grid,coefficient,"zero_flux").toarray()
    harmonic,rhs=_dirichlet_validation_system(production,grid,coefficient,spec.concentration_left,spec.concentration_right)
    arithmetic,_=_dirichlet_validation_system(independent_dense_operator(grid,coefficient,face_average="arithmetic"),
                                             grid,coefficient,spec.concentration_left,spec.concentration_right)
    raw={"layered__kappa":coefficient,"layered__exact_cell_average":exact,
         "layered__production_closed_operator":production,"layered__harmonic_dirichlet_operator":harmonic,
         "layered__arithmetic_dirichlet_operator":arithmetic,"layered__rhs":rhs}
    results={}
    for name,matrix in (("harmonic",harmonic),("arithmetic",arithmetic)):
        field=spsolve(sparse.csc_matrix(-matrix),rhs.ravel()).reshape(exact.shape)
        fx,fy=layered_fluxes(field,grid,coefficient,spec.concentration_left,spec.concentration_right,name)
        residual=matrix@field.ravel()+rhs.ravel()
        results[name]={"error":errors(field,exact),"flux_x_max_error":float(np.max(abs(fx-flux))),
            "flux_x_spread":float(np.ptp(fx)),"flux_x_mean":float(fx.mean()),
            "flux_y_max_absolute":float(np.max(abs(fy))),"linear_residual_linf":float(np.max(abs(residual))),
            "relative_linear_residual":float(np.linalg.norm(residual)/max(1,np.linalg.norm(rhs)))}
        raw[f"layered__{name}"]=field;raw[f"layered__{name}_flux_x"]=fx;raw[f"layered__{name}_flux_y"]=fy
    return {"grid":grid.to_dict(),"interface_x_m":interface,"series_resistance":resistance,
            "analytic_constant_flux":flux,"results":results,
            "reference":"piecewise linear analytic steady solution with serial resistances; centre values equal exact cell averages on aligned layers",
            "boundary_scope":"validation-only left/right Dirichlet half-cell terms added to a COPY of the production closed operator; top/bottom zero flux; production boundary API unchanged",
            "counterexample_scope":"arithmetic face means solve a different discrete resistance model; retained as an explicit contrast, never production output"},raw


def run_study(config:dict|Study)->tuple[dict,dict]:
    spec=config if isinstance(config,Study) else Study.model_validate(config)
    admission=check_budget(spec)
    spatial,raw=spatial_study(spec.spatial,reference_scaled_rate_limit=spec.budget.max_reference_scaled_rate)
    temporal,arrays=temporal_study(spec.temporal);raw.update(arrays)
    layered,arrays=layered_study(spec.layered);raw.update(arrays)
    expected=expected_raw_shapes(spec)
    if set(raw)!=set(expected) or any(raw[k].shape!=shape for k,shape in expected.items()):
        raise AssertionError("Generated raw-array schema does not match the preflight inventory.")
    report={"schema_version":1,"config":spec.model_dump(mode="json"),
            "configuration_id":content_id(spec.model_dump(mode="json")),"admission":admission,
            "spatial":spatial,"temporal":temporal,"layered":layered,
            "raw_array_shapes":{key:list(value) for key,value in expected.items()},
            "raw_array_dtype":"float64",
            "units":{"distance":"metres","time":"seconds","diffusivity":"m^2/s",
                     "concentration":"synthetic relative concentration","face_flux":"relative concentration * m/s"},
            "limitations":["Spatial study refines x for a y-independent mode; it does not certify arbitrary 2-D spatial convergence.",
                "Temporal order is measured relative to the fixed discrete matrix exponential, independently assembled on a small grid.",
                "Manufactured analytic references and floating-point matrix exponentials are not outward-rounded mathematical certificates.",
                "Half-cell Dirichlet conditions are used only by the validation system; production variable diffusion remains closed/periodic.",
                "Positive scalar cell-centred coefficients only; no tensor anisotropy, reactions, source, variable transport or observational calibration.",
                "No performance improvement claim. Solver timings are diagnostics; array/update budgets do not bound peak factor/eigensolver memory or wall time."]}
    return report,raw


def plot_report(report:dict,output:Path)->None:
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    rows=report["spatial"]["cases"]
    for key,label in (("spatial_error","Spatial vs exact cell averages"),("temporal_error","CN vs matrix exponential")):
        axes[0].loglog([r["h_x_m"] for r in rows],[max(r[key]["rms"],1e-18) for r in rows],"o-",label=label)
    axes[0].set(xlabel="x grid spacing (m)",ylabel="RMS concentration error",title="Smooth variable coefficient: separate errors")
    spatial_ticks=[r["h_x_m"] for r in rows]
    axes[0].set_xticks(spatial_ticks,labels=[f"{value:.3g}" for value in spatial_ticks])
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(fontsize=7)
    for boundary in report["temporal"]["boundaries"]:
        for method,rows in boundary["methods"].items():
            axes[1].loglog([r["dt_s"] for r in rows],[max(r["error"]["rms"],1e-18) for r in rows],
                           "o-" if boundary["boundary"]=="zero_flux" else "s--",label=f"{method} / {boundary['boundary']}")
    axes[1].set(xlabel="Time step (s)",ylabel="RMS error vs independent exp",title="Fixed-grid temporal convergence")
    temporal_ticks=[r["dt_s"] for r in report["temporal"]["boundaries"][0]["methods"]["explicit_euler"]]
    axes[1].set_xticks(temporal_ticks,labels=[f"{value:.3g}" for value in temporal_ticks])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].legend(fontsize=6)
    layer=report["layered"]
    values=[layer["results"][name]["flux_x_mean"] for name in ("harmonic","arithmetic")]
    bars=axes[2].bar(["Harmonic faces","Arithmetic faces"],values,color=["#397b9e","#c7774d"])
    reference=layer["analytic_constant_flux"]
    axes[2].bar_label(bars,labels=[f"{value:.6g}\nrelative error {100*abs(value-reference)/abs(reference):.3g}%" for value in values],
                      padding=4,fontsize=7)
    axes[2].margins(y=.25)
    axes[2].axhline(layer["analytic_constant_flux"],color="black",ls="--",label="Analytic serial-resistance flux")
    axes[2].set(ylabel="Mean x flux (relative concentration m/s)",title="Layer interface: steady validation only")
    axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(axis="y",alpha=.2)
    fig.savefig(output,dpi=160);plt.close(fig)


def _json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")


def verify_bundle(output:Path)->dict:
    root=Path(output);manifest=json.loads((root/"manifest.json").read_text())
    if manifest.get("schema_version")!=1 or manifest.get("status")!="completed":
        raise ValueError("Unsupported or incomplete variable study manifest.")
    files=manifest.get("files",{})
    required={"config.json","report.json","raw_fields.npz","convergence.csv","variable_evidence.png","REPORT.md"}
    if not isinstance(files,dict) or not required<=set(files):raise ValueError("Missing required variable study artifacts.")
    for name,info in files.items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:raise ValueError("Unsafe artifact path.")
        path=root/name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):raise ValueError("Missing or unsafe artifact.")
        if path.stat().st_size!=info["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest()!=info["sha256"]:
            raise ValueError("Artifact checksum mismatch: "+name)
    actual={str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual!=set(files)|{"manifest.json"}:raise ValueError("Bundle has unlisted files.")
    spec=Study.model_validate_json((root/"config.json").read_text())
    report=json.loads((root/"report.json").read_text())
    if report.get("schema_version")!=1 or report.get("configuration_id")!=content_id(spec.model_dump(mode="json")):
        raise ValueError("Report schema or configuration identity mismatch.")
    if report.get("config")!=spec.model_dump(mode="json"):raise ValueError("Report configuration differs from archived configuration.")
    if report.get("admission")!=check_budget(spec):raise ValueError("Report admission estimate differs from configuration.")
    expected=expected_raw_shapes(spec)
    if report.get("raw_array_dtype")!="float64":raise ValueError("Unexpected raw-array dtype declaration.")
    if report.get("raw_array_shapes")!={k:list(v) for k,v in expected.items()}:raise ValueError("Report raw-array inventory differs from configuration.")
    if [c.get("nx") for c in report.get("spatial",{}).get("cases",[])]!=list(spec.spatial.nx_values):
        raise ValueError("Spatial cases do not match configuration.")
    boundaries=report.get("temporal",{}).get("boundaries",[])
    if [b.get("boundary") for b in boundaries]!=list(spec.temporal.boundaries):raise ValueError("Temporal boundary cases do not match configuration.")
    for boundary in boundaries:
        if set(boundary.get("methods",{}))!=set(METHODS):raise ValueError("Temporal methods do not match the protocol.")
        for rows in boundary["methods"].values():
            if [r.get("requested_steps") for r in rows]!=list(spec.temporal.step_counts):raise ValueError("Temporal steps do not match configuration.")
    sources=report.get("source_snapshot_sha256",{})
    listed={name.removeprefix("source_snapshot/") for name in files if name.startswith("source_snapshot/")}
    required_sources={"backend/app/diffusion.py","backend/app/numerics/coefficients.py","backend/app/numerics/operators.py",
        "backend/app/numerics/solver.py","backend/app/research/experiments.py","backend/app/research/problems.py",
        "scripts/run_variable_diffusion_experiments.py","requirements.lock.txt"}
    if not isinstance(sources,dict) or set(sources)!=listed or not required_sources<=listed:
        raise ValueError("Missing or inconsistent source snapshot inventory.")
    if any(files["source_snapshot/"+name]["sha256"]!=digest for name,digest in sources.items()):
        raise ValueError("Source snapshot hashes disagree with manifest.")
    with zipfile.ZipFile(root/"raw_fields.npz") as archive:
        names=archive.namelist()
        if len(names)!=len(expected) or set(names)!={key+".npy" for key in expected}:
            raise ValueError("Raw archive has duplicate, missing or unexpected NPY members.")
        if sum(item.file_size for item in archive.infolist())>spec.budget.max_array_bytes+1_000_000:
            raise ValueError("Expanded raw archive exceeds its declared array budget.")
    for key,shape in expected.items():
        inspect_npz_array(root/"raw_fields.npz",key,shape,max_bytes=spec.budget.max_array_bytes,float64=True)
    with np.load(root/"raw_fields.npz",allow_pickle=False) as raw:
        if set(raw.files)!=set(expected):raise ValueError("Missing or unexpected raw arrays.")
        for key,shape in expected.items():
            values=raw[key]
            if values.shape!=shape:raise ValueError("Incorrect archived array shape: "+key)
            if values.dtype.kind not in "fiu" or not np.isfinite(values).all():raise ValueError("Raw arrays must be finite and real.")
            if key.endswith("__kappa") and np.any(values<=0):raise ValueError("Archived coefficient must remain positive.")
        if sum(raw[key].nbytes for key in raw.files)>spec.budget.max_array_bytes:raise ValueError("Raw arrays exceed their byte budget.")
    return {"status":"verified","files":len(files),"arrays":len(expected),
            "scope":"artifact bytes, schema, config/case identities, source inventory and raw-array layout; not mathematical accuracy"}


def _write_csv(path,report):
    rows=[]
    for row in report["spatial"]["cases"]:
        rows.append({"study":"spatial","boundary":"zero_flux","method":"crank_nicolson","nx":row["nx"],
            "step":row["h_x_m"],"rms":row["spatial_error"]["rms"],"linf":row["spatial_error"]["linf"],
            "observed_order":row["observed_rms_order"],"reference":"continuous exact cell average"})
    for boundary in report["temporal"]["boundaries"]:
        for method,points in boundary["methods"].items():
            for row in points:rows.append({"study":"temporal","boundary":boundary["boundary"],"method":method,
                "nx":boundary["grid"]["nx"],"step":row["dt_s"],"rms":row["error"]["rms"],"linf":row["error"]["linf"],
                "observed_order":row["observed_rms_order"],"reference":"independent dense matrix exponential"})
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)


def _markdown(report):
    lines=["# Variable diffusion: spatial, temporal and interface references","",
        "This bundle runs the production FE/BE/CN solver. Three distinct references separate spatial, temporal and material-interface errors. No performance or physical pollution claim is made.","",
        "| nx | Spatial RMS vs exact cell averages | CN temporal RMS | Temporal / spatial error | Observed spatial order |",
        "|---:|---:|---:|---:|---:|"]
    for row in report["spatial"]["cases"]:
        order=row["observed_rms_order"]
        ratio=row["temporal_to_spatial_rms_ratio"]
        ratio_text=f"{ratio:.6g}" if ratio is not None else "unresolved (zero represented spatial error)"
        lines.append(f"| {row['nx']} | {row['spatial_error']['rms']:.8g} | {row['temporal_error']['rms']:.8g} | {ratio_text} | {order if order is not None else 'unresolved'} |")
    lines += ["","The smooth mode solves the unforced closed-boundary PDE. Both initial and reference data are exact sinc cell averages. Only x is refined; y is constant. The fixed production-matrix exponential isolates CN time error. An unresolved ratio or large temporal error does not establish spatial order.","",
        "| Boundary | Method | Finest temporal RMS | Last observed temporal order |",
        "|---|---|---:|---:|"]
    for boundary in report["temporal"]["boundaries"]:
        for method,rows in boundary["methods"].items():
            last=rows[-1];lines.append(f"| {boundary['boundary']} | {method} | {last['error']['rms']:.8g} | {last['observed_rms_order']} |")
    lines += ["","Temporal tests hold a small rectangular grid fixed and use an independently assembled face matrix and symmetric eigendecomposition exponential. Their errors measure time integration, separately from continuous-PDE spatial error.","",
        "| Interface face rule | Solution RMS error | Mean x flux | Maximum x flux error |",
        "|---|---:|---:|---:|"]
    for name,row in report["layered"]["results"].items():
        lines.append(f"| {name} | {row['error']['rms']:.8g} | {row['flux_x_mean']:.8g} | {row['flux_x_max_error']:.8g} |")
    lines += ["",f"Analytic serial-resistance flux: {report['layered']['analytic_constant_flux']:.10g} relative-concentration m/s.","",
        "The steady interface validation adds left/right half-cell Dirichlet terms only to a COPY of the production closed matrix. The exact piecewise-linear profile and every x/y face flux are retained. Arithmetic face means are an explicitly different comparison discretization, never the production output.","",
        "Configuration, raw arrays, independent reference matrices, diagnostics, source snapshot and dependency lock are archived. Admission limits aggregate array bytes, estimated cell updates and spatial exponential T*maxOutgoing; it does not cap LU/eigensolver peak memory or wall time. Floating-point references are not interval-certified mathematical proofs.","",
        "Verify: `python -m scripts.run_variable_diffusion_experiments --verify BUNDLE`. Replot: `--replot BUNDLE` creates a fresh sibling PNG outside the sealed bundle. Replay from `source_snapshot/` with the prepared interpreter: `python -B -m scripts.run_variable_diffusion_experiments --config ../config.json --output /absolute/new/output`.","",
        "![Variable diffusion evidence](variable_evidence.png)",""]
    return "\n".join(lines)


def write_bundle(config,output:Path)->dict:
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(f"Output already exists: {output}")
    report,raw=run_study(config)
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix="."+output.name+"-",dir=output.parent))
    try:
        sources=sorted((ROOT/"backend").rglob("*.py"))+[Path(__file__).resolve(),ROOT/"scripts/__init__.py",ROOT/"requirements.lock.txt"]
        for source in sources:
            destination=stage/"source_snapshot"/source.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        report["provenance"]=provenance()
        report["source_snapshot_sha256"]={str(p.relative_to(stage/"source_snapshot")):hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted((stage/"source_snapshot").rglob("*")) if p.is_file()}
        _json(stage/"config.json",report["config"]);_json(stage/"report.json",report)
        np.savez_compressed(stage/"raw_fields.npz",**raw)
        _write_csv(stage/"convergence.csv",report);plot_report(report,stage/"variable_evidence.png")
        (stage/"REPORT.md").write_text(_markdown(report),encoding="utf-8")
        files={str(p.relative_to(stage)):{"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size}
               for p in sorted(stage.rglob("*")) if p.is_file()}
        _json(stage/"manifest.json",{"schema_version":1,"status":"completed","files":files})
        verify_bundle(stage)
        if output.exists():raise FileExistsError(f"Output already exists: {output}")
        stage.rename(output)
    finally:
        if stage.exists():shutil.rmtree(stage)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output",type=Path);action.add_argument("--verify",type=Path);action.add_argument("--replot",type=Path)
    args=parser.parse_args(argv)
    if args.verify:print(json.dumps(verify_bundle(args.verify)))
    elif args.replot:
        verify_bundle(args.replot)
        output=args.replot.parent/(args.replot.name+"-replot.png")
        if output.exists():raise FileExistsError(output)
        plot_report(json.loads((args.replot/"report.json").read_text()),output);print(str(output))
    else:
        result=write_bundle(json.loads(args.config.read_text()),args.output)
        print(json.dumps({"output":str(args.output.resolve()),"configuration_id":result["configuration_id"],
                          "arrays":result["admission"]["raw_array_count"]}))


if __name__=="__main__":main()
