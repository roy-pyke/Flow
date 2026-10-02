"""Independent references for prescribed conservative transport, sources and flux ledgers.

All concentrations are synthetic. Continuous characteristics, independent face
matrices and analytic mass integrals answer different validation questions.
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
from scipy.linalg import expm

from backend.app.diffusion import Grid
from backend.app.numerics import solve
from backend.app.numerics.forcing import PrescribedFields
from backend.app.research.experiments import provenance
from backend.app.research.problems import StrictSpec, content_id, inspect_npz_array

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/research/forced_transport_study.json"
METHODS = ("explicit_euler", "backward_euler", "crank_nicolson")
TRANSPORT_METHODS = ("advection_explicit", "imex_euler")


def _counts(values, minimum=3):
    if (not minimum <= len(values) <= 6 or any(not 8 <= n <= 4096 or n % 8 for n in values)
            or any(b <= a for a, b in zip(values, values[1:]))):
        raise ValueError("Step counts must increase strictly, be multiples of 8 in 8..4096, and contain 3..6 values.")


class PeriodicSpec(StrictSpec):
    length_x_m: float = Field(default=1, ge=.001, le=1000)
    length_y_m: float = Field(default=.25, ge=.001, le=1000)
    nx_values: tuple[int, ...] = (16, 32, 64, 128)
    ny: int = Field(default=4, ge=2, le=8)
    mean: float = Field(default=1, gt=0, le=100)
    wind_amplitude_m_s: float = Field(default=.25, gt=0, le=100)
    duration_s: float = Field(default=.8, gt=0, le=1000)
    dt_s: float = Field(default=.0005, gt=0)
    temporal_nx: int = Field(default=16, ge=4, le=64)
    step_counts: tuple[int, ...] = (32, 64, 128, 256)

    @model_validator(mode="after")
    def protocol(self):
        if (not 2 <= len(self.nx_values) <= 5 or any(not 4 <= n <= 128 for n in self.nx_values)
                or any(b <= a for a,b in zip(self.nx_values,self.nx_values[1:]))):
            raise ValueError("Periodic sizes must increase strictly with 2..5 sizes in 4..128.")
        _counts(self.step_counts)
        if not np.isfinite(self.duration_s/self.dt_s) or self.duration_s/self.dt_s > 100000:
            raise ValueError("Periodic spatial study exceeds its step-count cap.")
        # The prescribed five-knot wave reverses exactly; its largest |integral|
        # is 5*a*T/16. Limit characteristic compression and exponentials.
        if 2*np.pi/self.length_x_m*self.wind_amplitude_m_s*self.duration_s*5/16 > 5:
            raise ValueError("Characteristic compression exceeds the reference admission cap |s| <= 5.")
        return self


class SourceSpec(StrictSpec):
    length_x_m: float = Field(default=1, ge=.001, le=1000)
    length_y_m: float = Field(default=1, ge=.001, le=1000)
    nx: int = Field(default=12, ge=4, le=24)
    ny: int = Field(default=4, ge=2, le=8)
    diffusivity: float = Field(default=.03, gt=0, le=100)
    mean: float = Field(default=1, gt=0, le=100)
    amplitude: float = Field(default=.2, gt=0, le=100)
    source_mean: float = Field(default=.7, gt=0, le=100)
    source_mean_slope: float = Field(default=.3, ge=0, le=100)
    source_mode: float = Field(default=.3, gt=0, le=100)
    source_mode_slope: float = Field(default=.2, ge=0, le=100)
    duration_s: float = Field(default=.8, gt=0, le=1000)
    step_counts: tuple[int, ...] = (16, 32, 64, 128)

    @model_validator(mode="after")
    def protocol(self):
        _counts(self.step_counts)
        if self.nx*self.ny > 128:
            raise ValueError("Independent dense forced reference is limited to 128 cells.")
        if self.mean <= self.amplitude or self.source_mean <= self.source_mode or self.source_mean_slope < self.source_mode_slope:
            raise ValueError("Source protocol requires positive initial data and nonnegative prescribed source.")
        return self


class OpenSpec(StrictSpec):
    length_x_m: float = Field(default=1, ge=.001, le=1000)
    length_y_m: float = Field(default=.5, ge=.001, le=1000)
    nx: int = Field(default=16, ge=4, le=64)
    ny: int = Field(default=4, ge=2, le=8)
    diffusivity: float = Field(default=.02, ge=0, le=100)
    mean: float = Field(default=1, gt=0, le=100)
    concentration_slope: float = Field(default=.3, gt=0, le=100)
    inlet_velocity_m_s: float = Field(default=.6, gt=0, le=100)
    divergence_s_inv: float = Field(default=.4, gt=0, le=100)
    duration_s: float = Field(default=.8, gt=0, le=1000)
    step_counts: tuple[int, ...] = (32, 64, 128)

    @model_validator(mode="after")
    def protocol(self):
        _counts(self.step_counts)
        return self


class StudyBudget(StrictSpec):
    max_array_bytes: int = Field(default=128*1024**2, ge=1024, le=1024**3)
    max_cell_updates: int = Field(default=100_000_000, ge=1, le=2_000_000_000)
    max_reference_scaled_rate: float = Field(default=10000, gt=0, le=100000)


class Study(StrictSpec):
    schema_version: Literal[1] = 1
    periodic: PeriodicSpec = PeriodicSpec()
    source: SourceSpec = SourceSpec()
    open: OpenSpec = OpenSpec()
    budget: StudyBudget = StudyBudget()

    @model_validator(mode="after")
    def admission(self):
        check_budget(self)
        return self


def _forcing_shapes(prefix, nt, ny, nx):
    return {prefix+"forcing_times_s":(nt,), prefix+"velocity_x":(nt,ny,nx+1),
            prefix+"velocity_y":(nt,ny+1,nx), prefix+"source":(nt,ny,nx),
            **{prefix+"inflow_"+side:(nt,ny if side in {"left","right"} else nx)
               for side in ("left","right","bottom","top")}}


def expected_raw_shapes(spec: Study) -> dict[str, tuple[int, ...]]:
    p,s,o=spec.periodic,spec.source,spec.open; shapes={}
    for nx in p.nx_values:
        prefix=f"periodic_n{nx}__"
        shapes.update(_forcing_shapes(prefix,5,p.ny,nx))
        for key in ("numerical","exact_cell_average","semidiscrete"):
            shapes[prefix+key]=(3,p.ny,nx)
        for sign in ("positive","negative"):shapes[prefix+"operator_"+sign]=(nx,nx)
    prefix="periodic_temporal__"
    shapes.update(_forcing_shapes(prefix,5,p.ny,p.temporal_nx))
    shapes[prefix+"semidiscrete"]=(3,p.ny,p.temporal_nx)
    for sign in ("positive","negative"):shapes[prefix+"operator_"+sign]=(p.temporal_nx,p.temporal_nx)
    for count in p.step_counts:shapes[prefix+f"steps{count}"]=(3,p.ny,p.temporal_nx)
    prefix="source__";shapes.update(_forcing_shapes(prefix,2,s.ny,s.nx))
    for key in ("initial","mode_cell_average"):shapes[prefix+key]=(s.ny,s.nx)
    for key in ("semidiscrete","continuous_cell_average"):shapes[prefix+key]=(3,s.ny,s.nx)
    shapes[prefix+"independent_augmented_operator"]=(s.nx*s.ny+2,)*2
    shapes[prefix+"continuous_modal_operator"]=(3,3)
    for method in METHODS:
        for count in s.step_counts:shapes[prefix+f"{method}_steps{count}"]=(3,s.ny,s.nx)
    prefix="open__";shapes.update(_forcing_shapes(prefix,3,o.ny,o.nx))
    shapes[prefix+"exact_cell_average"]=(3,o.ny,o.nx)
    for method in TRANSPORT_METHODS:
        for count in o.step_counts:shapes[prefix+f"{method}_steps{count}"]=(3,o.ny,o.nx)
    return shapes


def check_budget(spec: Study) -> dict:
    p,s,o=spec.periodic,spec.source,spec.open
    for section in (p,s,o):
        if section.duration_s/min(section.step_counts) == 0:
            raise ValueError("Requested time step is not representable positively.")
    if np.any(np.diff(np.linspace(0,p.duration_s,5))<=0) or o.duration_s*.37 == 0:
        raise ValueError("Reference knot spacing is not representable positively.")
    raw_bytes=sum(int(np.prod(shape))*8 for shape in expected_raw_shapes(spec).values())
    updates=sum(nx*p.ny*(int(np.ceil(p.duration_s/p.dt_s))+8) for nx in p.nx_values)
    updates+=p.temporal_nx*p.ny*sum(n+8 for n in p.step_counts)
    updates+=3*s.nx*s.ny*sum(n+4 for n in s.step_counts)
    updates+=2*o.nx*o.ny*sum(n+5 for n in o.step_counts)
    periodic_rate=2*p.wind_amplitude_m_s*max(*p.nx_values,p.temporal_nx)/p.length_x_m
    source_rate=2*s.diffusivity*((s.nx/s.length_x_m)**2+(s.ny/s.length_y_m)**2)
    open_rate=(o.inlet_velocity_m_s+o.divergence_s_inv*o.length_x_m)*o.nx/o.length_x_m
    open_rate+=2*o.diffusivity*((o.nx/o.length_x_m)**2+(o.ny/o.length_y_m)**2)
    # Conservative upper bounds are computed without allocating fields or invoking solve.
    if p.dt_s*periodic_rate > .9 or p.duration_s/min(p.step_counts)*2*p.wind_amplitude_m_s*p.temporal_nx/p.length_x_m > .9:
        raise ValueError("Periodic requested step exceeds conservative global advective CFL admission.")
    if s.duration_s/min(s.step_counts)*source_rate > .9:
        raise ValueError("Source FE requested step exceeds diffusion CFL admission.")
    if o.duration_s/min(o.step_counts)*open_rate > .9:
        raise ValueError("Open requested step exceeds conservative global advective CFL admission.")
    scaled=max(p.duration_s*periodic_rate,s.duration_s*source_rate)
    if scaled>spec.budget.max_reference_scaled_rate:
        raise ValueError("Reference scaled rate exceeds the configured exponential-work admission cap.")
    if raw_bytes>spec.budget.max_array_bytes or updates>spec.budget.max_cell_updates:
        raise ValueError("Study exceeds aggregate raw-array or cell-update budget.")
    return {"raw_array_bytes":raw_bytes,"raw_array_count":len(expected_raw_shapes(spec)),
            "estimated_cell_updates_upper":updates,"reference_scaled_rate_bound":scaled,
            "max_reference_scaled_rate":spec.budget.max_reference_scaled_rate,
            "scope":"aggregate saved arrays, cell updates, reference dimension and scaled rate; excludes transient forcing copies, dense/LU workspaces, peak RSS and wall time"}


def errors(value,reference):
    difference=np.asarray(value)-np.asarray(reference)
    return {"rms":float(np.sqrt(np.mean(difference**2))),"linf":float(np.max(abs(difference)))}


def _orders(rows,step_key,error_key):
    rows[0]["observed_rms_order"]=None
    for a,b in zip(rows,rows[1:]):
        ea,eb=a[error_key]["rms"],b[error_key]["rms"]
        b["observed_rms_order"]=float(np.log(ea/eb)/np.log(a[step_key]/b[step_key])) if min(ea,eb)>0 else None


def periodic_clock(spec):
    return np.linspace(0,spec.duration_s,5),spec.wind_amplitude_m_s*np.array([1,1,-1,-1,1.])


def pwl_integral(times,values,end):
    """Independent exact trapezoid integration of the declared PWL input."""
    if not 0 <= end <= times[-1]:raise ValueError("Reference time outside prescribed interval.")
    total=0.
    for left,right,a,b in zip(times,times[1:],values,values[1:]):
        if end <= left:break
        stop=min(end,right);at_stop=a+(b-a)*(stop-left)/(right-left)
        total+=(stop-left)*(a+at_stop)/2
    return float(total)


def periodic_exact_cell_average(grid,time_s,spec):
    knots,amplitudes=periodic_clock(spec)
    s=2*np.pi/spec.length_x_m*pwl_integral(knots,amplitudes,time_s)
    theta=np.linspace(0,2*np.pi,grid.nx+1)
    inverse=2*np.arctan2(np.exp(-s/2)*np.sin(theta/2),np.exp(s/2)*np.cos(theta/2))
    inverse[0],inverse[-1]=0.,2*np.pi
    row=spec.mean*np.diff(inverse)/(2*np.pi/grid.nx)
    return np.broadcast_to(row,(grid.ny,grid.nx)).copy()


def independent_upwind_operator(nx,length,sign):
    """Reference 1-D conservative face assembly; no production stencil calls."""
    if sign not in (-1,1):raise ValueError("sign must be -1 or +1")
    matrix=np.zeros((nx,nx));dx=length/nx
    # The periodic seam velocity is exactly zero; internal faces counted once.
    for face in range(1,nx):
        velocity=sign*np.sin(2*np.pi*face/nx);left,right=face-1,face
        donor=left if velocity>=0 else right
        matrix[left,donor]-=velocity/dx;matrix[right,donor]+=velocity/dx
    return matrix


def periodic_semidiscrete(grid,output_times,spec):
    """Exact-in-time independent upwind system on each wind-sign interval.

    A(t)=abs(a(t))*A_sign commutes within one sign interval. Opposite-sign
    matrices generally do not commute; chronological products are essential.
    """
    knots,amplitudes=periodic_clock(spec)
    matrices={1:independent_upwind_operator(grid.nx,spec.length_x_m,1),
              -1:independent_upwind_operator(grid.nx,spec.length_x_m,-1)}
    events=set(float(t) for t in knots)
    for left,right,a,b in zip(knots,knots[1:],amplitudes,amplitudes[1:]):
        if a*b<0:events.add(float(left-a*(right-left)/(b-a)))
    events.update(float(t) for t in output_times)
    state=np.full(grid.nx,spec.mean);result=[];current=0.
    for target in output_times:
        for end in sorted(t for t in events if current<t<=target):
            area=pwl_integral(knots,amplitudes,end)-pwl_integral(knots,amplitudes,current)
            if area:state=expm(abs(area)*matrices[1 if area>0 else -1])@state
            current=end
        result.append(np.broadcast_to(state,(grid.ny,grid.nx)).copy())
    return np.asarray(result),matrices


def _save_forcing(raw,prefix,forcing):
    for key in ("times_s","velocity_x","velocity_y","source"):
        raw[prefix+("forcing_times_s" if key=="times_s" else key)]=np.array(getattr(forcing,key),copy=True)
    for side,value in forcing.inflow.items():raw[prefix+"inflow_"+side]=np.array(value,copy=True)


def _periodic_forcing(grid,spec):
    knots,amplitudes=periodic_clock(spec)
    row=np.sin(np.linspace(0,2*np.pi,grid.nx+1));row[0]=row[-1]=0.
    vx=amplitudes[:,None,None]*np.broadcast_to(row,(5,grid.ny,grid.nx+1))
    return PrescribedFields(grid,knots,velocity_x=vx,boundary="periodic")


def periodic_study(spec):
    rows=[];raw={};times=np.array([0,spec.duration_s/4,spec.duration_s])
    for nx in spec.nx_values:
        grid=Grid((0,0,spec.length_x_m,spec.length_y_m),nx,spec.ny)
        forcing=_periodic_forcing(grid,spec);initial=np.full((grid.ny,nx),spec.mean)
        before=initial.copy();numerical,diagnostics=solve(grid,initial,0,times,method="advection_explicit",backend="numpy",dt=spec.dt_s,boundary="periodic",forcing=forcing)
        reference,operators=periodic_semidiscrete(grid,times,spec)
        exact=np.asarray([periodic_exact_cell_average(grid,t,spec) for t in times])
        space,time=errors(reference[-1],exact[-1]),errors(numerical[-1],reference[-1])
        prefix=f"periodic_n{nx}__";_save_forcing(raw,prefix,forcing)
        raw.update({prefix+"numerical":numerical,prefix+"semidiscrete":reference,prefix+"exact_cell_average":exact,
                    prefix+"operator_positive":operators[1],prefix+"operator_negative":operators[-1]})
        rows.append({"nx":nx,"h_x_m":grid.dx,"dt_s":spec.dt_s,"spatial_error":space,"temporal_error":time,
                     "total_error":errors(numerical[-1],exact[-1]),
                     "temporal_to_spatial_rms_ratio":time["rms"]/space["rms"] if space["rms"] else None,
                     "compression_max_exact_cell_average":float(exact[1].max()),
                     "nonconservative_constant_error_at_compression":errors(initial,exact[1]),
                     "inputs_unmodified":bool(np.array_equal(before,initial)),"diagnostics":diagnostics})
    _orders(rows,"h_x_m","spatial_error")
    grid=Grid((0,0,spec.length_x_m,spec.length_y_m),spec.temporal_nx,spec.ny)
    forcing=_periodic_forcing(grid,spec);reference,operators=periodic_semidiscrete(grid,times,spec)
    prefix="periodic_temporal__";_save_forcing(raw,prefix,forcing);raw[prefix+"semidiscrete"]=reference
    raw[prefix+"operator_positive"],raw[prefix+"operator_negative"]=operators[1],operators[-1]
    temporal=[]
    for count in spec.step_counts:
        numerical,diagnostics=solve(grid,np.full((grid.ny,grid.nx),spec.mean),0,times,method="advection_explicit",backend="numpy",
                                    dt=spec.duration_s/count,boundary="periodic",forcing=forcing)
        raw[prefix+f"steps{count}"]=numerical
        temporal.append({"requested_steps":count,"dt_s":spec.duration_s/count,"error":errors(numerical[-1],reference[-1]),"diagnostics":diagnostics})
    _orders(temporal,"dt_s","error")
    return {"cases":rows,"temporal_cases":temporal,"output_times_s":times.tolist(),
            "clock_times_s":periodic_clock(spec)[0].tolist(),"wind_amplitudes_m_s":periodic_clock(spec)[1].tolist(),
            "signed_wind_integral_at_final":pwl_integral(*periodic_clock(spec),spec.duration_s),
            "continuous_reference":"c=c0/(cosh(s)+sinh(s)cos(2*pi*x/L)); s=(2*pi/L)*integral(a); inverse-characteristic differences give exact cell averages",
            "semidiscrete_reference":"independent 1-D conservative upwind face matrices; ordered exponentials split at every amplitude zero crossing and knot",
            "scope":"only x refinement of y-independent periodic compressive flow; continuous reversal returns uniform, upwind numerical diffusion does not reverse"},raw


def independent_closed_diffusion(grid,kappa):
    """Direct symmetric face-loop assembly independent of production code."""
    n=grid.nx*grid.ny;matrix=np.zeros((n,n))
    def face(a,b,rate):
        matrix[a,a]-=rate;matrix[b,b]-=rate;matrix[a,b]+=rate;matrix[b,a]+=rate
    for y in range(grid.ny):
        for x in range(grid.nx-1):face(y*grid.nx+x,y*grid.nx+x+1,kappa/grid.dx**2)
    for y in range(grid.ny-1):
        for x in range(grid.nx):face(y*grid.nx+x,(y+1)*grid.nx+x,kappa/grid.dy**2)
    return matrix


def source_mode(grid,spec):
    return np.broadcast_to(np.sinc(grid.dx/(2*spec.length_x_m))*np.cos(np.pi*grid.x/spec.length_x_m),(grid.ny,grid.nx)).copy()


def source_reference(grid,times,spec,*,continuous=False):
    mode=source_mode(grid,spec)
    if continuous:
        eigenvalue=-spec.diffusivity*(np.pi/spec.length_x_m)**2
        matrix=np.array([[eigenvalue,spec.source_mode_slope,spec.source_mode],[0,0,1],[0,0,0.]])
        amplitudes=[(expm(t*matrix)@np.array([spec.amplitude,0,1]))[0] for t in times]
        return np.asarray([spec.mean+spec.source_mean*t+spec.source_mean_slope*t*t/2+a*mode for t,a in zip(times,amplitudes)]),matrix
    n=grid.nx*grid.ny;matrix=np.zeros((n+2,n+2))
    matrix[:n,:n]=independent_closed_diffusion(grid,spec.diffusivity)
    matrix[:n,n]=(spec.source_mean_slope+spec.source_mode_slope*mode).ravel()
    matrix[:n,n+1]=(spec.source_mean+spec.source_mode*mode).ravel();matrix[n,n+1]=1
    initial=np.r_[(spec.mean+spec.amplitude*mode).ravel(),0,1]
    return np.asarray([(expm(t*matrix)@initial)[:n].reshape(grid.ny,grid.nx) for t in times]),matrix


def source_study(spec):
    grid=Grid((0,0,spec.length_x_m,spec.length_y_m),spec.nx,spec.ny)
    mode=source_mode(grid,spec);initial=spec.mean+spec.amplitude*mode
    knots=np.array([0,spec.duration_s]);times=np.array([0,spec.duration_s/2,spec.duration_s])
    forcing=PrescribedFields(grid,knots,source=np.asarray([spec.source_mean+spec.source_mean_slope*t+(spec.source_mode+spec.source_mode_slope*t)*mode for t in knots]))
    reference,matrix=source_reference(grid,times,spec);continuous,modal=source_reference(grid,times,spec,continuous=True)
    raw={"source__initial":initial.copy(),"source__mode_cell_average":mode,"source__semidiscrete":reference,
         "source__continuous_cell_average":continuous,"source__independent_augmented_operator":matrix,"source__continuous_modal_operator":modal}
    _save_forcing(raw,"source__",forcing);methods={}
    area=spec.length_x_m*spec.length_y_m
    injected=area*(spec.source_mean*spec.duration_s+spec.source_mean_slope*spec.duration_s**2/2)
    for method in METHODS:
        rows=[]
        for count in spec.step_counts:
            frames,diagnostics=solve(grid,initial,spec.diffusivity,times,method=method,backend="numpy" if method=="explicit_euler" else "scipy",
                                      dt=spec.duration_s/count,boundary="zero_flux",forcing=forcing)
            raw[f"source__{method}_steps{count}"]=frames
            rows.append({"requested_steps":count,"dt_s":spec.duration_s/count,"error":errors(frames[-1],reference[-1]),
                         "source_integral_error":diagnostics["cumulative_source_mass"]-injected,"diagnostics":diagnostics})
        _orders(rows,"dt_s","error");methods[method]=rows
    return {"grid":grid.to_dict(),"methods":methods,"output_times_s":times.tolist(),"analytic_injected_mass":injected,
            "spatial_error_at_fixed_grid":errors(reference[-1],continuous[-1]),
            "reference":"independent dense closed-face diffusion matrix augmented with clock and constant; exponential integrates affine-in-time mean and cosine source exactly",
            "scope":"fixed-grid time orders; nonzero diffusion makes CN error nontrivial even though trapezoid source mass is exact for this affine source"},raw


def open_integrals(spec,time):
    area=spec.length_x_m*spec.length_y_m;int_c=spec.mean*time+spec.concentration_slope*time*time/2
    inward=spec.length_y_m*spec.inlet_velocity_m_s*int_c
    outward=spec.length_y_m*(spec.inlet_velocity_m_s+spec.divergence_s_inv*spec.length_x_m)*int_c
    source=area*(spec.concentration_slope*time+spec.divergence_s_inv*int_c)
    return {"initial_mass":area*spec.mean,"final_mass":area*(spec.mean+spec.concentration_slope*time),
            "source_mass":source,"inward_mass":inward,"outward_mass":outward,"net_outward_mass":outward-inward}


def open_study(spec):
    grid=Grid((0,0,spec.length_x_m,spec.length_y_m),spec.nx,spec.ny)
    # An interior knot deliberately does not align with the output midpoint.
    knots=spec.duration_s*np.array([0,.37,1]);times=spec.duration_s*np.array([0,.5,1])
    concentration=spec.mean+spec.concentration_slope*knots
    vx=np.broadcast_to(spec.inlet_velocity_m_s+spec.divergence_s_inv*np.linspace(0,spec.length_x_m,spec.nx+1),(3,spec.ny,spec.nx+1)).copy()
    source=np.broadcast_to((spec.concentration_slope+spec.divergence_s_inv*concentration)[:,None,None],(3,spec.ny,spec.nx)).copy()
    inflow={"left":np.broadcast_to(concentration[:,None],(3,spec.ny)).copy()}
    forcing=PrescribedFields(grid,knots,velocity_x=vx,source=source,inflow=inflow,boundary="open")
    exact=np.asarray([np.full((spec.ny,spec.nx),spec.mean+spec.concentration_slope*t) for t in times])
    raw={"open__exact_cell_average":exact};_save_forcing(raw,"open__",forcing)
    integrals=open_integrals(spec,spec.duration_s);methods={}
    for method in TRANSPORT_METHODS:
        rows=[];kappa=spec.diffusivity
        for count in spec.step_counts:
            frames,diagnostics=solve(grid,exact[0],kappa,times,method=method,backend="numpy" if method=="advection_explicit" else "numpy_scipy",
                                      dt=spec.duration_s/count,boundary="open",forcing=forcing)
            raw[f"open__{method}_steps{count}"]=frames
            actual={key:diagnostics[name] for key,name in (("source_mass","cumulative_source_mass"),("outward_mass","cumulative_boundary_outward_mass"),("inward_mass","cumulative_boundary_inward_mass"))}
            mass=grid.dx*grid.dy*float(frames[-1].sum())
            independently_recomputed_balance=mass-integrals["initial_mass"]+actual["outward_mass"]-actual["inward_mass"]-actual["source_mass"]
            # For this affine uniform solution each stage integrand is affine.
            # Sum(dt_i^2) from the actual internal-step histogram determines the
            # left-rule error independently of solver source/flux evaluations.
            sum_squared_steps=sum(float(dt)**2*number for dt,number in diagnostics["actual_dt_histogram"].items())
            half_slope_step2=.5*spec.concentration_slope*sum_squared_steps
            predicted_errors={"source_mass":-spec.length_x_m*spec.length_y_m*spec.divergence_s_inv*half_slope_step2,
                "inward_mass":-spec.length_y_m*spec.inlet_velocity_m_s*half_slope_step2,
                "outward_mass":-spec.length_y_m*(spec.inlet_velocity_m_s+spec.divergence_s_inv*spec.length_x_m)*half_slope_step2}
            rows.append({"requested_steps":count,"dt_s":spec.duration_s/count,"diffusivity":kappa,"field_error":errors(frames,exact),
                "integral_errors":{key:actual[key]-integrals[key] for key in actual},
                "predicted_left_rule_errors":predicted_errors,"sum_actual_dt_squared":sum_squared_steps,
                "independently_recomputed_balance":independently_recomputed_balance,"diagnostics":diagnostics})
        methods[method]=rows
    return {"grid":grid.to_dict(),"methods":methods,"output_times_s":times.tolist(),"analytic_integrals":integrals,
            "continuous_reference":"u_x=v0+a*x; c=c0+b*t; q=b+a*c; left inflow=c; right outflow; zero exterior diffusive flux",
            "scope":"affine uniform field is reproduced to rounding, but individual left-rule source/in/out integrals have first-order errors; ledger closure is a separate algebraic property"},raw


def run_study(config):
    spec=config if isinstance(config,Study) else Study.model_validate(config)
    periodic,raw=periodic_study(spec.periodic);source,arrays=source_study(spec.source);raw.update(arrays)
    opened,arrays=open_study(spec.open);raw.update(arrays)
    expected=expected_raw_shapes(spec)
    if set(raw)!=set(expected) or any(raw[k].shape!=shape or not np.isfinite(raw[k]).all() for k,shape in expected.items()):
        raise AssertionError("Generated raw data do not match finite array schema.")
    report={"schema_version":1,"config":spec.model_dump(mode="json"),"configuration_id":content_id(spec.model_dump(mode="json")),
            "admission":check_budget(spec),"periodic":periodic,"source":source,"open":opened,
            "raw_array_shapes":{key:list(value) for key,value in expected.items()},"raw_array_dtype":"float64",
            "units":{"distance":"m","time":"s","wind":"m/s","diffusivity":"m^2/s","concentration":"synthetic relative concentration","source":"relative concentration/s","mass":"relative concentration*m^2"},
            "limitations":["Synthetic manufactured references, not measured pollution or observational calibration.",
                "Only y-independent periodic spatial refinement is measured; no general 2-D spatial convergence claim.",
                "Piecewise-linear prescribed inputs are the model. No inference about unresolved real winds, impulses, or discontinuous source signals.",
                "Conservative -div(u*c) differs from passive -u dot grad(c) for divergent wind; compression can increase the maximum and variance.",
                "Temporal references hold the spatial discretization fixed; characteristic cell averages separately measure spatial error.",
                "Open boundaries prescribe incoming advective concentration and zero exterior diffusive flux, not diffusion Dirichlet conditions.",
                "Ledger closure uses actual numerical stage quadrature and does not certify accurate individual physical time integrals.",
                "All references are floating-point, not outward-rounded mathematical certificates; no runtime or peak-memory guarantee."]}
    return report,raw


def plot_report(report,output):
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    ax=axes[0,0];rows=report["periodic"]["cases"]
    for key,label in (("spatial_error","Upwind space vs characteristics"),("temporal_error","FE time vs independent exponential")):
        ax.loglog([r["h_x_m"] for r in rows],[max(r[key]["rms"],1e-18) for r in rows],"o-",label=label)
    ax.set(xlabel="x grid spacing (m)",ylabel="Final RMS error",title="Periodic compression and reversal")
    ax.legend(fontsize=8)
    ax=axes[0,1]
    for method,rows in report["source"]["methods"].items():
        ax.loglog([r["dt_s"] for r in rows],[max(r["error"]["rms"],1e-18) for r in rows],"o-",label=method)
    rows=report["periodic"]["temporal_cases"]
    ax.loglog([r["dt_s"] for r in rows],[max(r["error"]["rms"],1e-18) for r in rows],"s--",label="prescribed upwind / FE")
    ax.set(xlabel="Requested step (s)",ylabel="RMS error vs fixed-grid exponential",title="Time errors: source and variable wind")
    ax.legend(fontsize=8)
    ax=axes[1,0]
    rows=report["open"]["methods"]["imex_euler"]
    for key,label in (("source_mass","Source integral"),("inward_mass","Inflow integral"),("outward_mass","Outflow integral")):
        ax.loglog([r["dt_s"] for r in rows],[max(abs(r["integral_errors"][key]),1e-18) for r in rows],"o-",label=label)
    ax.loglog([r["dt_s"] for r in rows],[max(abs(r["independently_recomputed_balance"]),1e-18) for r in rows],"s--",label="Mass-balance residual")
    ax.set(xlabel="Requested IMEX step (s)",ylabel="Absolute mass error",title="Accurate ledger does not imply exact integrals")
    ax.legend(fontsize=8)
    ax=axes[1,1]
    row=report["periodic"]["cases"][-1];d=row["diagnostics"];times=report["periodic"]["output_times_s"]
    ax.plot(times,d["max_history"],"o-",label="Numerical maximum")
    ax.plot(times,d["min_history"],"o-",label="Numerical minimum")
    ax.axhline(report["config"]["periodic"]["mean"],color="black",ls=":",label="Initial uniform concentration")
    ax.set(xlabel="Output time (s)",ylabel="Concentration",title="Divergent wind changes a uniform field")
    ax.legend(fontsize=8)
    ticks = [([row["h_x_m"] for row in report["periodic"]["cases"]], axes[0, 0]),
             (sorted({row["dt_s"] for rows in report["source"]["methods"].values() for row in rows}
                     | {row["dt_s"] for row in report["periodic"]["temporal_cases"]}), axes[0, 1]),
             ([row["dt_s"] for row in report["open"]["methods"]["imex_euler"]], axes[1, 0])]
    for values, axis in ticks:
        axis.set_xticks(values, labels=[format(value, ".5g") for value in values], fontsize=9)
        axis.xaxis.set_minor_formatter(NullFormatter())
    for ax in axes.ravel():ax.grid(alpha=.2)
    fig.savefig(output,dpi=160);plt.close(fig)


def _write_csv(path,report):
    rows=[]
    def append(study,method,step,error,order):
        rows.append({"study":study,"method":method,"step":step,"rms":error["rms"],"linf":error["linf"],"observed_order":order})
    for row in report["periodic"]["cases"]:append("periodic_spatial","upwind_semidiscrete",row["h_x_m"],row["spatial_error"],row["observed_rms_order"])
    for row in report["periodic"]["temporal_cases"]:append("periodic_temporal","advection_explicit",row["dt_s"],row["error"],row["observed_rms_order"])
    for method,points in report["source"]["methods"].items():
        for row in points:append("source_temporal",method,row["dt_s"],row["error"],row["observed_rms_order"])
    for method,points in report["open"]["methods"].items():
        for row in points:append("open_uniform_field",method,row["dt_s"],row["field_error"],None)
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)


def _markdown(report):
    lines=["# Prescribed transport, time-dependent sources and mass ledgers","",
        "Three independent reference families exercise the production solver. All fields are synthetic; there is no measured pollution or performance claim.","",
        "## Periodic conservative compression and reversal","",
        "The prescribed face velocity is u=a(t) sin(2 pi x/L), with a five-knot piecewise-linear amplitude that changes sign. The conservative equation is c_t=-div(u c). Starting from uniform c0, its exact density is c0/[cosh(s)+sinh(s) cos(2 pi x/L)], where s=(2 pi/L) integral(a). Inverse-characteristic angle differences give exact finite-volume cell averages. The input has zero final signed integral, so the continuous final density returns to uniform.","",
        "A separately assembled one-dimensional upwind matrix is advanced by ordered dense exponentials, split at every amplitude zero crossing. Matrices commute within a fixed-sign interval, not generally across sign changes. This fixed-grid reference isolates time error; its difference from the characteristic solution measures spatial error. Upwind diffusion does not reverse when the velocity reverses.","",
        "| nx | Spatial RMS | Time RMS | Time / space | Spatial order |","|---:|---:|---:|---:|---:|"]
    for row in report["periodic"]["cases"]:
        ratio=row["temporal_to_spatial_rms_ratio"]
        lines.append(f"| {row['nx']} | {row['spatial_error']['rms']:.8g} | {row['temporal_error']['rms']:.8g} | {ratio if ratio is not None else 'unresolved'} | {row['observed_rms_order']} |")
    lines += ["","Only x is refined and every field is constant in y. A constant-field nonconservative advection reference is deliberately retained as an error diagnostic at compression; it solves a different PDE. General divergent winds need not preserve maxima or variance.","",
        "## Time-dependent source with closed diffusion","",
        "The source is affine in time with a uniform mean and one exact cell-average cosine mode. A separately assembled closed diffusion matrix is augmented with clock and constant states; its exponential gives the fixed-grid reference. A continuous cosine-mode reference separately exposes spatial error. Nonzero diffusion keeps Crank–Nicolson time error nontrivial, although trapezoid source mass is exact for this affine source.","",
        "| Method | Finest RMS | Last time order | Source mass error |","|---|---:|---:|---:|"]
    for method,rows in report["source"]["methods"].items():
        row=rows[-1];lines.append(f"| {method} | {row['error']['rms']:.8g} | {row['observed_rms_order']} | {row['source_integral_error']:.8g} |")
    row=report["periodic"]["temporal_cases"][-1]
    lines += ["",f"Prescribed-wind FE last time order: {row['observed_rms_order']}; final RMS {row['error']['rms']:.8g}.","",
        "## Open-boundary manufactured solution and complete mass ledger","",
        "Choose u=v0+a*x, c=c0+b*t, q=b+a*c, and left incoming concentration c. The right boundary is outflow and exterior diffusive flux is zero. This uniform spatial solution is exact for FE transport and IMEX to rounding, including an interior forcing knot at 0.37T. Both methods use the same configured scalar diffusion, which annihilates this uniform solution.","",
        "The continuous source, inflow and outflow integrals are evaluated analytically. The actual numerical stage ledger satisfies R=M-M0+out-in-source. Its three individual left-rule integrals are nevertheless first-order approximations. For each affine integrand, the independent predicted error is minus one half its slope times sum(actual_dt squared); the saved histogram includes knot/output split steps.","",
        "| Method | Finest field Linf | Source integral error | Inflow integral error | Outflow integral error | Balance residual |","|---|---:|---:|---:|---:|---:|"]
    for method,rows in report["open"]["methods"].items():
        row=rows[-1];e=row["integral_errors"]
        lines.append(f"| {method} | {row['field_error']['linf']:.8g} | {e['source_mass']:.8g} | {e['inward_mass']:.8g} | {e['outward_mass']:.8g} | {row['independently_recomputed_balance']:.8g} |")
    lines += ["","## Archive and scope","",
        "The archive contains normalized configuration, all input forcing arrays, numerical and independent reference fields, reference matrices, every solver diagnostic, CSV, this report, the figure, the complete Python source snapshot and dependency lock. Admission bounds saved array bytes, estimated cell updates, reference dimensions and scaled rates. It is not a process-memory or time limit. Verification checks bytes, inventories, schema and identities; it does not rerun or mathematically certify the results.","",
        "Verify: `python -m scripts.run_forced_transport_experiments --verify BUNDLE`. Replot: `--replot BUNDLE` writes a fresh sibling PNG outside the sealed bundle. Replay inside `source_snapshot/` using the prepared interpreter: `python -B -m scripts.run_forced_transport_experiments --config ../config.json --output /absolute/new/output`.","",
        *["- "+line for line in report["limitations"]],"","![Forced transport evidence](forced_evidence.png)",""]
    return "\n".join(lines)


def _json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")


def _verify_report_clock(report,spec):
    def number(value,minimum=None,strict=False):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
            raise ValueError("Report scalar must be a finite number.")
        if minimum is not None and (value<=minimum if strict else value<minimum):
            raise ValueError("Report scalar lies outside its structural range.")
    def error(value):
        if not isinstance(value,dict):raise ValueError("Missing error metric mapping.")
        for key in ("rms","linf"):number(value.get(key),0)
    for section,fraction in (("periodic",.25),("source",.5),("open",.5)):
        duration=getattr(spec,section).duration_s
        expected=[0.,duration*fraction,duration]
        item=report.get(section,{})
        if item.get("output_times_s")!=expected:
            raise ValueError("Report output clock differs from the fixed study protocol.")
        records=(item["cases"]+item["temporal_cases"] if section=="periodic"
                 else [row for rows in item["methods"].values() for row in rows])
        for index,row in enumerate(records):
            number(row.get("dt_s"),0,strict=True)
            if section=="periodic" and index<len(item["cases"]):
                number(row.get("h_x_m"),0,strict=True)
                for key in ("spatial_error","temporal_error","total_error"):error(row.get(key))
            elif section in {"periodic","source"}:error(row.get("error"))
            else:
                error(row.get("field_error"))
                integrals=row.get("integral_errors")
                if not isinstance(integrals,dict):raise ValueError("Missing open integral-error mapping.")
                for key in ("source_mass","outward_mass","inward_mass"):number(integrals.get(key))
                number(row.get("independently_recomputed_balance"))
            diagnostics=row.get("diagnostics",{})
            if diagnostics.get("output_times_s")!=expected:
                raise ValueError("Diagnostic output clock differs from the study protocol.")
            for key in ("mass_history","min_history","max_history","mass_balance_history",
                        "cumulative_source_history","cumulative_boundary_outward_history","cumulative_boundary_inward_history"):
                values=diagnostics.get(key)
                if not isinstance(values,list) or len(values)!=3:
                    raise ValueError("Missing or malformed diagnostic output history.")
                for value in values:number(value)


def verify_bundle(output:Path)->dict:
    root=Path(output);manifest=json.loads((root/"manifest.json").read_text())
    if manifest.get("schema_version")!=1 or manifest.get("status")!="completed":
        raise ValueError("Unsupported or incomplete forced study manifest.")
    files=manifest.get("files",{})
    required={"config.json","report.json","raw_fields.npz","convergence.csv","forced_evidence.png","REPORT.md"}
    if not isinstance(files,dict) or not required<=set(files):raise ValueError("Missing required forced study artifacts.")
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
    if [c.get("nx") for c in report.get("periodic",{}).get("cases",[])]!=list(spec.periodic.nx_values):
        raise ValueError("Periodic spatial cases do not match configuration.")
    if [c.get("requested_steps") for c in report.get("periodic",{}).get("temporal_cases",[])]!=list(spec.periodic.step_counts):
        raise ValueError("Periodic temporal cases do not match configuration.")
    for section,methods in (("source",METHODS),("open",TRANSPORT_METHODS)):
        items=report.get(section,{}).get("methods",{})
        if set(items)!=set(methods):raise ValueError("Methods do not match the study protocol.")
        for rows in items.values():
            if [r.get("requested_steps") for r in rows]!=list(getattr(spec,section).step_counts):
                raise ValueError("Temporal steps do not match configuration.")
    _verify_report_clock(report,spec)
    sources=report.get("source_snapshot_sha256",{})
    listed={name.removeprefix("source_snapshot/") for name in files if name.startswith("source_snapshot/")}
    required_sources={"backend/__init__.py","backend/app/__init__.py","backend/app/numerics/__init__.py",
        "backend/app/research/__init__.py","scripts/__init__.py","backend/app/diffusion.py","backend/app/numerics/coefficients.py","backend/app/numerics/operators.py",
        "backend/app/numerics/solver.py","backend/app/numerics/forced_solver.py","backend/app/numerics/forcing.py","backend/app/research/experiments.py","backend/app/research/problems.py",
        "scripts/run_forced_transport_experiments.py","requirements.lock.txt"}
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
            if key.endswith("__forcing_times_s"):
                clock=(periodic_clock(spec.periodic)[0] if key.startswith("periodic_") else
                       np.array([0.,spec.source.duration_s]) if key.startswith("source__") else
                       spec.open.duration_s*np.array([0.,.37,1.]))
                if not np.array_equal(values,clock):raise ValueError("Archived forcing clock differs from the fixed protocol.")
        if sum(raw[key].nbytes for key in raw.files)>spec.budget.max_array_bytes:raise ValueError("Raw arrays exceed their byte budget.")
    return {"status":"verified","files":len(files),"arrays":len(expected),
            "scope":"artifact bytes, schema, config/case identities, source inventory and raw-array layout; not mathematical accuracy"}


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
        _write_csv(stage/"convergence.csv",report);plot_report(report,stage/"forced_evidence.png")
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
