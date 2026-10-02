"""Reproducible time-aware trajectory and finite time-expanded routing study.

PDE output spacing, internal solver dt, observation quadrature and search-time
rounding are distinct controls. Analytic references never freeze a moving trip.
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
import numpy as np
from numpy.polynomial import Polynomial
from pydantic import Field, model_validator

from backend.app.diffusion import Grid
from backend.app.numerics import solve
from backend.app.observations import build_edge_observer, build_path_trajectory, build_stationary_trajectory
from backend.app.research.decision import corridor_network, enumerate_simple_paths, path_costs
from backend.app.research.dynamic_routing import solve_time_expanded
from backend.app.research.experiments import provenance
from backend.app.research.problems import StrictSpec, content_id
from backend.app.research.travel import TimeDependentExposure, trajectory_from_schedule
from backend.app.routing import RoadNetwork
from backend.app.temporal_observations import build_trajectory_observer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/research/temporal_study.json"


class Case(StrictSpec):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    kappa: float = Field(ge=0, le=100)


class Study(StrictSpec):
    schema_version: Literal[1] = 1
    bounds_m: tuple[float, float, float, float] = (0, 0, 10, 10)
    nx: int = Field(default=40, ge=8, le=128)
    ny: int = Field(default=40, ge=8, le=128)
    speed_mps: float = Field(default=1.4, gt=0)
    departure_time_s: float = Field(default=0, ge=0)
    horizon_time_s: float = Field(default=12, gt=0)
    output_interval_s: float = Field(default=.125, gt=0)
    solver_dt_s: float = Field(default=.005, gt=0)
    lambda_weight: float = Field(default=1, ge=0)
    mean: float = Field(default=1, ge=0)
    amplitude: float = Field(default=.95, ge=0)
    cases: tuple[Case, ...]
    search_time_steps_s: tuple[float, ...] = (1, .5, .25, .125)

    @model_validator(mode="after")
    def protocol(self):
        Grid(self.bounds_m, self.nx, self.ny)
        if not self.cases or len(self.cases)>12 or len({c.id for c in self.cases})!=len(self.cases):
            raise ValueError("Require 1..12 distinct study cases.")
        if self.mean < self.amplitude or self.departure_time_s >= self.horizon_time_s:
            raise ValueError("Require nonnegative analytic concentration and departure before horizon.")
        if not self.search_time_steps_s or len(self.search_time_steps_s)>8 or any(t<=0 for t in self.search_time_steps_s):
            raise ValueError("Require 1..8 positive search time steps.")
        frame_ratio = self.horizon_time_s/self.output_interval_s
        step_ratio = self.horizon_time_s/self.solver_dt_s
        if not np.isfinite([frame_ratio, step_ratio]).all() or frame_ratio>1024 or step_ratio>100_000_000:
            raise ValueError("Study exceeds frame or estimated cell-update budget.")
        frames = int(np.ceil(frame_ratio))+1
        steps = int(np.ceil(step_ratio))+frames
        # Numerical, continuous-node and exact-cell-average arrays are all
        # retained for every case. This is an archive-array admission bound,
        # not a claim about sparse-solver or process peak RSS.
        raw_bytes = frames*self.nx*self.ny*8*3*len(self.cases) + self.nx*self.ny*8 + frames*8 + 65536
        if frames>1025 or raw_bytes>128*1024**2 or steps*self.nx*self.ny*len(self.cases)>100_000_000:
            raise ValueError("Study exceeds frame or estimated cell-update budget.")
        return self


def analytic_cosine_exposure(trajectory, *, kappa, mean, amplitude, wave_number, ymin=0.0):
    """Continuous diffusion solution, integrated exactly on each linear segment.

    A stationary segment uses zero velocity and contributes waiting exposure.
    expm1 avoids cancellation for short intervals and slowly changing fields.
    """
    total = 0.0
    beta = kappa*wave_number**2
    for start, end, t0, t1 in zip(trajectory.vertex_xy[:-1], trajectory.vertex_xy[1:],
                                trajectory.vertex_times_s[:-1], trajectory.vertex_times_s[1:]):
        duration = t1-t0
        q = complex(-beta, wave_number*(end[1]-start[1])/duration)
        integral = duration if q==0 else np.expm1(q*duration)/q
        total += mean*duration + amplitude*np.real(np.exp(-beta*t0+1j*wave_number*(start[1]-ymin))*integral)
    return float(total)


def polynomial_exposure(trajectory):
    """Independent exact integral of 1+.2x+.3y+.1t+.02xyt along a trajectory."""
    result=0.0
    for a,b,t0,t1 in zip(trajectory.vertex_xy[:-1],trajectory.vertex_xy[1:],
                         trajectory.vertex_times_s[:-1],trajectory.vertex_times_s[1:]):
        dt=t1-t0
        x=Polynomial([a[0],(b[0]-a[0])/dt]);y=Polynomial([a[1],(b[1]-a[1])/dt]);t=Polynomial([t0,1])
        integral=(1+.2*x+.3*y+.1*t+.02*x*y*t).integ()
        result+=integral(dt)-integral(0)
    return float(result)


def reconstruction_study():
    grid=Grid((0,0,10,10),8,8)
    network=RoadNetwork({"bounds":list(grid.bounds),"nodes":[{"id":"a","x":2,"y":2},{"id":"b","x":8,"y":7}],
        "edges":[{"u":"a","v":"b","key":0,"coordinates":[[2,2],[5,3],[8,7]]},
                 {"u":"b","v":"a","key":0,"coordinates":[[8,7],[5,3],[2,2]]}]})
    trajectory=build_path_trajectory(network,[0,1,0],speed_mps=4,departure_time_s=.4,waits_s=[.3,.2,.4,.5])
    times=np.array([0,.7,2.1,4.8,8.0,12.0])
    xx,yy=np.meshgrid(grid.x,grid.y)
    frames=np.stack([1+.2*xx+.3*yy+.1*t+.02*xx*yy*t for t in times])
    exact=polynomial_exposure(trajectory)
    observer=build_trajectory_observer(grid,times,trajectory)
    actual=float(observer.apply(frames))
    if abs(actual-exact)>2e-12*max(1,abs(exact)):
        raise AssertionError("Independent cubic trajectory integral failed.")
    quadrature=[]
    for control,steps in [("spatial_step_m",[.8,.4,.2,.1,.05]),("time_step_s",[.8,.4,.2,.1,.05])]:
        for step in steps:
            kwargs={"spatial_step_m":100.0,"time_step_s":100.0,control:step}
            op=build_trajectory_observer(grid,times,trajectory,quadrature="trapezoid",**kwargs)
            value=float(op.apply(frames))
            quadrature.append({"control":control,"step":step,"exposure":value,"absolute_error":abs(value-exact),"observer":op.metadata})
    frame_curve=[]
    beta=.3;start=trajectory.departure_time_s;end=trajectory.arrival_time_s
    exact_time=end-start+(np.exp(-beta*start)-np.exp(-beta*end))/beta
    for intervals in [4,8,16,32,64]:
        frame_times=12*np.linspace(0,1,intervals+1)**1.2
        field=np.broadcast_to((1+np.exp(-beta*frame_times))[:,None,None],(len(frame_times),grid.ny,grid.nx)).copy()
        op=build_trajectory_observer(grid,frame_times,trajectory)
        value=float(op.apply(field))
        frame_curve.append({"intervals":intervals,"maximum_frame_gap_s":float(np.diff(frame_times).max()),
                            "exposure":value,"absolute_error":abs(value-exact_time),"frame_times_s":frame_times.tolist()})
    return {"polynomial": "1 + .2*x + .3*y + .1*t + .02*x*y*t",
        "grid":grid.to_dict(),"frame_times_s":times.tolist(),"trajectory_xy":trajectory.vertex_xy.tolist(),
        "trajectory_times_s":trajectory.vertex_times_s.tolist(),"edge_occurrences":[0,1,0],"waits_s":[.3,.2,.4,.5],
        "analytic_exposure":exact,"gauss_exposure":actual,"gauss_absolute_error":abs(actual-exact),
        "quadrature_curves":quadrature,"frame_interpolation_curve":frame_curve,
        "frame_curve_analytic_exposure":float(exact_time),
        "scope":"manufactured reconstruction tests, not PDE outputs; point samples; all coordinates inside centre rectangle"}, {
        "polynomial_frames":frames,"polynomial_times_s":times,"trajectory_xy":trajectory.vertex_xy,"trajectory_times_s":trajectory.vertex_times_s}


def label_counterexample():
    """Independent edge-integral example; not a concentration-field inference."""
    points={"s":(0,0),"a":(1,0),"b":(0,1),"t":(2,0)}
    graph={"nodes":[{"id":key,"x":xy[0],"y":xy[1]} for key,xy in points.items()],
           "edges":[{"u":u,"v":v,"key":0,"coordinates":[points[u],points[v]]}
                    for u,v in [("s","a"),("s","b"),("b","a"),("a","t")]]}
    network=RoadNetwork(graph)
    result=solve_time_expanded(network,"s","t",departure_time_s=0,horizon_time_s=5,time_step_s=1,
        speed_mps=1,lambda_weight=1,edge_exposure=lambda i,t:100.0 if i==3 and t<3 else 0.0,
        waiting_exposure=lambda node,a,b:0.0,allow_wait=False)
    if result['edge_indices']!=[1,2,3] or result['objective']!=4.0:
        raise AssertionError("Node-time-label counterexample failed.")
    return {"network":graph,"result":result,"manual_optimum":4.0,"single_node_label_result":102.0,
        "edge_integral_rule":"a->t costs exposure 100 if departed before t=3, otherwise 0; other edges and waits exposure 0",
        "travel_time_fifo":True,"explanation":"Arrival at a at t=1 has smaller accumulated cost than t=3, but much larger remaining cost. Constant travel-time FIFO does not establish dominance for exposure cost.",
        "scope":"explicit callback-defined finite graph; not generated from the PDE field"}


def run_study(config: dict | Study):
    spec=config if isinstance(config,Study) else Study.model_validate(config)
    grid=Grid(spec.bounds_m,spec.nx,spec.ny)
    network,graph=corridor_network(spec.bounds_m,upper_fraction=.8,lower_fraction=.35)
    candidates=enumerate_simple_paths(network,"s","t")
    trajectories=[build_path_trajectory(network,path,spec.speed_mps,spec.departure_time_s) for path in candidates.paths]
    if max(t.arrival_time_s for t in trajectories)>spec.horizon_time_s:
        raise ValueError("Study horizon does not cover the complete candidate journeys.")
    intervals=int(np.ceil(spec.horizon_time_s/spec.output_interval_s))
    times=np.linspace(0,spec.horizon_time_s,intervals+1)
    wave=np.pi/(grid.bounds[3]-grid.bounds[1])
    cosine=np.broadcast_to(np.cos(wave*(grid.y[:,None]-grid.bounds[1])),(grid.ny,grid.nx))
    projection=np.sinc(wave*grid.dy/(2*np.pi))
    initial=spec.mean+spec.amplitude*projection*cosine
    durations=np.array([t.arrival_time_s-t.departure_time_s for t in trajectories])
    spatial=build_edge_observer(network,grid,speed_mps=spec.speed_mps,quadrature="gauss2_grid")
    rows=[];raw={"times_s":times,"initial_cell_averages":initial};last=None
    for case in spec.cases:
        fields,diagnostics=solve(grid,initial,case.kappa,times,method="crank_nicolson",backend="scipy",dt=spec.solver_dt_s,boundary="zero_flux")
        adapter=TimeDependentExposure(network,grid,times,fields,speed_mps=spec.speed_mps)
        moving=np.array([adapter.trajectory_exposure(t) for t in trajectories])
        reference=np.array([analytic_cosine_exposure(t,kappa=case.kappa,mean=spec.mean,amplitude=spec.amplitude,wave_number=wave,ymin=grid.bounds[1]) for t in trajectories])
        # Freeze the declared reconstruction at physical departure, even when
        # departure lies between saved frames; never substitute frame zero.
        from backend.app.temporal_observations import sample_time_series
        xx,yy=np.meshgrid(grid.x,grid.y);points=np.column_stack((xx.ravel(),yy.ravel()))
        frozen=sample_time_series(grid,times,fields,points,np.full(len(points),spec.departure_time_s)).reshape(grid.ny,grid.nx)
        fixed=path_costs(candidates,spatial.apply(frozen))
        approximate_cost=durations+spec.lambda_weight*moving
        exact_cost=durations+spec.lambda_weight*reference
        frozen_cost=durations+spec.lambda_weight*fixed
        chosen=int(np.argmin(approximate_cost));fixed_chosen=int(np.argmin(frozen_cost))
        continuous=spec.mean+spec.amplitude*np.exp(-case.kappa*wave**2*times[:,None,None])*cosine
        cell_reference=spec.mean+spec.amplitude*projection*np.exp(-case.kappa*wave**2*times[:,None,None])*cosine
        row={"id":case.id,"kappa":case.kappa,"lambda_weight":spec.lambda_weight,
            "frozen_selected":fixed_chosen,"moving_selected":chosen,"reference_minimizers":np.flatnonzero(exact_cost<=exact_cost.min()+1e-10).tolist(),
            "frozen_reference_regret_s":float(exact_cost[fixed_chosen]-exact_cost.min()),"moving_reference_regret_s":float(exact_cost[chosen]-exact_cost.min()),
            "maximum_cell_average_field_error":float(np.max(abs(fields-cell_reference))),
            "paths":[{"index":i,"edge_indices":list(candidates.paths[i]),"duration_s":float(durations[i]),
                "arrival_time_s":trajectories[i].arrival_time_s,"frozen_exposure":float(fixed[i]),
                "moving_exposure":float(moving[i]),"analytic_moving_exposure":float(reference[i]),
                "moving_exposure_error":float(abs(moving[i]-reference[i])),"frozen_objective_s":float(frozen_cost[i]),
                "moving_objective_s":float(approximate_cost[i]),"analytic_objective_s":float(exact_cost[i])} for i in range(len(trajectories))],
            "solver_diagnostics":diagnostics,"field_adapter":adapter.metadata()}
        rows.append(row);raw[case.id+"__fields"]=fields;raw[case.id+"__continuous_nodes"]=continuous
        raw[case.id+"__exact_cell_averages"]=cell_reference;last=(case,adapter,fields)
    case,adapter,fields=last
    search=[]
    for step in spec.search_time_steps_s:
        result=solve_time_expanded(network,"s","t",departure_time_s=spec.departure_time_s,horizon_time_s=spec.horizon_time_s,
            time_step_s=step,speed_mps=spec.speed_mps,lambda_weight=spec.lambda_weight,edge_exposure=adapter.edge_exposure,
            waiting_exposure=adapter.waiting_exposure,allow_wait=False,max_states=10000,max_transitions=50000)
        rerated=0.0
        if result['status']=='optimal':
            full=trajectory_from_schedule(network,result)
            independent_exposure=adapter.trajectory_exposure(full)
            if not np.isclose(independent_exposure,result['exposure'],rtol=2e-12,atol=1e-12):
                raise AssertionError("Whole-journey observation differs from searched action integrals.")
            for action in result['actions']:
                if action['type']=='travel':
                    trajectory=build_path_trajectory(network,[action['edge_index']],spec.speed_mps,action['start_time_s'])
                else:
                    node=network.nodes[action['node']]
                    trajectory=build_stationary_trajectory([node['x'],node['y']],action['start_time_s'],action['end_time_s'])
                rerated+=analytic_cosine_exposure(trajectory,kappa=case.kappa,mean=spec.mean,
                                                  amplitude=spec.amplitude,wave_number=wave,ymin=grid.bounds[1])
            reference_objective=result['elapsed_time_s']+spec.lambda_weight*rerated
            full_analytic=analytic_cosine_exposure(full,kappa=case.kappa,mean=spec.mean,amplitude=spec.amplitude,wave_number=wave,ymin=grid.bounds[1])
            if not np.isclose(full_analytic,rerated,rtol=2e-12,atol=1e-12):
                raise AssertionError("Whole-journey analytic reference differs from separate actions.")
            comparison={"analytic_exposure_of_returned_schedule":rerated,
                "whole_trajectory_exposure":independent_exposure,
                "whole_trajectory_vs_actions_absolute_difference":abs(independent_exposure-result['exposure']),
                "analytic_objective_of_returned_schedule_s":reference_objective,
                "difference_from_continuous_no_wait_candidate_optimum_s":reference_objective-min(p['analytic_objective_s'] for p in rows[-1]['paths']),
                "scope":"comparison with two no-wait paths; not a continuous-time regret certificate or error bound"}
        else:
            comparison={"status":"no_schedule_to_evaluate"}
        search.append({"time_step_s":step,"result":result,"analytic_schedule_evaluation":comparison})
    reconstruction,rec_arrays=reconstruction_study();raw.update(rec_arrays)
    report={"schema_version":1,"config":spec.model_dump(mode="json"),"configuration_id":content_id(spec.model_dump(mode="json")),
        "model":"c_t=kappa Laplacian(c), zero-flux walls, mean+amplitude*cos(pi*(y-ymin)/Ly)",
        "initialization":"exact finite-volume cell averages; spatial evaluation uses declared bilinear reconstruction",
        "observation":"piecewise bilinear space, piecewise linear physical time; grid/frame/trajectory-split Gauss2",
        "units":{"distance":"m","time":"s","concentration":"relative","exposure":"relative seconds","objective":"elapsed seconds + lambda*exposure"},
        "controls":{"solver_dt_s":spec.solver_dt_s,"saved_frame_times_s":times.tolist(),"path_quadrature":"gauss2_split",
                    "search_time_steps_s":list(spec.search_time_steps_s)},
        "candidate_scope":"the complete two simple paths, no voluntary waiting; continuous analytic moving-field comparison",
        "candidates":candidates.to_dict(network),"cases":rows,"reconstruction_validation":reconstruction,
        "node_time_label_counterexample":label_counterexample(),
        "time_expanded_case_id":case.id,"time_expanded_refinement":search,
        "limitations":["Synthetic PDE and artificial graph; no observed pollution calibration.",
            "Finite time-expanded optimum is for each declared rounded graph, not the continuous-time optimum.",
            "Rounding waits are charged even with voluntary waiting disabled; destination terminates at actual arrival.",
            "Output-frame interpolation, PDE discretization, path quadrature and search rounding are distinct errors.",
            "No adjoint, adaptive estimator, input uncertainty or end-to-end performance claim."]}
    return report,raw,graph


def plot_report(report,output: Path):
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout="constrained")
    cases=report['cases'];x=np.arange(len(cases))
    axes[0].bar(x-.18,[c['frozen_reference_regret_s'] for c in cases],.36,label='Frozen selection')
    axes[0].bar(x+.18,[c['moving_reference_regret_s'] for c in cases],.36,label='Moving-field selection')
    axes[0].set_xticks(x,[c['id'].replace('_','\n') for c in cases],fontsize=9)
    axes[0].set_ylabel('Analytic moving-field regret (s)');axes[0].legend();axes[0].set_title('Same full journey, common reference')
    rec=report['reconstruction_validation']
    for control,label in [('spatial_step_m','Spatial step (m)'),('time_step_s','Time step (s)')]:
        curve=[c for c in rec['quadrature_curves'] if c['control']==control]
        axes[1].loglog([c['step'] for c in curve],[c['absolute_error'] for c in curve],'o-',label=label)
    axes[1].set_xlabel('Requested maximum quadrature step');axes[1].set_ylabel('Absolute exposure error');axes[1].legend();axes[1].set_title('Fixed space-time reconstruction')
    curve=rec['frame_interpolation_curve']
    axes[2].loglog([c['maximum_frame_gap_s'] for c in curve],[c['absolute_error'] for c in curve],'o-')
    axes[2].set_xlabel('Maximum saved-frame gap (s)');axes[2].set_ylabel('Absolute exposure error');axes[2].set_title('Only temporal interpolation varies')
    from matplotlib.ticker import NullFormatter
    for ax in axes[1:]: ax.xaxis.set_minor_formatter(NullFormatter());ax.grid(alpha=.2)
    fig.savefig(output,dpi=160);plt.close(fig)


def _json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')


def verify_bundle(output: Path):
    output=Path(output);manifest=json.loads((output/'manifest.json').read_text())
    if manifest.get('schema_version')!=1 or manifest.get('status')!='completed': raise ValueError('Unknown or incomplete temporal study manifest.')
    files=manifest.get('files',{})
    if not {'config.json','report.json','raw_fields.npz','network.json','temporal_evidence.png','paths.csv','REPORT.md'}<=set(files):
        raise ValueError('Missing required temporal study artifacts.')
    for name,info in files.items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name: raise ValueError('Unsafe artifact path.')
        file=output/name
        if file.is_symlink() or not file.is_file() or file.resolve().is_relative_to(output.resolve()) is False: raise ValueError('Missing or unsafe artifact.')
        if file.stat().st_size!=info['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest()!=info['sha256']: raise ValueError('Artifact checksum mismatch: '+name)
    actual={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
    if actual!=set(files)|{'manifest.json'}:raise ValueError('Bundle has unlisted files.')
    config=Study.model_validate_json((output/'config.json').read_text())
    report=json.loads((output/'report.json').read_text())
    if report.get('schema_version')!=1 or report.get('configuration_id')!=content_id(config.model_dump(mode='json')):
        raise ValueError('Report schema or configuration identity mismatch.')
    if report.get('config')!=config.model_dump(mode='json'):
        raise ValueError('Report configuration differs from archived configuration.')
    if [row.get('id') for row in report.get('cases',[])]!=[case.id for case in config.cases]:
        raise ValueError('Report case identities do not match configuration.')
    source_hashes=report.get('source_snapshot_sha256',{})
    required_sources={'backend/app/temporal_observations.py','backend/app/observations.py',
        'backend/app/research/travel.py','backend/app/research/dynamic_routing.py',
        'scripts/run_temporal_experiments.py','requirements.lock.txt'}
    listed_sources={name.removeprefix('source_snapshot/') for name in files if name.startswith('source_snapshot/')}
    if not isinstance(source_hashes,dict) or not required_sources<=set(source_hashes) or set(source_hashes)!=listed_sources:
        raise ValueError('Missing or inconsistent source snapshot records.')
    if any(files['source_snapshot/'+name]['sha256']!=digest for name,digest in source_hashes.items()):
        raise ValueError('Source snapshot hashes disagree with the manifest.')
    graph=RoadNetwork.from_json(output/'network.json')
    if enumerate_simple_paths(graph,'s','t').to_dict(graph)!=report.get('candidates'):
        raise ValueError('Archived graph and candidate identities differ.')
    intervals=int(np.ceil(config.horizon_time_s/config.output_interval_s))
    times=np.linspace(0,config.horizon_time_s,intervals+1)
    n=config.ny,config.nx
    expected={'times_s':times.shape,'initial_cell_averages':n,
              'polynomial_frames':(6,8,8),'polynomial_times_s':(6,)}
    for case in config.cases:
        for suffix in ('fields','continuous_nodes','exact_cell_averages'):
            expected[case.id+'__'+suffix]=(len(times),*n)
    with zipfile.ZipFile(output/'raw_fields.npz') as archive:
        if sum(item.file_size for item in archive.infolist())>128*1024**2+65536:
            raise ValueError('Raw archive exceeds the declared array budget.')
    with np.load(output/'raw_fields.npz',allow_pickle=False) as raw:
        if set(raw.files)!=set(expected)|{'trajectory_xy','trajectory_times_s'}:
            raise ValueError('Unexpected or missing raw arrays.')
        for key,shape in expected.items():
            if raw[key].shape!=shape:raise ValueError('Incorrect archived array shape: '+key)
        for key in raw.files:
            if raw[key].dtype.kind not in 'fiu' or not np.isfinite(raw[key]).all():raise ValueError('Raw arrays must be finite and real.')
        if not np.array_equal(raw['times_s'],times):raise ValueError('Raw output times differ from configuration.')
        rec=report.get('reconstruction_validation',{})
        for key,reported in [('polynomial_times_s','frame_times_s'),('trajectory_xy','trajectory_xy'),('trajectory_times_s','trajectory_times_s')]:
            if not np.array_equal(raw[key],rec.get(reported)):raise ValueError('Trajectory/frame records differ from raw arrays.')
        from backend.app.observations import PathTrajectory
        PathTrajectory(raw['trajectory_xy'],raw['trajectory_times_s'],(),{})
    return {'status':'verified','files':len(files),'scope':'artifact hashes, configuration identity, graph candidates and raw-array schema; not scientific accuracy'}


def write_bundle(config,output: Path):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(f'Output already exists: {output}')
    report,raw,graph=run_study(config)
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent))
    try:
        sources=sorted((ROOT/'backend').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'scripts/__init__.py',ROOT/'requirements.lock.txt']
        for source in sources:
            destination=stage/'source_snapshot'/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
        report['provenance']=provenance()
        report['source_snapshot_sha256']={str(p.relative_to(stage/'source_snapshot')):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((stage/'source_snapshot').rglob('*')) if p.is_file()}
        _json(stage/'config.json',report['config']);_json(stage/'report.json',report);_json(stage/'network.json',graph)
        np.savez_compressed(stage/'raw_fields.npz',**raw)
        with (stage/'paths.csv').open('w',newline='') as f:
            rows=[{'case':c['id'],**p} for c in report['cases'] for p in c['paths']]
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        plot_report(report,stage/'temporal_evidence.png')
        lines=['# Time-aware trajectory and routing study','','All routes are evaluated over their full physical journey. The concentration and graph are synthetic.','',
            '| Case | Frozen selected | Moving selected | Frozen regret (s) | Moving regret (s) |','|---|---:|---:|---:|---:|']
        for c in report['cases']: lines.append(f"| {c['id']} | {c['frozen_selected']} | {c['moving_selected']} | {c['frozen_reference_regret_s']:.9g} | {c['moving_reference_regret_s']:.9g} |")
        lines+=['','The common reference is an independent closed-form continuous cosine-diffusion line integral, including the time of every segment. Initial PDE values are exact cell averages.','',
            'The polynomial reconstruction study includes nonuniform frame times, repeated edges and stationary waits. Grid/frame-split Gauss2 integrates its cubic trajectory restriction to roundoff. Trapezoid and output-frame refinement are measured separately.','',
            'Time-expanded results retain a label per node and arrival-time state. Arrival rounding creates an explicit charged wait; the destination terminates at actual arrival. Each result is optimal only on its stated finite rounded graph. Voluntary waiting is disabled in this refinement experiment and covered separately in tests.','',
            'No speedup or real-world pollution claim is made. Raw arrays, per-path results, source snapshots, and all assumptions are in this bundle.','',
            'Replay from source_snapshot with the prepared Python interpreter: `PYTHONDONTWRITEBYTECODE=1 python -m scripts.run_temporal_experiments --config ../config.json --output /absolute/new/output`. Bytecode writes are disabled to preserve the sealed source archive. Verify with `--verify BUNDLE`; redraw without recomputation with `--replot BUNDLE` (creates a separate sibling PNG).','',
            '![Evidence](temporal_evidence.png)','']
        (stage/'REPORT.md').write_text('\n'.join(lines))
        files={str(p.relative_to(stage)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(stage.rglob('*')) if p.is_file()}
        _json(stage/'manifest.json',{'schema_version':1,'status':'completed','files':files})
        verify_bundle(stage)
        if output.exists():raise FileExistsError(f'Output already exists: {output}')
        stage.rename(output)
    finally:
        if stage.exists():shutil.rmtree(stage)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--output',type=Path);actions.add_argument('--verify',type=Path);actions.add_argument('--replot',type=Path)
    args=parser.parse_args(argv)
    if args.verify: print(json.dumps(verify_bundle(args.verify)))
    elif args.replot:
        verify_bundle(args.replot)
        # Replot is an explicit export outside the sealed bundle.
        output=args.replot.parent/(args.replot.name+'-replot.png')
        if output.exists():raise FileExistsError(output)
        plot_report(json.loads((args.replot/'report.json').read_text()),output);print(str(output))
    else:
        result=write_bundle(json.loads(args.config.read_text()),args.output)
        print(json.dumps({'output':str(args.output.resolve()),'cases':len(result['cases']),'configuration_id':result['configuration_id']}))


if __name__=='__main__':main()
