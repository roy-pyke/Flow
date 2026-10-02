"""Independent temporal references, schedule integration and complete archives."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad

from backend.app.diffusion import Grid
from backend.app.observations import PathTrajectory, build_path_trajectory
from backend.app.temporal_observations import build_trajectory_observer
from backend.app.research.dynamic_routing import solve_time_expanded
from backend.app.research.travel import TimeDependentExposure, trajectory_from_schedule
from backend.app.routing import RoadNetwork
from scripts.run_temporal_experiments import (DEFAULT_CONFIG, Study, analytic_cosine_exposure,
    polynomial_exposure, run_study, verify_bundle, write_bundle)


def network():
    return RoadNetwork({'bounds':[0,0,10,10],'nodes':[{'id':'s','x':2,'y':2},{'id':'a','x':5,'y':4},{'id':'t','x':8,'y':3}],
        'edges':[{'u':'s','v':'a','key':0,'coordinates':[[2,2],[3,4],[5,4]]},
                 {'u':'a','v':'t','key':0,'coordinates':[[5,4],[7,5],[8,3]]}]})


def field_adapter(**kwargs):
    grid=Grid((0,0,10,10),8,8);times=np.array([0,.7,2.1,5,12.])
    xx,yy=np.meshgrid(grid.x,grid.y)
    frames=np.stack([1+.2*xx+.3*yy+.1*t+.02*xx*yy*t for t in times])
    return TimeDependentExposure(network(),grid,times,frames,**kwargs)


@pytest.mark.parametrize('kappa',[0,.01,5])
def test_cosine_closed_integral_matches_independent_adaptive_quadrature_with_waits(kappa):
    t=build_path_trajectory(network(),[0,1],speed_mps=2,departure_time_s=.3,waits_s=[.2,.7,.4])
    wave=np.pi/10;total=0
    for a,b,start,end in zip(t.vertex_xy[:-1],t.vertex_xy[1:],t.vertex_times_s[:-1],t.vertex_times_s[1:]):
        def value(time):
            y=a[1]+(b[1]-a[1])*(time-start)/(end-start)
            return 1+.95*np.exp(-kappa*wave*wave*time)*np.cos(wave*y)
        total+=quad(value,start,end,epsabs=1e-12,epsrel=1e-12)[0]
    assert analytic_cosine_exposure(t,kappa=kappa,mean=1,amplitude=.95,wave_number=wave)==pytest.approx(total,rel=2e-14)


def test_adapter_matches_polynomial_reference_and_owns_fields_and_cache():
    adapter=field_adapter(speed_mps=2,max_cache_entries=2)
    trajectory=build_path_trajectory(adapter.network,[0,1],2,.3,waits_s=[.1,.2,.3])
    assert adapter.trajectory_exposure(trajectory)==pytest.approx(polynomial_exposure(trajectory),rel=1e-14)
    expected=adapter.edge_exposure(0,.3)
    assert adapter.edge_exposure(0,.3)==expected
    assert adapter.cache_hits==1
    for start in (.4,.5,.6):adapter.edge_exposure(0,start)
    assert adapter.metadata()['cache_entries']==2
    assert not adapter.frames.flags.writeable and not adapter.times_s.flags.writeable
    new=TimeDependentExposure(adapter.network,adapter.grid,adapter.times_s,adapter.frames,speed_mps=2,max_cache_entries=0)
    new.edge_exposure(0,.3);assert new.metadata()['cache_entries']==0
    frames=adapter.frames.copy();times=adapter.times_s.copy()
    owned=TimeDependentExposure(adapter.network,adapter.grid,times,frames)
    frames[:]=99;times[:]=99
    assert owned.frames[0,0,0]!=99 and owned.times_s[0]==0


@pytest.mark.parametrize('bad',[-1,np.nan,np.inf])
def test_adapter_rejects_invalid_routing_fields(bad):
    a=field_adapter();frames=a.frames.copy();frames[0,0,0]=bad
    with pytest.raises(ValueError,match='finite nonnegative'):TimeDependentExposure(a.network,a.grid,a.times_s,frames)


def test_adapter_enforces_coverage_graph_join_and_budgets():
    a=field_adapter()
    with pytest.raises(ValueError,match='coverage'):a.edge_exposure(0,12)
    with pytest.raises(ValueError,match='coverage'):a.waiting_exposure('s',-1,0)
    with pytest.raises(ValueError,match='coverage'):a.waiting_exposure('s',12,np.nextafter(12.,np.inf))
    assert a.waiting_exposure('s',12,12)==0
    with pytest.raises(ValueError,match='byte budget'):field_adapter(max_field_bytes=10)
    with pytest.raises(ValueError,match='max_samples'):field_adapter(max_samples=1).edge_exposure(0,0)
    with pytest.raises(ValueError,match='increasing'):TimeDependentExposure(a.network,a.grid,[-1e308,1e308],a.frames[:2])
    graph=network();graph.edges[0]['coordinates'][0][0]+=.01
    with pytest.raises(ValueError,match='exact geometry'):TimeDependentExposure(graph,a.grid,a.times_s,a.frames)


def test_polyline_clock_matches_exact_search_horizon():
    coordinates=[[.1,.1],[.31529526772962263,.7067003751060119],[.5493078171597499,.5681219753894384],[.9,.9]]
    graph=RoadNetwork({'bounds':[0,0,1,1],'nodes':[{'id':'s','x':.1,'y':.1},{'id':'t','x':.9,'y':.9}],
        'edges':[{'u':'s','v':'t','key':0,'coordinates':coordinates}]})
    horizon=graph.edges[0]['length_m']/1.4
    adapter=TimeDependentExposure(graph,Grid((0,0,1,1),8,8),[0,horizon],np.ones((2,8,8)))
    result=solve_time_expanded(graph,'s','t',departure_time_s=0,horizon_time_s=horizon,time_step_s=.5,
        speed_mps=1.4,lambda_weight=1,edge_exposure=adapter.edge_exposure,waiting_exposure=adapter.waiting_exposure)
    assert result['arrival_time_s']==horizon
    trajectory=trajectory_from_schedule(graph,result)
    assert trajectory.arrival_time_s==horizon
    assert adapter.trajectory_exposure(trajectory)==pytest.approx(horizon,rel=1e-14)


def test_search_schedule_reconstruction_independently_includes_all_waits():
    a=field_adapter(speed_mps=2)
    result=solve_time_expanded(a.network,'s','t',departure_time_s=.3,horizon_time_s=12,time_step_s=1,
        speed_mps=2,lambda_weight=1,edge_exposure=a.edge_exposure,waiting_exposure=a.waiting_exposure,allow_wait=False)
    full=trajectory_from_schedule(a.network,result)
    assert result['rounding_wait_time_s']>0
    assert a.trajectory_exposure(full)==pytest.approx(result['exposure'],rel=1e-14)
    assert polynomial_exposure(full)==pytest.approx(result['exposure'],rel=1e-14)
    broken=copy.deepcopy(result);broken['actions'][1]['start_time_s']+=.01
    with pytest.raises(ValueError,match='uninterrupted'):trajectory_from_schedule(a.network,broken)
    broken=copy.deepcopy(result);broken['edge_indices']=[]
    with pytest.raises(ValueError,match='edge list'):trajectory_from_schedule(a.network,broken)


@pytest.fixture(scope='module')
def small_config():
    config=json.loads(DEFAULT_CONFIG.read_text());config.update(nx=16,ny=16,solver_dt_s=.01,output_interval_s=.25)
    return config


@pytest.fixture(scope='module')
def study(small_config):return run_study(small_config)


def test_actual_pde_freezing_equivalence_slow_change_and_failure(study):
    report,raw,graph=study
    cases={c['id']:c for c in report['cases']}
    stationary=cases['stationary_equivalent']
    for p in stationary['paths']:
        assert p['frozen_exposure']==pytest.approx(p['moving_exposure'],rel=2e-14)
    slow=cases['slow_evolution'];assert slow['frozen_selected']==slow['moving_selected']
    fast=cases['frozen_selects_wrong_route']
    assert fast['frozen_selected']==0 and fast['moving_selected']==1
    assert fast['moving_reference_regret_s']==0 and fast['frozen_reference_regret_s']>2.6
    assert max(p['moving_exposure_error'] for p in fast['paths'])<.003
    assert report['node_time_label_counterexample']['result']['objective']==4
    assert report['node_time_label_counterexample']['single_node_label_result']==102
    for row in report['time_expanded_refinement']:
        result=row['result'];audit=row['analytic_schedule_evaluation']
        assert result['rounding_wait_time_s']>0 and not result['continuous_time_optimum']
        assert audit['whole_trajectory_exposure']==pytest.approx(result['exposure'],rel=1e-13)
    assert raw['initial_cell_averages'].shape==(16,16)


def test_temporal_and_quadrature_refinements_are_separate(study):
    rec=study[0]['reconstruction_validation']
    assert rec['gauss_absolute_error']<1e-12
    curve=rec['frame_interpolation_curve']
    errors=[c['absolute_error'] for c in curve]
    assert all(b<a for a,b in zip(errors,errors[1:])) and errors[-1]<errors[0]/200
    for control in ('spatial_step_m','time_step_s'):
        errors=[c['absolute_error'] for c in rec['quadrature_curves'] if c['control']==control]
        assert errors[-1]<errors[0]/30


def test_translating_periodic_wave_followed_at_its_transport_speed():
    # c=2+.3*cos(k*(x-u*t)) solves constant-speed translation exactly. The
    # traveler follows x=1+u*t, so its continuous exposure is constant*3.
    trajectory=PathTrajectory(np.array([[1.,5.],[7.,5.]]),np.array([0.,3.]),(),{})
    wave=2*np.pi/10
    exact=3*(2+.3*np.cos(wave))
    errors=[]
    for n in (16,32,64):
        grid=Grid((0,0,10,10),n,8);times=np.linspace(0,3,2*n+1)
        frames=np.broadcast_to((2+.3*np.cos(wave*(grid.x[None,:]-2*times[:,None])))[:,None,:],(len(times),8,n)).copy()
        value=build_trajectory_observer(grid,times,trajectory,boundary='periodic').apply(frames)
        errors.append(abs(value-exact))
    assert all(b<a/3 for a,b in zip(errors,errors[1:]))
    assert errors[-1]<.001


@pytest.mark.parametrize('changes',[
    {'horizon_time_s':1}, {'schema_version':999}, {'solver_dt_s':1e-320},
    {'nx':128,'ny':128,'horizon_time_s':100,'output_interval_s':1/6,'solver_dt_s':1,
     'cases':[{'id':f'case_{i}','kappa':1} for i in range(8)]},
])
def test_study_rejects_incomplete_coverage_versions_and_aggregate_array_overflow(small_config,changes):
    with pytest.raises(ValueError):run_study({**small_config,**changes})


def rehash(bundle,name):
    m=json.loads((bundle/'manifest.json').read_text());p=bundle/name
    m['files'][name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    (bundle/'manifest.json').write_text(json.dumps(m))


def test_study_bundle_roundtrip_and_no_overwrite(tmp_path,small_config):
    out=tmp_path/'study';write_bundle(small_config,out)
    assert verify_bundle(out)['status']=='verified'
    with pytest.raises(FileExistsError):write_bundle(small_config,out)
    report=json.loads((out/'report.json').read_text());report['config']['mean']=99
    (out/'report.json').write_text(json.dumps(report));rehash(out,'report.json')
    with pytest.raises(ValueError,match='configuration differs'):verify_bundle(out)


@pytest.mark.parametrize('damage',['shape','nan','schema','source','extra'])
def test_checksums_do_not_replace_bundle_structure_validation(tmp_path,small_config,damage):
    out=tmp_path/'study';write_bundle(small_config,out)
    if damage in ('shape','nan'):
        with np.load(out/'raw_fields.npz',allow_pickle=False) as arrays:data={k:arrays[k] for k in arrays.files}
        if damage=='shape':data['initial_cell_averages']=np.ones((2,2))
        else:data['initial_cell_averages'][0,0]=np.nan
        np.savez_compressed(out/'raw_fields.npz',**data);rehash(out,'raw_fields.npz')
    elif damage in ('schema','source'):
        r=json.loads((out/'report.json').read_text())
        if damage=='schema':r['schema_version']=999
        else:r['source_snapshot_sha256']['backend/app/observations.py']='0'*64
        (out/'report.json').write_text(json.dumps(r));rehash(out,'report.json')
    else:(out/'unexpected.txt').write_text('unlisted')
    with pytest.raises(ValueError):verify_bundle(out)


def test_failed_render_does_not_publish_a_partial_bundle(tmp_path,small_config,monkeypatch):
    from scripts import run_temporal_experiments as module
    def fail(*args):raise OSError('disk full')
    monkeypatch.setattr(module,'plot_report',fail)
    with pytest.raises(OSError,match='disk full'):write_bundle(small_config,tmp_path/'failed')
    assert not (tmp_path/'failed').exists() and not list(tmp_path.glob('.failed-*'))
