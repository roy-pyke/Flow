"""Independent conservative characteristics, source stages and sealed study evidence."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
import zipfile

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from backend.app.diffusion import Grid
from scripts import run_forced_transport_experiments as module
from scripts.run_forced_transport_experiments import (DEFAULT_CONFIG, PeriodicSpec, Study,
    check_budget, expected_raw_shapes, independent_upwind_operator, main,
    periodic_clock, periodic_exact_cell_average, periodic_semidiscrete,
    pwl_integral, run_study, source_mode, source_reference, verify_bundle, write_bundle)


@pytest.fixture(scope="module")
def quick_config():
    config=json.loads(DEFAULT_CONFIG.read_text())
    config["periodic"].update(nx_values=[8,16,32,64],ny=2,dt_s=.0005,temporal_nx=8,step_counts=[32,64,128])
    config["source"].update(nx=6,ny=2,step_counts=[16,32,64])
    config["open"].update(nx=8,ny=2,step_counts=[32,64,128])
    return config


@pytest.fixture(scope="module")
def study(quick_config):return run_study(quick_config)


@pytest.mark.parametrize("time_fraction",[0,.25,.375,.625,.875,1])
def test_characteristic_cell_averages_against_independent_quadrature(time_fraction):
    spec=PeriodicSpec(nx_values=(8,16),ny=2)
    grid=Grid((0,0,spec.length_x_m,spec.length_y_m),16,2)
    time=time_fraction*spec.duration_s
    signed=2*np.pi/spec.length_x_m*pwl_integral(*periodic_clock(spec),time)
    actual=periodic_exact_cell_average(grid,time,spec)
    for i,x in enumerate(grid.x):
        expected=quad(lambda z:spec.mean/(np.cosh(signed)+np.sinh(signed)*np.cos(2*np.pi*z/spec.length_x_m)),
                      x-grid.dx/2,x+grid.dx/2,epsabs=1e-12,epsrel=1e-12)[0]/grid.dx
        np.testing.assert_allclose(actual[:,i],expected,rtol=2e-14,atol=2e-14)
    assert grid.dx*grid.dy*actual.sum()==pytest.approx(spec.mean*spec.length_x_m*spec.length_y_m,rel=2e-15)


def test_pwl_signed_integral_and_inverse_characteristic_compression():
    spec=PeriodicSpec();knots,values=periodic_clock(spec)
    assert pwl_integral(knots,values,spec.duration_s)==pytest.approx(0,abs=1e-16)
    assert pwl_integral(knots,values,spec.duration_s*3/8)==pytest.approx(5/16*spec.duration_s*spec.wind_amplitude_m_s)
    grid=Grid((0,0,1,.25),16,4)
    compressed=periodic_exact_cell_average(grid,spec.duration_s/4,spec)
    assert compressed.max()>spec.mean and compressed.min()<spec.mean
    np.testing.assert_allclose(periodic_exact_cell_average(grid,spec.duration_s,spec),spec.mean,atol=5e-15)
    with pytest.raises(ValueError):pwl_integral(knots,values,-1)


def test_independent_upwind_is_conservative_not_constant_preserving():
    positive=independent_upwind_operator(16,1,1);negative=independent_upwind_operator(16,1,-1)
    np.testing.assert_allclose(positive.sum(axis=0),0,atol=1e-14)
    np.testing.assert_allclose(negative.sum(axis=0),0,atol=1e-14)
    assert np.max(abs(positive@np.ones(16)))>5
    assert np.linalg.norm(positive@negative-negative@positive)>1
    for matrix in (positive,negative):
        off=matrix.copy();np.fill_diagonal(off,0)
        assert off.min()>=0 and np.diag(matrix).max()<=0


def test_chronological_sign_exponentials_match_independent_adaptive_ode():
    spec=PeriodicSpec(nx_values=(8,16),ny=2,temporal_nx=8)
    grid=Grid((0,0,1,.25),8,2);times=np.array([0,.2,.8])
    reference,operators=periodic_semidiscrete(grid,times,spec)
    knots,amplitude=periodic_clock(spec)
    def rhs(t,state):
        a=np.interp(t,knots,amplitude)
        return abs(a)*operators[1 if a>=0 else -1]@state
    numerical=solve_ivp(rhs,(0,.8),np.full(8,spec.mean),t_eval=times,rtol=1e-11,atol=1e-12,max_step=.002)
    assert numerical.success
    np.testing.assert_allclose(reference[:,0],numerical.y.T,rtol=2e-10,atol=2e-10)


def test_periodic_spatial_and_time_errors_are_separated(study):
    report,raw=study;rows=report["periodic"]["cases"]
    assert rows[-1]["observed_rms_order"]>.9
    assert all(row["temporal_to_spatial_rms_ratio"]<.1 for row in rows)
    assert all(row["inputs_unmodified"] for row in rows)
    assert rows[-1]["nonconservative_constant_error_at_compression"]["rms"]>.1
    assert abs(report["periodic"]["temporal_cases"][-1]["observed_rms_order"]-1)<.05
    for row in rows:
        prefix=f"periodic_n{row['nx']}__"
        np.testing.assert_array_equal(raw[prefix+"velocity_x"][:,:,0],raw[prefix+"velocity_x"][:,:,-1])
        assert row["diagnostics"]["max_internal_mass_balance_error"]<1e-13
        assert row["diagnostics"]["max_concentration"]>1.2
        assert row["diagnostics"]["cumulative_boundary_outward_mass"]==0
        assert row["spatial_error"]["rms"]>0  # Reversing wind cannot undo upwind diffusion.


def test_source_reference_matches_separate_discrete_cosine_modal_equation(quick_config):
    spec=Study.model_validate(quick_config).source;grid=Grid((0,0,1,1),spec.nx,spec.ny)
    times=np.array([0,.4,.8]);reference,_=source_reference(grid,times,spec)
    mode=source_mode(grid,spec)
    lam=-4*spec.diffusivity/grid.dx**2*np.sin(np.pi*grid.dx/2)**2
    for t,result in zip(times,reference):
        # Scalar Duhamel quadrature is independent of the augmented expm reference.
        a=spec.amplitude*np.exp(lam*t)+quad(lambda tau:np.exp(lam*(t-tau))*(spec.source_mode+spec.source_mode_slope*tau),0,t,epsabs=1e-13)[0]
        mean=spec.mean+spec.source_mean*t+.5*spec.source_mean_slope*t*t
        np.testing.assert_allclose(result,mean+a*mode,atol=1e-14,rtol=1e-14)


def test_time_dependent_source_fe_be_first_cn_second_and_stage_mass(study):
    report,_=study
    for method,rows in report["source"]["methods"].items():
        assert abs(rows[-1]["observed_rms_order"]-(2 if method=="crank_nicolson" else 1))<.03
        assert all(b["error"]["rms"]<a["error"]["rms"] for a,b in zip(rows,rows[1:]))
        for row in rows:
            d=row["diagnostics"]
            assert d["max_internal_mass_balance_error"]<1e-13
            assert d["cumulative_boundary_inward_mass"]==d["cumulative_boundary_outward_mass"]==0
            if method=="explicit_euler":assert row["source_integral_error"]<0
            elif method=="backward_euler":assert row["source_integral_error"]>0
            else:assert abs(row["source_integral_error"])<1e-14
    assert report["source"]["spatial_error_at_fixed_grid"]["rms"]>0


def test_open_exact_field_discrete_ledger_and_continuous_integral_errors(study):
    report,raw=study;opened=report["open"];analytic=opened["analytic_integrals"]
    assert abs(analytic["final_mass"]-analytic["initial_mass"]+analytic["outward_mass"]-analytic["inward_mass"]-analytic["source_mass"])<1e-14
    assert analytic["outward_mass"]>analytic["inward_mass"]>0
    for method,rows in opened["methods"].items():
        for row in rows:
            assert row["field_error"]["linf"]<1e-12
            assert abs(row["independently_recomputed_balance"])<1e-12
            assert row["diagnostics"]["max_internal_mass_balance_error"]<1e-12
            assert row["diagnostics"]["internal_steps"]>row["requested_steps"]  # 0.37T knot is respected.
            for key,error in row["integral_errors"].items():
                assert error<0
                assert error==pytest.approx(row["predicted_left_rule_errors"][key],rel=1e-8,abs=1e-13)
        assert abs(rows[-1]["integral_errors"]["source_mass"])/abs(rows[-2]["integral_errors"]["source_mass"])==pytest.approx(.5,rel=.03)
    np.testing.assert_array_equal(raw["open__inflow_right"],0)
    np.testing.assert_array_equal(raw["open__velocity_y"],0)


@pytest.mark.parametrize("section,changes",[
    ("periodic",{"nx_values":[16,8]}),("periodic",{"dt_s":1e-320}),
    ("periodic",{"length_x_m":.001}),("periodic",{"dt_s":1}),
    ("periodic",{"step_counts":[8,8,16]}),("periodic",{"duration_s":5e-324}),
    ("source",{"source_mean":.1}),("source",{"mean":.1}),
    ("source",{"nx":24,"ny":8}),("source",{"diffusivity":100}),
    ("open",{"inlet_velocity_m_s":100}),("open",{"diffusivity":100}),
    ("budget",{"max_array_bytes":1024}),("budget",{"max_cell_updates":10}),
    ("budget",{"max_reference_scaled_rate":.01}),
])
def test_rejected_protocol_stops_before_solver(quick_config,section,changes,monkeypatch):
    config=copy.deepcopy(quick_config);config[section].update(changes)
    def forbidden(*args,**kwargs):raise AssertionError("Rejected study must not call production solve")
    monkeypatch.setattr(module,"solve",forbidden)
    with pytest.raises(ValueError):run_study(config)


def test_schema_budget_and_default_protocol(study,quick_config):
    report,raw=study;spec=Study.model_validate(quick_config)
    assert set(raw)==set(expected_raw_shapes(spec))
    assert sum(array.nbytes for array in raw.values())==check_budget(spec)["raw_array_bytes"]
    assert report["admission"]["raw_array_count"]==len(raw)
    assert Study.model_validate_json(DEFAULT_CONFIG.read_text()).periodic.nx_values==(16,32,64,128)
    for mutation in ({"schema_version":999},{"hidden_method":True}):
        with pytest.raises(ValueError):Study.model_validate({**quick_config,**mutation})


@pytest.fixture(scope="module")
def saved_bundle(tmp_path_factory,quick_config):
    output=tmp_path_factory.mktemp("forced-evidence")/"bundle"
    write_bundle(quick_config,output)
    return output


def rehash(bundle,name):
    manifest=json.loads((bundle/"manifest.json").read_text());data=(bundle/name).read_bytes()
    manifest["files"][name]={"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
    (bundle/"manifest.json").write_text(json.dumps(manifest))


def test_atomic_bundle_inventory_verification_and_source_closure(saved_bundle,quick_config):
    result=verify_bundle(saved_bundle)
    assert result["status"]=="verified" and result["arrays"]==104
    report=json.loads((saved_bundle/"report.json").read_text())
    for name in ("backend/app/numerics/forcing.py","backend/app/numerics/forced_solver.py","backend/app/numerics/__init__.py","scripts/__init__.py"):
        assert name in report["source_snapshot_sha256"]
    assert (saved_bundle/"forced_evidence.png").stat().st_size>1000
    with pytest.raises(FileExistsError):write_bundle(quick_config,saved_bundle)


@pytest.mark.parametrize("damage",["shape","nonfinite","extra_array","source_hash","config","case","extra_file","missing_required_source","output_clock","diagnostic_clock","raw_clock","missing_metric","nonnumeric_metric"])
def test_rehashed_corruption_cannot_bypass_layout_or_identity(tmp_path,saved_bundle,damage):
    out=tmp_path/"damaged";shutil.copytree(saved_bundle,out)
    if damage in {"shape","nonfinite","extra_array","raw_clock"}:
        with np.load(out/"raw_fields.npz",allow_pickle=False) as values:raw={key:values[key] for key in values.files}
        if damage=="shape":raw["periodic_n8__numerical"]=np.ones((2,2))
        elif damage=="nonfinite":raw["periodic_n8__numerical"][0,0,0]=np.nan
        elif damage=="raw_clock":raw["open__forcing_times_s"][1]+=.01
        else:raw["undeclared"]=np.ones(2)
        np.savez_compressed(out/"raw_fields.npz",**raw);rehash(out,"raw_fields.npz")
    elif damage in {"source_hash","config","case","missing_required_source","output_clock","diagnostic_clock","missing_metric","nonnumeric_metric"}:
        report=json.loads((out/"report.json").read_text())
        if damage=="source_hash":report["source_snapshot_sha256"]["backend/app/numerics/forcing.py"]="0"*64
        elif damage=="config":report["config"]["periodic"]["mean"]=42
        elif damage=="case":report["source"]["methods"]["explicit_euler"][0]["requested_steps"]=999
        elif damage=="missing_metric":del report["periodic"]["cases"][0]["spatial_error"]
        elif damage=="nonnumeric_metric":report["open"]["methods"]["imex_euler"][0]["integral_errors"]["source_mass"]="wrong"
        elif damage=="output_clock":del report["periodic"]["output_times_s"]
        elif damage=="diagnostic_clock":report["periodic"]["cases"][0]["diagnostics"]["output_times_s"]=[0,.01,.8]
        else:
            name="backend/app/numerics/__init__.py";del report["source_snapshot_sha256"][name]
            manifest=json.loads((out/"manifest.json").read_text());del manifest["files"]["source_snapshot/"+name]
            (out/"manifest.json").write_text(json.dumps(manifest));(out/"source_snapshot"/name).unlink()
        (out/"report.json").write_text(json.dumps(report));rehash(out,"report.json")
    else:(out/"unexpected.txt").write_text("unlisted")
    with pytest.raises(ValueError):verify_bundle(out)


def test_forged_shape_is_rejected_before_numpy_loading(tmp_path,saved_bundle,monkeypatch):
    out=tmp_path/"huge";shutil.copytree(saved_bundle,out)
    archive=out/"raw_fields.npz"
    with zipfile.ZipFile(archive) as z:members={name:z.read(name) for name in z.namelist()}
    stream=io.BytesIO();np.lib.format.write_array_header_1_0(stream,{"descr":"<f8","fortran_order":False,"shape":(2**40,)})
    members["periodic_n8__numerical.npy"]=stream.getvalue()
    with zipfile.ZipFile(archive,"w") as z:
        for name,data in members.items():z.writestr(name,data)
    rehash(out,"raw_fields.npz")
    def forbidden(*args,**kwargs):raise AssertionError("np.load must not inspect a forged huge allocation")
    monkeypatch.setattr(module.np,"load",forbidden)
    with pytest.raises(ValueError):verify_bundle(out)


def test_duplicate_npy_members_rejected(tmp_path,saved_bundle):
    out=tmp_path/"duplicate";shutil.copytree(saved_bundle,out)
    with zipfile.ZipFile(out/"raw_fields.npz","a") as z:
        with pytest.warns(UserWarning):z.writestr("periodic_n8__numerical.npy",z.read("periodic_n8__numerical.npy"))
    rehash(out,"raw_fields.npz")
    with pytest.raises(ValueError,match="duplicate"):verify_bundle(out)


def test_failed_render_cleans_staging_and_cannot_publish(tmp_path,quick_config,monkeypatch):
    def fail(*args):raise OSError("simulated render failure")
    monkeypatch.setattr(module,"plot_report",fail)
    output=tmp_path/"failed"
    with pytest.raises(OSError,match="render failure"):write_bundle(quick_config,output)
    assert not output.exists() and not list(tmp_path.glob(".failed-*"))


def test_verify_replot_never_changes_sealed_bundle(tmp_path,saved_bundle):
    out=tmp_path/"replot";shutil.copytree(saved_bundle,out)
    before={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob("*") if p.is_file()}
    main(["--verify",str(out)]);main(["--replot",str(out)])
    assert (out.parent/(out.name+"-replot.png")).is_file()
    after={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob("*") if p.is_file()}
    assert before==after
    with pytest.raises(FileExistsError):main(["--replot",str(out)])
