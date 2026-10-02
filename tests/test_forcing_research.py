"""Versioned prescribed inputs, allocation admission and immutable full replay."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from scipy.integrate import quad

from backend.app.research import experiments
from backend.app.research.problems import (ExperimentSpec, capabilities, check_budget, forcing_array_payload,
    forcing_fields, imported_forcing_arrays, load_spec)


def config():
    return {"problem": {"bounds": [0,0,2,1], "model": "advection_diffusion", "kappa": .01,
        "velocity": [0,0], "boundary": "open", "initial": {"kind":"constant","value":.2},
        "forcing": {"kind":"analytic","times_s":[0,.1,.2],
            "wind":{"kind":"uniform","value":[.5,0]},"wind_scale":[1,0,-1],
            "source":{"kind":"constant","value":.1},"source_scale":[0,1,0],
            "inflow":{"left":1,"right":.5}}},
        "numerical":{"nx":6,"ny":4,"output_times_s":[0,.1,.2],
            "method":"imex_euler","backend":"numpy_scipy","dt":.01}}


def source_config(method="crank_nicolson", backend="scipy"):
    value=config()
    value["problem"].update(model="diffusion",boundary="zero_flux",kappa=0,
                             initial={"kind":"constant","value":0})
    value["problem"]["forcing"].update(wind={"kind":"uniform","value":[0,0]},
        source={"kind":"constant","value":2},inflow={})
    value["numerical"].update(method=method,backend=backend,dt=.03)
    return value


def imported(tmp_path):
    value=config();spec=ExperimentSpec.model_validate(value)
    payload=forcing_array_payload(forcing_fields(spec))
    path=tmp_path/"external_forcing.npz";np.savez_compressed(path,**payload)
    value["problem"]["forcing"]={"kind":"npz","path":path.name,
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bounds":[0,0,2,1],
        "nx":6,"ny":4,"times_s":[0,.1,.2]}
    return value,path


def rehash(bundle,name):
    path=bundle/"manifest.json";manifest=json.loads(path.read_text());file=bundle/name
    manifest["artifacts"][name]={"sha256":hashlib.sha256(file.read_bytes()).hexdigest(),"size_bytes":file.stat().st_size}
    path.write_text(json.dumps(manifest))


def test_legacy_ids_are_unchanged_when_new_fields_are_absent_or_explicit_none():
    root=Path(__file__).resolve().parents[1]
    old=load_spec(root/"configs/research/cosine_diffusion.json")
    assert old.identities()["configuration_id"]=="cc0c6576d77b611821326b02ac19fbb1ceab7ba09a1050d42d9fb8ef92a060a9"
    value=old.model_dump(mode="json");value["problem"]["forcing"]=None
    assert ExperimentSpec.model_validate(value).identities()==old.identities()
    assert "forcing" not in old.physical_descriptor()


def test_shared_knot_reversal_and_triangular_source_are_resolved_independently():
    spec=ExperimentSpec.model_validate(config());forcing=forcing_fields(spec)
    assert forcing.at(0).velocity_x[0,0]==.5
    assert forcing.at(.2).velocity_x[0,0]==-.5
    assert forcing.at(.05).velocity_x[0,0]==pytest.approx(.25)
    assert forcing.at(.05).source[0,0]==pytest.approx(.05)
    assert forcing.at(.15).source[0,0]==pytest.approx(.05)
    np.testing.assert_array_equal(forcing.inflow["left"],1)
    np.testing.assert_array_equal(forcing.inflow["right"],.5)
    assert spec.problem.forcing.generator_version==1


@pytest.mark.parametrize("wind",[
    {"kind":"shear","axis":"x","mean_speed_m_s":.4,"gradient_s_inv":.2},
    {"kind":"shear","axis":"y","mean_speed_m_s":.4,"gradient_s_inv":.2},
    {"kind":"rotation","angular_rate_s_inv":.3},
])
def test_linear_winds_use_exact_normal_face_averages_and_periodic_seams(wind):
    value=config();value["problem"]["boundary"]="periodic"
    value["problem"]["forcing"].update(wind=wind,wind_scale=None,inflow={})
    spec=ExperimentSpec.model_validate(value);grid=spec.grid();forcing=forcing_fields(spec)
    if wind["kind"]=="rotation":
        np.testing.assert_allclose(forcing.velocity_x[0,:,0],-.3*(grid.y-.5))
        np.testing.assert_allclose(forcing.velocity_y[0,0,:],.3*(grid.x-1))
    elif wind["axis"]=="x":
        np.testing.assert_allclose(forcing.velocity_x[0,:,0],.4+.2*(grid.y-.5))
        np.testing.assert_array_equal(forcing.velocity_y,0)
    else:
        np.testing.assert_allclose(forcing.velocity_y[0,0,:],.4+.2*(grid.x-1))
        np.testing.assert_array_equal(forcing.velocity_x,0)
    np.testing.assert_array_equal(forcing.velocity_x[:,:,0],forcing.velocity_x[:,:,-1])
    np.testing.assert_array_equal(forcing.velocity_y[:,0,:],forcing.velocity_y[:,-1,:])


def test_gaussian_source_is_an_exact_cell_average_not_a_center_sample():
    value=source_config();value["problem"]["forcing"].update(
        source={"kind":"gaussian","center":[.31,.27],"sigma_m":.12,"amplitude":2},source_scale=None)
    spec=ExperimentSpec.model_validate(value);grid=spec.grid();forcing=forcing_fields(spec)
    x=quad(lambda p:np.exp(-.5*((p-.31)/.12)**2),0,grid.dx,epsabs=1e-12)[0]/grid.dx
    y=quad(lambda p:np.exp(-.5*((p-.27)/.12)**2),0,grid.dy,epsabs=1e-12)[0]/grid.dy
    assert forcing.source[0,0,0]==pytest.approx(2*x*y,rel=2e-14)


@pytest.mark.parametrize("change",[
    {"times_s":[.01,.2]}, {"times_s":[0,.1,.1]}, {"times_s":[0,.1]},
    {"wind_scale":[1,2]}, {"inflow_scale":[1,-1,1]}, {"generator_version":2},
    {"velocity_unit":"km/h"}, {"temporal_extrapolation":"clamp"},
])
def test_forcing_schema_rejects_ambiguous_or_uncovered_inputs(change):
    value=config();value["problem"]["forcing"].update(change)
    with pytest.raises(ValueError):ExperimentSpec.model_validate(value)


def test_forcing_coverage_is_strict_at_next_float_after_last_knot():
    value=config();value["numerical"]["output_times_s"]=[0,float(np.nextafter(.2,np.inf))]
    with pytest.raises(ValueError,match="complete solve interval"):ExperimentSpec.model_validate(value)


def test_old_velocity_cannot_be_combined_with_prescribed_wind_and_cpp_rejected():
    value=config();value["problem"]["velocity"]=[.1,0]
    with pytest.raises(ValueError,match="legacy velocity"):ExperimentSpec.model_validate(value)
    value=source_config("explicit_euler","cpp")
    with pytest.raises(ValueError,match="prescribed forcing"):ExperimentSpec.model_validate(value)


def test_closed_transport_requires_zero_normal_wind_and_pure_diffusion_disallows_wind():
    value=config();value["problem"]["boundary"]="zero_flux"
    value["problem"]["forcing"]["inflow"]={}
    with pytest.raises(ValueError):forcing_fields(ExperimentSpec.model_validate(value))
    value["problem"]["forcing"]["wind"]["value"]=[0,0]
    assert not forcing_fields(ExperimentSpec.model_validate(value)).has_velocity
    value=source_config();value["problem"]["boundary"]="periodic"
    value["problem"]["forcing"]["wind"]["value"]=[1,0]
    with pytest.raises(ValueError,match="Pure diffusion"):forcing_fields(ExperimentSpec.model_validate(value))


def test_knots_and_aggregate_prescribed_arrays_are_counted_before_solve(tmp_path,monkeypatch):
    value=source_config("explicit_euler","numpy")
    value["problem"]["forcing"].update(times_s=[0,1],wind_scale=None,source_scale=None)
    value["numerical"].update(output_times_s=[0,1],dt=.5)
    value["budget"]={"max_cell_updates":96}
    assert check_budget(ExperimentSpec.model_validate(value))["estimated_steps_upper"]==4
    value["problem"]["forcing"]["times_s"]=list(np.linspace(0,1,9))
    with pytest.raises(ValueError,match="work budget"):check_budget(ExperimentSpec.model_validate(value))
    value,path=imported(tmp_path);value["budget"]={"max_forcing_bytes":1024}
    def forbidden(*args,**kwargs):raise AssertionError("Must reject before constructing prescribed fields.")
    monkeypatch.setattr(experiments,"forcing_fields",forbidden)
    with pytest.raises(ValueError,match="forcing byte budget"):
        experiments.run_experiment(value,tmp_path/"never",base_dir=tmp_path)
    assert not (tmp_path/"never").exists()


def test_npz_header_blocks_forged_shape_before_numpy_allocation(tmp_path,monkeypatch):
    value,path=imported(tmp_path)
    with zipfile.ZipFile(path) as archive:members={item.filename:archive.read(item) for item in archive.infolist()}
    forged=io.BytesIO();np.lib.format.write_array_header_1_0(forged,
        {"descr":"<f8","fortran_order":False,"shape":(10**9,10**9,10**9)})
    members["source.npy"]=forged.getvalue()
    with zipfile.ZipFile(path,"w") as archive:
        for key,raw in members.items():archive.writestr(key,raw)
    value["problem"]["forcing"]["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    def forbidden(*args,**kwargs):raise AssertionError("Must reject a forged header before np.load.")
    monkeypatch.setattr(np,"load",forbidden)
    with pytest.raises(ValueError,match="header shape"):
        imported_forcing_arrays(ExperimentSpec.model_validate(value),tmp_path)


@pytest.mark.parametrize("method,backend",[("explicit_euler","auto"),("backward_euler","scipy"),("crank_nicolson","scipy")])
def test_source_only_archive_mass_ledger_and_replay(method,backend,tmp_path):
    value=source_config(method,backend)
    result=experiments.run_experiment(value,tmp_path/"bundle")
    ledger=result["diagnostics"]
    assert abs(ledger["final_mass"]-ledger["cumulative_source_mass"])<1e-12
    assert abs(ledger["mass_balance_error"])<1e-12
    if method=="crank_nicolson":
        assert ledger["cumulative_source_mass"]==pytest.approx(.4,rel=2e-14)
    assert experiments.verify_bundle(tmp_path/"bundle")["status"]=="verified"
    replay=experiments.replay_bundle(tmp_path/"bundle",tmp_path/"replay")
    assert replay["replay_comparison"]["all_arrays_exact"]
    assert len(replay["replay_comparison"]["forcing_arrays"])==8


def test_transport_archive_preserves_all_inputs_and_bidirectional_boundary_ledger(tmp_path):
    value,path=imported(tmp_path);original=path.read_bytes()
    result=experiments.run_experiment(value,tmp_path/"bundle",base_dir=tmp_path)
    ledger=result["diagnostics"]
    assert ledger["cumulative_boundary_inward_mass"]>0
    assert ledger["cumulative_boundary_outward_mass"]>0
    assert abs(ledger["final_mass"]-ledger["initial_mass"]+ledger["cumulative_boundary_outward_mass"]
               -ledger["cumulative_boundary_inward_mass"]-ledger["cumulative_source_mass"])<1e-12
    assert (tmp_path/"bundle/forcing_input.npz").read_bytes()==original
    first=ExperimentSpec.model_validate(value).identities()
    moved=deepcopy(value);moved["problem"]["forcing"]["path"]="/another/computer/input.npz"
    assert ExperimentSpec.model_validate(moved).identities()==first
    path.unlink()
    assert experiments.verify_bundle(tmp_path/"bundle")["status"]=="verified"
    replay=experiments.replay_bundle(tmp_path/"bundle",tmp_path/"replay")
    assert replay["replay_comparison"]["all_arrays_exact"]


def test_analytic_archive_verification_never_regenerates_sampling_formula(tmp_path,monkeypatch):
    experiments.run_experiment(config(),tmp_path/"bundle")
    def forbidden(*args,**kwargs):raise AssertionError("Integrity uses archived arrays, not current analytic code.")
    monkeypatch.setattr(experiments,"forcing_fields",forbidden)
    assert experiments.verify_bundle(tmp_path/"bundle")["status"]=="verified"


@pytest.mark.parametrize("artifact",["forcing.npz","forcing_input.npz","result_record"])
def test_legacy_bundle_rejects_unconfigured_forcing_before_replay_creates_output(tmp_path,artifact):
    value=source_config();value["problem"].pop("forcing")
    bundle=tmp_path/"bundle";experiments.run_experiment(value,bundle)
    if artifact=="result_record":
        path=bundle/"result.json";result=json.loads(path.read_text())
        result["forcing"]=None;path.write_text(json.dumps(result));rehash(bundle,path.name)
    else:
        (bundle/artifact).write_bytes(b"unused artifact with a valid manifest checksum")
        rehash(bundle,artifact)
    with pytest.raises(ValueError,match="reserved forcing"):
        experiments.verify_bundle(bundle)
    with pytest.raises(ValueError,match="reserved forcing"):
        experiments.replay_bundle(bundle,tmp_path/"replay")
    assert not (tmp_path/"replay").exists()


def test_analytic_bundle_rejects_reserved_imported_forcing_artifact(tmp_path):
    bundle=tmp_path/"bundle";experiments.run_experiment(config(),bundle)
    (bundle/"forcing_input.npz").write_bytes(b"unconfigured input")
    rehash(bundle,"forcing_input.npz")
    with pytest.raises(ValueError,match="reserved imported forcing input"):
        experiments.verify_bundle(bundle)


def test_imported_forcing_y_direction_is_explicit_and_rejects_reversed_axes(tmp_path):
    value,_=imported(tmp_path)
    spec=ExperimentSpec.model_validate(value)
    assert spec.problem.forcing.y_direction=="south_to_north"
    assert spec.physical_descriptor()["forcing"]["y_direction"]=="south_to_north"
    value["problem"]["forcing"]["y_direction"]="north_to_south"
    with pytest.raises(ValueError,match="y_direction"):
        ExperimentSpec.model_validate(value)


def test_analytic_replay_reports_generator_changes_in_all_resolved_array_comparison(tmp_path,monkeypatch):
    experiments.run_experiment(config(),tmp_path/"bundle")
    original=experiments.forcing_fields
    def changed(spec,base_dir="."):
        from backend.app.numerics.forcing import PrescribedFields
        old=original(spec,base_dir)
        return PrescribedFields(old.grid,old.times_s,velocity_x=old.velocity_x,velocity_y=old.velocity_y,
                                source=old.source+.01,inflow=old.inflow,boundary=old.boundary)
    monkeypatch.setattr(experiments,"forcing_fields",changed)
    replay=experiments.replay_bundle(tmp_path/"bundle",tmp_path/"changed_replay")
    comparison=replay["replay_comparison"]
    assert not comparison["all_arrays_exact"]
    assert comparison["forcing_arrays"]["source"]["linf"]==pytest.approx(.01)
    assert not comparison["forcing_arrays"]["source"]["exact"]
    assert comparison["linf"]>0
    assert experiments.verify_bundle(tmp_path/"bundle")["status"]=="verified"


def test_capabilities_explicitly_include_closed_prescribed_transport_and_source_startup():
    combos=capabilities()["prescribed_forcing"]["supported_combinations"]
    assert len(combos)==28
    assert any(c["model"]=="advection_diffusion" and c["boundary"]=="zero_flux"
               and c["method"]=="imex_euler" and c["backend"]=="auto" for c in combos)
    assert any(c["model"]=="diffusion" and c["startup"]=="rannacher" for c in combos)
    assert not any(c["backend"]=="cpp" for c in combos)


@pytest.mark.parametrize("damage",["record","shape","nan","boundary","imported_values","external_path"])
def test_semantic_forcing_tamper_is_rejected_even_after_valid_artifact_checksums(tmp_path,damage):
    value,_=imported(tmp_path);bundle=tmp_path/"bundle"
    experiments.run_experiment(value,bundle,base_dir=tmp_path)
    if damage=="record":
        path=bundle/"result.json";result=json.loads(path.read_text())
        result["forcing"]["schema_version"]=99;path.write_text(json.dumps(result));rehash(bundle,path.name)
    elif damage=="external_path":
        path=bundle/"config.json";data=json.loads(path.read_text())
        data["problem"]["forcing"]["path"]="../external_forcing.npz"
        path.write_text(json.dumps(data));rehash(bundle,path.name)
    else:
        with np.load(bundle/"forcing.npz",allow_pickle=False) as saved:arrays={key:saved[key] for key in saved.files}
        if damage=="shape":arrays["source"]=np.ones((3,2,2))
        elif damage=="nan":arrays["source"][0,0,0]=np.nan
        elif damage=="boundary":arrays["inflow_left"][0,0]=-1
        else:arrays["source"][0,0,0]=np.nextafter(0.,1.)
        if damage in {"boundary","imported_values"}:
            path=bundle/"result.json";result=json.loads(path.read_text())
            result["forcing"]=experiments._forcing_record(arrays);path.write_text(json.dumps(result));rehash(bundle,path.name)
        np.savez_compressed(bundle/"forcing.npz",**arrays);rehash(bundle,"forcing.npz")
    with pytest.raises(ValueError):experiments.verify_bundle(bundle)


def test_prescribed_sample_config_is_valid_and_uses_reversal_and_triangular_gaussian():
    root=Path(__file__).resolve().parents[1]
    spec=load_spec(root/"configs/research/prescribed_transport.json")
    driving=forcing_fields(spec)
    assert driving.has_velocity and driving.has_source and driving.has_inflow
    assert driving.velocity_x[0,0,0]>0 and driving.velocity_x[-1,0,0]<0
    np.testing.assert_array_equal(driving.source[0],0)
    np.testing.assert_array_equal(driving.source[-1],0)
    assert driving.source[1].max()>0
