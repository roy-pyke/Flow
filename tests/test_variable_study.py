"""Independent variable-coefficient references and complete study archives."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from backend.app.diffusion import Grid
from scripts.run_variable_diffusion_experiments import (DEFAULT_CONFIG, SpatialSpec, Study,
    check_budget, expected_raw_shapes, independent_dense_operator, main, run_study,
    smooth_coefficient, smooth_exact_cell_average, verify_bundle, write_bundle)


@pytest.fixture(scope="module")
def quick_config():
    config=json.loads(DEFAULT_CONFIG.read_text())
    config["spatial"].update(nx_values=[8,16,32],ny=2,dt_s=.001)
    config["temporal"].update(nx=6,ny=4,step_counts=[8,16,32])
    config["layered"].update(nx=16,ny=2)
    return config


@pytest.fixture(scope="module")
def study(quick_config):return run_study(quick_config)


def test_smooth_cell_averages_against_independent_quadrature():
    spec=SpatialSpec(nx_values=(8,16),ny=2)
    grid=Grid((0,0,spec.length_x_m,spec.length_y_m),8,2)
    time=.2;actual=smooth_exact_cell_average(grid,time,spec)
    for i,x in enumerate(grid.x):
        expected=quad(lambda z:spec.mean+spec.amplitude*np.exp(-spec.base_diffusivity*np.pi**2*time/spec.length_x_m**2)*
            (np.cos(np.pi*z/spec.length_x_m)+spec.shape_b*np.cos(3*np.pi*z/spec.length_x_m)),
            x-grid.dx/2,x+grid.dx/2,epsabs=1e-12,epsrel=1e-12)[0]/grid.dx
        np.testing.assert_allclose(actual[:,i],expected,rtol=2e-15,atol=2e-15)


def test_smooth_unforced_pde_identity_and_closed_face_flux():
    spec=SpatialSpec(nx_values=(8,16),ny=2,shape_b=.08)
    grid=Grid((0,0,spec.length_x_m,spec.length_y_m),16,2)
    theta=np.pi*grid.x/spec.length_x_m
    derivative=-np.pi/spec.length_x_m*(np.sin(theta)+3*spec.shape_b*np.sin(3*theta))
    product=smooth_coefficient(grid,spec.base_diffusivity,spec.shape_b)[0]*derivative
    independent=-spec.base_diffusivity*np.pi/spec.length_x_m*(np.sin(theta)+spec.shape_b/3*np.sin(3*theta))
    np.testing.assert_allclose(product,independent,rtol=2e-15,atol=2e-15)
    # Analytic flux derivative is -D*pi^2/L^2*(cos(theta)+b*cos(3theta)),
    # and both boundary fluxes vanish; no artificial volume source is needed.
    for wall in (0,np.pi):
        assert abs(np.sin(wall)+spec.shape_b/3*np.sin(3*wall))<2e-15


def test_independent_face_reference_counts_both_periodic_two_cell_faces():
    grid=Grid((0,0,2,2),2,2);coefficient=np.array([[1.,3.],[2.,4.]])
    closed=independent_dense_operator(grid,coefficient)
    periodic=independent_dense_operator(grid,coefficient,"periodic")
    np.testing.assert_allclose(periodic,2*closed,rtol=3e-16,atol=2e-15)
    np.testing.assert_allclose(periodic.sum(axis=1),0,atol=1e-14)
    np.testing.assert_array_equal(periodic,periodic.T)
    for bad in (np.zeros((2,2)),np.ones((1,2)),np.full((2,2),np.nan)):
        with pytest.raises(ValueError):independent_dense_operator(grid,bad)


def test_spatial_convergence_separates_cn_time_error(study):
    report,raw=study;rows=report["spatial"]["cases"]
    assert all(row["inputs_unmodified"] for row in rows)
    assert all(row["time_error_below_ten_percent_of_spatial"] for row in rows)
    assert all(1.9<row["observed_rms_order"]<2.1 for row in rows[1:])
    assert rows[-1]["spatial_error"]["rms"]<rows[0]["spatial_error"]["rms"]/14
    for row in rows:
        key=f"spatial_n{row['nx']}__"
        assert raw[key+"kappa"].min()>0
        assert raw[key+"initial"].min()>0
        assert np.max(abs(raw[key+"numerical"]-raw[key+"exact_cell_average"]))==row["total_error"]["linf"]


def test_temporal_first_second_orders_against_independent_dense_reference(study):
    report,_=study
    for boundary in report["temporal"]["boundaries"]:
        assert boundary["operator_max_absolute_difference"]<1e-11
        assert boundary["independent_matrix_symmetry_residual"]==0
        assert boundary["independent_matrix_row_sum_residual"]<1e-12
        assert boundary["largest_eigenvalue"]<1e-11
        for method,rows in boundary["methods"].items():
            target=2 if method=="crank_nicolson" else 1
            assert abs(rows[-1]["observed_rms_order"]-target)<.08
            assert all(b["error"]["rms"]<a["error"]["rms"] for a,b in zip(rows,rows[1:]))


def test_layered_serial_resistance_and_all_face_fluxes(study):
    report,raw=study;layer=report["layered"];flux=layer["analytic_constant_flux"]
    assert flux==pytest.approx(1/(.5/1+.5/20),rel=1e-15)
    harmonic=layer["results"]["harmonic"]
    assert harmonic["error"]["linf"]<1e-12
    np.testing.assert_allclose(raw["layered__harmonic_flux_x"],flux,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(raw["layered__harmonic_flux_y"],0,atol=1e-12)
    assert layer["results"]["arithmetic"]["error"]["linf"]>.01
    assert layer["results"]["arithmetic"]["flux_x_max_error"]>.05
    # The saved production matrix remains closed; the boundary terms occur
    # in a separate validation copy, not a new production boundary option.
    np.testing.assert_allclose(raw["layered__production_closed_operator"].sum(axis=1),0,atol=1e-12)
    assert not np.array_equal(raw["layered__production_closed_operator"],raw["layered__harmonic_dirichlet_operator"])


@pytest.mark.parametrize("section,changes",[
    ("spatial",{"shape_b":1/9}), ("spatial",{"shape_b":0}),
    ("spatial",{"nx_values":[16,8]}), ("spatial",{"dt_s":1e-320}),
    ("spatial",{"length_x_m":1e-8}),
    ("temporal",{"duration_cfl_multiple":32}), ("temporal",{"boundaries":["open"]}),
    ("temporal",{"boundaries":["periodic","periodic"]}), ("temporal",{"nx":24,"ny":16}),
    ("layered",{"interface_fraction":.31}), ("layered",{"kappa_right":1}),
    ("budget",{"max_array_bytes":1024}), ("budget",{"max_cell_updates":10}),
])
def test_invalid_or_over_budget_protocol_rejected_before_solver(quick_config,section,changes,monkeypatch):
    from scripts import run_variable_diffusion_experiments as module
    config=copy.deepcopy(quick_config);config[section].update(changes)
    def forbidden(*args,**kwargs):raise AssertionError("Solver must not run for rejected configuration")
    monkeypatch.setattr(module,"solve",forbidden)
    with pytest.raises(ValueError):run_study(config)


def test_default_schema_and_aggregate_raw_array_budget(quick_config,study):
    spec=Study.model_validate(quick_config);report,raw=study
    expected=expected_raw_shapes(spec)
    assert set(raw)==set(expected)
    assert sum(array.nbytes for array in raw.values())==check_budget(spec)["raw_array_bytes"]
    assert report["admission"]["raw_array_count"]==len(raw)
    default=Study.model_validate_json(DEFAULT_CONFIG.read_text())
    assert list(default.spatial.nx_values)==[16,32,64,128]
    with pytest.raises(ValueError):Study.model_validate({**quick_config,"schema_version":999})
    with pytest.raises(ValueError):Study.model_validate({**quick_config,"unrecorded_switch":True})


@pytest.fixture(scope="module")
def saved_bundle(tmp_path_factory,quick_config):
    output=tmp_path_factory.mktemp("variable-evidence")/"bundle"
    write_bundle(quick_config,output)
    return output


def rehash(bundle,name):
    manifest=json.loads((bundle/"manifest.json").read_text());data=(bundle/name).read_bytes()
    manifest["files"][name]={"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
    (bundle/"manifest.json").write_text(json.dumps(manifest))


def test_atomic_bundle_verification_and_raw_array_inventory(saved_bundle,quick_config):
    result=verify_bundle(saved_bundle)
    assert result["status"]=="verified" and result["arrays"]==55
    with pytest.raises(FileExistsError):write_bundle(quick_config,saved_bundle)
    report=json.loads((saved_bundle/"report.json").read_text())
    assert "backend/app/numerics/coefficients.py" in report["source_snapshot_sha256"]
    assert (saved_bundle/"variable_evidence.png").stat().st_size>1000


@pytest.mark.parametrize("damage",["shape","nonfinite","extra_array","coefficient_sign","source","config","case","extra_file"])
def test_rehashed_corruption_does_not_bypass_schema(tmp_path,saved_bundle,damage):
    import shutil
    out=tmp_path/"damaged";shutil.copytree(saved_bundle,out)
    if damage in {"shape","nonfinite","extra_array","coefficient_sign"}:
        with np.load(out/"raw_fields.npz",allow_pickle=False) as values:raw={key:values[key] for key in values.files}
        if damage=="shape":raw["spatial_n8__initial"]=np.ones((2,2))
        elif damage=="nonfinite":raw["spatial_n8__initial"][0,0]=np.nan
        elif damage=="extra_array":raw["undeclared"]=np.ones(2)
        else:raw["spatial_n8__kappa"][0,0]=-1
        np.savez_compressed(out/"raw_fields.npz",**raw);rehash(out,"raw_fields.npz")
    elif damage in {"source","config","case"}:
        report=json.loads((out/"report.json").read_text())
        if damage=="source":report["source_snapshot_sha256"]["backend/app/numerics/operators.py"]="0"*64
        elif damage=="config":report["config"]["spatial"]["mean"]=42
        else:report["temporal"]["boundaries"][0]["methods"]["explicit_euler"][0]["requested_steps"]=999
        (out/"report.json").write_text(json.dumps(report));rehash(out,"report.json")
    else:(out/"unexpected.txt").write_text("not listed")
    with pytest.raises(ValueError):verify_bundle(out)


def test_failed_render_cannot_publish_partial_bundle(tmp_path,quick_config,monkeypatch):
    from scripts import run_variable_diffusion_experiments as module
    def fail(*args):raise OSError("simulated full disk")
    monkeypatch.setattr(module,"plot_report",fail)
    output=tmp_path/"failed"
    with pytest.raises(OSError,match="full disk"):write_bundle(quick_config,output)
    assert not output.exists() and not list(tmp_path.glob(".failed-*"))


def test_zero_represented_error_is_rendered_as_unresolved(study):
    from scripts.run_variable_diffusion_experiments import spatial_study, _markdown
    small=SpatialSpec(nx_values=(4,8),ny=2,base_diffusivity=1e-300,duration_s=.1,dt_s=.1)
    spatial,_=spatial_study(small)
    assert all(row["temporal_to_spatial_rms_ratio"] is None for row in spatial["cases"])
    report=copy.deepcopy(study[0]);report["spatial"]=spatial
    assert "unresolved (zero represented spatial error)" in _markdown(report)


@pytest.mark.parametrize("damage",["huge_shape","short_payload","object_dtype","duplicate_member"])
def test_npy_headers_are_rejected_before_numpy_load(tmp_path,saved_bundle,monkeypatch,damage):
    import io
    import shutil
    import zipfile
    from scripts import run_variable_diffusion_experiments as module
    out=tmp_path/"forged";shutil.copytree(saved_bundle,out)
    path=out/"raw_fields.npz";target="spatial_n8__initial.npy"
    with zipfile.ZipFile(path) as archive:entries={name:archive.read(name) for name in archive.namelist()}
    if damage!="duplicate_member":
        header=io.BytesIO()
        np.lib.format.write_array_header_1_0(header,{
            "descr":"|O" if damage=="object_dtype" else "<f8", "fortran_order":False,
            "shape":(2**30,2**30) if damage=="huge_shape" else (2,8)})
        entries[target]=header.getvalue()
    with zipfile.ZipFile(path,"w") as archive:
        for name,data in entries.items():archive.writestr(name,data)
        if damage=="duplicate_member":
            with pytest.warns(UserWarning,match="Duplicate name"):archive.writestr(target,entries[target])
    rehash(out,"raw_fields.npz")
    def forbidden(*args,**kwargs):raise AssertionError("np.load must not run before rejected NPY header validation")
    monkeypatch.setattr(module.np,"load",forbidden)
    with pytest.raises(ValueError):verify_bundle(out)


def test_replot_uses_saved_data_and_exports_outside_sealed_bundle(saved_bundle,monkeypatch):
    from scripts import run_variable_diffusion_experiments as module
    def forbidden(*args):raise AssertionError("Replot must not recompute the study")
    monkeypatch.setattr(module,"run_study",forbidden)
    before=(saved_bundle/"manifest.json").read_bytes()
    main(["--replot",str(saved_bundle)])
    assert saved_bundle.with_name(saved_bundle.name+"-replot.png").exists()
    assert (saved_bundle/"manifest.json").read_bytes()==before
    assert verify_bundle(saved_bundle)["status"]=="verified"
    with pytest.raises(FileExistsError):main(["--replot",str(saved_bundle)])
