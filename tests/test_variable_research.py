"""Variable material configuration, real CFL budgets and portable input archives."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from backend.app.research import experiments
from backend.app.research.problems import (ExperimentSpec, capabilities, check_budget,
                                         diffusivity_field, load_spec)


def config():
    return {"problem": {"bounds": [0, 0, 2, 1],
                        "kappa": {"kind": "layered", "axis": "x", "interface_m": 1,
                                  "left": .1, "right": 1},
                        "initial": {"kind": "cosine", "modes": [1, 0]}},
            "numerical": {"nx": 8, "ny": 4, "output_times_s": [0, .02],
                          "method": "crank_nicolson", "backend": "scipy", "dt": .002}}


def imported(tmp_path, values=None):
    values = np.arange(1, 33, dtype=np.float32).reshape(4, 8)/32 if values is None else values
    source = tmp_path / "material.npz"
    np.savez_compressed(source, values=values)
    settings = config()
    settings["problem"]["kappa"] = {"kind": "npz", "path": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bounds": [0, 0, 2, 1],
        "nx": 8, "ny": 4}
    return settings, source


def rehash(bundle, name):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    file = bundle / name
    manifest["artifacts"][name] = {"sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                                    "size_bytes": file.stat().st_size}
    path.write_text(json.dumps(manifest))


def test_existing_scalar_configuration_id_is_unchanged():
    root = Path(__file__).resolve().parents[1]
    spec = load_spec(root / "configs/research/cosine_diffusion.json")
    assert spec.identities() == {
        "physical_id": "10098008cbee0282672a3bbbe41eedeba4ead1130f1d13cf94ab3676228f7df4",
        "numerical_id": "def9c77fc5b72d9abbc634e43cdd9892c678df0c70e8bf134ab4a0e8d210b5ba",
        "configuration_id": "cc0c6576d77b611821326b02ac19fbb1ceab7ba09a1050d42d9fb8ef92a060a9"}
    assert isinstance(spec.problem.kappa, float)
    settings = config(); settings["problem"]["kappa"] = 0
    assert diffusivity_field(ExperimentSpec.model_validate(settings)) == 0.0


@pytest.mark.parametrize("axis,interface,split", [("x", 1, 4), ("y", .5, 2)])
def test_layered_physical_interface_defines_cell_materials(axis, interface, split):
    settings = config(); settings["problem"]["kappa"].update(axis=axis, interface_m=interface)
    spec = ExperimentSpec.model_validate(settings)
    values = diffusivity_field(spec)
    assert values.shape == (4, 8)
    if axis == "x":
        np.testing.assert_array_equal(values[:, :split], .1)
        np.testing.assert_array_equal(values[:, split:], 1)
    else:
        np.testing.assert_array_equal(values[:split], .1)
        np.testing.assert_array_equal(values[split:], 1)
    changed = deepcopy(settings); changed["numerical"].update(nx=16, ny=8)
    assert ExperimentSpec.model_validate(changed).identities()["physical_id"] == spec.identities()["physical_id"]


@pytest.mark.parametrize("change", [
    {"interface_m": .7}, {"interface_m": 0}, {"interface_m": 2},
    {"left": 0}, {"right": -1}, {"right": np.inf}, {"axis": "z"},
])
def test_invalid_layer_material_or_subcell_interface_rejected(change):
    settings = config(); settings["problem"]["kappa"].update(change)
    with pytest.raises(ValueError):
        ExperimentSpec.model_validate(settings)


def test_decimal_grid_face_is_recognized_without_admitting_a_subcell_interface():
    settings = config()
    settings["problem"].update(bounds=[0, 0, 1, 1])
    settings["problem"]["kappa"]["interface_m"] = .3
    settings["numerical"]["nx"] = 10
    field = diffusivity_field(ExperimentSpec.model_validate(settings))
    np.testing.assert_array_equal(field[:, :3], .1)
    settings["problem"]["kappa"]["interface_m"] += 1e-10
    with pytest.raises(ValueError, match="grid face"):
        ExperimentSpec.model_validate(settings)


@pytest.mark.parametrize("offset", [0, 500000, -500000])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_physical_layer_faces_are_valid_after_coordinate_translation(offset, axis):
    settings = config()
    settings["problem"]["bounds"] = ([offset, 0, offset + 1, 1] if axis == "x"
                                      else [0, offset, 1, offset + 1])
    settings["numerical"].update(nx=10 if axis == "x" else 2, ny=2 if axis == "x" else 10)
    settings["problem"]["kappa"].update(axis=axis, interface_m=offset + .3)
    values = diffusivity_field(ExperimentSpec.model_validate(settings))
    if axis == "y":
        values = values.T
    np.testing.assert_array_equal(values[:, :3], .1)
    np.testing.assert_array_equal(values[:, 3:], 1)
    # Translation must not turn a true subcell interface into a material face.
    settings["problem"]["kappa"]["interface_m"] = offset + .300001
    with pytest.raises(ValueError, match="grid face"):
        ExperimentSpec.model_validate(settings)


@pytest.mark.parametrize("width", [.25, 8])
def test_layer_grid_rejects_coordinate_precision_that_cannot_resolve_faces(width):
    settings = config()
    offset = float(2**50)  # Coordinate ULP is 0.25 m.
    settings["problem"]["bounds"] = [offset, 0, offset + width, 1]
    settings["numerical"].update(nx=10, ny=2)
    settings["problem"]["kappa"]["interface_m"] = offset + (width / 2)
    # Width .25 cannot represent even the intended interior interface; width
    # 8 has representable faces but too few ULPs for reliable alignment checks.
    with pytest.raises(ValueError, match="physical domain|physical-coordinate precision"):
        ExperimentSpec.model_validate(settings)


@pytest.mark.parametrize("change", ["cpp", "transport", "open"])
def test_unsupported_variable_equations_and_backends_rejected(change):
    settings = config()
    if change == "cpp":
        settings["numerical"].update(method="explicit_euler", backend="cpp")
    elif change == "transport":
        settings["problem"].update(model="advection_diffusion", boundary="periodic")
        settings["problem"]["initial"] = {"kind": "constant"}
        settings["numerical"].update(method="advection_explicit", backend="numpy")
    else:
        settings["problem"]["boundary"] = "open"
    with pytest.raises(ValueError):
        ExperimentSpec.model_validate(settings)


def test_field_cfl_budget_uses_face_outgoing_rates_and_boundary():
    settings = config()
    settings["problem"].update(bounds=[0, 0, 2, 2], initial={"kind": "constant"})
    settings["problem"]["kappa"].update(left=1, right=9)
    settings["numerical"].update(nx=2, ny=2, method="explicit_euler", backend="auto", dt=None,
                                  output_times_s=[0, .2])
    settings["budget"] = {"max_cell_updates": 20}
    admission = check_budget(ExperimentSpec.model_validate(settings))
    # Interior horizontal harmonic face is 1.8. A right cell has one vertical
    # material face of 9: rate=10.8. A maximum-kappa scalar guess would be 36.
    assert admission["cfl_limit_s"] == pytest.approx(1/10.8)
    assert admission["admission_dt_s"] == pytest.approx(.9/10.8)
    assert admission["estimated_steps_upper"] == 5
    settings["problem"]["boundary"] = "periodic"
    with pytest.raises(ValueError, match="work budget"):
        check_budget(ExperimentSpec.model_validate(settings))


def test_output_preflight_precedes_input_reads_and_allocations(tmp_path, monkeypatch):
    settings, _ = imported(tmp_path)
    settings["numerical"].update(output_times_s=list(range(20)))
    settings["budget"] = {"max_output_bytes": 1024}
    def forbidden(*args, **kwargs):
        raise AssertionError("An over-budget output must not resolve input arrays.")
    monkeypatch.setattr(experiments, "diffusivity_field", forbidden)
    with pytest.raises(ValueError, match="byte budget"):
        experiments.run_experiment(settings, tmp_path / "never", base_dir=tmp_path)
    assert not (tmp_path / "never").exists()


def test_work_preflight_precedes_initial_condition_allocation(tmp_path, monkeypatch):
    settings = config(); settings["budget"] = {"max_cell_updates": 1}
    def forbidden(*args, **kwargs):
        raise AssertionError("Work admission must precede initial construction.")
    monkeypatch.setattr(experiments, "initial_field", forbidden)
    with pytest.raises(ValueError, match="work budget"):
        experiments.run_experiment(settings, tmp_path / "never")


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_imported_material_requires_strictly_positive_finite_values(tmp_path, bad):
    values = np.ones((4, 8)); values[0, 0] = bad
    settings, _ = imported(tmp_path, values)
    with pytest.raises(ValueError, match="strictly positive"):
        diffusivity_field(ExperimentSpec.model_validate(settings), tmp_path)


@pytest.mark.parametrize("change", [{"nx": 16}, {"bounds": [0, 0, 3, 1]},
                                    {"unit": "relative"}, {"axis_order": "x,y"},
                                    {"interpretation": "cell_average"}])
def test_imported_material_geometry_and_unit_contract(tmp_path, change):
    settings, _ = imported(tmp_path); settings["problem"]["kappa"].update(change)
    with pytest.raises(ValueError):
        ExperimentSpec.model_validate(settings)


def test_import_identity_uses_content_and_geometry_not_machine_path(tmp_path):
    settings, _ = imported(tmp_path)
    first = ExperimentSpec.model_validate(settings)
    settings["problem"]["kappa"]["path"] = "/different/machine/material.npz"
    assert ExperimentSpec.model_validate(settings).identities() == first.identities()
    settings["problem"]["kappa"]["sha256"] = "0"*64
    assert ExperimentSpec.model_validate(settings).identities()["physical_id"] != first.identities()["physical_id"]


def test_import_header_budget_rejects_forged_large_shape_before_numpy_load(tmp_path, monkeypatch):
    settings, source = imported(tmp_path)
    forged = io.BytesIO()
    np.lib.format.write_array_header_1_0(forged, {"descr": "<f8", "fortran_order": False,
                                               "shape": (10**9, 10**9)})
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("values.npy", forged.getvalue())
    settings["problem"]["kappa"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    def forbidden(*args, **kwargs):
        raise AssertionError("Forged shape must be rejected before np.load.")
    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="header shape"):
        diffusivity_field(ExperimentSpec.model_validate(settings), tmp_path)


@pytest.mark.parametrize("method,backend", [("explicit_euler", "auto"),
                                           ("backward_euler", "scipy"), ("crank_nicolson", "scipy")])
def test_variable_bundle_records_exact_coefficient_and_replays_all_arrays(tmp_path, method, backend):
    settings = config(); settings["numerical"].update(method=method, backend=backend, dt=.001)
    result = experiments.run_experiment(settings, tmp_path / "bundle")
    assert result["diffusivity"]["shape"] == [4, 8]
    assert result["diagnostics"]["diffusivity"]["type"] == "cell_centred_field"
    if method == "explicit_euler":
        assert result["diagnostics"]["backend"] == "numpy"
        assert result["diagnostics"]["backend_selection"] == "variable_diffusivity_capability"
    assert experiments.verify_bundle(tmp_path / "bundle")["status"] == "verified"
    replay = experiments.replay_bundle(tmp_path / "bundle", tmp_path / "replay")
    comparison = replay["replay_comparison"]
    assert comparison["all_arrays_exact"] and comparison["linf"] == 0
    assert set(comparison["arrays"]) == {"initial", "times_s", "frames", "diffusivity"}


def test_imported_input_is_preserved_byte_exact_and_independent_of_original_or_caches(tmp_path):
    settings, source = imported(tmp_path)
    original_bytes = source.read_bytes()
    result = experiments.run_experiment(settings, tmp_path / "bundle", base_dir=tmp_path)
    assert (tmp_path / "bundle/kappa_input.npz").read_bytes() == original_bytes
    source.unlink()
    assert experiments.verify_bundle(tmp_path / "bundle")["status"] == "verified"
    replay = experiments.replay_bundle(tmp_path / "bundle", tmp_path / "replay")
    assert replay["replay_comparison"]["all_arrays_exact"]
    assert replay["diffusivity"] == result["diffusivity"]
    archived = json.loads((tmp_path / "bundle/config.json").read_text())
    assert archived["problem"]["kappa"]["path"] == "kappa_input.npz"


@pytest.mark.parametrize("damage", ["schema", "hash", "shape", "zero", "imported_values", "external_path"])
def test_semantic_coefficient_archive_tampering_rejected_after_valid_checksums(tmp_path, damage):
    settings, _ = imported(tmp_path)
    bundle = tmp_path / "bundle"
    experiments.run_experiment(settings, bundle, base_dir=tmp_path)
    if damage in {"schema", "hash"}:
        path = bundle / "result.json"; result = json.loads(path.read_text())
        result["diffusivity"]["schema_version" if damage == "schema" else "values_sha256"] = 9 if damage == "schema" else "0"*64
        path.write_text(json.dumps(result)); rehash(bundle, path.name)
    elif damage == "external_path":
        path = bundle / "config.json"; value = json.loads(path.read_text())
        value["problem"]["kappa"]["path"] = "../material.npz"
        path.write_text(json.dumps(value)); rehash(bundle, path.name)
    else:
        with np.load(bundle / "fields.npz", allow_pickle=False) as raw:
            arrays = {key: raw[key] for key in raw.files}
        if damage == "shape": arrays["diffusivity"] = np.ones((2, 2))
        elif damage == "zero": arrays["diffusivity"][0, 0] = 0
        else:
            arrays["diffusivity"][0, 0] = np.nextafter(arrays["diffusivity"][0, 0], np.inf)
            path = bundle / "result.json"; result = json.loads(path.read_text())
            result["diffusivity"] = experiments._diffusivity_record(arrays["diffusivity"])
            path.write_text(json.dumps(result)); rehash(bundle, path.name)
        np.savez_compressed(bundle / "fields.npz", **arrays); rehash(bundle, "fields.npz")
    with pytest.raises(ValueError):
        experiments.verify_bundle(bundle)


def test_analytic_coefficient_integrity_does_not_reconstruct_material_formula(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    experiments.run_experiment(config(), bundle)
    def forbidden(*args, **kwargs):
        raise AssertionError("Integrity verification must use archived material values.")
    monkeypatch.setattr(experiments, "diffusivity_field", forbidden)
    assert experiments.verify_bundle(bundle)["status"] == "verified"


def test_capabilities_expose_variable_auto_and_rannacher_combinations():
    supported = capabilities()["variable_diffusivity_combinations"]
    assert {"model": "diffusion", "method": "crank_nicolson", "backend": "auto",
            "boundary": "periodic", "startup": "rannacher"} in supported
    assert all(case["backend"] != "cpp" and case["boundary"] != "open" for case in supported)
