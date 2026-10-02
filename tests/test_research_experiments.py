"""Research problem semantics and artifact integrity, independently of HTTP."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
from scipy.integrate import quad

from backend.app.research import experiments
from backend.app.research.problems import ExperimentSpec, check_budget, initial_field


def config():
    return {"problem": {"bounds": [0, 0, 2, 1.5], "kappa": 0.07,
                        "initial": {"kind": "cosine", "modes": [1, 1]}},
            "numerical": {"nx": 12, "ny": 10, "output_times_s": [0, 0.1],
                          "method": "crank_nicolson", "backend": "scipy", "dt": 0.01}}


def test_physical_identity_does_not_change_with_discretization():
    a = ExperimentSpec.model_validate(config())
    values = config()
    values["numerical"].update(nx=24, method="backward_euler")
    b = ExperimentSpec.model_validate(values)
    assert a.identities()["physical_id"] == b.identities()["physical_id"]
    assert a.identities()["numerical_id"] != b.identities()["numerical_id"]
    values["problem"]["kappa"] = 0.08
    c = ExperimentSpec.model_validate(values)
    assert b.identities()["physical_id"] != c.identities()["physical_id"]


def test_cosine_is_projected_to_cell_averages():
    spec = ExperimentSpec.model_validate(config())
    grid = spec.grid()
    values = initial_field(spec)
    xx = np.cos(np.pi*grid.x/2)*np.sinc(grid.dx/4)
    yy = np.cos(np.pi*grid.y/1.5)*np.sinc(grid.dy/3)
    np.testing.assert_allclose(values, 1+0.5*yy[:, None]*xx[None, :], atol=1e-15)
    changed = config()
    changed["numerical"]["initialization"] = "point_sample"
    assert not np.array_equal(values, initial_field(ExperimentSpec.model_validate(changed)))


def test_gaussian_average_matches_independent_quadrature():
    values = config()
    values["problem"]["initial"] = {"kind": "gaussian", "center": [0.31, 0.47],
                                    "sigma_m": 0.2, "amplitude": 2, "background": 0.3}
    spec = ExperimentSpec.model_validate(values)
    g = spec.grid()
    integral_x = quad(lambda x: np.exp(-0.5*((x-0.31)/0.2)**2), 0, g.dx)[0] / g.dx
    integral_y = quad(lambda y: np.exp(-0.5*((y-0.47)/0.2)**2), 0, g.dy)[0] / g.dy
    assert initial_field(spec)[0, 0] == pytest.approx(0.3+2*integral_x*integral_y, abs=1e-14)


@pytest.mark.parametrize("change", [
    {"schema_version": 2},
    {"problem": {"velocity": [1, 0]}},
    {"numerical": {"output_times_s": [0.1, 0.1]}},
    {"numerical": {"backend": "cpp"}},
    {"numerical": {"dt": float("nan")}},
    {"problem": {"boundary": "periodic"}},
    {"problem": {"unimplemented_source": 1}},
])
def test_invalid_or_unsupported_contract_rejected(change):
    values = config()
    for key, item in change.items():
        if isinstance(item, dict):
            values[key].update(item)
        else:
            values[key] = item
    with pytest.raises(ValueError):
        ExperimentSpec.model_validate(values)


def test_work_and_output_budget_reject_before_solve():
    values = config()
    values["budget"] = {"max_cell_updates": 1}
    with pytest.raises(ValueError, match="work budget"):
        check_budget(ExperimentSpec.model_validate(values))
    values["budget"] = {"max_output_bytes": 1024}
    with pytest.raises(ValueError, match="byte budget"):
        check_budget(ExperimentSpec.model_validate(values))


def test_immutable_bundle_and_replay(tmp_path):
    target = tmp_path / "first"
    result = experiments.run_experiment(config(), target)
    assert result["archive_mode"] == "full_initial_and_output_arrays"
    assert experiments.verify_bundle(target)["status"] == "verified"
    replay = experiments.replay_bundle(target, tmp_path / "replay")
    assert replay["replay_comparison"]["linf"] == 0
    with pytest.raises(FileExistsError):
        experiments.run_experiment(config(), target)
    with (target / "fields.npz").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        experiments.verify_bundle(target)


def test_failed_publication_leaves_no_completed_bundle(tmp_path, monkeypatch):
    original = experiments._json
    def fail_manifest(path, value):
        if path.name == "manifest.json":
            raise OSError("simulated disk failure")
        return original(path, value)
    monkeypatch.setattr(experiments, "_json", fail_manifest)
    with pytest.raises(OSError, match="disk failure"):
        experiments.run_experiment(config(), tmp_path / "run")
    assert list(tmp_path.iterdir()) == []


def test_array_input_is_portable_and_hash_verified(tmp_path):
    source = tmp_path / "input.npz"
    data = np.ones((10, 12))
    np.savez(source, values=data)
    values = config()
    values["problem"]["initial"] = {"kind": "npz", "path": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "bounds": [0, 0, 2, 1.5], "nx": 12, "ny": 10}
    spec = ExperimentSpec.model_validate(values)
    experiments.run_experiment(spec, tmp_path / "bundle", base_dir=tmp_path)
    source.unlink()
    assert experiments.verify_bundle(tmp_path / "bundle")["status"] == "verified"
    assert experiments.replay_bundle(tmp_path / "bundle", tmp_path / "replay")["replay_comparison"]["linf"] == 0
    changed = deepcopy(values)
    changed["problem"]["initial"]["path"] = "a_different_path.npz"
    assert ExperimentSpec.model_validate(changed).identities() == spec.identities()


def test_bundle_path_traversal_and_unlisted_content_rejected(tmp_path):
    target = tmp_path / "bundle"
    experiments.run_experiment(config(), target)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["../outside"] = {"sha256": "0"*64, "size_bytes": 0}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe"):
        experiments.verify_bundle(target)


def test_array_grid_mismatch_cannot_be_silently_resampled():
    values = config()
    values["problem"]["initial"] = {"kind": "npz", "path": "unused.npz", "sha256": "0"*64,
                                     "bounds": [0, 0, 2, 1.5], "nx": 24, "ny": 20}
    with pytest.raises(ValueError, match="resampling"):
        ExperimentSpec.model_validate(values)


def refresh_artifact_record(bundle, name):
    """Keep checksums valid while testing semantic validation separately."""
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifact = bundle / name
    manifest["artifacts"][name] = {
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "size_bytes": artifact.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest))


def create_array_bundle(tmp_path):
    source = tmp_path / "input.npz"
    np.savez(source, values=np.ones((10, 12)))
    values = config()
    values["problem"]["initial"] = {"kind": "npz", "path": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "bounds": [0, 0, 2, 1.5], "nx": 12, "ny": 10}
    bundle = tmp_path / "bundle"
    experiments.run_experiment(values, bundle, base_dir=tmp_path)
    return bundle, source


def test_analytic_bundle_verification_is_independent_of_math_library_roundoff(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    experiments.run_experiment(config(), bundle)
    evaluator = experiments.initial_field
    monkeypatch.setattr(experiments, "initial_field",
                        lambda *args, **kwargs: np.nextafter(evaluator(*args, **kwargs), np.inf))
    assert experiments.verify_bundle(bundle)["status"] == "verified"
    replay = experiments.replay_bundle(bundle, tmp_path / "replay")
    assert replay["replay_comparison"]["linf"] > 0
    assert replay["replay_comparison"]["linf"] < 1e-12


def test_bundle_rejects_unknown_result_schema_with_valid_checksum(tmp_path):
    bundle = tmp_path / "bundle"
    experiments.run_experiment(config(), bundle)
    result_path = bundle / "result.json"
    result = json.loads(result_path.read_text())
    result["schema_version"] = 999
    result_path.write_text(json.dumps(result))
    refresh_artifact_record(bundle, "result.json")
    with pytest.raises(ValueError, match="result schema"):
        experiments.verify_bundle(bundle)


@pytest.mark.parametrize("external_path", ["absolute", "parent_relative", "missing_archive"])
def test_imported_bundle_requires_manifest_listed_in_bundle_input(tmp_path, external_path):
    bundle, source = create_array_bundle(tmp_path)
    config_path = bundle / "config.json"
    values = json.loads(config_path.read_text())
    if external_path != "missing_archive":
        values["problem"]["initial"]["path"] = str(source) if external_path == "absolute" else "../input.npz"
        config_path.write_text(json.dumps(values))
        refresh_artifact_record(bundle, "config.json")
    (bundle / "initial_input.npz").unlink()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"].pop("initial_input.npz")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest-listed initial_input"):
        experiments.verify_bundle(bundle)


def test_imported_archive_hash_must_match_physical_configuration(tmp_path):
    bundle, _ = create_array_bundle(tmp_path)
    np.savez(bundle / "initial_input.npz", values=np.full((10, 12), 2.0))
    refresh_artifact_record(bundle, "initial_input.npz")
    with pytest.raises(ValueError, match="initial input checksum"):
        experiments.verify_bundle(bundle)


def test_imported_initial_values_must_match_archived_input_exactly(tmp_path):
    bundle, _ = create_array_bundle(tmp_path)
    with np.load(bundle / "fields.npz", allow_pickle=False) as archive:
        fields = {key: archive[key].copy() for key in archive.files}
    fields["initial"][0, 0] = np.nextafter(fields["initial"][0, 0], np.inf)
    np.savez_compressed(bundle / "fields.npz", **fields)
    refresh_artifact_record(bundle, "fields.npz")
    with pytest.raises(ValueError, match="initial values differ"):
        experiments.verify_bundle(bundle)
