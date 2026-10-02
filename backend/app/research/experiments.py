"""Local, immutable research bundles with explicit numerical provenance.

The destination is created by one directory rename after every file is written
and validated. Existing bundles are never overwritten. Hash verification checks
integrity; it is not a scientific accuracy certificate or a trusted signature.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import tempfile
from time import perf_counter
import zipfile

import numpy as np

from ..numerics import solve
from .problems import ArrayInitial, ExperimentSpec, check_budget, content_id, initial_field, load_spec

ROOT = Path(__file__).resolve().parents[3]


def _json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def provenance() -> dict:
    sources = sorted((ROOT / "backend" / "app").rglob("*.py"))
    sources += sorted((ROOT / "cpp" / "src").glob("*"))
    sources += [ROOT / "cpp" / "CMakeLists.txt", ROOT / "cpp" / "pyproject.toml",
                ROOT / "cpp" / "source_hash.py", ROOT / "requirements.lock.txt",
                ROOT / "scripts" / "run_research.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "git_dirty": dirty, "source_sha256": hashes, "source_id": content_id(hashes),
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in ["numpy", "scipy", "pydantic"]}}


def verify_bundle(path: Path | str) -> dict:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "completed":
        raise ValueError("Unsupported or incomplete research bundle.")
    required = {"config.json", "result.json", "fields.npz"}
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict) or not required <= set(artifacts):
        raise ValueError("Research bundle is missing required artifact records.")
    for name, details in artifacts.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name or len(rel.parts) != 1:
            raise ValueError("Unsafe bundle artifact path.")
        file = root / name
        if file.is_symlink() or not file.is_file():
            raise ValueError(f"Missing or nonregular bundle artifact: {name}")
        if file.stat().st_size != details["size_bytes"] or hashlib.sha256(file.read_bytes()).hexdigest() != details["sha256"]:
            raise ValueError(f"Research bundle checksum mismatch: {name}")
    expected = set(artifacts) | {"manifest.json"}
    if {f.name for f in root.iterdir()} != expected:
        raise ValueError("Research bundle contains unlisted artifacts.")
    spec = load_spec(root / "config.json")
    result = json.loads((root / "result.json").read_text())
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise ValueError("Unsupported research result schema version.")
    if result.get("identities") != spec.identities():
        raise ValueError("Bundle result/configuration identity mismatch.")
    if isinstance(spec.problem.initial, ArrayInitial):
        initial = spec.problem.initial
        if initial.path != "initial_input.npz" or initial.path not in artifacts:
            raise ValueError("Imported initial data must be archived as manifest-listed initial_input.npz.")
        if artifacts[initial.path]["sha256"] != initial.sha256:
            raise ValueError("Archived initial input checksum differs from the configuration.")
    max_fields = spec.budget.max_output_bytes + spec.numerical.nx*spec.numerical.ny*8 + 1_000_000
    with zipfile.ZipFile(root / "fields.npz") as archive:
        if sum(i.file_size for i in archive.infolist()) > max_fields:
            raise ValueError("Archived fields exceed declared array budget.")
    with np.load(root / "fields.npz", allow_pickle=False) as fields:
        if set(fields.files) != {"initial", "times_s", "frames"}:
            raise ValueError("Unexpected field arrays.")
        n = spec.numerical
        if fields["frames"].shape != (len(n.output_times_s), n.ny, n.nx) or fields["initial"].shape != (n.ny, n.nx):
            raise ValueError("Archived field dimensions do not match the configuration.")
        if not np.array_equal(fields["times_s"], n.output_times_s):
            raise ValueError("Archived physical times do not match the configuration.")
        for key in fields.files:
            values = fields[key]
            if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
                raise ValueError("Archived arrays must be finite real numbers.")
        # The archive's checksums, dimensions and finite-value checks establish
        # integrity without re-evaluating an analytic formula using today's math
        # libraries. Replay performs that numerical evaluation separately.
        # Imported data have a preserved byte-identical input, so their converted
        # values must exactly match the initial array actually archived.
        if isinstance(spec.problem.initial, ArrayInitial):
            if not np.array_equal(fields["initial"], initial_field(spec, root)):
                raise ValueError("Archived initial values differ from the preserved imported input.")
    return {"status": "verified", "configuration_id": spec.identities()["configuration_id"],
            "artifacts": len(artifacts), "validation_scope": "bundle integrity and schema, not scientific accuracy"}


def run_experiment(spec: ExperimentSpec | dict, output: Path | str, *, base_dir: Path | str = ".") -> dict:
    spec = spec if isinstance(spec, ExperimentSpec) else ExperimentSpec.model_validate(spec)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Research bundle already exists: {output}")
    admission = check_budget(spec)
    initial = initial_field(spec, base_dir)
    n, p = spec.numerical, spec.problem
    start = perf_counter()
    frames, diagnostics = solve(spec.grid(), initial, p.kappa, n.output_times_s,
                               method=n.method, backend=n.backend, dt=n.dt, boundary=p.boundary,
                               velocity=p.velocity, startup=n.startup)
    elapsed = (perf_counter()-start)*1000
    result = {"schema_version": 1, "identities": spec.identities(),
              "grid": spec.grid().to_dict(), "initialization": n.initialization,
              "diagnostics": diagnostics, "solve_call_ms": elapsed,
              "admission_estimate": admission, "provenance": provenance(),
              "archive_mode": "full_initial_and_output_arrays",
              "claims": "synthetic or imported numerical problem; no observational calibration implied"}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        archived = spec.model_dump(mode="json")
        if isinstance(p.initial, ArrayInitial):
            source = Path(base_dir) / p.initial.path
            shutil.copyfile(source, stage / "initial_input.npz")
            archived["problem"]["initial"]["path"] = "initial_input.npz"
        _json(stage / "config.json", archived)
        _json(stage / "result.json", result)
        with (stage / "fields.npz").open("wb") as handle:
            np.savez_compressed(handle, initial=initial, times_s=np.asarray(n.output_times_s), frames=frames)
            handle.flush()
            os.fsync(handle.fileno())
        artifacts = {f.name: {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "size_bytes": f.stat().st_size}
                     for f in sorted(stage.iterdir())}
        _json(stage / "manifest.json", {"schema_version": 1, "status": "completed", "artifacts": artifacts})
        verify_bundle(stage)
        # A competing completed writer has a nonempty destination, which rename
        # refuses; callers must assign a distinct output name per execution.
        if output.exists():
            raise FileExistsError(f"Research bundle already exists: {output}")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {**result, "output": str(output)}


def replay_bundle(path: Path | str, output: Path | str) -> dict:
    path = Path(path)
    verify_bundle(path)
    result = run_experiment(load_spec(path / "config.json"), output, base_dir=path)
    with np.load(path / "fields.npz", allow_pickle=False) as prior, np.load(Path(output) / "fields.npz", allow_pickle=False) as current:
        difference = current["frames"]-prior["frames"]
        comparison = {"linf": float(np.max(np.abs(difference))), "rms": float(np.sqrt(np.mean(difference**2))),
                      "comparison_scope": "recomputed raw frames; not a cross-platform bitwise guarantee"}
    return {**result, "replay_comparison": comparison}
