"""Immutable dataset generations with an atomically replaced CURRENT.json pointer.

Readers resolve the pointer once and use that immutable directory for the entire
operation. Publication never mutates a visible generation. Atomic visibility uses
same-filesystem rename; fsync also requests local filesystem crash durability.
Concurrent writers are last-writer-wins. Old/orphan generations are retained so
active readers remain valid. Legacy flat data directories are read-only inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Callable
from uuid import uuid4

DATASET_SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def resolve_dataset_dir(path: Path | str, *, verify: bool = True) -> Path:
    """Resolve one generation; verify its complete manifest by default."""
    root = Path(path)
    pointer = root / "CURRENT.json"
    if not pointer.exists():
        # A failed first publication must not masquerade as a flat dataset.
        if (root / "versions").exists():
            raise ValueError(f"Dataset has no published generation: {root}")
        return root
    current = json.loads(pointer.read_text(encoding="utf-8"))
    if current.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("Unsupported dataset pointer schema version")
    version = current.get("version", "")
    if not isinstance(version, str) or len(version) != 32 or any(c not in "0123456789abcdef" for c in version):
        raise ValueError("Invalid dataset generation identifier")
    generation = root / "versions" / version
    manifest_path = generation / "manifest.json"
    if not manifest_path.is_file() or sha256(manifest_path) != current.get("manifest_sha256"):
        raise ValueError("Dataset manifest is missing or its hash does not match")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION or manifest.get("version") != version:
        raise ValueError("Unsupported or inconsistent dataset manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Dataset manifest has no artifacts")
    for name, metadata in artifacts.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Dataset artifact names must be plain filenames")
        artifact = generation / name
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(f"Dataset artifact is missing: {name}")
        if verify and (artifact.stat().st_size != metadata.get("size_bytes") or sha256(artifact) != metadata.get("sha256")):
            raise ValueError(f"Dataset artifact hash does not match: {name}")
    return generation


def publish_dataset(output_dir: Path | str, build: Callable[[Path], None], *, kind: str) -> Path:
    """Build and seal a generation, then atomically publish its pointer.

    `build` must finish all scientific validation before returning. Failures before
    pointer replacement leave any existing published generation byte-for-byte
    unchanged. Failures after replacement can only expose the complete new bundle.
    Do not use output_dir for a legacy flat dataset: those are deliberately frozen.
    """
    root = Path(output_dir)
    if root.exists() and any(p.name not in {"versions", "CURRENT.json"} and not p.name.startswith(".CURRENT-") for p in root.iterdir()):
        raise ValueError(f"Refusing to publish into a non-versioned directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    versions = root / "versions"
    versions.mkdir(exist_ok=True)
    version = uuid4().hex
    staging = versions / f".staging-{version}"
    generation = versions / version
    pointer_tmp = root / f".CURRENT-{version}.json"
    staging.mkdir()
    try:
        build(staging)
        files = sorted(staging.iterdir())
        if not files or any(not path.is_file() or path.is_symlink() for path in files):
            raise ValueError("Dataset generation must contain regular artifact files")
        if any(path.name == "manifest.json" for path in files):
            raise ValueError("manifest.json is reserved for dataset publication")
        write_json(staging / "manifest.json", {
            "schema_version": DATASET_SCHEMA_VERSION, "version": version, "kind": kind,
            "artifacts": {path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size} for path in files},
        })
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        _sync_directory(staging)
        os.replace(staging, generation)
        _sync_directory(versions)
        write_json(pointer_tmp, {"schema_version": DATASET_SCHEMA_VERSION, "version": version, "manifest_sha256": sha256(generation / "manifest.json")})
        with pointer_tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(pointer_tmp, root / "CURRENT.json")
        _sync_directory(root)
        _sync_directory(root.parent)
        return generation
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        pointer_tmp.unlink(missing_ok=True)


def validate_source(source_dir: Path | str, region: dict) -> tuple[Path, dict]:
    """Validate exact-region reuse before any generator output is written.

    Subregion reuse is intentionally unsupported: query, CRS and analysis bounds
    must match. Legacy provenance has no analysis_crs, so region.json provides it.
    """
    source = resolve_dataset_dir(source_dir)
    for name in ("region.json", "source.json", "osm_walk.graphml.gz"):
        if not (source / name).is_file():
            raise ValueError(f"Required source artifact is missing: {name}")
    saved_region = json.loads((source / "region.json").read_text(encoding="utf-8"))
    provenance = json.loads((source / "source.json").read_text(encoding="utf-8"))
    from pyproj import CRS
    if CRS.from_user_input(saved_region["crs"]) != CRS.from_user_input(region["crs"]):
        raise ValueError("Configured CRS differs from the saved snapshot region")
    if "analysis_crs" in provenance and CRS.from_user_input(provenance["analysis_crs"]) != CRS.from_user_input(region["crs"]):
        raise ValueError("Provenance CRS differs from the configured CRS")
    for key in ("center_lon", "center_lat", "width_m", "height_m", "download_buffer_m", "network_type"):
        if saved_region.get(key) != region.get(key):
            raise ValueError(f"Configured {key} differs from the saved snapshot region")
    canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
    if canonical(provenance.get("query_polygon")) != canonical(region["download_geometry"]):
        raise ValueError("Configured query polygon differs from the saved snapshot")
    if canonical(saved_region.get("download_geometry")) != canonical(region["download_geometry"]):
        raise ValueError("Saved region query polygon differs from the configuration")
    if provenance.get("analysis_bounds_m") != region["bounds"] or saved_region.get("bounds") != region["bounds"]:
        raise ValueError("Configured analysis bounds differ from the saved snapshot")
    if provenance.get("network_type") != region["network_type"]:
        raise ValueError("Configured network type differs from provenance")
    snapshots = provenance.get("snapshots", {})
    if "osm_walk.graphml.gz" not in snapshots:
        raise ValueError("Source provenance does not contain the graph snapshot hash")
    for name, metadata in snapshots.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Snapshot names must be plain filenames")
        path = source / name
        if not path.is_file():
            raise ValueError(f"Required snapshot is missing: {name}")
        if path.stat().st_size != metadata.get("size_bytes") or sha256(path) != metadata.get("sha256"):
            raise ValueError(f"Source snapshot hash does not match: {name}")
    return source, provenance


def copy_source(source: Path, target: Path, provenance: dict) -> None:
    for name in ("source.json", *provenance["snapshots"]):
        shutil.copyfile(source / name, target / name)
