"""Fetch OSM walking data into an immutable dataset, preserving raw responses.

Existing source snapshots are validated before reuse. Fetching or refreshing never
overwrites the bundled demo; only a complete source bundle becomes current.
"""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import osmnx as ox
from pyproj import CRS
from shapely.geometry import shape

try:
    from .prepare_region import ROOT, prepare_region
    from .dataset_io import copy_source, publish_dataset, resolve_dataset_dir, sha256, validate_source, write_json
except ImportError:
    from prepare_region import ROOT, prepare_region
    from dataset_io import copy_source, publish_dataset, resolve_dataset_dir, sha256, validate_source, write_json


def fetch(
    config: Path | str = ROOT / "configs/region.json", refresh: bool = False, *,
    source_dir: Path | str = ROOT / "data/demo", output_dir: Path | str | None = None,
    raw_dir: Path | str | None = None,
) -> dict:
    region = prepare_region(config)
    output = Path(output_dir) if output_dir is not None else ROOT / "data/sources" / region["region_id"]
    source_root = Path(source_dir)
    # Prefer a prior published download at the output when one exists, so repeat
    # commands don't silently revert to the older bundled source.
    candidate = output if (output / "CURRENT.json").exists() else source_root
    source_markers = ("region.json", "source.json", "osm_walk.graphml.gz", "CURRENT.json", "versions")
    has_source = any((candidate / name).exists() for name in source_markers)
    if has_source and not refresh:
        source, saved = validate_source(candidate, region)
        if candidate.resolve() != output.resolve():
            def reuse(stage: Path) -> None:
                write_json(stage / "region.json", region)
                copy_source(source, stage, saved)
                validate_source(stage, region)
            publish_dataset(output, reuse, kind="osm-source")
        print(f"Using verified source snapshot {source / 'osm_walk.graphml.gz'}")
        return saved

    raw = Path(raw_dir) if raw_dir is not None else ROOT / "data/raw"
    # Every acquisition gets a fresh cache directory: raw responses belong to this
    # acquisition only. OSMnx's global settings are restored even on failure.
    cache = raw / "overpass_cache" / uuid4().hex
    cache.mkdir(parents=True, exist_ok=True)
    settings = {key: getattr(ox.settings, key) for key in ("use_cache", "cache_folder", "log_console", "requests_timeout")}
    started = datetime.now(timezone.utc).isoformat()
    try:
        ox.settings.use_cache = True
        ox.settings.cache_folder = str(cache)
        ox.settings.log_console = True
        ox.settings.requests_timeout = 180
        graph = ox.graph_from_polygon(shape(region["download_geometry"]), network_type="walk", simplify=True, retain_all=True)
        if not graph or graph.number_of_edges() == 0:
            raise ValueError("Downloaded OSM graph is empty; no dataset was published")
        if CRS.from_user_input(graph.graph.get("crs")) != CRS.from_epsg(4326):
            raise ValueError("Downloaded OSM graph must declare geographic EPSG:4326")
        responses = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(cache.glob("*.json"))]
        completed = datetime.now(timezone.utc).isoformat()
        result = {
            "source": "OpenStreetMap contributors", "license": "ODbL-1.0",
            "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
            "attribution_url": "https://www.openstreetmap.org/copyright",
            "overpass_url": ox.settings.overpass_url,
            "download_started_utc": started, "download_completed_utc": completed,
            "osm_timestamps": sorted({r.get("osm3s", {}).get("timestamp_osm_base", "unknown") for r in responses}),
            "query_polygon": region["download_geometry"], "analysis_bounds_m": region["bounds"],
            "analysis_crs": region["crs"], "source_crs": "EPSG:4326",
            "query_note": "OSMnx may add its own topology buffer outside the configured download polygon.",
            "network_type": "walk",
            "walking_direction_policy": "OSMnx walking graphs are bidirectional, including motor-vehicle one-way streets; this is not a vehicle routing graph.",
            "osmnx_version": ox.__version__, "raw_nodes": len(graph), "raw_directed_edges": graph.number_of_edges(),
        }
        def build(stage: Path) -> None:
            write_json(stage / "region.json", region)
            snapshot = stage / "osm_walk.graphml.gz"
            ox.save_graphml(graph, filepath=snapshot)
            raw_responses = stage / "osm_overpass_responses.json.gz"
            with raw_responses.open("wb") as target:
                with gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as zipped:
                    zipped.write(json.dumps(responses, separators=(",", ":"), allow_nan=False).encode())
            result["snapshots"] = {p.name: {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p in (snapshot, raw_responses)}
            write_json(stage / "source.json", result)
            validate_source(stage, region)
        publish_dataset(output, build, kind="osm-source")
    finally:
        for key, value in settings.items():
            setattr(ox.settings, key, value)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/region.json")
    parser.add_argument("--refresh", action="store_true", help="Publish a new source generation fetched from current OSM data")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/demo")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()
    fetch(args.config, args.refresh, source_dir=args.source_dir, output_dir=args.output_dir, raw_dir=args.raw_dir)
