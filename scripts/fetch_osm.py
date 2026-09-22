"""Fetch real OpenStreetMap walking data; preserve original responses and graph.

The initial run requires internet. Processing can reuse the bundled snapshot.
No raster tile server is downloaded.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import osmnx as ox
from shapely.geometry import shape

from prepare_region import ROOT, prepare_region


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(
        config: Path = ROOT / "configs/region.json", 
        refresh: bool = False
        ) -> dict:
    raw, demo = ROOT / "data/raw", ROOT / "data/demo"
    raw.mkdir(parents=True, exist_ok=True)
    demo.mkdir(parents=True, exist_ok=True)
    region = prepare_region(config)
    (demo / "region.json").write_text(
        json.dumps(region, indent=2) + "\n"
    )
    snapshot, provenance = demo / "osm_walk.graphml.gz", demo / "source.json"
    if snapshot.exists() and provenance.exists() and not refresh:
        saved = json.loads(provenance.read_text())
        if json.dumps(saved["query_polygon"], sort_keys=True) != json.dumps(region["download_geometry"], sort_keys=True):
            raise ValueError("The configured region differs from the saved snapshot; use --refresh to fetch that region")
        print(f"Using saved source snapshot {snapshot}")
        return saved
    ox.settings.use_cache = True
    # A fresh cache folder prevents --refresh from reusing yesterday's response.
    cache = raw / "overpass_cache"
    if refresh:
        cache = cache / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ox.settings.cache_folder = str(cache)
    ox.settings.log_console = True
    ox.settings.requests_timeout = 180
    started = datetime.now(timezone.utc).isoformat()
    graph = ox.graph_from_polygon(shape(region["download_geometry"]), network_type="walk", simplify=True, retain_all=True)
    ox.save_graphml(graph, filepath=snapshot)
    cached = sorted(cache.glob("*.json"))
    raw_responses = demo / "osm_overpass_responses.json.gz"
    responses = [json.loads(path.read_text()) for path in cached]
    with raw_responses.open("wb") as target:
        with gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as zipped:
            zipped.write(json.dumps(responses, separators=(",", ":")).encode())
    result = {
        "source": "OpenStreetMap contributors", "license": "ODbL-1.0",
        "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
        "attribution_url": "https://www.openstreetmap.org/copyright",
        "overpass_url": ox.settings.overpass_url,
        "download_started_utc": started,
        "download_completed_utc": datetime.now(timezone.utc).isoformat(),
        "osm_timestamps": sorted({r.get("osm3s", {}).get("timestamp_osm_base", "unknown") for r in responses}),
        "query_polygon": region["download_geometry"], "analysis_bounds_m": region["bounds"],
        "query_note": "OSMnx adds its own 500 m topology buffer to the configured 500 m download buffer.",
        "network_type": "walk",
        "walking_direction_policy": "OSMnx walking graphs are bidirectional, including motor-vehicle one-way streets; this is not a vehicle routing graph.",
        "osmnx_version": ox.__version__, "raw_nodes": len(graph), "raw_directed_edges": graph.number_of_edges(),
        "snapshots": {p.name: {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p in (snapshot, raw_responses)},
    }
    provenance.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/region.json")
    parser.add_argument("--refresh", action="store_true", help="Explicitly replace the saved snapshot with current OSM data")
    args = parser.parse_args()
    fetch(args.config, args.refresh)
