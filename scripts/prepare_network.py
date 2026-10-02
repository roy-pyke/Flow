"""Validate, project and publish a preserved OSM graph as an immutable dataset."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, box

try:  # Support both python -m scripts.prepare_network and direct script execution.
    from .prepare_region import ROOT, prepare_region
    from .dataset_io import copy_source, publish_dataset, sha256, validate_source, write_json
except ImportError:
    from prepare_region import ROOT, prepare_region
    from dataset_io import copy_source, publish_dataset, sha256, validate_source, write_json


def text_tag(value) -> str:
    return "; ".join(str(v) for v in value) if isinstance(value, list) else str(value or "")


def prepare_network(
    config: Path | str = ROOT / "configs/region.json", *,
    source_dir: Path | str = ROOT / "data/demo",
    output_dir: Path | str | None = None,
    endpoint_tolerance_m: float = 0.1,
) -> dict:
    if isinstance(endpoint_tolerance_m, bool) or not math.isfinite(endpoint_tolerance_m) or endpoint_tolerance_m < 0:
        raise ValueError("endpoint_tolerance_m must be finite and nonnegative")
    region = prepare_region(config)
    source_dir, provenance = validate_source(source_dir, region)
    source = source_dir / "osm_walk.graphml.gz"
    source_hash = sha256(source)
    raw_graph = ox.load_graphml(source)
    if CRS.from_user_input(raw_graph.graph.get("crs")) != CRS.from_user_input(provenance.get("source_crs", "EPSG:4326")):
        raise ValueError("Snapshot graph CRS differs from source provenance")
    if not isinstance(raw_graph, nx.MultiDiGraph) or raw_graph.number_of_edges() == 0:
        raise ValueError("Source snapshot must contain a nonempty directed multigraph")
    graph = ox.project_graph(raw_graph, to_crs=region["crs"])
    domain = box(*region["bounds"])
    original_nodes, original_edges = len(graph), graph.number_of_edges()
    invalid_nodes = [node for node, data in graph.nodes(data=True) if not all(math.isfinite(float(data.get(axis, math.nan))) for axis in ("x", "y"))]
    graph.remove_nodes_from(invalid_nodes)
    outside = [node for node, data in graph.nodes(data=True) if not domain.covers(Point(data["x"], data["y"]))]
    graph.remove_nodes_from(outside)
    removed_invalid = removed_crossing = removed_endpoints = 0
    for u, v, key, data in list(graph.edges(keys=True, data=True)):
        line = data.get("geometry")
        if line is None:
            line = LineString([(graph.nodes[u]["x"], graph.nodes[u]["y"]), (graph.nodes[v]["x"], graph.nodes[v]["y"])])
        if (line.geom_type != "LineString" or line.is_empty or not line.is_valid
                or not math.isfinite(line.length) or line.length <= 0
                or any(len(xy) != 2 or not all(math.isfinite(v) for v in xy) for xy in line.coords)):
            graph.remove_edge(u, v, key)
            removed_invalid += 1
            continue
        if not domain.covers(line):
            graph.remove_edge(u, v, key)
            removed_crossing += 1
            continue
        coords = list(line.coords)
        start = Point(graph.nodes[u]["x"], graph.nodes[u]["y"])
        end = Point(graph.nodes[v]["x"], graph.nodes[v]["y"])
        if start.distance(Point(coords[-1])) < start.distance(Point(coords[0])):
            line = LineString(coords[::-1])
        if start.distance(Point(line.coords[0])) > endpoint_tolerance_m or end.distance(Point(line.coords[-1])) > endpoint_tolerance_m:
            graph.remove_edge(u, v, key)
            removed_endpoints += 1
            continue
        data["geometry"] = line
        data["length_m"] = line.length
    isolates = list(nx.isolates(graph))
    graph.remove_nodes_from(isolates)
    if graph.number_of_edges() == 0 or not graph:
        raise ValueError("No usable edges remain in the analysis region; no dataset was published")
    components = sorted(nx.weakly_connected_components(graph), key=lambda c: (-len(c), min(c)))
    membership = {node: i for i, component in enumerate(components) for node in component}
    to_geo = Transformer.from_crs(region["crs"], "EPSG:4326", always_xy=True)
    nodes = []
    for node, data in sorted(graph.nodes(data=True)):
        lon, lat = to_geo.transform(data["x"], data["y"], errcheck=True)
        nodes.append({"id": int(node), "x": float(data["x"]), "y": float(data["y"]), "lon": lon, "lat": lat, "component": membership[node]})
    edges, features, table_edges = [], [], []
    seen_geometries = set()
    for u, v, key, data in sorted(graph.edges(keys=True, data=True)):
        line = data["geometry"]
        edge = {"u": int(u), "v": int(v), "key": int(key), "length_m": line.length,
                "coordinates": [list(c) for c in line.coords], "name": text_tag(data.get("name")),
                "highway": text_tag(data.get("highway")), "osmid": text_tag(data.get("osmid"))}
        edges.append(edge)
        table_edges.append({k: value for k, value in edge.items() if k != "coordinates"} | {"component": membership[u], "geometry": line})
        signature = line.normalize().wkb
        if signature not in seen_geometries:
            seen_geometries.add(signature)
            geographic = [to_geo.transform(*xy, errcheck=True) for xy in line.coords]
            features.append({"type": "Feature", "properties": {"name": edge["name"], "highway": edge["highway"], "length_m": line.length},
                             "geometry": {"type": "LineString", "coordinates": geographic}})
    result = {"region_id": region["region_id"], "crs": region["crs"], "bounds": region["bounds"],
              "geographic_bounds": region["geographic_bounds"], "center_lon": region["center_lon"], "center_lat": region["center_lat"],
              "nodes": nodes, "edges": edges, "walk_speed_mps": region["walk_speed_mps"],
              "source_snapshot_sha256": source_hash,
              "source_provenance_sha256": sha256(source_dir / "source.json"),
              "dataset_schema_version": 1,
              "attribution": "© OpenStreetMap contributors (ODbL-1.0)",
              "cleaning": {"raw_nodes": original_nodes, "raw_directed_edges": original_edges,
                           "nodes_outside_domain": len(outside), "invalid_nodes_removed": len(invalid_nodes),
                           "isolated_nodes_removed": len(isolates), "invalid_edges_removed": removed_invalid,
                           "boundary_crossing_edges_removed": removed_crossing,
                           "endpoint_mismatch_edges_removed": removed_endpoints, "endpoint_tolerance_m": endpoint_tolerance_m,
                           "policy": "Retain all components; reject nonfinite, invalid, endpoint-mismatched or boundary-crossing geometry without moving endpoints."}}
    # Serialization also rejects any nonfinite value before output staging starts.
    json.dumps(result, allow_nan=False)
    nodes_table = gpd.GeoDataFrame([{**n, "geometry": Point(n["x"], n["y"])} for n in nodes], crs=region["crs"])
    edges_table = gpd.GeoDataFrame(table_edges, crs=region["crs"])

    def build(stage: Path) -> None:
        write_json(stage / "region.json", region)
        copy_source(source_dir, stage, provenance)
        write_json(stage / "network.json", result)
        write_json(stage / "roads.geojson", {"type": "FeatureCollection", "features": features})
        nodes_table.to_parquet(stage / "nodes.parquet", index=False)
        edges_table.to_parquet(stage / "edges.parquet", index=False)
        # Read back both tables before publishing; serializer errors cannot expose a
        # new JSON paired with old or truncated Parquet files.
        for name, expected in (("nodes", nodes_table), ("edges", edges_table)):
            actual = gpd.read_parquet(stage / f"{name}.parquet")
            if len(actual) != len(expected) or actual.crs != expected.crs or not actual.geometry.equals(expected.geometry):
                raise ValueError(f"{name} Parquet roundtrip differs from generated data")
        validate_source(stage, region)

    output = Path(output_dir) if output_dir is not None else ROOT / "data/datasets" / region["region_id"]
    published = publish_dataset(output, build, kind="walking-network")
    print(json.dumps({"nodes": len(nodes), "directed_edges": len(edges), "display_polylines": len(features),
                      "components": len(components), "published_dir": str(published)}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/region.json")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/demo")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--endpoint-tolerance-m", type=float, default=0.1)
    args = parser.parse_args()
    prepare_network(args.config, source_dir=args.source_dir, output_dir=args.output_dir, endpoint_tolerance_m=args.endpoint_tolerance_m)
