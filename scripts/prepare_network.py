"""Project and clean the preserved OSM graph, retaining complete polylines."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import LineString, Point, box, mapping

from prepare_region import ROOT, prepare_region


def text_tag(value) -> str:
    return "; ".join(str(v) for v in value) if isinstance(value, list) else str(value or "")


def prepare_network(config: Path = ROOT / "configs/region.json") -> dict:
    region = prepare_region(config)
    demo = ROOT / "data/demo"
    source = demo / "osm_walk.graphml.gz"
    graph = ox.project_graph(ox.load_graphml(source), to_crs=region["crs"])
    domain = box(*region["bounds"])
    original_nodes, original_edges = len(graph), graph.number_of_edges()
    outside = [node for node, data in graph.nodes(data=True) if not domain.covers(Point(data["x"], data["y"]))]
    graph.remove_nodes_from(outside)
    removed_invalid = removed_crossing = 0
    # Reject paths that leave the simulation rectangle, even if both endpoints
    # are inside. Keeping their coordinates would silently sample outside it.
    for u, v, key, data in list(graph.edges(keys=True, data=True)):
        line = data.get("geometry")
        if line is None:
            line = LineString([(graph.nodes[u]["x"], graph.nodes[u]["y"]), (graph.nodes[v]["x"], graph.nodes[v]["y"])])
        if line.is_empty or not line.is_valid or line.length <= 0 or line.geom_type != "LineString":
            graph.remove_edge(u, v, key)
            removed_invalid += 1
            continue
        if not domain.covers(line):
            graph.remove_edge(u, v, key)
            removed_crossing += 1
            continue
        coords = list(line.coords)
        start = Point(graph.nodes[u]["x"], graph.nodes[u]["y"])
        if start.distance(Point(coords[-1])) < start.distance(Point(coords[0])):
            line = LineString(coords[::-1])
        data["geometry"] = line
        data["length_m"] = line.length
    graph.remove_nodes_from(list(nx.isolates(graph)))
    components = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
    membership = {node: i for i, component in enumerate(components) for node in component}
    to_geo = Transformer.from_crs(region["crs"], "EPSG:4326", always_xy=True)
    nodes = []
    for node, data in sorted(graph.nodes(data=True)):
        lon, lat = to_geo.transform(data["x"], data["y"])
        nodes.append({"id": int(node), "x": data["x"], "y": data["y"], "lon": lon, "lat": lat, "component": membership[node]})
    edges, features, table_edges = [], [], []
    seen_geometries = set()
    for u, v, key, data in sorted(graph.edges(keys=True, data=True)):
        line = data["geometry"]
        edge = {"u": int(u), "v": int(v), "key": int(key), "length_m": line.length,
                "coordinates": [list(c) for c in line.coords], "name": text_tag(data.get("name")),
                "highway": text_tag(data.get("highway")), "osmid": text_tag(data.get("osmid"))}
        edges.append(edge)
        table_edges.append({k: v for k, v in edge.items() if k != "coordinates"} | {"component": membership[u], "geometry": line})
        # Display each physical polyline once, while the graph keeps directions.
        signature = line.normalize().wkb
        if signature not in seen_geometries:
            seen_geometries.add(signature)
            geographic = [to_geo.transform(*xy) for xy in line.coords]
            features.append({"type": "Feature", "properties": {"name": edge["name"], "highway": edge["highway"], "length_m": line.length},
                             "geometry": {"type": "LineString", "coordinates": geographic}})
    result = {"region_id": region["region_id"], "crs": region["crs"], "bounds": region["bounds"],
              "geographic_bounds": region["geographic_bounds"], "center_lon": region["center_lon"], "center_lat": region["center_lat"],
              "nodes": nodes, "edges": edges, "walk_speed_mps": region["walk_speed_mps"],
              "source_snapshot_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "attribution": "© OpenStreetMap contributors (ODbL-1.0)",
              "cleaning": {"raw_nodes": original_nodes, "raw_directed_edges": original_edges,
                           "nodes_outside_domain": len(outside), "invalid_edges_removed": removed_invalid,
                           "boundary_crossing_edges_removed": removed_crossing,
                           "policy": "Retain all components; discard edges not wholly contained in the metric analysis rectangle."}}
    serialized = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
    (demo / "network.json").write_text(serialized + "\n")
    roads = {"type": "FeatureCollection", "features": features}
    (demo / "roads.geojson").write_text(json.dumps(roads, separators=(",", ":"), ensure_ascii=False) + "\n")
    gpd.GeoDataFrame([{**n, "geometry": Point(n["x"], n["y"])} for n in nodes], crs=region["crs"]).to_parquet(demo / "nodes.parquet", index=False)
    gpd.GeoDataFrame(table_edges, crs=region["crs"]).to_parquet(demo / "edges.parquet", index=False)
    print(json.dumps({"nodes": len(nodes), "directed_edges": len(edges), "display_polylines": len(features), "components": len(components)}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/region.json")
    prepare_network(parser.parse_args().config)
