"""Exercise the generators using small offline OSM-shaped directed multigraphs."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import Transformer
import pytest
from shapely.geometry import LineString

from scripts import dataset_io
from scripts import fetch_osm
from scripts import prepare_network as generator
from scripts.prepare_region import prepare_region


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def case(tmp_path):
    config = tmp_path / "region-config.json"
    data = {"region_id": "offline-case", "center_lon": -122.4149, "center_lat": 37.7599,
            "crs": "EPSG:32610", "width_m": 100, "height_m": 100,
            "download_buffer_m": 10, "network_type": "walk", "walk_speed_mps": 1.4}
    config.write_text(json.dumps(data))
    region = prepare_region(config)
    x, y = region["center_m"]
    to_geo = Transformer.from_crs(region["crs"], "EPSG:4326", always_xy=True)
    graph = nx.MultiDiGraph(crs="EPSG:4326", simplified=True)
    points = {1: (x - 20, y), 2: (x + 20, y), 3: (x, y + 20), 4: (x + 100, y)}
    for node, point in points.items():
        lon, lat = to_geo.transform(*point)
        graph.add_node(node, x=lon, y=lat)
    def line(coords):
        return LineString([to_geo.transform(*point) for point in coords])
    # The first edge has reversed geometry; its reverse is already correctly oriented.
    graph.add_edge(1, 2, key=0, geometry=line([points[2], points[1]]), osmid=10)
    graph.add_edge(2, 1, key=0, geometry=line([points[2], points[1]]), osmid=10)
    # Parallel edge must be preserved, not flattened into a simple graph.
    graph.add_edge(1, 2, key=1, geometry=line([points[1], (x, y + 8), points[2]]), osmid=11)
    # Endpoints inside but polyline leaves the analysis rectangle.
    graph.add_edge(1, 2, key=2, geometry=line([points[1], (x, y + 80), points[2]]), osmid=12)
    # Endpoints are both bad, although the geometry itself is valid and in bounds.
    graph.add_edge(1, 2, key=3, geometry=line([(x - 10, y), (x + 10, y)]), osmid=13)
    # Source node lies outside; node 3 is isolated.
    graph.add_edge(4, 1, key=0, geometry=line([points[4], points[1]]), osmid=14)
    source = tmp_path / "source"
    source.mkdir()
    def save_graph(replacement=graph):
        snapshot = source / "osm_walk.graphml.gz"
        ox.save_graphml(replacement, filepath=snapshot)
        dataset_io.write_json(source / "region.json", region)
        provenance = {"query_polygon": region["download_geometry"], "analysis_bounds_m": region["bounds"],
                      "analysis_crs": region["crs"], "source_crs": "EPSG:4326", "network_type": "walk",
                      "snapshots": {snapshot.name: {"sha256": dataset_io.sha256(snapshot), "size_bytes": snapshot.stat().st_size}}}
        dataset_io.write_json(source / "source.json", provenance)
    save_graph()
    return {"config": config, "region": region, "source": source, "graph": graph,
            "save_graph": save_graph, "output": tmp_path / "output", "data": data}


def generate(case, **kwargs):
    return generator.prepare_network(case["config"], source_dir=case["source"], output_dir=case["output"], **kwargs)


def test_generator_preserves_parallel_edges_reverses_geometry_and_removes_bad_edges(case):
    source_before = tree_bytes(case["source"])
    result = generate(case)
    assert [(e["u"], e["v"], e["key"]) for e in result["edges"]] == [(1, 2, 0), (1, 2, 1), (2, 1, 0)]
    assert {n["id"] for n in result["nodes"]} == {1, 2}
    assert result["cleaning"]["endpoint_mismatch_edges_removed"] == 1
    assert result["cleaning"]["boundary_crossing_edges_removed"] == 1
    assert result["cleaning"]["nodes_outside_domain"] == 1
    assert result["cleaning"]["isolated_nodes_removed"] == 1
    nodes = {n["id"]: n for n in result["nodes"]}
    for edge in result["edges"]:
        assert edge["coordinates"][0] == pytest.approx([nodes[edge["u"]]["x"], nodes[edge["u"]]["y"]], abs=1e-6)
        assert edge["coordinates"][-1] == pytest.approx([nodes[edge["v"]]["x"], nodes[edge["v"]]["y"]], abs=1e-6)
    current = dataset_io.resolve_dataset_dir(case["output"])
    assert json.loads((current / "network.json").read_text()) == result
    edges = gpd.read_parquet(current / "edges.parquet")
    assert list(edges.key) == [0, 1, 0]
    np.testing.assert_allclose(edges.length_m, edges.geometry.length)
    assert tree_bytes(case["source"]) == source_before
    # A resolved generation remains self-contained and can serve as a new input.
    generator.prepare_network(case["config"], source_dir=case["output"], output_dir=case["output"].parent / "second")


@pytest.mark.parametrize("corruption", ["query", "crs", "hash", "missing", "bounds", "graph_crs"])
def test_bad_source_rejected_before_any_output_write(case, corruption):
    source = case["source"]
    provenance = json.loads((source / "source.json").read_text())
    if corruption == "query":
        provenance["query_polygon"]["coordinates"][0][0][0] += .1
    elif corruption == "crs":
        provenance["analysis_crs"] = "EPSG:32611"
    elif corruption == "hash":
        provenance["snapshots"]["osm_walk.graphml.gz"]["sha256"] = "0" * 64
    elif corruption == "bounds":
        provenance["analysis_bounds_m"][0] += 1
    elif corruption == "graph_crs":
        provenance["source_crs"] = "EPSG:32610"
    else:
        (source / "osm_walk.graphml.gz").unlink()
    dataset_io.write_json(source / "source.json", provenance)
    before = tree_bytes(source)
    with pytest.raises(ValueError):
        generate(case)
    assert not case["output"].exists()
    assert tree_bytes(source) == before


@pytest.mark.parametrize("empty_source", [True, False])
def test_empty_source_or_cleaned_graph_does_not_write(case, empty_source):
    graph = case["graph"].copy()
    graph.remove_edges_from(list(graph.edges(keys=True)))
    if not empty_source:
        # Only retained raw edge crosses the boundary and is removed by cleaning.
        original = case["graph"].edges[1, 2, 2]
        graph.add_edge(1, 2, key=2, **original)
    case["save_graph"](graph)
    with pytest.raises(ValueError, match="nonempty|No usable edges"):
        generate(case)
    assert not case["output"].exists()


@pytest.mark.parametrize("key,value", [("width_m", float("nan")), ("height_m", float("inf")),
    ("download_buffer_m", float("nan")), ("download_buffer_m", -1), ("walk_speed_mps", 0),
    ("walk_speed_mps", float("inf")), ("walk_speed_mps", True), ("center_lon", float("nan")),
    ("center_lat", 91), ("crs", "EPSG:4326"), ("crs", "EPSG:2227"), ("region_id", "../escape")])
def test_invalid_config_rejected_without_outputs(case, key, value):
    case["data"][key] = value
    case["config"].write_text(json.dumps(case["data"]))
    with pytest.raises(ValueError):
        generate(case)
    assert not case["output"].exists()


def test_generator_failure_mid_parquet_keeps_previous_generation_unchanged(case, monkeypatch):
    generate(case)
    current_before = (case["output"] / "CURRENT.json").read_bytes()
    resolved_before = dataset_io.resolve_dataset_dir(case["output"])
    artifacts_before = tree_bytes(resolved_before)
    original = gpd.GeoDataFrame.to_parquet
    def fail_on_edges(self, path, *args, **kwargs):
        if Path(path).name == "edges.parquet":
            raise OSError("injected parquet failure")
        return original(self, path, *args, **kwargs)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fail_on_edges)
    with pytest.raises(OSError, match="injected"):
        generate(case)
    assert (case["output"] / "CURRENT.json").read_bytes() == current_before
    assert dataset_io.resolve_dataset_dir(case["output"]) == resolved_before
    assert tree_bytes(resolved_before) == artifacts_before
    assert not list((case["output"] / "versions").glob(".staging-*"))


def test_atomic_pointer_failure_keeps_old_bundle_visible(case, monkeypatch):
    generate(case)
    pointer_before = (case["output"] / "CURRENT.json").read_bytes()
    old = dataset_io.resolve_dataset_dir(case["output"])
    original = dataset_io.os.replace
    def fail_pointer(source, destination):
        if Path(destination).name == "CURRENT.json":
            assert (Path(source).parent / "CURRENT.json").read_bytes() == pointer_before
            raise OSError("injected pointer failure")
        return original(source, destination)
    monkeypatch.setattr(dataset_io.os, "replace", fail_pointer)
    with pytest.raises(OSError, match="injected"):
        generate(case)
    assert dataset_io.resolve_dataset_dir(case["output"]) == old
    assert (case["output"] / "CURRENT.json").read_bytes() == pointer_before


def test_fetch_mismatch_or_download_failure_never_clobbers_source(case, monkeypatch):
    before = tree_bytes(case["source"])
    case["data"]["width_m"] += 1
    case["config"].write_text(json.dumps(case["data"]))
    def no_download(*args, **kwargs):
        raise RuntimeError("injected download failure")
    monkeypatch.setattr(fetch_osm.ox, "graph_from_polygon", no_download)
    with pytest.raises(ValueError, match="differs"):
        fetch_osm.fetch(case["config"], source_dir=case["source"], output_dir=case["output"])
    assert tree_bytes(case["source"]) == before
    assert not case["output"].exists()
    settings_before = ox.settings.cache_folder
    with pytest.raises(RuntimeError, match="injected download"):
        fetch_osm.fetch(case["config"], refresh=True, source_dir=case["source"], output_dir=case["output"], raw_dir=case["output"].parent / "raw")
    assert tree_bytes(case["source"]) == before
    assert not case["output"].exists()
    assert ox.settings.cache_folder == settings_before


def test_fetch_verified_reuse_and_refresh_publish_complete_generations(case, monkeypatch):
    def download(*args, **kwargs):
        Path(ox.settings.cache_folder, "response.json").write_text(json.dumps({"elements": [], "osm3s": {"timestamp_osm_base": "offline-fixture"}}))
        return case["graph"]
    monkeypatch.setattr(fetch_osm.ox, "graph_from_polygon", download)
    before = tree_bytes(case["source"])
    fetch_osm.fetch(case["config"], source_dir=case["source"], output_dir=case["output"])
    first = dataset_io.resolve_dataset_dir(case["output"])
    new_source = fetch_osm.fetch(case["config"], refresh=True, source_dir=case["source"], output_dir=case["output"], raw_dir=case["output"].parent / "raw")
    second = dataset_io.resolve_dataset_dir(case["output"])
    assert first != second and first.is_dir()
    assert new_source["osm_timestamps"] == ["offline-fixture"]
    assert (second / "osm_overpass_responses.json.gz").is_file()
    assert tree_bytes(case["source"]) == before


def test_generators_refuse_legacy_directory_as_publication_target(case):
    before = tree_bytes(case["source"])
    with pytest.raises(ValueError, match="non-versioned"):
        generator.prepare_network(case["config"], source_dir=case["source"], output_dir=case["source"])
    assert tree_bytes(case["source"]) == before


def test_manifest_verification_detects_artifact_damage(case):
    generate(case)
    directory = dataset_io.resolve_dataset_dir(case["output"])
    (directory / "network.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        dataset_io.resolve_dataset_dir(case["output"])


def test_endpoint_check_runs_after_reversal_and_does_not_move_geometry(case):
    graph = case["graph"].copy()
    x, y = case["region"]["center_m"]
    to_geo = Transformer.from_crs(case["region"]["crs"], "EPSG:4326", always_xy=True)
    # Reversal will put its start within tolerance but its end still misses v by 2m.
    graph.add_edge(1, 2, key=9, geometry=LineString([to_geo.transform(x + 18, y), to_geo.transform(x - 20, y)]), osmid=99)
    # A small allowed mismatch must survive without silently snapping coordinates.
    graph.add_edge(1, 2, key=10, geometry=LineString([to_geo.transform(x - 19.95, y), to_geo.transform(x + 20, y)]), osmid=100)
    case["save_graph"](graph)
    result = generate(case)
    assert result["cleaning"]["endpoint_mismatch_edges_removed"] == 2
    assert not any(edge["key"] == 9 for edge in result["edges"])
    edge = next(edge for edge in result["edges"] if edge["key"] == 10)
    node = next(node for node in result["nodes"] if node["id"] == 1)
    assert edge["coordinates"][0][0] - node["x"] == pytest.approx(.05, abs=1e-6)


def test_nonfinite_and_zero_length_geometries_are_removed(case, monkeypatch):
    # Invalid coordinates may be rejected inside projection; inject already
    # projected bad geometries here to exercise the generator's own validation.
    projected = ox.project_graph(case["graph"], to_crs=case["region"]["crs"])
    x, y = case["region"]["center_m"]
    with np.errstate(invalid="ignore"):
        projected.add_edge(1, 2, key=20, geometry=LineString([(x - 20, y), (float("nan"), y), (x + 20, y)]))
    projected.add_edge(1, 2, key=21, geometry=LineString([(x, y), (x, y)]))
    monkeypatch.setattr(generator.ox, "project_graph", lambda *args, **kwargs: projected)
    result = generate(case)
    assert result["cleaning"]["invalid_edges_removed"] == 2
    assert all(edge["key"] not in {20, 21} for edge in result["edges"])


def test_failed_refresh_snapshot_serialization_keeps_published_source(case, monkeypatch):
    fetch_osm.fetch(case["config"], source_dir=case["source"], output_dir=case["output"])
    pointer = (case["output"] / "CURRENT.json").read_bytes()
    published = dataset_io.resolve_dataset_dir(case["output"])
    before = tree_bytes(published)
    monkeypatch.setattr(fetch_osm.ox, "graph_from_polygon", lambda *args, **kwargs: case["graph"])
    def broken_save(graph, filepath):
        Path(filepath).write_bytes(b"partial graph")
        raise OSError("injected graph serialization failure")
    monkeypatch.setattr(fetch_osm.ox, "save_graphml", broken_save)
    with pytest.raises(OSError, match="injected graph"):
        fetch_osm.fetch(case["config"], refresh=True, source_dir=case["source"], output_dir=case["output"], raw_dir=case["output"].parent / "raw")
    assert (case["output"] / "CURRENT.json").read_bytes() == pointer
    assert dataset_io.resolve_dataset_dir(case["output"]) == published
    assert tree_bytes(published) == before


def test_unpublished_first_generation_is_not_treated_as_legacy(tmp_path):
    root = tmp_path / "interrupted-dataset"
    def interrupted(stage):
        dataset_io.write_json(stage / "network.json", {"generation": "unfinished"})
        raise RuntimeError("interrupted before Parquet")
    with pytest.raises(RuntimeError):
        dataset_io.publish_dataset(root, interrupted, kind="test")
    with pytest.raises(ValueError, match="no published generation"):
        dataset_io.resolve_dataset_dir(root)
