"""Cross-check the committed real-data artifacts without network access."""
import gzip
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import mapbox_vector_tile
import numpy as np
import pytest
from pmtiles.reader import Reader, MmapSource
from pmtiles.tile import Compression
from shapely.geometry import LineString, box

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data/demo"


@pytest.fixture(scope="module")
def network():
    return json.loads((DEMO / "network.json").read_text())


def test_geometries_preserve_endpoints_directions_and_metric_lengths(network):
    nodes = {n["id"]: n for n in network["nodes"]}
    domain = box(*network["bounds"])
    edge_ids = set()
    for edge in network["edges"]:
        key = (edge["u"], edge["v"], edge["key"])
        assert key not in edge_ids
        edge_ids.add(key)
        line = LineString(edge["coordinates"])
        assert line.is_valid and domain.covers(line)
        assert line.length == pytest.approx(edge["length_m"], abs=1e-8)
        assert line.length > 0
        for coordinate, node in [(edge["coordinates"][0], nodes[edge["u"]]), (edge["coordinates"][-1], nodes[edge["v"]])]:
            assert coordinate == pytest.approx([node["x"], node["y"]], abs=1e-6)
    assert len(nodes) > 1000  # Protect against replacing real data with a tiny toy grid.
    assert any(edge["key"] > 0 for edge in network["edges"])
    assert any(len(edge["coordinates"]) > 3 for edge in network["edges"])


def test_geoparquet_roundtrip_matches_routing_data(network):
    nodes = gpd.read_parquet(DEMO / "nodes.parquet")
    edges = gpd.read_parquet(DEMO / "edges.parquet")
    assert nodes.crs.to_epsg() == 32610
    assert edges.crs.to_epsg() == 32610
    assert len(nodes) == len(network["nodes"])
    assert len(edges) == len(network["edges"])
    np.testing.assert_allclose(edges.length_m, edges.geometry.length, atol=1e-8)


def test_original_osm_snapshot_and_generated_files_are_hash_verified():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for filename, metadata in manifest["artifacts"].items():
        contents = (ROOT / filename).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == metadata["sha256"], filename
        assert len(contents) == metadata["size_bytes"], filename
    with gzip.open(DEMO / "osm_overpass_responses.json.gz", "rt") as source:
        responses = json.load(source)
    assert any("elements" in response for response in responses)
    assert all(response["osm3s"]["timestamp_osm_base"] for response in responses)


def test_offline_pmtiles_contains_real_vector_roads():
    with (DEMO / "roads.pmtiles").open("rb") as source:
        reader = Reader(MmapSource(source))
        header = reader.header()
        metadata = reader.metadata()
        assert header["addressed_tiles_count"] > 0
        assert header["tile_compression"] == Compression.GZIP
        assert metadata["vector_layers"][0]["id"] == "roads"
        # Mission center at zoom 13; independently evaluate slippy map indices.
        import math
        lon, lat, z = -122.4149, 37.7599, 13
        x = int((lon + 180) / 360 * 2 ** z)
        y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2 ** z)
        tile = reader.get(z, x, y)
        decoded = mapbox_vector_tile.decode(gzip.decompress(tile))
        assert len(decoded["roads"]["features"]) > 50
    style = json.loads((DEMO / "style.json").read_text())
    assert "glyphs" not in style and "sprite" not in style
    assert "https://" not in json.dumps(style["sources"])


def test_quality_report_has_no_invalid_or_outside_geometries():
    report = json.loads((ROOT / "reports/data_quality.json").read_text())
    assert report["status"] == "passed"
    for metric in ["invalid_geometries", "orphan_edge_endpoints", "nonpositive_lengths", "edges_outside_domain", "duplicate_edge_ids"]:
        assert report["quality"][metric] == 0
