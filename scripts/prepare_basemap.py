"""Create a compact local vector PMTiles basemap directly from licensed OSM data.

No web tiles are cached. MapLibre >=5.11 renders labels from local system fonts,
so this style makes no glyph, font, sprite, or third-party network requests.
"""
from __future__ import annotations

import gzip
import json
import math

import geopandas as gpd
import mapbox_vector_tile
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer
from shapely.geometry import box

from prepare_region import ROOT

EARTH_HALF = 20037508.342789244


def tile_xy(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def tile_bounds(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    width = 2 * EARTH_HALF / 2 ** z
    return -EARTH_HALF + x * width, EARTH_HALF - (y + 1) * width, -EARTH_HALF + (x + 1) * width, EARTH_HALF - y * width


def prepare_basemap() -> dict:
    demo = ROOT / "data/demo"
    region = json.loads((demo / "region.json").read_text())
    roads = gpd.read_file(demo / "roads.geojson").to_crs("EPSG:3857")
    west, south, east, north = region["geographic_bounds"]
    min_zoom, max_zoom = 11, 16
    tiles = []
    index = roads.sindex
    # Geometry is encoded in Web Mercator, then quantized within each tile.
    for z in range(min_zoom, max_zoom + 1):
        left, top = tile_xy(west, north, z)
        right, bottom = tile_xy(east, south, z)
        for x in range(left, right + 1):
            for y in range(top, bottom + 1):
                bounds = tile_bounds(x, y, z)
                envelope = box(*bounds)
                features = []
                for row_id in index.query(envelope, predicate="intersects"):
                    row = roads.iloc[int(row_id)]
                    line = row.geometry.intersection(envelope)
                    if line.is_empty or line.geom_type not in ("LineString", "MultiLineString"):
                        continue
                    features.append({"geometry": line, "properties": {"name": str(row["name"] or ""), "highway": str(row["highway"] or "")}})
                if features:
                    data = mapbox_vector_tile.encode({"name": "roads", "features": features},
                        default_options={"quantize_bounds": bounds, "extents": 4096})
                    tiles.append((zxy_to_tileid(z, x, y), gzip.compress(data, mtime=0)))
    archive = demo / "roads.pmtiles"
    with archive.open("wb") as output:
        writer = Writer(output)
        for tile_id, data in sorted(tiles):
            writer.write_tile(tile_id, data)
        writer.finalize({"tile_type": TileType.MVT, "tile_compression": Compression.GZIP,
                         "min_zoom": min_zoom, "max_zoom": max_zoom,
                         "min_lon_e7": round(west * 1e7), "min_lat_e7": round(south * 1e7),
                         "max_lon_e7": round(east * 1e7), "max_lat_e7": round(north * 1e7),
                         "center_zoom": 13, "center_lon_e7": round(region["center_lon"] * 1e7),
                         "center_lat_e7": round(region["center_lat"] * 1e7)},
                        {"name": "Flow Mission walking streets", "description": "Locally generated OSM street vector tiles; idealized diffusion experiment.",
                         "attribution": "© OpenStreetMap contributors", "type": "baselayer", "version": "1",
                         "vector_layers": [{"id": "roads", "minzoom": min_zoom, "maxzoom": max_zoom,
                                            "fields": {"name": "String", "highway": "String"}}]})
    style = {
        "version": 8, "name": "Flow offline streets",
        "metadata": {"license": "ODbL-1.0", "requires": "MapLibre GL JS >=5.11 for local font rendering", "offline": True},
        "sources": {"streets": {"type": "vector", "url": "pmtiles:///data/demo/roads.pmtiles", "attribution": "© OpenStreetMap contributors"}},
        "layers": [
            {"id": "background", "type": "background", "paint": {"background-color": "#e8eee9"}},
            {"id": "road-casing", "source": "streets", "source-layer": "roads", "type": "line",
             "paint": {"line-color": "#bdc9c1", "line-width": ["interpolate", ["linear"], ["zoom"], 11, 0.6, 16, 5.0]}},
            {"id": "roads", "source": "streets", "source-layer": "roads", "type": "line",
             "paint": {"line-color": "#fffef7", "line-width": ["interpolate", ["linear"], ["zoom"], 11, 0.25, 16, 3.2]}},
            {"id": "road-labels", "source": "streets", "source-layer": "roads", "type": "symbol", "minzoom": 14,
             "filter": ["!=", ["get", "name"], ""],
             "layout": {"symbol-placement": "line", "text-field": ["get", "name"], "text-font": ["Arial", "sans-serif"],
                        "text-size": 11, "symbol-spacing": 350, "text-max-angle": 30},
             "paint": {"text-color": "#53685e", "text-halo-color": "#fffef7", "text-halo-width": 1.5}},
        ]}
    (demo / "style.json").write_text(json.dumps(style, indent=2) + "\n")
    (demo / "boundary.geojson").write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": region["geometry"]}) + "\n")
    result = {"archive": str(archive.relative_to(ROOT)), "tiles": len(tiles), "size_bytes": archive.stat().st_size, "zoom_range": [min_zoom, max_zoom]}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    prepare_basemap()
