# Mission walking dataset

Real OpenStreetMap data for a 4 km × 4 km rectangle around Mission, San Francisco.
Coordinates for computation are WGS 84 / UTM zone 10N (EPSG:32610), in meters.

- `osm_overpass_responses.json.gz`: original Overpass API response, retained for provenance and redistribution under ODbL 1.0.
- `osm_walk.graphml.gz`: downloaded, simplified OSMnx walking graph before analysis-domain cleaning.
- `source.json`: source dates, query polygon, source hashes and license information.
- `network.json`: full directed multigraph consumed by the numerical backend. Each edge keeps its complete oriented polyline and `(u, v, key)` identifier.
- `nodes.parquet`, `edges.parquet`: projected GeoParquet tables.
- `network.duckdb`: the same two tables for local SQL analysis. Install DuckDB Spatial during online setup, then use `LOAD spatial` offline.
- `roads.geojson`: unique physical polylines in longitude/latitude for display.
- `roads.pmtiles`: 129 locally generated vector tiles, zoom 11–16 (MapLibre can overzoom above 16), source layer `roads`.
- `style.json`: offline MapLibre style; street labels use local system fonts, requiring MapLibre GL JS ≥5.11. No sprites or remote glyphs are needed. When loading the style programmatically, replace its PMTiles URL with `pmtiles://` plus the absolute same-origin URL if your PMTiles adapter requires it.
- `region.json`, `boundary.geojson`: exact metric domain and geographic boundary.

From the repository root, rebuild without downloading current OSM data:

```sh
python scripts/prepare_region.py
python scripts/fetch_osm.py
python scripts/prepare_network.py
python scripts/prepare_basemap.py
python scripts/report_data_quality.py
```

`fetch_osm.py` uses the saved snapshot by default. Use `--refresh` only to explicitly download a new source version. On a new machine, execute `python scripts/report_data_quality.py --install-spatial` once while online.

All connected components are retained. Clicking on different components can legitimately produce an unreachable route. Walking graphs are bidirectional under OSMnx's walking policy; car one-way restrictions are not applied. Roads whose complete geometry leaves the analysis rectangle are excluded. The basemap shows streets only, and is not a full urban reference map.

© [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
The geographic database and derivative data in this directory are offered under the [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/). The original source snapshot is included. This data license applies to the geographic data, not to the application source code. No OpenStreetMap standard raster tiles were downloaded or cached.
