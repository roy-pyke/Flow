# Flow

**An offline diffusion and walking-route laboratory for San Francisco's Mission District.** Run an idealized release, inspect a fixed-scale concentration field, and compare the shortest walk with a route that trades walking time for lower modeled exposure.

Built with React / TypeScript, MapLibre / PMTiles, ECharts, FastAPI, NumPy, OSMnx, GeoParquet and DuckDB Spatial. The scientific core implements its own conservative diffusion solver and Dijkstra / A* searches.

## Run locally

Requirements: **Python 3.12**, **Node.js 22.12+** and npm. The repository includes an attributed OSM snapshot and prepared offline map assets (about 22 MB); no API key or additional geographic-data download is needed for the demo.

```bash
git clone https://github.com/roy-pyke/Flow.git
cd Flow
./setup.sh       # one-time online dependency installation and frontend build
./start.sh       # subsequent starts work offline
```

Open **http://127.0.0.1:8000**. On macOS, `Start Flow.command` also starts the prepared application. Keep its terminal open. Stop with Ctrl+C. Set `FLOW_PORT=8001 ./start.sh` if port 8000 is occupied.

The service listens only on `127.0.0.1`. UI assets, vector tiles, simulations, routes and exports use the local service. Street labels render with fonts already installed on the computer; no font files, glyph server or sprite service are requested. No external basemap or cloud computation is required. Setup downloads dependencies and the DuckDB Spatial extension once; ordinary offline use then needs no network connection.

## Explore

1. The app opens a default Mission District experiment.
2. Select **Start A**, **Destination B** or **Release source**, then click the map.
3. Set the diffusion coefficient and run a 30-minute simulation.
4. Move the timeline to inspect the field at a fixed time.
5. Adjust the route preference to compare distance, walking time and modeled exposure.
6. Run the six-point preference sweep, inspect reports, and export CSV / JSON.

The field uses a fixed concentration scale across frames. Routes use a **frozen field** at the selected time, including for walks longer than that simulated interval. Exposure is relative concentration integrated over walking time, measured in model seconds; it is not a pollutant dose or a health prediction.

## Implemented scope

- Real OpenStreetMap walking network in a 4 × 4 km projected rectangle: directed parallel edges and complete road polylines, with preserved raw snapshots, timestamps and hashes.
- Local PMTiles basemap, locally rendered street labels, OSM attribution and geographic overlays.
- Double-precision cell-centered finite-volume diffusion with zero-flux walls, Gaussian release of width 250 m, 160 × 160 cells at 25 m, and conservative explicit Euler updates.
- Automatic time steps at most `0.9 h² / (4 κ)`, with shortened steps to reach every 30-second output time through 1,800 seconds.
- Bilinear concentration interpolation and trapezoidal exposure integration over every segment of a road, sampled at spacing no larger than half a grid cell.
- Custom Dijkstra and A*, retaining parallel-edge identity. A* uses straight-line walking time as an admissible lower bound.
- Parameter sweeps for λ = 0, 0.5, 1, 2, 5, 10; sensitivity at κ = 5, 20, 50 and frozen times 0, 10, 30 minutes; separate solver, integration and search timings.
- Persisted experiments with parameters, data/source hashes and code version; JSON, CSV and Parquet outputs.

The routing objective is `length / 1.4 + λ × exposure`, with `exposure = ∫ c ds / 1.4`. The constant walking speed is 1.4 m/s. Neither wind, terrain nor buildings influences the diffusion model. Reflecting walls are an experimental assumption. Sampled trade-offs do not claim a complete Pareto frontier.

## Evidence and verification

- [Data quality](reports/data_quality.md)
- [Numerical validation](reports/validation.md)
- [Reproducible analysis and measured timings](reports/analysis.md)
- [Plan 1 delivery record and demonstration](artifacts/plan1-2026-09-17/PROGRESS.md)
- [Original development plan](PLAN_1.md) (Chinese)

```bash
.venv/bin/python -m pytest -q
npm --prefix frontend run build
npm --prefix frontend run lint
.venv/bin/python -m scripts.run_experiments
```

Checks cover constant-state preservation, mass drift below 1e-10, nonnegative diffusion, a cosine analytical solution with 32² / 64² / 128² refinement, NetworkX cost agreement, parallel/directed edges, unreachable destinations, equal endpoints, invalid clicks, constant-field road integrals, and API-to-export behavior. Euler remains first order in time; the approximately second-order convergence experiment jointly refines with `dt = 0.1 h²`.

Performance numbers in reports are measurements on the recorded machine, not universal guarantees. The initial geometry sampling table has a one-time preparation cost; subsequent frame integrations reuse it.

## Files and reproducibility

| Path | Contents |
| --- | --- |
| `backend/app/` | Diffusion, interpolation, graph algorithms, validation and local API |
| `frontend/` | Interactive map, controls, charts, timeline and reports |
| `configs/region.json` | Region, metric CRS and walking speed |
| `data/demo/` | Bundled OSM source snapshot, processed road tables, network and map assets |
| `data/manifest.json` | Dataset provenance and file hashes |
| `data/experiments/` | Locally generated scenario metadata and experiment JSON / Parquet (ignored by Git) |
| `scripts/` | Repeatable data preparation and experiment generation |
| `reports/` | Committed numerical, data-quality and analysis evidence |
| `artifacts/plan1-2026-09-17/` | Delivery record, browser evidence, video and original-source backup |

A simulation ID is derived from parameters, the dataset hash and numerical source hash. Evicted simulations are recomputed from saved metadata. Experiments remain on disk after a server restart. Refresh data deliberately; a new OSM snapshot can change routing and report results.

Data preparation commands and attribution are documented in [the bundled data README](data/demo/README.md). The included OSM source timestamp is **2026-09-17 17:56:05 UTC**. The processed graph contains **9,611 nodes and 28,498 directed edges**; all 44 connected components are retained. Its walking policy permits both directions, including on streets that are one-way for cars. Edges whose full geometry leaves the analysis rectangle are excluded.

After setup, rerunning preparation from the bundled snapshot works offline. `fetch_osm.py` reuses that snapshot unless `--refresh` is explicitly supplied; a fresh download requires internet access. Generated GeoParquet tables are audited with DuckDB Spatial, and `data/manifest.json` records artifact sizes and SHA-256 hashes. The local PMTiles archive is generated directly from the road geometries; no OSM standard raster tiles are downloaded or cached.

## API

`GET /api/config`, `POST /api/simulations`, `GET /api/simulations/{id}/frames/{frame}`, `POST /api/routes`, `POST /api/experiments`, `GET /api/experiments`, `GET /api/experiments/{id}/export?format=json|csv`, and `GET /api/reports`. The machine-readable schema is at `/openapi.json`.

## Data attribution and later work

Map and road data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). The bundled geographic database and derived map data are offered under the [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/); original source snapshots, provenance and licensing details are included in [data/demo/](data/demo/README.md). The numerical field is synthetic and explicitly labeled.

Caltrans PeMS / CWWP traffic ingestion, advection, implicit solvers, C++ optimization and observational calibration are later work. They are not part of this diffusion release. [PLAN_1.md](PLAN_1.md) preserves the original scope and rationale.
