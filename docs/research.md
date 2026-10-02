# Research extension: first executable milestone

This directory extends Flow V2 on `codex/research-v3`. The current milestone provides a reproducible static-field experiment: define a PDE problem, compute a field, integrate directed road polylines, compare complete small-graph candidate sets, and evaluate the selected route against independent references. Concentration is synthetic and relative. Dynamic travel, variable coefficients, higher-order transport, iterative implicit solvers, adjoints, adaptive error control and input uncertainty remain future work.

## Run and replay a PDE experiment

From the project root after `./setup.sh`:

```bash
.venv/bin/python -m scripts.run_research capabilities
.venv/bin/python -m scripts.run_research run configs/research/cosine_diffusion.json --output data/experiments/research/my-cosine
.venv/bin/python -m scripts.run_research verify data/experiments/research/my-cosine
.venv/bin/python -m scripts.run_research replay data/experiments/research/my-cosine --output data/experiments/research/my-cosine-replay
```

Each output directory must be new. A completed bundle contains configuration, diagnostics, provenance, full initial/output arrays and SHA-256 hashes. NPZ inputs are copied into the bundle. Publication renames a completed temporary directory; verification checks integrity and schema, not physical truth. Replay recomputes with the current implementation and reports the raw-field difference; cross-platform bitwise agreement is not required. The decision study below additionally archives its source files.

The Python entry is `backend.app.research.experiments.run_experiment`. `ExperimentSpec` separates physical identity from numerical identity. A grid refinement changes the numerical identity; changing diffusivity, boundary or initial data changes the physical identity. The configuration uses Cartesian meters, seconds, relative concentration, `(ny,nx)` arrays and increasing south-to-north rows. It supports constant, Gaussian, cosine and hash-checked NPZ initial data. Analytical initial fields default to actual cell averages; explicit `point_sample` preserves the old convention. NPZ inputs must declare their grid, domain and interpretation; they are not silently resampled or flipped.

The supported model still has constant diffusivity and constant velocity. `capabilities` enumerates legal method/backend/boundary/startup combinations; it reports supported initial types separately. Schema versions and unsupported combinations are rejected. Array-output and estimated cell-update admission budgets do not bound process RSS, sparse-factorization peak memory or wall time. The existing geographic UI retains its V2 request schema and source controls; this richer configuration is currently a CLI/Python interface.

## Field-to-road operator

`build_edge_observer(network, grid, ...)` constructs a sparse matrix `H` with one row per directed edge. `H @ c` is concentration integrated over travel time along the full polyline. Metadata records the grid, typed `(u,v,key)` identity, geometry hash, quadrature, speed, sample count, CSR bytes and construction cost.

The default trapezoid rule matches the existing sampling policy. `gauss2_grid` splits each segment at the bilinear reconstruction knots and integrates each resulting quadratic segment using two-point Gauss quadrature. This is an accurate reference for the declared reconstruction, not the continuous PDE solution. Periodic interpolation wraps across the seam. Inside a zero-flux or open domain, outer half-cells extend their nearest cell-center value constantly; outside points are rejected. This open-boundary reconstruction is a sampling convention, distinct from the PDE inflow-flux condition.

`apply_batch` supports frame batches and explicit chunks. Linear operator calls accept signed values for algebraic checks; routing requires finite nonnegative fields. `RoadNetwork.edge_exposures` caches operators up to eight entries / 64 MiB of CSR storage, with a separate eight-entry / 32 MiB sample-array cache. These are retention budgets, not peak-construction budgets. Oversized operators can be applied without being cached. Network geometry is immutable during a network instance's lifetime; construct a new instance after changing geometry.

`build_path_trajectory` constructs position and arrival time from an ordered edge sequence, speed and departure time. Repeated edges are preserved. It requires geometry joins to agree exactly, because inventing short uncharged connectors would alter the trajectory. This is trajectory construction only: temporal field integration and waiting costs are not implemented yet. The map still snaps clicks to graph nodes within 250 m and omits those off-network connectors. Saved route sweeps now retain node/edge sequences and typed edge keys, in addition to the dataset identity and costs.

## Decision counterexamples and references

```bash
.venv/bin/python -m scripts.run_decision_experiments --config configs/research/decision_study.json --output reports/research/my-decision-study
```

The five predefined cases use the production Backward Euler solver on a cosine Neumann diffusion problem and a fixed two-corridor graph. They include a large field error with the correct route, a small error with the wrong route, exact and near ties, and substantial positive regret. The preference parameter near a route switch is deliberately constructed from the thresholds. These are counterexamples, not estimates of how often real routes fail.

Independent references distinguish the continuous analytical solution, the exact semi-discrete cosine eigenmode, bilinear reconstruction and road quadrature. Regret is the reference cost of the chosen route minus the reference optimum over the declared candidate set. The simple-path oracle preserves parallel edges and refuses to claim completeness if its node/path budget is exceeded. It does not optimize arbitrary walks or time-dependent routes.

The decision module implements conditional `T ε`, `λ T ε`, `2 ε + η`, heterogeneous candidate intervals, direct path-difference intervals and a nonnegative lower-bound graph comparison. Inputs must satisfy the documented uniform-error/search/candidate assumptions. Bounds are numerically checked using floating-point arithmetic, without outward-rounded interval arithmetic. The output therefore records assumptions and evidence scope; it is not a machine-checked proof or an uncertainty estimate for observations.

Each study archives configuration, graph, raw fields, per-path CSV, a figure, report, source snapshot and dependency lock. Its manifest lists hashes for all artifacts except the manifest itself. Hashes establish file integrity, not authenticity. To repeat archived code, use the project's prepared Python interpreter from the saved `source_snapshot/` directory and pass `--config ../config.json --output /absolute/new/directory`. It uses the interpreter's current installed dependencies; it does not recreate an OS automatically.

## Measure the observation implementation

```bash
.venv/bin/python -m scripts.benchmark_observations --output reports/research/my-observation-benchmark
```

This measures only road observation on the bundled 28,498-edge Mission graph. It checks equal outputs before comparing reusable sparse multiplication with direct interpolation plus aggregation. Construction and application are separated, batch sizes and raw repetitions are saved, and independent curves separate quadrature error from field reconstruction error. Cold costs and repeated-use amortization matter: a fast warm multiplication alone does not establish a faster one-shot pipeline. This is not an end-to-end application or PDE-solver speedup claim.

## Prepare a new network without changing the bundled demo

```bash
.venv/bin/python scripts/prepare_region.py --output output/region-preview.json
.venv/bin/python scripts/prepare_network.py --source-dir data/demo --output-dir data/datasets/mission-rebuilt
```

The existing snapshot is validated before generation. The output uses `CURRENT.json` and immutable `versions/<id>/` directories. Readers must resolve `CURRENT.json` once with `scripts.dataset_io.resolve_dataset_dir`, verify that generation, and keep that directory for the operation. Failed staging leaves the previous complete generation usable. Concurrent publication is last-writer-wins; old generations are retained for existing readers.

`fetch_osm.py --source-dir data/demo --output-dir data/sources/mission` publishes a validated source copy; `--refresh` intentionally downloads a new snapshot. For a changed region, supply a matching configuration and a freshly acquired matching source. Subregion reuse is currently rejected. The network generator checks geometry finiteness, orientation, both endpoint distances, domain containment, parallel edges and Parquet roundtrips.

This new publication path produces research network tables and source provenance. PMTiles and DuckDB report generation remain the old demo pipeline; the UI still loads `data/demo`. Do not run the old basemap/report scripts expecting them to publish a new versioned map bundle.

## Recorded milestone evidence

On the local macOS environment, all **217 tests** passed with native execution required; frontend build and lint passed. The rebuilt Mission network retains 9,611 nodes and 28,498 directed edges. The cosine archive replay had zero raw-frame difference, and replay from the decision study's archived source reproduced all 35 saved arrays exactly in this environment. These are local results, not a new Linux CI claim.

The [five-case decision report](../reports/research/decision_baseline/REPORT.zh-CN.md) includes a nodal field error of about 0.000275 that nevertheless selects the wrong near-critical route, with reference objective regret of 0.059576 seconds. Another case has error 0.072629 and still selects a reference-optimal route. The deliberate construction and references are part of the saved report.

The [observation benchmark](../reports/research/observation_baseline/README.md) records warm application ratios of 21.75, 37.19 and 39.91 for batches of 1, 8 and 32 fields. Maximum discrepancy is below 3.5e-13 relative-concentration seconds. Full observation setup is about 77.77 ms for CSR versus 5.89 ms for samples. Consequently a single one-frame call is slower with fresh CSR setup (about 78.01 versus 11.10 ms); repeated one-frame use amortizes after roughly 15 calls on this run. Graph loading, PDE and routing are outside these ratios. Cold totals sum measured setup and median application; they are not repeated cold-start benchmarks.

To redraw the observation figure without rerunning any timings:

```bash
.venv/bin/python scripts/benchmark_observations.py --replot reports/research/observation_baseline/benchmark.json
```

## Validation and remaining scope

```bash
FLOW_REQUIRE_NATIVE=1 .venv/bin/python -m pytest -q
npm --prefix frontend run build
npm --prefix frontend run lint
```

The committed `reports/numerics/` and top-level original route reports are historical V2 evidence. They have not been regenerated for every source change in this branch. New evidence lives under `reports/research/`. This milestone establishes the static research workflow; it does not close the complete Research roadmap or establish calibrated real-world pollution decisions.
