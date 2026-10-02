# Research workflows: spatial and temporal decisions

This directory extends Flow V2 on `codex/research-v3`. It provides reproducible static and time-aware experiments: define a PDE problem, compute saved fields, integrate complete journeys, compare paths and evaluate results against independent references. Concentration is synthetic and relative. Time-expanded search returns the optimum of its declared finite graph. Pure diffusion additionally supports positive static material fields with conservative harmonic face fluxes; see [variable diffusion](variable_diffusion.md) for the derivation, inputs and independent references. General continuous-time optimization, variable wind, source terms, higher-order transport, iterative implicit solvers, adjoints, adaptive error control and input uncertainty remain future work.

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

Diffusivity can be a nonnegative scalar or, for closed/periodic pure diffusion, a strictly positive cell material field declared through an aligned `layered` or hash-checked `npz` input. Variable fields use NumPy FE or SciPy BE/CN; explicit C++ requests are rejected and `auto` FE selects NumPy by capability. Advection-diffusion retains constant diffusivity and wind. `capabilities` enumerates legal combinations and input types. Schema versions and unsupported combinations are rejected. Array-output and estimated cell-update admission budgets do not bound process RSS, sparse-factorization peak memory or wall time. The existing geographic UI retains its V2 request schema and source controls; this richer configuration is currently a CLI/Python interface.

## Field-to-road operator

`build_edge_observer(network, grid, ...)` constructs a sparse matrix `H` with one row per directed edge. `H @ c` is concentration integrated over travel time along the full polyline. Metadata records the grid, typed `(u,v,key)` identity, geometry hash, quadrature, speed, sample count, CSR bytes and construction cost.

The default trapezoid rule matches the existing sampling policy. `gauss2_grid` splits each segment at the bilinear reconstruction knots and integrates each resulting quadratic segment using two-point Gauss quadrature. This is an accurate reference for the declared reconstruction, not the continuous PDE solution. Periodic interpolation wraps across the seam. Inside a zero-flux or open domain, outer half-cells extend their nearest cell-center value constantly; outside points are rejected. This open-boundary reconstruction is a sampling convention, distinct from the PDE inflow-flux condition.

`apply_batch` supports frame batches and explicit chunks. Linear operator calls accept signed values for algebraic checks; routing requires finite nonnegative fields. `RoadNetwork.edge_exposures` caches operators up to eight entries / 64 MiB of CSR storage, with a separate eight-entry / 32 MiB sample-array cache. These are retention budgets, not peak-construction budgets. Oversized operators can be applied without being cached. Network geometry is immutable during a network instance's lifetime; construct a new instance after changing geometry.

`build_path_trajectory` constructs position and arrival time from an ordered edge sequence, speed and departure time. Repeated edges are preserved. Its optional `waits_s` has one wait before each edge occurrence plus a final wait. Stationary waits contribute local concentration integrated over elapsed time. Geometry joins must agree exactly, because inventing short uncharged connectors would alter the trajectory. Each edge finishes at `edge_start + edge.length_m/speed`, matching the search clock even at strict horizon boundaries. The map still snaps clicks to graph nodes within 250 m and omits those off-network connectors. Saved route sweeps retain node/edge sequences and typed edge keys, in addition to the dataset identity and costs.

## Time-aware observations and finite graph search

```bash
.venv/bin/python -m scripts.run_temporal_experiments --config configs/research/temporal_study.json --output reports/research/my-temporal-study
.venv/bin/python -m scripts.run_temporal_experiments --verify reports/research/my-temporal-study
.venv/bin/python -m scripts.run_temporal_experiments --replot reports/research/my-temporal-study
```

Outputs must be new. Replot reads saved data and writes a separate sibling PNG, leaving the sealed bundle unchanged. A bundle contains full arrays, configuration, graph, CSV, figure, report and source snapshot. Verification checks artifact hashes, configuration identity, candidate identities and raw-array schema. Hashes establish integrity, not authenticity or scientific truth. Replay from `source_snapshot/` with `PYTHONDONTWRITEBYTECODE=1 /path/to/Flow-Research/.venv/bin/python -m scripts.run_temporal_experiments --config ../config.json --output /absolute/new/directory`; disabling bytecode writes preserves the sealed archive. The protocol preflights aggregate stored arrays across all cases, not just one case. This budget does not bound peak process memory.

`sample_time_series(grid, times, frames, positions, query_times, boundary)` uses bilinear space and linear physical-time interpolation. Nonuniform frame times are supported; all queries and the entire journey must be covered by the saved frames. No last-frame freezing or temporal extrapolation is performed. Spatial walls and periodic seams use the same rules as the static observer.

`build_trajectory_observer(grid, times, trajectory, ...)` returns a sparse row acting on flattened `(frame,y,x)` values. Its weights are nonnegative and sum to journey duration, including waits. `gauss2_split` splits at trajectory bends, spatial reconstruction knots and frame times. The integrand is at most cubic within each piece, so two-point Gauss integrates the declared reconstruction to roundoff. It does not remove PDE discretization or output-frame interpolation error. `trapezoid` retains mandatory splits and accepts independent `spatial_step_m` and `time_step_s` controls. Its sample budget is checked before allocating evaluation arrays and the sparse operator.

`TimeDependentExposure` in `backend.app.research.travel` connects nonnegative owned field arrays to edge and waiting integrals. It enforces exact geometry-to-node joins and a bounded scalar cache. `trajectory_from_schedule` reconstructs a whole journey from explicit actions, rejecting gaps or uncharged connectors; the study independently re-evaluates this trajectory to check action accounting.

`solve_time_expanded` in `backend.app.research.dynamic_routing` keeps a label for each **node and time**. The lattice has origin zero; an off-lattice departure is an explicit source state. A nonterminal edge travels for its actual duration and rounds arrival upward to the next lattice time; the extra wait is recorded and charged time plus exposure. Arrival at the destination terminates at its actual time. Optional voluntary waits advance one time state. The finite horizon admits cycles without infinite search; node/time state and transition budgets reject exhaustion without claiming a partial optimum. Deterministic ties preserve typed parallel-edge identities.

The returned optimum applies only to that rounded finite graph, its supplied costs and declared waiting policy. Constant travel-time FIFO does not justify keeping only one lowest-cost label per spatial node when exposure changes with time. A saved counterexample gives cost **4** using node/time states versus **102** when the necessary later arrival is discarded. Time-grid refinement is descriptive: different rounding/waiting schedules are different discrete problems, so generic monotonicity or continuous-time error bounds are not promised.

The [temporal report](../reports/research/temporal_baseline/REPORT.md) includes stationary-equivalent, slowly evolving and freezing-failure cases using the production CN diffusion solver. Every path is evaluated over its full journey under an independent continuous analytical solution. In the rapid-change case, frozen-field selection has reference objective regret **2.645896 s**; moving-field selection has zero regret over the two no-wait candidates. Separate studies isolate quadrature and saved-frame interpolation error, with nonuniform frames, repeated edges and waits. Translating-wave, periodic-boundary and intentional-wait behavior are also checked by tests. This remains synthetic model evidence.

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

The variable-diffusion milestone passes **459 tests** with native execution required. Its [independent study](../reports/research/variable_diffusion_baseline/REPORT.md) records x-direction spatial orders 1.9933, 1.9983 and 1.9996, first-order FE/BE and second-order CN time convergence, and a layered harmonic-face flux error of 1.33e-14. All 66 archived arrays replay exactly from the saved source in the recorded local environment. The [operator benchmark](../reports/research/variable_operator_baseline/benchmark.json) finds lower retained array storage for matrix-free application but faster warm CSC multiplication on this configuration; it does not establish an end-to-end speedup.

The completed temporal milestone passed **318 tests** with native execution required. Frontend build and lint passed in the first milestone; no frontend files changed in the temporal or variable-diffusion milestones. The rebuilt Mission network retains 9,611 nodes and 28,498 directed edges. The cosine archive replay had zero raw-frame difference, and replay from the static decision study's archived source reproduced all 35 saved arrays exactly in its recorded environment. The temporal study adds 15 raw arrays, reproduced exactly from its archived source in the same local environment. These are local results, not a new Linux CI claim.

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

The committed `reports/numerics/` and top-level original route reports are historical V2 evidence. They have not been regenerated for every source change in this branch. Each `reports/research/` bundle retains its original code and measurement provenance. The current implementation adds static, temporal and variable-diffusion research workflows; it does not close the complete Research roadmap or establish calibrated real-world pollution decisions. These research controls are CLI/Python interfaces; the map UI remains a frozen-field experiment.
