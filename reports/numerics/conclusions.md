# Measured findings

These results are generated from the committed experiment scripts. They concern the recorded problems, machine and build; raw samples are preserved alongside the figures.

## Accuracy, structure and counterexamples

All **69** predeclared convergence/order and time-error-control checks passed. Maximum normalized transport mass-balance residual across 32 boundary scenarios was **5.385e-15**.

| Diffusion method | Final observed L2 time order |
|---|---:|
| Forward Euler | 1.00082 |
| Backward Euler | 0.99918 |
| Crank–Nicolson | 2.00000 |

The stiff spike experiment produced CN minimum **-0.94835** despite decreasing variance energy. With the tested Rannacher startup the minimum over the initial/output states was **0.00000**. This demonstrates the difference between linear stability and positivity; it is not a proof of positivity for arbitrary Rannacher runs.

IMEX remains first order and subject to the advection CFL. Upwind spatial convergence is approximately first order, including translated/decayed Fourier validation; its numerical diffusion is distinct from the physical κ. Open boundaries lose mass through modeled outflow while satisfying the discrete flux balance.

## Implementation speed

| Identical FE workload | NumPy median (ms) | C++ median (ms) | NumPy / C++ |
|---|---:|---:|---:|
| final_80 | 1.4581 | 0.6325 | 2.31× |
| final_160 | 16.8328 | 5.7070 | 2.95× |
| final_320 | 264.1471 | 84.6216 | 3.12× |
| final_640 | 5885.9721 | 1379.7108 | 4.27× |
| default_160_61 | 21.4921 | 10.3778 | 2.07× |
| conversion_160_61 | 21.9284 | 10.2716 | 2.13× |
| zero_duration_overhead | 0.0177 | 0.3336 | 0.05× |

The final-frame workloads show the benefit of a complete native loop and reusable buffers relative to the original vectorized NumPy FE implementation. Writing 61 frames reduces that advantage; a zero-duration call exposes native dispatch/build-validation overhead and can be slower. These timings exclude HTTP, JSON, browser rendering and the higher-level shared solver's additional normalized diagnostics. They do not measure an implicit-method speedup. See benchmark.json for exact timed scopes, raw samples, P90, memory and compiler flags.

## Cost at the same error

The table selects the fastest tested step satisfying each target, with all setup and solve work included. A dash means the tested steps did not attain the target; no extrapolated timing is substituted. The small fixed grid limits conclusions about large sparse systems.

| Problem | LU cache | L2 target | Method | Fastest measured median (ms) |
|---|---|---:|---|---:|
| diffusion | cold | 1e-03 | Forward Euler | 0.2851 |
| diffusion | cold | 1e-03 | Backward Euler | 0.6536 |
| diffusion | cold | 1e-03 | Crank–Nicolson | 0.6793 |
| diffusion | cold | 1e-04 | Forward Euler | 0.5372 |
| diffusion | cold | 1e-04 | Backward Euler | 1.4683 |
| diffusion | cold | 1e-04 | Crank–Nicolson | 0.6793 |
| diffusion | cold | 1e-05 | Forward Euler | — |
| diffusion | cold | 1e-05 | Backward Euler | — |
| diffusion | cold | 1e-05 | Crank–Nicolson | 0.7962 |
| diffusion | warm | 1e-03 | Forward Euler | 0.2752 |
| diffusion | warm | 1e-03 | Backward Euler | 0.4532 |
| diffusion | warm | 1e-03 | Crank–Nicolson | 0.4266 |
| diffusion | warm | 1e-04 | Forward Euler | 0.5406 |
| diffusion | warm | 1e-04 | Backward Euler | 1.2856 |
| diffusion | warm | 1e-04 | Crank–Nicolson | 0.4266 |
| diffusion | warm | 1e-05 | Forward Euler | — |
| diffusion | warm | 1e-05 | Backward Euler | — |
| diffusion | warm | 1e-05 | Crank–Nicolson | 0.4862 |
| transport | cold | 1e-03 | Explicit upwind | 1.3609 |
| transport | cold | 1e-03 | IMEX Euler | 2.4721 |
| transport | cold | 1e-04 | Explicit upwind | — |
| transport | cold | 1e-04 | IMEX Euler | — |
| transport | cold | 1e-05 | Explicit upwind | — |
| transport | cold | 1e-05 | IMEX Euler | — |
| transport | warm | 1e-03 | Explicit upwind | 1.3468 |
| transport | warm | 1e-03 | IMEX Euler | 2.1385 |
| transport | warm | 1e-04 | Explicit upwind | — |
| transport | warm | 1e-04 | IMEX Euler | — |
| transport | warm | 1e-05 | Explicit upwind | — |
| transport | warm | 1e-05 | IMEX Euler | — |

BE's stability permits larger steps but does not remove its first-order truncation error. It is useful when damping and robustness matter and the chosen tolerance permits that error. For this smooth-mode study CN can reach tight errors with fewer steps; cold LU costs can dominate at small size. Compare the recorded target-specific timings instead of assuming every implicit solve is faster.

## Effect on route decisions

All candidates use the same graph, endpoints, source, frozen time, preference and fine road-sampling points. Candidate routes are re-evaluated in the finest reference field. The reference is numerical, with an additional refinement check; it is not observed truth.

| Model | Reference refinement | Relative change in optimum cost | Candidate reference regrets (s) |
|---|---|---:|---|
| diffusion | 320² → 640² | 9.087e-06 | Forward Euler: 0.000e+00, Backward Euler: 0.000e+00, Crank–Nicolson: 0.000e+00 |
| advection_diffusion | 320² → 640² | 6.205e-04 | Explicit upwind: 0.000e+00, IMEX Euler: 0.000e+00 |

In this scenario, measurable field and edge-exposure errors coexist with unchanged selected paths. That is a limited observation, not evidence that PDE accuracy never affects route choice. NetworkX checks agree independently for each frozen graph; graph correctness and PDE sensitivity answer different questions. Transport reference uncertainty is larger because the first-order upwind discretization converges more slowly.

## Limits and reproducibility

The public app uses a prescribed constant wind, synthetic relative concentration, a fixed rectangle and frozen-field walking exposure. There are no meteorological observations, calibration, buildings, reactions, traffic-flow PDEs or time-dependent route optimization. The native backend currently accelerates zero-flux explicit diffusion only; sparse implicit work uses SciPy/SuperLU. Interactive budgets limit each field array to 64 MiB, retained fields to 256 MiB, and retained LU arrays separately to 256 MiB; temporary allocations and the interpreter are additional. Larger experiments belong in scripts. Full commands and mathematical conventions are in [methods.md](methods.md), environment/source hashes in [summary.json](summary.json).
