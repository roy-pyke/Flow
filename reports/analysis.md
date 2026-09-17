# Reproducible analysis

Data SHA-256: `5d684aea94e9b61840e812bba1e804ad43817895329c7c935a036efe0a4d3ce6`. Environment: `macOS-26.5.2-arm64-arm-64bit`, Python 3.12.6.

All 6 lambda settings agree with NetworkX for both custom algorithms (absolute cost error < 1e-7).

| κ (m²/s) | Frozen time (min) | Shortest exposure | Weighted exposure | Solve (ms) | Route total (ms) |
|---|---|---|---|---|---|
| 5 | 0 | 508.027 | 1.868 | 10.16 | 27.82 |
| 5 | 10 | 492.666 | 2.923 | 10.16 | 18.29 |
| 5 | 30 | 465.239 | 5.857 | 10.16 | 18.94 |
| 20 | 0 | 508.027 | 1.868 | 27.31 | 17.38 |
| 20 | 10 | 453.003 | 7.711 | 27.31 | 18.33 |
| 20 | 30 | 380.047 | 28.194 | 27.31 | 18.63 |
| 50 | 0 | 508.027 | 1.868 | 51.61 | 19.49 |
| 50 | 10 | 395.034 | 22.733 | 51.61 | 19.12 |
| 50 | 30 | 292.493 | 70.812 | 51.61 | 20.38 |

![Route trade-offs](tradeoffs.png)

The Gaussian broadens with diffusion; effects depend on source placement, coefficient and frozen time. Exposure is dimensionless relative concentration integrated over walking time, with no calibration to pollutants. No wind, buildings, terrain or time-evolving exposure along a walk is modeled. The reflecting rectangular boundary is a modeling assumption.

Measurements include solver time, edge integration, and separate path-search timing in the JSON/CSV files. Browser transport/rendering is not included. Reproduce with `.venv/bin/python -m scripts.run_experiments`.
