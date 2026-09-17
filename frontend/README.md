# Flow browser laboratory

React 19, TypeScript, Vite, MapLibre GL JS, local PMTiles, and Apache ECharts.

From the repository root, use the one-command launcher documented in the main README to serve the production build and Python API together. The production browser makes no external resource requests: map tiles, JavaScript, CSS, street text, data, and API responses all come from the local server. Street and neighborhood labels use system fonts. OpenStreetMap attribution remains visible.

For frontend development:

```sh
cd frontend
npm ci
npm run dev
```

The development server proxies `/api` and `/data` to `http://127.0.0.1:8000`; run the Python service separately. `npm run build` performs TypeScript checking and creates `dist/`. `npm run lint` runs Oxlint.

The laboratory supports point selection, a 5–50 m²/s diffusion coefficient, 61 frozen-field snapshots, a fixed absolute concentration scale, A*/Dijkstra route comparison, six-value preference sweeps, and JSON/CSV export. Reports display convergence, sensitivity, timings, search effort, road quality, and full reproducible data. The PDE and routing computations run exclusively in the local Python service.

The display is an idealized experiment, not a real-world pollution or health prediction. See the repository model documentation for equations, units, assumptions, and validation.
