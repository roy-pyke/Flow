# Browser and offline acceptance

Verified 2026-09-17 with the production build served by `./start.sh` on `127.0.0.1:8000`, using a dedicated Playwright Chromium session.

## Offline method

The browser cache was cleared with Chrome DevTools `Network.clearBrowserCache`, caching was disabled, cookies were cleared, and all HTTP requests outside `127.0.0.1` / `localhost` were blocked with `internetdisconnected`. The page was then reloaded. This is browser-level external-network isolation; the computer's Wi-Fi setting was not changed. Localhost remains available by design for this local application.

After the complete workflow, browser resource timing reported **zero external HTTP resources**. Browser console reported **0 errors, 0 warnings**. The locally bundled MapLibre worker loaded successfully, and PMTiles range requests returned HTTP 206.

## Verified user actions

- Cold page load: real streets and street names, three location markers, Gaussian field and two distinct routes render. See [laboratory.png](laboratory.png).
- λ = 0: both displayed routes become 3.74 km, with 0% model-exposure difference and 0 additional minutes. Dijkstra and A* can both be selected.
- Timeline moves from 10 to 30 minutes and updates the heat field and route metrics.
- κ changes from 20 to 50; running the simulation updates the field.
- Map click changes the release position; rerunning updates the field. Restoring example points returns the default scenario.
- Six-preference sweep generates the chart and all six rows, with convergence, sensitivity and algorithm benchmark charts visible below. See [experiments.png](experiments.png).
- Both CSV and JSON links downloaded files through the browser while external networking was blocked. The actual downloads are [browser-export.csv](browser-export.csv) and [browser-export.json](browser-export.json). Both contain six rows; data versions match; CSV includes simulation, source/endpoints, frame, coefficient and code/data provenance.
- Narrow viewport at 390 × 844: responsive layout has no document-level horizontal overflow. See [mobile.png](mobile.png).

[Demo video](demo.webm): approximately two minutes; actual captured interaction replayed at 2× speed, without audio. Performance measurements come from the reports, not the recording playback speed.

## Issues caught and fixed before delivery

- The first frontend build omitted MapLibre's worker module; explicit bundling restored the map and offline operation.
- Default coefficient serialization caused one restart-recovery case to fail; normalized validated defaults fixed it, with a regression test.
- Some distant out-of-region coordinates raised a projection exception; the API now returns a readable 422 input error, with a regression test.

This browser acceptance complements the 34 automated scientific, data, and API tests. It does not claim cross-browser or physical Wi-Fi-disconnection testing.
