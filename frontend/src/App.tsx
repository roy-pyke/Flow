import { useCallback, useEffect, useMemo, useState } from 'react'
import MapView from './MapView'
import Chart from './Chart'
import EvidenceCharts from './EvidenceCharts'
import { api, minutes, number } from './types'
import type { Config, Experiment, Frame, Point, PointKind, Reports, Routes, Simulation } from './types'
import './App.css'

const initialPoints: Record<PointKind, Point> = { start: [-122.43, 37.76], end: [-122.40, 37.76], source: [-122.4149, 37.7599] }
const pretty = (key: string) => key.replaceAll('_', ' ').replace(/\b\w/g, letter => letter.toUpperCase())
const message = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong. Please try again.'
const same = (a: Point, b: Point) => a[0] === b[0] && a[1] === b[1]
const formatValue = (value: unknown): string => typeof value === 'number' ? Math.abs(value) > 0 && Math.abs(value) < 0.001 ? value.toExponential(2) : number(value, 4) : typeof value === 'boolean' ? value ? 'Pass' : 'Fail' : String(value)
function ReportSection({ title, data }: { title: string; data: Record<string, unknown> | null | undefined }) {
  const metrics = Object.entries(data?.quality && typeof data.quality === 'object' ? data.quality : data ?? {}).filter(([, value]) => ['number', 'string', 'boolean'].includes(typeof value)).slice(0, 9)
  return <section className="report-card"><div className="section-eyebrow">VERIFICATION</div><h3>{title}</h3>{data ? <><dl className="report-metrics">{metrics.map(([key, value]) => <div key={key}><dt>{pretty(key)}</dt><dd>{formatValue(value)}</dd></div>)}</dl><details><summary>View complete reproducible report</summary><pre>{JSON.stringify(data, null, 2)}</pre></details></> : <p className="muted">No saved report yet. Run the validation and analysis commands described in the project README.</p>}</section>
}
function RouteCard({ name, kind, route }: { name: string; kind: 'shortest' | 'weighted'; route: Routes['shortest'] | undefined }) {
  return <div className={`route-card ${kind}`}><div className="route-card-head"><span><i className={`line-key ${kind}`} />{name}</span><span>{kind === 'shortest' ? 'BASELINE' : 'OPTIMIZED'}</span></div><div className="route-main"><strong>{route ? number(route.distance_m / 1000, 2) : '—'}<small>km</small></strong><span>{route ? minutes(route.time_s) : '— min'} walk</span></div><div className="route-secondary"><span>Model exposure</span><strong>{number(route?.exposure, 2)}<small> rel. s</small></strong></div></div>
}
function App() {
  const [page, setPage] = useState<'lab' | 'reports'>('lab')
  const [config, setConfig] = useState<Config | null>(null)
  const [points, setPoints] = useState(initialPoints)
  const [mode, setMode] = useState<PointKind>('start')
  const [kappa, setKappa] = useState(20)
  const [lambda, setLambda] = useState(5)
  const [algorithm, setAlgorithm] = useState<'astar' | 'dijkstra'>('astar')
  const [simulation, setSimulation] = useState<Simulation | null>(null)
  const [simulated, setSimulated] = useState<{ source: Point; kappa: number } | null>(null)
  const [frameIndex, setFrameIndex] = useState(20)
  const [frame, setFrame] = useState<Frame | null>(null)
  const [routes, setRoutes] = useState<Routes | null>(null)
  const [reports, setReports] = useState<Reports | null>(null)
  const [experiment, setExperiment] = useState<Experiment | null>(null)
  const [busy, setBusy] = useState('Preparing local laboratory')
  const [routeBusy, setRouteBusy] = useState(false)
  const [error, setError] = useState('')
  const [showHeat, setShowHeat] = useState(true)
  const [playing, setPlaying] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const dirty = !!simulated && (!same(simulated.source, points.source) || simulated.kappa !== kappa)

  useEffect(() => {
    const controller = new AbortController()
    const boot = async () => {
      try {
        const settings = await api<Config>('/config', undefined, controller.signal)
        setConfig(settings); setPoints(settings.defaults)
        api<Reports>('/reports', undefined, controller.signal).then(setReports).catch(() => {})
        const result = await api<Simulation>('/simulations', { source: settings.defaults.source, kappa: 20 }, controller.signal)
        setSimulation(result); setSimulated({ source: settings.defaults.source, kappa: 20 }); setBusy('')
      } catch (cause) { if (!controller.signal.aborted) { setError(message(cause)); setBusy('') } }
    }
    void boot()
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!simulation || dirty) return
    const controller = new AbortController()
    const timer = window.setTimeout(async () => {
      setRouteBusy(true)
      try {
        const [nextFrame, nextRoutes] = await Promise.all([
          api<Frame>(`/simulations/${simulation.id}/frames/${frameIndex}`, undefined, controller.signal).then(value => { if (!controller.signal.aborted) setFrame(value); return value }),
          api<Routes>('/routes', { simulation_id: simulation.id, frame: frameIndex, start: points.start, end: points.end, lambda_weight: lambda, algorithm }, controller.signal),
        ])
        if (!controller.signal.aborted) { setFrame(nextFrame); setRoutes(nextRoutes); setError(''); setRouteBusy(false) }
      } catch (cause) { if (!controller.signal.aborted) { setRoutes(null); setError(message(cause)); setRouteBusy(false); setPlaying(false) } }
    }, 100)
    return () => { clearTimeout(timer); controller.abort() }
  }, [simulation, frameIndex, points.start, points.end, lambda, algorithm, dirty, refresh])

  useEffect(() => {
    if (!playing || !simulation || dirty || routeBusy) return
    const timer = window.setTimeout(() => { setExperiment(null); setFrameIndex(value => value >= simulation.times.length - 1 ? (setPlaying(false), value) : value + 1) }, 450)
    return () => clearTimeout(timer)
  }, [playing, simulation, dirty, routeBusy, frameIndex])

  const pick = useCallback((point: Point) => {
    if (busy) return
    const bounds = config?.region.geographic_bounds
    if (bounds && (point[0] < bounds[0] || point[0] > bounds[2] || point[1] < bounds[1] || point[1] > bounds[3])) {
      setError('Choose a point inside the 4 × 4 km study area.'); return
    }
    setPoints(previous => ({ ...previous, [mode]: point })); setPlaying(false); setError(''); setExperiment(null)
  }, [config, mode, busy])
  const runSimulation = async () => {
    setBusy('Solving the diffusion field'); setError(''); setPlaying(false)
    try {
      const result = await api<Simulation>('/simulations', { source: points.source, kappa })
      setSimulation(result); setSimulated({ source: [...points.source], kappa }); setExperiment(null)
    } catch (cause) { setError(message(cause)) }
    finally { setBusy('') }
  }
  const runExperiment = async () => {
    if (!simulation || dirty) return
    setBusy('Scanning six route preferences'); setError(''); setPlaying(false)
    try {
      const result = await api<Experiment>('/experiments', { simulation_id: simulation.id, frame: frameIndex, start: points.start, end: points.end, lambda_weight: lambda, algorithm })
      setExperiment(result); setPage('reports')
      api<Reports>('/reports').then(setReports).catch(() => {})
    } catch (cause) { setError(message(cause)) }
    finally { setBusy('') }
  }
  const exportComparison = () => {
    const content = { data_version: config?.data_version, model: 'Idealized pure diffusion with a frozen field', parameters: { source: simulated?.source, kappa: simulated?.kappa, start: points.start, end: points.end, lambda_weight: lambda, time_s: frame?.time_s, algorithm }, simulation, routes }
    const url = URL.createObjectURL(new Blob([JSON.stringify(content, null, 2)], { type: 'application/json' }))
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = `flow-comparison-${simulation?.id ?? 'export'}.json`; anchor.click(); URL.revokeObjectURL(url)
  }
  const chart = useMemo(() => ({
    tooltip: { trigger: 'axis' }, grid: { left: 65, right: 28, top: 40, bottom: 50 },
    xAxis: { type: 'value', name: 'Walking time (min)', nameLocation: 'middle', nameGap: 32, scale: true, splitLine: { lineStyle: { color: '#ecede8' } } },
    yAxis: { type: 'value', name: 'Model exposure (rel. s)', nameTextStyle: { align: 'left' }, scale: true, splitLine: { lineStyle: { color: '#ecede8' } } },
    series: [{ type: 'line', name: 'Sampled trade-offs', symbolSize: 10, smooth: false, lineStyle: { width: 2 }, data: experiment?.rows.map(row => [Number((row.time_s / 60).toFixed(3)), Number(row.exposure.toFixed(3))]) ?? [], label: { show: true, position: 'top', formatter: (params: { dataIndex: number }) => `λ ${experiment?.rows[params.dataIndex].lambda_weight}` } }],
  }), [experiment])
  const perfChart = useMemo(() => ({
    tooltip: { trigger: 'axis' }, grid: { left: 92, right: 30, top: 30, bottom: 35 },
    xAxis: { type: 'value', name: 'ms', splitLine: { lineStyle: { color: '#ecede8' } } }, yAxis: { type: 'category', data: ['Simulation', 'Integration', 'Search'] },
    series: [{ type: 'bar', barWidth: 22, data: [simulation?.elapsed_ms ?? 0, routes?.timings?.integration_ms ?? 0, routes?.timings?.search_ms ?? 0], itemStyle: { borderRadius: [0, 4, 4, 0] } }],
  }), [simulation, routes])
  const reduction = routes && routes.shortest.exposure > 1e-10 ? 100 * (1 - routes.weighted.exposure / routes.shortest.exposure) : 0
  const extraTime = routes ? (routes.weighted.time_s - routes.shortest.time_s) / 60 : 0
  const selectedTime = simulation?.times[frameIndex] ?? frameIndex * 30

  return <div className="app-shell">
    <header className="app-header"><a className="brand" href="#" onClick={event => { event.preventDefault(); setPage('lab') }}><span className="brand-symbol"><i /><i /><i /></span><strong>flow<span>.</span></strong><span className="brand-divider" /><span className="brand-description">SPATIAL EXPERIMENT LAB</span></a><nav aria-label="Main navigation"><button className={page === 'lab' ? 'active' : ''} onClick={() => setPage('lab')}>Laboratory</button><button className={page === 'reports' ? 'active' : ''} onClick={() => setPage('reports')}>Experiments <span>↗</span></button></nav><div className="local-status"><span />{config ? 'Local workspace' : 'Connecting locally'}</div></header>
    <div className="intro-strip"><div><span className="section-eyebrow">EXPERIMENT 001 / URBAN DIFFUSION</span><h1>A different way from A to B.</h1></div><p>Explore how a little extra distance<br />can change your exposure to a modeled field.</p><span className="model-tag">IDEALIZED MODEL</span></div>
    {error && <div className="alert" role="alert"><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}
    <main className={`laboratory ${page !== 'lab' ? 'hidden' : ''}`}>
      <aside className="parameters panel">
        <div className="panel-title"><h2>Set up the experiment</h2><span>01</span></div>
        <div className="parameter-section"><label className="field-title">Place your points <span>SELECT, THEN CLICK MAP</span></label><div className="point-list">{(['start', 'end', 'source'] as const).map(kind => <button key={kind} className={`point-button ${mode === kind ? 'selected' : ''}`} onClick={() => setMode(kind)}><span className={`point-icon ${kind}`}>{kind === 'start' ? 'A' : kind === 'end' ? 'B' : '✳'}</span><span><strong>{kind === 'start' ? 'Starting point' : kind === 'end' ? 'Destination' : 'Release source'}</strong><small>{number(points[kind][1], 4)}, {number(points[kind][0], 4)}</small></span><span className="point-action">⌖</span></button>)}</div><button className="text-button" disabled={!config || !!busy} onClick={() => { if (config) { setPoints(config.defaults); setKappa(20); setLambda(5); setFrameIndex(20); setError(''); setExperiment(null) } }}>↺ Restore example points</button></div>
        <div className="parameter-section"><div className="field-title">Diffusion coefficient <span className="math-symbol">κ</span></div><div className="range-value"><strong>{kappa}</strong><span>m²/s</span></div><input aria-label="Diffusion coefficient" disabled={!!busy} type="range" min="5" max="50" step="1" value={kappa} onChange={event => { setKappa(Number(event.target.value)); setPlaying(false); setExperiment(null) }} /><div className="range-labels"><span>5 · slower spread</span><span>50 · faster spread</span></div><button className="primary-button" disabled={!!busy || !config} onClick={runSimulation}><span>{busy && !busy.includes('Scanning') ? '◌' : '▶'}</span>{busy && !busy.includes('Scanning') ? 'Running simulation…' : 'Run simulation'}<span>→</span></button>{dirty && <p className="pending-note">Parameters changed. Run the simulation to update the field.</p>}</div>
        <div className="parameter-section route-preferences"><div className="field-title">Exposure preference <span className="math-symbol">λ</span></div><div className="range-value"><strong>{lambda.toFixed(1)}</strong><span>penalty weight</span></div><input aria-label="Exposure preference" disabled={!!busy} type="range" min="0" max="10" step="0.5" value={lambda} onChange={event => { setLambda(Number(event.target.value)); setExperiment(null) }} /><div className="range-labels"><span>Shortest path</span><span>Less exposure</span></div><label className="algorithm-label">Search algorithm<select aria-label="Search algorithm" disabled={!!busy} value={algorithm} onChange={event => { setAlgorithm(event.target.value as 'astar' | 'dijkstra'); setExperiment(null) }}><option value="astar">A* search</option><option value="dijkstra">Dijkstra</option></select></label></div>
        <div className="model-note"><span className="note-icon">i</span><p>A mathematical experiment.<br />Pure diffusion, no wind or buildings. Routes use a frozen snapshot, at a walking speed of 1.4 m/s.</p></div>
      </aside>
      <section className="map-column" aria-label="Map and time controls"><MapView config={config} points={points} mode={mode} onPick={pick} simulation={simulation} frame={frame} routes={dirty || routeBusy ? null : routes} showHeat={showHeat} visible={page === 'lab'} /><div className="timeline"><div className="timeline-heading"><div><span className="section-eyebrow">TIME SINCE RELEASE</span><strong>{Math.floor(selectedTime / 60).toString().padStart(2, '0')}<span>:</span>{(selectedTime % 60).toString().padStart(2, '0')}<small>min : sec</small></strong></div><label className="layer-switch"><input type="checkbox" checked={showHeat} onChange={event => setShowHeat(event.target.checked)} /> Diffusion layer</label></div><div className="timeline-track"><button className="play-button" aria-label={playing ? 'Pause timeline' : 'Play timeline'} disabled={!simulation || dirty || !!busy} onClick={() => { if (frameIndex === 60) setFrameIndex(0); setPlaying(!playing) }}>{playing ? 'Ⅱ' : '▶'}</button><div className="time-range"><input aria-label="Time since release" type="range" min="0" max={simulation ? simulation.times.length - 1 : 60} step="1" value={frameIndex} disabled={!simulation || dirty || !!busy} onChange={event => { setFrameIndex(Number(event.target.value)); setPlaying(false); setExperiment(null) }} /><div className="range-labels"><span>0 min</span><span>10 min</span><span>20 min</span><span>30 min</span></div></div></div><div className="concentration-key"><span>Relative concentration</span><i /><span>0</span><span>1</span><small>Fixed scale across all frames</small></div></div></section>
      <aside className="results panel"><div className="panel-title"><h2>Compare the routes</h2><span>02</span></div><div className="results-status"><span className={routeBusy || busy || dirty ? 'status-dot waiting' : 'status-dot'} />{dirty ? 'Simulation update needed' : routeBusy ? 'Computing routes…' : busy ? busy : routes ? `Frozen field at ${minutes(frame?.time_s ?? selectedTime)}` : 'Waiting for simulation'}</div><div className={routeBusy || dirty ? 'route-results stale' : 'route-results'}><RouteCard name="Shortest route" kind="shortest" route={routes?.shortest} /><RouteCard name="Exposure-aware" kind="weighted" route={routes?.weighted} /><div className="takeaway"><span>THE TRADE-OFF</span><strong>{routes ? `${number(reduction)}%` : '—'}<small> less model exposure</small></strong><p>{routes ? `${number(extraTime)} extra minutes of walking at λ = ${lambda.toFixed(1)}.` : 'Your route comparison will appear here.'}</p></div></div><button className="secondary-button" disabled={!simulation || !!busy || dirty || routeBusy} onClick={() => setRefresh(value => value + 1)}>↻ Compare routes</button><div className="computation"><span className="section-eyebrow">COMPUTATION</span><div><span>Diffusion solve</span><strong>{number(simulation?.elapsed_ms)} ms</strong></div><div><span>Road integration</span><strong>{number(routes?.timings?.integration_ms)} ms</strong></div><div><span>Route search</span><strong>{number(routes?.timings?.search_ms)} ms</strong></div><div><span>Mass drift</span><strong>{simulation?.mass_drift === undefined ? '—' : simulation.mass_drift.toExponential(1)}</strong></div></div><button className="experiment-link" disabled={!simulation || !!busy || dirty || routeBusy} onClick={runExperiment}><span><strong>Find the trade-off</strong><small>Scan 6 preferences & view the report</small></span><span>↗</span></button><button className="text-button export-button" disabled={!routes || dirty || routeBusy || !!busy} onClick={exportComparison}>↓ Export current comparison</button></aside>
    </main>
    {page === 'reports' && <main className="reports-page"><div className="report-heading"><div><span className="section-eyebrow">REPRODUCIBLE RESULTS</span><h2>From model to evidence.</h2><p>Inspect the sampled trade-offs, measured performance, and numerical checks.</p></div><div className="report-actions"><button className="secondary-button" onClick={() => { api<Reports>('/reports').then(setReports).catch(cause => setError(message(cause))) }}>↻ Refresh reports</button><button className="primary-button" disabled={!simulation || dirty || !!busy || routeBusy} onClick={runExperiment}>{busy ? 'Working…' : 'Run parameter sweep'}<span>→</span></button></div></div><div className="report-grid"><section className="report-card tradeoff-chart"><div className="card-heading"><div><span className="section-eyebrow">ROUTE PREFERENCE SWEEP</span><h3>Time versus model exposure</h3></div>{experiment && <div className="download-links"><a href={`/api/experiments/${experiment.id}/export?format=csv`} download>↓ CSV</a><a href={`/api/experiments/${experiment.id}/export?format=json`} download>↓ JSON</a></div>}</div>{experiment ? <><Chart option={chart} label="Sampled route trade-offs: walking time versus model exposure" /><p className="chart-caption">λ = 0, 0.5, 1, 2, 5, 10. These are sampled optima, not a complete Pareto frontier.</p></> : <div className="chart-empty"><span>↗</span><strong>One route is only part of the story.</strong><p>Run a parameter sweep to see the cost of reducing exposure at six different preferences.</p><button className="secondary-button" disabled={!simulation || dirty || !!busy || routeBusy} onClick={runExperiment}>Run the experiment</button></div>}</section><section className="report-card"><span className="section-eyebrow">MEASURED ON THIS MACHINE</span><h3>Where the computation happens</h3><Chart option={perfChart} label="Simulation, road integration, and search duration in milliseconds" /><p className="chart-caption">Wall-clock time for the latest simulation and route request. Results vary with hardware and caching.</p></section></div>{experiment && <section className="report-card sweep-table"><h3>Parameter sweep</h3><div className="table-scroll"><table><thead><tr><th>Preference λ</th><th>Distance</th><th>Walking time</th><th>Model exposure</th><th>Objective cost</th></tr></thead><tbody>{experiment.rows.map(row => <tr key={row.lambda_weight}><td>{row.lambda_weight}</td><td>{number(row.distance_m / 1000, 3)} km</td><td>{minutes(row.time_s)}</td><td>{number(row.exposure, 3)} rel. s</td><td>{number(row.cost, 3)}</td></tr>)}</tbody></table></div></section>}<EvidenceCharts reports={reports} /><div className="report-grid thirds"><ReportSection title="Road data quality" data={reports?.data_quality} /><ReportSection title="Numerical & algorithm validation" data={reports?.validation} /><ReportSection title="Sensitivity & benchmark analysis" data={reports?.analysis} /></div><section className="assumptions"><span className="note-icon">i</span><div><h3>Read the model before reading the result.</h3><p>Concentration is a dimensionless relative field from a Gaussian release (initial peak 1, width 250 m). The 160 × 160 cell-centered grid uses 25 m cells, zero-flux boundaries, and a stable explicit Euler step. Exposure is the integral of concentration over walking time in a frozen field. This laboratory does not predict real pollution or health risk. Roads and model data are served locally; map labels use system fonts.</p><code>∂c/∂t = κ∇²c &nbsp; · &nbsp; Dₑ = ∫ₑ c ds/v &nbsp; · &nbsp; wₑ = ℓₑ/v + λDₑ</code></div></section></main>}
    <footer className="app-footer"><span>FLOW LAB <i /> IDEALIZED DIFFUSION EXPERIMENT</span><span>{config ? `Dataset ${config.data_version.slice(0, 18)}` : 'Local data'}<b>·</b>160 × 160 grid<b>·</b>61 time frames</span><span>Built to explore. Designed to question.</span></footer>
  </div>
}
export default App
