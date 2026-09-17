import { useMemo } from 'react'
import Chart from './Chart'
import { number } from './types'
import type { Reports } from './types'

type Convergence = { n: number; l2_error: number; observed_order: number | null }
type Sensitivity = { kappa: number; time_s: number; shortest_exposure: number; weighted_exposure: number; weighted_distance_m: number; shortest_distance_m: number }
type Benchmark = { lambda_weight: number; algorithm: string; cost: number; search_ms: number; visited: number }
const splitLine = { lineStyle: { color: '#ecede8' } }

export default function EvidenceCharts({ reports }: { reports: Reports | null }) {
  const convergence = useMemo(() => (Array.isArray(reports?.validation?.convergence) ? reports.validation.convergence : []) as Convergence[], [reports])
  const sensitivity = useMemo(() => (Array.isArray(reports?.analysis?.sensitivity) ? reports.analysis.sensitivity : []) as Sensitivity[], [reports])
  const benchmark = useMemo(() => (Array.isArray(reports?.analysis?.benchmark) ? reports.analysis.benchmark : []) as Benchmark[], [reports])
  const convergenceChart = useMemo(() => ({
    tooltip: { trigger: 'axis', valueFormatter: (value: number) => value.toExponential(4) },
    legend: { bottom: 0, textStyle: { fontSize: 10, color: '#829079' } },
    grid: { left: 83, right: 30, top: 45, bottom: 80 },
    xAxis: { type: 'log', logBase: 2, name: 'Cells per side (n)', nameLocation: 'middle', nameGap: 30, min: 32, max: 128, splitLine },
    yAxis: { type: 'log', name: 'L² error', nameTextStyle: { align: 'left' }, axisLabel: { formatter: (value: number) => value.toExponential(0) }, splitLine },
    series: [
      { name: 'Measured error', type: 'line', symbolSize: 9, data: convergence.map(row => [row.n, row.l2_error]) },
      { name: 'Second-order reference', type: 'line', symbol: 'none', lineStyle: { type: 'dashed', width: 1.5 }, data: convergence.map(row => [row.n, (convergence[0]?.l2_error ?? 0) * ((convergence[0]?.n ?? 32) / row.n) ** 2]) },
    ],
  }), [convergence])
  const sensitivityChart = useMemo(() => ({
    tooltip: { trigger: 'axis' }, legend: { bottom: 0, textStyle: { fontSize: 10, color: '#829079' } },
    grid: { left: 60, right: 25, top: 45, bottom: 80 },
    xAxis: { type: 'value', name: 'Time since release (min)', nameLocation: 'middle', nameGap: 30, splitLine },
    yAxis: { type: 'value', name: 'Exposure-aware route (rel. s)', nameTextStyle: { align: 'left' }, splitLine },
    series: [...new Set(sensitivity.map(row => row.kappa))].map(kappa => ({
      name: `κ = ${kappa} m²/s`, type: 'line', symbolSize: 8,
      data: sensitivity.filter(row => row.kappa === kappa).sort((a, b) => a.time_s - b.time_s).map(row => [row.time_s / 60, Number(row.weighted_exposure.toFixed(3))]),
    })),
  }), [sensitivity])
  const benchmarkChart = useMemo(() => ({
    tooltip: { trigger: 'axis' }, legend: { bottom: 0, textStyle: { fontSize: 10, color: '#829079' } },
    grid: { left: 55, right: 25, top: 45, bottom: 80 },
    xAxis: { type: 'category', data: [...new Set(benchmark.map(row => row.lambda_weight))].map(value => `λ ${value}`) },
    yAxis: { type: 'value', name: 'Search duration (ms)', nameTextStyle: { align: 'left' }, splitLine },
    series: ['astar', 'dijkstra'].map(algorithm => ({
      name: algorithm === 'astar' ? 'A* search' : 'Dijkstra', type: 'bar', barMaxWidth: 20,
      data: benchmark.filter(row => row.algorithm === algorithm).map(row => Number(row.search_ms.toFixed(3))),
      itemStyle: { borderRadius: [3, 3, 0, 0] },
    })),
  }), [benchmark])
  const comparisons = benchmark.filter(row => row.algorithm === 'astar').map(row => {
    const reference = benchmark.find(other => other.algorithm === 'dijkstra' && other.lambda_weight === row.lambda_weight)
    return { lambda: row.lambda_weight, astar: row, dijkstra: reference }
  })
  if (!convergence.length && !sensitivity.length && !benchmark.length) return null
  return <>
    <div className="report-grid">
      {convergence.length > 0 && <section className="report-card"><span className="section-eyebrow">ANALYTICAL SOLUTION CHECK</span><h3>Refine the grid. Measure the error.</h3><Chart option={convergenceChart} label="Log-log convergence plot comparing measured L2 error with second-order reference on 32, 64 and 128 square grids" /><p className="chart-caption">Observed orders: {convergence.filter(row => row.observed_order !== null).map(row => number(row.observed_order!, 4)).join(', ')}. Δt = 0.1h²; joint refinement approaches order 2. Explicit Euler remains first-order in time.</p></section>}
      {sensitivity.length > 0 && <section className="report-card"><span className="section-eyebrow">DIFFUSION & TIME SENSITIVITY</span><h3>The same streets, a changing field.</h3><Chart option={sensitivityChart} label="Exposure on optimized routes at 0, 10 and 30 minutes, comparing diffusion coefficients 5, 20 and 50" /><p className="chart-caption">Saved experiment at λ = {String((reports?.analysis?.parameters as Record<string, unknown> | undefined)?.lambda_weight ?? 5)} with fixed endpoints and source. Each point recomputes its optimum in that frozen field.</p></section>}
    </div>
    {sensitivity.length > 0 && <section className="report-card sweep-table"><span className="section-eyebrow">SENSITIVITY RESULTS</span><h3>Compare baseline and exposure-aware routes</h3><div className="table-scroll"><table><thead><tr><th>κ (m²/s)</th><th>Snapshot</th><th>Shortest exposure</th><th>Weighted exposure</th><th>Extra distance</th></tr></thead><tbody>{sensitivity.map(row => <tr key={`${row.kappa}-${row.time_s}`}><td>{row.kappa}</td><td>{number(row.time_s / 60)} min</td><td>{number(row.shortest_exposure, 3)} rel. s</td><td>{number(row.weighted_exposure, 3)} rel. s</td><td>{number(row.weighted_distance_m - row.shortest_distance_m)} m</td></tr>)}</tbody></table></div></section>}
    {benchmark.length > 0 && <div className="report-grid"><section className="report-card"><span className="section-eyebrow">ALGORITHM BENCHMARK</span><h3>Two searches, the same optimal cost.</h3><Chart option={benchmarkChart} label="A star and Dijkstra search runtime comparison across six route preference values" /><p className="chart-caption">Saved benchmark from the reported machine. A* uses straight-line walking time as its admissible lower bound. See the full analysis for NetworkX reference costs.</p></section><section className="report-card sweep-table"><span className="section-eyebrow">SEARCH EFFORT & AGREEMENT</span><h3>Vertices visited and cost difference</h3><div className="table-scroll"><table><thead><tr><th>λ</th><th>A* visits</th><th>Dijkstra visits</th><th>|Δ cost|</th></tr></thead><tbody>{comparisons.map(row => <tr key={row.lambda}><td>{row.lambda}</td><td>{number(row.astar.visited, 0)}</td><td>{number(row.dijkstra?.visited, 0)}</td><td>{row.dijkstra ? Math.abs(row.astar.cost - row.dijkstra.cost).toExponential(1) : '—'}</td></tr>)}</tbody></table></div></section></div>}
  </>
}
