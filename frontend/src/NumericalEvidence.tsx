import { number } from './types'
import type { Reports, Simulation } from './types'
import { methodNames } from './numerics'

const scientific = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? Math.abs(value) < .001 && value !== 0 ? value.toExponential(2) : number(value, 5) : '—'

export function SimulationDiagnostics({ simulation }: { simulation: Simulation | null }) {
  if (!simulation) return null
  const data = simulation.diagnostics ?? {}
  const method = String(data.method ?? simulation.parameters?.method ?? 'explicit_euler')
  const backend = String(data.backend ?? simulation.parameters?.backend ?? 'numpy')
  const minimum = data.min_concentration ?? data.min
  const open = simulation.parameters?.boundary === 'open'
  const rows = [
    ['Minimum concentration', minimum], ['Maximum concentration', data.max_concentration ?? data.max],
    [open ? 'Relative mass change (open)' : 'Relative mass drift', data.relative_mass_drift ?? simulation.mass_drift],
    ['Relative flux balance error', data.relative_mass_balance_error], ['Max. linear residual', data.linear_residual_max],
    ['Internal time steps', data.internal_steps], ['Smallest actual step (s)', data.actual_dt_min_s], ['Largest actual step (s)', data.actual_dt_max_s],
  ]
  return <section className="simulation-diagnostics"><span className="section-eyebrow">LAST COMPLETED SOLVE</span><div className="solver-badge"><strong>{methodNames[method] ?? method}</strong><span>{backend}</span></div><p className="diagnostic-scope">Extrema and balance cover the initial state and saved output frames.</p><dl>{rows.map(([label, value]) => <div key={String(label)} className={label === 'Minimum concentration' && typeof value === 'number' && value < 0 ? 'negative-diagnostic' : ''}><dt>{String(label)}</dt><dd>{scientific(value)}</dd></div>)}</dl>{typeof minimum === 'number' && minimum < 0 && <p className="negative-note">The solver produced negative concentration. Values are preserved and shown in magenta. Routing rejects negative fields.</p>}<details><summary>Parameters and full diagnostics</summary><pre>{JSON.stringify({ parameters: simulation.parameters, diagnostics: data }, null, 2)}</pre></details></section>
}

const figures = [
  { file: 'temporal_convergence', title: 'Separate time error from space error.', label: 'TIME CONVERGENCE', text: 'Fixed spatial grid, refined time steps. Compare the observed first-order and second-order regimes against an independent reference.' },
  { file: 'spatial_convergence', title: 'Measure the spatial discretization.', label: 'SPACE CONVERGENCE', text: 'Compare diffusion and upwind transport errors with their respective references. Time error must be controlled before interpreting spatial order.' },
  { file: 'stability_positivity', title: 'Stable does not always mean positive.', label: 'STABILITY & POSITIVITY', text: 'Inspect large-step Crank–Nicolson oscillations and the effect of Rannacher startup. No clipping or mass renormalization hides the result.' },
  { file: 'transport_balance', title: 'Account for what leaves the domain.', label: 'TRANSPORT & BOUNDARIES', text: 'Periodic conservation and open-boundary mass balance test different claims. Outflow is included in the open-boundary accounting.' },
  { file: 'benchmark', title: 'The same method, two implementations.', label: 'NUMPY / C++', text: 'Compare identical Forward Euler workloads. Repeated samples, output policy, conversion and allocation costs are recorded in the downloadable data.' },
  { file: 'work_precision', title: 'Compare methods at the same error.', label: 'WORK–PRECISION', text: 'Implicit methods use fewer steps, but matrix setup and factorization have a cost. Cold and reused-factorization timings are distinguished.' },
  { file: 'routing_sensitivity', title: 'Does field error change the decision?', label: 'PDE ERROR → ROUTE COST', text: 'Re-evaluate each selected route in the same numerical reference field. A different route shape alone does not prove a worse decision.' },
]
export default function NumericalEvidence({ reports }: { reports: Reports | null }) {
  return <section className="numerical-evidence">
    <div className="science-heading"><div><span className="section-eyebrow">SCIENTIFIC COMPUTING / V2</span><h2>Methods, counterexamples, evidence.</h2><p>Time and space accuracy, conservative transport, and an independently implemented C++ kernel.</p></div><div className="download-links"><a href="/reports/numerics/methods.md" download>↓ Method notes</a><a href="/reports/numerics/conclusions.md" download>↓ Findings</a><a href="/reports/numerics/validation.json" download>↓ Validation</a></div></div>
    {reports?.numerics ? <div className="report-grid science-figures">{figures.map(figure => <section className="report-card scientific-figure" key={figure.file}><span className="section-eyebrow">{figure.label}</span><h3>{figure.title}</h3><a href={`/reports/numerics/${figure.file}.png`} target="_blank" rel="noreferrer"><img src={`/reports/numerics/${figure.file}.png`} alt={`${figure.label.toLowerCase()} experiment results; raw data is available below`} loading="lazy" /></a><p className="chart-caption">{figure.text}</p><div className="download-links"><a href={`/reports/numerics/${figure.file}.csv`} download>↓ Raw CSV</a><a href={`/reports/numerics/${figure.file}.png`} download>↓ Figure</a></div></section>)}</div> : <div className="science-empty"><strong>No saved numerical evidence is available yet.</strong><p>Run the reproducible numerical experiments from the README, then refresh reports. Current simulation diagnostics remain available in the laboratory.</p></div>}
  </section>
}
