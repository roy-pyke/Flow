import type { Config, NumericsSettings } from './types'

import { methodNames } from './numerics'

const backends: Record<string, string[]> = {
  explicit_euler: ['numpy', 'cpp', 'auto'], backward_euler: ['scipy'], crank_nicolson: ['scipy'],
  advection_explicit: ['numpy'], imex_euler: ['numpy_scipy'],
}
const backendNames: Record<string, string> = { numpy: 'NumPy', cpp: 'C++', scipy: 'SciPy sparse', numpy_scipy: 'NumPy + SciPy', auto: 'Automatic (record actual)' }

type Props = { value: NumericsSettings; onChange: (value: NumericsSettings) => void; config: Config | null; disabled: boolean }
export default function NumericalControls({ value, onChange, config, disabled }: Props) {
  const update = (patch: Partial<NumericsSettings>) => onChange({ ...value, ...patch })
  const allowedBackends = config?.numerics?.methods?.[value.method] ?? backends[value.method]
  const transport = value.model === 'advection_diffusion'
  return <details className="numerics-controls" open>
    <summary><span>Numerical method</span><small>{methodNames[value.method]}</small></summary>
    <fieldset disabled={disabled}>
      <label>Model<select aria-label="Physical model" value={value.model} onChange={event => {
        const model = event.target.value as NumericsSettings['model']
        update(model === 'diffusion'
          ? { model, method: 'explicit_euler', backend: 'numpy', boundary: 'zero_flux', velocity: [0, 0], startup: 'none' }
          : { model, method: 'imex_euler', backend: 'numpy_scipy', boundary: 'open', velocity: [1, 0], startup: 'none' })
      }}><option value="diffusion">Pure diffusion</option><option value="advection_diffusion">Advection–diffusion</option></select></label>
      <label>Time integrator<select aria-label="Time integrator" value={value.method} onChange={event => {
        const method = event.target.value as NumericsSettings['method']
        update({ method, backend: backends[method][0] as NumericsSettings['backend'], startup: 'none' })
      }}>{(transport ? ['advection_explicit', 'imex_euler'] : ['explicit_euler', 'backward_euler', 'crank_nicolson']).map(method => <option key={method} value={method}>{methodNames[method]}</option>)}</select></label>
      <label>Implementation<select aria-label="Numerical backend" value={value.backend} onChange={event => update({ backend: event.target.value as NumericsSettings['backend'], ...(event.target.value === 'cpp' ? { boundary: 'zero_flux' } : {}) })}>{allowedBackends.map(backend => <option key={backend} value={backend} disabled={backend === 'cpp' && config?.numerics?.native?.available === false}>{backendNames[backend] ?? backend}{backend === 'cpp' && config?.numerics?.native?.available === false ? ' · not installed' : ''}</option>)}</select></label>
      <div className="numeric-pair"><label>Cells along x<select aria-label="Grid cells x" value={value.nx} onChange={event => update({ nx: Number(event.target.value) })}>{(config?.numerics?.grids ?? [80, 160, 320]).map(n => <option key={n} value={n}>{n}</option>)}</select></label><label>Cells along y<select aria-label="Grid cells y" value={value.ny} onChange={event => update({ ny: Number(event.target.value) })}>{(config?.numerics?.grids ?? [80, 160, 320]).map(n => <option key={n} value={n}>{n}</option>)}</select></label></div>
      <label>Boundary<select aria-label="Boundary condition" value={value.boundary} onChange={event => update({ boundary: event.target.value as NumericsSettings['boundary'] })}>{transport ? <option value="open">Open · zero inflow</option> : <option value="zero_flux">Closed · zero flux</option>}<option value="periodic" disabled={value.backend === 'cpp'}>Periodic · analytical study</option></select></label>
      {transport && <div className="numeric-pair"><label>Wind x (m/s)<input aria-label="Wind x" type="number" min="-10" max="10" step="0.1" value={value.velocity[0]} onChange={event => update({ velocity: [Number(event.target.value), value.velocity[1]] })} /></label><label>Wind y (m/s)<input aria-label="Wind y" type="number" min="-10" max="10" step="0.1" value={value.velocity[1]} onChange={event => update({ velocity: [value.velocity[0], Number(event.target.value)] })} /></label></div>}
      <div className="step-control"><label className="checkbox-label"><input type="checkbox" aria-label="Automatic time step" checked={value.dt === null} onChange={event => update({ dt: event.target.checked ? null : 5 })} />Automatic time step</label>{value.dt !== null && <label>Requested Δt (s)<input type="number" aria-label="Requested time step" min="0.001" max="1800" step="any" value={value.dt} onChange={event => update({ dt: Number(event.target.value) })} /></label>}</div>
      {value.method === 'crank_nicolson' && <label className="checkbox-label"><input aria-label="Rannacher startup" type="checkbox" checked={value.startup === 'rannacher'} onChange={event => update({ startup: event.target.checked ? 'rannacher' : 'none' })} />Rannacher startup</label>}
      <div className="numeric-pair"><label>Duration (s)<input aria-label="Simulation duration" type="number" min="30" max="1800" step="30" value={value.duration_s} onChange={event => update({ duration_s: Number(event.target.value) })} /></label><label>Output every (s)<input aria-label="Frame interval" type="number" min="1" max="1800" step="1" value={value.frame_interval_s} onChange={event => update({ frame_interval_s: Number(event.target.value) })} /></label></div>
    </fieldset>
    <p className="numerics-hint">{value.method === 'crank_nicolson' ? 'Second order in time, but large steps may oscillate. Startup damps sharp initial transients; it does not guarantee positivity.' : value.method === 'backward_euler' ? 'First order in time. Implicit stability does not guarantee an accurate large time step.' : value.method === 'imex_euler' ? 'First order: implicit diffusion, explicit upwind transport. The advection CFL limit still applies.' : 'First order in time. An explicit CFL limit applies; invalid requested steps are rejected.'}</p>
    <p className="numerics-hint">{value.boundary === 'open' ? 'Open boundaries release outward mass. Check the flux balance, not mass drift alone.' : value.boundary === 'periodic' ? 'The field wraps across opposite edges. This is a verification boundary, not a city model.' : 'Closed boundaries conserve mass. The field cannot leave the study area.'}</p>
  </details>
}
