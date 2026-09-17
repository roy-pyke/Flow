import type { NumericsSettings } from './types'

export const initialNumerics: NumericsSettings = {
  model: 'diffusion', method: 'explicit_euler', backend: 'numpy', nx: 160, ny: 160,
  dt: null, boundary: 'zero_flux', velocity: [0, 0], startup: 'none',
  duration_s: 1800, frame_interval_s: 30,
}
export const methodNames: Record<string, string> = {
  explicit_euler: 'Forward Euler', backward_euler: 'Backward Euler', crank_nicolson: 'Crank–Nicolson',
  advection_explicit: 'Explicit upwind + diffusion', imex_euler: 'IMEX Euler',
}
