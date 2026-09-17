import type { Feature, Geometry } from 'geojson'

export type Point = [number, number]
export type PointKind = 'start' | 'end' | 'source'
export type Config = {
  region: { geographic_bounds: [number, number, number, number]; [key: string]: unknown }
  defaults: Record<PointKind, Point>
  data_version: string
  numerics?: { methods: Record<string, string[]>; grids: number[]; native: { available: boolean; [key: string]: unknown } }
}
export type NumericsSettings = {
  model: 'diffusion' | 'advection_diffusion'
  method: 'explicit_euler' | 'backward_euler' | 'crank_nicolson' | 'advection_explicit' | 'imex_euler'
  backend: 'numpy' | 'cpp' | 'scipy' | 'numpy_scipy' | 'auto'
  nx: number; ny: number; dt: number | null; boundary: 'zero_flux' | 'periodic' | 'open'
  velocity: Point; startup: 'none' | 'rannacher'; duration_s: number; frame_interval_s: number
}
export type Simulation = {
  id: string; times: number[]; geographic_corners: Point[]; bounds: number[]
  mass_drift: number; elapsed_ms: number
  parameters?: Partial<NumericsSettings> & { source?: Point; kappa?: number; [key: string]: unknown }
  diagnostics?: Record<string, unknown>
}
export type Frame = { time_s: number; values: number[][]; min: number; max: number; difference?: number[][]; reference_time_s?: number; difference_min?: number; difference_max?: number }
export type Route = {
  geojson: Feature<Geometry>; distance_m: number; time_s: number
  exposure: number; cost: number; elapsed_ms: number; visited: number
}
export type Routes = { shortest: Route; weighted: Route; timings: { integration_ms: number; search_ms: number } }
export type ExperimentRow = { lambda_weight: number; distance_m: number; time_s: number; exposure: number; cost: number; elapsed_ms: number }
export type Experiment = { id: string; rows: ExperimentRow[] }
export type Reports = { data_quality: Record<string, unknown>; validation: Record<string, unknown> | null; analysis: Record<string, unknown> | null; numerics?: Record<string, unknown> | null }

export async function api<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method: body === undefined ? 'GET' : 'POST', signal,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => null)
    const detail = Array.isArray(error?.detail) ? error.detail.map((item: { msg?: string }) => item.msg ?? 'Invalid parameter').join(' ') : error?.detail
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status}). Please check the local server.`)
  }
  return response.json()
}
export const number = (value: number | undefined, digits = 1) => Number.isFinite(value) ? value!.toLocaleString('en-US', { maximumFractionDigits: digits }) : '—'
export const minutes = (seconds: number) => `${number(seconds / 60)} min`
