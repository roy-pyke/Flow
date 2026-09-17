import { useEffect, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import { Protocol } from 'pmtiles'
import mapWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import type { FeatureCollection } from 'geojson'
import type { Config, Frame, Point, PointKind, Routes, Simulation } from './types'
import 'maplibre-gl/dist/maplibre-gl.css'

const empty: FeatureCollection = { type: 'FeatureCollection', features: [] }
maplibregl.setWorkerUrl(mapWorkerUrl)
const protocol = new Protocol()
maplibregl.addProtocol('pmtiles', protocol.tile)
const labels: [string, Point][] = [
  ['MISSION DISTRICT', [-122.416, 37.759]], ['THE CASTRO', [-122.435, 37.7605]],
  ['POTRERO HILL', [-122.4005, 37.758]], ['NOE VALLEY', [-122.429, 37.747]],
  ['SOMA', [-122.41, 37.776]], ['BERNAL HEIGHTS', [-122.416, 37.741]],
]
function heatImage(frame: Frame): string {
  const height = frame.values.length
  const width = frame.values[0]?.length ?? 0
  const canvas = document.createElement('canvas')
  canvas.width = width; canvas.height = height
  const context = canvas.getContext('2d')!
  const pixels = context.createImageData(width, height)
  // The API uses south-to-north rows. Image coordinates start in the northwest.
  // All frames use the same absolute [0, 1] concentration color scale.
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const c = Math.min(1, Math.max(0, frame.values[height - 1 - y][x]))
    const i = (y * width + x) * 4
    pixels.data[i] = Math.round(252 - c * 19)
    pixels.data[i + 1] = Math.round(208 - c * 107)
    pixels.data[i + 2] = Math.round(81 - c * 13)
    pixels.data[i + 3] = Math.round(Math.min(0.86, c * 2.5) * 255)
  }
  context.putImageData(pixels, 0, 0)
  return canvas.toDataURL()
}

type Props = {
  config: Config | null; points: Record<PointKind, Point>; mode: PointKind
  onPick: (point: Point) => void; simulation: Simulation | null; frame: Frame | null
  routes: Routes | null; showHeat: boolean; visible: boolean
}
export default function MapView({ config, points, mode, onPick, simulation, frame, routes, showHeat, visible }: Props) {
  const container = useRef<HTMLDivElement>(null)
  const map = useRef<maplibregl.Map | null>(null)
  const markers = useRef<Partial<Record<PointKind, maplibregl.Marker>>>({})
  const click = useRef(onPick)
  const [ready, setReady] = useState(false)
  const [mapError, setMapError] = useState('')
  useEffect(() => { click.current = onPick }, [onPick])
  useEffect(() => {
    if (!container.current) return
    const instance = new maplibregl.Map({
      container: container.current,
      center: [-122.4149, 37.7599], zoom: 13.25,
      attributionControl: false,
      style: {
        version: 8,
        sources: { roads: { type: 'vector', url: `pmtiles://${window.location.origin}/data/demo/roads.pmtiles`, attribution: '<a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">© OpenStreetMap contributors</a> · ODbL' } },
        layers: [
          { id: 'background', type: 'background', paint: { 'background-color': '#e9ece4' } },
          { id: 'road-outline', type: 'line', source: 'roads', 'source-layer': 'roads', paint: { 'line-color': '#c7cdc3', 'line-width': ['interpolate', ['linear'], ['zoom'], 12, 2, 16, 12], 'line-opacity': .85 } },
          { id: 'roads', type: 'line', source: 'roads', 'source-layer': 'roads', paint: { 'line-color': '#fafbf6', 'line-width': ['interpolate', ['linear'], ['zoom'], 12, 1, 16, 9] } },
          { id: 'road-labels', type: 'symbol', source: 'roads', 'source-layer': 'roads', minzoom: 13, filter: ['!=', ['get', 'name'], ''], layout: { 'symbol-placement': 'line', 'text-field': ['get', 'name'], 'text-font': ['Arial', 'sans-serif'], 'text-size': 10, 'symbol-spacing': 260, 'text-max-angle': 30 }, paint: { 'text-color': '#75816e', 'text-halo-color': '#fafbf6', 'text-halo-width': 1.5 } },
        ],
      },
    })
    map.current = instance
    instance.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    instance.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left')
    instance.addControl(new maplibregl.AttributionControl({ compact: false }), 'bottom-right')
    instance.on('click', event => click.current([event.lngLat.lng, event.lngLat.lat]))
    instance.on('error', event => {
      setMapError(`Map could not finish loading: ${event.error?.message ?? 'unknown error'}. Reload after checking the local map resources.`)
    })
    instance.on('load', () => {
      for (const name of ['shortest', 'weighted']) {
        instance.addSource(name, { type: 'geojson', data: empty })
        instance.addLayer({ id: `${name}-halo`, type: 'line', source: name, paint: { 'line-color': '#ffffff', 'line-width': name === 'shortest' ? 7 : 8, 'line-opacity': .9 }, layout: { 'line-cap': 'round', 'line-join': 'round' } })
        instance.addLayer({ id: name, type: 'line', source: name, paint: { 'line-color': name === 'shortest' ? '#e5a445' : '#187f77', 'line-width': name === 'shortest' ? 3 : 4, ...(name === 'shortest' ? { 'line-dasharray': [2, 1.1] } : {}) }, layout: { 'line-cap': 'round', 'line-join': 'round' } })
      }
      labels.forEach(([name, point]) => {
        const element = document.createElement('div'); element.className = 'neighborhood-label'; element.textContent = name
        new maplibregl.Marker({ element }).setLngLat(point).addTo(instance)
      })
      setReady(true)
    })
    const observer = new ResizeObserver(() => instance.resize())
    observer.observe(container.current)
    return () => { observer.disconnect(); instance.remove(); map.current = null; markers.current = {}; setReady(false) }
  }, [])
  useEffect(() => {
    if (!map.current || !ready || !config) return
    const b = config.region.geographic_bounds
    if (b) map.current.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 35, duration: 0 })
  }, [config, ready])
  useEffect(() => {
    if (!map.current || !ready) return
    for (const kind of ['start', 'end', 'source'] as const) {
      if (!markers.current[kind]) {
        const element = document.createElement('div'); element.className = `map-pin pin-${kind}`
        element.textContent = kind === 'start' ? 'A' : kind === 'end' ? 'B' : '✳'
        element.title = kind === 'source' ? 'Gaussian release source' : kind === 'start' ? 'Start point' : 'End point'
        markers.current[kind] = new maplibregl.Marker({ element }).setLngLat(points[kind]).addTo(map.current)
      }
      markers.current[kind]!.setLngLat(points[kind])
    }
  }, [points, ready])
  useEffect(() => {
    if (!map.current || !ready || !frame || !simulation) return
    const url = heatImage(frame)
    const coordinates = simulation.geographic_corners as [Point, Point, Point, Point]
    const source = map.current.getSource('diffusion') as maplibregl.ImageSource | undefined
    if (source) source.updateImage({ url, coordinates })
    else {
      map.current.addSource('diffusion', { type: 'image', url, coordinates })
      map.current.addLayer({ id: 'diffusion', type: 'raster', source: 'diffusion', paint: { 'raster-opacity': 0.8, 'raster-fade-duration': 0, 'raster-resampling': 'linear' } }, 'road-labels')
    }
    map.current.setLayoutProperty('diffusion', 'visibility', showHeat ? 'visible' : 'none')
  }, [frame, simulation, ready, showHeat])
  useEffect(() => {
    if (!map.current || !ready) return
    for (const kind of ['shortest', 'weighted'] as const) {
      const source = map.current.getSource(kind) as maplibregl.GeoJSONSource
      source.setData(routes?.[kind].geojson ?? empty)
    }
  }, [routes, ready])
  useEffect(() => { if (visible) map.current?.resize() }, [visible])
  return <div className={`map-view mode-${mode}`}>
    <div ref={container} className="map-canvas" aria-label="Interactive offline Mission District map. Click to set the selected point." />
    <div className="map-location"><span className="crosshair">⌖</span><div><strong>Mission, San Francisco</strong><span>4 × 4 km study area · EPSG:32610</span></div><span className="local-badge">LOCAL</span></div>
    <div className="map-tip"><span className={`point-dot ${mode}`} /> Click the map to place {mode === 'start' ? 'start A' : mode === 'end' ? 'destination B' : 'the release source'}</div>
    <div className="map-legend"><span><i className="line-key shortest" />Shortest</span><span><i className="line-key weighted" />Exposure-aware</span></div>
    {mapError && <div className="map-error" role="alert">{mapError}</div>}
  </div>
}
