import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000', '/data': 'http://127.0.0.1:8000' } },
  build: { rollupOptions: { output: { manualChunks: (id) => id.includes('maplibre') || id.includes('pmtiles') ? 'map' : id.includes('echarts') || id.includes('zrender') ? 'charts' : undefined } } },
})
