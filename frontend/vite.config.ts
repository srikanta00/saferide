import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev only: the Vite server forwards API, video and snapshot requests to the Python server.
const api = process.env.SAFERIDE_API ?? 'http://127.0.0.1:8765'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': api,
      '/stream.mjpg': api,
      '/snapshots': api,
    },
  },
})
