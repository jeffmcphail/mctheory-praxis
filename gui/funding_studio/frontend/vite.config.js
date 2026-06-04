import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Funding Studio frontend (Cycle 54b). Port 5174 so it coexists with
// mcb_studio (5173); proxies /api + /ws to the funding_studio backend (8002).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      '/api': 'http://localhost:8002',
      '/ws': {
        target: 'ws://localhost:8002',
        ws: true,
      },
    },
  },
})
