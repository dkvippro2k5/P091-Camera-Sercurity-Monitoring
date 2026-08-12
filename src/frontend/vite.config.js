import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Everything under /api is proxied to FastAPI so the browser sees one origin. That means
// no CORS preflight on the MJPEG and SSE streams, which are the two things most likely to
// break under a cross-origin setup.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
