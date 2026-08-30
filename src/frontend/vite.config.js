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
        // FastAPI binds to IPv4 in local development; avoid Node trying ::1 first
        // and reporting an AggregateError while the backend is restarting.
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // Proxy the WebSocket upgrade too, not just plain HTTP.
        //
        // Two routes depend on this and both fail silently without it: the live video socket
        // (`/api/v1/cameras/{id}/video`, the grid's H.264 feed) and the manager dashboard's
        // `/api/v1/manager/live`. Without `ws`, Vite answers the upgrade itself rather than
        // forwarding it — the browser sees a socket that opens and then closes with no
        // frames, so the grid tile stays black and nothing in the console explains why. The
        // client's MJPEG fallback only engages after the socket has already failed, and in
        // dev that failure is this proxy rather than a real encoder problem.
        //
        // Production is unaffected: nginx already forwards `Upgrade`/`Connection` for
        // `location /api/` (see nginx.conf). This only closes the dev/prod gap.
        ws: true,
      },
    },
  },
})
