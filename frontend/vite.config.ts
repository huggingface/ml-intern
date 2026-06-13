import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const parsePort = (value: string | undefined, fallback: number) => {
  const parsed = Number.parseInt(value ?? '', 10)
  return Number.isFinite(parsed) ? parsed : fallback
}

export default defineConfig(() => {
  const backendProxyTarget =
    process.env.VITE_BACKEND_PROXY_TARGET ?? 'http://localhost:7860'
  const devServerPort = parsePort(process.env.VITE_DEV_SERVER_PORT, 5173)

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: devServerPort,
      proxy: {
        '/api': {
          target: backendProxyTarget,
          changeOrigin: true,
          ws: true, // Proxy WebSocket connections (/api/ws/...)
        },
        '/auth': {
          target: backendProxyTarget,
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: false,
    },
  }
})
