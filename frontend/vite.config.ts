import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

declare const process: any;

const API_TARGET = (typeof process !== 'undefined' && process?.env?.VITE_API_BASE_URL) || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: API_TARGET, changeOrigin: true },
      '/health': { target: API_TARGET, changeOrigin: true },
    },
  },
  build: { outDir: 'dist', sourcemap: false },
})
