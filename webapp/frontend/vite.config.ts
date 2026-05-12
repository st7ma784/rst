import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: process.env.BACKEND_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: (process.env.BACKEND_URL || 'http://localhost:8000').replace(/^http/, 'ws'),
        ws: true,
      }
    }
  }
})
