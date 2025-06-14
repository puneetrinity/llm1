import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: 'dist',
    sourcemap: true,
    minify: 'esbuild',
    target: 'es2020',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          charts: ['recharts'],
          utils: ['axios']
        }
      }
    }
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    strictPort: true,
    proxy: {
      // Proxy API calls to backend
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false
      },
      '/v1': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false
      },
      '/health': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false
      },
      '/metrics': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false
      },
      '/auth': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false
      },
      '/admin': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false
      },
      '/docs': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false
      },
      '/openapi.json': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false
      },
      // WebSocket proxy
      '/ws': {
        target: 'ws://localhost:8001',
        ws: true,
        changeOrigin: true
      }
    }
  },
  preview: {
    host: '0.0.0.0',
    port: 3000,
    strictPort: true
  },
  define: {
    // Make sure environment variables are available
    'process.env': process.env
  }
})
