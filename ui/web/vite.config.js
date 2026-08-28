import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    // The dev server serves the interface; the API still belongs to the Node
    // process that owns the engine, so those paths are forwarded to it.
    proxy: {
      '/api': {
        target: 'http://localhost:5174',
        changeOrigin: true,
        // Server-sent events die behind a buffering proxy.
        configure: (proxy) => proxy.on('proxyRes', (res) => { res.headers['x-no-buffer'] = '1'; }),
      },
    },
  },
});
