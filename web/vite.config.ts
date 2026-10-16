import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// server.js serves web/dist under /app, so every asset URL starts with /app/.
export default defineConfig({
  base: '/app/',
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
    // ECharts is most of the bundle and changes rarely: keep it in its own cached chunk.
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      output: {
        manualChunks: {
          echarts: ['echarts/core', 'echarts/charts', 'echarts/components', 'echarts/renderers'],
        },
      },
    },
  },
});
