import { defineConfig } from 'vite';

export default defineConfig({
  base: '/maplibre-gl-nightlayer/',
  assetsInclude: ['**/*.zip'],
  resolve: {
    dedupe: ['maplibre-gl'],
  },
  build: {
    chunkSizeWarningLimit: Infinity,
  },
});
