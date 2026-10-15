import { defineConfig } from 'vite';

export default defineConfig({
  base: '/maplibre-gl-nightlayer/',
  assetsInclude: ['**/*.zip', '**/*.bin.gz'],
  resolve: {
    dedupe: ['maplibre-gl'],
  },
  build: {
    chunkSizeWarningLimit: Infinity,
  },
});
