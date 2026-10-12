import { defineConfig } from 'vite';

export default defineConfig({
  base: '/maplibre-gl-nightlayer/',
  assetsInclude: ['**/*.zip'],
  build: {
    chunkSizeWarningLimit: Infinity,
  },
});
