import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  base: '/maplibre-gl-nightlayer/',
  assetsInclude: ['**/*.zip', '**/*.bin.gz'],
  server: {
    // The example imports the library sources and versioned LUT from ../src.
    fs: { allow: [fileURLToPath(new URL('..', import.meta.url))] },
    watch: {
      // Acquisition writes here continuously; the gallery polls JSON itself.
      ignored: ['**/public/globe-size/coordinate-grid/five-d-reference/**/*.{json,png,log,jsonl,tmp,lock}'],
    },
  },
  resolve: {
    dedupe: ['maplibre-gl'],
  },
  build: {
    chunkSizeWarningLimit: Infinity,
  },
});
