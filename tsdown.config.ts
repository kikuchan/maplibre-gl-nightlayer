import { defineConfig } from 'tsdown';

export default defineConfig([
  {
    entry: { main: 'src/main.ts' },
    format: 'esm',
    platform: 'browser',
    target: 'esnext',
    minify: true,
    dts: true,
    deps: { neverBundle: true },
  },
  {
    entry: { nightlayer: 'src/main.ts' },
    format: 'iife',
    platform: 'browser',
    target: 'esnext',
    minify: true,
    dts: false,
    globalName: 'nightLayer',
    deps: {
      neverBundle: ['maplibre-gl'],
    },
    outputOptions: {
      entryFileNames: 'nightlayer.min.js',
      exports: 'named',
      globals: { 'maplibre-gl': 'maplibregl' },
    },
  },
]);
