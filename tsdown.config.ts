import { defineConfig, type TsdownPlugin } from 'tsdown';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

function inlineAssets(): TsdownPlugin {
  return {
    name: 'nightlayer-inline-assets',
    resolveId(source, importer) {
      if (importer && /\/(lighting-lut\.zip|starfield\.webp|star-catalog\.bin\.gz)\?inline$/.test(source))
        return `\0${resolve(dirname(importer), source.slice(0, -7))}.js`;
    },
    load(id) {
      if (!id.startsWith('\0') || !/\/(lighting-lut\.zip|starfield\.webp|star-catalog\.bin\.gz)\.js$/.test(id)) return;
      const encoded = readFileSync(id.slice(1, -3)).toString('base64');
      const mime = id.endsWith('.webp.js')
        ? 'image/webp'
        : id.endsWith('.bin.gz.js')
          ? 'application/gzip'
          : 'application/zip';
      return `export default "data:${mime};base64,${encoded}";`;
    },
  };
}

export default defineConfig([
  {
    entry: { main: 'src/main.ts' },
    format: 'esm',
    platform: 'browser',
    target: 'esnext',
    minify: true,
    plugins: [inlineAssets()],
    dts: true,
    deps: { neverBundle: true },
  },
  {
    entry: { nightlayer: 'src/main.ts' },
    format: 'iife',
    platform: 'browser',
    target: 'esnext',
    minify: true,
    plugins: [inlineAssets()],
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
