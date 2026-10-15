import { defineConfig, type TsdownPlugin } from 'tsdown';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

function inlineLUT(): TsdownPlugin {
  return {
    name: 'nightlayer-inline-lut',
    resolveId(source, importer) {
      if (importer && source.endsWith('/lighting-lut.zip?inline'))
        return `\0${resolve(dirname(importer), source.slice(0, -7))}.js`;
    },
    load(id) {
      if (!id.startsWith('\0') || !id.endsWith('/lighting-lut.zip.js')) return;
      const encoded = readFileSync(id.slice(1, -3)).toString('base64');
      return `export default "data:application/zip;base64,${encoded}";`;
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
    plugins: [inlineLUT()],
    dts: true,
    deps: { neverBundle: true },
  },
  {
    entry: { nightlayer: 'src/main.ts' },
    format: 'iife',
    platform: 'browser',
    target: 'esnext',
    minify: true,
    plugins: [inlineLUT()],
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
