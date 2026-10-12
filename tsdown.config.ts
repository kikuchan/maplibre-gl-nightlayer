import { defineConfig, type TsdownPlugin } from 'tsdown';
import { readFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';

function lutAssets(format: 'esm' | 'iife'): TsdownPlugin {
  return {
    name: 'nightlayer-lut-assets',
    resolveId(source, importer) {
      if (importer && source.endsWith('/lighting-lut.zip')) return `\0${resolve(dirname(importer), source)}.js`;
    },
    load(id) {
      if (!id.startsWith('\0') || !id.endsWith('/lighting-lut.zip.js')) return;
      const path = id.slice(1, -3),
        fileName = basename(path);
      this.emitFile({ type: 'asset', fileName, source: readFileSync(path) });
      const base = format === 'esm' ? 'import.meta.url' : '(document.currentScript?.src ?? document.baseURI)';
      return `export default new URL(${JSON.stringify(fileName)}, ${base}).href;`;
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
    plugins: [lutAssets('esm')],
    dts: true,
    deps: { neverBundle: true },
  },
  {
    entry: { nightlayer: 'src/main.ts' },
    format: 'iife',
    platform: 'browser',
    target: 'esnext',
    minify: true,
    plugins: [lutAssets('iife')],
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
