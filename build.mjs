import esbuild from 'esbuild';
import { globalExternals } from "@fal-works/esbuild-plugin-global-externals";

const globals = {
  'maplibre-gl': {
    varName: 'maplibregl',
    namedExports: ['createTileMesh', 'MercatorCoordinate'],
    defaultExport: false,
  }
};

esbuild.build({
  entryPoints: ["src/main.ts"],
  format: 'iife',
  outfile: "dist/nightlayer.min.js",
  minify: true,
  bundle: true,
  plugins: [globalExternals(globals)],
  globalName: 'nightLayer',
});

esbuild.build({
  entryPoints: ["src/main.ts"],
  format: 'esm',
  outfile: "dist/main.js",
  minify: true,
});
