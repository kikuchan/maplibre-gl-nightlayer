/** IEEE binary16, including signed zero and subnormals. */
export function decodeFloat16(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 31,
    fraction = bits & 1023;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return exponent === 0 ? sign * fraction * 2 ** -24 : sign * (1024 + fraction) * 2 ** (exponent - 25);
}

const crcTable = Uint32Array.from({ length: 256 }, (_, value) => {
  for (let bit = 0; bit < 8; bit++) value = (value >>> 1) ^ (value & 1 ? 0xedb88320 : 0);
  return value >>> 0;
});
function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const value of bytes) crc = (crc >>> 8) ^ crcTable[(crc ^ value) & 255];
  return (crc ^ 0xffffffff) >>> 0;
}

/** Strict local-header ZIP decoder for atmosphere archives. */
export async function decodeLUTZip(bytes: Uint8Array<ArrayBuffer>): Promise<Map<string, Uint8Array<ArrayBuffer>>> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const text = new TextDecoder(),
    entries = new Map<string, Uint8Array<ArrayBuffer>>();
  let offset = 0;
  while (offset + 4 <= bytes.length && view.getUint32(offset, true) === 0x04034b50) {
    if (offset + 30 > bytes.length) throw new Error('Truncated LUT ZIP header');
    const flags = view.getUint16(offset + 6, true),
      method = view.getUint16(offset + 8, true);
    const checksum = view.getUint32(offset + 14, true),
      compressedSize = view.getUint32(offset + 18, true),
      size = view.getUint32(offset + 22, true);
    const nameLength = view.getUint16(offset + 26, true),
      extraLength = view.getUint16(offset + 28, true);
    const start = offset + 30 + nameLength + extraLength,
      end = start + compressedSize;
    if ((flags & ~0x800) !== 0 || method !== 8 || size === 0xffffffff || end > bytes.length)
      throw new Error('Unsupported LUT ZIP entry');
    const name = text.decode(bytes.subarray(offset + 30, offset + 30 + nameLength));
    if ((offset === 0 && name !== 'metadata.json') || entries.has(name)) throw new Error('Invalid LUT ZIP entry order');
    const value = new Uint8Array(
      await new Response(
        new Blob([bytes.subarray(start, end)]).stream().pipeThrough(new DecompressionStream('deflate-raw')),
      ).arrayBuffer(),
    );
    if (value.length !== size || crc32(value) !== checksum) throw new Error('Corrupt LUT ZIP entry');
    entries.set(name, value);
    offset = end;
  }
  if (offset + 4 > bytes.length || view.getUint32(offset, true) !== 0x02014b50) throw new Error('Truncated LUT ZIP');
  return entries;
}

/** Observer grid and direct RGB transmittance layout. */
type ObserverLayout = {
  version: 8;
  counts: [16, 64, 8, 64];
  axisNodes: number[][];
  axisOrder: ['p', 'nu', 'phi', 'beta'];
  encoding: 'shared-shape7-logA-split-float16';
  axisCDF: number[][];
  angleRangeDegrees: [-90, 100.0858569625031];
  angleAsinhScaleDegrees: 0.5;
  transmission: {
    encoding: 'linear-rgb-transmission-rgba32f';
    size: [256, 256];
    domain: [number, number];
    columnScale: [number, number];
    mieExtinction: number;
  };
  radianceScale: number;
};

export type LUTMetadata = {
  format: 'atmosphere-lut-v1';
  palette: {
    encoding: 'stable-r-theta-linear-rgb-rgba16f';
    color: {
      size: [256, 64, 2];
      uvScale: [[number, number], [number, number]];
      uvBias: [[number, number], [number, number]];
      minRadius: [number, number];
    };
  };
  observer: ObserverLayout;
  scattering: {
    encoding: 'pca-tensor-bspline-f16-v1' | 'pca-tensor-bspline-mixed-v2';
    p0Height: 0.002;
    /** Controls of the tensor B-spline: [p, beta, nu, s, phi]. */
    counts: [16, 64, 32, 16, 8];
    /** Per-frame dense grid [nu, s, phi], and the space-entry grid [nu, phi, beta]. */
    render: [128, 256, 32];
    /** Dense nu is asinh(relative elevation / renderNuScale degrees), finer than the chart. */
    renderNuScale: number;
    boundaryRender: [64, 16, 64];
    skyCounts: [32, 8];
    skyRender: [128, 32];
    /** PCA ranks of [L_R, theta_R, L_M, theta_M]. */
    ranks: [number, number, number, number];
    /** Clamped cubic knot vectors of the smooth physical chart. */
    knots: { p: number[]; beta: number[]; nu: number[]; s: number[]; phi: number[] };
    heightRange: [number, number];
    epsilon: { minimum: number; decades: number; center: number; width: number };
  };
  refraction?: {
    version: 2;
    encoding: 'fixed-map-celestial-v2';
    size: [128, 64, 3];
    refractivity: number;
    scaleHeightKm: number;
  };
  solarRadiance: [number, number, number];
  solarAngularRadius: number;
  mieG: number;
  skyReference: number;
  irradianceHeightCount: number;
  irradianceHeightMax: number;
  densityIntegrals: { width: 128; height: 128; version: 2; encoding: 'path-normalized-log-tail-rational-rgba32f' };
};
/** One field: C[(p,beta),(nu,s,phi)] = mean + A[(p,beta),k] B[k,(nu,s,phi)]. */
export type ScatteringFactors = {
  mean: Float32Array<ArrayBuffer>;
  a: Float32Array<ArrayBuffer>;
  b: Float32Array<ArrayBuffer>;
  rank: number;
};
export type LUTData = {
  /** [L_R, theta_R, L_M, theta_M] low-rank finite-path factors. */
  scattering: ScatteringFactors[];
  skyNodes: Float32Array<ArrayBuffer>;
  palette: Uint16Array<ArrayBuffer>;
  transmission: Float32Array<ArrayBuffer>;
  metadata: LUTMetadata;
  irradiance: Float32Array<ArrayBuffer>;
  optical: Float32Array<ArrayBuffer>;
  refraction?: Float32Array<ArrayBuffer>;
};

let data: Promise<LUTData> | undefined;
const DEFAULT_LUT_URL = new URL('./atmosphere-lut.zip', import.meta.url).href;

/** Decode the atmosphere archive; DirectionTransport owns its GPU representation. */
export function loadTransportLUT(source?: string): Promise<LUTData> {
  if (source !== undefined) return readLUT(source);
  if (data) return data;
  const pending = readLUT();
  data = pending;
  void pending.catch(() => {
    data = undefined;
  });
  return pending;
}

export async function readLUT(source: string = DEFAULT_LUT_URL): Promise<LUTData> {
  const response = await fetch(source);
  if (!response.ok) throw new Error(`Failed to load NightLayer LUT (${response.status}): ${source}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const entries = await decodeLUTZip(bytes);
  const metadata: LUTMetadata = JSON.parse(new TextDecoder().decode(entries.get('metadata.json')!));
  if (metadata?.format !== 'atmosphere-lut-v1') throw new Error('Only atmosphere-lut-v1 LUT archives are supported');
  const scattering = metadata.scattering;
  const mixed = scattering?.encoding === 'pca-tensor-bspline-mixed-v2';
  const names = [
    'metadata.json',
    'scattering-pca.f16',
    ...(mixed ? ['scattering-mie-pair.f32'] : []),
    'sky-scattering.f16',
    'palette.rgba16f',
    'irradiance.f32',
    'density-integrals.rgba32f',
    'transmittance.rgba32f',
    ...(metadata.refraction ? ['refraction.rgba32f'] : []),
  ];
  if (entries.size !== names.length || names.some((name) => !entries.has(name)))
    throw new Error('Unexpected atmosphere LUT ZIP entries');
  const color = metadata.palette?.color;
  if (
    metadata.palette?.encoding !== 'stable-r-theta-linear-rgb-rgba16f' ||
    !color ||
    JSON.stringify(color.size) !== '[256,64,2]' ||
    [color.uvScale, color.uvBias].some(
      (rows) =>
        !Array.isArray(rows) ||
        rows.length !== 2 ||
        rows.some((row) => !Array.isArray(row) || row.length !== 2 || row.some((v) => !Number.isFinite(v))),
    ) ||
    color.uvScale.some((row) => row.some((v) => v <= 0)) ||
    !Array.isArray(color.minRadius) ||
    color.minRadius.length !== 2 ||
    color.minRadius.some(
      (v) => !Number.isFinite(v) || v <= 0 || !Number.isFinite(Math.fround(v)) || Math.fround(v) <= 0,
    )
  )
    throw new Error('Invalid RGB palette layout');
  const layout = metadata.observer;
  if (
    !layout ||
    layout.version !== 8 ||
    layout.encoding !== 'shared-shape7-logA-split-float16' ||
    JSON.stringify(layout.axisOrder) !== '["p","nu","phi","beta"]' ||
    JSON.stringify(layout.counts) !== '[16,64,8,64]' ||
    !Array.isArray(layout.axisNodes) ||
    layout.axisNodes.length !== 4 ||
    layout.axisNodes.some(
      (row, axis) =>
        !Array.isArray(row) ||
        row.length !== layout.counts[axis] ||
        row[0] !== 0 ||
        row[row.length - 1] !== 1 ||
        row.some((v, i) => !Number.isFinite(v) || (i > 0 && Math.fround(v) <= Math.fround(row[i - 1]))),
    ) ||
    !Array.isArray(layout.axisCDF) ||
    layout.axisCDF.length !== 4 ||
    layout.axisCDF.some(
      (row) =>
        !Array.isArray(row) ||
        row.length !== 65 ||
        row[0] !== 0 ||
        row[64] !== 1 ||
        row.some((v, i) => !Number.isFinite(v) || (i > 0 && Math.fround(v) <= Math.fround(row[i - 1]))),
    ) ||
    !Number.isFinite(layout.radianceScale) ||
    layout.radianceScale <= 0 ||
    JSON.stringify(layout.angleRangeDegrees) !== '[-90,100.0858569625031]' ||
    layout.angleAsinhScaleDegrees !== 0.5
  )
    throw new Error('Invalid shared7 suffix layout');
  const transmissionLayout = layout.transmission;
  if (
    !transmissionLayout ||
    transmissionLayout.encoding !== 'linear-rgb-transmission-rgba32f' ||
    JSON.stringify(transmissionLayout.size) !== '[256,256]' ||
    !Array.isArray(transmissionLayout.domain) ||
    transmissionLayout.domain.length !== 2 ||
    transmissionLayout.domain.some((v) => !Number.isFinite(v) || v <= 0) ||
    !Array.isArray(transmissionLayout.columnScale) ||
    transmissionLayout.columnScale.length !== 2 ||
    transmissionLayout.columnScale.some((v) => !Number.isFinite(v) || v <= 0) ||
    !Number.isFinite(transmissionLayout.mieExtinction) ||
    transmissionLayout.mieExtinction < 0
  )
    throw new Error('Invalid RGB transmission layout');
  if (
    !scattering ||
    (!mixed && scattering.encoding !== 'pca-tensor-bspline-f16-v1') ||
    scattering.p0Height !== 0.002 ||
    JSON.stringify(scattering.counts) !== '[16,64,32,16,8]' ||
    JSON.stringify(scattering.render) !== '[128,256,32]' ||
    !(scattering.renderNuScale > 0 && scattering.renderNuScale <= 0.5) ||
    JSON.stringify(scattering.boundaryRender) !== '[64,16,64]' ||
    JSON.stringify(scattering.skyCounts) !== '[32,8]' ||
    JSON.stringify(scattering.skyRender) !== '[128,32]' ||
    !Array.isArray(scattering.ranks) ||
    scattering.ranks.length !== 4 ||
    scattering.ranks.some((k) => !Number.isSafeInteger(k) || k < 1 || k > 256) ||
    !scattering.knots ||
    (['p', 'beta', 'nu', 's', 'phi'] as const).some((axis, i) => {
      const knots = scattering.knots[axis];
      // phi merges its end pairs: 8 controls from 10 basis functions.
      const count = [16, 64, 32, 16, 10][i] + 4;
      return (
        !Array.isArray(knots) ||
        knots.length !== count ||
        knots.slice(0, 4).some((v) => v !== 0) ||
        knots.slice(-4).some((v) => v !== 1) ||
        knots.some((v, j) => !Number.isFinite(v) || (j > 0 && v < knots[j - 1]))
      );
    }) ||
    !Array.isArray(scattering.heightRange) ||
    scattering.heightRange.length !== 2 ||
    !(scattering.heightRange[0] > 0 && scattering.heightRange[1] > scattering.heightRange[0]) ||
    !scattering.epsilon ||
    [scattering.epsilon.minimum, scattering.epsilon.decades, scattering.epsilon.center, scattering.epsilon.width].some(
      (v) => !Number.isFinite(v),
    ) ||
    !(scattering.epsilon.minimum > 0 && scattering.epsilon.width > 0)
  )
    throw new Error('Invalid scattering layout');
  if (
    !Array.isArray(metadata.solarRadiance) ||
    metadata.solarRadiance.length !== 3 ||
    metadata.solarRadiance.some((v) => !Number.isFinite(v) || v < 0) ||
    !Number.isFinite(metadata.solarAngularRadius) ||
    metadata.solarAngularRadius <= 0 ||
    metadata.solarAngularRadius >= Math.PI / 2 ||
    !Number.isFinite(metadata.mieG) ||
    Math.abs(metadata.mieG) >= 1 ||
    !Number.isFinite(metadata.skyReference) ||
    metadata.skyReference <= 0
  )
    throw new Error('Invalid LUT metadata');
  if (
    !Number.isSafeInteger(metadata.irradianceHeightCount) ||
    metadata.irradianceHeightCount < 2 ||
    metadata.irradianceHeightCount > 256 ||
    !Number.isFinite(metadata.irradianceHeightMax) ||
    metadata.irradianceHeightMax <= 0 ||
    metadata.irradianceHeightMax > 100000
  )
    throw new Error('Invalid irradiance height grid');
  const densityIntegrals = metadata.densityIntegrals;
  if (
    !densityIntegrals ||
    densityIntegrals.version !== 2 ||
    densityIntegrals.encoding !== 'path-normalized-log-tail-rational-rgba32f' ||
    densityIntegrals.width !== 128 ||
    densityIntegrals.height !== 128
  )
    throw new Error('Invalid density integral layout');
  function floats(name: string, half = false): Float32Array<ArrayBuffer> {
    const entry = entries.get(name)!;
    const stride = half ? 2 : 4;
    if (entry.length % stride !== 0) throw new Error(`Invalid LUT array: ${name}`);
    const source = new DataView(entry.buffer, entry.byteOffset, entry.byteLength);
    const result = new Float32Array(entry.length / stride);
    for (let i = 0; i < result.length; i++) {
      result[i] = half ? decodeFloat16(source.getUint16(i * 2, true)) : source.getFloat32(i * 4, true);
      if (!Number.isFinite(result[i])) throw new Error(`Nonfinite LUT value: ${name}`);
    }
    return result;
  }
  const irradiance = floats('irradiance.f32');
  const packed = floats('scattering-pca.f16', true);
  const slices = 16 * 64,
    controls = 32 * 16 * 8;
  const factors: ScatteringFactors[] = [];
  const pair = mixed ? floats('scattering-mie-pair.f32') : null;
  if (pair && pair.length !== 2 * controls) throw new Error('Invalid Mie factor dimensions');
  let offset = 0;
  for (const [field, rank] of scattering.ranks.entries()) {
    const take = (length: number) => packed.slice(offset, (offset += length));
    if (pair && field === 2) {
      const a = take(slices * rank),
        b = new Float32Array(rank * controls);
      b.set(pair.subarray(controls));
      b.set(take((rank - 1) * controls), controls);
      factors.push({ mean: pair.subarray(0, controls), a, b, rank });
    } else {
      factors.push({ mean: take(controls), a: take(slices * rank), b: take(rank * controls), rank });
    }
  }
  if (offset !== packed.length) throw new Error('Invalid scattering factor dimensions');
  const skyNodes = floats('sky-scattering.f16', true);
  if (skyNodes.length !== 16 * 64 * 8 * 32 * 4) throw new Error('Invalid sky node dimensions');
  const colorBytes = entries.get('palette.rgba16f')!;
  if (colorBytes.byteLength !== 256 * 64 * 2 * 4 * 2) throw new Error('Invalid RGB palette dimensions');
  const colorView = new DataView(colorBytes.buffer, colorBytes.byteOffset, colorBytes.byteLength);
  const palette = new Uint16Array(colorBytes.byteLength / 2);
  for (let i = 0; i < palette.length; i++) {
    const value = colorView.getUint16(i * 2, true);
    if ((value & 0x7c00) === 0x7c00 || (i % 4 === 3 && value !== 0x3c00)) throw new Error('Invalid RGB palette texel');
    palette[i] = value;
  }
  const optical = floats('density-integrals.rgba32f');
  const refraction = metadata.refraction ? floats('refraction.rgba32f') : undefined;
  if (
    metadata.refraction &&
    (metadata.refraction.version !== 2 ||
      metadata.refraction.encoding !== 'fixed-map-celestial-v2' ||
      JSON.stringify(metadata.refraction.size) !== '[128,64,3]' ||
      metadata.refraction.refractivity !== 0.00029 ||
      metadata.refraction.scaleHeightKm !== 8 ||
      refraction?.length !== 128 * 64 * 3 * 4)
  )
    throw new Error('Invalid refraction layout');
  const transmission = floats('transmittance.rgba32f');
  // Keep signed/unclamped RGB values: clamping before filtering changes the result.
  if (
    transmission.length !== 256 * 256 * 4 ||
    transmission.subarray(0, 4).some((v) => v !== 1) ||
    transmission.some((v, i) => i % 4 === 3 && v !== 1)
  )
    throw new Error('Invalid RGB transmission texels');
  const heightCount = metadata.irradianceHeightCount;
  const solarCount = irradiance.length / (7 * heightCount);
  const opticalSize = densityIntegrals.width * densityIntegrals.height;
  if (!Number.isSafeInteger(solarCount) || solarCount < 2 || optical.length !== opticalSize * 4)
    throw new Error('Invalid LUT dimensions');
  for (let row = 0; row < heightCount; row++) {
    for (let column = 0; column < solarCount; column++) {
      const i = (row * solarCount + column) * 7;
      if ((column > 0 && irradiance[i] <= irradiance[i - 7]) || irradiance[i] !== irradiance[column * 7])
        throw new Error('Invalid irradiance coordinates');
    }
  }
  irradianceAxis(irradiance, solarCount);
  for (let row = 0; row < densityIntegrals.height; row++) {
    const start = row * densityIntegrals.width * 4;
    for (let col = 0; col < densityIntegrals.width; col++) {
      const i = start + col * 4;
      if (optical[i + 2] < 0 || optical[i + 3] !== 0) throw new Error('Invalid compact optical tail texel');
    }
  }
  return { metadata, scattering: factors, skyNodes, palette, transmission, irradiance, optical, refraction };
}

/** Uniform lattice containing every solar-altitude knot of the irradiance table. */
export type IrradianceAxis = { first: number; inverseStep: number; count: number };

/** The non-uniform knots all lie on one uniform lattice. Storing the fractional
 * knot index at every lattice point lets the shader find its column directly:
 * the index is linear between lattice points because every knot is one, so
 * interpolating it reproduces piecewise-linear knot interpolation exactly. */
function irradianceAxis(irradiance: Float32Array, solarCount: number): IrradianceAxis {
  const knot = (column: number) => irradiance[column * 7];
  let step = Infinity;
  for (let column = 1; column < solarCount; column++) step = Math.min(step, knot(column) - knot(column - 1));
  const first = knot(0);
  for (let column = 0; column < solarCount; column++) {
    const position = (knot(column) - first) / step;
    if (Math.abs(position - Math.round(position)) > 1e-6)
      throw new Error('Irradiance altitude knots are not on a uniform lattice');
  }
  const count = Math.round((knot(solarCount - 1) - first) / step) + 1;
  if (count > 65536) throw new Error('Irradiance altitude lattice is too fine');
  return { first, inverseStep: 1 / step, count };
}

export function getIrradianceAxis(lut: LUTData): IrradianceAxis {
  return irradianceAxis(lut.irradiance, lut.irradiance.length / (7 * lut.metadata.irradianceHeightCount));
}

/** Upload the small surface-illumination LUT, independently of screen transport.
 * Rows: [0,H) sky, [H,2H) sun, then the lattice-to-column index map (red). */
export function uploadIrradiance(gl: WebGL2RenderingContext, lut: LUTData): WebGLTexture {
  const height = lut.metadata.irradianceHeightCount;
  const samples = lut.irradiance.length / 7;
  const width = samples / height;
  const axis = getIrradianceAxis(lut);
  const indexRows = Math.ceil(axis.count / width);
  const textureHeight = 2 * height + indexRows;
  const limit = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  if (width > limit || textureHeight > limit) throw new Error('Irradiance LUT exceeds GPU limits');
  const pixels = new Float32Array(width * textureHeight * 4);
  for (let i = 0; i < samples; i++) {
    pixels.set(lut.irradiance.subarray(i * 7 + 1, i * 7 + 4), i * 4);
    pixels[i * 4 + 3] = lut.irradiance[i * 7];
    pixels.set(lut.irradiance.subarray(i * 7 + 4, i * 7 + 7), (width * height + i) * 4);
    pixels[(width * height + i) * 4 + 3] = lut.irradiance[i * 7];
  }
  for (let lattice = 0, column = 0; lattice < axis.count; lattice++) {
    const altitude = axis.first + lattice / axis.inverseStep;
    while (column < width - 2 && lut.irradiance[(column + 1) * 7] <= altitude) column++;
    const a = lut.irradiance[column * 7],
      b = lut.irradiance[(column + 1) * 7];
    pixels[(2 * width * height + lattice) * 4] = column + Math.min(1, Math.max(0, (altitude - a) / (b - a)));
  }
  const texture = gl.createTexture();
  if (!texture) throw new Error('Failed to allocate irradiance texture');
  const binding = gl.getParameter(gl.TEXTURE_BINDING_2D);
  const unpack = [
    [gl.UNPACK_ALIGNMENT, 1],
    [gl.UNPACK_FLIP_Y_WEBGL, 0],
    [gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 0],
    [gl.UNPACK_ROW_LENGTH, 0],
    [gl.UNPACK_IMAGE_HEIGHT, 0],
    [gl.UNPACK_SKIP_PIXELS, 0],
    [gl.UNPACK_SKIP_ROWS, 0],
    [gl.UNPACK_SKIP_IMAGES, 0],
  ].map(([parameter, value]) => ({ parameter, value, previous: Number(gl.getParameter(parameter)) }));
  const buffer = gl.getParameter(gl.PIXEL_UNPACK_BUFFER_BINDING);
  try {
    gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null);
    for (const { parameter, value } of unpack) gl.pixelStorei(parameter, value);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, textureHeight, 0, gl.RGBA, gl.FLOAT, pixels);
    return texture;
  } catch (error) {
    gl.deleteTexture(texture);
    throw error;
  } finally {
    gl.bindTexture(gl.TEXTURE_2D, binding);
    gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, buffer);
    for (const { parameter, previous } of unpack) gl.pixelStorei(parameter, previous);
  }
}
