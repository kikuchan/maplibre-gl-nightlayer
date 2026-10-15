import embeddedLUT from './lighting-lut.zip?inline';

export type LUTMetadata = {
  gridSize: [number, number, number, number, number];
  solarRadiance: [number, number, number];
  solarAngularRadius: number;
  mieG: number;
  skyReference: number;
};

type LUTData = {
  metadata: LUTMetadata;
  irradiance: Float32Array<ArrayBuffer>;
  rayleigh: Float32Array<ArrayBuffer>;
  mie: Float32Array<ArrayBuffer>;
  optical: Float32Array<ArrayBuffer>;
};

const data = new Map<string | undefined, Promise<LUTData>>();
const crcTable = Uint32Array.from({ length: 256 }, (_, value) => {
  for (let bit = 0; bit < 8; bit++) value = (value >>> 1) ^ (value & 1 ? 0xedb88320 : 0);
  return value >>> 0;
});

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const value of bytes) crc = (crc >>> 8) ^ crcTable[(crc ^ value) & 255];
  return (crc ^ 0xffffffff) >>> 0;
}

function loadLUT(source?: string | URL): Promise<LUTData> {
  const key = source?.toString();
  const cached = data.get(key);
  if (cached) return cached;
  const pending = readLUT(key);
  data.set(key, pending);
  void pending.catch(() => data.delete(key));
  return pending;
}

async function readLUT(source?: string): Promise<LUTData> {
  let bytes: Uint8Array<ArrayBuffer>;
  if (source === undefined) {
    const binary = atob(embeddedLUT.slice(embeddedLUT.indexOf(',') + 1));
    bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  } else {
    const response = await fetch(source);
    if (!response.ok) throw new Error('Failed to load NightLayer LUT');
    bytes = new Uint8Array(await response.arrayBuffer());
  }
  const view = new DataView(bytes.buffer);
  const text = new TextDecoder();
  const entries = new Map<string, Uint8Array<ArrayBuffer>>();
  let offset = 0;
  while (offset + 4 <= bytes.length && view.getUint32(offset, true) === 0x04034b50) {
    if (offset + 30 > bytes.length) throw new Error('Truncated LUT ZIP header');
    const flags = view.getUint16(offset + 6, true);
    const method = view.getUint16(offset + 8, true);
    const checksum = view.getUint32(offset + 14, true);
    const compressedSize = view.getUint32(offset + 18, true);
    const size = view.getUint32(offset + 22, true);
    const nameLength = view.getUint16(offset + 26, true);
    const extraLength = view.getUint16(offset + 28, true);
    const start = offset + 30 + nameLength + extraLength;
    const end = start + compressedSize;
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
  const header = entries.get('metadata.json');
  if (!header) throw new Error('Missing LUT metadata');
  const metadata: LUTMetadata = JSON.parse(text.decode(header));
  if (
    !Array.isArray(metadata.gridSize) ||
    metadata.gridSize.length !== 5 ||
    metadata.gridSize.some((n) => !Number.isSafeInteger(n) || n < 3 || n > 256) ||
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
  function floats(name: string): Float32Array<ArrayBuffer> {
    const entry = entries.get(name);
    if (!entry || entry.length % 4 !== 0) throw new Error(`Invalid LUT array: ${name}`);
    const source = new DataView(entry.buffer);
    const result = new Float32Array(entry.length / 4);
    for (let i = 0; i < result.length; i++) {
      result[i] = source.getFloat32(i * 4, true);
      if (!Number.isFinite(result[i])) throw new Error(`Nonfinite LUT value: ${name}`);
    }
    return result;
  }
  const irradiance = floats('irradiance.bin');
  const rayleigh = floats('rayleigh.bin');
  const mie = floats('mie.bin');
  const optical = floats('optical.bin');
  const [h, v, d, s, a] = metadata.gridSize;
  if (
    entries.size !== 5 ||
    irradiance.length < 14 ||
    irradiance.length % 7 !== 0 ||
    rayleigh.length !== 2 * h * v * d * s * a * 3 ||
    mie.length !== rayleigh.length ||
    optical.length !== 2 * h * v * d * 3
  )
    throw new Error('Invalid LUT dimensions');
  for (let i = 0; i < irradiance.length; i += 7)
    if (i > 0 && irradiance[i] <= irradiance[i - 7]) throw new Error('Invalid irradiance coordinates');
  if (optical.some((v) => v < 0)) throw new Error('Negative optical coefficient');
  return { metadata, irradiance, rayleigh, mie, optical };
}

function volumeTiles(grid: number[], limit: number): [number, number, number] {
  const [h, v, d, s, a] = grid;
  const block = [d + 2, v + 2, h + 2];
  let best: [number, number, number] | undefined;
  let score = [Infinity, Infinity, Infinity];
  for (let x = 1; x * block[0] <= limit; x++) {
    for (let y = 1; y * block[1] <= limit; y++) {
      const z = Math.ceil((2 * s * a) / (x * y));
      const extent = [x * block[0], y * block[1], z * block[2]];
      if (extent.some((n) => n > limit)) continue;
      const next = [x * y * z, Math.max(...extent), extent.reduce((sum, n) => sum + n * n, 0)];
      const firstDifference = next.findIndex((n, i) => n !== score[i]);
      if (firstDifference >= 0 && next[firstDifference] < score[firstDifference]) {
        best = [x, y, z];
        score = next;
      }
    }
  }
  if (!best) throw new Error('Atmosphere LUT exceeds the GPU texture size limit');
  return best;
}

export async function uploadLUT(
  gl: WebGL2RenderingContext,
  source?: string | URL,
): Promise<{
  metadata: LUTMetadata;
  tiles: [number, number, number];
  textures: [WebGLTexture, WebGLTexture, WebGLTexture, WebGLTexture];
}> {
  const lut = await loadLUT(source);
  const [h, v, d, s, a] = lut.metadata.gridSize;
  const tiles = volumeTiles(lut.metadata.gridSize, Math.min(256, gl.getParameter(gl.MAX_3D_TEXTURE_SIZE)));
  const extent = [d + 2, v + 2, h + 2].map((n, i) => n * tiles[i]);
  const textures: WebGLTexture[] = [];
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
  function texture(target: number): void {
    const result = gl.createTexture();
    if (!result) throw new Error('Failed to allocate LUT texture');
    textures.push(result);
    gl.bindTexture(target, result);
    gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    for (const axis of target === gl.TEXTURE_3D
      ? [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]
      : [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T])
      gl.texParameteri(target, axis, gl.CLAMP_TO_EDGE);
  }
  try {
    gl.activeTexture(gl.TEXTURE0);
    for (const { parameter, value } of unpack) gl.pixelStorei(parameter, value);
    for (const source of [lut.rayleigh, lut.mie]) {
      const packed = new Float32Array(extent[0] * extent[1] * extent[2] * 3);
      for (let page = 0; page < 2 * s * a; page++) {
        const tx = (page % tiles[0]) * (d + 2);
        const ty = (Math.floor(page / tiles[0]) % tiles[1]) * (v + 2);
        const tz = Math.floor(page / (tiles[0] * tiles[1])) * (h + 2);
        for (let z = 0; z < h + 2; z++)
          for (let y = 0; y < v + 2; y++)
            for (let x = 0; x < d + 2; x++) {
              const input =
                (((page * h + Math.min(h - 1, Math.max(0, z - 1))) * v + Math.min(v - 1, Math.max(0, y - 1))) * d +
                  Math.min(d - 1, Math.max(0, x - 1))) *
                3;
              const output = (((tz + z) * extent[1] + ty + y) * extent[0] + tx + x) * 3;
              packed.set(source.subarray(input, input + 3), output);
            }
      }
      texture(gl.TEXTURE_3D);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGB32F, extent[0], extent[1], extent[2], 0, gl.RGB, gl.FLOAT, packed);
    }
    if (2 * h > gl.getParameter(gl.MAX_3D_TEXTURE_SIZE)) throw new Error('Optical LUT exceeds GPU limits');
    texture(gl.TEXTURE_3D);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGB32F, d, v, 2 * h, 0, gl.RGB, gl.FLOAT, lut.optical);
    const count = lut.irradiance.length / 7;
    if (count > gl.getParameter(gl.MAX_TEXTURE_SIZE)) throw new Error('Irradiance LUT exceeds GPU limits');
    const irradiance = new Float32Array(count * 8);
    for (let i = 0; i < count; i++) {
      irradiance.set(lut.irradiance.subarray(i * 7 + 1, i * 7 + 4), i * 4);
      irradiance[i * 4 + 3] = lut.irradiance[i * 7];
      irradiance.set(lut.irradiance.subarray(i * 7 + 4, i * 7 + 7), (count + i) * 4);
      irradiance[(count + i) * 4 + 3] = lut.irradiance[i * 7];
    }
    texture(gl.TEXTURE_2D);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, count, 2, 0, gl.RGBA, gl.FLOAT, irradiance);
    if (gl.getError() !== gl.NO_ERROR) throw new Error('Failed to upload LUT textures');
    return {
      metadata: lut.metadata,
      tiles,
      textures: textures as [WebGLTexture, WebGLTexture, WebGLTexture, WebGLTexture],
    };
  } catch (error) {
    for (const value of textures) gl.deleteTexture(value);
    throw error;
  } finally {
    for (const { parameter, previous } of unpack) gl.pixelStorei(parameter, previous);
  }
}
