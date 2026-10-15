import embeddedMilkyWay from './starfield.webp?inline';
import embeddedCatalog from './star-catalog.bin.gz?inline';
import { StarPoints } from './star-points.js';

export type Starfield = {
  textures: [WebGLTexture];
  points: StarPoints;
  backgroundScale: number;
};
type Catalog = {
  stars: Float32Array<ArrayBuffer>;
  backgroundScale: number;
};
let catalog: Promise<Catalog> | undefined;

function embeddedBytes(url: string): Uint8Array<ArrayBuffer> {
  return Uint8Array.from(atob(url.slice(url.indexOf(',') + 1)), (char) => char.charCodeAt(0));
}

function loadCatalog(): Promise<Catalog> {
  if (catalog) return catalog;
  catalog = (async () => {
    const buffer = await new Response(
      new Blob([embeddedBytes(embeddedCatalog)]).stream().pipeThrough(new DecompressionStream('gzip')),
    ).arrayBuffer();
    const header = new DataView(buffer);
    const count = header.getUint32(4, true);
    if (header.getUint32(0, true) !== 0x53545054 || buffer.byteLength !== 16 + count * 36)
      throw new Error('Invalid embedded star catalogue');
    return {
      stars: new Float32Array(buffer, 16, count * 9),
      backgroundScale: header.getFloat32(8, true),
    };
  })();
  void catalog.catch(() => {
    catalog = undefined;
  });
  return catalog;
}

/** Earth-fixed (+X east at Greenwich, +Y north, +Z Greenwich) to J2000 equatorial. */
export function celestialRotation(date: Date): number[] {
  const days = date.getTime() / 86400000 - 10957.5;
  const t = days / 36525;
  const radians = Math.PI / 180;
  const sidereal = ((280.46061837 + 360.98564736629 * days + 0.000387933 * t * t - t ** 3 / 38710000) % 360) * radians;
  // IAU 1976 precession: invert J2000 -> mean equator/equinox of date.
  // This mean-equator model omits nutation and annual aberration. Individual
  // proper motions are applied separately to catalogue directions in the shader.
  const zeta = ((2306.2181 * t + 0.30188 * t * t + 0.017998 * t ** 3) * radians) / 3600;
  const z = ((2306.2181 * t + 1.09468 * t * t + 0.018203 * t ** 3) * radians) / 3600;
  const theta = ((2004.3109 * t - 0.42665 * t * t - 0.041833 * t ** 3) * radians) / 3600;
  return [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ].flatMap(([east, north, greenwich]) => {
    const x = greenwich * Math.cos(sidereal) - east * Math.sin(sidereal);
    const y = greenwich * Math.sin(sidereal) + east * Math.cos(sidereal);
    const a = x * Math.cos(z) + y * Math.sin(z);
    const b = -x * Math.sin(z) + y * Math.cos(z);
    const c = a * Math.cos(theta) + north * Math.sin(theta);
    const d = -a * Math.sin(theta) + north * Math.cos(theta);
    return [c * Math.cos(zeta) + b * Math.sin(zeta), -c * Math.sin(zeta) + b * Math.cos(zeta), d];
  });
}

export async function uploadStarfield(gl: WebGL2RenderingContext): Promise<Starfield> {
  const data = await loadCatalog();
  const image = await createImageBitmap(new Blob([embeddedBytes(embeddedMilkyWay)], { type: 'image/webp' }), {
    colorSpaceConversion: 'none',
    premultiplyAlpha: 'none',
    imageOrientation: 'none',
  });
  const textures: WebGLTexture[] = [];
  let points: StarPoints | undefined;
  const unpack = [
    gl.UNPACK_ALIGNMENT,
    gl.UNPACK_FLIP_Y_WEBGL,
    gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,
    gl.UNPACK_ROW_LENGTH,
    gl.UNPACK_IMAGE_HEIGHT,
    gl.UNPACK_SKIP_PIXELS,
    gl.UNPACK_SKIP_ROWS,
    gl.UNPACK_SKIP_IMAGES,
  ].map((parameter) => ({ parameter, previous: Number(gl.getParameter(parameter)) }));
  function texture(): WebGLTexture {
    const value = gl.createTexture();
    if (!value) throw new Error('Failed to create NightLayer star texture');
    textures.push(value);
    gl.bindTexture(gl.TEXTURE_2D, value);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return value;
  }
  try {
    for (const { parameter } of unpack) gl.pixelStorei(parameter, parameter === gl.UNPACK_ALIGNMENT ? 1 : 0);
    gl.activeTexture(gl.TEXTURE0);
    texture();
    // sRGB storage decodes before filtering; mipmaps retain linear-light averages.
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.SRGB8_ALPHA8, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.generateMipmap(gl.TEXTURE_2D);
    points = new StarPoints(gl, data.stars);
    if (gl.getError() !== gl.NO_ERROR) throw new Error('Failed to upload NightLayer star texture');
    return { textures: textures as Starfield['textures'], points, backgroundScale: data.backgroundScale };
  } catch (error) {
    for (const value of textures) gl.deleteTexture(value);
    points?.dispose(gl);
    throw error;
  } finally {
    for (const { parameter, previous } of unpack) gl.pixelStorei(parameter, previous);
    image.close();
  }
}
