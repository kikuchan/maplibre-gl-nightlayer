import { compileProgram, uniformLocation } from './gl.js';
import { SENSOR_PITCH_MM } from './receiver.js';
import type { CustomRenderMethodInput } from 'maplibre-gl';
import { bindForwardFrame, forwardFrame, forwardGeometryGLSL, type ProjectionCamera } from './surface.js';
import { refractionGLSL, type DirectionTransport } from './atmosphere.js';
import embeddedMilkyWay from './starfield.webp?inline';
import embeddedCatalog from './star-catalog.bin.gz?inline';

export type Starfield = {
  textures: [WebGLTexture];
  points: StarPoints;
  backgroundScale: number;
};
type Catalog = { stars: Float32Array<ArrayBuffer>; backgroundScale: number };
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
    if (header.getUint32(0, true) !== 0x5354444b || buffer.byteLength !== 16 + count * 40)
      throw new Error('Invalid embedded star catalogue');
    return { stars: new Float32Array(buffer, 16, count * 10), backgroundScale: header.getFloat32(8, true) };
  })();
  void catalog.catch(() => {
    catalog = undefined;
  });
  return catalog;
}

/** Earth-fixed (+X east at Greenwich, +Y north, +Z Greenwich) to J2000 equatorial. */
export function celestialRotation(date: Date): number[] {
  const days = date.getTime() / 86400000 - 10957.5;
  const t = days / 36525,
    radians = Math.PI / 180;
  const sidereal = ((280.46061837 + 360.98564736629 * days + 0.000387933 * t * t - t ** 3 / 38710000) % 360) * radians;
  // Inverse IAU 1976 precession; nutation and annual aberration are omitted.
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
    const a = x * Math.cos(z) + y * Math.sin(z),
      b = -x * Math.sin(z) + y * Math.cos(z);
    const c = a * Math.cos(theta) + north * Math.sin(theta),
      d = -a * Math.sin(theta) + north * Math.cos(theta);
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

/** Schlyter's low-precision lunar ephemeris, for illumination rather than occultations.
 * https://stjarnhimlen.se/comp/ppcomp.html */
export function lunarLight(date: Date): { position: number[]; phaseAngle: number; strength: number } {
  const rad = Math.PI / 180;
  const sin = (angle: number) => Math.sin(angle * rad),
    cos = (angle: number) => Math.cos(angle * rad);
  const days = date.getTime() / 86400000 - 10956;
  const node = 125.1228 - 0.0529538083 * days;
  const inclination = 5.1454;
  const periapsis = 318.0634 + 0.1643573223 * days;
  const anomaly = 115.3654 + 13.0649929509 * days;
  const eccentricity = 0.0549;
  let eccentric = anomaly * rad;
  for (let i = 0; i < 8; i++)
    eccentric -=
      (eccentric - eccentricity * Math.sin(eccentric) - anomaly * rad) / (1 - eccentricity * Math.cos(eccentric));
  const x = Math.cos(eccentric) - eccentricity,
    y = Math.sqrt(1 - eccentricity ** 2) * Math.sin(eccentric);
  const trueAnomaly = Math.atan2(y, x) / rad;
  let distance = 60.2666 * Math.hypot(x, y);
  const argument = trueAnomaly + periapsis;
  const ecliptic = [
    cos(node) * cos(argument) - sin(node) * sin(argument) * cos(inclination),
    sin(node) * cos(argument) + cos(node) * sin(argument) * cos(inclination),
    sin(argument) * sin(inclination),
  ];
  let longitude = Math.atan2(ecliptic[1], ecliptic[0]) / rad,
    latitude = Math.asin(ecliptic[2]) / rad;
  const sunAnomaly = 356.047 + 0.9856002585 * days;
  const sunPeriapsis = 282.9404 + 0.0000470935 * days;
  const sunLongitude = sunAnomaly + sunPeriapsis;
  const meanLongitude = anomaly + periapsis + node;
  const elongation = meanLongitude - sunLongitude,
    argumentLatitude = meanLongitude - node;
  longitude +=
    -1.274 * sin(anomaly - 2 * elongation) +
    0.658 * sin(2 * elongation) -
    0.186 * sin(sunAnomaly) -
    0.059 * sin(2 * anomaly - 2 * elongation) -
    0.057 * sin(anomaly - 2 * elongation + sunAnomaly) +
    0.053 * sin(anomaly + 2 * elongation) +
    0.046 * sin(2 * elongation - sunAnomaly) +
    0.041 * sin(anomaly - sunAnomaly) -
    0.035 * sin(elongation) -
    0.031 * sin(anomaly + sunAnomaly) -
    0.015 * sin(2 * argumentLatitude - 2 * elongation) +
    0.011 * sin(anomaly - 4 * elongation);
  latitude +=
    -0.173 * sin(argumentLatitude - 2 * elongation) -
    0.055 * sin(anomaly - argumentLatitude - 2 * elongation) -
    0.046 * sin(anomaly + argumentLatitude - 2 * elongation) +
    0.033 * sin(argumentLatitude + 2 * elongation) +
    0.017 * sin(2 * anomaly + argumentLatitude);
  distance += -0.58 * cos(anomaly - 2 * elongation) - 0.46 * cos(2 * elongation);
  const obliquity = 23.4393 - 3.563e-7 * days;
  const ex = cos(longitude) * cos(latitude),
    ey = sin(longitude) * cos(latitude),
    ez = sin(latitude);
  const equatorial = [ex, ey * cos(obliquity) - ez * sin(obliquity), ey * sin(obliquity) + ez * cos(obliquity)];
  const j2000Days = days - 1.5,
    t = j2000Days / 36525;
  const sidereal = 280.46061837 + 360.98564736629 * j2000Days + 0.000387933 * t * t - t ** 3 / 38710000;
  const kilometres = distance * 6378.14;
  const position = [
    (equatorial[1] * cos(sidereal) - equatorial[0] * sin(sidereal)) * kilometres,
    equatorial[2] * kilometres,
    (equatorial[0] * cos(sidereal) + equatorial[1] * sin(sidereal)) * kilometres,
  ];
  const sunTrueLongitude = sunLongitude + 1.915 * sin(sunAnomaly) + 0.02 * sin(2 * sunAnomaly);
  const sunDistance = 149597870.7 * (1 - 0.01671 * cos(sunAnomaly));
  const separationCos = cos(latitude) * cos(longitude - sunTrueLongitude);
  const separationSin = Math.sqrt(Math.max(0, 1 - separationCos ** 2));
  const phaseAngle = Math.atan2(sunDistance * separationSin, kilometres - sunDistance * separationCos);
  const phaseDegrees = phaseAngle / rad;
  // Empirical lunar phase law, faded over the final ten degrees to new Moon.
  const crescent = Math.max(0, Math.min(1, (180 - phaseDegrees) / 10));
  const strength =
    0.25 *
    (384400 / kilometres) ** 2 *
    10 ** (-0.4 * (0.026 * phaseDegrees + 4e-9 * phaseDegrees ** 4)) *
    crescent *
    crescent *
    (3 - 2 * crescent);
  return { position, phaseAngle, strength };
}

// An illustrative optical model, not a measured lens: 90% sharp core and
// 10% scattered wing. Both Gaussians are integrated over each sensor pixel.
const PSF_CORE_SIGMA_MM = 0.0125;
const PSF_WING_SIGMA_MM = 0.05;
const PSF_RADIUS_MM = 4 * PSF_WING_SIGMA_MM;
const ZERO_CLEAR = new Float32Array(4);

/** Rasterize only stellar footprints; store flux, not radiance, in half floats. */
export class StarPoints {
  readonly texture: WebGLTexture;
  #program?: WebGLProgram;
  #buffer: WebGLBuffer;
  #vao: WebGLVertexArrayObject;
  #framebuffer: WebGLFramebuffer;
  #count: number;
  #quads: boolean;
  #pointLimit: number;
  #width = 0;
  #height = 0;
  #signature?: string;
  #transport?: DirectionTransport;
  #variant?: string;

  constructor(gl: WebGL2RenderingContext, stars: Float32Array<ArrayBuffer>) {
    this.#pointLimit = (gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) as Float32Array)[1];
    this.#quads = this.#pointLimit < 3;
    const texture = gl.createTexture(),
      buffer = gl.createBuffer(),
      vao = gl.createVertexArray(),
      framebuffer = gl.createFramebuffer();
    if (!texture || !buffer || !vao || !framebuffer) {
      gl.deleteTexture(texture);
      gl.deleteBuffer(buffer);
      gl.deleteVertexArray(vao);
      gl.deleteFramebuffer(framebuffer);
      throw new Error('Failed to allocate NightLayer point stars');
    }
    this.texture = texture;
    this.#buffer = buffer;
    this.#vao = vao;
    this.#framebuffer = framebuffer;
    this.#count = stars.length / 10;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, stars, gl.STATIC_DRAW);
    // The tenth float is an estimated angular diameter (radians). The current
    // PSF renderer uses the first nine floats, while retaining the wider stride.
    for (let i = 0; i < 3; i++) {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, 3, gl.FLOAT, false, 40, i * 12);
      if (this.#quads) gl.vertexAttribDivisor(i, 1);
    }
    gl.bindVertexArray(null);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, 1, 1, 0, gl.RGBA, gl.HALF_FLOAT, null);
  }

  render(
    gl: WebGL2RenderingContext,
    camera: ProjectionCamera,
    rotation: number[],
    options: CustomRenderMethodInput,
    years: number,
    size: readonly [number, number],
    transport?: DirectionTransport,
  ): void {
    // Stars are received on the logical sensor grid: one texel per sensor pixel.
    const [width, height] = size;
    const pixelScale = [1, 1] as const;
    const projection = options.defaultProjectionData;
    const variant = options.shaderData.variantName;
    const signature = JSON.stringify([
      width,
      height,
      variant,
      projection.mainMatrix,
      projection.fallbackMatrix,
      projection.projectionTransition,
      camera.origin,
      rotation,
      years,
      pixelScale,
    ]);
    if (signature === this.#signature && transport === this.#transport) return;
    const radius = PSF_RADIUS_MM / SENSOR_PITCH_MM;
    const pointSize = 2 * Math.ceil(radius * Math.max(...pixelScale)) + 3;
    const quads = pointSize > this.#pointLimit;
    if (!this.#program || quads !== this.#quads || variant !== this.#variant) {
      const source = (shader: string) => {
        const expanded = shader.replace(
          '// FORWARD_PROJECTION_PRELUDE',
          `${options.shaderData.define}\n${options.shaderData.vertexShaderPrelude}`,
        );
        return quads ? expanded.replace('\n', '\n#define STAR_QUADS\n') : expanded;
      };
      const program = compileProgram(gl, source(starFragment), source(starVertex));
      if (this.#program) gl.deleteProgram(this.#program);
      this.#program = program;
      this.#quads = quads;
      this.#variant = variant;
      gl.bindVertexArray(this.#vao);
      for (let i = 0; i < 3; i++) gl.vertexAttribDivisor(i, quads ? 1 : 0);
      gl.bindVertexArray(null);
    }
    const framebuffer = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
    try {
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      const resized = width !== this.#width || height !== this.#height;
      if (resized) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, width, height, 0, gl.RGBA, gl.HALF_FLOAT, null);
      }
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.#framebuffer);
      if (resized) {
        gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texture, 0);
        if (gl.checkFramebufferStatus(gl.DRAW_FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
          throw new Error('Incomplete NightLayer point-star framebuffer');
        this.#width = width;
        this.#height = height;
      }
      gl.viewport(0, 0, width, height);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.STENCIL_TEST);
      gl.disable(gl.SCISSOR_TEST);
      gl.disable(gl.CULL_FACE);
      gl.colorMask(true, true, true, true);
      // Clear this attachment without touching the shared clear-color state.
      // Reading COLOR_CLEAR_VALUE back can synchronise the command stream.
      gl.clearBufferfv(gl.COLOR, 0, ZERO_CLEAR);
      gl.enable(gl.BLEND);
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(this.#program);
      if (transport) transport.bind(this.#program);
      else gl.uniform1i(uniformLocation(gl, this.#program, 'u_refraction'), 0);
      gl.uniform3fv(uniformLocation(gl, this.#program, 'u_refraction_origin'), camera.origin);
      gl.uniform1f(
        uniformLocation(gl, this.#program, 'u_refraction_height'),
        Math.max(0, Math.hypot(...camera.origin) - 6371),
      );
      this.#transport = transport;
      gl.bindVertexArray(this.#vao);
      bindForwardFrame(gl, this.#program, forwardFrame(camera, options));
      gl.uniformMatrix3fv(uniformLocation(gl, this.#program, 'u_celestial_rotation'), false, rotation);
      gl.uniform1f(uniformLocation(gl, this.#program, 'u_star_years'), years);
      gl.uniform2f(uniformLocation(gl, this.#program, 'u_viewport'), width, height);
      gl.uniform2f(uniformLocation(gl, this.#program, 'u_pixel_scale'), pixelScale[0], pixelScale[1]);
      gl.uniform1f(uniformLocation(gl, this.#program, 'u_point_size'), pointSize);
      gl.uniform1f(uniformLocation(gl, this.#program, 'u_psf_radius'), radius);
      gl.uniform2f(
        uniformLocation(gl, this.#program, 'u_psf_sigma'),
        PSF_CORE_SIGMA_MM / SENSOR_PITCH_MM,
        PSF_WING_SIGMA_MM / SENSOR_PITCH_MM,
      );
      if (this.#quads) gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.#count);
      else gl.drawArrays(gl.POINTS, 0, this.#count);
      this.#signature = signature;
    } finally {
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, framebuffer);
      gl.disable(gl.BLEND);
    }
  }

  dispose(gl: WebGL2RenderingContext): void {
    gl.deleteTexture(this.texture);
    gl.deleteFramebuffer(this.#framebuffer);
    gl.deleteBuffer(this.#buffer);
    gl.deleteVertexArray(this.#vao);
    if (this.#program) gl.deleteProgram(this.#program);
  }
}

const starVertex = `#version 300 es
precision highp float;
precision highp int;
// FORWARD_PROJECTION_PRELUDE
layout(location = 0) in vec3 a_direction;
layout(location = 1) in vec3 a_flux;
layout(location = 2) in vec3 a_motion;
uniform mat3 u_celestial_rotation;
uniform float u_star_years;
uniform vec2 u_viewport;
uniform vec2 u_pixel_scale;
uniform float u_point_size;
uniform float u_psf_radius;
#ifdef STAR_QUADS
out vec2 v_pixel;
#else
flat out vec2 v_offset;
#endif
flat out vec3 v_flux;
${forwardGeometryGLSL}
${refractionGLSL}
uniform vec3 u_refraction_origin;
uniform float u_refraction_height;
void main() {
  vec3 direction = transpose(u_celestial_rotation) * normalize(a_direction + a_motion * u_star_years);
  direction=refractionScreenRay(u_refraction_origin,u_refraction_height,direction);
  vec4 projected = projectSky(direction);
  vec2 clip = projected.xy / projected.w;
  bool visible = projected.w > 0.0 && !any(isnan(clip)) && !any(isinf(clip))
    && all(lessThanEqual(abs(clip), vec2(1.0) + (2.0 * u_psf_radius * u_pixel_scale + 1.0) / u_viewport));
  v_flux = a_flux * 1e12;
#ifdef STAR_QUADS
  vec2 corner = vec2(float((gl_VertexID == 1 || gl_VertexID == 2 || gl_VertexID == 4) ? 1 : -1),
                     float((gl_VertexID == 2 || gl_VertexID == 4 || gl_VertexID == 5) ? 1 : -1));
   v_pixel = corner * (u_psf_radius * u_pixel_scale + .5);
  gl_Position = visible ? vec4(clip + v_pixel * 2.0 / u_viewport, 0, 1) : vec4(2,2,2,1);
#else
  // Point primitives are clipped by their centre, not by their footprint.
  // Move offscreen centres to the nearest edge pixel and undo that translation
  // in the fragment shader, so stars enter/leave the frame continuously.
  vec2 centre = (clip * .5 + .5) * u_viewport;
  vec2 clamped = clamp(centre, vec2(.5), u_viewport - .5);
  v_offset = clamped - centre;
  gl_PointSize = u_point_size;
  gl_Position = visible ? vec4(clamped / u_viewport * 2.0 - 1.0, 0, 1) : vec4(2,2,2,1);
#endif
}`;

const starFragment = `#version 300 es
precision highp float;
#ifdef STAR_QUADS
in vec2 v_pixel;
#else
flat in vec2 v_offset;
#endif
flat in vec3 v_flux;
uniform vec2 u_pixel_scale;
uniform float u_point_size;
uniform vec2 u_psf_sigma;
out vec4 fragColor;
// Gaussian upper-tail probability for x >= 0 (absolute error < 7.5e-8).
// Subtract tails on the same side to avoid losing faint wings to 1-CDF.
float tail(float x) {
  float t = 1.0 / (1.0 + .2316419 * x);
  return .3989422804 * exp(-.5 * x * x) * t *
    (.319381530 + t * (-.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
}
float interval(float lo, float hi) {
  if (lo >= 0.0) return tail(lo) - tail(hi);
  if (hi <= 0.0) return tail(-hi) - tail(-lo);
  return 1.0 - tail(-lo) - tail(hi);
}
float gaussianPixel(vec2 pixel, float sigma) {
  vec2 lo = clamp((pixel - .5) / (u_pixel_scale * sigma), -4.0, 4.0);
  vec2 hi = clamp((pixel + .5) / (u_pixel_scale * sigma), -4.0, 4.0);
  float normalization = 1.0 - 2.0 * tail(4.0);
  return max(0.0, interval(lo.x, hi.x) * interval(lo.y, hi.y)) /
    (normalization * normalization);
}
void main() {
#ifdef STAR_QUADS
  vec2 pixel = v_pixel;
#else
  vec2 pixel = (gl_PointCoord - .5) * vec2(u_point_size, -u_point_size) + v_offset;
#endif
  // Each component is truncated at +/-4 sigma per axis and renormalized.
  // Brightness scales this fixed PSF, never its width or total flux.
  float weight = .9 * gaussianPixel(pixel, u_psf_sigma.x)
    + .1 * gaussianPixel(pixel, u_psf_sigma.y);
  if (weight == 0.0) discard;
  fragColor = vec4(v_flux * weight, 0);
}`;
