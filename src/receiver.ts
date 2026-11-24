import { compileProgram, uniformLocation } from './gl.js';

/** A fixed sensor pitch per CSS pixel; a 960 px-high view spans 24 mm. */
export const SENSOR_PITCH_MM: number = 24 / 960;

export function focalLength(fovDegrees: number, sensorHeightMm = 24): number {
  return sensorHeightMm / (2 * Math.tan((fovDegrees * Math.PI) / 360));
}

const METER_SIZE = 16;
const SAMPLE_INTERVAL = 250;
const EXPOSURE_METER_MIN_LOG = -40;
const EXPOSURE_METER_LOG_RANGE = 44;
const MAX_EXPOSURE = 32;

type ExposureSample = { log: number; weight: number };

/** Centre-weighted scene metering with independent extended-highlight protection. */
function meterExposure(samples: ExposureSample[], photographicGain = 1): number {
  if (!samples.length) return 0;
  const sorted = [...samples].sort((a, b) => a.log - b.log);
  const total = sorted.reduce((sum, sample) => sum + sample.weight, 0);
  const percentile = (fraction: number) => {
    let weight = 0;
    for (const sample of sorted) {
      weight += sample.weight;
      if (weight >= fraction * total) return sample.log;
    }
    return sorted.at(-1)!.log;
  };
  const gainEV = Math.log2(photographicGain);
  const ceiling = percentile(0.95);
  const mean = sorted.reduce((sum, sample) => sum + Math.min(sample.log, ceiling) * sample.weight, 0) / total - gainEV;
  // Fade the daylight 18% anchor toward 0.9% at night, rather than lifting darkness to grey.
  const night = Math.max(0, Math.min(1, (Math.log2(0.18) - mean - 6) / 12));
  const targetEV = Math.log2(0.18) - night * Math.log2(20);
  // Extended highlights stay below linear 1; sparse stars are excluded before metering.
  return Math.min(targetEV - mean, -(percentile(0.98) - gainEV));
}

/** Low-resolution metering of unexposed receiver RGB, with asynchronous readback. */
export class AutoExposure {
  #framebuffer?: WebGLFramebuffer;
  #color?: WebGLRenderbuffer;
  #buffer?: WebGLBuffer;
  #fence?: WebGLSync;
  #pixels = new Uint8Array(METER_SIZE * METER_SIZE * 4);
  #dirty = true;
  #sampleTime = -Infinity;
  #updateTime?: number;
  #target = 0;
  #meterTarget?: number;
  #sampleGain = 1;
  value = 0;

  constructor(exposure = 0) {
    this.setExposure(exposure);
  }
  setExposure(exposure: number): void {
    this.value = exposure;
    this.#target = exposure;
    this.#meterTarget = undefined;
    this.#updateTime = undefined;
  }
  invalidate(): void {
    this.#dirty = true;
  }
  get needsFrame(): boolean {
    return this.#dirty || !!this.#fence || Math.abs(this.#target - this.value) > 0.02;
  }
  update(gl: WebGL2RenderingContext, now: number, photographicGain = 1, compensation = 0): void {
    if (this.#fence) {
      const status = gl.clientWaitSync(this.#fence, 0, 0);
      if (status === gl.WAIT_FAILED) throw new Error('Failed to read exposure metering');
      if (status !== gl.TIMEOUT_EXPIRED) {
        const previous = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.#buffer!);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.#pixels);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, previous);
        gl.deleteSync(this.#fence);
        this.#fence = undefined;
        const samples: ExposureSample[] = [];
        for (let y = 0; y < METER_SIZE; y++)
          for (let x = 0; x < METER_SIZE; x++) {
            const i = (y * METER_SIZE + x) * 4;
            if (this.#pixels[i + 3] === 0) continue;
            const radius = (((x + 0.5) / METER_SIZE) * 2 - 1) ** 2 + (((y + 0.5) / METER_SIZE) * 2 - 1) ** 2;
            samples.push({
              log: (this.#pixels[i] / 255) * EXPOSURE_METER_LOG_RANGE + EXPOSURE_METER_MIN_LOG,
              weight: Math.exp(-2 * radius),
            });
          }
        if (samples.length) this.#meterTarget = meterExposure(samples, this.#sampleGain);
        else this.#meterTarget ??= this.value + Math.log2(photographicGain) - compensation;
      }
    }
    if (this.#meterTarget !== undefined)
      this.#target = Math.max(
        -12,
        Math.min(MAX_EXPOSURE, this.#meterTarget - Math.log2(photographicGain) + compensation),
      );
    const elapsed = Math.min(0.1, Math.max(0, (now - (this.#updateTime ?? now)) / 1000));
    this.#updateTime = now;
    const seconds = 0.2;
    this.value += (this.#target - this.value) * (1 - Math.exp(-elapsed / seconds));
    if (Math.abs(this.#target - this.value) <= 0.02) this.value = this.#target;
  }
  measure(gl: WebGL2RenderingContext, now: number, draw: () => void, photographicGain = 1): void {
    if (!this.#dirty || this.#fence || now - this.#sampleTime < SAMPLE_INTERVAL) return;
    const read = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING);
    const write = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING);
    const renderbuffer = gl.getParameter(gl.RENDERBUFFER_BINDING);
    const buffer = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING);
    const viewport = gl.getParameter(gl.VIEWPORT) as Int32Array;
    const pack = [gl.PACK_ALIGNMENT, gl.PACK_ROW_LENGTH, gl.PACK_SKIP_PIXELS, gl.PACK_SKIP_ROWS].map((parameter) => ({
      parameter,
      value: gl.getParameter(parameter) as number,
    }));
    try {
      if (!this.#framebuffer) {
        this.#framebuffer = gl.createFramebuffer() ?? undefined;
        this.#color = gl.createRenderbuffer() ?? undefined;
        this.#buffer = gl.createBuffer() ?? undefined;
        if (!this.#framebuffer || !this.#color || !this.#buffer)
          throw new Error('Failed to allocate exposure metering');
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.#framebuffer);
        gl.bindRenderbuffer(gl.RENDERBUFFER, this.#color);
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, METER_SIZE, METER_SIZE);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, this.#color);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
          throw new Error('Incomplete exposure metering framebuffer');
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.#buffer);
        gl.bufferData(gl.PIXEL_PACK_BUFFER, this.#pixels.byteLength, gl.STREAM_READ);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.#framebuffer);
      gl.viewport(0, 0, METER_SIZE, METER_SIZE);
      draw();
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.#buffer!);
      for (const { parameter } of pack) gl.pixelStorei(parameter, parameter === gl.PACK_ALIGNMENT ? 1 : 0);
      gl.readPixels(0, 0, METER_SIZE, METER_SIZE, gl.RGBA, gl.UNSIGNED_BYTE, 0);
      this.#fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0) ?? undefined;
      if (!this.#fence) throw new Error('Failed to schedule exposure metering');
      gl.flush();
      this.#sampleTime = now;
      this.#sampleGain = photographicGain;
      this.#dirty = false;
    } finally {
      for (const { parameter, value } of pack) gl.pixelStorei(parameter, value);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buffer);
      gl.bindRenderbuffer(gl.RENDERBUFFER, renderbuffer);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, read);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, write);
      gl.viewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    }
  }
  dispose(gl: WebGL2RenderingContext): void {
    if (this.#fence) gl.deleteSync(this.#fence);
    if (this.#buffer) gl.deleteBuffer(this.#buffer);
    if (this.#color) gl.deleteRenderbuffer(this.#color);
    if (this.#framebuffer) gl.deleteFramebuffer(this.#framebuffer);
  }
}

export type DevelopmentProcess = 'linear' | 'aces' | 'agx' | 'filmic';
export type Development = {
  process: DevelopmentProcess;
  exposure: number;
  autoExposure: boolean;
  compensation: number;
  gamma: number;
};
export type DevelopmentInput = false | DevelopmentProcess | Partial<Development>;
export const MAP_DEVELOPMENT: Readonly<Development> = {
  process: 'linear',
  exposure: 0,
  autoExposure: false,
  compensation: 0,
  gamma: 1,
};

/** False selects the fixed map presentation; objects are partial updates. */
export function developmentSettings(
  input: DevelopmentInput,
  previous: Development | false = false,
): Development | false {
  if (input === false) return false;
  if (typeof input !== 'string' && (!input || typeof input !== 'object' || Array.isArray(input)))
    throw new TypeError('Development must be false, a process name or an options object');
  const values = {
    ...MAP_DEVELOPMENT,
    process: 'aces',
    ...previous,
    ...(typeof input === 'string' ? { process: input } : input),
  };
  if (!['linear', 'aces', 'agx', 'filmic'].includes(values.process))
    throw new RangeError('Development process must be linear, aces, agx or filmic');
  for (const [key, min, max] of [
    ['exposure', -12, MAX_EXPOSURE],
    ['compensation', -12, 12],
    ['gamma', 0.1, 4],
  ] as const)
    if (!Number.isFinite(values[key]) || values[key] < min || values[key] > max)
      throw new RangeError(`Development ${key} must be between ${min} and ${max}`);
  if (typeof values.autoExposure !== 'boolean') throw new TypeError('autoExposure must be boolean');
  return {
    process: values.process as DevelopmentProcess,
    exposure: values.exposure,
    autoExposure: values.autoExposure,
    compensation: values.compensation,
    gamma: values.gamma,
  };
}

const developmentProcessIndex = (process: DevelopmentProcess): number =>
  ['linear', 'aces', 'agx', 'filmic'].indexOf(process);

/** Fixed Filmic Blender Base Contrast approximation; no editable curve/LUT. */
function filmicLook(x: number): number {
  x = Math.max(0, Math.min(1, x));
  if (x === 0 || x === 1) return x;
  const t = 2 * x - 1;
  return Math.max(
    0,
    Math.min(
      1,
      (((((((-0.547031202325636 * t + 0.4724197994292706) * t + 1.364337590376409) * t - 0.8878732788888453) * t -
        1.5566985487040412) *
        t +
        0.384378716381371) *
        t +
        1.2005825090543176) *
        t +
        0.5698844146771623) *
        x,
    ),
  );
}
const FILMIC_GREY_GAMMA = Math.log(0.5) / Math.log(filmicLook(10 / 16.5));

const developmentGLSL: string = `
uniform vec2 u_dither_scale;
uniform float u_display_gamma;
uniform int u_development;
float filmicBaseLook(float x) {
  x = clamp(x, 0.0, 1.0);
  if (x <= 0.0 || x >= 1.0) return x;
  float t = 2.0*x - 1.0;
  return clamp((((((((-.547031202325636*t + .4724197994292706)*t
    + 1.364337590376409)*t - .8878732788888453)*t - 1.5566985487040412)*t
    + .384378716381371)*t + 1.2005825090543176)*t + .5698844146771623)*x, 0.0, 1.0);
}
float filmicChannel(float value) {
  if (value <= 0.0) return 0.0;
  float x = (log2(value) - log2(.18) + 10.0) / 16.5;
  float display = pow(filmicBaseLook(x), ${FILMIC_GREY_GAMMA});
  return display <= .04045 ? display / 12.92 : pow((display+.055)/1.055, 2.4);
}
vec3 filmicRGB(vec3 linear) {
  float peak = max(max(linear.r, linear.g), linear.b);
  float desaturation = pow(clamp((peak - 11.52) / 5886.72, 0.0, 1.0), 1.1);
  vec3 color = mix(linear, vec3(peak), desaturation);
  return vec3(filmicChannel(color.r), filmicChannel(color.g), filmicChannel(color.b));
}
// Blender-derived analytic AgX base look; see THIRD-PARTY-NOTICES.
vec3 agxRGB(vec3 color) {
  if (max(max(color.r, color.g), color.b) <= 0.0) return vec3(0);
  const mat3 to2020 = mat3(
    .6274, .0691, .0164, .3293, .9195, .0880, .0433, .0113, .8956);
  const mat3 toSRGB = mat3(
    1.6605, -.1246, -.0182, -.5876, 1.1329, -.1006, -.0728, -.0083, 1.1187);
  const mat3 inset = mat3(
    .856627153315983, .137318972929847, .11189821299995,
    .0951212405381588, .761241990602591, .0767994186031903,
    .0482516061458583, .101439036467562, .811302368396859);
  const mat3 outset = mat3(
    1.1271005818144368, -.1413297634984383, -.14132976349843826,
    -.11060664309660323, 1.157823702216272, -.11060664309660294,
    -.016493938717834573, -.016493938717834257, 1.2519364065950405);
  color = inset * (to2020 * color);
  color = clamp((log2(max(color, vec3(1e-10))) + 12.47393) / 16.499999, 0.0, 1.0);
  vec3 x2 = color * color, x4 = x2 * x2;
  color = 15.5 * x4 * x2 - 40.14 * x4 * color + 31.96 * x4
    - 6.868 * x2 * color + .4298 * x2 + .1191 * color - .00232;
  color = pow(max(outset * color, vec3(0)), vec3(2.2));
  return clamp(toSRGB * color, 0.0, 1.0);
}
// Stephen Hill fitted RRT/ODT with the three.js viewing-environment scale.
vec3 acesRGB(vec3 color) {
  const mat3 inputMatrix = mat3(
    .59719, .07600, .02840, .35458, .90834, .13383, .04823, .01566, .83777);
  const mat3 outputMatrix = mat3(
    1.60475, -.10208, -.00327, -.53108, 1.10813, -.07276, -.07367, -.00605, 1.07602);
  vec3 v = inputMatrix * (color / .6);
  vec3 inverse = 1.0 / max(v, vec3(1));
  vec3 scaled = v * inverse;
  vec3 a = scaled * (scaled + .0245786 * inverse) - .000090537 * inverse * inverse;
  vec3 b = scaled * (.983729 * scaled + .4329510 * inverse) + .238081 * inverse * inverse;
  return clamp(outputMatrix * (a / b), 0.0, 1.0);
}
vec3 displayRadiance(vec3 linear) {
  linear = max(linear * exp2(u_exposure), vec3(0));
  if (u_development == 1) linear = acesRGB(linear);
  else if (u_development == 2) linear = agxRGB(linear);
  else if (u_development == 3) linear = filmicRGB(linear);
  linear = clamp(linear, 0.0, 1.0);
  if (u_display_gamma != 1.0) linear = pow(linear, vec3(1.0 / u_display_gamma));
  vec3 color = mix(1.055 * pow(linear, vec3(1.0 / 2.4)) - 0.055,
                   12.92 * linear, lessThanEqual(linear, vec3(0.0031308)));
  vec2 logicalPixel = floor(gl_FragCoord.xy * u_dither_scale) + .5;
  float noise = fract(52.9829189 * fract(dot(logicalPixel, vec2(.06711056, .00583715)))) - .5;
  vec3 dithered = clamp(color + noise / 255.0, 0.0, 1.0);
  dithered = mix(color, dithered, greaterThan(color, vec3(0)));
  return mix(dithered, color, greaterThanEqual(color, vec3(1)));
}`;

/** Uncompressed scanline OpenEXR, planar FLOAT RGB, linear sRGB primaries.
 * Input is RGBA float32 in WebGL's bottom-up order. No exposure, clamping or
 * display transfer function is applied here. Optional Z is optical-axis metres,
 * bottom-up like RGB; positive infinity denotes sky/no forward surface.
 */
export function encodeEXR(
  width: number,
  height: number,
  rgba: Float32Array,
  metadata: object,
  depth?: Float32Array,
): Uint8Array<ArrayBuffer> {
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width <= 0 ||
    height <= 0 ||
    rgba.length !== width * height * 4 ||
    (depth !== undefined && depth.length !== width * height)
  )
    throw new Error('Invalid EXR image dimensions');
  const text = new TextEncoder();
  const string = (value: string) => text.encode(value + '\0');
  const concat = (...parts: Uint8Array[]) => {
    const result = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
    let offset = 0;
    for (const part of parts) {
      result.set(part, offset);
      offset += part.length;
    }
    return result;
  };
  const ints = (...values: number[]) => {
    const bytes = new Uint8Array(values.length * 4),
      view = new DataView(bytes.buffer);
    values.forEach((value, i) => view.setInt32(i * 4, value, true));
    return bytes;
  };
  const floats = (...values: number[]) => {
    const bytes = new Uint8Array(values.length * 4),
      view = new DataView(bytes.buffer);
    values.forEach((value, i) => view.setFloat32(i * 4, value, true));
    return bytes;
  };
  const attribute = (name: string, type: string, bytes: Uint8Array) =>
    concat(string(name), string(type), ints(bytes.length), bytes);
  const channel = (name: string) => concat(string(name), ints(2), new Uint8Array(4), ints(1, 1));
  const header = concat(
    ints(20000630, 2),
    attribute(
      'channels',
      'chlist',
      concat(channel('B'), channel('G'), channel('R'), ...(depth ? [channel('Z')] : []), new Uint8Array(1)),
    ),
    attribute('compression', 'compression', new Uint8Array([0])),
    attribute('dataWindow', 'box2i', ints(0, 0, width - 1, height - 1)),
    attribute('displayWindow', 'box2i', ints(0, 0, width - 1, height - 1)),
    attribute('lineOrder', 'lineOrder', new Uint8Array([0])),
    attribute('pixelAspectRatio', 'float', floats(1)),
    attribute('screenWindowCenter', 'v2f', floats(0, 0)),
    attribute('screenWindowWidth', 'float', floats(1)),
    attribute('chromaticities', 'chromaticities', floats(0.64, 0.33, 0.3, 0.6, 0.15, 0.06, 0.3127, 0.329)),
    attribute('comments', 'string', text.encode(JSON.stringify(metadata))),
    new Uint8Array(1),
  );
  const rowBytes = width * (depth ? 16 : 12);
  const output = new Uint8Array(header.length + height * 8 + height * (8 + rowBytes));
  output.set(header);
  const view = new DataView(output.buffer);
  let offset = header.length + height * 8;
  for (let y = 0; y < height; y++) {
    view.setBigUint64(header.length + y * 8, BigInt(offset), true);
    view.setInt32(offset, y, true);
    view.setUint32(offset + 4, rowBytes, true);
    offset += 8;
    for (const component of [2, 1, 0])
      for (let x = 0; x < width; x++) {
        const value = rgba[((height - 1 - y) * width + x) * 4 + component];
        if (!Number.isFinite(value)) throw new Error('Nonfinite receiver signal; EXR capture aborted');
        view.setFloat32(offset, value, true);
        offset += 4;
      }
    if (depth)
      for (let x = 0; x < width; x++) {
        const value = depth[(height - 1 - y) * width + x];
        if (!(value > 0)) throw new Error('Invalid EXR depth');
        view.setFloat32(offset, value, true);
        offset += 4;
      }
  }
  return output;
}

type Target = { texture: WebGLTexture; framebuffer: WebGLFramebuffer };

/** Float reception on the logical sensor grid -> preview development.
 * The composite writes one radiance sample per logical (CSS) pixel using the
 * inverse LUT's stencil. Display maps device pixels onto that logical grid. */
export class Receiver {
  readonly raw: Target;
  #display: WebGLProgram;
  #vao: WebGLVertexArrayObject;
  #width = 0;
  #height = 0;

  constructor(gl: WebGL2RenderingContext) {
    const display = compileProgram(gl, displayFragment, vertex);
    let vao: WebGLVertexArrayObject | null = null;
    try {
      vao = gl.createVertexArray();
      if (!vao) throw new Error('Failed to allocate receiver vertex array');
      this.#display = display;
      this.#vao = vao;
      this.raw = this.#target(gl);
    } catch (error) {
      gl.deleteProgram(display);
      if (vao) gl.deleteVertexArray(vao);
      throw error;
    }
  }

  #target(gl: WebGL2RenderingContext): Target {
    const texture = gl.createTexture(),
      framebuffer = gl.createFramebuffer();
    if (!texture || !framebuffer) {
      gl.deleteTexture(texture);
      gl.deleteFramebuffer(framebuffer);
      throw new Error('Failed to allocate receiver target');
    }
    return { texture, framebuffer };
  }

  get width(): number {
    return this.#width;
  }
  get height(): number {
    return this.#height;
  }

  resize(gl: WebGL2RenderingContext, width: number, height: number): void {
    if (width === this.#width && height === this.#height) return;
    const framebuffer = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING);
    const texture = gl.getParameter(gl.TEXTURE_BINDING_2D);
    try {
      gl.bindTexture(gl.TEXTURE_2D, this.raw.texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.raw.framebuffer);
      gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.raw.texture, 0);
      if (gl.checkFramebufferStatus(gl.DRAW_FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error('Incomplete float receiver framebuffer');

      this.#width = width;
      this.#height = height;
    } finally {
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, framebuffer);
      gl.bindTexture(gl.TEXTURE_2D, texture);
    }
  }

  prepareDisplay(
    gl: WebGL2RenderingContext,
    exposure: number,
    opacity: number,
    map: WebGLTexture,
    surfaceRay: WebGLTexture,
    skyOpacity: number,
    development: Development,
    deviceWidth: number,
    deviceHeight: number,
  ): void {
    gl.useProgram(this.#display);
    gl.bindVertexArray(this.#vao);
    for (const [unit, name, texture] of [
      [0, 'u_scene', this.raw.texture],
      [1, 'u_map', map],
      [2, 'u_surface_ray', surfaceRay],
    ] as const) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.uniform1i(uniformLocation(gl, this.#display, name), unit);
    }
    gl.uniform1f(uniformLocation(gl, this.#display, 'u_exposure'), exposure);
    gl.uniform1f(uniformLocation(gl, this.#display, 'u_display_gamma'), development.gamma);
    gl.uniform1i(uniformLocation(gl, this.#display, 'u_development'), developmentProcessIndex(development.process));
    gl.uniform1f(uniformLocation(gl, this.#display, 'u_opacity'), opacity);
    gl.uniform1f(uniformLocation(gl, this.#display, 'u_sky_opacity'), skyOpacity);
    gl.uniform2f(uniformLocation(gl, this.#display, 'u_viewport'), deviceWidth, deviceHeight);
    gl.uniform2f(
      uniformLocation(gl, this.#display, 'u_dither_scale'),
      this.#width / deviceWidth,
      this.#height / deviceHeight,
    );
  }

  draw(gl: WebGL2RenderingContext, metering: boolean): void {
    gl.uniform1i(uniformLocation(gl, this.#display, 'u_metering'), metering ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  dispose(gl: WebGL2RenderingContext): void {
    gl.deleteTexture(this.raw.texture);
    gl.deleteFramebuffer(this.raw.framebuffer);
    gl.deleteProgram(this.#display);
    gl.deleteVertexArray(this.#vao);
  }
}

const vertex = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const displayFragment = `#version 300 es
precision highp float;
// Night RAW reaches ~2^-34; lower-precision samplers can flush it to zero.
precision highp sampler2D;
uniform sampler2D u_scene, u_map, u_surface_ray;
uniform vec2 u_viewport;
uniform float u_exposure, u_opacity, u_sky_opacity;
uniform bool u_metering;
out vec4 fragColor;
${developmentGLSL}
void main() {
  vec2 uv = u_metering ? gl_FragCoord.xy / 16.0 : gl_FragCoord.xy / u_viewport;
  ivec2 size = textureSize(u_scene, 0);
  ivec2 pixel = clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1);
  vec4 scene = texelFetch(u_scene, pixel, 0);
  float rayKind = sign(texelFetch(u_surface_ray, pixel, 0).z);
  float skyOpacity = rayKind == 0.0 ? 0.0 : u_sky_opacity;
  float coverage = rayKind > 0.0 ? 1.0 : skyOpacity;
  if (coverage > 0.0 && coverage < 1.0) scene /= coverage;
  if (u_metering) {
    if (scene.a < 0.0 || coverage <= 0.0) { fragColor = vec4(0); return; }
    float logY = clamp(log2(max(scene.a, exp2(-40.0))), -40.0, 4.0);
    fragColor = vec4(vec3((logY + 40.0) / 44.0), coverage); return;
  }
  float opacity = clamp(u_opacity, 0.0, 1.0);
  if (rayKind < 0.0) {
    // Fade the developed sky to black, never to the underlying map colour.
    vec3 sky = coverage > 0.0 ? displayRadiance(scene.rgb) * (coverage * opacity) : vec3(0);
    fragColor = vec4(sky, 1.0);
    return;
  }
  vec4 map = texelFetch(u_map, ivec2(gl_FragCoord.xy), 0);
  vec3 color = rayKind > 0.0 && scene.a >= -1.5 ? displayRadiance(scene.rgb) : vec3(0);
  fragColor = mix(map, vec4(color, 1.0), opacity);
}`;
