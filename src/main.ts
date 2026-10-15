import type { CustomLayerInterface, CustomRenderMethodInput, Map as MaplibreMap } from 'maplibre-gl';
import { MercatorCoordinate } from 'maplibre-gl';
import { uploadLUT, type LUTMetadata } from './lut.js';
import { AutoExposure, EXPOSURE_METER_MIN_LOG, EXPOSURE_METER_LOG_RANGE, MAX_EXPOSURE } from './exposure.js';
import { celestialRotation, uploadStarfield, type Starfield } from './stars.js';
import { lunarLight } from './moon.js';
import { compileProgram } from './gl-program.js';

type Color3 = [number, number, number];
type Color4 = [number, number, number, number];
type Color = Color3 | Color4;
type Options = {
  date?: Date | null;
  /** Whole-layer opacity, including atmospheric effects. Default: 0.5. */
  opacity?: number;
  /** Night illumination override; alpha controls replacement strength. Default: transparent. */
  color?: Color;
  /** Day illumination override; alpha controls replacement strength. Default: transparent. */
  daytimeColor?: Color;
  twilightSteps?: number;
  /** Gamma of intermediate twilight opacity. Default: 1. Ignored for continuous illumination. */
  twilightAttenuation?: number;
  updateInterval?: number;
  /**
   * Blend from vacuum to atmospheric lighting and transport, from 0 to 1. The Sun remains visible. Default: 0.
   */
  atmosphere?: number;
  /** Exposure adjustment in stops, from -12 to 24. Default: 0. */
  exposure?: number;
  /** Automatically meter scene brightness; exposure becomes a compensation. Default: false. */
  autoExposure?: boolean;
  /** Show the embedded, date-aligned stars and Milky Way. Default: true. */
  stars?: boolean;
  /** Custom LUT ZIP URL. Uses the embedded LUT when omitted. */
  lut?: string | URL;
};
type Target = { texture: WebGLTexture; framebuffer: WebGLFramebuffer };
const radians = (degrees: number) => (degrees * Math.PI) / 180;
const degrees = (radians: number) => (radians * 180) / Math.PI;
const THRESHOLDS = Array.from({ length: 15 }, (_, i) => Math.sin(radians(i === 0 ? -0.8333333 : -6 * i)));

/** Subsolar longitude/latitude; omitted or null date means now. */
export function getSubsolarPoint(date?: Date | null): { lng: number; lat: number } {
  date ??= new Date();
  if (!Number.isFinite(date.getTime())) throw new RangeError('Invalid date');
  const year = date.getUTCFullYear();
  const month = date.getUTCMonth();
  const leapDay = month > 1 && year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0) ? 1 : 0;
  const dayOfYear = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334][month] + date.getUTCDate() + leapDay;
  const UTC =
    date.getUTCHours() + date.getUTCMinutes() / 60 + date.getUTCSeconds() / 3600 + date.getUTCMilliseconds() / 3600000;
  const D = dayOfYear + UTC / 24;
  const n = (2 * Math.PI) / 365.24;
  const e = radians(23.44);
  const E = 0.0167;
  const A = (D + 9) * n;
  const B = A + 2 * E * Math.sin((D - 3) * n);
  const C = (A - Math.atan2(Math.sin(B), Math.cos(B) * Math.cos(e))) / Math.PI;
  const EOT = 720 * (C - Math.trunc(C + 0.5));
  return { lng: -15 * (UTC - 12 + EOT / 60), lat: degrees(Math.asin(Math.sin(-e) * Math.cos(B))) };
}

function premultiply(color: Color4): number[] {
  const [r, g, b, a] = color.map((value) => value / 255);
  return [r ** 2.2 * a, g ** 2.2 * a, b ** 2.2 * a, a];
}

function atmosphereCamera(map: MaplibreMap, options: CustomRenderMethodInput) {
  const projection = options.defaultProjectionData;
  const p = options.projectionMatrix;
  const kilometres = 1 / (MercatorCoordinate.fromLngLat(map.getCenter()).meterInMercatorCoordinateUnits() * 1000);
  const inverse = [
    1 / p[0],
    0,
    0,
    0,
    0,
    1 / p[5],
    0,
    0,
    0,
    0,
    0,
    1 / p[14],
    p[8] / p[0],
    p[9] / p[5],
    -1,
    p[10] / p[14],
  ];
  const transition = Math.max(0, Math.min(1, projection.projectionTransition));
  const camera = (matrix: ArrayLike<number>, mercator: boolean) => {
    const view = Array.from({ length: 16 }, (_, i) =>
      [0, 1, 2, 3].reduce((sum, k) => sum + inverse[k * 4 + (i % 4)] * matrix[Math.floor(i / 4) * 4 + k], 0),
    );
    // Invert the native affine view, preserving its depth-to-distance scale.
    const cross = (a: number[], b: number[]) => [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
    const a = view.slice(0, 3),
      b = view.slice(4, 7),
      c = view.slice(8, 11);
    const rows = [cross(b, c), cross(c, a), cross(a, b)];
    const determinant = a.reduce((sum, n, i) => sum + n * rows[0][i], 0);
    if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-30)
      throw new Error('Invalid atmosphere camera transform');
    const position = rows.map((row) => -row.reduce((sum, n, i) => sum + n * view[12 + i], 0) / determinant);
    const inverseView = Array.from({ length: 9 }, (_, i) => rows[i % 3][Math.floor(i / 3)] / determinant);
    if (mercator) {
      const latitude = Math.atan(Math.sinh(Math.PI * (1 - 2 * position[1]))),
        longitude = (position[0] - 0.5) * 2 * Math.PI;
      const up = [
        Math.sin(longitude) * Math.cos(latitude),
        Math.sin(latitude),
        Math.cos(longitude) * Math.cos(latitude),
      ];
      const origin = up.map((n) => n * (6371 + position[2] * kilometres));
      return {
        origin,
        position,
        inverseView,
      };
    }
    return { origin: position.map((n) => n * 6371), inverseView };
  };
  const mercator = transition < 1 ? camera(projection.fallbackMatrix, true) : undefined;
  const globe = transition > 0 ? camera(projection.mainMatrix, false) : undefined;
  // Keep the physical observer continuous, including Mercator terrain elevation.
  const origin =
    mercator && globe
      ? mercator.origin.map((n, i) => n + (globe.origin[i] - n) * transition)
      : (mercator ?? globe)!.origin;
  let mercatorView = Array(9).fill(0);
  if (mercator) {
    const radius = Math.hypot(...mercator.origin);
    const up = mercator.origin.map((n) => n / radius);
    const longitude = Math.atan2(up[0], up[2]);
    const east = [Math.cos(longitude), 0, -Math.sin(longitude)];
    const north = [up[1] * east[2], up[2] * east[0] - up[0] * east[2], -up[1] * east[0]];
    // Mercator Y points south. Convert the inverse view at the observer, not at
    // each tile endpoint, so sky and terrain share one physical viewing frame.
    const horizontal = 2 * Math.PI * 6371 * Math.hypot(up[0], up[2]);
    mercatorView = Array.from({ length: 9 }, (_, i) => {
      const column = Math.floor(i / 3) * 3,
        row = i % 3;
      return (
        (east[row] * mercator.inverseView[column] - north[row] * mercator.inverseView[column + 1]) * horizontal +
        up[row] * mercator.inverseView[column + 2] * kilometres
      );
    });
  }
  return {
    origin,
    inverseProjection: inverse,
    mercatorView,
    globeView: globe?.inverseView.map((n) => n * 6371) ?? Array(9).fill(0),
    mercatorPosition: mercator?.position ?? [0, 0, 0],
    mercatorKilometres: kilometres,
  };
}

export class NightLayer implements CustomLayerInterface {
  id = 'nightlayer';
  type = 'custom' as const;
  renderingMode = '2d' as const;
  #date: Date | null = null;
  #opacity: number;
  #color: Color4 = [0, 0, 0, 0];
  #daytimeColor: Color4 = [0, 0, 0, 0];
  #twilightSteps: number;
  #twilightAttenuation = 1;
  #updateInterval: number;
  #atmosphere = 0;
  #exposure = 0;
  #autoExposure = false;
  #stars = true;
  #starfield?: Starfield;
  #exposureMeter?: AutoExposure;
  #exposureTimer?: ReturnType<typeof setTimeout>;
  #lut?: string | URL;
  #lutTextures?: [WebGLTexture, WebGLTexture, WebGLTexture, WebGLTexture];
  #lutMetadata?: LUTMetadata;
  #lutTiles?: [number, number, number];
  #mapColor?: Target & { width: number; height: number };
  #timer?: ReturnType<typeof setInterval>;
  #map?: MaplibreMap;
  #attachment?: object;
  #lightingProgram?: WebGLProgram;
  #vertexArray?: WebGLVertexArrayObject;
  #surfaceRenderer?: NativeSurface;
  #skyRenderer?: AtmosphereSurface;
  #groundRenderer?: SeaLevelSurface;
  #surface?: Target & { normals: WebGLTexture; depth: WebGLRenderbuffer; width: number; height: number };
  #surfaceReady = false;
  #surfaceSignature?: string;
  #invalidateSurface = () => {
    this.#surfaceSignature = undefined;
  };

  constructor(opts: Options = {}) {
    this.#opacity = opts.opacity ?? 0.5;
    this.#twilightSteps = opts.twilightSteps ?? 0;
    this.#updateInterval = opts.updateInterval ?? 10000;
    this.#lut = opts.lut;
    this.setAtmosphere(opts.atmosphere ?? 0);
    this.setExposure(opts.exposure ?? 0);
    this.setAutoExposure(opts.autoExposure ?? false);
    this.setStars(opts.stars ?? true);
    this.setTwilightAttenuation(opts.twilightAttenuation ?? 1);
    this.setColor(opts.color ?? [0, 0, 0, 0]);
    this.setDaytimeColor(opts.daytimeColor ?? [0, 0, 0, 0]);
    this.setDate(opts.date ?? null);
  }

  #repaint(): void {
    this.#exposureMeter?.invalidate();
    this.#map?.triggerRepaint();
  }
  #setIntervalTimer(): void {
    if (this.#timer !== undefined) clearInterval(this.#timer);
    this.#timer = undefined;
    if (this.#map && this.#date === null && this.#updateInterval > 0) {
      this.#timer = setInterval(() => {
        this.#repaint();
      }, this.#updateInterval);
    }
  }
  getSubsolarPoint(): { lng: number; lat: number } {
    return getSubsolarPoint(this.#date);
  }
  getDate(): Date | null {
    return this.#date ? new Date(this.#date.getTime()) : null;
  }
  setDate(date: Date | null): void {
    if (date !== null && !Number.isFinite(date.getTime())) throw new RangeError('Invalid date');
    this.#date = date === null ? null : new Date(date.getTime());
    this.#setIntervalTimer();
    this.#repaint();
  }
  getOpacity(): number {
    return this.#opacity;
  }
  setOpacity(opacity: number): void {
    this.#opacity = opacity;
    this.#repaint();
  }
  getColor(): Color4 {
    return this.#color;
  }
  setColor(color: Color): void {
    this.#color = [...(color.slice(0, 3) as Color3), color[3] ?? 255];
    this.#repaint();
  }
  getDaytimeColor(): Color4 {
    return this.#daytimeColor;
  }
  setDaytimeColor(color: Color): void {
    this.#daytimeColor = [...(color.slice(0, 3) as Color3), color[3] ?? 255];
    this.#repaint();
  }
  getTwilightSteps(): number {
    return this.#twilightSteps;
  }
  setTwilightSteps(steps: number): void {
    this.#twilightSteps = steps;
    this.#repaint();
  }
  getTwilightAttenuation(): number {
    return this.#twilightAttenuation;
  }
  setTwilightAttenuation(gamma: number): void {
    if (!Number.isFinite(gamma) || gamma <= 0) throw new RangeError('Twilight attenuation must be finite and positive');
    this.#twilightAttenuation = gamma;
    this.#repaint();
  }
  getUpdateInterval(): number {
    return this.#updateInterval;
  }
  setUpdateInterval(interval: number): void {
    this.#updateInterval = interval;
    this.#setIntervalTimer();
  }
  getAtmosphere(): number {
    return this.#atmosphere;
  }
  setAtmosphere(atmosphere: number): void {
    if (!Number.isFinite(atmosphere) || atmosphere < 0 || atmosphere > 1)
      throw new RangeError('Atmosphere must be between 0 and 1');
    this.#atmosphere = atmosphere;
    this.#repaint();
  }

  getExposure(): number {
    return this.#exposure;
  }
  setExposure(exposure: number): void {
    if (!Number.isFinite(exposure) || exposure < -12 || exposure > MAX_EXPOSURE)
      throw new RangeError(`Exposure must be between -12 and ${MAX_EXPOSURE} stops`);
    this.#exposure = exposure;
    this.#repaint();
  }

  getAutoExposure(): boolean {
    return this.#autoExposure;
  }

  setAutoExposure(enabled: boolean): void {
    this.#autoExposure = enabled;
    if (this.#exposureTimer !== undefined) clearTimeout(this.#exposureTimer);
    this.#exposureTimer = undefined;
    if (!enabled && this.#map) {
      this.#exposureMeter?.dispose(this.#map.painter.context.gl);
      this.#exposureMeter = undefined;
    }
    this.#repaint();
  }

  getStars(): boolean {
    return this.#stars;
  }
  setStars(enabled: boolean): void {
    this.#stars = enabled;
    this.#repaint();
  }

  onAdd(map: MaplibreMap, context: WebGLRenderingContext | WebGL2RenderingContext): void {
    const gl = context as WebGL2RenderingContext;
    if (!gl.getExtension('OES_texture_float_linear')) throw new Error('NightLayer requires OES_texture_float_linear');
    this.#vertexArray = gl.createVertexArray() ?? undefined;
    if (!this.#vertexArray) throw new Error('Failed to create NightLayer vertex array');
    this.#lightingProgram = createProgram(gl, lightingFragment);
    this.#surfaceRenderer = new NativeSurface();
    this.#skyRenderer = new AtmosphereSurface(gl);
    this.#groundRenderer = new SeaLevelSurface(gl);
    this.#map = map;
    const attachment = (this.#attachment = {});
    uploadStarfield(gl)
      .then((starfield) => {
        if (this.#attachment !== attachment) {
          for (const texture of starfield.textures) gl.deleteTexture(texture);
          starfield.points.dispose(gl);
          return;
        }
        this.#starfield = starfield;
        this.#repaint();
      })
      .catch((error) => {
        if (this.#attachment === attachment) map.fire('error', { error });
      })
      .finally(() => map.painter.context.setDirty());
    uploadLUT(gl, this.#lut)
      .then(({ metadata, tiles, textures }) => {
        if (this.#attachment !== attachment) {
          for (const texture of textures) gl.deleteTexture(texture);
          return;
        }
        this.#lutTextures = textures;
        this.#lutMetadata = metadata;
        this.#lutTiles = tiles;
        map.painter.context.setDirty();
        this.#repaint();
      })
      .catch((error) => {
        if (this.#attachment === attachment) map.fire('error', { error });
      })
      .finally(() => map.painter.context.setDirty());
    map.on('sourcedata', this.#invalidateSurface);
    map.on('terrain', this.#invalidateSurface);
    this.#setIntervalTimer();
  }
  #createTexture(gl: WebGL2RenderingContext): WebGLTexture {
    const texture = gl.createTexture();
    if (!texture) throw new Error('Failed to create NightLayer texture');
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }
  #createTarget(gl: WebGL2RenderingContext, width: number, height: number): Target {
    const texture = this.#createTexture(gl);
    const framebuffer = gl.createFramebuffer();
    if (!framebuffer) {
      gl.deleteTexture(texture);
      throw new Error('Failed to create NightLayer framebuffer');
    }
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      gl.deleteTexture(texture);
      gl.deleteFramebuffer(framebuffer);
      throw new Error('Incomplete NightLayer framebuffer');
    }
    return { texture, framebuffer };
  }
  #deleteSurface(gl: WebGL2RenderingContext): void {
    if (!this.#surface) return;
    gl.deleteTexture(this.#surface.texture);
    gl.deleteFramebuffer(this.#surface.framebuffer);
    gl.deleteRenderbuffer(this.#surface.depth);
    gl.deleteTexture(this.#surface.normals);
    this.#surface = undefined;
  }
  #deleteMapColor(gl: WebGL2RenderingContext): void {
    if (!this.#mapColor) return;
    gl.deleteTexture(this.#mapColor.texture);
    gl.deleteFramebuffer(this.#mapColor.framebuffer);
    this.#mapColor = undefined;
  }
  #copyMapColor(gl: WebGL2RenderingContext): void {
    const width = gl.drawingBufferWidth,
      height = gl.drawingBufferHeight;
    const draw = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
    const read = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
    if (!this.#mapColor || this.#mapColor.width !== width || this.#mapColor.height !== height) {
      this.#deleteMapColor(gl);
      const texture = this.#createTexture(gl);
      const framebuffer = gl.createFramebuffer();
      if (!framebuffer) {
        gl.deleteTexture(texture);
        throw new Error('Failed to create NightLayer map colour framebuffer');
      }
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      if (gl.checkFramebufferStatus(gl.DRAW_FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        gl.deleteTexture(texture);
        gl.deleteFramebuffer(framebuffer);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, draw);
        throw new Error('Incomplete NightLayer map colour framebuffer');
      }
      this.#mapColor = { texture, framebuffer, width, height };
    }
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, draw);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.#mapColor.framebuffer);
    gl.blitFramebuffer(0, 0, width, height, 0, 0, width, height, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, read);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, draw);
  }
  onRemove(map: MaplibreMap, context: WebGLRenderingContext | WebGL2RenderingContext): void {
    const gl = context as WebGL2RenderingContext;
    if (this.#timer !== undefined) clearInterval(this.#timer);
    if (this.#exposureTimer !== undefined) clearTimeout(this.#exposureTimer);
    this.#exposureTimer = undefined;
    this.#exposureMeter?.dispose(gl);
    this.#exposureMeter = undefined;
    this.#timer = undefined;
    map.off('sourcedata', this.#invalidateSurface);
    map.off('terrain', this.#invalidateSurface);
    this.#surfaceRenderer?.dispose(gl);
    this.#surfaceRenderer = undefined;
    this.#skyRenderer?.dispose(gl);
    this.#skyRenderer = undefined;
    this.#groundRenderer?.dispose(gl);
    this.#groundRenderer = undefined;
    this.#deleteSurface(gl);
    this.#deleteMapColor(gl);
    for (const texture of this.#lutTextures ?? []) gl.deleteTexture(texture);
    for (const texture of this.#starfield?.textures ?? []) gl.deleteTexture(texture);
    this.#starfield?.points.dispose(gl);
    this.#starfield = undefined;
    if (this.#lightingProgram) gl.deleteProgram(this.#lightingProgram);
    if (this.#vertexArray) gl.deleteVertexArray(this.#vertexArray);
    this.#lutTextures = undefined;
    this.#lutMetadata = undefined;
    this.#lutTiles = undefined;
    this.#attachment = undefined;
    this.#lightingProgram = undefined;
    this.#vertexArray = undefined;
    this.#surfaceReady = false;
    this.#surfaceSignature = undefined;
    this.#map = undefined;
  }
  prerender(gl: WebGL2RenderingContext, options: CustomRenderMethodInput): void {
    if (!this.#map || !this.#surfaceRenderer || this.#opacity <= 0) return;
    const projection = options.defaultProjectionData;
    const signature = JSON.stringify([
      gl.drawingBufferWidth,
      gl.drawingBufferHeight,
      Array.from(projection.mainMatrix),
      Array.from(projection.fallbackMatrix),
      projection.projectionTransition,
      this.#map.getCenter(),
      this.#map.getZoom(),
      this.#map.getPitch(),
      this.#map.getBearing(),
      this.#map.getTerrain(),
      this.#map.getRenderWorldCopies(),
    ]);
    if (this.#surfaceReady && signature === this.#surfaceSignature) return;
    this.#exposureMeter?.invalidate();
    this.#surfaceReady = false;
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('NightLayer requires EXT_color_buffer_float');
    const width = Math.max(1, gl.drawingBufferWidth);
    const height = Math.max(1, gl.drawingBufferHeight);
    if (!this.#surface || this.#surface.width !== width || this.#surface.height !== height) {
      this.#deleteSurface(gl);
      const target = this.#createTarget(gl, width, height);
      const depth = gl.createRenderbuffer();
      if (!depth) throw new Error('Failed to create NightLayer depth buffer');
      gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
      const normals = this.#createTexture(gl);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, normals, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error('Incomplete NightLayer normal framebuffer');
      this.#surface = { ...target, normals, depth, width, height };
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.#surface.framebuffer);
    gl.viewport(0, 0, width, height);
    gl.disable(gl.SCISSOR_TEST);
    gl.colorMask(true, true, true, true);
    gl.depthMask(true);
    gl.clearColor(0, 0, 0, 0);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    this.#skyRenderer!.render(this.#map, gl, options);
    this.#groundRenderer!.render(this.#map, gl, options);
    const terrain = this.#map.getTerrain();
    const source = terrain ? this.#map.getSource(terrain.source) : undefined;
    const tileSize = source && 'tileSize' in source ? (source.tileSize ?? 512) * 2 : 256;
    const tiles = this.#map.coveringTiles({
      tileSize,
      roundZoom: !terrain,
      ...(terrain ? { terrain: this.#map.terrain, minzoom: 0, maxzoom: 22 } : {}),
    });
    this.#surfaceRenderer.render(this.#map, gl, tiles, options);
    this.#surfaceReady = true;
    this.#surfaceSignature = signature;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
  render(context: WebGLRenderingContext | WebGL2RenderingContext, options: CustomRenderMethodInput): void {
    if (!this.#surfaceReady || !this.#lightingProgram || !this.#lutTextures || !this.#starfield || this.#opacity <= 0)
      return;
    const gl = context as WebGL2RenderingContext;
    const program = this.#lightingProgram;
    const date = this.#date ?? new Date();
    const rotation = celestialRotation(date);
    const camera = atmosphereCamera(this.#map!, options);
    const mercatorWeight = Math.max(0, Math.min(1, 1 - options.defaultProjectionData.projectionTransition));
    const starYears = Math.max(-200, Math.min(200, (date.getTime() / 86400000 - 10957.5) / 365.25));
    if (this.#stars) this.#starfield.points.render(gl, camera, rotation, mercatorWeight, starYears);
    const now = performance.now();
    if (this.#autoExposure) {
      this.#exposureMeter ??= new AutoExposure();
      this.#exposureMeter.update(gl, now);
    }
    gl.useProgram(program);
    gl.bindVertexArray(this.#vertexArray!);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.CULL_FACE);
    gl.colorMask(true, true, true, true);
    gl.disable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.uniform1f(gl.getUniformLocation(program, 'u_opacity'), this.#opacity);
    gl.uniform4fv(gl.getUniformLocation(program, 'u_night_color'), premultiply(this.#color));
    gl.uniform4fv(gl.getUniformLocation(program, 'u_daytime_color'), premultiply(this.#daytimeColor));
    gl.uniform1f(gl.getUniformLocation(program, 'u_twilight_steps'), this.#twilightSteps);
    gl.uniform1f(gl.getUniformLocation(program, 'u_twilight_attenuation'), this.#twilightAttenuation);
    const { lng, lat } = getSubsolarPoint(date);
    const moon = lunarLight(date);
    gl.uniform3fv(gl.getUniformLocation(program, 'u_moon_position'), moon.position);
    gl.uniform1f(gl.getUniformLocation(program, 'u_moon_phase'), moon.phaseAngle);
    // Approximate extraterrestrial solar photopic illuminance, shared by lunar
    // and natural-night calibration. LUT RGB is daylight-normalized, not lux.
    gl.uniform1f(gl.getUniformLocation(program, 'u_moon_ratio'), moon.strength / 127500);
    gl.uniformMatrix3fv(gl.getUniformLocation(program, 'u_celestial_rotation'), false, rotation);
    gl.uniform1i(gl.getUniformLocation(program, 'u_stars_enabled'), this.#stars ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(program, 'u_milkyway_scale'), this.#starfield.backgroundScale);
    gl.uniform3fv(gl.getUniformLocation(program, 'u_solar_direction'), [
      Math.sin(radians(lng)) * Math.cos(radians(lat)),
      Math.sin(radians(lat)),
      Math.cos(radians(lng)) * Math.cos(radians(lat)),
    ]);
    gl.uniform1fv(gl.getUniformLocation(program, 'u_thresholds[0]'), THRESHOLDS);
    gl.uniform2f(gl.getUniformLocation(program, 'u_viewport'), gl.drawingBufferWidth, gl.drawingBufferHeight);
    const projectionMatrix = options.projectionMatrix;
    gl.uniform1f(
      gl.getUniformLocation(program, 'u_max_sun_edge'),
      2 / (Math.abs(projectionMatrix[5]) * gl.drawingBufferHeight),
    );
    this.#copyMapColor(gl);
    const layout = this.#lutMetadata!;
    const solarY = layout.solarRadiance.reduce((sum, value, i) => sum + value * [0.2126, 0.7152, 0.0722][i], 0);
    gl.uniform1f(
      gl.getUniformLocation(program, 'u_night_irradiance'),
      // Ground coefficients multiply reflectance: Lambertian radiance is E / pi.
      (solarY * Math.sin(layout.solarAngularRadius) ** 2 * 0.0008) / 127500,
    );
    gl.uniform1f(
      gl.getUniformLocation(program, 'u_star_solar_irradiance'),
      solarY * Math.PI * Math.sin(layout.solarAngularRadius) ** 2,
    );
    gl.uniform1f(gl.getUniformLocation(program, 'u_atmosphere_strength'), this.#atmosphere);
    gl.uniform1f(gl.getUniformLocation(program, 'u_exposure'), this.#exposure + (this.#exposureMeter?.value ?? 0));
    gl.uniform1f(gl.getUniformLocation(program, 'u_sky_reference'), layout.skyReference);
    gl.uniform1f(gl.getUniformLocation(program, 'u_irradiance_height_max'), layout.irradianceHeightMax);
    gl.uniform3fv(gl.getUniformLocation(program, 'u_camera_origin'), camera.origin);
    gl.uniformMatrix4fv(gl.getUniformLocation(program, 'u_inverse_projection'), false, camera.inverseProjection);
    gl.uniformMatrix3fv(gl.getUniformLocation(program, 'u_mercator_view'), false, camera.mercatorView);
    gl.uniformMatrix3fv(gl.getUniformLocation(program, 'u_globe_view'), false, camera.globeView);
    gl.uniform1f(gl.getUniformLocation(program, 'u_mercator_weight'), mercatorWeight);
    gl.uniform1fv(gl.getUniformLocation(program, 'u_direct_counts[0]'), layout.gridSize);
    gl.uniform3fv(gl.getUniformLocation(program, 'u_direct_tiles'), this.#lutTiles!);
    gl.uniform3fv(gl.getUniformLocation(program, 'u_direct_solar_radiance'), layout.solarRadiance);
    gl.uniform1f(gl.getUniformLocation(program, 'u_direct_sun_cos_radius'), Math.cos(layout.solarAngularRadius));
    gl.uniform1f(gl.getUniformLocation(program, 'u_fog_mie_g'), layout.mieG);
    for (const [unit, name, texture] of [
      [1, 'u_surface', this.#surface!.texture],
      [3, 'u_normals', this.#surface!.normals],
      [5, 'u_direct_rayleigh', this.#lutTextures[0]],
      [6, 'u_direct_mie', this.#lutTextures[1]],
      [7, 'u_direct_optical', this.#lutTextures[2]],
      [10, 'u_map_color', this.#mapColor!.texture],
      [11, 'u_irradiance', this.#lutTextures[3]],
      [12, 'u_milkyway', this.#starfield.textures[0]],
      [13, 'u_star_points', this.#starfield.points.texture],
    ] as const) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(unit >= 5 && unit <= 7 ? gl.TEXTURE_3D : gl.TEXTURE_2D, texture ?? null);
      gl.uniform1i(gl.getUniformLocation(program, name), unit);
    }
    const metering = gl.getUniformLocation(program, 'u_metering');
    this.#exposureMeter?.measure(gl, now, () => {
      gl.uniform1i(metering, 1);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    });
    gl.uniform1i(metering, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (this.#exposureTimer !== undefined) clearTimeout(this.#exposureTimer);
    this.#exposureTimer = undefined;
    if (this.#exposureMeter?.needsFrame) {
      this.#exposureTimer = setTimeout(() => {
        this.#exposureTimer = undefined;
        this.#map?.triggerRepaint();
      }, 33);
    }
  }
}
export default NightLayer;

const vertex = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

function createProgram(gl: WebGL2RenderingContext, fragment: string, vertexSource = vertex): WebGLProgram {
  return compileProgram(gl, fragment, vertexSource);
}

const lightingFragment = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler3D;
uniform sampler2D u_surface, u_normals, u_irradiance, u_map_color;
uniform sampler2D u_milkyway, u_star_points;
uniform mat3 u_celestial_rotation;
uniform bool u_stars_enabled;
uniform float u_star_solar_irradiance, u_milkyway_scale;
uniform sampler3D u_direct_rayleigh, u_direct_mie, u_direct_optical;
uniform float u_direct_counts[5], u_direct_sun_cos_radius;
uniform vec3 u_direct_tiles, u_direct_solar_radiance;
uniform float u_atmosphere_strength, u_fog_mie_g, u_sky_reference;
uniform float u_irradiance_height_max;
uniform float u_exposure;
uniform bool u_metering;
uniform float u_max_sun_edge;
uniform vec3 u_camera_origin;
uniform mat4 u_inverse_projection;
uniform mat3 u_mercator_view, u_globe_view;
uniform float u_mercator_weight;
uniform vec2 u_viewport;
uniform vec3 u_solar_direction;
uniform vec3 u_moon_position;
uniform float u_moon_ratio, u_moon_phase, u_night_irradiance;
uniform vec4 u_night_color, u_daytime_color;
uniform float u_opacity, u_twilight_steps, u_twilight_attenuation;
uniform float u_thresholds[15];
out vec4 fragColor;
vec3 displayRadiance(vec3 linear) {
  // Narkowicz ACES filmic fit (not a full ACES color-management pipeline).
  // Values beyond this bound already map to display white; cap before squaring.
  linear = clamp(linear * exp2(u_exposure), 0.0, 64.0);
  linear = clamp((linear * (2.51 * linear + 0.03)) / (linear * (2.43 * linear + 0.59) + 0.14), 0.0, 1.0);
  vec3 color = mix(1.055 * pow(linear, vec3(1.0 / 2.4)) - 0.055,
                   12.92 * linear, lessThanEqual(linear, vec3(0.0031308)));
  // Stable, zero-centred half-LSB dither only at the final display conversion.
  float noise = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) - 0.5;
  vec3 dithered = clamp(color + noise / 255.0, 0.0, 1.0);
  dithered = mix(color, dithered, greaterThan(color, vec3(0.0)));
  return mix(dithered, color, greaterThanEqual(color, vec3(1.0)));
}
float directAxis(float coordinate, int axis) {
  return clamp(coordinate, 0.0, 1.0) * (u_direct_counts[axis] - 1.0);
}
void directTransport(vec3 geometry, vec2 solar, float branch, float airLength, vec2 phases,
                     out vec3 transmission, out vec3 scattering) {
  vec3 p = vec3(directAxis(geometry.x, 0), directAxis(geometry.y, 1), directAxis(geometry.z, 2));
  vec2 angular = vec2(directAxis(solar.x, 3), directAxis(solar.y, 4));
  vec2 counts = vec2(u_direct_counts[3], u_direct_counts[4]);
  vec2 low = min(floor(angular), counts - 2.0), f = angular - low;
  vec3 block = vec3(u_direct_counts[2], u_direct_counts[1], u_direct_counts[0]);
  vec3 tau = texture(u_direct_optical, (vec3(p.z, p.y, branch * block.z + p.x) + 0.5) / (block * vec3(1.0, 1.0, 2.0))).rgb;
  vec3 rayleigh = vec3(0.0), mie = vec3(0.0);
  for (int corner = 0; corner < 4; corner++) {
    vec2 offset = vec2(float(corner & 1), float((corner >> 1) & 1));
    vec2 w = mix(1.0 - f, f, offset), a = low + offset;
    int page = (int(branch) * int(counts.x) + int(a.x)) * int(counts.y) + int(a.y);
    ivec3 tiles = ivec3(u_direct_tiles);
    vec3 tile = vec3(page % tiles.x, (page / tiles.x) % tiles.y, page / (tiles.x * tiles.y));
    vec3 position = (tile * (block + 2.0) + p.zyx + 1.5) / ((block + 2.0) * u_direct_tiles);
    rayleigh += texture(u_direct_rayleigh, position).rgb * w.x * w.y;
    mie += texture(u_direct_mie, position).rgb * w.x * w.y;
  }
  transmission = exp(-tau * airLength);
  scattering = (rayleigh * phases.x + mie * phases.y) * airLength;
}
void transportAngles(vec3 up, vec3 direction, vec3 light, float mu, out vec2 solar, out vec2 phases) {
  float sun = clamp(dot(up, light), -1.0, 1.0);
  float nu = clamp(dot(direction, light), -1.0, 1.0);
  float denominator = sqrt(max(0.0, (1.0 - mu * mu) * (1.0 - sun * sun)));
  float azimuth = denominator > 1e-7 ? acos(clamp((nu - mu * sun) / denominator, -1.0, 1.0)) : 0.0;
  solar = vec2((asinh(sun * sinh(4.0)) / 4.0 + 1.0) * 0.5, azimuth / ${Math.PI});
  float phaseR = 3.0 / (16.0 * ${Math.PI}) * (1.0 + nu * nu);
  float g = u_fog_mie_g;
  float phaseM = 3.0 / (8.0 * ${Math.PI}) * (1.0 - g * g) / (2.0 + g * g)
    * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
  phases = vec2(phaseR, phaseM);
}
void atmosphereRay(vec3 origin, vec3 direction, float distance,
                   out vec3 transmission, out vec3 scattering, out bool occluded) {
  transmission = vec3(1.0); scattering = vec3(0.0); occluded = false;
  float radius = length(origin), t = dot(origin, direction);
  float h = radius - 6371.0, radial = h * (12742.0 + h);
  float topRadial = (100.0 - h) * (12842.0 + h);
  float topDiscriminant = t * t + topRadial;
  if (topDiscriminant < 0.0) return;
  float topRoot = sqrt(topDiscriminant);
  float end = t > 0.0 ? topRadial / max(t + topRoot, 1e-20) : -t + topRoot;
  if (end <= 0.0) return;
  float start = 0.0;
  // Vacuum before entry; below-ground density remains zero.
  if (h > 100.0) start = (radial - 100.0 * 12842.0) / max(-t + topRoot, 1e-20);
  float groundDiscriminant = t * t - radial;
  if (h < 0.0) start = -t + sqrt(max(0.0, groundDiscriminant));
  float groundDistance = 1e30;
  if (h >= 0.0 && t < 0.0 && groundDiscriminant >= 0.0)
    groundDistance = max(0.0, radial / max(-t + sqrt(groundDiscriminant), 1e-20));
  occluded = h >= 0.0 && t < 0.0 && groundDiscriminant >= 0.0 && groundDistance <= distance + 0.001;
  end = min(end, distance);
  end = min(end, groundDistance);
  // Preserve Earth occlusion in vacuum; only transport through air is skipped.
  if (u_atmosphere_strength <= 0.0) return;
  if (end <= start) return;
  origin += direction * start;
  radius = length(origin);
  vec3 up = origin / radius;
  h = clamp(radius - 6371.0, 0.0, 100.0);
  float mu = clamp(dot(up, direction), -1.0, 1.0);
  float horizon = -sqrt(h * (12742.0 + h)) / (6371.0 + h);
  bool ground = mu < horizon;
  float view = sqrt(clamp((mu - horizon) / (ground ? -1.0 - horizon : 1.0 - horizon), 0.0, 1.0));
  t = (6371.0 + h) * mu;
  topRadial = (100.0 - h) * (12842.0 + h);
  topRoot = sqrt(max(0.0, t * t + topRadial));
  float maximum = ground ? h * (12742.0 + h) / max(-t + sqrt(max(0.0, t * t - h * (12742.0 + h))), 1e-20)
    : (t > 0.0 ? topRadial / max(t + topRoot, 1e-20) : -t + topRoot);
  float airLength = min(end - start, maximum);
  if (airLength <= 0.0) return;
  vec3 geometry = vec3(log(1.0 + sqrt(h)) / log(11.0), view, log(1.0 + 255.0 * airLength / max(maximum, 1e-20)) / log(256.0));
  vec2 solar, phases;
  transportAngles(up, direction, u_solar_direction, mu, solar, phases);
  directTransport(geometry, solar, ground ? 0.0 : 1.0, airLength, phases, transmission, scattering);
  if (u_moon_ratio > 1e-12) {
    vec3 moonTransmission, moonScattering;
    transportAngles(up, direction, normalize(u_moon_position - origin), mu, solar, phases);
    directTransport(geometry, solar, ground ? 0.0 : 1.0, airLength, phases, moonTransmission, moonScattering);
    scattering += moonScattering * u_moon_ratio;
  }
}
ivec2 surfacePixel(ivec2 pixel, bool hasMapColor) {
  // Position/normal targets have exactly the framebuffer's resolution.
  // Resolve MSAA coverage once, then use the same texel for both attachments.
  if (texelFetch(u_surface, pixel, 0).a > 0.0 || !hasMapColor) return pixel;
  ivec2 result = pixel;
  float nearest = 1e30;
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    ivec2 offset = ivec2(x, y);
    float distance = dot(vec2(offset), vec2(offset));
    if (distance >= nearest) continue;
    ivec2 candidate = clamp(pixel + offset, ivec2(0), textureSize(u_surface, 0) - 1);
    if (texelFetch(u_surface, candidate, 0).a <= 0.0) continue;
    result = candidate;
    nearest = distance;
  }
  return result;
}
float horizonDip(float height) {
  float h = max(0.0, height) * 0.001;
  return atan(sqrt(h * (12742.0 + h)), 6371.0);
}
void surfaceIrradiance(float altitude, float height, out vec3 sky, out vec3 sun) {
  sky = vec3(0.0); sun = vec3(0.0);
  ivec2 size = textureSize(u_irradiance, 0);
  float h = clamp(height, 0.0, u_irradiance_height_max);
  float row = sqrt(h / u_irradiance_height_max) * float(size.y / 2 - 1);
  altitude += degrees(horizonDip(h));
  int low = 0, high = size.x - 1;
  if (altitude < texelFetch(u_irradiance, ivec2(0, 0), 0).a) return;
  while (high - low > 1) {
    int middle = (low + high) / 2;
    if (altitude < texelFetch(u_irradiance, ivec2(middle, 0), 0).a) high = middle;
    else low = middle;
  }
  vec4 a = texelFetch(u_irradiance, ivec2(low, 0), 0), b = texelFetch(u_irradiance, ivec2(high, 0), 0);
  float fraction = clamp((altitude - a.a) / (b.a - a.a), 0.0, 1.0);
  // Shared solar knots and a square-root height grid allow hardware bilinear interpolation.
  vec2 coordinate = vec2(float(low) + fraction + 0.5, row + 0.5) / vec2(size);
  sky = texture(u_irradiance, coordinate).rgb;
  sun = texture(u_irradiance, coordinate + vec2(0.0, 0.5)).rgb;
}
vec3 surfaceRadiance(vec3 color, vec3 normal, vec3 up, float height) {
  float solarHeight = clamp(dot(up, u_solar_direction), -1.0, 1.0);
  vec3 illumination;
  float weight;
  if (u_twilight_steps > 0.0) {
    // Twilight categories depend only on solar altitude, never on terrain or sky brightness.
    float band = 1.0;
    for (int i = 1; i < 15; i++) if (solarHeight < u_thresholds[i]) band += 1.0;
    band = ceil(min(band, u_twilight_steps));
    if (solarHeight > u_thresholds[0]) band = 0.0;
    weight = 1.0 - pow(clamp(band / u_twilight_steps, 0.0, 1.0), u_twilight_attenuation);
    illumination = vec3(weight);
  } else {
    vec3 sky, sun;
    surfaceIrradiance(degrees(asin(solarHeight)), height, sky, sun);
    float atmosphericDay = clamp(dot(sky, vec3(0.2126, 0.7152, 0.0722)) / u_sky_reference, 0.0, 1.0);
    // Visible fraction of the finite solar disk across the geometric horizon.
    float solarRadius = acos(u_direct_sun_cos_radius);
    float horizonPosition = clamp((asin(solarHeight) + horizonDip(height)) / solarRadius, -1.0, 1.0);
    float visibleSun = (acos(-horizonPosition) + horizonPosition * sqrt(max(0.0, 1.0 - horizonPosition * horizonPosition))) / ${Math.PI};
    vec3 vacuumSun = u_direct_solar_radiance * (${Math.PI} * sin(solarRadius) * sin(solarRadius)) * visibleSun;
    sky *= u_atmosphere_strength;
    sun = mix(vacuumSun, sun, u_atmosphere_strength);
    illumination = sky * clamp(0.5 * (1.0 + dot(normal, up)), 0.0, 1.0)
      + sun * max(0.0, dot(normal, u_solar_direction));
    vec3 moonDirection = normalize(u_moon_position - up * (6371.0 + height * .001));
    float moonAltitude = asin(clamp(dot(up, moonDirection), -1.0, 1.0));
    vec3 moonSky, moonSun;
    surfaceIrradiance(degrees(moonAltitude), height, moonSky, moonSun);
    float moonHorizon = clamp((moonAltitude + horizonDip(height)) / solarRadius, -1.0, 1.0);
    float moonVisible = (acos(-moonHorizon) + moonHorizon * sqrt(max(0.0, 1.0 - moonHorizon * moonHorizon))) / ${Math.PI};
    vec3 vacuumMoon = u_direct_solar_radiance * (sin(solarRadius) * sin(solarRadius)) * moonVisible;
    illumination += u_moon_ratio * (moonSky * u_atmosphere_strength * clamp(.5 * (1.0 + dot(normal, up)), 0.0, 1.0)
      + mix(vacuumMoon, moonSun, u_atmosphere_strength) * max(0.0, dot(normal, moonDirection)));
    // Combined stars/Milky Way/airglow/zodiacal floor: not another copy of each
    // catalogue star. In vacuum retain only a rough extraterrestrial-star share.
    illumination += vec3(u_night_irradiance * mix(.25, 1.0, u_atmosphere_strength))
      * clamp(.5 * (1.0 + dot(normal, up)), 0.0, 1.0);
    weight = mix(visibleSun, atmosphericDay, u_atmosphere_strength);
  }
  float day = weight * u_daytime_color.a;
  float night = (1.0 - weight) * u_night_color.a;
  illumination = illumination * (1.0 - day - night)
    + weight * u_daytime_color.rgb + (1.0 - weight) * u_night_color.rgb;
  return pow(clamp(color, 0.0, 1.0), vec3(2.2)) * illumination;
}
void main() {
  vec2 screen = u_metering ? gl_FragCoord.xy / 16.0 * u_viewport : gl_FragCoord.xy;
  vec4 mapColor = texelFetch(u_map_color, ivec2(screen), 0);
  ivec2 pixel = surfacePixel(ivec2(screen), mapColor.a > 0.0);
  vec4 surface = texelFetch(u_surface, pixel, 0);
  if (surface.a <= 0.0) {
    fragColor = u_metering ? vec4(0.0) : mapColor;
    return;
  }
  vec4 normalHeight = texelFetch(u_normals, pixel, 0);
  vec3 up = normalize(surface.xyz);
  vec2 clip = screen / u_viewport * 2.0 - 1.0;
  vec3 view = (u_inverse_projection * vec4(clip, 0.0, 1.0)).xyz;
  vec3 direction = vec3(0.0);
  float kilometresPerDepth = 0.0;
  if (u_mercator_weight > 0.0) {
    vec3 ray = u_mercator_view * view;
    direction += normalize(ray) * u_mercator_weight;
    kilometresPerDepth += length(ray) * u_mercator_weight;
  }
  if (u_mercator_weight < 1.0) {
    vec3 ray = u_globe_view * view;
    direction += normalize(ray) * (1.0 - u_mercator_weight);
    kilometresPerDepth += length(ray) * (1.0 - u_mercator_weight);
  }
  direction = normalize(direction);
  vec3 rayDx = dFdx(direction), rayDy = dFdy(direction);
  float sunAngle = atan(length(cross(direction, u_solar_direction)), dot(direction, u_solar_direction));
  float radius = acos(u_direct_sun_cos_radius);
  // Derivatives across unrelated terrain/sky endpoints must not spread the disk
  // around the planet's limb. Bound softness by one angular screen pixel.
  float edge = clamp(0.5 * fwidth(sunAngle), 1e-7, max(u_max_sun_edge, 1e-7));
  float sunCoverage = 1.0 - smoothstep(radius - edge, radius + edge, sunAngle);
  vec3 transmission, scattering;
  bool occluded;
  // Keep the sky's viewing direction, but stop at captured terrain depth so
  // mountains hide the air behind them. Alpha stores clip W (view-space depth),
  // not geographic distance; convert it with the same observer-local ray metric.
  // Sky has unbounded depth. The ray function also clips to Earth/atmosphere.
  float distance = surface.a * kilometresPerDepth;
  atmosphereRay(u_camera_origin, direction, distance, transmission, scattering, occluded);
  vec3 linear = vec3(0.0);
  if (mapColor.a > 0.0) {
    vec3 tileColor = clamp(mapColor.rgb / mapColor.a, 0.0, 1.0);
    linear = surfaceRadiance(tileColor, normalize(normalHeight.xyz), up, normalHeight.a) * mapColor.a;
  }
  if (mapColor.a < 1.0 && normalHeight.a < 100000.0 && u_atmosphere_strength > 0.0) {
    vec3 skyTransmission, skyScattering;
    bool skyOccluded;
    // Fill missing ground tiles with the local zenith sky, before viewing-ray transport.
    vec3 skyOrigin = up * (6371.0 + max(0.0, normalHeight.a * 0.001));
    atmosphereRay(skyOrigin, up, 1e30, skyTransmission, skyScattering, skyOccluded);
    linear += skyScattering * u_atmosphere_strength * (1.0 - mapColor.a);
  }
  // A position on the ground is not a solar-disk background, even without tiles.
  if (!occluded && normalHeight.a >= 100000.0) {
    vec3 background = u_direct_solar_radiance * sunCoverage;
    vec3 toMoon = u_moon_position - u_camera_origin;
    vec3 moonDirection = normalize(toMoon);
    float moonRadius = asin(1737.4 / length(toMoon));
    float moonSeparation = length(cross(direction, moonDirection));
    float moonEdge = max(length(rayDx), length(rayDy));
    float moonCoverage = dot(direction, moonDirection) > 0.0
      ? 1.0 - smoothstep(sin(moonRadius) - moonEdge * .5, sin(moonRadius) + moonEdge * .5, moonSeparation) : 0.0;
    if (moonCoverage > 0.0) {
      // Analytic lunar sphere: the visible terminator follows the phase angle;
      // its disk-average flux is normalized to the empirical lunar phase law.
      vec3 sunTangent = u_solar_direction - moonDirection * dot(u_solar_direction, moonDirection);
      if (length(sunTangent) < 1e-6)
        sunTangent = cross(moonDirection, abs(moonDirection.y) < .9 ? vec3(0,1,0) : vec3(1,0,0));
      sunTangent = normalize(sunTangent);
      vec3 side = cross(moonDirection, sunTangent);
      vec2 disk = vec2(dot(direction, sunTangent), dot(direction, side)) / sin(moonRadius);
      vec3 lunarNormal = vec3(disk, sqrt(max(0.0, 1.0 - dot(disk, disk))));
      float lit = max(0.0, dot(lunarNormal, vec3(sin(u_moon_phase), 0.0, cos(u_moon_phase))));
      float lambert = (sin(u_moon_phase) + (${Math.PI} - u_moon_phase) * cos(u_moon_phase)) / ${Math.PI};
      float moonRadiance = u_star_solar_irradiance * u_moon_ratio
        / max(${Math.PI} * sin(moonRadius) * sin(moonRadius), 1e-12);
      background = mix(background, vec3(moonRadiance * lit / max((2.0 / 3.0) * lambert, 1e-6)), moonCoverage);
    }
    if (u_stars_enabled) {
      vec3 celestial = u_celestial_rotation * direction;
      // NASA's map is centered at RA 0h, increasing left; its first row is north.
      vec2 uv = vec2(0.5 - atan(celestial.y, celestial.x) / 6.28318530718,
                     0.5 - asin(clamp(celestial.z, -1.0, 1.0)) / 3.14159265359);
      // Explicit gradients avoid the RA wrap seam and divergent ground/sky derivatives.
      float equatorial = max(dot(celestial.xy, celestial.xy), 1e-8);
      vec3 du = vec3(celestial.y, -celestial.x, 0.0) / (6.28318530718 * equatorial);
      vec3 dv = vec3(0.0, 0.0, -1.0 / (3.14159265359 * sqrt(equatorial)));
      vec3 dx = u_celestial_rotation * rayDx, dy = u_celestial_rotation * rayDy;
      vec3 stars = textureGrad(u_milkyway, uv, vec2(dot(du, dx), dot(dv, dx)), vec2(dot(du, dy), dot(dv, dy))).rgb * u_milkyway_scale;
      // Sparse point sources are deliberately excluded from 16x16 scene metering.
      if (!u_metering) stars += texelFetch(u_star_points, ivec2(screen), 0).rgb
        * 1e-12 / max(length(cross(rayDx, rayDy)), 1e-30);
      background += stars * u_star_solar_irradiance * (1.0 - sunCoverage) * (1.0 - moonCoverage);
    }
    linear += background * (1.0 - mapColor.a);
  }
  linear = linear * mix(vec3(1.0), transmission, u_atmosphere_strength) + scattering * u_atmosphere_strength;
  if (u_metering) {
    // Empty space is not an underexposed subject, even though it has a valid sky endpoint.
    float cameraHeight = length(u_camera_origin) - 6371.0;
    float t = dot(u_camera_origin, direction);
    float radial = (100.0 - cameraHeight) * (12842.0 + cameraHeight);
    if (cameraHeight > 100.0 && (t >= 0.0 || t * t + radial < 0.0)) {
      fragColor = vec4(0.0);
      return;
    }
    float luminance = dot(max(linear, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
    fragColor = vec4(vec3((clamp(log2(max(luminance, exp2(${EXPOSURE_METER_MIN_LOG.toFixed(1)}))), ${EXPOSURE_METER_MIN_LOG.toFixed(1)}, 4.0) - ${EXPOSURE_METER_MIN_LOG.toFixed(1)}) / ${EXPOSURE_METER_LOG_RANGE.toFixed(1)}), 1.0);
    return;
  }
  // Whole-layer opacity is applied once, including coverage, sky and the solar disk.
  vec4 completed = vec4(displayRadiance(linear), 1.0);
  fragColor = mix(mapColor, completed, clamp(u_opacity, 0.0, 1.0));
}`;

const ATMOSPHERE_MESH_STEPS = 16;
const ATMOSPHERE_AZIMUTH_STEPS = 64;

/** Project an infinite sky mesh, but store finite physical atmospheric endpoints. */
class AtmosphereSurface {
  #program: WebGLProgram;
  #vertices: WebGLBuffer;
  #vertexArray: WebGLVertexArrayObject;
  #indices: WebGLBuffer;

  constructor(gl: WebGL2RenderingContext) {
    this.#program = createProgram(
      gl,
      `#version 300 es
      precision highp float;
      in highp vec3 v_direction;
      uniform vec3 u_camera_origin;
      layout(location = 0) out vec4 position;
      layout(location = 1) out vec4 normalHeight;
      void main() {
        vec3 direction = normalize(v_direction);
        float cameraRadius = length(u_camera_origin);
        float height = cameraRadius - 6371.0;
        float t = dot(u_camera_origin, direction);
        float radial = (100.0 - height) * (12842.0 + height);
        float discriminant = t * t + radial;
        float root = sqrt(max(0.0, discriminant));
        float exitDistance = t > 0.0 ? radial / max(t + root, 1e-20) : -t + root;
        bool crossesAtmosphere = discriminant >= 0.0 && exitDistance > 0.0;
        // Space directions also need a valid endpoint, but never acquire air density.
        float spaceDistance = max(1000.0, cameraRadius * 0.1);
        vec3 endpoint = u_camera_origin + direction * (crossesAtmosphere ? exitDistance : spaceDistance);
        float radius = length(endpoint);
        vec3 up = endpoint / radius;
        // Sky has a direction, but no finite surface at which to stop transport.
        position = vec4(up, 1e30);
        normalHeight = vec4(up, crossesAtmosphere ? 100000.0 : (radius - 6371.0) * 1000.0);
      }`,
      `#version 300 es
      precision highp float;
      layout(location = 0) in vec3 a_clip;
      layout(location = 1) in vec3 a_direction;
      out highp vec3 v_direction;
      void main() {
        // CPU projection has already taken the infinite-distance limit (w = 0).
        // Preserve angular coverage without clipping at the map's far plane.
        gl_Position = vec4(a_clip.xy, 0.0, a_clip.z);
        v_direction = a_direction;
      }`,
    );
    const vertices = gl.createBuffer();
    const vertexArray = gl.createVertexArray();
    const indices = gl.createBuffer();
    if (!vertices || !vertexArray || !indices) throw new Error('Failed to allocate NightLayer atmosphere mesh');
    this.#vertices = vertices;
    this.#vertexArray = vertexArray;
    this.#indices = indices;
    gl.bindVertexArray(vertexArray);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertices);
    for (let i = 0; i < 2; i++) {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, 3, gl.FLOAT, false, 24, i * 12);
    }
    const indexData: number[] = [];
    const steps = ATMOSPHERE_MESH_STEPS;
    const azimuthSteps = ATMOSPHERE_AZIMUTH_STEPS;
    for (let y = 0; y < steps; y++)
      for (let x = 0; x < azimuthSteps; x++) {
        const a = y * (azimuthSteps + 1) + x;
        const b = a + 1,
          c = a + azimuthSteps + 1,
          d = c + 1;
        indexData.push(a, b, c, c, b, d);
      }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indices);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indexData), gl.STATIC_DRAW);
  }

  render(map: MaplibreMap, gl: WebGL2RenderingContext, options: CustomRenderMethodInput): void {
    const projection = options.defaultProjectionData;
    const transition = Math.max(0, Math.min(1, projection.projectionTransition));
    const camera = atmosphereCamera(map, options);
    const { origin } = camera;
    const centreLongitude = Math.atan2(origin[0], origin[2]);
    const cameraRadius = Math.hypot(...origin);
    const cameraUp = origin.map((n) => n / cameraRadius);
    const east = [Math.cos(centreLongitude), 0, -Math.sin(centreLongitude)];
    const north = [cameraUp[1] * east[2], cameraUp[2] * east[0] - cameraUp[0] * east[2], -cameraUp[1] * east[0]];
    const horizontalKilometres = 2 * Math.PI * 6371 * Math.max(Math.hypot(cameraUp[0], cameraUp[2]), 1e-7);
    const data: number[] = [];
    const steps = ATMOSPHERE_MESH_STEPS;
    const physicalDirection = (azimuth: number, elevation: number) =>
      cameraUp.map(
        (n, i) =>
          (east[i] * Math.sin(azimuth) + north[i] * Math.cos(azimuth)) * Math.cos(elevation) + n * Math.sin(elevation),
      );
    // Globe surrounds the observer; Mercator has only the upward hemisphere.
    // Corresponding latitude rows keep the topology fixed through projection morphs.
    for (let y = 0; y <= steps; y++) {
      const mercatorElevation = ((y / steps) * Math.PI) / 2;
      const globeElevation = (y / steps) * Math.PI - Math.PI / 2;
      for (let x = 0; x <= ATMOSPHERE_AZIMUTH_STEPS; x++) {
        const azimuth = x === ATMOSPHERE_AZIMUTH_STEPS ? 0 : (x * 2 * Math.PI) / ATMOSPHERE_AZIMUTH_STEPS;
        const globeDirection = physicalDirection(azimuth, globeElevation);
        const mercatorDirection = physicalDirection(azimuth, mercatorElevation);
        const mercator = [
          (Math.sin(azimuth) * Math.cos(mercatorElevation)) / horizontalKilometres,
          (-Math.cos(azimuth) * Math.cos(mercatorElevation)) / horizontalKilometres,
          Math.sin(mercatorElevation) / camera.mercatorKilometres,
        ];
        const globe = globeDirection.map((n) => n / 6371);
        // Project directions with homogeneous w = 0: no altitude, camera
        // translation, or subtraction of large coordinates is involved.
        const clip = [0, 1, 3].map((row) => {
          const multiply = (matrix: ArrayLike<number>, direction: number[]) =>
            matrix[row] * direction[0] + matrix[4 + row] * direction[1] + matrix[8 + row] * direction[2];
          return (
            (transition < 1 ? multiply(projection.fallbackMatrix, mercator) * (1 - transition) : 0) +
            (transition > 0 ? multiply(projection.mainMatrix, globe) * transition : 0)
          );
        });
        const direction = mercatorDirection.map((n, i) => n + (globeDirection[i] - n) * transition);
        data.push(...clip, ...direction);
      }
    }
    gl.useProgram(this.#program);
    gl.bindVertexArray(this.#vertexArray);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.#vertices);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.DYNAMIC_DRAW);
    gl.uniform3fv(gl.getUniformLocation(this.#program, 'u_camera_origin'), origin);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.drawElements(gl.TRIANGLES, steps * ATMOSPHERE_AZIMUTH_STEPS * 6, gl.UNSIGNED_SHORT, 0);
    // Terrain overwrites the positions; its cleared native depth buffer is untouched.
    map.painter.context.setDirty();
  }

  dispose(gl: WebGL2RenderingContext): void {
    gl.deleteProgram(this.#program);
    gl.deleteBuffer(this.#vertices);
    gl.deleteVertexArray(this.#vertexArray);
    gl.deleteBuffer(this.#indices);
  }
}

/** Tile-independent sea level, following the native projection in every mode. */
class SeaLevelSurface {
  #program: WebGLProgram;
  #vertices: WebGLBuffer;
  #vertexArray: WebGLVertexArrayObject;
  #sphere = new NativeSurface();

  constructor(gl: WebGL2RenderingContext) {
    this.#program = createProgram(
      gl,
      `#version 300 es
      precision highp float;
      in highp vec3 v_ground;
      uniform vec2 u_mercator_centre;
      layout(location = 0) out vec4 position;
      layout(location = 1) out vec4 normalHeight;
      void main() {
        vec2 mercator = u_mercator_centre + v_ground.xy / v_ground.z;
        float longitude = (fract(mercator.x) - 0.5) * ${2 * Math.PI};
        // Inverse Mercator remains finite even beyond the native latitude strip.
        // Use a decaying exponential, avoiding sinh overflow toward infinity.
        float angle = ${Math.PI} * (1.0 - 2.0 * mercator.y);
        float e = exp(-abs(angle));
        float latitudeSin = sign(angle) * (1.0 - e * e) / (1.0 + e * e);
        float latitudeCos = 2.0 * e / (1.0 + e * e);
        vec3 up = vec3(sin(longitude) * latitudeCos, latitudeSin, cos(longitude) * latitudeCos);
        // Ground vertices are homogeneous, including points at infinity. Undo
        // their interpolated world W to recover the finite plane's clip depth.
        float depth = v_ground.z > 0.0 ? 1.0 / (gl_FragCoord.w * v_ground.z) : 1e30;
        position = vec4(up, depth);
        normalHeight = vec4(up, 0.0);
      }`,
      `#version 300 es
      precision highp float;
      layout(location = 0) in vec3 a_clip;
      layout(location = 1) in vec3 a_ground;
      out highp vec3 v_ground;
      void main() {
        // The fallback plane must not stop at the native tiles' far clip.
        gl_Position = vec4(a_clip.xy, 0.0, a_clip.z);
        v_ground = a_ground;
      }`,
    );
    const vertices = gl.createBuffer();
    const vertexArray = gl.createVertexArray();
    if (!vertices || !vertexArray) throw new Error('Failed to allocate NightLayer Mercator ground mesh');
    this.#vertices = vertices;
    this.#vertexArray = vertexArray;
    gl.bindVertexArray(vertexArray);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertices);
    for (let i = 0; i < 2; i++) {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, 3, gl.FLOAT, false, 24, i * 12);
    }
  }

  render(map: MaplibreMap, gl: WebGL2RenderingContext, options: CustomRenderMethodInput): void {
    const projection = options.defaultProjectionData;
    const camera = atmosphereCamera(map, options);
    if (projection.projectionTransition > 0) {
      // Geographic coverage does not require any loaded source tiles. Reuse the
      // native world mesh, including its poles and globe-side clipping, at 0 m.
      const tiles = map.coveringTiles({ tileSize: 512, minzoom: 0, maxzoom: 0 });
      this.#sphere.render(map, gl, tiles, options, true);
      return;
    }
    const { mercatorPosition } = camera;
    const [x, y] = mercatorPosition;
    // Relative homogeneous coordinates preserve precision near the camera.
    // Homogeneous vertices at infinity extend sea level beyond both latitude
    // edges without choosing a finite extent. World-copy settings still decide
    // whether longitude is unbounded or restricted to the original world.
    const centre = [0, 0, 1];
    const east = [1, 0, 0],
      west = [-1, 0, 0];
    const north = [0, -1, 0],
      south = [0, 1, 0];
    const ground = map.getRenderWorldCopies()
      ? [centre, east, north, centre, north, west, centre, west, south, centre, south, east]
      : [[-x, 0, 1], [1 - x, 0, 1], north, [-x, 0, 1], south, [1 - x, 0, 1]];
    const matrix = projection.fallbackMatrix;
    const data = ground.flatMap(([dx, dy, w]) => {
      const clip = [0, 1, 3].map(
        (row) =>
          (matrix[row] * x + matrix[4 + row] * y + matrix[12 + row]) * w + matrix[row] * dx + matrix[4 + row] * dy,
      );
      return [...clip, dx, dy, w];
    });
    gl.useProgram(this.#program);
    gl.bindVertexArray(this.#vertexArray);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.#vertices);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.DYNAMIC_DRAW);
    gl.uniform2f(gl.getUniformLocation(this.#program, 'u_mercator_centre'), x % 1, y);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.drawArrays(gl.TRIANGLES, 0, ground.length);
    map.painter.context.setDirty();
  }

  dispose(gl: WebGL2RenderingContext): void {
    this.#sphere.dispose(gl);
    gl.deleteProgram(this.#program);
    gl.deleteBuffer(this.#vertices);
    gl.deleteVertexArray(this.#vertexArray);
  }
}

type Program = ReturnType<MaplibreMap['painter']['programCache']['getProgram']>;
type Terrain = NonNullable<MaplibreMap['terrain']>;
type Mesh = ReturnType<Terrain['getTerrainMesh']>;
type Tile = ReturnType<MaplibreMap['coveringTiles']>[number];

/**
 * MapLibre WebGL adapter. Borrow the native meshes and draw machinery;
 * own only the position-output programs and their VAOs. Tile enumeration stays
 * with map.coveringTiles({terrain}), outside this adapter.
 */
class NativeSurface {
  #terrain?: Terrain;
  #programs = new Map<WebGLProgram, Program>();
  #segments = new Map<Mesh['segments'], Mesh['segments']>();

  #createProgram(gl: WebGL2RenderingContext, native: Program, seaLevel: boolean): Program {
    const shader = gl
      .getAttachedShaders(native.program)
      ?.find((shader) => gl.getShaderParameter(shader, gl.SHADER_TYPE) === gl.VERTEX_SHADER);
    const source = shader && gl.getShaderSource(shader);
    if (!source || !source.includes('projectTile')) {
      throw new Error('NightLayer: unsupported MapLibre terrain vertex shader');
    }
    // Keep the native height sampling, pole handling, skirts and projection
    // intact. The wrapper only adds the geographic direction as a varying.
    const position = source.includes('in vec3 a_pos3d') ? 'a_pos3d.xy' : 'a_pos';
    const globe = /^#define GLOBE\b/m.test(source);
    const vertexSource =
      source.replace(/^(#version[^\n]*\n)/, '$1#define main nightlayerTerrainMain\n') +
      `
      #undef main
      uniform vec4 u_nightlayer_tile;
      out highp vec3 v_nightlayer_direction;
      out highp vec3 v_nightlayer_normal;
      out highp float v_nightlayer_elevation;
      void main() {
        nightlayerTerrainMain();
        ${
          seaLevel
            ? `
        // Globe's Z clips the planet's invisible side and poles during morphing.
        // Flat sea level instead extends past the native tile far plane.
        #ifndef GLOBE
          gl_Position.z = 0.0;
        #endif
        `
            : ''
        }
        v_nightlayer_elevation = ${source.includes('get_elevation(a_pos3d.xy)') ? `get_elevation(${position}) - (a_pos3d.z == 1.0 ? u_ele_delta : 0.0)` : '0.0'};
        #ifdef GLOBE
          v_nightlayer_direction = projectToSphere(${position}, ${position});
        #else
          v_nightlayer_direction = vec3(${position}, 0.0);
        #endif
        vec2 mercator = u_nightlayer_tile.xy + ${position} * u_nightlayer_tile.zw;
        float longitude = mercator.x * ${2 * Math.PI} + ${Math.PI};
        float latitude = atan(sinh(${Math.PI} * (1.0 - 2.0 * mercator.y)));
        vec3 up = vec3(sin(longitude) * cos(latitude), sin(latitude), cos(longitude) * cos(latitude));
        v_nightlayer_normal = up;
        ${
          source.includes('get_elevation(a_pos3d.xy)')
            ? `
        vec2 stepSize = 1.0 / (u_terrain_dim * vec2(u_terrain_matrix[0][0], u_terrain_matrix[1][1]));
        vec2 heightDelta = vec2(
          get_elevation(${position} + vec2(stepSize.x, 0.0)) - get_elevation(${position} - vec2(stepSize.x, 0.0)),
          get_elevation(${position} + vec2(0.0, stepSize.y)) - get_elevation(${position} - vec2(0.0, stepSize.y)));
        vec2 distance = 2.0 * stepSize * u_nightlayer_tile.zw * ${2 * Math.PI * 6371000} * max(cos(latitude), 0.000001);
        vec2 slope = heightDelta / distance;
        vec3 east = vec3(cos(longitude), 0.0, -sin(longitude));
        vec3 north = vec3(-sin(longitude) * sin(latitude), cos(latitude), -cos(longitude) * sin(latitude));
        v_nightlayer_normal = normalize(up - slope.x * east + slope.y * north);
        `
            : ''
        }
      }`;
    const fragmentSource = `#version 300 es
      precision highp float;
      in highp vec3 v_nightlayer_direction;
      in highp vec3 v_nightlayer_normal;
      in highp float v_nightlayer_elevation;
      uniform vec4 u_nightlayer_tile;
      layout(location = 0) out vec4 fragColor;
      layout(location = 1) out vec4 normalColor;
      void main() {
        ${
          globe
            ? 'vec3 direction = v_nightlayer_direction;'
            : `
        // MapLibre's tile scale already includes 1 / EXTENT.
        vec2 mercator = u_nightlayer_tile.xy + v_nightlayer_direction.xy * u_nightlayer_tile.zw;
        float longitude = mercator.x * ${2 * Math.PI} + ${Math.PI};
        float t = exp(${Math.PI} * (1.0 - 2.0 * mercator.y));
        float latitudeSin = (t * t - 1.0) / (t * t + 1.0);
        float latitudeCos = 2.0 * t / (t * t + 1.0);
        vec3 direction = vec3(sin(longitude) * latitudeCos, latitudeSin, cos(longitude) * latitudeCos);`
        }
        vec3 up = normalize(direction);
        fragColor = vec4(up, 1.0 / gl_FragCoord.w);
        normalColor = vec4(${source.includes('get_elevation(a_pos3d.xy)') ? 'normalize(v_nightlayer_normal)' : 'up'}, v_nightlayer_elevation);
      }`;
    const program = gl.createProgram();
    if (!program) throw new Error('NightLayer: failed to allocate native terrain position program');
    const shaders: WebGLShader[] = [];
    try {
      for (const [type, text] of [
        [gl.VERTEX_SHADER, vertexSource],
        [gl.FRAGMENT_SHADER, fragmentSource],
      ] as const) {
        const shader = gl.createShader(type);
        if (!shader) throw new Error('NightLayer: failed to allocate native terrain shader');
        shaders.push(shader);
        gl.shaderSource(shader, text);
        gl.compileShader(shader);
        gl.attachShader(program, shader);
      }
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(
          shaders.map((shader) => gl.getShaderInfoLog(shader)).join('\n') ||
            gl.getProgramInfoLog(program) ||
            'NightLayer: failed to link native terrain position program',
        );
      }
      // Use native block bindings and Program.draw(), so DEM/projection UBOs
      // are populated by MapLibre itself rather than duplicating their layout.
      for (let i = 0; i < gl.getProgramParameter(program, gl.ACTIVE_UNIFORM_BLOCKS); i++) {
        const name = gl.getActiveUniformBlockName(program, i)!;
        const nativeIndex = gl.getUniformBlockIndex(native.program, name);
        if (nativeIndex === gl.INVALID_INDEX) throw new Error(`NightLayer: unsupported native uniform block ${name}`);
        gl.uniformBlockBinding(
          program,
          i,
          gl.getActiveUniformBlockParameter(native.program, nativeIndex, gl.UNIFORM_BLOCK_BINDING),
        );
      }
      const result = Object.create(native) as Program;
      result.program = program;
      const float = gl.getUniformLocation(program, 'u_ele_delta');
      result.fixedUniforms = {};
      if (native.fixedUniforms.u_ele_delta) {
        const elevation = Object.create(native.fixedUniforms.u_ele_delta) as (typeof native.fixedUniforms)[string];
        elevation.set = (value: number) => gl.uniform1f(float, value);
        result.fixedUniforms.u_ele_delta = elevation;
      }
      result.terrainUniforms = { ...native.terrainUniforms };
      for (const name of ['u_depth', 'u_terrain'] as const) {
        const location = gl.getUniformLocation(program, name);
        const uniform = Object.create(native.terrainUniforms[name]) as (typeof native.terrainUniforms)[typeof name];
        uniform.set = (value: number) => gl.uniform1i(location, value);
        result.terrainUniforms[name] = uniform;
      }
      return result;
    } catch (error) {
      gl.deleteProgram(program);
      throw error;
    } finally {
      for (const shader of shaders) {
        gl.deleteShader(shader);
      }
    }
  }

  render(
    map: MaplibreMap,
    gl: WebGL2RenderingContext,
    tiles: Tile[],
    options: CustomRenderMethodInput,
    seaLevel = false,
  ): void {
    const terrain = seaLevel ? undefined : map.terrain;
    const painter = map.painter;
    const projection = map.style.projection;
    if (
      !painter?.programCache?.getProgram ||
      !projection?.getMeshFromTileID ||
      (terrain && (!terrain.getTerrainMesh || !terrain.getTerrainData))
    ) {
      throw new Error('NightLayer: unsupported MapLibre native terrain interface');
    }
    if (this.#terrain && this.#terrain !== terrain) this.dispose(gl);
    this.#terrain = terrain;
    const native = painter.programCache.getProgram({
      name: terrain ? 'terrainDepth' : 'clippingMask',
      projectionShaderVariant: {
        name: projection.shaderVariantName,
        define: projection.shaderDefine,
        prelude: projection.shaderPreludeCode,
      },
      showOverdrawInspector: false,
      useTerrain: !!map.terrain,
      defines: [],
    });
    let program = this.#programs.get(native.program);
    if (!program) {
      program = this.#createProgram(gl, native, seaLevel);
      this.#programs.set(native.program, program);
    }
    const context = painter.context;
    context.setDirty();
    const depth = { func: seaLevel ? gl.ALWAYS : gl.LEQUAL, mask: !seaLevel, range: [0, 1] as [number, number] };
    const stencil: Parameters<Program['draw']>[3] = {
      test: { func: gl.ALWAYS, mask: 0 },
      ref: 0,
      mask: 0,
      fail: gl.KEEP,
      depthFail: gl.KEEP,
      pass: gl.KEEP,
    };
    const colour: Parameters<Program['draw']>[4] = {
      blendFunction: [gl.ONE, gl.ZERO],
      blendColor: context.blendColor.get(),
      mask: [true, true, true, true],
    };
    const cull = { enable: true, mode: gl.BACK, frontFace: gl.CCW };
    for (const tile of tiles) {
      const mesh = terrain
        ? terrain.getTerrainMesh(tile, options.defaultProjectionData.projectionTransition > 0)
        : projection.getMeshFromTileID(context, tile.canonical, false, true, 'raster');
      let segments = this.#segments.get(mesh.segments);
      if (!segments) {
        segments = Object.create(mesh.segments) as Mesh['segments'];
        segments.segments = mesh.segments.get().map((segment) => ({ ...segment, vaos: {} }));
        this.#segments.set(mesh.segments, segments);
      }
      const data = terrain?.getTerrainData(tile) ?? null;
      const tileProjection = options.getProjectionData({
        tileID: tile,
        applyTerrainMatrix: false,
        applyGlobeMatrix: true,
      });
      context.program.set(program.program);
      gl.uniform4fv(gl.getUniformLocation(program.program, 'u_nightlayer_tile'), tileProjection.tileMercatorCoords);
      program.draw(
        context,
        gl.TRIANGLES,
        depth,
        stencil,
        colour,
        cull,
        { u_ele_delta: terrain?.getSkirtLength(map.getZoom()) ?? 0 },
        // Native clipping-mask programs have no terrain block and ignore this argument.
        data!,
        tileProjection,
        'nightlayer-surface',
        mesh.vertexBuffer,
        mesh.indexBuffer,
        segments,
      );
    }
    context.setDirty();
  }

  dispose(gl: WebGL2RenderingContext): void {
    for (const segments of this.#segments.values()) segments.destroy();
    for (const program of this.#programs.values()) gl.deleteProgram(program.program);
    this.#segments.clear();
    this.#programs.clear();
    this.#terrain = undefined;
  }
}
