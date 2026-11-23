import type { CustomLayerInterface, CustomRenderMethodInput, Map as MaplibreMap } from 'maplibre-gl';
import {
  projectionCamera as atmosphereCamera,
  supportsProjection,
  earthIntersectionGLSL,
  inverseLookupGLSL,
  mercatorSphereGLSL,
  inverseSkyGLSL,
  ForwardSurface,
  groundHorizonChordSquared,
  bindRelativePosition,
  relativePositionGLSL,
  TileSurfaceProgram,
  tileVertexPrelude,
  depthSample,
  viewAxis,
  opticalDepthMeters,
  twilightSettings,
  twilightGLSL,
  type WorldCoordinates,
  type DepthSample,
  type Twilight,
  type TwilightInput,
} from './surface.js';
import { getIrradianceAxis, loadTransportLUT, uploadIrradiance, type IrradianceAxis, type LUTMetadata } from './lut.js';
import {
  DirectionTransport,
  hermiteOpticalGLSL,
  hermiteScatteringGLSL,
  MIE_SHADOW_STEP,
  atmosphereFade,
  atmosphereSettings,
  type Atmosphere,
  type AtmosphereInput,
} from './atmosphere.js';
import { celestialRotation, uploadStarfield, lunarLight, type Starfield } from './celestial.js';
import { compileProgram, uniformLocation, SurfaceTargets } from './gl.js';
import {
  focalLength,
  receiverGain,
  SENSOR_PITCH_MM,
  encodeEXR,
  Receiver,
  AutoExposure,
  developmentSettings,
  MAP_DEVELOPMENT,
  type Development,
  type DevelopmentInput,
} from './receiver.js';
import { shadowFragment, shadowVertexGLSL, TerrainShadow, TERRAIN_SHADOW_SIZE, ShadowIntervals } from './shadow.js';
import { ExtrusionRenderer } from './extrusion.js';

export type { DepthSample, Color, Color3, Color4, Twilight, TwilightInput, TwilightSteps } from './surface.js';
export type { Development, DevelopmentInput, DevelopmentProcess } from './receiver.js';
export type { Atmosphere, AtmosphereInput } from './atmosphere.js';

type Options = {
  date?: Date | null;
  /** Whole-layer opacity, including atmospheric effects. Default: 0.5. */
  opacity?: number;
  /** Map-only stepped surface lighting; null/omission keeps continuous lighting.
   * Defaults: black with alpha 0.95, attenuation 0.5. Ignored by camera/physical export. */
  twilight?: TwilightInput;
  updateInterval?: number;
  /**
   * Sky scattering, finite-path fog, and celestial backgrounds/disks. Default: false.
   */
  atmosphere?: AtmosphereInput;
  /** False: fixed linear map presentation. A process string abbreviates an object. */
  development?: DevelopmentInput;
  /** Atmosphere LUT archive URL. Defaults to atmosphere-lut.zip next to the module. */
  lut?: string;
};
type Target = { texture: WebGLTexture; framebuffer: WebGLFramebuffer };
const radians = (degrees: number) => (degrees * Math.PI) / 180;
const degrees = (radians: number) => (radians * 180) / Math.PI;
const THRESHOLDS = Array.from({ length: 4 }, (_, i) => Math.sin(radians(i === 0 ? -0.8333333 : -6 * i)));

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

export class NightLayer implements CustomLayerInterface {
  id = 'nightlayer';
  type = 'custom' as const;
  renderingMode = '2d' as const;
  #date: Date | null = null;
  #opacity: number;
  #twilight: Twilight | null = null;
  #updateInterval: number;
  #atmosphere: Atmosphere = atmosphereSettings(false);
  #shadowSignature?: string;
  #shadowDEMs: object[] = [];
  #terrainShadow?: TerrainShadow;
  #shadowIntervals?: ShadowIntervals;
  #shadowIntervalsReady = false;
  #shadowRenderer?: NativeSurface;
  #extrusions = new ExtrusionRenderer();
  #buildings: ReturnType<ExtrusionRenderer['collect']> = [];
  #development: Development | false = false;
  #capture?: { resolve: (blob: Blob) => void; reject: (error: Error) => void; exposure: 'preview' | 'raw' };
  #receiver?: Receiver;
  #logicalWidth = 0;
  #logicalHeight = 0;
  #starfield?: Starfield;
  #exposureMeter?: AutoExposure;
  #exposureTimer?: ReturnType<typeof setTimeout>;
  #directionTransport?: DirectionTransport;
  #irradianceTexture?: WebGLTexture;
  #lutMetadata?: LUTMetadata;
  #irradianceAxis?: IrradianceAxis;
  #lutSource?: string;
  #lutRequest?: { source?: string; promise: Promise<void> };
  #mapColor?: Target & { width: number; height: number };
  #timer?: ReturnType<typeof setInterval>;
  #map?: MaplibreMap;
  #attachment?: object;
  #lightingProgram?: WebGLProgram;
  #mieOcclusionProgram?: WebGLProgram;
  #mieOcclusion?: Target & { width: number; height: number };
  #vertexArray?: WebGLVertexArrayObject;
  #surfaceRenderer?: NativeSurface;
  #forwardSurface?: ForwardSurface;
  #surface?: SurfaceTargets;
  #viewAxis: number[] = [0, 0, -1];
  #surfaceReady = false;
  #surfaceTiles: Tile[] = [];
  #surfaceSignature?: string;
  #surfaceCoordinates?: WorldCoordinates;
  #invalidateSurface = () => {
    this.#surfaceSignature = undefined;
    // Includes DEM border backfills, which can mutate a loaded DEM in place.
    this.#shadowSignature = undefined;
  };
  #contextLost = () => {
    this.#capture?.reject(new Error('WebGL context lost before EXR capture'));
    this.#capture = undefined;
  };

  constructor(opts: Options = {}) {
    this.#opacity = opts.opacity ?? 0.5;
    this.#updateInterval = opts.updateInterval ?? 10000;
    this.#lutSource = opts.lut;
    this.setAtmosphere(opts.atmosphere ?? false);
    this.setDevelopment(opts.development ?? false);
    this.setTwilight(opts.twilight);
    this.setDate(opts.date ?? null);
  }

  #repaint(): void {
    if (this.#map && !supportsProjection(this.#map)) return;
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
  getTwilight(): Twilight | null {
    return this.#twilight ? { ...this.#twilight, color: [...this.#twilight.color] } : null;
  }
  setTwilight(twilight?: TwilightInput): void {
    this.#twilight = twilightSettings(twilight);
    this.#repaint();
  }
  getUpdateInterval(): number {
    return this.#updateInterval;
  }
  setUpdateInterval(interval: number): void {
    this.#updateInterval = interval;
    this.#setIntervalTimer();
  }
  getAtmosphere(): Atmosphere {
    return { ...this.#atmosphere };
  }
  setAtmosphere(atmosphere: AtmosphereInput): void {
    this.#atmosphere = atmosphereSettings(atmosphere);
    this.#repaint();
  }
  getDevelopment(): Development | false {
    return this.#development && { ...this.#development };
  }
  setDevelopment(input: DevelopmentInput): void {
    const previous = this.#development;
    const next = developmentSettings(input, previous);
    const exposure = next ? next.exposure : 0;
    if (exposure !== (previous ? previous.exposure : 0)) this.#exposureMeter?.setExposure(exposure);
    this.#development = next;
    if (!next || !next.autoExposure) {
      if (this.#exposureTimer !== undefined) clearTimeout(this.#exposureTimer);
      this.#exposureTimer = undefined;
      if (this.#exposureMeter && this.#map) this.#exposureMeter.dispose(this.#map.painter.context.gl);
      this.#exposureMeter = undefined;
    }
    if (!!previous !== !!next) this.#repaint();
    else this.#map?.triggerRepaint();
  }

  /** Last rendered logical pixel at canvas-relative CSS coordinates (top-left).
   * Returns the nearest opaque surface depth and coverage for each pixel.
   * Returns null before a valid frame, outside the canvas, or for unsupported CRS. */
  getDepthAt(x: number, y: number): DepthSample | null {
    if (!this.#map || !supportsProjection(this.#map) || !this.#surfaceReady || !this.#surface) return null;
    const canvas = this.#map.getCanvas();
    if (
      !Number.isFinite(x) ||
      !Number.isFinite(y) ||
      x < 0 ||
      y < 0 ||
      x >= canvas.clientWidth ||
      y >= canvas.clientHeight
    )
      return null;
    const pixelX = Math.floor((x * this.#surface.width) / canvas.clientWidth);
    const pixelY = this.#surface.height - 1 - Math.floor((y * this.#surface.height) / canvas.clientHeight);
    return depthSample(this.#surface.readRays(pixelX, pixelY, 1, 1), this.#viewAxis);
  }

  /** Capture linear FLOAT RGB + Z (axial metres) OpenEXR, without preview tone mapping.
   * Pass { exposure: 'raw' } to preserve the unexposed receiver values. */
  captureEXR(options: { exposure?: 'preview' | 'raw' } = {}): Promise<Blob> {
    if (this.#map && !supportsProjection(this.#map))
      return Promise.reject(new Error('NightLayer: EXR capture is unavailable for CRS projections'));
    if (!this.#map || !this.#lightingProgram || !this.#starfield || !this.#directionTransport)
      return Promise.reject(new Error('NightLayer is not ready for capture'));
    if (this.#capture) return Promise.reject(new Error('An EXR capture is already pending'));
    return new Promise((resolve, reject) => {
      this.#capture = { resolve, reject, exposure: options.exposure ?? 'preview' };
      this.#map!.triggerRepaint();
    });
  }

  /** Replace the LUT after loading succeeds; omit the URL to restore the default atmosphere archive. */
  setLUT(source?: string): Promise<void> {
    const map = this.#map,
      attachment = this.#attachment;
    if (!map || !attachment) {
      this.#lutSource = source;
      return Promise.resolve();
    }
    if (this.#lutRequest && this.#lutRequest.source === source) return this.#lutRequest.promise;
    if (this.#directionTransport && this.#lutSource === source) {
      this.#lutRequest = undefined;
      return Promise.resolve();
    }
    const gl = map.painter.context.gl;
    const request = { source, promise: Promise.resolve() };
    this.#lutRequest = request;
    request.promise = loadTransportLUT(source)
      .then((lut) => {
        if (this.#attachment !== attachment || this.#lutRequest !== request) return;
        const transport = new DirectionTransport(gl, lut);
        let irradiance: WebGLTexture;
        try {
          irradiance = uploadIrradiance(gl, lut);
        } catch (error) {
          transport.dispose();
          throw error;
        }
        this.#directionTransport?.dispose();
        if (this.#irradianceTexture) gl.deleteTexture(this.#irradianceTexture);
        this.#directionTransport = transport;
        this.#irradianceTexture = irradiance;
        this.#lutMetadata = lut.metadata;
        this.#irradianceAxis = getIrradianceAxis(lut);
        this.#lutSource = source;
        this.#repaint();
      })
      .finally(() => {
        if (this.#lutRequest === request) this.#lutRequest = undefined;
        map.painter.context.setDirty();
      });
    return request.promise;
  }

  onAdd(map: MaplibreMap, context: WebGLRenderingContext | WebGL2RenderingContext): void {
    const gl = context as WebGL2RenderingContext;
    if (typeof gl.texImage3D !== 'function') throw new Error('NightLayer requires WebGL 2');
    const missing = ['OES_texture_float_linear', 'EXT_color_buffer_float', 'EXT_float_blend'].filter(
      (name) => !gl.getExtension(name),
    );
    if (missing.length) throw new Error(`NightLayer: missing required WebGL extensions: ${missing.join(', ')}`);
    if (typeof Float16Array === 'undefined') throw new Error('NightLayer requires Float16Array support');
    this.#vertexArray = gl.createVertexArray() ?? undefined;
    if (!this.#vertexArray) throw new Error('Failed to create NightLayer vertex array');
    this.#lightingProgram = createProgram(gl, lightingFragment);
    this.#mieOcclusionProgram = createProgram(gl, mieOcclusionFragment);
    this.#receiver = new Receiver(gl);
    this.#surfaceRenderer = new NativeSurface();
    this.#forwardSurface = new ForwardSurface(gl);
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
    void this.setLUT(this.#lutSource).catch((error) => {
      if (this.#attachment === attachment) map.fire('error', { error });
    });
    map.on('sourcedata', this.#invalidateSurface);
    map.on('terrain', this.#invalidateSurface);
    map.on('webglcontextlost', this.#contextLost);
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
  #deleteSurface(_gl: WebGL2RenderingContext): void {
    if (!this.#surface) return;
    this.#surface.dispose();
    this.#surface = undefined;
  }
  #deleteMapColor(gl: WebGL2RenderingContext): void {
    if (!this.#mapColor) return;
    gl.deleteTexture(this.#mapColor.texture);
    gl.deleteFramebuffer(this.#mapColor.framebuffer);
    this.#mapColor = undefined;
  }
  /** Logical (CSS) pixel grid of the map canvas: the virtual sensor's pixels. */
  #sensorSize(gl: WebGL2RenderingContext): [number, number] {
    const canvas = this.#map!.getCanvas();
    return [
      Math.max(1, canvas.clientWidth || gl.drawingBufferWidth),
      Math.max(1, canvas.clientHeight || gl.drawingBufferHeight),
    ];
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
    this.#capture?.reject(new Error('NightLayer removed before EXR capture'));
    this.#capture = undefined;
    const gl = context as WebGL2RenderingContext;
    if (this.#timer !== undefined) clearInterval(this.#timer);
    if (this.#exposureTimer !== undefined) clearTimeout(this.#exposureTimer);
    this.#exposureTimer = undefined;
    this.#exposureMeter?.dispose(gl);
    this.#exposureMeter = undefined;
    this.#receiver?.dispose(gl);
    this.#receiver = undefined;
    this.#timer = undefined;
    map.off('sourcedata', this.#invalidateSurface);
    map.off('terrain', this.#invalidateSurface);
    map.off('webglcontextlost', this.#contextLost);
    this.#surfaceRenderer?.dispose(gl);
    this.#shadowRenderer?.dispose(gl);
    this.#extrusions.dispose(gl);
    this.#buildings = [];
    this.#terrainShadow?.dispose(gl);
    this.#shadowSignature = undefined;
    this.#shadowDEMs = [];
    this.#shadowRenderer = undefined;
    this.#terrainShadow = undefined;
    this.#shadowIntervals?.dispose(gl);
    this.#shadowIntervals = undefined;
    this.#shadowIntervalsReady = false;
    this.#surfaceRenderer = undefined;
    this.#forwardSurface?.dispose(gl);
    this.#forwardSurface = undefined;
    this.#deleteSurface(gl);
    this.#deleteMapColor(gl);
    this.#directionTransport?.dispose();
    if (this.#irradianceTexture) gl.deleteTexture(this.#irradianceTexture);
    for (const texture of this.#starfield?.textures ?? []) gl.deleteTexture(texture);
    this.#starfield?.points.dispose(gl);
    this.#starfield = undefined;
    if (this.#lightingProgram) gl.deleteProgram(this.#lightingProgram);
    if (this.#mieOcclusionProgram) gl.deleteProgram(this.#mieOcclusionProgram);
    if (this.#mieOcclusion) {
      gl.deleteTexture(this.#mieOcclusion.texture);
      gl.deleteFramebuffer(this.#mieOcclusion.framebuffer);
    }
    if (this.#vertexArray) gl.deleteVertexArray(this.#vertexArray);
    this.#directionTransport = undefined;
    this.#irradianceTexture = undefined;
    this.#lutMetadata = undefined;
    this.#irradianceAxis = undefined;
    this.#lutRequest = undefined;
    this.#attachment = undefined;
    this.#lightingProgram = undefined;
    this.#mieOcclusionProgram = undefined;
    this.#mieOcclusion = undefined;
    this.#vertexArray = undefined;
    this.#surfaceReady = false;
    this.#surfaceSignature = undefined;
    this.#map = undefined;
  }
  prerender(gl: WebGL2RenderingContext, options: CustomRenderMethodInput): void {
    if (this.#map && !supportsProjection(this.#map)) {
      this.#surfaceReady = false;
      this.#surfaceSignature = undefined;
      this.#shadowSignature = undefined;
      this.#capture?.reject(new Error('NightLayer: EXR capture is unavailable for CRS projections'));
      this.#capture = undefined;
      return;
    }
    if (!this.#map || !this.#surfaceRenderer || this.#opacity <= 0) return;
    const projection = options.defaultProjectionData;
    const coordinates = this.#map.style.projection?.worldCoordinateHelper;
    if (!coordinates) throw new Error('NightLayer: geographic projection mapping is unavailable');
    // Only layers already in the native colour at composite time can write endpoints.
    const layerIds = this.#map.getLayersOrder();
    const layerIndex = layerIds.indexOf(this.id);
    const buildings = this.#extrusions.collect(this.#map, layerIds.slice(0, Math.max(0, layerIndex)));
    this.#buildings = buildings;
    const [width, height] = this.#sensorSize(gl);
    const signature = JSON.stringify([
      width,
      height,
      coordinates.name,
      options.shaderData.variantName,
      Array.from(projection.mainMatrix),
      Array.from(projection.fallbackMatrix),
      projection.projectionTransition,
      this.#map.getCenter(),
      this.#map.getZoom(),
      this.#map.getPitch(),
      this.#map.getBearing(),
      this.#map.getTerrain(),
      this.#map.getRenderWorldCopies(),
      buildings.map((building) => building.signature),
    ]);
    if (this.#surfaceReady && signature === this.#surfaceSignature && coordinates === this.#surfaceCoordinates) return;
    if (coordinates !== this.#surfaceCoordinates) this.#shadowSignature = undefined;
    this.#exposureMeter?.invalidate();
    this.#surfaceReady = false;
    if (!this.#surface || this.#surface.width !== width || this.#surface.height !== height) {
      this.#deleteSurface(gl);
      this.#surface = new SurfaceTargets(gl, width, height);
    }
    this.#map.painter.context.setDirty();
    this.#surface.beginLut();
    const camera = atmosphereCamera(this.#map, options);
    this.#forwardSurface!.render(gl, camera, options);
    // Overlay the native map meshes only within the spherical horizon.
    const tiles = this.#map.terrain
      ? this.#map.terrain.tileManager.getRenderableTiles().map((tile) => tile.tileID)
      : this.#map.coveringTiles({ tileSize: 256, roundZoom: true });
    this.#surfaceRenderer.render(this.#map, gl, tiles, options, { depth3D: buildings.length > 0 });
    this.#extrusions.render(this.#map, options, buildings, { camera });
    this.#viewAxis = viewAxis(camera);
    this.#surfaceTiles = tiles;
    this.#surfaceReady = true;
    this.#surfaceSignature = signature;
    this.#surfaceCoordinates = coordinates;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.#map.painter.context.setDirty();
  }
  render(context: WebGLRenderingContext | WebGL2RenderingContext, options: CustomRenderMethodInput): void {
    if (
      !this.#map ||
      !supportsProjection(this.#map) ||
      !this.#surfaceReady ||
      !this.#lightingProgram ||
      !this.#directionTransport ||
      !this.#irradianceTexture ||
      !this.#starfield ||
      (this.#opacity <= 0 && !this.#capture)
    )
      return;
    const gl = context as WebGL2RenderingContext;
    const destination = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
    [this.#logicalWidth, this.#logicalHeight] = this.#sensorSize(gl);
    // The logical receiver resolves the inverse-LUT samples in linear radiance.
    this.#receiver!.resize(gl, this.#logicalWidth, this.#logicalHeight);
    const program = this.#lightingProgram;
    const date = this.#date ?? new Date();
    const rotation = celestialRotation(date);
    const camera = atmosphereCamera(this.#map!, options);
    const fade = atmosphereFade(this.#map.getZoom(), options.defaultProjectionData.projectionTransition);
    const skyStrength = this.#atmosphere.sky * fade;
    const fogStrength = this.#atmosphere.fog * fade;
    const skyOpacity = this.#atmosphere.sky > 0 || this.#atmosphere.stars ? fade : 0;
    const atmosphere = Math.max(skyStrength, fogStrength);
    const development = this.#development || MAP_DEVELOPMENT;
    const { lng, lat } = getSubsolarPoint(date);
    const moon = lunarLight(date);
    const solarDirection = [
      Math.sin(radians(lng)) * Math.cos(radians(lat)),
      Math.sin(radians(lat)),
      Math.cos(radians(lng)) * Math.cos(radians(lat)),
    ];
    // Generate the light planes before shadow/star draws. The GPU generator
    // owns its drawing state; restore the known map target without querying GL.
    if (atmosphere > 0)
      this.#directionTransport.update({
        origin: camera.origin,
        solarDirection,
        moonPosition: moon.position,
        moonRatio: moon.strength / 127500,
      });
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, destination);
    const terrain = this.#map!.terrain;
    // Keep offscreen resident casters when fitting acquisition to the camera
    // cone; no tiles are requested from the Sun's view.
    const loaded =
      fogStrength > 0 && terrain
        ? terrain.tileManager.tileManager
            .getRenderableIds()
            .sort()
            .flatMap((id) => {
              const tile = terrain.tileManager.tileManager.getTileByID(id);
              return tile?.dem ? [tile] : [];
            })
        : [];
    const buildings = this.#buildings;
    const mieShadows = fogStrength > 0 && (loaded.length > 0 || !terrain || buildings.length > 0);
    if (mieShadows) {
      // Existing terrain visibility is a local parallel-beam approximation.
      // Its tangent direction follows the same refractive light inverse as
      // surface illumination; the geographic caster mesh keeps its projection.
      const shadowLight = this.#directionTransport.apparentLight(camera.origin, solarDirection);
      this.#terrainShadow ??= new TerrainShadow(gl);
      this.#shadowRenderer ??= new NativeSurface(true);
      // Fitting and warping depend on actual screen coverage, including view
      // direction. Unchanged views still reuse the caster map and its bounds.
      const signature = JSON.stringify([
        camera.origin.map((v) => v.toFixed(5)),
        camera.inverseProjection,
        camera.planarView,
        camera.sphereView,
        shadowLight,
        this.#map!.getTerrain(),
        this.#surfaceSignature,
        loaded.map((tile) => [tile.tileID.key, tile.dem!.min, tile.dem!.max]),
      ]);
      const dirty =
        signature !== this.#shadowSignature ||
        loaded.length !== this.#shadowDEMs.length ||
        // sourcedata already invalidates in-place DEM/border changes. Native
        // needsTerrainPrepare is an upload flag, not a content revision; it can
        // remain true indefinitely on offscreen tiles not selected as a DEM.
        loaded.some((tile, i) => tile.dem !== this.#shadowDEMs[i]);
      if (dirty) {
        const casterTiles = terrain
          ? terrainShadowTiles(
              this.#surfaceTiles,
              loaded.map((tile) => tile.tileID),
              terrain!.tileManager.deltaZoom,
            )
          : [];
        const terrainMax = loaded.reduce((max, tile) => Math.max(max, tile.dem!.max * terrain!.exaggeration), 0);
        const casterMax = buildings.reduce((max, building) => Math.max(max, building.maxElevation), terrainMax);
        this.#terrainShadow!.begin(gl, camera, shadowLight, casterMax);
        const shadow = this.#terrainShadow!;
        if (terrain) this.#shadowRenderer!.render(this.#map, gl, casterTiles, options, { shadow });
        this.#extrusions.render(this.#map, options, buildings, { shadow });
        this.#terrainShadow!.finish(gl);
        this.#shadowSignature = signature;
        this.#shadowDEMs = loaded.map((tile) => tile.dem!);
        this.#shadowIntervalsReady = false;
      }
      if (!this.#shadowIntervalsReady) {
        this.#shadowIntervals ??= new ShadowIntervals(gl);
        this.#shadowIntervals.update(gl, this.#terrainShadow!);
        this.#shadowIntervalsReady = true;
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, destination);
    }
    const starYears = Math.max(-200, Math.min(200, (date.getTime() / 86400000 - 10957.5) / 365.25));
    if (this.#atmosphere.stars && skyOpacity > 0)
      this.#starfield.points.render(
        gl,
        camera,
        rotation,
        options,
        starYears,
        [this.#logicalWidth, this.#logicalHeight],
        this.#directionTransport,
      );
    // Snapshot native colour before submitting the shadow integration. The copy
    // preserves shared GL state using queries; do not put those behind this draw.
    this.#copyMapColor(gl);
    // Dedicated shadow integration on a decimated logical sensor grid. The
    // composite evaluates its own endpoint Mie and upsamples the removed fraction.
    if (mieShadows) {
      const width = Math.ceil(this.#logicalWidth / MIE_SHADOW_STEP),
        height = Math.ceil(this.#logicalHeight / MIE_SHADOW_STEP);
      if (!this.#mieOcclusion || this.#mieOcclusion.width !== width || this.#mieOcclusion.height !== height) {
        if (this.#mieOcclusion) {
          gl.deleteTexture(this.#mieOcclusion.texture);
          gl.deleteFramebuffer(this.#mieOcclusion.framebuffer);
        }
        this.#mieOcclusion = { ...this.#createTarget(gl, width, height), width, height };
      }
      const shadowProgram = this.#mieOcclusionProgram!;
      gl.useProgram(shadowProgram);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.#mieOcclusion.framebuffer);
      gl.bindVertexArray(this.#vertexArray!);
      gl.viewport(0, 0, width, height);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.STENCIL_TEST);
      gl.disable(gl.SCISSOR_TEST);
      gl.disable(gl.CULL_FACE);
      gl.disable(gl.BLEND);
      gl.colorMask(true, true, true, true);
      gl.uniform3fv(uniformLocation(gl, shadowProgram, 'u_camera_origin'), camera.origin);
      gl.uniform3fv(uniformLocation(gl, shadowProgram, 'u_solar_direction'), solarDirection);
      gl.uniform3fv(uniformLocation(gl, shadowProgram, 'u_moon_position'), moon.position);
      gl.uniform1f(uniformLocation(gl, shadowProgram, 'u_moon_ratio'), 0);
      gl.uniform1f(uniformLocation(gl, shadowProgram, 'u_atmosphere_strength'), atmosphere);
      gl.uniform1f(uniformLocation(gl, shadowProgram, 'u_fog_mie_g'), this.#lutMetadata!.mieG);
      const radius = Math.hypot(...camera.origin);
      gl.uniform1f(uniformLocation(gl, shadowProgram, 'u_ray4d_observer_height'), radius - 6371);
      gl.uniform3fv(
        uniformLocation(gl, shadowProgram, 'u_ray4d_observer_up'),
        camera.origin.map((v) => v / radius),
      );
      gl.uniform1i(uniformLocation(gl, shadowProgram, 'u_terrain_mie_shadows'), 1);
      for (const [unit, name, texture] of [[0, 'u_surface_ray', this.#surface!.rays]] as const) {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.uniform1i(uniformLocation(gl, shadowProgram, name), unit);
      }
      this.#directionTransport.bind(shadowProgram);
      this.#shadowIntervals!.bind(gl, shadowProgram, this.#terrainShadow!);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindFramebuffer(gl.FRAMEBUFFER, destination);
    }
    const now = performance.now();
    if (development.autoExposure && this.#development) {
      this.#exposureMeter ??= new AutoExposure(development.exposure);
      this.#exposureMeter.update(gl, now, 1, development.compensation);
      this.#development.exposure = this.#exposureMeter.value;
    }
    gl.useProgram(program);
    gl.bindVertexArray(this.#vertexArray!);
    gl.viewport(0, 0, this.#logicalWidth, this.#logicalHeight);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.CULL_FACE);
    gl.colorMask(true, true, true, true);
    gl.disable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.uniform1f(uniformLocation(gl, program, 'u_sky_opacity'), skyOpacity);
    gl.uniform4fv(uniformLocation(gl, program, 'u_twilight_color'), this.#twilight?.color ?? [0, 0, 0, 0.95]);
    gl.uniform1f(uniformLocation(gl, program, 'u_twilight_attenuation'), this.#twilight?.attenuation ?? 0.5);
    gl.uniform3fv(uniformLocation(gl, program, 'u_moon_position'), moon.position);
    gl.uniform1f(uniformLocation(gl, program, 'u_moon_phase'), moon.phaseAngle);
    // Approximate extraterrestrial solar photopic illuminance, shared by lunar
    // and natural-night calibration. LUT RGB is daylight-normalized, not lux.
    gl.uniform1f(uniformLocation(gl, program, 'u_moon_ratio'), moon.strength / 127500);
    gl.uniformMatrix3fv(uniformLocation(gl, program, 'u_celestial_rotation'), false, rotation);
    gl.uniform1i(uniformLocation(gl, program, 'u_stars_enabled'), this.#atmosphere.stars ? 1 : 0);
    gl.uniform1i(uniformLocation(gl, program, 'u_disks_enabled'), this.#atmosphere.stars ? 1 : 0);
    gl.uniform1f(uniformLocation(gl, program, 'u_sky_strength'), skyStrength);
    gl.uniform1f(uniformLocation(gl, program, 'u_fog_strength'), fogStrength);
    gl.uniform1f(uniformLocation(gl, program, 'u_milkyway_scale'), this.#starfield.backgroundScale);
    gl.uniform3fv(uniformLocation(gl, program, 'u_solar_direction'), solarDirection);
    gl.uniform1fv(uniformLocation(gl, program, 'u_thresholds[0]'), THRESHOLDS);
    gl.uniform2f(uniformLocation(gl, program, 'u_viewport'), this.#logicalWidth, this.#logicalHeight);
    const layout = this.#lutMetadata!;
    const solarY = layout.solarRadiance.reduce((sum, value, i) => sum + value * [0.2126, 0.7152, 0.0722][i], 0);
    // Ground coefficients multiply reflectance: Lambertian radiance is E / pi.
    const nightIrradiance = (solarY * Math.sin(layout.solarAngularRadius) ** 2 * 0.0008) / 127500;
    gl.uniform1f(uniformLocation(gl, program, 'u_night_irradiance'), nightIrradiance);
    gl.uniform1f(
      uniformLocation(gl, program, 'u_star_solar_irradiance'),
      solarY * Math.PI * Math.sin(layout.solarAngularRadius) ** 2,
    );
    gl.uniform1f(uniformLocation(gl, program, 'u_atmosphere_strength'), atmosphere);
    const fov = this.#map!.getVerticalFieldOfView();
    const sensorHeight = this.#logicalHeight * SENSOR_PITCH_MM;
    gl.uniform1f(uniformLocation(gl, program, 'u_photography_gain'), receiverGain(fov, sensorHeight));
    gl.uniform1f(uniformLocation(gl, program, 'u_irradiance_height_max'), layout.irradianceHeightMax);
    const axis = this.#irradianceAxis!;
    gl.uniform4f(
      uniformLocation(gl, program, 'u_irradiance_axis'),
      axis.first,
      axis.inverseStep,
      axis.count,
      layout.irradianceHeightCount,
    );
    gl.uniform3fv(uniformLocation(gl, program, 'u_camera_origin'), camera.origin);
    const observerRadius = Math.hypot(...camera.origin);
    gl.uniform1f(uniformLocation(gl, program, 'u_ray4d_observer_height'), observerRadius - 6371);
    gl.uniform3fv(
      uniformLocation(gl, program, 'u_ray4d_observer_up'),
      camera.origin.map((v) => v / observerRadius),
    );
    gl.uniform4f(
      uniformLocation(gl, program, 'u_sensor_projection'),
      camera.inverseProjection[0],
      camera.inverseProjection[5],
      camera.inverseProjection[12],
      camera.inverseProjection[13],
    );
    gl.uniform1f(uniformLocation(gl, program, 'u_fog_mie_g'), layout.mieG);
    gl.uniform3fv(uniformLocation(gl, program, 'u_direct_solar_radiance'), layout.solarRadiance);
    gl.uniform1f(uniformLocation(gl, program, 'u_direct_sun_cos_radius'), Math.cos(layout.solarAngularRadius));
    for (const [unit, name, texture] of [
      [0, 'u_surface_ray', this.#surface!.rays],
      [10, 'u_map_color', this.#mapColor!.texture],
      [11, 'u_irradiance', this.#irradianceTexture],
      [12, 'u_milkyway', this.#starfield.textures[0]],
      [13, 'u_star_points', this.#starfield.points.texture],
    ] as const) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, texture ?? null);
      gl.uniform1i(uniformLocation(gl, program, name), unit);
    }
    this.#directionTransport.bind(program);
    gl.uniform1i(uniformLocation(gl, program, 'u_terrain_mie_shadows'), mieShadows ? 1 : 0);
    // A valid 2D sampler is needed even when shadows are disabled.
    gl.activeTexture(gl.TEXTURE14);
    gl.bindTexture(gl.TEXTURE_2D, mieShadows ? this.#mieOcclusion!.texture : this.#mapColor!.texture);
    gl.uniform1i(uniformLocation(gl, program, 'u_shadowed_mie'), 14);
    const artistic = this.#development === false;
    const exportingPhysical = !!this.#capture && artistic;
    const setDisplayGains = (enabled: boolean) => {
      gl.uniform1f(
        uniformLocation(gl, program, 'u_twilight_steps'),
        enabled && artistic ? (this.#twilight?.steps ?? 0) : 0,
      );
      gl.uniform2f(uniformLocation(gl, program, 'u_display_star_limits'), 0.01, 2800);
      gl.uniform1f(uniformLocation(gl, program, 'u_display_star_gain'), enabled && artistic ? 2 ** 20 : 1);
      // Preserve the former atmosphere=0 fill, independently of sky/fog strength.
      gl.uniform1f(
        uniformLocation(gl, program, 'u_display_ambient_irradiance'),
        enabled && artistic ? nightIrradiance * ((2 ** 25 - 1) * 0.25) : 0,
      );
    };
    setDisplayGains(!exportingPhysical);
    const drawReceiver = () => {
      this.#surface!.beginReceiver(this.#receiver!.raw.texture);
      gl.uniform1i(uniformLocation(gl, program, 'u_solar_pass'), 0);
      for (const ground of [false, true]) {
        if (!ground && skyOpacity <= 0) continue;
        this.#surface!.selectGround(ground);
        gl.uniform1i(uniformLocation(gl, program, 'u_ground_pass'), ground ? 1 : 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      if (this.#atmosphere.stars && skyOpacity > 0) {
        this.#surface!.beginSolar();
        gl.uniform1i(uniformLocation(gl, program, 'u_solar_pass'), 1);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        this.#surface!.endSolar();
      }
      gl.uniform1i(uniformLocation(gl, program, 'u_solar_pass'), 0);
      this.#surface!.endReceiver();
    };
    drawReceiver();
    if (this.#capture) this.#captureFrame(gl, date, fov);
    if (exportingPhysical) {
      // Only export frames need a second lighting draw: preserve physical RAW,
      // then restore the artistic receiver image for the normal preview.
      setDisplayGains(true);
      drawReceiver();
    }
    this.#receiver!.prepareDisplay(
      gl,
      development.exposure,
      this.#opacity,
      this.#mapColor!.texture,
      this.#surface!.rays,
      skyOpacity,
      development,
      gl.drawingBufferWidth,
      gl.drawingBufferHeight,
    );
    this.#exposureMeter?.measure(gl, now, () => this.#receiver!.draw(gl, true));
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, destination);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    this.#receiver!.draw(gl, false);
    if (this.#exposureTimer !== undefined) clearTimeout(this.#exposureTimer);
    this.#exposureTimer = undefined;
    if (this.#exposureMeter?.needsFrame) {
      this.#exposureTimer = setTimeout(() => {
        this.#exposureTimer = undefined;
        this.#map?.triggerRepaint();
      }, 33);
    }
  }

  #captureFrame(gl: WebGL2RenderingContext, date: Date, fov: number): void {
    const capture = this.#capture!;
    this.#capture = undefined;
    // EXR pixels are the logical sensor pixels of the receiver.
    const width = this.#receiver!.width,
      height = this.#receiver!.height;
    const read = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING),
      draw = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING);
    const buffer = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING);
    const pack = [gl.PACK_ALIGNMENT, gl.PACK_ROW_LENGTH, gl.PACK_SKIP_PIXELS, gl.PACK_SKIP_ROWS].map(
      (parameter) => [parameter, gl.getParameter(parameter)] as const,
    );
    try {
      if (gl.isContextLost()) throw new Error('WebGL context lost during EXR capture');
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.#receiver!.raw.framebuffer);
      const pixels = new Float32Array(width * height * 4);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      for (const [parameter] of pack) gl.pixelStorei(parameter, parameter === gl.PACK_ALIGNMENT ? 1 : 0);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, pixels);
      if (gl.isContextLost()) throw new Error('WebGL context lost during EXR capture');
      const development = this.#development || MAP_DEVELOPMENT;
      const exposureEV = capture.exposure === 'preview' ? development.exposure : 0;
      const gain = 2 ** exposureEV;
      if (gain !== 1)
        for (let i = 0; i < pixels.length; i += 4) {
          pixels[i] *= gain;
          pixels[i + 1] *= gain;
          pixels[i + 2] *= gain;
        }
      const center = this.#map!.getCenter();
      const rays = this.#surface!.readRays();
      const depth = new Float32Array(width * height);
      for (let i = 0; i < depth.length; i++) depth[i] = opticalDepthMeters(rays, this.#viewAxis, i * 4);
      const exr = encodeEXR(
        width,
        height,
        pixels,
        {
          model: 'NightLayer ideal linear RGB receiver; normalized, not sensor electrons',
          colorSpace: 'linear sRGB / D65',
          exportExposureEV: exposureEV,
          exportExposureMode: capture.exposure,
          sensorHeightMm: this.#logicalHeight * SENSOR_PITCH_MM,
          sensorWidthMm: this.#logicalWidth * SENSOR_PITCH_MM,
          logicalWidth: this.#logicalWidth,
          logicalHeight: this.#logicalHeight,
          sensorPitchMmPerLogicalPixel: SENSOR_PITCH_MM,
          focalLengthMm: focalLength(fov, this.#logicalHeight * SENSOR_PITCH_MM),
          verticalFovDegrees: fov,
          referenceExposure: 'f/4, 1/125 s, ISO 100 at exposure 0 EV',
          date: date.toISOString(),
          center: [center.lng, center.lat],
          zoom: this.#map!.getZoom(),
          pitch: this.#map!.getPitch(),
          bearing: this.#map!.getBearing(),
          roll: this.#map!.getRoll(),
          atmosphere: this.#atmosphere,
          previewExposureStops: development.exposure,
          previewDisplayGamma: development.gamma,
          displayGammaBakedIn: false,
          previewExposureBakedIn: false,
          displayLightingBakedIn: false,
          opacityBakedIn: false,
          depthChannel: 'Z',
          depthUnits: 'metres',
          depthDefinition: 'physical optical-axis distance to the first surface sample',
          depthInfinity: 'positive infinity for sky or no forward surface',
        },
        depth,
      );
      capture.resolve(new Blob([exr], { type: 'image/x-exr' }));
    } catch (error) {
      capture.reject(error instanceof Error ? error : new Error(String(error)));
    } finally {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, read);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, draw);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buffer);
      for (const [parameter, value] of pack) gl.pixelStorei(parameter, value);
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

// The shadow pass uses the same finite-ray setup and cumulative fit as the
// composite. Dead-code elimination drops unused transmission, Rayleigh, phase
// and lunar output. It uses the resolved surface endpoints of every
// MIE_SHADOW_STEP-th logical pixel and
// stores the removed fraction of the solar Mie and its path length (RGBA32F;
// w < 0: nothing occluded). The composite upsamples it bilaterally.
const mieOcclusionFragment = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
${inverseLookupGLSL}
uniform sampler2D u_surface_ray;
uniform vec3 u_camera_origin,u_solar_direction,u_moon_position;
uniform float u_moon_ratio,u_fog_mie_g,u_atmosphere_strength;
${hermiteScatteringGLSL}
${hermiteOpticalGLSL}
out vec4 fragColor;
void main(){
   ivec2 pixel=min(ivec2(gl_FragCoord.xy)*${MIE_SHADOW_STEP},textureSize(u_surface_ray,0)-1);
   vec4 ray=texelFetch(u_surface_ray,pixel,0);
   float distance=ray.z;
  terrainMieFraction=vec3(0);
  terrainMieLength=-1.0;
   if(distance>1e-6){
     vec3 transmission,scattering;
     atmosphereRay(u_camera_origin,decodeDirection(ray.xy),distance,false,transmission,scattering);
  }
  fragColor=vec4(terrainMieFraction,terrainMieLength);
}`;

const lightingFragment = `#version 300 es
#define NIGHTLAYER_COMPOSITE_PASS
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DArray;
${inverseLookupGLSL}
uniform sampler2D u_surface_ray, u_irradiance, u_map_color;
uniform sampler2D u_milkyway, u_star_points;
uniform bool u_ground_pass;
uniform bool u_solar_pass;
uniform mat3 u_celestial_rotation;
uniform bool u_stars_enabled;
uniform bool u_disks_enabled;
uniform float u_sky_strength,u_fog_strength;
uniform float u_star_solar_irradiance, u_milkyway_scale;
uniform float u_display_star_gain, u_display_ambient_irradiance;
uniform vec2 u_display_star_limits;
uniform float u_direct_sun_cos_radius;
uniform vec3 u_direct_solar_radiance;
uniform float u_atmosphere_strength, u_fog_mie_g;
uniform float u_sky_opacity;
uniform float u_irradiance_height_max;
// First lattice altitude (deg), lattices per degree, lattice count, height rows.
uniform vec4 u_irradiance_axis;
uniform float u_photography_gain;
uniform vec3 u_camera_origin;
uniform vec4 u_sensor_projection;
uniform vec2 u_viewport;
uniform vec3 u_solar_direction;
uniform vec3 u_moon_position;
uniform float u_moon_ratio, u_moon_phase, u_night_irradiance;
${twilightGLSL}
layout(location=0) out vec4 fragColor;
${earthIntersectionGLSL}
${inverseSkyGLSL}
${hermiteScatteringGLSL}
${hermiteOpticalGLSL}
float horizonDip(float height) {
  float h = max(0.0, height) * 0.001;
  return atan(sqrt(h * (12742.0 + h)), 6371.0);
}
// Artistic boosts only: return to physical gain in bright surroundings.
// Both the physical-brightness ramp and the added-light fade are linear.
// Stop-valued UI controls only set the maximum multiplier, not the fade shape.
// The thresholds are display policy, not changes to atmospheric extinction.
float darkPreviewGain(float gain, float brightness, float darkLimit, float brightLimit) {
  if (gain <= 1.0) return gain;
  float darkness = 1.0 - clamp((brightness - darkLimit) / (brightLimit - darkLimit), 0.0, 1.0);
  return 1.0 + (gain - 1.0) * darkness;
}
float photopicScale() {
  // Convert daylight-normalized radiance/irradiance into cd/m² or lux.
  return 127500.0 / max(u_star_solar_irradiance, 1e-20);
}
void surfaceIrradiance(float altitude, float height, out vec3 sky, out vec3 sun) {
  sky = vec3(0.0); sun = vec3(0.0);
  ivec2 size = textureSize(u_irradiance, 0);
  float h = clamp(height, 0.0, u_irradiance_height_max);
  int heightCount = int(u_irradiance_axis.w);
  float row = sqrt(h / u_irradiance_height_max) * float(heightCount - 1);
  altitude += degrees(horizonDip(h));
  if (altitude < u_irradiance_axis.x) return;
  // Every non-uniform knot lies on this uniform lattice; the stored fractional
  // knot index is linear between lattice points, so this is exactly the
  // piecewise-linear knot interpolation, without a per-pixel search.
  int count = int(u_irradiance_axis.z);
  float lattice = min((altitude - u_irradiance_axis.x) * u_irradiance_axis.y, float(count - 1));
  int i = min(int(lattice), count - 2);
  int first = 2 * heightCount * size.x + i;
  float a = texelFetch(u_irradiance, ivec2(first % size.x, first / size.x), 0).r;
  float b = texelFetch(u_irradiance, ivec2((first + 1) % size.x, (first + 1) / size.x), 0).r;
  float column = mix(a, b, lattice - float(i));
  // Shared solar knots and a square-root height grid allow hardware bilinear interpolation.
  vec2 coordinate = vec2(column + 0.5, row + 0.5) / vec2(size);
  sky = texture(u_irradiance, coordinate).rgb;
  sun = texture(u_irradiance, coordinate + vec2(0.0, float(heightCount)/float(size.y))).rgb;
}
float ambientIrradiance() {
  // Physical night light retains its atmosphere/vacuum transition. Map-only
  // visibility fill is a separate, fixed addition; physical exports use zero.
  return u_night_irradiance * mix(.25, 1.0, u_atmosphere_strength)
    + u_display_ambient_irradiance;
}
vec3 missingSurfaceRadiance(vec3 up, float height, vec3 towardCamera) {
  if(u_twilight_steps>0.0)
    return twilightIllumination(clamp(dot(up,u_solar_direction),-1.0,1.0));
  // No albedo tile: show the diffuse sky illumination of a horizontal surface,
  // not a second atmospheric path or the direct solar/lunar beam. These LUT
  // coefficients use the same radiance units as the normal surface lighting.
  vec3 sky, unusedSun;
  surfaceIrradiance(degrees(asin(clamp(dot(up, u_solar_direction), -1.0, 1.0))), height, sky, unusedSun);
  vec3 moonDirection = normalize(u_moon_position - up * (6371.0 + height * .001));
  vec3 moonSky;
  surfaceIrradiance(degrees(asin(clamp(dot(up, moonDirection), -1.0, 1.0))), height, moonSky, unusedSun);
  vec3 illumination = (sky + moonSky * u_moon_ratio) * u_atmosphere_strength
    + vec3(ambientIrradiance());
  return illumination * max(0.0, dot(up, towardCamera));
}
vec3 surfaceRadiance(vec3 color, vec3 normal, vec3 up, float height) {
  float solarHeight = clamp(dot(up, u_solar_direction), -1.0, 1.0);
  vec3 illumination;
  if (u_twilight_steps > 0.0) {
    illumination = twilightIllumination(solarHeight);
  } else {
    vec3 sky, sun;
    surfaceIrradiance(degrees(asin(solarHeight)), height, sky, sun);
    // Visible fraction of the finite solar disk across the geometric horizon.
    float solarRadius = acos(u_direct_sun_cos_radius);
    float horizonPosition = clamp((asin(solarHeight) + horizonDip(height)) / solarRadius, -1.0, 1.0);
    float visibleSun = (acos(-horizonPosition) + horizonPosition * sqrt(max(0.0, 1.0 - horizonPosition * horizonPosition))) / ${Math.PI};
    // Irradiance LUTs store E/pi for Lambertian surface radiance. The solar
    // disk's pi*sin(radius)^2 solid-angle factor loses pi in that conversion.
    vec3 vacuumSun = u_direct_solar_radiance * (sin(solarRadius) * sin(solarRadius)) * visibleSun;
    sky *= u_atmosphere_strength;
    sun = mix(vacuumSun, sun, u_atmosphere_strength);
    illumination = sky * clamp(0.5 * (1.0 + dot(normal, up)), 0.0, 1.0)
      + sun * max(0.0, dot(normal, u_atmosphere_strength>0.0
        ?refractionLight(height*.001,up,u_solar_direction):u_solar_direction));
    vec3 moonDirection = normalize(u_moon_position - up * (6371.0 + height * .001));
    float moonAltitude = asin(clamp(dot(up, moonDirection), -1.0, 1.0));
    vec3 moonSky, moonSun;
    surfaceIrradiance(degrees(moonAltitude), height, moonSky, moonSun);
    float moonHorizon = clamp((moonAltitude + horizonDip(height)) / solarRadius, -1.0, 1.0);
    float moonVisible = (acos(-moonHorizon) + moonHorizon * sqrt(max(0.0, 1.0 - moonHorizon * moonHorizon))) / ${Math.PI};
    vec3 vacuumMoon = u_direct_solar_radiance * (sin(solarRadius) * sin(solarRadius)) * moonVisible;
    illumination += u_moon_ratio * (moonSky * u_atmosphere_strength * clamp(.5 * (1.0 + dot(normal, up)), 0.0, 1.0)
      + mix(vacuumMoon, moonSun, u_atmosphere_strength)
        * max(0.0, dot(normal, u_atmosphere_strength>0.0
          ?refractionLight(height*.001,up,moonDirection):moonDirection)));
    // Combined stars/Milky Way/airglow/zodiacal floor: not another copy of each
    // catalogue star. In vacuum retain only a rough extraterrestrial-star share.
    // Map fill is constant through day/twilight/night and independent of sky/fog.
    // Camera/physical export retain only the physical night illumination.
    illumination += vec3(ambientIrradiance())
      * clamp(.5 * (1.0 + dot(normal, up)), 0.0, 1.0);
  }
  return color * illumination;
}
// Native map colour is rendered at device resolution; one logical sensor pixel
// receives the exact area average of the device pixels it covers. Averaging
// is done on adjusted linear premultiplied albedo, since reflected radiance is
// linear in albedo; native colour is kept only for coverage and pass-through.
vec4 logicalMapColor(ivec2 pixel, out vec3 linearAlbedo) {
  vec2 size = vec2(textureSize(u_map_color, 0));
  vec2 scale = size / u_viewport;
  vec2 lo = vec2(pixel) * scale, hi = min(lo + scale, size);
  vec4 color = vec4(0);
  linearAlbedo = vec3(0);
  for (int y = int(floor(lo.y)); y < int(ceil(hi.y)); y++)
    for (int x = int(floor(lo.x)); x < int(ceil(hi.x)); x++) {
      vec2 overlap = max(vec2(0), min(hi, vec2(x, y) + 1.0) - max(lo, vec2(x, y)));
      float weight = overlap.x * overlap.y;
      vec4 value = texelFetch(u_map_color, ivec2(x, y), 0);
      color += value * weight;
      if (value.a > 0.0)
        linearAlbedo += pow(clamp(value.rgb / value.a, 0.0, 1.0), vec3(2.2)) * value.a * weight;
    }
  vec2 area = hi - lo;
  linearAlbedo /= area.x * area.y;
  return color / (area.x * area.y);
}
vec4 shadeSurfaceSample(vec2 screen,vec4 hit,vec3 direction,vec2 dx,vec2 dy) {
  terrainMiePixel = ivec2(screen);
  bool hasSurface = hit.z > 0.0;
  float distance = hasSurface ? hit.z : 1e30;
  vec3 up = vec3(0.0);
  float height = 0.0;
  uint attributes = 0u;
  bool fallback = false;
  if (hasSurface) {
    vec3 endpoint = u_camera_origin + direction * distance;
    float radius = length(endpoint);
    if (radius <= 0.0) return vec4(0.0, 0.0, 0.0, -2.0);
    up = endpoint / radius;
    attributes = uint(hit.w);
    fallback = (attributes >> 20) != 0u;
    height = fallback ? 0.0 : (radius - 6371.0) * 1000.0;
  }
  // The inverse LUT owns coverage. Native colour cannot turn a sky sample into
  // a surface hit, and missing native colour cannot turn a surface hit into sky.
  vec3 mapLinear = vec3(0.0);
  vec4 mapColor = vec4(0.0);
  if (hasSurface && !fallback) mapColor = logicalMapColor(ivec2(screen), mapLinear);
  vec2 clip = screen / u_viewport * 2.0 - 1.0;
  // Optical sensor acceptance is a lens property, independent of terrain slope
  // and map projection. It is not another world-space viewing-direction solve.
  vec3 sensor = vec3(clip * u_sensor_projection.xy + u_sensor_projection.zw, -1.0);
  float angularArea = abs(u_sensor_projection.x * u_sensor_projection.y) / pow(length(sensor), 3.0);
  vec3 transmission, scattering;
  if ((hasSurface ? u_fog_strength : u_sky_strength) > 0.0)
    atmosphereRay(u_camera_origin,direction,distance,!hasSurface,transmission,scattering);
  else {
    transmission=vec3(1);scattering=vec3(0);
    atmosphereBackgroundOrigin=u_camera_origin;atmosphereBackgroundDirection=direction;
    // Sky intensity controls radiance, not the apparent position of stars.
    if(!hasSurface&&u_refraction&&u_stars_enabled)
      refractionExternal(u_camera_origin,max(u_ray4d_observer_height,0.0),direction);
  }
  // Keep the central celestial ray while evaluating its footprint below.
  vec3 backgroundDirection=atmosphereBackgroundDirection;
  vec3 backgroundOrigin=atmosphereBackgroundOrigin;
  vec3 backgroundMoonDirection=normalize(u_moon_position-backgroundOrigin);
  vec3 linear = vec3(0.0);
  vec3 ground = vec3(0.0);
  vec3 pointStars = vec3(0.0);
  if (hasSurface) {
    if (mapColor.a > 0.0)
      ground = surfaceRadiance(mapLinear, unpackNormal(attributes), up, height);
    if (mapColor.a < 1.0)
      ground += missingSurfaceRadiance(up, height, -direction) * (1.0 - mapColor.a);
    ground *= mix(vec3(1.0), transmission, u_fog_strength);
  }
  // A position on the ground is not a solar-disk background, even without tiles.
  if (!hasSurface) {
    mat3 angularBasis = skyBasis(direction);
    vec3 rayDx = angularBasis * vec3(dx, 0.0);
    vec3 rayDy = angularBasis * vec3(dy, 0.0);
    vec3 moonDx=rayDx,moonDy=rayDy;
    if(u_refraction){
      // Carry the sky-only pixel offsets through the same exit map.
      vec3 exitDx=refractionExternal(u_camera_origin,max(u_ray4d_observer_height,0.0),normalize(direction+rayDx));
      vec3 moonDirectionDx=normalize(u_moon_position-atmosphereBackgroundOrigin);
      vec3 exitDy=refractionExternal(u_camera_origin,max(u_ray4d_observer_height,0.0),normalize(direction+rayDy));
      vec3 moonDirectionDy=normalize(u_moon_position-atmosphereBackgroundOrigin);
      rayDx=exitDx-backgroundDirection;rayDy=exitDy-backgroundDirection;
      moonDx=rayDx-(moonDirectionDx-backgroundMoonDirection);
      moonDy=rayDy-(moonDirectionDy-backgroundMoonDirection);
      atmosphereBackgroundOrigin=backgroundOrigin;
      atmosphereBackgroundDirection=backgroundDirection;
    }
    float pixelSolidAngle = length(cross(rayDx, rayDy));
    if(pixelSolidAngle<=1e-20)pixelSolidAngle=4.0*angularArea/(u_viewport.x*u_viewport.y);
    float sunAngle = atan(length(cross(backgroundDirection, u_solar_direction)), dot(backgroundDirection, u_solar_direction));
    float radius = acos(u_direct_sun_cos_radius);
    float edge = max(0.5 * (length(rayDx) + length(rayDy)), 1e-7);
    float sunCoverage = u_disks_enabled ? 1.0 - smoothstep(radius - edge, radius + edge, sunAngle) : 0.0;
    // Direct solar radiance is added by the fractional-coverage pass. Keep the
    // disk coverage here only to suppress stars behind the disk.
    vec3 background = vec3(0);
    vec3 toMoon = u_moon_position - backgroundOrigin;
    vec3 moonDirection = backgroundMoonDirection;
    float moonRadius = asin(1737.4 / length(toMoon));
    float moonSeparation = length(cross(backgroundDirection, moonDirection));
    float moonEdge = u_refraction
      ?max(length(moonDx),length(moonDy)):max(length(rayDx),length(rayDy));
    float moonCoverage = u_disks_enabled && dot(backgroundDirection, moonDirection) > 0.0
      ? 1.0 - smoothstep(sin(moonRadius) - moonEdge * .5, sin(moonRadius) + moonEdge * .5, moonSeparation) : 0.0;
    if (moonCoverage > 0.0) {
      // Analytic lunar sphere: the visible terminator follows the phase angle;
      // its disk-average flux is normalized to the empirical lunar phase law.
      vec3 sunTangent = u_solar_direction - moonDirection * dot(u_solar_direction, moonDirection);
      if (length(sunTangent) < 1e-6)
        sunTangent = cross(moonDirection, abs(moonDirection.y) < .9 ? vec3(0,1,0) : vec3(1,0,0));
      sunTangent = normalize(sunTangent);
      vec3 side = cross(moonDirection, sunTangent);
      vec2 disk = vec2(dot(backgroundDirection, sunTangent), dot(backgroundDirection, side)) / sin(moonRadius);
      vec3 lunarNormal = vec3(disk, sqrt(max(0.0, 1.0 - dot(disk, disk))));
      float lit = max(0.0, dot(lunarNormal, vec3(sin(u_moon_phase), 0.0, cos(u_moon_phase))));
      float lambert = (sin(u_moon_phase) + (${Math.PI} - u_moon_phase) * cos(u_moon_phase)) / ${Math.PI};
      float moonRadiance = u_star_solar_irradiance * u_moon_ratio
        / max(${Math.PI} * sin(moonRadius) * sin(moonRadius), 1e-12);
      background = mix(background, vec3(moonRadiance * lit / max((2.0 / 3.0) * lambert, 1e-6)), moonCoverage);
    }
    if (u_stars_enabled) {
      // Local physical sky background, before reception/exposure/artistic gains.
      // A ray missing the atmosphere has zero background even beside a bright Earth.
      float skyLuminance = dot(max(scattering * u_sky_strength, vec3(0)), vec3(.2126,.7152,.0722)) * photopicScale();
      float starGain = darkPreviewGain(u_display_star_gain, skyLuminance, u_display_star_limits.x, u_display_star_limits.y);
      vec3 celestial = u_celestial_rotation * backgroundDirection;
      // NASA's map is centered at RA 0h, increasing left; its first row is north.
      vec2 uv = vec2(0.5 - atan(celestial.y, celestial.x) / 6.28318530718,
                     0.5 - asin(clamp(celestial.z, -1.0, 1.0)) / 3.14159265359);
      // Explicit gradients avoid the RA wrap seam and divergent ground/sky derivatives.
      float equatorial = max(dot(celestial.xy, celestial.xy), 1e-8);
      vec3 du = vec3(celestial.y, -celestial.x, 0.0) / (6.28318530718 * equatorial);
      vec3 dv = vec3(0.0, 0.0, -1.0 / (3.14159265359 * sqrt(equatorial)));
      vec3 dx = u_celestial_rotation * rayDx, dy = u_celestial_rotation * rayDy;
      vec3 stars = textureGrad(u_milkyway, uv, vec2(dot(du, dx), dot(dv, dx)), vec2(dot(du, dy), dot(dv, dy))).rgb * u_milkyway_scale * starGain;
      // Sparse point sources are deliberately excluded from 16x16 scene metering.
        vec3 points = texelFetch(u_star_points, ivec2(screen), 0).rgb * 1e-12 / pixelSolidAngle;
        pointStars = points * u_star_solar_irradiance * starGain * (1.0 - sunCoverage) * (1.0 - moonCoverage) * (1.0 - mapColor.a);
      background += stars * u_star_solar_irradiance * (1.0 - sunCoverage) * (1.0 - moonCoverage);
    }
    float mediumIndex=u_refraction?1.0+.00029*
      (exp(-clamp(u_ray4d_observer_height,0.0,100.0)/8.0)-exp(-12.5)):1.0;
    linear += background * (mediumIndex*mediumIndex) * (1.0 - mapColor.a);
    pointStars*=mediumIndex*mediumIndex;
  }
  // Ground already includes its own geographic path's extinction. Only sky
  // backgrounds use the screen ray's transmission; do not attenuate ground twice.
  linear = ground + linear * mix(vec3(1.0), transmission, u_sky_strength)
    + scattering * (hasSurface ? u_fog_strength : u_sky_strength);
  // Ideal receiver: fixed reference exposure sets collecting area, uniform directional
  // acceptance, and a fixed-height virtual sensor. Sensor height = 2 units,
  // pixel area = 4 / viewportHeight^2. Reference area = 9 units^2 calibrates
  // the on-axis response to 1 at the default FOV (2*atan(1/3)). Exposure
  // is controlled by a single EV gain during preview development.
  // This is not a finite-aperture lens simulation.
  // A * solidAngle / pixelArea = 9 * angularArea * viewportHeight/viewportWidth.
  // Apply to ALL transported radiance, before both metering and development.
  float reception = 9.0 * angularArea * (u_viewport.y / u_viewport.x) * u_photography_gain;
  linear *= reception;
  vec3 withoutPoints = linear;
  linear += pointStars * mix(vec3(1.0), transmission, u_sky_strength) * reception;
    // Alpha is metering luminance, excluding sparse point sources, not opacity.
    float meterY = dot(max(withoutPoints, vec3(0)), vec3(.2126,.7152,.0722));
    float cameraHeight = length(u_camera_origin) - 6371.0;
    float t = dot(u_camera_origin, direction);
    float radial = (100.0 - cameraHeight) * (12842.0 + cameraHeight);
    if (!hasSurface && cameraHeight > 100.0 && (t >= 0.0 || t * t + radial < 0.0))
      meterY=-1.0;
    // Display restores sky radiance before metering and development.
    return vec4(linear, meterY)*(hasSurface?1.0:u_sky_opacity);
}
// Never difference the sky ray against an adjacent ground endpoint.
vec3 skyRayStep(ivec2 pixel,ivec2 axis,vec3 center){
  ivec2 size=textureSize(u_surface_ray,0);
  ivec2 hi=min(pixel+axis,size-1),lo=max(pixel-axis,ivec2(0));
  vec4 a=texelFetch(u_surface_ray,lo,0),b=texelFetch(u_surface_ray,hi,0);
  bool left=a.z<0.0&&any(notEqual(lo,pixel)),right=b.z<0.0&&any(notEqual(hi,pixel));
  if(left&&right)return .5*(decodeDirection(b.xy)-decodeDirection(a.xy));
  if(left)return center-decodeDirection(a.xy);
  if(right)return decodeDirection(b.xy)-center;
  return vec3(0);
}
void shadeSolarDisk(ivec2 pixel) {
  if(!u_disks_enabled)discard;
  vec4 ray=texelFetch(u_surface_ray,pixel,0);
  if(ray.z>=0.0)discard;
  vec3 center=decodeDirection(ray.xy);
  float radius=acos(u_direct_sun_cos_radius);
  float separation=atan(length(cross(center,u_solar_direction)),dot(center,u_solar_direction));
  // Conservative bound for this atmosphere's limb deflection plus pixel
  // footprint. The expensive forward-map evaluations stay near the Sun.
  vec3 dx=skyRayStep(pixel,ivec2(1,0),center),dy=skyRayStep(pixel,ivec2(0,1),center);
  if(separation>radius+.04+length(dx)+length(dy))discard;
  vec3 sum=vec3(0);
  float hits=0.0,meteredSum=0.0;
  float h=max(u_ray4d_observer_height,0.0);
    float mediumIndex=u_refraction?1.0+.00029*
    (exp(-clamp(h,0.0,100.0)/8.0)-exp(-12.5)):1.0;
  for(int y=0;y<4;y++)for(int x=0;x<4;x++) {
    vec2 offset=(vec2(x,y)+.5)/4.0-.5;
    vec3 direction=normalize(center+dx*offset.x+dy*offset.y);
    vec3 transmission,unusedScattering;
    atmosphereRay(u_camera_origin,direction,1e30,true,transmission,unusedScattering);
    if(u_refraction&&u_sky_strength<=0.0)
      refractionExternal(u_camera_origin,h,direction);
    vec3 exterior=atmosphereBackgroundDirection;
    float angle=atan(length(cross(exterior,u_solar_direction)),dot(exterior,u_solar_direction));
    if(angle>=radius||all(lessThanEqual(transmission,vec3(0))))continue;
    vec3 toMoon=u_moon_position-atmosphereBackgroundOrigin;
    vec3 moon=normalize(toMoon);
    if(dot(exterior,moon)>0.0&&length(cross(exterior,moon))<1737.4/length(toMoon))continue;
    vec2 screen=vec2(pixel)+.5+offset;
    vec2 clip=screen/u_viewport*2.0-1.0;
    vec3 sensor=vec3(clip*u_sensor_projection.xy+u_sensor_projection.zw,-1);
    float angularArea=abs(u_sensor_projection.x*u_sensor_projection.y)/pow(length(sensor),3.0);
    float reception=9.0*angularArea*(u_viewport.y/u_viewport.x)*u_photography_gain;
    vec3 value=u_direct_solar_radiance*mix(vec3(1),transmission,u_sky_strength)
      *(mediumIndex*mediumIndex)*reception;
    sum+=value;hits+=1.0;
    // Preserve the existing exclusion of empty space from scene metering.
    float along=dot(u_camera_origin,direction);
    float radial=(100.0-h)*(12842.0+h);
    if(h<=100.0||(along<0.0&&along*along+radial>=0.0))
      meteredSum+=dot(max(value,vec3(0)),vec3(.2126,.7152,.0722));
  }
  if(hits==0.0)discard;
  // Keep small disk fractions in the radiance.
  // Native ground coverage is applied separately by the shared stencil.
  fragColor=vec4(sum,meteredSum)*(u_sky_opacity/16.0);
}
void main() {
  vec2 screen=gl_FragCoord.xy;
  ivec2 pixel=ivec2(screen);
  if(u_solar_pass) { shadeSolarDisk(pixel); return; }
  vec4 hit=texelFetch(u_surface_ray,pixel,0);
  if(u_ground_pass ? hit.z<=0.0 : hit.z>=0.0)discard;
  vec3 direction=decodeDirection(hit.xy);
  vec2 dx=vec2(0),dy=vec2(0);
  if(!u_ground_pass) {
    mat3 tangent=transpose(skyBasis(direction));
    vec3 rayDx=skyRayStep(pixel,ivec2(1,0),direction),rayDy=skyRayStep(pixel,ivec2(0,1),direction);
    dx=(tangent*rayDx).xy;
    dy=(tangent*rayDy).xy;
  }
  fragColor=shadeSurfaceSample(screen,hit,direction,dx,dy);
}`;

type Terrain = NonNullable<MaplibreMap['terrain']>;
type Tile = ReturnType<MaplibreMap['coveringTiles']>[number];

/** Match visible receiver meshes, and fill only their uncovered surroundings.
 * DEM source IDs are deltaZoom levels below the mesh IDs expected by MapLibre.
 * Never draw coarse parent terrain over a more detailed receiver or caster.
 */
function terrainShadowTiles(visible: Tile[], sources: Tile[], deltaZoom: number): Tile[] {
  const expand = (tile: Tile, levels: number): Tile[] =>
    levels === 0 ? [tile] : tile.children(22).flatMap((child) => expand(child, levels - 1));
  const candidates = [...sources.flatMap((tile) => expand(tile, deltaZoom)), ...visible];
  const receivers = new Set(visible.map((tile) => tile.key));
  const roots = new Map(candidates.map((tile) => [tile.scaledTo(0).key, tile.scaledTo(0)]));
  const visit = (tile: Tile, covered: boolean, descendants: Tile[]): Tile[] => {
    if (receivers.has(tile.key)) return [tile];
    covered ||= descendants.some((candidate) => candidate.key === tile.key);
    const children = descendants.filter((candidate) => candidate.canonical.z > tile.canonical.z);
    if (children.length === 0) return covered ? [tile] : [];
    return tile.children(22).flatMap((child) =>
      visit(
        child,
        covered,
        children.filter((candidate) => candidate.key === child.key || candidate.isChildOf(child)),
      ),
    );
  };
  return Array.from(roots.values()).flatMap((root) =>
    visit(
      root,
      false,
      candidates.filter((tile) => tile.key === root.key || tile.isChildOf(root)),
    ),
  );
}

/**
 * MapLibre WebGL adapter. Consume native tile meshes and DEM descriptors;
 * own the shaders, uniforms and VAOs. Tile enumeration stays
 * with map.coveringTiles({terrain}), outside this adapter.
 */
class NativeSurface {
  #terrain?: Terrain;
  #programs = new Map<string, TileSurfaceProgram>();

  constructor(private readonly shadow = false) {}

  #createProgram(
    gl: WebGL2RenderingContext,
    options: CustomRenderMethodInput,
    terrain: boolean,
    depth3D: boolean,
  ): TileSurfaceProgram {
    const position = 'a_tile.xy';
    const globe = /\bGLOBE\b/.test(options.shaderData.define);
    const vertexSource =
      tileVertexPrelude(options, terrain) +
      `
      ${mercatorSphereGLSL}
      ${relativePositionGLSL}
      uniform vec4 u_nightlayer_tile;
      ${
        this.shadow
          ? `uniform mat4 u_shadow_matrix; uniform vec3 u_shadow_origin;
      ${shadowVertexGLSL}`
          : ''
      }
      out highp vec3 v_nightlayer_direction;
      out highp vec3 v_nightlayer_delta;
      out highp vec3 v_nightlayer_normal;
      out highp float v_nightlayer_elevation;
      vec3 tileDelta(vec2 position) {
        #ifdef GLOBE
          if(position.y < -32767.5 || position.y > 32766.5)
            return localPoleDelta(sign(position.y)*-1.0);
        #endif
        return localSphereDelta(position,u_nightlayer_tile.zw);
      }
      void main() {
        float elevation=terrainElevation(${position});
        float skirt=a_tile.z==1.0 ? u_skirt : 0.0;
        gl_Position=${terrain || depth3D ? `projectTileFor3D(${position},elevation-skirt)` : `projectTile(${position},${position})`};
        v_nightlayer_elevation = elevation-skirt;
        #ifdef GLOBE
          v_nightlayer_direction = projectToSphere(${position}, ${position});
        #else
          v_nightlayer_direction = vec3(${position}, 0.0);
        #endif
        vec2 mercator = u_nightlayer_tile.xy + ${position} * u_nightlayer_tile.zw;
        float longitude = mercator.x * ${2 * Math.PI} + ${Math.PI};
        #ifdef GLOBE
          // Use the receiver's exact native sphere mapping, including pole
          // sentinel vertices. Reinterpreting those as Mercator Y bends caps.
          vec3 up = v_nightlayer_direction;
        #else
          vec3 up = mercatorSphere(mercator);
        #endif
        float latitudeSin = up.y;
        float latitudeCos = length(up.xz);
        v_nightlayer_normal = up;
        v_nightlayer_delta = tileDelta(vec2(${position}));
        ${
          this.shadow
            ? `
        // Native skirt vertices duplicate the edge XY. At the same elevation
        // their triangles have zero area; real pole caps keep their own XY.
        v_nightlayer_elevation = elevation;
        vec3 light=(u_shadow_matrix * vec4(localEndpoint(v_nightlayer_delta,v_nightlayer_elevation), 1.0)).xyz;
        gl_Position=shadowPosition(light,elevation*.001);
        `
            : ''
        }
        ${
          terrain
            ? `
        // Polar cap vertices are native sentinels, not DEM sample positions.
        // A finite difference across that sentinel samples unrelated clamped
        // DEM texels and divides by zero meridian width at the pole.
        if(latitudeCos>1e-6) {
        vec2 stepSize = 1.0 / (u_dem_size * vec2(u_dem_matrix[0][0], u_dem_matrix[1][1]));
        vec2 heightDelta = vec2(
          terrainElevation(${position} + vec2(stepSize.x, 0.0)) - terrainElevation(${position} - vec2(stepSize.x, 0.0)),
          terrainElevation(${position} + vec2(0.0, stepSize.y)) - terrainElevation(${position} - vec2(0.0, stepSize.y)));
        vec2 distance = 2.0 * stepSize * u_nightlayer_tile.zw * ${2 * Math.PI * 6371000} * max(latitudeCos, 0.000001);
        vec2 slope = heightDelta / distance;
        vec3 east = vec3(cos(longitude), 0.0, -sin(longitude));
        vec3 north = vec3(-sin(longitude) * latitudeSin, latitudeCos, -cos(longitude) * latitudeSin);
        v_nightlayer_normal = normalize(up - slope.x * east + slope.y * north);
        }
        `
            : ''
        }
      }`;
    const fragmentSource = this.shadow
      ? shadowFragment
      : `#version 300 es
      precision highp float;
      precision highp int;
      // Forward endpoint attributes use ordinary perspective interpolation.
      in highp vec3 v_nightlayer_direction;
      in highp vec3 v_nightlayer_delta;
      in highp vec3 v_nightlayer_normal;
      in highp float v_nightlayer_elevation;
      uniform vec4 u_nightlayer_tile;
      uniform vec4 u_observer_horizon; // observer up and squared horizon chord
      layout(location = 0) out vec4 hit;
      ${inverseLookupGLSL}
      ${mercatorSphereGLSL}
      ${relativePositionGLSL}
      void main() {
        // Include native stitch walls as visible receivers so tile gaps use
        // the same endpoint as the map instead of the background atmosphere.
        vec3 delta=${globe ? 'localNormalizeDelta(v_nightlayer_delta)' : 'localSphereDelta(v_nightlayer_direction.xy,u_nightlayer_tile.zw)'};
        // Geographic output and relative ray share one endpoint definition.
        vec3 up = normalize(u_local_to_world*(u_local_up+delta));
        vec3 chord = up - u_observer_horizon.xyz;
        if(u_observer_horizon.w<4.0&&dot(chord,chord)>u_observer_horizon.w)discard;
        hit=packGroundHit(localEndpoint(delta,v_nightlayer_elevation),
          ${terrain ? 'normalize(v_nightlayer_normal)' : 'up'},
          false);
      }`;
    return new TileSurfaceProgram(gl, vertexSource, fragmentSource);
  }

  render(
    map: MaplibreMap,
    gl: WebGL2RenderingContext,
    tiles: Tile[],
    options: CustomRenderMethodInput,
    { shadow, depth3D = false }: { shadow?: TerrainShadow; depth3D?: boolean } = {},
  ): void {
    const terrain = map.terrain;
    const painter = map.painter;
    const projection = map.style.projection;
    if (
      !painter?.context ||
      !projection?.getMeshFromTileID ||
      (terrain && (!terrain.getTerrainMesh || !terrain.getTerrainData))
    ) {
      throw new Error('NightLayer: unsupported MapLibre native terrain interface');
    }
    if (this.#terrain && this.#terrain !== terrain) this.dispose(gl);
    this.#terrain = terrain;
    const globeMesh = /\bGLOBE\b/.test(options.shaderData.define);
    // The 2D globe projection substitutes a clipping Z. Use the same physical
    // 3D depth for terrainless tiles and buildings when comparing their depths.
    const globeDepth = !terrain && depth3D && globeMesh;
    const key = `${options.shaderData.variantName}/${!!terrain}/${globeDepth}`;
    let program = this.#programs.get(key);
    if (!program) {
      program = this.#createProgram(gl, options, !!terrain, globeDepth);
      this.#programs.set(key, program);
    }
    const context = painter.context;
    context.setDirty();
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(true);
    gl.depthRange(0, 1);
    if (shadow) gl.disable(gl.STENCIL_TEST);
    else {
      // Record the same ground hit in the LUT and its receiver stencil.
      gl.enable(gl.STENCIL_TEST);
      gl.stencilMask(0xff);
      gl.stencilFunc(gl.ALWAYS, 1, 0xff);
      gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
    }
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.BLEND);
    gl.colorMask(true, true, true, true);
    if (shadow) gl.disable(gl.CULL_FACE);
    else {
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.BACK);
      gl.frontFace(gl.CCW);
    }
    const camera = atmosphereCamera(map, options);
    if (!shadow) {
      const radius = Math.hypot(...camera.origin);
      const horizon = groundHorizonChordSquared(
        radius,
        atmosphereFade(map.getZoom(), options.defaultProjectionData.projectionTransition),
      );
      gl.useProgram(program.program);
      gl.uniform4f(
        uniformLocation(gl, program.program, 'u_observer_horizon'),
        camera.origin[0] / radius,
        camera.origin[1] / radius,
        camera.origin[2] / radius,
        horizon,
      );
    }
    if (shadow) gl.viewport(0, 0, TERRAIN_SHADOW_SIZE, TERRAIN_SHADOW_SIZE);
    for (const tile of tiles) {
      const mesh = terrain
        ? terrain.getTerrainMesh(tile, globeMesh)
        : projection.getMeshFromTileID(context, tile.canonical, false, true, 'raster');
      const data = terrain?.getTerrainData(tile) ?? null;
      const tileProjection = options.getProjectionData({
        tileID: tile,
        applyTerrainMatrix: false,
        applyGlobeMatrix: true,
      });
      program.bind(gl, tileProjection, data, terrain?.getSkirtLength(map.getZoom()) ?? 0);
      shadow?.bindProjection(gl, program.program);
      gl.uniform4fv(uniformLocation(gl, program.program, 'u_nightlayer_tile'), tileProjection.tileMercatorCoords);
      bindRelativePosition(gl, program.program, tileProjection.tileMercatorCoords, camera.origin);
      program.draw(gl, mesh);
    }
    context.setDirty();
  }

  dispose(gl: WebGL2RenderingContext): void {
    for (const program of this.#programs.values()) program.dispose(gl);
    this.#programs.clear();
    this.#terrain = undefined;
  }
}
