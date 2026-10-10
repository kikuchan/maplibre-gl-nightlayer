import { createTileMesh } from 'maplibre-gl';
import type { CustomLayerInterface, CustomRenderMethodInput, Map as MaplibreMap } from 'maplibre-gl';

type Mesh = { vertices: WebGLBuffer; indices: WebGLBuffer; count: number };

const TWILIGHT_THRESHOLD_SINES = Array.from({ length: 15 }, (_, i) => Math.sin(radians(i === 0 ? -0.8333333 : -6 * i)));

function radians(deg: number) {
  return (deg * Math.PI) / 180;
}

function degrees(rad: number) {
  return (rad / Math.PI) * 180;
}

function premultiplyAlpha(c: number[]) {
  const [r, g, b, a] = c.map((x) => x / 255);
  return [r * a, g * a, b * a, a];
}

/**
 * Get the subsolar point (longitude and latitude) at the given date.
 * @param date If null, the current date is used.
 * @returns The subsolar point.
 */
export function getSubsolarPoint(date?: Date | null): { lng: number; lat: number } {
  date ??= new Date();
  if (!Number.isFinite(date.getTime())) throw new RangeError('Invalid date');

  // based on https://en.wikipedia.org/wiki/Equation_of_time#Alternative_calculation
  const year = date.getUTCFullYear();
  const month = date.getUTCMonth();
  const leapDay = month > 1 && year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0) ? 1 : 0;
  const dayOfYear = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334][month] + date.getUTCDate() + leapDay;
  const UTC =
    date.getUTCHours() + date.getUTCMinutes() / 60 + date.getUTCSeconds() / 3600 + date.getUTCMilliseconds() / 3600000;
  const D = dayOfYear + UTC / 24;
  const n = (2 * Math.PI) / 365.24;

  const e = radians(23.44); // Earth's axial tilt
  const E = 0.0167; // Earth's orbital eccentricity

  const A = (D + 9) * n;
  const B = A + 2 * E * Math.sin((D - 3) * n);
  const C = (A - Math.atan2(Math.sin(B), Math.cos(B) * Math.cos(e))) / Math.PI;

  const EOT = 720 * (C - Math.trunc(C + 0.5));

  const lng = -15 * (UTC - 12 + EOT / 60);
  const lat = degrees(Math.asin(Math.sin(-e) * Math.cos(B)));

  return { lng, lat };
}

type Color3 = [number, number, number];
type Color4 = [number, number, number, number];
type Color = Color3 | Color4;

/**
 * Options for the NightLayer.
 */
type Options = {
  /**
   * Date for the shadow.
   * If null, the current date is used.
   */
  date?: Date | null;

  /**
   * Opacity of the layer.
   * 0.0 means fully transparent, 1.0 means fully opaque
   * default: 0.5
   */
  opacity?: number;

  /**
   * Color of the shadow: [r, g, b, a]
   * Each value should be in the range [0, 255].
   * default: [0, 0, 0, 255]
   */
  color?: Color;

  /**
   * Color of the daytime: [r, g, b, a]
   * Each value should be in the range [0, 255].
   * default: [0, 0, 0, 0]
   */
  daytimeColor?: Color;

  /**
   * Number of twilight steps.
   * 0 means no steps (gradation), 1 means one step (day/night), etc.
   * default: 0
   */
  twilightSteps?: number;

  /**
   * Attenuation factor for each twilight step.
   * 0.0 means no attenuation, 1.0 means full attenuation
   * default: 0.5
   */
  twilightAttenuation?: number;

  /**
   * Update interval in milliseconds.
   * Periodically triggers a repaint.
   * default: 10000
   */
  updateInterval?: number;
};

/**
 * A custom Maplibre GL JS layer that renders the night side of the Earth.
 */
export class NightLayer implements CustomLayerInterface {
  id = 'nightlayer';
  type = 'custom' as const;
  renderingMode = '2d' as const;

  #date!: Date | null;
  #opacity: number;
  #color!: Color4;
  #daytimeColor!: Color4;

  #updateIntervalTimer?: number;
  #updateInterval: number;

  #twilightSteps: number;
  #twilightAttenuation: number;

  #programCache?: Map<string, WebGLProgram>;
  #meshCache?: Map<string, Mesh>;
  #map?: MaplibreMap;

  /**
   * Create a new NightLayer instance. Intended to be passed to `map.addLayer`.
   * @param opts The options.
   */
  constructor(opts: Options = {}) {
    this.#opacity = opts.opacity ?? 0.5;

    this.#twilightSteps = opts.twilightSteps ?? 0;
    this.#twilightAttenuation = opts.twilightAttenuation ?? 0.5;
    this.#updateInterval = opts.updateInterval ?? 10000;

    this.setColor(opts.color ?? [0, 0, 0, 255]);
    this.setDaytimeColor(opts.daytimeColor ?? [0, 0, 0, 0]);
    this.setDate(opts.date ?? null);
  }

  #setIntervalTimer() {
    const activate = this.#map !== undefined && this.#date === null;
    if (this.#updateIntervalTimer) {
      clearInterval(this.#updateIntervalTimer);
      this.#updateIntervalTimer = undefined;
    }
    if (activate && this.#updateInterval > 0) {
      this.#updateIntervalTimer = setInterval(() => this.#map?.triggerRepaint(), this.#updateInterval);
    }
  }

  /**
   * Get the current subsolar point (longitude and latitude)
   * @returns The subsolar point.
   */
  getSubsolarPoint(): { lng: number; lat: number } {
    return getSubsolarPoint(this.#date);
  }

  /**
   * Get the date for the shadow.
   * @returns The date. If null, the current date is used.
   */
  getDate(): Date | null {
    return this.#date ? new Date(this.#date.getTime()) : null;
  }

  /**
   * Set the date for the shadow.
   * @param date If null, the current date is used.
   */
  setDate(date: Date | null): void {
    if (date !== null && !Number.isFinite(date.getTime())) throw new RangeError('Invalid date');
    this.#date = date === null ? null : new Date(date.getTime());
    this.#setIntervalTimer();
    this.#map?.triggerRepaint();
  }

  /**
   * Get the opacity of the shadow.
   * @returns The opacity. 0.0 means fully transparent, 1.0 means fully opaque.
   */
  getOpacity(): number {
    return this.#opacity;
  }

  /**
   * Set the opacity of the shadow.
   * @param opacity 0.0 means fully transparent, 1.0 means fully opaque.
   */
  setOpacity(opacity: number): void {
    this.#opacity = opacity;
    this.#map?.triggerRepaint();
  }

  /**
   * Get the color of the shadow.
   * @returns The color. Each value is in the range [0, 255].
   */
  getColor(): Color4 {
    return this.#color;
  }

  /**
   * Set the color of the shadow.
   * @param color Each value should be in the range [0, 255].
   */
  setColor(color: Color): void {
    this.#color = [...(color.slice(0, 3) as Color3), color?.[3] ?? 255];
    this.#map?.triggerRepaint();
  }

  /**
   * Get the color of the daytime.
   * @returns The color [r, g, b, a]. Each value is in the range [0, 255].
   */
  getDaytimeColor(): Color4 {
    return this.#daytimeColor;
  }

  /**
   * Set the color of the daytime.
   * @param color Each value should be in the range [0, 255].
   */
  setDaytimeColor(color: Color): void {
    this.#daytimeColor = [...(color.slice(0, 3) as Color3), color?.[3] ?? 255];
    this.#map?.triggerRepaint();
  }

  /**
   * Get the number of twilight steps.
   * @returns The steps. 0 means no steps (gradation), 1 means one step (day/night), etc.
   */
  getTwilightSteps(): number {
    return this.#twilightSteps;
  }

  /**
   * Set the number of twilight steps.
   * @param steps 0 means no steps (gradation), 1 means one step (day/night), etc.
   */
  setTwilightSteps(steps: number): void {
    this.#twilightSteps = steps;
    this.#map?.triggerRepaint();
  }

  /**
   * Get the attenuation factor for each twilight step.
   * @returns The attenuation. 0.0 means no attenuation, 1.0 means full attenuation.
   */
  getTwilightAttenuation(): number {
    return this.#twilightAttenuation;
  }

  /**
   * Set the attenuation factor for each twilight step.
   * @param attenuation 0.0 means no attenuation, 1.0 means full attenuation.
   */
  setTwilightAttenuation(attenuation: number): void {
    this.#twilightAttenuation = attenuation;
    this.#map?.triggerRepaint();
  }

  /**
   * Get the update interval.
   * @returns The interval in milliseconds.
   */
  getUpdateInterval(): number {
    return this.#updateInterval;
  }

  /**
   * Set the update interval.
   * @param interval The interval in milliseconds..
   */
  setUpdateInterval(interval: number): void {
    this.#updateInterval = interval;
    this.#setIntervalTimer();
  }

  onAdd(map: MaplibreMap, _gl: WebGLRenderingContext | WebGL2RenderingContext): void {
    this.#programCache = new Map();
    this.#meshCache = new Map();
    this.#map = map;
    this.#setIntervalTimer();
  }

  onRemove(_map: MaplibreMap, gl: WebGLRenderingContext | WebGL2RenderingContext): void {
    if (this.#updateIntervalTimer !== undefined) {
      clearInterval(this.#updateIntervalTimer);
      this.#updateIntervalTimer = undefined;
    }
    if (this.#programCache) {
      for (const program of this.#programCache.values()) gl.deleteProgram(program);
    }
    if (this.#meshCache) {
      for (const mesh of this.#meshCache.values()) {
        gl.deleteBuffer(mesh.vertices);
        gl.deleteBuffer(mesh.indices);
      }
    }
    this.#programCache = undefined;
    this.#meshCache = undefined;
    this.#map = undefined;
  }

  #createProgram(
    gl: WebGLRenderingContext | WebGL2RenderingContext,
    shaderData: CustomRenderMethodInput['shaderData'],
  ) {
    const vertexSource = `#version 300 es
      precision highp float;

      in vec2 a_position;
      out vec2 v_position;
      out float v_pole;

      ${shaderData.vertexShaderPrelude}
      ${shaderData.define}

      void main() {
        gl_Position = projectTile(a_position, a_position);
        v_position = vec2(a_position.x, clamp(a_position.y, 0.0, 8192.0)) / 8192.0 - 0.5;
        v_pole = a_position.y < -32767.5 ? 1.0 : (a_position.y > 32766.5 ? -1.0 : 0.0);
      }
`;

    const fragmentSource = `#version 300 es
      precision highp float;

      in vec2 v_position;
      in float v_pole;
      out vec4 fragColor;

      uniform vec4 u_tile; // exp(mercator latitude), sin(latitude), cos(latitude), tile size
      uniform vec3 u_sun; // sin(solar latitude), cos(solar latitude) * cos/sin(relative longitude)
      uniform float u_center_height;
      uniform float u_thresholds[15];
      uniform float u_opacity;
      uniform float u_twilight_steps;
      uniform float u_twilight_attenuation;
      uniform vec4 u_daytime_color;
      uniform vec4 u_night_color;

      float expMinusOne(float x) {
        if (abs(x) < 0.001) return x * (1.0 + x * (0.5 + x * (1.0 / 6.0 + x / 24.0)));
        return exp(x) - 1.0;
      }

      void main() {
        // Work with small tile-local differences, not rounded global coordinates.
        float longitudeDelta = v_position.x * u_tile.w * ${2 * Math.PI};
        float e = expMinusOne(-v_position.y * u_tile.w * ${2 * Math.PI});
        float t = u_tile.x;
        float denominator = t * t + 1.0;
        float deltaSquared = t * t * e * (2.0 + e);
        float product = denominator * (denominator + deltaSquared);
        float sinDelta = 2.0 * deltaSquared / product;
        float cosDelta = 2.0 * t * (e * denominator - deltaSquared) / product;

        float poleAngle = v_pole * ${radians(90 - 85.0511287798066)};
        float poleCosDelta = -2.0 * pow(sin(poleAngle * 0.5), 2.0);
        float poleSin = sin(poleAngle);
        float latitudeSin = u_tile.y + sinDelta;
        float latitudeCos = u_tile.z + cosDelta;
        sinDelta += latitudeSin * poleCosDelta + latitudeCos * poleSin;
        cosDelta += latitudeCos * poleCosDelta - latitudeSin * poleSin;
        latitudeCos = u_tile.z + cosDelta;

        float longitudeCosDelta = -2.0 * pow(sin(longitudeDelta * 0.5), 2.0);
        float heightDelta = u_sun.x * sinDelta + u_sun.y * cosDelta
          + latitudeCos * (u_sun.y * longitudeCosDelta + u_sun.z * sin(longitudeDelta));
        float twilightStepAngle = 6.0;
        float twilightSteps = u_twilight_steps;

        // Attenuation for each twilightStepAngle
        float att = u_twilight_attenuation;
        float twilightLevel;
        if (twilightSteps > 0.) {
          // Threshold residuals are computed in float64 on the CPU before upload.
          twilightLevel = 1.0;
          for (int i = 1; i < 15; i++) {
            if (u_thresholds[i] + heightDelta < 0.0) twilightLevel += 1.0;
          }
          twilightLevel = ceil(min(twilightLevel, twilightSteps));
          if (u_thresholds[0] + heightDelta > 0.0) twilightLevel = 0.0;
        } else {
          float altitude = degrees(asin(clamp(u_center_height + heightDelta, -1.0, 1.0)));
          twilightLevel = -altitude / twilightStepAngle;
        }
        float brightness = clamp(pow(clamp(1. - att, 0., 1.), twilightLevel), 0., 1.);
        fragColor = mix(u_night_color, u_daytime_color, brightness) * u_opacity;
      }`;

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    if (!vertexShader) throw new Error('failed to create vertex shader');
    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    if (!fragmentShader) {
      gl.deleteShader(vertexShader);
      throw new Error('failed to create fragment shader');
    }

    let program: WebGLProgram | null = null;
    try {
      gl.shaderSource(vertexShader, vertexSource);
      gl.compileShader(vertexShader);
      gl.shaderSource(fragmentShader, fragmentSource);
      gl.compileShader(fragmentShader);

      program = gl.createProgram();
      if (!program) throw new Error('failed to create program');
      gl.attachShader(program, vertexShader);
      gl.attachShader(program, fragmentShader);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program) || 'failed to link program');
      }
      return program;
    } catch (error) {
      if (program) gl.deleteProgram(program);
      program = null;
      throw error;
    } finally {
      if (program) {
        gl.detachShader(program, vertexShader);
        gl.detachShader(program, fragmentShader);
      }
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
    }
  }

  #getMesh(
    gl: WebGLRenderingContext | WebGL2RenderingContext,
    granularity: number,
    north: boolean,
    south: boolean,
  ): Mesh {
    const key = `${granularity}:${north}:${south}`;
    const cached = this.#meshCache!.get(key);
    if (cached) return cached;
    const data = createTileMesh(
      {
        granularity,
        generateBorders: false,
        extendToNorthPole: north,
        extendToSouthPole: south,
      },
      '16bit',
    );
    const vertices = gl.createBuffer();
    const indices = gl.createBuffer();
    if (!vertices || !indices) {
      if (vertices) gl.deleteBuffer(vertices);
      if (indices) gl.deleteBuffer(indices);
      throw new Error('failed to create tile mesh buffers');
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, vertices);
    gl.bufferData(gl.ARRAY_BUFFER, data.vertices, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indices);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data.indices, gl.STATIC_DRAW);
    const mesh = { vertices, indices, count: data.indices.byteLength / 2 };
    this.#meshCache!.set(key, mesh);
    return mesh;
  }

  render(gl: WebGLRenderingContext | WebGL2RenderingContext, options: CustomRenderMethodInput): void {
    if (!this.#programCache) return;
    if (!this.#meshCache) return;
    if (!this.#map) return;

    const shaderData = options.shaderData;
    const programKey = shaderData.variantName;
    if (!this.#programCache.has(programKey)) {
      this.#programCache.set(programKey, this.#createProgram(gl, shaderData));
    }
    const program = this.#programCache.get(programKey);
    if (!program) return;

    const { lng: subsolarLng, lat: subsolarLat } = this.getSubsolarPoint();
    const solarLongitude = radians(subsolarLng);
    const solarLatitude = radians(subsolarLat);
    const solarSin = Math.sin(solarLatitude);
    const solarCos = Math.cos(solarLatitude);

    gl.useProgram(program);

    gl.uniform1f(gl.getUniformLocation(program, 'u_opacity'), this.#opacity);
    gl.uniform4fv(gl.getUniformLocation(program, 'u_night_color'), premultiplyAlpha(this.#color));
    gl.uniform4fv(gl.getUniformLocation(program, 'u_daytime_color'), premultiplyAlpha(this.#daytimeColor));
    gl.uniform1f(gl.getUniformLocation(program, 'u_twilight_steps'), this.#twilightSteps);
    gl.uniform1f(gl.getUniformLocation(program, 'u_twilight_attenuation'), this.#twilightAttenuation);

    const globe = options.defaultProjectionData.projectionTransition > 0;
    const tiles = this.#map.coveringTiles({ tileSize: 256 });
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.enable(gl.CULL_FACE);
    gl.frontFace(gl.CCW);
    gl.cullFace(gl.BACK);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    for (const tile of tiles) {
      const { x, y, z } = tile.canonical;
      const size = 2 ** -z;
      const projectionData = options.getProjectionData({
        tileID: { canonical: { x, y, z }, wrap: tile.wrap },
        applyGlobeMatrix: true,
      });
      gl.uniformMatrix4fv(gl.getUniformLocation(program, 'u_projection_matrix'), false, projectionData.mainMatrix);
      gl.uniformMatrix4fv(
        gl.getUniformLocation(program, 'u_projection_fallback_matrix'),
        false,
        projectionData.fallbackMatrix,
      );
      gl.uniform4fv(
        gl.getUniformLocation(program, 'u_projection_tile_mercator_coords'),
        projectionData.tileMercatorCoords,
      );
      gl.uniform4fv(gl.getUniformLocation(program, 'u_projection_clipping_plane'), projectionData.clippingPlane);
      gl.uniform1f(gl.getUniformLocation(program, 'u_projection_transition'), projectionData.projectionTransition);

      // Compute the tile center in float64; the shader computes small local differences.
      const t = Math.exp(Math.PI * (1 - 2 * (y + 0.5) * size));
      const latitudeSin = (t * t - 1) / (t * t + 1);
      const latitudeCos = (2 * t) / (t * t + 1);
      const longitude = radians((x + 0.5) * size * 360 - 180);
      const longitudeDifference = solarLongitude - longitude;
      const sun = [solarSin, solarCos * Math.cos(longitudeDifference), solarCos * Math.sin(longitudeDifference)];
      const centerHeight = sun[0] * latitudeSin + sun[1] * latitudeCos;
      const thresholds = TWILIGHT_THRESHOLD_SINES.map((threshold) => centerHeight - threshold);
      gl.uniform4fv(gl.getUniformLocation(program, 'u_tile'), [t, latitudeSin, latitudeCos, size]);
      gl.uniform3fv(gl.getUniformLocation(program, 'u_sun'), sun);
      gl.uniform1f(gl.getUniformLocation(program, 'u_center_height'), centerHeight);
      gl.uniform1fv(gl.getUniformLocation(program, 'u_thresholds[0]'), thresholds);

      // Match MapLibre's raster tile subdivision: 128 at z=0, at least 32 per tile.
      const granularity = globe ? Math.max(Math.floor(128 * size), 32) : 1;
      const mesh = this.#getMesh(gl, granularity, globe && y === 0, globe && y === 2 ** z - 1);
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.vertices);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.indices);
      const index = gl.getAttribLocation(program, 'a_position');
      gl.enableVertexAttribArray(index);
      gl.vertexAttribPointer(index, 2, gl.SHORT, false, 0, 0);
      gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0);
    }
  }
}

export default NightLayer;
