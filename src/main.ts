import { MercatorCoordinate, createTileMesh } from 'maplibre-gl';
import type { CustomLayerInterface, CustomRenderMethodInput, Map as MaplibreMap, TileMesh } from 'maplibre-gl';
import type { mat4 } from 'gl-matrix';

function radians(deg: number) {
  return (deg * Math.PI) / 180;
}

function degrees(rad: number) {
  return (rad / Math.PI) * 180;
}

type Color = [number, number, number];

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
   * Opacity of the shadow.
   * 0.0 means fully transparent, 1.0 means fully opaque
   */
  opacity?: number;

  /**
   * Color of the shadow.
   * Each value should be in the range [0, 255].
   */
  color?: Color;

  /**
   * Number of twilight steps.
   * 0 means no steps (gradation), 1 means one step (day/night), etc.
   */
  twilightSteps?: number;

  /**
   * Angle for each twilight step in degrees.
   */
  twilightStepAngle?: number;

  /**
   * Attenuation factor for each twilight step.
   * 0.0 means no attenuation, 1.0 means full attenuation
   */
  twilightAttenuation?: number;
};

/**
 * A custom Maplibre GL JS layer that renders the night side of the Earth.
 */
export class NightLayer implements CustomLayerInterface {
  id = 'nightlayer';
  type = 'custom' as const;
  renderingMode = '2d' as const;

  #date: Date | null;
  #opacity: number;
  #color: Color;

  #twilightSteps: number;
  #twilightStepAngle: number;
  #twilightAttenuation: number;

  #programCache?: Map<string, WebGLProgram>;
  #map?: MaplibreMap;
  #vbo?: WebGLBuffer;
  #ibo?: WebGLBuffer;
  #indices?: number;

  #modelSignature?: string;

  /**
   * Create a new NightLayer instance. Intended to be passed to `map.addLayer`.
   * @param opts The options.
   */
  constructor(opts: Options = {}) {
    this.#date = opts.date ?? null;
    this.#opacity = opts.opacity ?? 0.5;
    this.#color = opts.color ?? [0, 0, 0];

    this.#twilightSteps = opts.twilightSteps ?? 0;
    this.#twilightStepAngle = opts.twilightStepAngle ?? 6;
    this.#twilightAttenuation = opts.twilightAttenuation ?? 0.5;
  }

  /**
   * Get the subsolar point (longitude and latitude) at the given date.
   * @param date If null, the current date is used.
   */
  getSubsolarPoint(date: Date) {
    // based on https://en.wikipedia.org/wiki/Equation_of_time#Alternative_calculation
    const D = (date.getTime() - Date.UTC(date.getUTCFullYear(), 0, 0)) / 86400000;
    const n = (2 * Math.PI) / 365.24;

    const e = radians(23.44); // Earth's axial tilt
    const E = 0.0167; // Earth's orbital eccentricity

    const A = (D + 9) * n;
    const B = A + 2 * E * Math.sin((D - 3) * n);
    const C = (A - Math.atan2(Math.sin(B), Math.cos(B) * Math.cos(e))) / Math.PI;

    const EOT = 720 * (C - Math.trunc(C + 0.5));

    const UTC = date.getUTCHours() + date.getUTCMinutes() / 60 + date.getUTCSeconds() / 3600;

    const subsolarLng = -15 * (UTC - 12 + EOT / 60);
    const subsolarLat = degrees(Math.asin(Math.sin(-e) * Math.cos(B)));

    return {
      subsolarLng,
      subsolarLat,
    };
  }

  /**
   * Get the date for the shadow.
   * @returns The date. If null, the current date is used.
   */
  getDate() {
    return this.#date;
  }

  /**
   * Set the date for the shadow.
   * @param date If null, the current date is used.
   */
  setDate(date: Date | null) {
    this.#date = date;
    this.#map?.triggerRepaint();
  }

  /**
   * Get the opacity of the shadow.
   * @returns The opacity. 0.0 means fully transparent, 1.0 means fully opaque.
   */
  getOpacity() {
    return this.#opacity;
  }

  /**
   * Set the opacity of the shadow.
   * @param opacity 0.0 means fully transparent, 1.0 means fully opaque.
   */
  setOpacity(opacity: number) {
    this.#opacity = opacity;
    this.#map?.triggerRepaint();
  }

  /**
   * Get the color of the shadow.
   * @returns The color. Each value is in the range [0, 255].
   */
  getColor() {
    return this.#color;
  }

  /**
   * Set the color of the shadow.
   * @param color Each value should be in the range [0, 255].
   */
  setColor(color: Color) {
    this.#color = color;
    this.#map?.triggerRepaint();
  }

  /**
   * Get the number of twilight steps.
   * @returns The steps. 0 means no steps (gradation), 1 means one step (day/night), etc.
   */
  getTwilightSteps() {
    return this.#twilightSteps;
  }

  /**
   * Set the number of twilight steps.
   * @param steps 0 means no steps (gradation), 1 means one step (day/night), etc.
   */
  setTwilightSteps(steps: number) {
    this.#twilightSteps = steps;
    this.#map?.triggerRepaint();
  }

  /**
   * Get the angle for each twilight step.
   * @returns The angle in degrees.
   */
  getTwilightStepAngle() {
    return this.#twilightStepAngle;
  }

  /**
   * Set the angle for each twilight step.
   * @param angle in degrees.
   */
  setTwilightStepAngle(angle: number) {
    this.#twilightStepAngle = angle;
    this.#map?.triggerRepaint();
  }

  /**
   * Get the attenuation factor for each twilight step.
   * @returns The attenuation. 0.0 means no attenuation, 1.0 means full attenuation.
   */
  getTwilightAttenuation() {
    return this.#twilightAttenuation;
  }

  /**
   * Set the attenuation factor for each twilight step.
   * @param attenuation 0.0 means no attenuation, 1.0 means full attenuation.
   */
  setTwilightAttenuation(attenuation: number) {
    this.#twilightAttenuation = attenuation;
    this.#map?.triggerRepaint();
  }

  onAdd(map: MaplibreMap, gl: WebGLRenderingContext | WebGL2RenderingContext) {
    this.#programCache = new Map();
    this.#vbo = gl.createBuffer()!;
    this.#ibo = gl.createBuffer()!;
    this.#map = map;
  }

  onRemove() {
    this.#programCache = undefined;
    this.#map = undefined;
    this.#vbo = undefined;
    this.#ibo = undefined;
  }

  #createProgram(
    gl: WebGLRenderingContext | WebGL2RenderingContext,
    shaderData: { vertexShaderPrelude?: string; define?: string },
  ) {
    const isV4 = !shaderData.vertexShaderPrelude;

    const vertexSource = `#version 300 es
      precision highp float;

      in vec2 a_position;
      out vec2 v_position;

${
  isV4
    ? `
      uniform mat4 u_matrix;

      vec4 projectTile(vec2 pos) {
        return u_matrix * vec4(pos, 0., 1.);
      }
`
    : `
      ${shaderData.vertexShaderPrelude}
      ${shaderData.define}
`
}

      void main() {
        gl_Position = projectTile(a_position);
        v_position = a_position;
      }
`;

    const fragmentSource = `#version 300 es
      precision highp float;

      in vec2 v_position;
      out vec4 fragColor;

      uniform vec2 u_subsolar;
      uniform float u_opacity;
      uniform float u_twilight_step_angle;
      uniform float u_twilight_steps;
      uniform float u_twilight_attenuation;
      uniform vec3 u_color;

      vec2 mercatorToLngLat(vec2 mercator) {
        // 0 <= x <= 1, 0 <= y <= 1
        float x = mercator.x;
        float y = mercator.y;

        float lng = x * 360.0 - 180.0;
        float lat = degrees(2.0 * atan(exp(${Math.PI} * (1.0 - 2.0 * y))) - ${Math.PI / 2});

        return vec2(lng, lat);
      }

      void main() {
        vec2 lnglat = mercatorToLngLat(v_position);

        vec2 observer = radians(lnglat);
        vec2 subsolar = radians(u_subsolar);
        float A = sin(observer.y) * sin(subsolar.y);
        float B = cos(observer.y) * cos(subsolar.y) * cos(subsolar.x - observer.x);
        float altitude = degrees(asin(A + B));
        float twilightStepAngle = u_twilight_step_angle;
        float twilightSteps = u_twilight_steps;

        // Attenuation for each twilightStepAngle
        float att = u_twilight_attenuation;
        float twilightLevel = -altitude / twilightStepAngle;
        if (twilightSteps > 0.) {
          twilightLevel = ceil(clamp(twilightLevel, 0., twilightSteps));
        }
        float brightness = clamp(pow(clamp(1. - att, 0., 1.), twilightLevel), 0., 1.);
        float darkness = (1. - brightness);

        fragColor = vec4(v_position, 0.0, 1.0);
        fragColor = vec4(u_color / 255., 1.0) * darkness * u_opacity;
      }`;

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    if (!vertexShader) throw new Error('failed to create vertex shader');
    gl.shaderSource(vertexShader, vertexSource);
    gl.compileShader(vertexShader);

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    if (!fragmentShader) throw new Error('failed to create fragment shader');
    gl.shaderSource(fragmentShader, fragmentSource);
    gl.compileShader(fragmentShader);

    const program = gl.createProgram()!;
    if (!program) throw new Error('failed to create program');
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    return program;
  }

  #updateModel(gl: WebGLRenderingContext | WebGL2RenderingContext) {
    if (!this.#map) return;
    if (!this.#vbo) return;
    if (!this.#ibo) return;

    const [w, e] = this.#map.getBounds().toArray();
    const xmin = Math.floor(MercatorCoordinate.fromLngLat(w).x);
    const xmax = Math.ceil(MercatorCoordinate.fromLngLat(e).x);

    const modelSignature = this.#map.style?.projection?.name === 'globe' ? 'globe' : [xmin, xmax].join(':');
    if (modelSignature === this.#modelSignature) return;

    let meshBuffers: TileMesh;
    if (modelSignature === 'globe') {
      // model for globe projection
      meshBuffers = createTileMesh(
        {
          granularity: 100,
          generateBorders: false,
          extendToNorthPole: true,
          extendToSouthPole: true,
        },
        '16bit',
      );

      meshBuffers.vertices = new Float32Array(new Int16Array(meshBuffers.vertices)).map((x) => x / 8192.0).buffer;
    } else {
      const vertices = new Float32Array([xmin, 0, xmax, 0, xmin, 1, xmin, 1, xmax, 0, xmax, 1]).buffer;

      meshBuffers = {
        vertices,
        indices: new Uint16Array([0, 1, 2, 3, 4, 5]).buffer,
        uses32bitIndices: false,
      };
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.#vbo);
    gl.bufferData(gl.ARRAY_BUFFER, meshBuffers.vertices, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.#ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, meshBuffers.indices, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

    this.#indices = meshBuffers.indices.byteLength / 2;
    this.#modelSignature = modelSignature;
  }

  // v4 interface
  render(gl: WebGLRenderingContext | WebGL2RenderingContext, matrix: mat4, options: CustomRenderMethodInput): void;
  // v5 interface
  render(gl: WebGLRenderingContext | WebGL2RenderingContext, options: CustomRenderMethodInput): void;
  render(
    gl: WebGLRenderingContext | WebGL2RenderingContext,
    matrix: mat4 | CustomRenderMethodInput,
    options?: CustomRenderMethodInput,
  ) {
    if (!options) {
      // v5 interface
      options = matrix as CustomRenderMethodInput;
    }

    if (!this.#programCache) return;
    if (!this.#vbo) return;
    if (!this.#ibo) return;
    if (!this.#map) return;

    const shaderData = options.shaderData || {};
    const programKey = shaderData.variantName || 'default';
    if (!this.#programCache.has(programKey)) {
      this.#programCache.set(programKey, this.#createProgram(gl, shaderData));
    }
    const program = this.#programCache.get(programKey);
    if (!program) return;

    const { subsolarLng, subsolarLat } = this.getSubsolarPoint(this.#date || new Date());

    gl.useProgram(program);

    gl.uniform2fv(gl.getUniformLocation(program, 'u_subsolar'), [subsolarLng, subsolarLat]);
    gl.uniform1f(gl.getUniformLocation(program, 'u_opacity'), this.#opacity);
    gl.uniform3fv(gl.getUniformLocation(program, 'u_color'), this.#color);
    gl.uniform1f(gl.getUniformLocation(program, 'u_twilight_steps'), this.#twilightSteps);
    gl.uniform1f(gl.getUniformLocation(program, 'u_twilight_step_angle'), this.#twilightStepAngle);
    gl.uniform1f(gl.getUniformLocation(program, 'u_twilight_attenuation'), this.#twilightAttenuation);

    if ('getProjectionDataForCustomLayer' in this.#map.transform) {
      // v5 interface
      const projectionData = this.#map.transform.getProjectionDataForCustomLayer(true);

      // The magic: based on test/examples/globe-custom-simple.html
      gl.uniformMatrix4fv(
        gl.getUniformLocation(program, 'u_projection_fallback_matrix'),
        false,
        projectionData.fallbackMatrix, // convert mat4 from gl-matrix to a plain array
      );
      gl.uniformMatrix4fv(
        gl.getUniformLocation(program, 'u_projection_matrix'),
        false,
        projectionData.mainMatrix, // convert mat4 from gl-matrix to a plain array
      );
      gl.uniform4f(
        gl.getUniformLocation(program, 'u_projection_tile_mercator_coords'),
        ...projectionData.tileMercatorCoords,
      );
      gl.uniform4f(gl.getUniformLocation(program, 'u_projection_clipping_plane'), ...projectionData.clippingPlane);
      gl.uniform1f(gl.getUniformLocation(program, 'u_projection_transition'), projectionData.projectionTransition);
    } else {
      // v4 interface
      gl.uniformMatrix4fv(gl.getUniformLocation(program, 'u_matrix'), false, matrix as mat4);
    }

    this.#updateModel(gl);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.#vbo);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.#ibo);

    const index = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(index);
    gl.vertexAttribPointer(index, 2, gl.FLOAT, false, 0, 0);

    gl.drawElements(gl.TRIANGLES, this.#indices!, gl.UNSIGNED_SHORT, 0);
  }
}

export default NightLayer;
