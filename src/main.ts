import type { CustomLayerInterface, Map as MaplibreMap } from 'maplibre-gl';
import { MercatorCoordinate } from 'maplibre-gl';
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
   * Date.
   * If null, the current date is used.
   */
  date?: Date | null;

  /**
   * Opacity.
   * 0.0 means fully transparent, 1.0 means fully opaque
   */
  opacity?: number;

  /**
   * RGB color.
   * Each value is in the range [0, 255].
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

  #program?: WebGLProgram;
  #map?: MaplibreMap;
  #arrayBuffer?: WebGLBuffer;

  /**
   * Create a new NightLayer instance. Intended to be passed to `map.addLayer`.
   * @param opts Options.
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
   * @param date Date. If null, the current date is used.
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
   * Get the date of the night layer.
   * @returns Date. If null, the current date is used.
   */
  getDate() {
    return this.#date;
  }

  /**
   * Set the date of the night layer.
   * @param date Date. If null, the current date is used.
   */
  setDate(date: Date | null) {
    this.#date = date;
    this.#map?.triggerRepaint();
  }

  /**
   * Get the opacity of the night layer.
   * @returns Opacity. 0.0 means fully transparent, 1.0 means fully opaque.
   */
  getOpacity() {
    return this.#opacity;
  }

  /**
   * Set the opacity of the night layer.
   * @param v Opacity. 0.0 means fully transparent, 1.0 means fully opaque.
   */
  setOpacity(v: number) {
    this.#opacity = v;
    this.#map?.triggerRepaint();
  }

  /**
   * Get the color of the night layer.
   * @returns RGB color. Each value is in the range [0, 255].
   */
  getColor() {
    return this.#color;
  }

  /**
   * Set the color of the night layer.
   * @param v RGB color. Each value should be in the range [0, 255].
   */
  setColor(v: Color) {
    this.#color = v;
    this.#map?.triggerRepaint();
  }

  /**
   * Get the number of twilight steps.
   * @returns 0 means no steps (gradation), 1 means one step (day/night), etc.
   */
  getTwilightSteps() {
    return this.#twilightSteps;
  }

  /**
   * Set the number of twilight steps.
   * @param v 0 means no steps (gradation), 1 means one step (day/night), etc.
   */
  setTwilightSteps(v: number) {
    this.#twilightSteps = v;
    this.#map?.triggerRepaint();
  }

  /**
   * Get the angle for each twilight step.
   * @returns angle in degrees.
   */
  getTwilightStepAngle() {
    return this.#twilightStepAngle;
  }

  /**
   * Set the angle for each twilight step.
   * @param v angle in degrees.
   */
  setTwilightStepAngle(v: number) {
    this.#twilightStepAngle = v;
    this.#map?.triggerRepaint();
  }

  /**
   * Get the attenuation factor for each twilight step.
   * @returns 0.0 means no attenuation, 1.0 means full attenuation.
   */
  getTwilightAttenuation() {
    return this.#twilightAttenuation;
  }

  /**
   * Set the attenuation factor for each twilight step.
   * @param v 0.0 means no attenuation, 1.0 means full attenuation.
   */
  setTwilightAttenuation(v: number) {
    this.#twilightAttenuation = v;
    this.#map?.triggerRepaint();
  }

  onAdd(map: MaplibreMap, gl: WebGLRenderingContext) {
    const vertexSource = `
      precision highp float;

      attribute vec2 a_position;

      uniform mat4 u_matrix;

      varying vec2 v_position;

      void main() {
          vec4 tmp = u_matrix * vec4(a_position, 0.0, 1.0);
          gl_Position = tmp;
          v_position = a_position;
      }`;

    const fragmentSource = `
      precision highp float;

      uniform vec2 u_subsolar;
      uniform float u_opacity;
      uniform float u_twilight_step_angle;
      uniform float u_twilight_steps;
      uniform float u_twilight_attenuation;
      uniform vec3 u_color;

      varying vec2 v_position;

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

        gl_FragColor = vec4(u_color / 255., 1.0) * darkness * u_opacity;
      }`;

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    if (!vertexShader) throw new Error('failed to create vertex shader');
    gl.shaderSource(vertexShader, vertexSource);
    gl.compileShader(vertexShader);

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    if (!fragmentShader) throw new Error('failed to create fragment shader');
    gl.shaderSource(fragmentShader, fragmentSource);
    gl.compileShader(fragmentShader);

    this.#program = gl.createProgram()!;
    if (!this.#program) throw new Error('failed to create program');
    gl.attachShader(this.#program, vertexShader);
    gl.attachShader(this.#program, fragmentShader);
    gl.linkProgram(this.#program);

    this.#arrayBuffer = gl.createBuffer()!;

    this.#map = map;
  }

  onRemove() {
    this.#map = undefined;
    this.#arrayBuffer = undefined;
    this.#program = undefined;
  }

  render(gl: WebGLRenderingContext, matrix: mat4) {
    if (!this.#arrayBuffer) return;
    if (!this.#program) return;
    if (!this.#map) return;

    const { subsolarLng, subsolarLat } = this.getSubsolarPoint(this.#date || new Date());

    const [w, e] = this.#map.getBounds().toArray();
    const xmin = Math.floor(MercatorCoordinate.fromLngLat(w).x);
    const xmax = Math.ceil(MercatorCoordinate.fromLngLat(e).x);

    gl.useProgram(this.#program);

    gl.uniformMatrix4fv(gl.getUniformLocation(this.#program, 'u_matrix'), false, matrix);
    gl.uniform2fv(gl.getUniformLocation(this.#program, 'u_subsolar'), [subsolarLng, subsolarLat]);
    gl.uniform1f(gl.getUniformLocation(this.#program, 'u_opacity'), this.#opacity);
    gl.uniform3fv(gl.getUniformLocation(this.#program, 'u_color'), this.#color);
    gl.uniform1f(gl.getUniformLocation(this.#program, 'u_twilight_steps'), this.#twilightSteps);
    gl.uniform1f(gl.getUniformLocation(this.#program, 'u_twilight_step_angle'), this.#twilightStepAngle);
    gl.uniform1f(gl.getUniformLocation(this.#program, 'u_twilight_attenuation'), this.#twilightAttenuation);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.#arrayBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([xmin, 0, xmax, 0, xmin, 1, xmin, 1, xmax, 0, xmax, 1]),
      gl.DYNAMIC_DRAW,
    );
    const index = gl.getAttribLocation(this.#program, 'a_position');
    gl.enableVertexAttribArray(index);
    gl.vertexAttribPointer(index, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}

export default NightLayer;
