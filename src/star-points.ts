import { compileProgram } from './gl-program.js';

type Camera = {
  inverseProjection: number[];
  mercatorView: number[];
  globeView: number[];
};

/** Rasterize only stellar footprints; store flux, not radiance, in half floats. */
export class StarPoints {
  readonly texture: WebGLTexture;
  #program: WebGLProgram;
  #buffer: WebGLBuffer;
  #vao: WebGLVertexArrayObject;
  #framebuffer: WebGLFramebuffer;
  #count: number;
  #quads: boolean;
  #width = 0;
  #height = 0;
  #signature?: string;

  constructor(gl: WebGL2RenderingContext, stars: Float32Array<ArrayBuffer>) {
    this.#quads = (gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) as Float32Array)[1] < 7;
    const source = (shader: string) => (this.#quads ? shader.replace('\n', '\n#define STAR_QUADS\n') : shader);
    const program = compileProgram(gl, source(starFragment), source(starVertex));
    const texture = gl.createTexture(),
      buffer = gl.createBuffer(),
      vao = gl.createVertexArray(),
      framebuffer = gl.createFramebuffer();
    if (!texture || !buffer || !vao || !framebuffer) {
      gl.deleteProgram(program);
      gl.deleteTexture(texture);
      gl.deleteBuffer(buffer);
      gl.deleteVertexArray(vao);
      gl.deleteFramebuffer(framebuffer);
      throw new Error('Failed to allocate NightLayer point stars');
    }
    this.texture = texture;
    this.#program = program;
    this.#buffer = buffer;
    this.#vao = vao;
    this.#framebuffer = framebuffer;
    this.#count = stars.length / 9;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, stars, gl.STATIC_DRAW);
    for (let i = 0; i < 3; i++) {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, 3, gl.FLOAT, false, 36, i * 12);
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

  render(gl: WebGL2RenderingContext, camera: Camera, rotation: number[], mercatorWeight: number, years: number): void {
    const width = gl.drawingBufferWidth,
      height = gl.drawingBufferHeight;
    const signature = JSON.stringify([
      width,
      height,
      camera.inverseProjection,
      camera.mercatorView,
      camera.globeView,
      rotation,
      mercatorWeight,
      years,
    ]);
    if (signature === this.#signature) return;
    const framebuffer = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
    const clearColor = gl.getParameter(gl.COLOR_CLEAR_VALUE) as Float32Array;
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
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(this.#program);
      gl.bindVertexArray(this.#vao);
      gl.uniformMatrix4fv(
        gl.getUniformLocation(this.#program, 'u_inverse_projection'),
        false,
        camera.inverseProjection,
      );
      gl.uniformMatrix3fv(gl.getUniformLocation(this.#program, 'u_mercator_view'), false, camera.mercatorView);
      gl.uniformMatrix3fv(gl.getUniformLocation(this.#program, 'u_globe_view'), false, camera.globeView);
      gl.uniformMatrix3fv(gl.getUniformLocation(this.#program, 'u_celestial_rotation'), false, rotation);
      gl.uniform1f(gl.getUniformLocation(this.#program, 'u_mercator_weight'), mercatorWeight);
      gl.uniform1f(gl.getUniformLocation(this.#program, 'u_star_years'), years);
      gl.uniform2f(gl.getUniformLocation(this.#program, 'u_viewport'), width, height);
      if (this.#quads) gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, this.#count);
      else gl.drawArrays(gl.POINTS, 0, this.#count);
      this.#signature = signature;
    } finally {
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, framebuffer);
      gl.clearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
      gl.disable(gl.BLEND);
    }
  }

  dispose(gl: WebGL2RenderingContext): void {
    gl.deleteTexture(this.texture);
    gl.deleteFramebuffer(this.#framebuffer);
    gl.deleteBuffer(this.#buffer);
    gl.deleteVertexArray(this.#vao);
    gl.deleteProgram(this.#program);
  }
}

const starVertex = `#version 300 es
precision highp float;
layout(location = 0) in vec3 a_direction;
layout(location = 1) in vec3 a_flux;
layout(location = 2) in vec3 a_motion;
uniform mat4 u_inverse_projection;
uniform mat3 u_mercator_view, u_globe_view, u_celestial_rotation;
uniform float u_mercator_weight, u_star_years;
uniform vec2 u_viewport;
#ifdef STAR_QUADS
out vec2 v_pixel;
#else
flat out vec2 v_offset;
#endif
flat out vec3 v_flux;
void addRay(mat3 view, vec3 point, float weight, inout vec3 ray, inout vec3 dx, inout vec3 dy) {
  vec3 value = view * point;
  float distance = length(value);
  vec3 direction = value / distance;
  vec3 x = view[0] * u_inverse_projection[0][0];
  vec3 y = view[1] * u_inverse_projection[1][1];
  ray += direction * weight;
  dx += (x - direction * dot(direction, x)) * (weight / distance);
  dy += (y - direction * dot(direction, y)) * (weight / distance);
}
void main() {
  vec3 direction = transpose(u_celestial_rotation) * normalize(a_direction + a_motion * u_star_years);
  mat3 view = mat3(0.0);
  if (u_mercator_weight > 0.0) view += u_mercator_view / length(u_mercator_view[0]) * u_mercator_weight;
  if (u_mercator_weight < 1.0) view += u_globe_view / length(u_globe_view[0]) * (1.0 - u_mercator_weight);
  vec3 point = inverse(view) * direction;
  vec2 scale = vec2(u_inverse_projection[0][0], u_inverse_projection[1][1]);
  vec2 clip = (-point.xy / point.z - u_inverse_projection[3].xy) / scale;
  // Pure projections have a closed-form inverse. Refine the mixed viewing ray
  // during Globe/Mercator morphing, rather than changing the starfield's frame.
  if (u_mercator_weight > 0.0 && u_mercator_weight < 1.0) {
    vec3 tangentX = normalize(cross(direction, abs(direction.y) < .9 ? vec3(0,1,0) : vec3(1,0,0)));
    vec3 tangentY = cross(direction, tangentX);
    for (int i = 0; i < 4; i++) {
      vec3 ray = vec3(0), dx = vec3(0), dy = vec3(0);
      vec3 p = (u_inverse_projection * vec4(clip, 0, 1)).xyz;
      addRay(u_mercator_view, p, u_mercator_weight, ray, dx, dy);
      addRay(u_globe_view, p, 1.0 - u_mercator_weight, ray, dx, dy);
      mat2 jacobian = mat2(dot(dx,tangentX), dot(dx,tangentY), dot(dy,tangentX), dot(dy,tangentY));
      if (abs(determinant(jacobian)) > 1e-12)
        clip -= inverse(jacobian) * vec2(dot(ray,tangentX), dot(ray,tangentY));
    }
  }
  bool visible = point.z < 0.0 && !any(isnan(clip)) && !any(isinf(clip))
    && all(lessThanEqual(abs(clip), vec2(1.0) + 6.0 / u_viewport));
  v_flux = a_flux * 1e12;
#ifdef STAR_QUADS
  vec2 corner = vec2(float((gl_VertexID == 1 || gl_VertexID == 2 || gl_VertexID == 4) ? 1 : -1),
                     float((gl_VertexID == 2 || gl_VertexID == 4 || gl_VertexID == 5) ? 1 : -1));
  v_pixel = corner * 3.0;
  gl_Position = visible ? vec4(clip + v_pixel * 2.0 / u_viewport, 0, 1) : vec4(2,2,2,1);
#else
  // Point primitives are clipped by their centre, not by their footprint.
  // Move offscreen centres to the nearest edge pixel and undo that translation
  // in the fragment shader, so stars enter/leave the frame continuously.
  vec2 centre = (clip * .5 + .5) * u_viewport;
  vec2 clamped = clamp(centre, vec2(.5), u_viewport - .5);
  v_offset = clamped - centre;
  gl_PointSize = 7.0;
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
out vec4 fragColor;
void main() {
#ifdef STAR_QUADS
  vec2 pixel = v_pixel;
#else
  vec2 pixel = (gl_PointCoord - .5) * vec2(7.0, -7.0) + v_offset;
#endif
  float radius2 = dot(pixel, pixel);
  if (radius2 > 9.0) discard;
  float weight = exp(-radius2 / .72) / (6.28318530718 * .36);
  fragColor = vec4(v_flux * weight, 0);
}`;
