import { compileProgram, uniformLocation } from './gl.js';
import type { ProjectionCamera } from './surface.js';

/** Conservative camera-cone box in radially warped solar XY. */
function shadowBeamBox(
  camera: ProjectionCamera,
  axes: readonly number[][],
  reach: number,
  scale: number,
  radius: number,
  size: number,
): Float32Array {
  const frames = [camera.planarView, camera.sphereView].filter((view) => view.some((v) => v !== 0));
  if (frames.length !== 1) return new Float32Array([-1, -1, 2, 2]);
  const view = frames[0],
    p = camera.inverseProjection;
  const transform = (v: number[]) =>
    [0, 1, 2].map((row) => v.reduce((sum, n, column) => sum + view[column * 3 + row] * n, 0));
  const forward = transform([p[12], p[13], p[14]]),
    right = transform([p[0], 0, 0]),
    up = transform([0, p[5], 0]);
  const dot = (a: readonly number[], b: readonly number[]) => a.reduce((sum, v, i) => sum + v * b[i], 0);
  const normal = [
    right[1] * up[2] - right[2] * up[1],
    right[2] * up[0] - right[0] * up[2],
    right[0] * up[1] - right[1] * up[0],
  ];
  const minimum = Math.abs(dot(normal, forward)) / Math.hypot(...normal);
  if (!(minimum > 0)) return new Float32Array([-1, -1, 2, 2]);
  const result = new Float32Array(4);
  const padding = (2 * Math.PI + 0.5) / size;
  for (let i = 0; i < 2; i++) {
    const center = dot(axes[i], forward);
    const spread = Math.abs(dot(axes[i], right)) + Math.abs(dot(axes[i], up));
    const low = Math.min(0, Math.max(-reach, ((center - spread) * reach) / minimum));
    const high = Math.max(0, Math.min(reach, ((center + spread) * reach) / minimum));
    // Radial compression moves toward zero; axis-wise bounds enclose the warp.
    const lo = (scale * Math.asinh(low / scale)) / radius - padding;
    const hi = (scale * Math.asinh(high / scale)) / radius + padding;
    const width = (hi - lo) / (1 - 4 / size);
    result[i] = (lo + hi - width) * 0.5;
    result[i + 2] = width;
  }
  return result;
}

/** Epipolar slice range [start, extent], in turns. */
export function shadowAlphaRange(camera: ProjectionCamera, axes: readonly number[][], size: number): Float32Array {
  const full = new Float32Array([0, 1]);
  // A flat camera cone cannot bound physical rays to spherical endpoints.
  if (camera.planarView.some((v) => v !== 0)) return full;
  const frames = [camera.planarView, camera.sphereView].filter((view) => view.some((v) => v !== 0));
  if (frames.length !== 1) return full;
  const view = frames[0],
    p = camera.inverseProjection;
  const transform = (v: number[]) =>
    [0, 1, 2].map((row) => v.reduce((sum, n, column) => sum + view[column * 3 + row] * n, 0));
  const angles: number[] = [];
  for (const sx of [-1, 1])
    for (const sy of [-1, 1]) {
      const corner = transform([p[12] + sx * p[0], p[13] + sy * p[5], p[14]]);
      const x = axes[0].reduce((sum, v, i) => sum + v * corner[i], 0);
      const y = axes[1].reduce((sum, v, i) => sum + v * corner[i], 0);
      if (Math.hypot(x, y) < 1e-9 * Math.hypot(...corner)) return full;
      angles.push((((Math.atan2(y, x) / (2 * Math.PI)) % 1) + 1) % 1);
    }
  angles.sort((a, b) => a - b);
  let gap = angles[0] + 1 - angles[3],
    start = angles[0];
  for (let i = 1; i < 4; i++)
    if (angles[i] - angles[i - 1] > gap) {
      gap = angles[i] - angles[i - 1];
      start = angles[i];
    }
  if (gap <= 0.5) return full;
  const extent = (1 - gap) / (1 - 4 / size);
  if (extent >= 1) return full;
  return new Float32Array([(((start - (2 * extent) / size) % 1) + 1) % 1, extent]);
}

const shadowBeamGLSL: string = `
uniform vec4 u_shadow_beam_bounds;
vec2 shadowMapUV(vec2 warped){
  return (warped-u_shadow_beam_bounds.xy)/u_shadow_beam_bounds.zw;
}
vec2 shadowMapPoint(vec2 uv){
  return u_shadow_beam_bounds.xy+uv*u_shadow_beam_bounds.zw;
}
`;

/** Depth residuals relative to a sphere enclosing the observer and all casters.
 * Subtract the reference after interpolation; use radial asinh only for XY sampling. */
const shadowGeometryGLSL: string = `
uniform float u_shadow_scale;
float shadowAsinhRatio(float x){
  // A series preserves the linear core against float32 cancellation.
  float x2=x*x;
  return x<.1?1.0+x2*(-1.0/6.0+x2*3.0/40.0):log(x+sqrt(1.0+x2))/x;
}
float shadowWarpRadius(float radius){
  return u_shadow_scale>0.0?radius*shadowAsinhRatio(radius/u_shadow_scale):radius;
}
float shadowUnwarpRadius(float radius){
  if(u_shadow_scale==0.0)return radius;
  float x=radius/u_shadow_scale,x2=x*x;
  if(x<.1)return radius*(1.0+x2*(1.0/6.0+x2/120.0));
  float e=exp(x);
  return u_shadow_scale*.5*(e-1.0/e);
}
vec2 shadowWarpPoint(vec2 point){
  return u_shadow_scale>0.0?point*shadowAsinhRatio(length(point)/u_shadow_scale):point;
}
vec2 shadowUnwarpPoint(vec2 point){
  float radius=length(point);
  return radius>0.0?point*(shadowUnwarpRadius(radius)/radius):point;
}
uniform vec4 u_shadow_earth;
uniform float u_shadow_reference_excess;
uniform float u_shadow_reference_height;
uniform bool u_shadow_reference_clipped;
const float SHADOW_EMPTY=65504.0;
const float SHADOW_DEPTH_LOG_RANGE=11.7830137;
float shadowHardwareDepth(float residual){
  float a=abs(residual);
  float encoded=log(a+sqrt(a*a+1.0));
  return .5+sign(residual)*encoded/(2.0*SHADOW_DEPTH_LOG_RANGE);
}
float shadowReferenceDiscriminant(vec2 xy){
  if(u_shadow_reference_clipped){
    float radius=6371.0+u_shadow_reference_height;
    float transverse=length(u_shadow_earth.xy+xy);
    return (radius-transverse)*(radius+transverse);
  }
  return u_shadow_earth.w-dot(xy,xy+2.0*u_shadow_earth.xy);
}
float shadowReferenceDepth(vec2 xy){
  float d=shadowReferenceDiscriminant(xy);
  if(d<=0.0)return -u_shadow_earth.z;
  float root=sqrt(d);
  if(!u_shadow_reference_clipped&&u_shadow_earth.z<0.0)
    return (dot(xy,xy+2.0*u_shadow_earth.xy)-u_shadow_reference_excess)/(-u_shadow_earth.z+root);
  return -u_shadow_earth.z-root;
}
float shadowReceiverResidual(vec3 ray,float distance){
  float z=u_shadow_earth.z+ray.z*distance;
  float d=shadowReferenceDiscriminant(ray.xy*distance);
  if(d<=0.0)return z;
  float root=sqrt(d);
  float excess=distance*(2.0*dot(u_shadow_earth.xyz,ray)+dot(ray,ray)*distance)-u_shadow_reference_excess;
  if(u_shadow_reference_clipped){
    float radius=6371.0+u_shadow_reference_height;
    float endpointRadius=length(u_shadow_earth.xyz+ray*distance);
    excess=(endpointRadius-radius)*(endpointRadius+radius);
  }
  return z<0.0?excess/(z-root):z+root;
}
`;

export const TERRAIN_SHADOW_SIZE = 1024;
const SIZE = TERRAIN_SHADOW_SIZE;

/** The physical solar basis is independent of screen fitting. */
function solarAxes(sun: number[]): number[][] {
  const cross = (a: number[], b: number[]) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const length = Math.hypot(...sun);
  const light = sun.map((v) => v / length);
  const right = cross(Math.abs(light[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0], light);
  const rightLength = Math.hypot(...right);
  const x = right.map((v) => v / rightLength);
  // Toward the Sun is nearer (smaller depth).
  return [x, cross(light, x), light.map((v) => -v)];
}

/** Radially compressed samples of reference-sphere depth residuals in kilometres.
 * Smaller Z is closer to the Sun; 65504 means no sampled caster. The solar basis
 * is independent of the camera; the acquisition box follows its view. Both
 * acquisition and the tree retain camera-centred radial asinh coordinates.
 */
export const shadowProjectionGLSL: string = `
${shadowGeometryGLSL}
${shadowBeamGLSL}
uniform float u_shadow_radius; // asinh-compressed radial extent, km.
float shadowAlpha(vec2 p){
  return dot(p,p)>0.0?fract(atan(p.y,p.x)/6.283185307179586+1.0):0.0;
}
// Slice rows cover [start, start+extent) turns. A full circle keeps the
// repeating row coordinate; a camera-fitted range clamps to its edge rows.
uniform vec2 u_shadow_alpha_range;
float shadowRow(float alpha){
  if(u_shadow_alpha_range.y>=1.0)return alpha;
  float rows=float(${SIZE});
  return clamp(fract(alpha-u_shadow_alpha_range.x)/u_shadow_alpha_range.y,.5/rows,1.0-.5/rows);
}
// Turn angle of row coordinate r (row centre r=(i+.5)/rows).
float shadowRowAlpha(float r){
  return u_shadow_alpha_range.x+r*u_shadow_alpha_range.y;
}
vec2 shadowPhysicalPoint(vec2 uv){
  return shadowUnwarpPoint(shadowMapPoint(uv)*u_shadow_radius);
}
`;

export const shadowVertexGLSL: string = `
${shadowProjectionGLSL}
out highp float v_shadow_depth;
vec4 shadowPosition(vec3 light,float elevation) {
  v_shadow_depth=light.z;
  // Physical sample p/N is stored at texel center (p+.5)/N.
  return vec4(shadowMapUV(shadowWarpPoint(light.xy)/u_shadow_radius)*2.0-1.0+1.0/${SIZE.toFixed(1)},0,1);
}
`;

/** Deliberate vertex-only warp: raster edges are straight and depth is affine
 * in compressed XY. Reference subtraction happens after interpolation, so the
 * storage reference cannot bend the interpolated surface. Coverage/depth can
 * differ from the original native triangle.
 * Axis-spanning triangles are deliberately not split or reconstructed.
 * These acquisition errors are not included in the sampled-range uncertainty.
 */
export const shadowFragment: string = `#version 300 es
precision highp float;
in highp float v_shadow_depth;
${shadowProjectionGLSL}
out float shadowDepth;
void main() {
  vec2 xy=shadowPhysicalPoint((gl_FragCoord.xy-.5)/${SIZE.toFixed(1)});
  shadowDepth=v_shadow_depth-shadowReferenceDepth(xy);
  gl_FragDepth=shadowHardwareDepth(shadowDepth);
}
`;

/** Sample the analytic sea-level sphere into the same finite-resolution field. */
export const earthShadowFragment: string = `#version 300 es
precision highp float;
${shadowProjectionGLSL}
// Observer in solar coordinates and reference-sphere discriminant at gamma=0.
// CPU-evaluated invariants avoid subtracting squared Earth-sized GPU values.
out float depth;
void main() {
  vec2 uv=(gl_FragCoord.xy-.5)/${SIZE.toFixed(1)};
  vec2 xy=shadowPhysicalPoint(uv);
  float reference=shadowReferenceDiscriminant(xy);
  float h=u_shadow_reference_height;
  float excess=h*(12742.0+h);
  float discriminant=reference-excess;
  if(discriminant<0.0) discard;
  depth=reference>0.0?excess/(sqrt(reference)+sqrt(discriminant)):-sqrt(discriminant);
  gl_FragDepth=shadowHardwareDepth(depth);
}
`;

export class TerrainShadow {
  readonly texture: WebGLTexture;
  readonly framebuffer: WebGLFramebuffer;
  #depth: WebGLRenderbuffer;
  #earthProgram: WebGLProgram;
  #earth = new Float32Array(4);
  #referenceHeight = 0;
  #referenceExcess = 0;
  #referenceClipped = false;
  #beamBounds = new Float32Array([-1, -1, 2, 2]);
  #alphaRange = new Float32Array([0, 1]);
  #vao: WebGLVertexArrayObject;
  readonly matrix: Float32Array<ArrayBuffer> = new Float32Array(16);
  radius = 1;
  scale = 1;
  origin: number[] = [0, 0, 0];
  revision = 0;

  constructor(gl: WebGL2RenderingContext) {
    const texture = gl.createTexture();
    const framebuffer = gl.createFramebuffer();
    if (!texture || !framebuffer) throw new Error('Failed to allocate terrain shadow map');
    this.texture = texture;
    this.framebuffer = framebuffer;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, SIZE, SIZE, 0, gl.RED, gl.FLOAT, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    this.#depth = gl.createRenderbuffer()!;
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.#depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, SIZE, SIZE);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.#depth);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
      throw new Error('Incomplete terrain shadow framebuffer');
    this.#vao = gl.createVertexArray()!;
    this.#earthProgram = compileProgram(
      gl,
      earthShadowFragment,
      `#version 300 es
void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));gl_Position=vec4(p*2.0-1.0,0,1);}
`,
    );
  }

  begin(gl: WebGL2RenderingContext, camera: ProjectionCamera, sun: number[], maxCasterElevation: number): void {
    const observer = camera.origin;
    const axes = solarAxes(sun);
    // The core radius follows observer altitude, not receiver
    // moments or projection. The floor is one float32 Earth-radius ULP in km.
    this.scale = Math.max(Math.hypot(...observer) - 6371, 2 ** -11);
    // A segment outside the opaque sphere is at most the sum of the observer's
    // and terrain's tangent lengths. Its transverse radius cannot be larger.
    const height = Math.max(0, Math.hypot(...observer) - 6371);
    const casterHeight = Math.max(0, maxCasterElevation) * 0.001;
    const reach = Math.sqrt(height * (12742 + height)) + Math.sqrt(casterHeight * (12742 + casterHeight));
    this.radius = (this.scale * Math.asinh(Math.max(reach, this.scale) / this.scale)) / (1 - 2 / SIZE);
    this.#beamBounds.set(shadowBeamBox(camera, axes, reach, this.scale, this.radius, SIZE));
    this.#alphaRange.set(shadowAlphaRange(camera, axes, SIZE));
    const earthObserver = axes.map((axis) => axis.reduce((sum, v, i) => sum + v * observer[i], 0));
    const observerRadius = Math.hypot(...observer);
    const observerHeight = observerRadius - 6371;
    // No caster may rise above the reference sphere: a caster crossing its
    // solar silhouette meets the square-root edge of the reference depth.
    this.#referenceHeight = Math.min(Math.max(observerHeight, casterHeight), 100);
    this.#referenceClipped = observerHeight > 100;
    const referenceRadius = 6371 + this.#referenceHeight;
    const transverse = Math.hypot(earthObserver[0], earthObserver[1]);
    // Float64 invariants: radius excess, and the discriminant at the solar axis.
    this.#referenceExcess = (referenceRadius - observerRadius) * (referenceRadius + observerRadius);
    this.#earth.set([
      ...earthObserver,
      this.#referenceClipped
        ? (referenceRadius - transverse) * (referenceRadius + transverse)
        : earthObserver[2] ** 2 + this.#referenceExcess,
    ]);
    // Matrix outputs physical light-space kilometres, centred on the observer.
    // Radial sampling changes only XY density, not the physical light basis.
    this.matrix.fill(0);
    for (let row = 0; row < 3; row++) for (let i = 0; i < 3; i++) this.matrix[i * 4 + row] = axes[row][i];
    this.matrix[15] = 1;
    this.origin = observer.slice();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.viewport(0, 0, SIZE, SIZE);
    gl.disable(gl.SCISSOR_TEST);
    gl.depthMask(true);
    gl.clearDepth(1);
    gl.colorMask(true, true, true, true);
    gl.clearBufferfv(gl.COLOR, 0, new Float32Array([65504, 0, 0, 0]));
    gl.clear(gl.DEPTH_BUFFER_BIT);
  }

  /** Compose sea level into the same caster map, using the same depth order. */
  finish(gl: WebGL2RenderingContext): void {
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.viewport(0, 0, SIZE, SIZE);
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.STENCIL_TEST);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.bindVertexArray(this.#vao);
    gl.disable(gl.BLEND);
    gl.colorMask(true, true, true, true);
    gl.depthFunc(gl.LEQUAL);
    gl.useProgram(this.#earthProgram);
    this.bindProjection(gl, this.#earthProgram);
    gl.uniform4fv(uniformLocation(gl, this.#earthProgram, 'u_shadow_earth'), this.#earth);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.revision++;
  }

  bindProjection(gl: WebGL2RenderingContext, program: WebGLProgram): void {
    gl.uniformMatrix4fv(uniformLocation(gl, program, 'u_shadow_matrix'), false, this.matrix);
    gl.uniform3fv(uniformLocation(gl, program, 'u_shadow_origin'), this.origin);
    gl.uniform1f(uniformLocation(gl, program, 'u_shadow_radius'), this.radius);
    gl.uniform4fv(uniformLocation(gl, program, 'u_shadow_beam_bounds'), this.#beamBounds);
    gl.uniform2fv(uniformLocation(gl, program, 'u_shadow_alpha_range'), this.#alphaRange);
    gl.uniform1f(uniformLocation(gl, program, 'u_shadow_scale'), this.scale);
    gl.uniform1f(uniformLocation(gl, program, 'u_shadow_reference_height'), this.#referenceHeight);
    gl.uniform1f(uniformLocation(gl, program, 'u_shadow_reference_excess'), this.#referenceExcess);
    gl.uniform1i(uniformLocation(gl, program, 'u_shadow_reference_clipped'), this.#referenceClipped ? 1 : 0);
    gl.uniform4fv(uniformLocation(gl, program, 'u_shadow_earth'), this.#earth);
  }

  dispose(gl: WebGL2RenderingContext): void {
    gl.deleteTexture(this.texture);
    gl.deleteFramebuffer(this.framebuffer);
    gl.deleteRenderbuffer(this.#depth);
    gl.deleteProgram(this.#earthProgram);
    gl.deleteVertexArray(this.#vao);
  }
}

/** A beta min/max tree per epipolar slice of the filtered caster map. */
export class ShadowIntervals {
  // Keep alternate levels in their final targets. A reduction never samples its draw attachment.
  readonly #levels: { texture: WebGLTexture; framebuffer: WebGLFramebuffer }[];
  #vao: WebGLVertexArrayObject;
  #leaves: WebGLProgram;
  #sliceReduce: WebGLProgram;

  constructor(gl: WebGL2RenderingContext) {
    this.#leaves = compileProgram(gl, shadowLeavesFragment, shadowIntervalVertex);
    this.#sliceReduce = compileProgram(gl, shadowSliceReduceFragment, shadowIntervalVertex);
    this.#vao = gl.createVertexArray()!;
    this.#levels = [2 * SIZE, SIZE].map((width) => {
      const result = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, result);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, width, SIZE, 0, gl.RG, gl.FLOAT, null);
      const framebuffer = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, result, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error('Incomplete shadow bounds framebuffer');
      return { texture: result, framebuffer };
    });
  }
  update(gl: WebGL2RenderingContext, shadow: TerrainShadow): void {
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.#levels[0].framebuffer);
    gl.bindVertexArray(this.#vao);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
    gl.colorMask(true, true, true, true);
    gl.useProgram(this.#leaves);
    shadow.bindProjection(gl, this.#leaves);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, shadow.texture);
    gl.uniform1i(uniformLocation(gl, this.#leaves, 'u_shadow_depth'), 0);
    gl.viewport(SIZE, 0, SIZE, SIZE);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.useProgram(this.#sliceReduce);
    gl.activeTexture(gl.TEXTURE1);
    gl.uniform1i(uniformLocation(gl, this.#sliceReduce, 'u_shadow_bounds'), 1);
    let source = 0;
    for (let width = SIZE / 2; width >= 1; width /= 2) {
      gl.bindTexture(gl.TEXTURE_2D, this.#levels[source].texture);
      source = 1 - source;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.#levels[source].framebuffer);
      gl.viewport(width, 0, width, SIZE);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
  }
  bind(gl: WebGL2RenderingContext, program: WebGLProgram, shadow: TerrainShadow): void {
    shadow.bindProjection(gl, program);
    gl.activeTexture(gl.TEXTURE14);
    gl.bindTexture(gl.TEXTURE_2D, this.#levels[0].texture);
    gl.uniform1i(uniformLocation(gl, program, 'u_shadow_bounds'), 14);
    gl.activeTexture(gl.TEXTURE13);
    gl.bindTexture(gl.TEXTURE_2D, this.#levels[1].texture);
    gl.uniform1i(uniformLocation(gl, program, 'u_shadow_bounds_odd'), 13);
    gl.activeTexture(gl.TEXTURE15);
    gl.bindTexture(gl.TEXTURE_2D, shadow.texture);
    gl.uniform1i(uniformLocation(gl, program, 'u_shadow_depth'), 15);
  }
  dispose(gl: WebGL2RenderingContext): void {
    for (const { texture, framebuffer } of this.#levels) {
      gl.deleteTexture(texture);
      gl.deleteFramebuffer(framebuffer);
    }
    gl.deleteVertexArray(this.#vao);
    gl.deleteProgram(this.#leaves);
    gl.deleteProgram(this.#sliceReduce);
  }
}

const shadowIntervalVertex: string = `#version 300 es
void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));gl_Position=vec4(p*2.0-1.0,0,1);}`;
const shadowLeavesFragment: string = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D u_shadow_depth;
${shadowProjectionGLSL}
out vec2 bounds;
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy),size=textureSize(u_shadow_depth,0);
  float count=float(size.x),angle=shadowRowAlpha((float(p.y)+.5)/count)*6.283185307179586;
  float radius=(float(p.x-size.x)+.5)/count;
  vec2 warped=vec2(cos(angle),sin(angle))*radius;
  vec2 uv=shadowMapUV(warped);
  if(any(lessThan(uv,vec2(0)))||any(greaterThan(uv,vec2(1)-1.0/vec2(size)))){
    bounds=vec2(6.283185307179586);return;
  }
  float residual=texture(u_shadow_depth,uv+.5/vec2(size)).r;
  float gamma=shadowUnwarpRadius(radius*u_shadow_radius);
  float depth=shadowReferenceDepth(vec2(cos(angle),sin(angle))*gamma)+residual;
  bounds=vec2(atan(gamma,-depth));
}`;
const shadowSliceReduceFragment: string = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D u_shadow_bounds;
out vec2 bounds;
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy);
  p.x*=2;
  vec2 a=texelFetch(u_shadow_bounds,p,0).rg;
  vec2 b=texelFetch(u_shadow_bounds,p+ivec2(1,0),0).rg;
  bounds=vec2(min(a.x,b.x),max(a.y,b.y));
}`;

/** Buffer visibility changes, then evaluate cumulative Mie only at their boundaries.
 * Adjacent equal runs share a boundary; each change contributes (w_before − w_after) P(t).
 * Outside-map intervals stay visible, and the endpoint uses its already-known P value. */
export const terrainMieShadowGLSL: string = `
uniform bool u_terrain_mie_shadows;
uniform highp sampler2D u_shadow_bounds,u_shadow_bounds_odd,u_shadow_depth;
uniform mat4 u_shadow_matrix;
${shadowProjectionGLSL}
vec2 shadowNodeBounds(int node,int span,float alpha){
  // span=1,4,16,... lives with the leaves; span=2,8,32,... in the half-width atlas.
  float x=(float(node)+.5)/float(${2 * SIZE});
  if((span&0x55555555)!=0)return textureLod(u_shadow_bounds,vec2(x,alpha),0.0).rg;
  return textureLod(u_shadow_bounds_odd,vec2(2.0*x,alpha),0.0).rg;
}
float shadowTerminalWeight(vec3 ray,vec2 interval){
  float midpoint=interval.x+(interval.y-interval.x)*.5;
  vec2 uv=shadowMapUV(shadowWarpPoint(ray.xy*midpoint)/u_shadow_radius);
  if(any(lessThan(uv,vec2(0)))||any(greaterThan(uv,vec2(1)-1.0/vec2(textureSize(u_shadow_depth,0)))))return 0.0;
  uv+=.5/vec2(textureSize(u_shadow_depth,0));
  float residual=texture(u_shadow_depth,uv).r;
  return shadowReceiverResidual(ray,midpoint)>residual?1.0:0.0;
}
vec3 terrainMiePrefix(float distance);
vec3 terrainMiePrefixEnd(float distance);
float shadowCellDistance(int cell,int samples,float transverse,float distance){
  if(cell==0)return 0.0;
  return transverse==0.0?distance:min(shadowUnwarpRadius(float(cell)/float(samples)*u_shadow_radius)/transverse,distance);
}
vec3 terrainMieOcclusion(vec3 direction,float distance){
  if(distance<=0.0)return vec3(0);
  vec3 ray=(u_shadow_matrix*vec4(direction,0)).xyz;
  int samples=textureSize(u_shadow_bounds,0).y;
  float transverse=length(ray.xy);
  float endGamma=shadowWarpRadius(transverse*distance)/u_shadow_radius;
  float alpha=shadowRow(shadowAlpha(ray.xy));
  float beta=atan(transverse,-ray.z);
  // Stackless min/max traversal (Chen et al., Figure 5).
  int node=1,first=0,span=samples;
  vec3 removed=vec3(0);
  float cursor=0.0,current=0.0;
  bool finishing=false;
  vec4 events=vec4(0);
  int count=0;
  float sign=-1.0;
  while(true){
    while(!finishing&&count<4){
      float eventT=cursor,w=current;
      int eventCell=-1;
      if(first>=samples||transverse>0.0&&float(first)/float(samples)>=endGamma){cursor=shadowCellDistance(first,samples,transverse,distance);finishing=true;continue;}
      float gamma=float(first+span)/float(samples);
      bool terminal=gamma>=endGamma;
      vec2 bounds=span>1||!terminal?shadowNodeBounds(node,span,alpha):vec2(0);
      bool mixed=beta>=bounds.x&&beta<bounds.y;
      if(terminal&&span>1){
        if(mixed){node*=2;span/=2;continue;}
        int leaf=clamp(int(ceil(endGamma*float(samples)))-1,first,first+span-1);
        cursor=shadowCellDistance(first,samples,transverse,distance);
        if(leaf>first){
          float leafStart=transverse==0.0?distance:shadowUnwarpRadius(float(leaf)/float(samples)*u_shadow_radius)/transverse;
          vec2 run=vec2(cursor,min(leafStart,distance));
          if(run.y>run.x){eventT=run.x;w=beta>=bounds.y?1.0:0.0;}
          cursor=run.y;
        }
        node=samples+leaf;first=leaf;span=1;
        if(cursor>=distance)finishing=true;
      }else{
        if(!terminal&&mixed&&span>1){node*=2;span/=2;continue;}
        if(terminal){
          vec2 segment=vec2(shadowCellDistance(first,samples,transverse,distance),distance);
          float weight=shadowTerminalWeight(ray,segment);
          if(segment.y>segment.x){eventT=segment.x;w=weight;}
          cursor=distance;finishing=true;
        }else{
          w=beta>=bounds.y?1.0:0.0;eventCell=first;
          first+=span;node++;
          int up=int(log2(float(node&-node)));
          node>>=up;span<<=up;
          if(node==1){cursor=shadowCellDistance(first,samples,transverse,distance);finishing=true;}
        }
      }
      if(w!=current){
        if(eventCell>=0)eventT=shadowCellDistance(eventCell,samples,transverse,distance);
        if(eventT<distance){events[count]=eventT;count++;}
        else removed+=(current-w)*terrainMiePrefixEnd(distance);
        current=w;
      }
    }
    bool open=finishing&&current>0.0&&cursor<distance;
    if(open&&count<4){events[count]=cursor;count++;current=0.0;open=false;}
    for(int k=0;k<count;k++){removed+=sign*terrainMiePrefix(events[k]);sign=-sign;}
    count=0;
    if(finishing&&!open)break;
  }
  if(current>0.0)removed+=current*terrainMiePrefixEnd(distance);
  return removed;
}
`;
