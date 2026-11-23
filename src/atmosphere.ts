import { compileProgram, uniformLocation } from './gl.js';
import type { LUTData } from './lut.js';
import { terrainMieShadowGLSL } from './shadow.js';

export type Atmosphere = { sky: number; fog: number; stars: boolean };
export type AtmosphereInput =
  | boolean
  | number
  | {
      sky?: boolean | number;
      fog?: boolean | number;
      stars?: boolean;
    };
function strength(value: boolean | number): number {
  const n = typeof value === 'boolean' ? Number(value) : value;
  if (!Number.isFinite(n) || n < 0 || n > 1) throw new RangeError('Atmosphere strength must be between 0 and 1');
  return n;
}
/** Scalars set sky/fog together; object omissions default to fully enabled. */
export function atmosphereSettings(input: AtmosphereInput): Atmosphere {
  if (typeof input === 'number' || typeof input === 'boolean') {
    const value = strength(input);
    return { sky: value, fog: value, stars: value > 0 };
  }
  if (!input || typeof input !== 'object' || Array.isArray(input))
    throw new TypeError('Atmosphere must be a boolean, number or options object');
  const stars = input.stars ?? true;
  if (typeof stars !== 'boolean') throw new TypeError('Atmosphere stars must be boolean');
  return { sky: strength(input.sky ?? true), fog: strength(input.fog ?? true), stars };
}
/** Keep globe atmosphere; restore the physical Mercator model at zoom 8..10. */
export function atmosphereFade(zoom: number, globeWeight: number): number {
  const progress = Math.max(0, Math.min(1, (zoom - 8) / 2));
  const fade = progress * progress * (3 - 2 * progress);
  return fade + (1 - fade) * Math.max(0, Math.min(1, globeWeight));
}

/** Cubic B-spline basis of a clamped knot vector at x (Cox–de Boor): first control and weights. */
function cubicBasis(knots: readonly number[], x: number): { first: number; weights: number[] } {
  const controls = knots.length - 4;
  x = Math.min(Math.max(x, 0), 1 - 1e-12);
  let span = 3;
  while (span < controls - 1 && x >= knots[span + 1]) span++;
  const values = [1, 0, 0, 0],
    left = [0, 0, 0, 0],
    right = [0, 0, 0, 0];
  for (let degree = 1; degree <= 3; degree++) {
    left[degree] = x - knots[span + 1 - degree];
    right[degree] = knots[span + degree] - x;
    let saved = 0;
    for (let j = 0; j < degree; j++) {
      const term = values[j] / (right[j + 1] + left[degree - j]);
      values[j] = saved + right[j + 1] * term;
      saved = left[degree - j] * term;
    }
    values[degree] = saved;
  }
  return { first: span - 3, weights: values };
}

/** Basis rows at the given positions (or uniform x = i / (n - 1)): four weights and their
 * control indices. With mergeEnds, the first/last control pairs are merged (zero end derivative). */
function basisRows(knots: readonly number[], positions: number | readonly number[], mergeEnds = false): Float32Array {
  const controls = knots.length - 4;
  const xs =
    typeof positions === 'number' ? Array.from({ length: positions }, (_, i) => i / (positions - 1)) : positions;
  const rows = new Float32Array(xs.length * 8);
  for (let sample = 0; sample < xs.length; sample++) {
    const { first, weights } = cubicBasis(knots, xs[sample]);
    const merged = new Map<number, number>();
    for (let j = 0; j < 4; j++) {
      const control = mergeEnds ? Math.max(0, Math.min(controls - 3, first + j - 1)) : first + j;
      merged.set(control, (merged.get(control) ?? 0) + weights[j]);
    }
    let c = 0;
    for (const [control, weight] of merged) {
      rows[sample * 8 + c] = weight;
      rows[sample * 8 + 4 + c] = control;
      c++;
    }
  }
  return rows;
}

const ANGLE_MIN = Math.asinh(-180),
  ANGLE_SPAN = Math.asinh(200.1717139250062) - ANGLE_MIN;
/** Dense-grid nu: asinh of the relative elevation at a finer scale (degrees) than the
 * spline chart (0.5 degrees), concentrating samples at the horizon. */
function denseNuChart(scale: number): { low: number; inverseSpan: number } {
  const low = Math.asinh(-90 / scale);
  return { low, inverseSpan: 1 / (Math.asinh(100.0858569625031 / scale) - low) };
}
/** Spline-chart positions of the dense nu samples. */
function denseNuPositions(count: number, scale: number): number[] {
  const { low, inverseSpan } = denseNuChart(scale);
  return Array.from({ length: count }, (_, i) => {
    const degrees = scale * Math.sinh(low + i / (count - 1) / inverseSpan);
    return (Math.asinh(degrees / 0.5) - ANGLE_MIN) / ANGLE_SPAN;
  });
}

/** Resident low-rank scattering factors and the per-frame generator.
 *
 * C[(p,beta),(nu,s,phi)] = mean + sum_k a_k(p,beta) B_k(nu,s,phi), per field
 * [L_R, theta_R, L_M, theta_M]. For one observer/light, a_k comes from the cubic
 * (p, beta) spline; the slice is composed, then expanded into a dense [nu, s, phi]
 * grid that the pixel shader reads with one hardware-trilinear fetch.
 * Updates own drawing state; the caller rebinds its known draw framebuffer.
 * No state queries, uploads or readbacks occur during an update.
 */
class ScatteringGpu {
  // Resident: 0 A, 1 B (array), 2 mean, 3 basis rows, 4 sky nodes (array).
  // Scratch: 5 weights, 6 slice, 7 phi expansion, 8 dense nu expansion, 9 sky nu expansion.
  private readonly textures: WebGLTexture[] = [];
  private readonly programs: WebGLProgram[] = [];
  private readonly framebuffer: WebGLFramebuffer;
  private readonly vao: WebGLVertexArrayObject;
  private readonly rank: number;
  private phiCount = 0;
  private distanceCount = 0;
  private readonly knots: { p: number[]; beta: number[] };
  private readonly locations: Record<string, WebGLUniformLocation | null> = {};

  constructor(
    private readonly gl: WebGL2RenderingContext,
    lut: LUTData,
    finiteBoundary: WebGLTexture,
    skyVolume: WebGLTexture,
  ) {
    const { scattering, skyNodes } = lut;
    const layout = lut.metadata.scattering;
    if (scattering.length !== 4 || skyNodes.length !== 16 * 64 * 8 * 32 * 4)
      throw new Error('Invalid resident scattering dimensions');
    this.rank = Math.max(...scattering.map((field) => field.rank));
    this.knots = { p: layout.knots.p, beta: layout.knots.beta };
    this.framebuffer = gl.createFramebuffer()!;
    this.vao = gl.createVertexArray()!;
    if (!this.framebuffer || !this.vao) {
      gl.deleteFramebuffer(this.framebuffer);
      gl.deleteVertexArray(this.vao);
      throw new Error('Failed to allocate scattering generator');
    }
    // Construction is asynchronous with respect to the map's render loop.
    // Preserve shared state once here, never in the per-frame generator.
    const draw = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
    const viewport = gl.getParameter(gl.VIEWPORT) as Int32Array;
    const program = gl.getParameter(gl.CURRENT_PROGRAM) as WebGLProgram | null;
    const vao = gl.getParameter(gl.VERTEX_ARRAY_BINDING) as WebGLVertexArrayObject | null;
    const active = gl.getParameter(gl.ACTIVE_TEXTURE) as number;
    const mask = gl.getParameter(gl.COLOR_WRITEMASK) as boolean[];
    const capabilities = [
      gl.DEPTH_TEST,
      gl.STENCIL_TEST,
      gl.SCISSOR_TEST,
      gl.CULL_FACE,
      gl.BLEND,
      gl.RASTERIZER_DISCARD,
    ];
    const enabled = capabilities.map((capability) => gl.isEnabled(capability));
    const bindings = Array.from({ length: 10 }, (_, unit) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      return [
        gl.getParameter(gl.TEXTURE_BINDING_2D) as WebGLTexture | null,
        gl.getParameter(gl.TEXTURE_BINDING_2D_ARRAY) as WebGLTexture | null,
      ] as const;
    });
    try {
      const slices = 16 * 64,
        controls = 32 * 16 * 8,
        rank = this.rank;
      // RGBA = field. A: x = k, y = p * 64 + beta.  B: x = s * 8 + phi, y = nu, layer = k.
      const a = new Float16Array(rank * slices * 4),
        b = new Float16Array(controls * rank * 4),
        // Two adjacent 128 x 32 planes: mean and B[0]. Their Mie L lanes must
        // survive upload in float32; all higher components stay in RGBA16F.
        leading = new Float32Array(controls * 2 * 4);
      scattering.forEach((field, c) => {
        for (let i = 0; i < controls; i++) {
          leading[i * 4 + c] = field.mean[i];
          leading[(controls + i) * 4 + c] = field.b[i];
        }
        for (let row = 0; row < slices; row++)
          for (let k = 0; k < field.rank; k++) a[(row * rank + k) * 4 + c] = field.a[row * field.rank + k];
        for (let k = 1; k < field.rank; k++)
          for (let i = 0; i < controls; i++) b[(k * controls + i) * 4 + c] = field.b[k * controls + i];
      });
      this.texture(gl.TEXTURE_2D, rank, slices, 1, new Uint16Array(a.buffer));
      this.texture(gl.TEXTURE_2D_ARRAY, 128, 32, rank, new Uint16Array(b.buffer));
      this.texture(gl.TEXTURE_2D, 128, 64, 1, leading);
      // Rows: [0,1] nu64 and [2,3] phi16 (space entry); [4,5] dense nu, [6,7] dense s,
      // [8,9] dense phi. Each pair: four weights, then four control indices.
      const rows = new Float32Array(256 * 10 * 4);
      const put = (row: number, values: Float32Array) => {
        for (let x = 0; x < values.length / 8; x++)
          for (let i = 0; i < 4; i++) {
            rows[(row * 256 + x) * 4 + i] = values[x * 8 + i];
            rows[((row + 1) * 256 + x) * 4 + i] = values[x * 8 + 4 + i];
          }
      };
      const [nu, s, phi] = layout.render;
      [this.phiCount, this.distanceCount] = [phi, s];
      put(0, basisRows(layout.knots.nu, layout.boundaryRender[0]));
      put(2, basisRows(layout.knots.phi, layout.boundaryRender[1], true));
      put(4, basisRows(layout.knots.nu, denseNuPositions(nu, layout.renderNuScale)));
      put(6, basisRows(layout.knots.s, s));
      put(8, basisRows(layout.knots.phi, phi, true));
      this.texture(gl.TEXTURE_2D, 256, 10, 1, rows);
      const skyPacked = new Float16Array(skyNodes.length);
      for (let p = 0; p < 16; p++)
        for (let b = 0; b < 64; b++)
          for (let f = 0; f < 8; f++)
            for (let n = 0; n < 32; n++) {
              const from = (((p * 64 + b) * 8 + f) * 32 + n) * 4;
              const out = (((b * 16 + p) * 8 + f) * 32 + n) * 4;
              for (let c = 0; c < 4; c++) skyPacked[out + c] = skyNodes[from + c];
            }
      this.texture(gl.TEXTURE_2D_ARRAY, 32, 128, 64, new Uint16Array(skyPacked.buffer));
      this.texture(gl.TEXTURE_2D, rank, 1, 1, null);
      this.texture(gl.TEXTURE_2D, 128, 32, 1, null);
      this.texture(gl.TEXTURE_2D, phi, 512, 1, null);
      this.texture(gl.TEXTURE_2D, nu, phi * 16, 1, null);
      this.texture(gl.TEXTURE_2D, 128, 8 * 4, 1, null);
      const fragments = [
        scatteringWeightsFragment,
        scatteringComposeFragment,
        scatteringPhiFragment,
        scatteringNuFragment,
        scatteringDenseNuFragment,
        scatteringDenseDistanceFragment,
        scatteringSkyNuFragment,
        scatteringSkyPhiFragment,
      ];
      for (const fragment of fragments) this.programs.push(compileProgram(gl, fragment, scatteringVertex));
      const samplers: [number, string, number][] = [
        [0, 'u_a', 0],
        [1, 'u_b', 1],
        [1, 'u_mean', 2],
        [1, 'u_weights', 5],
        [2, 'u_slice', 6],
        [2, 'u_basis', 3],
        [3, 'u_phi', 7],
        [3, 'u_basis', 3],
        [4, 'u_phi', 7],
        [4, 'u_basis', 3],
        [5, 'u_nu', 8],
        [5, 'u_basis', 3],
        [6, 'u_sky', 4],
        [7, 'u_sky_nu', 9],
      ];
      for (const [i, name, unit] of samplers) {
        gl.useProgram(this.programs[i]);
        gl.uniform1i(uniformLocation(gl, this.programs[i], name), unit);
      }
      gl.useProgram(this.programs[1]);
      gl.uniform1i(uniformLocation(gl, this.programs[1], 'u_rank'), rank);
      for (const [key, i, name] of [
        ['first', 0, 'u_first'],
        ['pw', 0, 'u_pw'],
        ['bw', 0, 'u_bw'],
        ['densePhi', 5, 'u_phi_layer'],
        ['phiRow', 2, 'u_phi_row'],
        ['skyNode', 6, 'u_node'],
        ['skyFraction', 7, 'u_fraction'],
      ] as const)
        this.locations[key] = uniformLocation(gl, this.programs[i], name);
      this.prepare();
      this.attach2D(this.textures[5]);
      if (gl.checkFramebufferStatus(gl.DRAW_FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error('Incomplete scattering generator framebuffer');
      // The 100 km boundary retains beta slices for space-entry rays.
      const [, , betaCount] = layout.boundaryRender;
      for (let i = 0; i < betaCount; i++) {
        this.slice(1, i / (betaCount - 1), true);
        gl.framebufferTextureLayer(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, finiteBoundary, 0, i);
        this.draw(3, 128, 256);
        this.sky(skyVolume, i, 15, i);
      }
    } catch (error) {
      this.dispose();
      throw error;
    } finally {
      for (const [unit, binding] of bindings.entries()) {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, binding[0]);
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, binding[1]);
      }
      gl.activeTexture(active);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, draw);
      gl.viewport(viewport[0], viewport[1], viewport[2], viewport[3]);
      gl.useProgram(program);
      gl.bindVertexArray(vao);
      gl.colorMask(mask[0], mask[1], mask[2], mask[3]);
      for (const [i, capability] of capabilities.entries()) {
        if (enabled[i]) gl.enable(capability);
        else gl.disable(capability);
      }
    }
  }

  private texture(
    target: number,
    width: number,
    height: number,
    depth: number,
    data: Uint16Array | Float32Array | null,
  ): void {
    const gl = this.gl;
    const texture = gl.createTexture();
    if (!texture) throw new Error('Failed to allocate resident scattering texture');
    this.textures.push(texture);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(target, texture);
    gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const half = data instanceof Uint16Array;
    const format = half ? gl.RGBA16F : gl.RGBA32F;
    const type = half ? gl.HALF_FLOAT : gl.FLOAT;
    if (target === gl.TEXTURE_2D_ARRAY) gl.texImage3D(target, 0, format, width, height, depth, 0, gl.RGBA, type, data);
    else gl.texImage2D(target, 0, format, width, height, 0, gl.RGBA, type, data);
  }

  private prepare(): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, this.framebuffer);
    gl.bindVertexArray(this.vao);
    for (const capability of [
      gl.DEPTH_TEST,
      gl.STENCIL_TEST,
      gl.SCISSOR_TEST,
      gl.CULL_FACE,
      gl.BLEND,
      gl.RASTERIZER_DISCARD,
    ])
      gl.disable(capability);
    gl.colorMask(true, true, true, true);
    for (const [unit, texture] of this.textures.entries()) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(unit === 1 || unit === 4 ? gl.TEXTURE_2D_ARRAY : gl.TEXTURE_2D, texture);
    }
  }

  private attach2D(texture: WebGLTexture): void {
    this.gl.framebufferTexture2D(this.gl.DRAW_FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, texture, 0);
  }

  private draw(program: number, width: number, height: number, y = 0): void {
    const gl = this.gl;
    gl.useProgram(this.programs[program]);
    gl.viewport(0, y, width, height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Compose the slice controls at smooth-chart (p, beta) and expand them over phi
   * (16 space-entry samples, or the dense count). */
  private slice(p: number, beta: number, boundary = false): void {
    const gl = this.gl;
    const pb = cubicBasis(this.knots.p, p),
      bb = cubicBasis(this.knots.beta, beta);
    gl.useProgram(this.programs[0]);
    gl.uniform2i(this.locations.first, pb.first, bb.first);
    gl.uniform4fv(this.locations.pw, pb.weights);
    gl.uniform4fv(this.locations.bw, bb.weights);
    this.attach2D(this.textures[5]);
    this.draw(0, this.rank, 1);
    this.attach2D(this.textures[6]);
    this.draw(1, 128, 32);
    this.attach2D(this.textures[7]);
    gl.useProgram(this.programs[2]);
    gl.uniform1i(this.locations.phiRow, boundary ? 2 : 8);
    this.draw(2, boundary ? 16 : this.phiCount, 512);
  }

  /** Sky-exit anchors stay on the warped observer node grid. */
  private sky(target: WebGLTexture, layer: number, p: number, beta: number): void {
    const gl = this.gl;
    const pl = Math.floor(p),
      bl = Math.floor(beta);
    this.attach2D(this.textures[9]);
    gl.useProgram(this.programs[6]);
    gl.uniform2i(this.locations.skyNode, pl, bl);
    this.draw(6, 128, 32);
    gl.framebufferTextureLayer(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, target, 0, layer);
    gl.useProgram(this.programs[7]);
    // Split coordinates on the CPU so tiny fractions near high-index nodes do
    // not disappear when a full p/beta coordinate is converted to float32.
    gl.uniform2f(this.locations.skyFraction, p - pl, beta - bl);
    this.draw(7, 128, 32);
  }

  /** Update the dense finite grid and the sky volume's two observer/light slices. */
  update(
    target: WebGLTexture,
    skyVolume: WebGLTexture,
    light: number,
    finite: readonly [number, number],
    sky: readonly [number, number],
  ): void {
    const gl = this.gl;
    this.prepare();
    this.slice(finite[0], finite[1]);
    const phiCount = this.phiCount;
    this.attach2D(this.textures[8]);
    this.draw(4, 128, phiCount * 16);
    gl.useProgram(this.programs[5]);
    for (let phi = 0; phi < phiCount; phi++) {
      gl.framebufferTextureLayer(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, target, 0, light * phiCount + phi);
      gl.uniform1i(this.locations.densePhi, phi);
      this.draw(5, 128, this.distanceCount);
    }
    this.sky(skyVolume, 64 + light, sky[0], sky[1]);
  }

  dispose(): void {
    const gl = this.gl;
    for (const texture of this.textures) gl.deleteTexture(texture);
    for (const program of this.programs) gl.deleteProgram(program);
    gl.deleteFramebuffer(this.framebuffer);
    gl.deleteVertexArray(this.vao);
  }
}

const scatteringVertex: string = `#version 300 es
void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));gl_Position=vec4(p*2.0-1.0,0,1);}`;

const scatteringHeader = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DArray;
out vec4 color;
`;

// a_k(p, beta) for every rank k: the cubic (p, beta) spline of A, weights from the CPU.
const scatteringWeightsFragment: string = `${scatteringHeader}
uniform sampler2D u_a;
uniform ivec2 u_first;
uniform vec4 u_pw,u_bw;
void main(){
  int k=int(gl_FragCoord.x);
  color=vec4(0);
  for(int i=0;i<4;i++)for(int j=0;j<4;j++)
    color+=u_pw[i]*u_bw[j]*texelFetch(u_a,ivec2(k,(u_first.x+i)*64+u_first.y+j),0);
}`;

// Slice controls: mean + sum_k a_k B_k, x = s * 8 + phi, y = nu.
const scatteringComposeFragment: string = `${scatteringHeader}
uniform sampler2D u_weights,u_mean;
uniform sampler2DArray u_b;
uniform int u_rank;
void main(){
  ivec2 pixel=ivec2(gl_FragCoord.xy);
  vec4 value=texelFetch(u_mean,pixel,0);
  value+=texelFetch(u_weights,ivec2(0),0)*texelFetch(u_mean,pixel+ivec2(0,32),0);
  for(int k=1;k<u_rank;k++)value+=texelFetch(u_weights,ivec2(k,0),0)*texelFetch(u_b,ivec3(pixel,k),0);
  color=value;
}`;

// Phi expansion to uniform chart samples: x = phi, y = nu * 16 + s.
const scatteringPhiFragment: string = `${scatteringHeader}
uniform sampler2D u_slice,u_basis;
uniform int u_phi_row;
vec4 control(int n,int s,int f){
  vec4 v=texelFetch(u_slice,ivec2(s*8+f,n),0);
  v.xz=clamp(v.xz,vec2(-40),vec2(3,2.5));
  return v;
}
void main(){
  ivec2 pixel=ivec2(gl_FragCoord.xy);
  int n=pixel.y/16,s=pixel.y%16;
  vec4 weights=texelFetch(u_basis,ivec2(pixel.x,u_phi_row),0);
  ivec4 indices=ivec4(texelFetch(u_basis,ivec2(pixel.x,u_phi_row+1),0));
  color=vec4(0);
  for(int i=0;i<4;i++)color+=weights[i]*control(n,s,indices[i]);
}`;

// Neither float render-target conversion nor packHalf2x16 promises the same
// rounding as Float16Array on every backend. Produce exact half-representable
// float32 values using round-to-nearest, ties-to-even before storing them.
const roundHalfGLSL = `
float roundHalf(float value){
  uint bits=floatBitsToUint(value),signBit=bits&0x80000000u;
  uint magnitude=bits&0x7fffffffu;
  float a=uintBitsToFloat(magnitude);
  if(a<0.00006103515625){
    float rounded=roundEven(a*16777216.0)*0.000000059604644775390625;
    return uintBitsToFloat(floatBitsToUint(rounded)|signBit);
  }
  if(a>=65520.0)return uintBitsToFloat(signBit|0x7f800000u);
  magnitude=(magnitude+0xfffu+((magnitude>>13u)&1u))&0xffffe000u;
  return uintBitsToFloat(magnitude|signBit);
}
vec4 roundHalf(vec4 value){
  return vec4(roundHalf(value.x),roundHalf(value.y),roundHalf(value.z),roundHalf(value.w));
}`;

// Space-entry controls: nu64 x phi16 tiles, one tile per species/control; Z is beta.
const scatteringNuFragment: string = `${scatteringHeader}
uniform sampler2D u_phi,u_basis;
${roundHalfGLSL}
void main(){
  ivec2 pixel=ivec2(gl_FragCoord.xy);
  int logical=pixel.x/64+(pixel.y/16)*2;
  int s=logical%16,species=logical/16;
  int v=pixel.x%64,f=pixel.y%16;
  int first=int(texelFetch(u_basis,ivec2(v,1),0).x);
  vec4 weights=texelFetch(u_basis,ivec2(v,0),0);
  vec4 a=vec4(0),b=vec4(0);
  for(int i=0;i<4;i++){
    a+=weights[i]*texelFetch(u_phi,ivec2(f,(first+i)*16+s),0);
    b+=weights[i]*texelFetch(u_phi,ivec2(f,(first+i)*16+min(s+1,15)),0);
  }
  vec4 value=species==0?vec4(a.xy,b.xy):vec4(a.zw,b.zw);
  color=roundHalf(value);
}`;

// Dense nu expansion: x = dense nu sample, y = phi * 16 + s control.
const scatteringDenseNuFragment: string = `${scatteringHeader}
uniform sampler2D u_phi,u_basis;
void main(){
  ivec2 pixel=ivec2(gl_FragCoord.xy);
  int f=pixel.y/16,s=pixel.y%16;
  int first=int(texelFetch(u_basis,ivec2(pixel.x,5),0).x);
  vec4 weights=texelFetch(u_basis,ivec2(pixel.x,4),0);
  color=vec4(0);
  for(int i=0;i<4;i++)color+=weights[i]*texelFetch(u_phi,ivec2(f,(first+i)*16+s),0);
}`;

// Dense distance expansion into one phi layer of the 3D grid: x = dense nu, y = dense s.
const scatteringDenseDistanceFragment: string = `${scatteringHeader}
uniform sampler2D u_nu,u_basis;
uniform int u_phi_layer;
${roundHalfGLSL}
void main(){
  ivec2 pixel=ivec2(gl_FragCoord.xy);
  int first=int(texelFetch(u_basis,ivec2(pixel.y,7),0).x);
  vec4 weights=texelFetch(u_basis,ivec2(pixel.y,6),0);
  vec4 value=vec4(0);
  for(int i=0;i<4;i++)value+=weights[i]*texelFetch(u_nu,ivec2(pixel.x,u_phi_layer*16+first+i),0);
  color=roundHalf(value);
}`;

const skyPchipGLSL = `
vec4 slope(vec4 a,vec4 b){return mix(vec4(0),2.0*a*b/(a+b),greaterThan(a*b,vec4(0)));}
vec4 pchip(vec4 a,vec4 b,vec4 c,vec4 d,float t,int i,int count){
  vec4 m0=i==0?c-b:slope(b-a,c-b),m1=i==count-2?c-b:slope(c-b,d-c);
  return (2.0*t*t*t-3.0*t*t+1.0)*b+(-2.0*t*t*t+3.0*t*t)*c
    +(t*t*t-2.0*t*t+t)*m0+(t*t*t-t*t)*m1;
}`;

// Four blocks of eight phi rows, one block per (p,beta) corner, in float32.
const scatteringSkyNuFragment: string = `${scatteringHeader}
uniform sampler2DArray u_sky;
uniform ivec2 u_node;
${skyPchipGLSL}
void main(){
  int row=int(gl_FragCoord.y),corner=row/8;
  ivec2 node=min(u_node+ivec2(corner/2,corner%2),ivec2(15,63));
  float x=((gl_FragCoord.x-.5)/127.0)*31.0;
  int n=int(min(floor(x),30.0));
  int y=node.x*8+row%8;
  color=pchip(
    texelFetch(u_sky,ivec3(max(n-1,0),y,node.y),0),
    texelFetch(u_sky,ivec3(n,y,node.y),0),
    texelFetch(u_sky,ivec3(n+1,y,node.y),0),
    texelFetch(u_sky,ivec3(min(n+2,31),y,node.y),0),x-float(n),n,32);
}`;

// PCHIP is nonlinear: finish phi at each corner before mixing p and beta.
const scatteringSkyPhiFragment: string = `${scatteringHeader}
uniform sampler2D u_sky_nu;
uniform vec2 u_fraction;
${roundHalfGLSL}
${skyPchipGLSL}
vec4 node(int corner,int n,int f,float t){
  int y=corner*8;
  return pchip(texelFetch(u_sky_nu,ivec2(n,y+max(f-1,0)),0),
    texelFetch(u_sky_nu,ivec2(n,y+f),0),texelFetch(u_sky_nu,ivec2(n,y+f+1),0),
    texelFetch(u_sky_nu,ivec2(n,y+min(f+2,7)),0),t,f,8);
}
void main(){
  int n=int(gl_FragCoord.x);
  float x=((gl_FragCoord.y-.5)/31.0)*7.0;
  int f=int(min(floor(x),6.0));
  float fraction=x-float(f);
  vec2 t=u_fraction;
  vec4 value=mix(mix(node(0,n,f,fraction),node(1,n,f,fraction),t.y),
    mix(node(2,n,f,fraction),node(3,n,f,fraction),t.y),t.x);
  value=vec4(exp(value.x),value.y,exp(value.z),value.w);
  color=roundHalf(value);
}`;

// Refraction: CPU apparent light and the matching celestial shader atlas.
function refractedLight(
  table: Float32Array | undefined,
  origin: ArrayLike<number>,
  light: ArrayLike<number>,
): number[] {
  const direction = Array.from(light);
  const radius = Math.hypot(...Array.from(origin));
  const height = Math.max(radius - 6371, 0);
  if (!table || height >= 100) return direction;
  const up = Array.from(origin, (v) => v / radius);
  const mu = Math.max(
    -1,
    Math.min(
      1,
      up.reduce((sum, v, i) => sum + v * direction[i], 0),
    ),
  );
  const p = 63 * Math.cbrt(Math.min(height / 100, 1));
  const first = 63 * Math.cbrt(0.002 / 100);
  const y = p < 2 ? 1 + Math.max(p - first, 0) / (2 - first) : p;
  const sample = (x: number, channel: number): number => {
    const xi = Math.min(Math.floor(x), 126),
      yi = Math.min(Math.floor(y), 62);
    const at = (xx: number, yy: number) => table[((2 * 64 + yy) * 128 + xx) * 4 + channel];
    const a = at(xi, yi) * (1 - x + xi) + at(xi + 1, yi) * (x - xi);
    const b = at(xi, yi + 1) * (1 - x + xi) + at(xi + 1, yi + 1) * (x - xi);
    return a * (1 - y + yi) + b * (y - yi);
  };
  const elevation = Math.asin(mu);
  const relative = Math.max(((elevation - sample(0, 2)) * 180) / Math.PI, 0);
  const x = 127 * Math.min(Math.log1p(relative / 0.0001) / Math.log(1800001), 1);
  const angle = elevation + sample(x, 0);
  const horizontal = direction.map((v, i) => v - up[i] * mu);
  const length = Math.hypot(...horizontal);
  return length < 1e-6 ? direction : horizontal.map((v, i) => (v / length) * Math.cos(angle) + up[i] * Math.sin(angle));
}

/** Full-path columns, exit map and local lighting; finite extinction uses straight-ray tails. */
export const refractionGLSL: string = `
uniform bool u_refraction;
uniform highp sampler2DArray u_refraction_lut;
vec3 atmosphereBackgroundOrigin,atmosphereBackgroundDirection;
float refractionElevation(float mu) {
  return atan(mu,sqrt(max((1.0-mu)*(1.0+mu),0.0)));
}
float refractionHeight(float h){
  float p=63.0*pow(clamp(h/100.0,0.0,1.0),1.0/3.0);
  float first=63.0*pow(.002/100.0,1.0/3.0);
  return p<2.0?1.0+max(p-first,0.0)/(2.0-first):p;
}
vec2 refractionUV(float h,float mu) {
  float tangent=sqrt(max(h*(12742.0+h),0.0));
  float horizontal=sqrt(max((1.0-mu)*(1.0+mu),0.0));
  // Form the small relative angle before atan to retain limb precision.
  float relative=degrees(atan(mu*6371.0+horizontal*tangent,
                               horizontal*6371.0-mu*tangent));
  float x=relative<0.0
    ?63.0*(1.0-log(1.0+sqrt(-relative/.02))/log(1.0+sqrt(4500.0)))
    :64.0+63.0*log(1.0+relative/.02)/log(5006.0);
  float y=refractionHeight(h);
  return (vec2(clamp(x,0.0,127.0),y)+.5)/vec2(128,64);
}
vec3 refractionSkyColumns(float h,float mu) {
  vec2 uv=refractionUV(h,mu);
  uv.x=max(uv.x,64.5/128.0);
  vec3 columns=texture(u_refraction_lut,vec3(uv,0.0)).rgb;
  return max(vec3(exp(columns.xy),columns.z),vec3(0));
}
void refractionBackground(float h,vec3 up,vec3 direction) {
  float mu=clamp(dot(up,direction),-1.0,1.0);
  vec2 uv=refractionUV(h,mu);
  // Celestial rays stay on the escaping branch, including exact tangency.
  uv.x=max(uv.x,64.5/128.0);
  vec4 exit=texture(u_refraction_lut,vec3(uv,1.0));
  exit.x+=refractionElevation(mu);
  vec3 horizontal=direction-up*mu;
  float horizontalLength=length(horizontal);
  if(horizontalLength<1e-6) {
    atmosphereBackgroundOrigin=6471.0*up;
    atmosphereBackgroundDirection=direction;
    return;
  }
  horizontal/=horizontalLength;
  atmosphereBackgroundOrigin=6471.0*(cos(exit.y)*up+sin(exit.y)*horizontal);
  atmosphereBackgroundDirection=cos(exit.x)*horizontal+sin(exit.x)*up;
}
vec3 refractionLight(float h,vec3 up,vec3 light) {
  if(!u_refraction||h>=100.0)return light;
  float y=(refractionHeight(h)+.5)/64.0;
  float lower=texture(u_refraction_lut,vec3(.5/128.0,y,2.0)).z;
  float mu=clamp(dot(up,light),-1.0,1.0);
  float elevation=refractionElevation(mu);
  float relative=max(degrees(elevation-lower),0.0);
  float x=log(1.0+relative/.0001)/log(1800001.0);
  float angle=elevation+texture(u_refraction_lut,vec3((clamp(x,0.0,1.0)*127.0+.5)/128.0,y,2.0)).x;
  vec3 horizontal=light-up*mu;
  return length(horizontal)<1e-6?light:normalize(horizontal)*cos(angle)+up*sin(angle);
}
vec3 refractionExternal(vec3 origin,float h,vec3 direction) {
  atmosphereBackgroundOrigin=origin;atmosphereBackgroundDirection=direction;
  if(h>100.0){
    float along=dot(origin,direction),impact=length(cross(origin,direction));
    if(along>=0.0||impact>=6471.0)return direction;
    float root=sqrt(max((6471.0-impact)*(6471.0+impact),0.0));
    float entry=(h-100.0)*(h+12842.0)/max(-along+root,1e-20);
    origin+=entry*direction;h=100.0;
  }
  refractionBackground(max(h,0.0),normalize(origin),direction);
  return atmosphereBackgroundDirection;
}
vec3 refractionScreenRay(vec3 origin,float h,vec3 exterior) {
  if(!u_refraction)return exterior;
  vec3 up=normalize(origin);
  float mu=dot(up,exterior);
  vec3 horizontal=exterior-up*mu;
  if(length(horizontal)<1e-6)return exterior;
  horizontal=normalize(horizontal);
  float lo=-atan(sqrt(h*(12742.0+h)),6371.0),hi=1.5707963267948966;
  float target=refractionElevation(mu);
  vec3 tangent=cos(lo)*horizontal+sin(lo)*up;
  if(refractionElevation(clamp(dot(up,refractionExternal(origin,h,tangent)),-1.0,1.0))>target)return -up;
  for(int i=0;i<20;i++){
    float mid=(lo+hi)*.5;
    vec3 ray=cos(mid)*horizontal+sin(mid)*up;
    float actual=refractionElevation(clamp(dot(up,refractionExternal(origin,h,ray)),-1.0,1.0));
    if(actual<target)lo=mid;else hi=mid;
  }
  float angle=(lo+hi)*.5;
  return cos(angle)*horizontal+sin(angle)*up;
}
`;

export type DirectionTransportFrame = {
  origin: ArrayLike<number>;
  solarDirection: ArrayLike<number>;
  moonPosition: ArrayLike<number>;
  moonRatio: number;
};

/** Shadow-pass decimation per axis on the logical sensor grid (1 = full). */
export const MIE_SHADOW_STEP = 2;

const LUT_OBSERVER_TOLERANCE_KM = 0.1 / 1000;

/** Resident scattering curves, expanded on the GPU for each observer/light pair. */
export class DirectionTransport {
  private readonly textures: WebGLTexture[] = [];
  private readonly previous = Array.from({ length: 8 }, () => NaN);
  private observerOrigin?: number[];
  private readonly scatteringGpu: ScatteringGpu;
  private disposed = false;

  constructor(
    private readonly gl: WebGL2RenderingContext,
    private readonly lut: LUTData,
  ) {
    if (!gl.getExtension('OES_texture_float_linear'))
      throw new Error('Scattering coefficient filtering requires OES_texture_float_linear');
    const layout = lut.metadata.observer;
    const binding = gl.getParameter(gl.TEXTURE_BINDING_2D_ARRAY);
    const binding2D = gl.getParameter(gl.TEXTURE_BINDING_2D);
    const binding3D = gl.getParameter(gl.TEXTURE_BINDING_3D);
    try {
      this.scatteringGpu = this.unpack(() => {
        // Keep the dense finite sampler's depth and normalized coordinates unchanged.
        // The last two layers are unused now that the sky has its own volume.
        const [nu, s, phi] = lut.metadata.scattering.render;
        this.image(gl.TEXTURE_3D, nu, s, 2 * phi + 2, null, true, true);
        const axes = new Float32Array(128 * 2 * 4);
        for (let i = 0; i < 65; i++) for (let a = 0; a < 4; a++) axes[i * 4 + a] = layout.axisCDF[a][i];
        for (let a = 0; a < 4; a++)
          for (let i = 0; i < layout.counts[a]; i++) axes[(128 + i) * 4 + a] = layout.axisNodes[a][i];
        this.image(gl.TEXTURE_2D, 128, 2, 1, axes, false);
        const tail = lut.metadata.densityIntegrals;
        this.image(gl.TEXTURE_2D, tail.width, tail.height, 1, lut.optical, true);
        this.image(gl.TEXTURE_2D, ...layout.transmission.size, 1, lut.transmission, true);
        this.image(gl.TEXTURE_2D_ARRAY, ...lut.metadata.palette.color.size, lut.palette, true);
        // Space-entry finite controls: XY tiles for species/distance; Z is beta.
        this.image(gl.TEXTURE_3D, 128, 256, 64, null, true, true);
        // Sky exits share one volume: 64 space-entry beta slices, then Sun/Moon observer slices.
        this.image(gl.TEXTURE_3D, 128, 32, 66, null, true, true);
        if (lut.refraction) this.image(gl.TEXTURE_2D_ARRAY, 128, 64, 3, lut.refraction, true);
        return new ScatteringGpu(gl, lut, this.textures[5], this.textures[6]);
      });
    } catch (error) {
      this.dispose();
      throw error;
    } finally {
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, binding);
      gl.bindTexture(gl.TEXTURE_2D, binding2D);
      gl.bindTexture(gl.TEXTURE_3D, binding3D);
    }
  }

  apparentLight(origin: ArrayLike<number>, direction: ArrayLike<number>): number[] {
    return refractedLight(this.lut.refraction, origin, direction);
  }

  private unpack<T>(action: () => T): T {
    const gl = this.gl;
    const names = [
      gl.UNPACK_ALIGNMENT,
      gl.UNPACK_ROW_LENGTH,
      gl.UNPACK_IMAGE_HEIGHT,
      gl.UNPACK_SKIP_PIXELS,
      gl.UNPACK_SKIP_ROWS,
      gl.UNPACK_SKIP_IMAGES,
      gl.UNPACK_FLIP_Y_WEBGL,
      gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,
    ];
    const saved = names.map((name) => gl.getParameter(name));
    const buffer = gl.getParameter(gl.PIXEL_UNPACK_BUFFER_BINDING);
    try {
      gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null);
      names.forEach((name) => gl.pixelStorei(name, name === gl.UNPACK_ALIGNMENT ? 1 : 0));
      return action();
    } finally {
      names.forEach((name, i) => gl.pixelStorei(name, saved[i]));
      gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, buffer);
    }
  }

  private image(
    target: number,
    width: number,
    height: number,
    depth: number,
    pixels: Float32Array | Uint16Array | null,
    linear: boolean,
    half = pixels instanceof Uint16Array,
  ): void {
    const gl = this.gl;
    const limit = gl.getParameter(target === gl.TEXTURE_3D ? gl.MAX_3D_TEXTURE_SIZE : gl.MAX_TEXTURE_SIZE);
    if (Math.max(width, height, target === gl.TEXTURE_3D ? depth : 0) > limit)
      throw new Error('Scattering LUT exceeds GPU limits');
    const texture = gl.createTexture();
    if (!texture) throw new Error('Failed to allocate scattering texture');
    this.textures.push(texture);
    gl.bindTexture(target, texture);
    for (const name of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER])
      gl.texParameteri(target, name, linear ? gl.LINEAR : gl.NEAREST);
    for (const name of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R])
      gl.texParameteri(target, name, gl.CLAMP_TO_EDGE);
    if (target === gl.TEXTURE_2D_ARRAY || target === gl.TEXTURE_3D)
      gl.texImage3D(
        target,
        0,
        half ? gl.RGBA16F : gl.RGBA32F,
        width,
        height,
        depth,
        0,
        gl.RGBA,
        pixels instanceof Uint16Array ? gl.HALF_FLOAT : gl.FLOAT,
        pixels,
      );
    else gl.texImage2D(target, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, pixels);
  }

  private warp(x: number, axis: number): number {
    const row = this.lut.metadata.observer.axisCDF[axis];
    const index = Math.max(0, Math.min(1, x)) * 64;
    const lo = Math.min(Math.floor(index), 63);
    const value = row[lo] + (row[lo + 1] - row[lo]) * (index - lo);
    const nodes = this.lut.metadata.observer.axisNodes[axis];
    let lower = 0,
      upper = nodes.length - 1;
    while (upper - lower > 1) {
      const mid = (lower + upper) >> 1;
      if (nodes[mid] <= value) lower = mid;
      else upper = mid;
    }
    return (lower + (value - nodes[lower]) / (nodes[upper] - nodes[lower])) / (nodes.length - 1);
  }

  update(frame: DirectionTransportFrame): void {
    if (this.disposed) throw new Error('Directional transport has been disposed');
    // Anchor only the LUT observer: pure rotation can perturb the recovered camera position.
    const camera = frame.origin;
    let origin = this.observerOrigin;
    if (
      !origin ||
      Math.hypot(camera[0] - origin[0], camera[1] - origin[1], camera[2] - origin[2]) > LUT_OBSERVER_TOLERANCE_KM
    )
      this.observerOrigin = origin = Array.from(camera);
    const radius = Math.hypot(...origin);
    const height = Math.max(0, Math.min(100, radius - 6371));
    const horizon = Math.sqrt(height * (12742 + height));
    const up = origin.map((v) => v / radius);
    const moon = Array.from(frame.moonPosition, (v, i) => v - origin[i]);
    const length = Math.hypot(...moon);
    const [np, , , nb] = this.lut.metadata.observer.counts;
    let p = this.warp(horizon / Math.sqrt(1284200), 0) * (np - 1);
    const h0 = this.lut.metadata.scattering.p0Height;
    const p0 = this.warp(Math.sqrt(h0 * (12742 + h0)) / Math.sqrt(1284200), 0) * (np - 1);
    if (p < 1) p = Math.max(0, p - p0) / (1 - p0);
    // Finite paths use the smooth chart: horizon distance across the fitted heights.
    const [low, high] = this.lut.metadata.scattering.heightRange.map((h) => Math.sqrt(h * (12742 + h)));
    const finiteP = Math.max(0, Math.min(1, (horizon - low) / (high - low)));
    const dip = Math.atan(horizon / 6371);
    for (let light = 0; light < 2; light++) {
      if (light === 1 && frame.moonRatio <= 1e-12) continue;
      const direction = light === 0 ? frame.solarDirection : moon.map((v) => v / length);
      const sine = up.reduce((sum, v, i) => sum + v * direction[i], 0);
      const angle = ((Math.asin(Math.max(-1, Math.min(1, sine))) + dip) * 180) / Math.PI;
      const mapped = (Math.asinh(angle / 0.5) - Math.asinh(-180)) / (Math.asinh(200.1717139250062) - Math.asinh(-180));
      const state = [finiteP, Math.max(0, Math.min(1, mapped)), p, this.warp(mapped, 3) * (nb - 1)];
      if (state.every((v, i) => this.previous[light * 4 + i] === v)) continue;
      this.scatteringGpu.update(this.textures[0], this.textures[6], light, [state[0], state[1]], [state[2], state[3]]);
      this.previous.splice(light * 4, 4, ...state);
    }
  }

  bind(program: WebGLProgram): void {
    if (this.disposed) throw new Error('Directional transport has been disposed');
    const gl = this.gl,
      layout = this.lut.metadata.observer,
      palette = this.lut.metadata.palette;
    for (const [i, [unit, target, name]] of (
      [
        [2, gl.TEXTURE_3D, 'u_hermite'],
        [7, gl.TEXTURE_2D, 'u_hermite_axes'],
        [8, gl.TEXTURE_2D, 'u_tail'],
        [5, gl.TEXTURE_2D, 'u_transmission'],
        [9, gl.TEXTURE_2D_ARRAY, 'u_hermite_rgb'],
        [4, gl.TEXTURE_3D, 'u_hermite_boundary'],
        [1, gl.TEXTURE_3D, 'u_hermite_sky'],
      ] as const
    ).entries()) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(target, this.textures[i]);
      gl.uniform1i(uniformLocation(gl, program, name), unit);
    }
    gl.uniform4iv(uniformLocation(gl, program, 'u_hermite_counts'), layout.counts);
    const epsilon = this.lut.metadata.scattering.epsilon;
    gl.uniform4f(
      uniformLocation(gl, program, 'u_scattering_epsilon'),
      epsilon.minimum,
      epsilon.decades,
      epsilon.center,
      epsilon.width,
    );
    const dense = denseNuChart(this.lut.metadata.scattering.renderNuScale);
    gl.uniform3f(
      uniformLocation(gl, program, 'u_scattering_nu'),
      this.lut.metadata.scattering.renderNuScale,
      dense.low,
      dense.inverseSpan,
    );
    gl.uniform2fv(
      uniformLocation(gl, program, 'u_hermite_color_scale[0]'),
      Float32Array.from(palette.color.uvScale.flat()),
    );
    gl.uniform2fv(
      uniformLocation(gl, program, 'u_hermite_color_bias[0]'),
      Float32Array.from(palette.color.uvBias.flat()),
    );
    gl.uniform2fv(uniformLocation(gl, program, 'u_hermite_color_min'), palette.color.minRadius);
    gl.uniform1f(uniformLocation(gl, program, 'u_observer_radiance_scale'), layout.radianceScale);
    gl.uniform1i(uniformLocation(gl, program, 'u_refraction'), this.lut.refraction ? 1 : 0);
    gl.activeTexture(gl.TEXTURE6);
    // Without refraction any 2D array keeps the sampler valid.
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.lut.refraction ? this.textures[7] : this.textures[4]);
    gl.uniform1i(uniformLocation(gl, program, 'u_refraction_lut'), 6);
    const transmission = layout.transmission;
    gl.uniform2fv(uniformLocation(gl, program, 'u_transmission_domain'), transmission.domain);
    gl.uniform2fv(uniformLocation(gl, program, 'u_transmission_scale'), transmission.columnScale);
    gl.uniform1f(uniformLocation(gl, program, 'u_transmission_mie'), transmission.mieExtinction);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.scatteringGpu?.dispose();
    for (const texture of this.textures) this.gl.deleteTexture(texture);
    this.textures.length = 0;
  }
}

export const hermiteScatteringGLSL: string = `
precision highp sampler2DArray;
precision highp sampler3D;
uniform sampler3D u_hermite,u_hermite_boundary,u_hermite_sky;
uniform sampler2DArray u_hermite_rgb;
uniform sampler2D u_hermite_axes;
uniform vec2 u_hermite_color_scale[2],u_hermite_color_bias[2],u_hermite_color_min;
// Finite distance chart: eps = x * 10^(y * (.5 + .5 tanh((relative - z) / w))).
uniform vec4 u_scattering_epsilon;
// Dense-grid nu: (asinh(relative degrees / x) - y) * z.
uniform vec3 u_scattering_nu;
vec2 scatteringEndColumns;
float scatteringRelativeElevation;
uniform ivec4 u_hermite_counts;
uniform float u_observer_radiance_scale;
uniform float u_ray4d_observer_height;
uniform vec3 u_ray4d_observer_up;
float hermiteWarp(float value,int axis) {
  float index=clamp(value,0.0,1.0)*64.0; int lo=min(int(floor(index)),63);
  float mapped=mix(texelFetch(u_hermite_axes,ivec2(lo,0),0)[axis],texelFetch(u_hermite_axes,ivec2(lo+1,0),0)[axis],index-float(lo));
  // The embedded grid is uniform except for its shorter final interval.
  // Retain interpolation between the stored knots, including that last interval.
  float stepSize=texelFetch(u_hermite_axes,ivec2(1,1),0)[axis];
  int lower=min(int(floor(mapped/stepSize)),u_hermite_counts[axis]-2),upper=lower+1;
  float a=texelFetch(u_hermite_axes,ivec2(lower,1),0)[axis],b=texelFetch(u_hermite_axes,ivec2(upper,1),0)[axis];
  return (float(lower)+(mapped-a)/(b-a))/float(u_hermite_counts[axis]-1);
}
float scatteringAsinh(float x) {
  // Avoid overflowing x*x for very small Mie ground-column fractions.
  return x>10000.0?log(x)+0.6931471805599453:log(x+sqrt(x*x+1.0));
}
// Smooth chart (no warp): x = asinh relative-elevation fraction of the view, y = phi / pi,
// z = the dense-grid nu coordinate (finer asinh scale at the horizon).
vec3 scatteringAngles(float height,vec3 up,vec3 direction,vec3 sun) {
  float horizon=sqrt(height*(12742.0+height)),dip=atan(horizon,6371.0);
  float mv=clamp(dot(up,direction),-1.0,1.0),ms=clamp(dot(up,sun),-1.0,1.0);
  vec3 vd=direction-up*mv,sd=sun-up*ms;
  float vl=length(vd),sl=length(sd);
  float phi=vl*sl>1e-10 ? atan(abs(dot(up,cross(vd,sd))),dot(vd,sd)) : ${Math.PI / 2};
  float angle=degrees(atan(mv,vl)+dip)/.5;
  float mapped=(sign(angle)*log(abs(angle)+sqrt(angle*angle+1.0))-(${Math.asinh(-180)}))/${Math.asinh(200.1717139250062) - Math.asinh(-180)};
  float fine=angle*.5/u_scattering_nu.x;
  fine=(sign(fine)*log(abs(fine)+sqrt(fine*fine+1.0))-u_scattering_nu.y)*u_scattering_nu.z;
  return clamp(vec3(mapped,phi/${Math.PI},fine),0.0,1.0);
}
// Sun elevation in the same chart, at the observer (space entry: per pixel).
float scatteringSunFraction(float height,vec3 up,vec3 sun) {
  float horizon=sqrt(height*(12742.0+height));
  float mu=clamp(dot(up,sun),-1.0,1.0),horizontal=sqrt(max((1.0-mu)*(1.0+mu),0.0));
  float angle=degrees(atan(mu*6371.0+horizontal*horizon,horizontal*6371.0-mu*horizon))/.5;
  return (sign(angle)*log(abs(angle)+sqrt(angle*angle+1.0))-(${Math.asinh(-180)}))/${Math.asinh(200.1717139250062) - Math.asinh(-180)};
}
// Distance coordinate of the finite spline from u = column / endColumns.
float scatteringCoordinate(float u) {
  vec4 e=u_scattering_epsilon;
  float x=clamp((scatteringRelativeElevation-e.z)/e.w,-10.0,10.0);
  float eps=e.x*pow(10.0,e.y*(.5+.5*tanh(x)));
  return clamp(scatteringAsinh(u/eps)/scatteringAsinh(1.0/eps),0.0,1.0);
}
vec4 scatteringBasis(float s,out int first) {
  float x=clamp(s,0.0,1.0)*13.0;
  first=min(int(floor(x)),12);
  float t=x-float(first);
  bool reverse=first>10;
  int edge=reverse?12-first:first;
  if(reverse)t=1.0-t;
  vec4 a,b,c,d;
  if(edge==0){
    a=vec4(-1,1.75,-11.0/12.0,1.0/6.0);
    b=vec4(3,-4.5,1.5,0);c=vec4(-3,3,0,0);d=vec4(1,0,0,0);
  }else if(edge==1){
    a=vec4(-.25,7.0/12.0,-.5,1.0/6.0);
    b=vec4(.75,-1.25,.5,0);c=vec4(-.75,.25,.5,0);d=vec4(.25,7.0/12.0,1.0/6.0,0);
  }else{
    a=vec4(-1,3,-3,1)/6.0;b=vec4(.5,-1,.5,0);
    c=vec4(-.5,0,.5,0);d=vec4(1,4,1,0)/6.0;
  }
  vec4 weights=((a*t+b)*t+c)*t+d;
  return reverse?weights.wzyx:weights;
}
// The texel-center range of each XY tile is isolated from adjacent distance controls.
vec4 scatteringBoundary(vec2 pixel,int control,float beta) {
  vec2 tile=vec2(control%2,control/2)*vec2(64,16);
  return texture(u_hermite_boundary,vec3((pixel+tile)/vec2(128,256),(beta+.5)/64.0));
}
vec3 scatteringPalette(int species,float r,float theta) {
  r=max(r,0.0);
  vec2 colorUV=vec2(theta,log2(max(r,u_hermite_color_min[species])))
    *u_hermite_color_scale[species]+u_hermite_color_bias[species];
  return r*texture(u_hermite_rgb,vec3(colorUV,float(species))).rgb;
}
// Finite paths. beta < 0: one hardware-trilinear fetch of the per-frame dense grid
// [nu, s, phi] holding (L_R, theta_R, L_M, theta_M). beta >= 0 (space entry):
// spline controls on nu64 x phi16, cubic in the distance coordinate.
vec3 finiteColor(vec3 angles,int light,int species,float u,float beta) {
  float s=scatteringCoordinate(u),L,theta;
  if(beta<0.0) {
    vec3 size=vec3(textureSize(u_hermite,0));
    float phis=(size.z-2.0)*.5;
    vec3 texel=vec3(angles.z*(size.x-1.0),s*(size.y-1.0),float(light)*phis+angles.y*(phis-1.0));
    vec4 v=texture(u_hermite,(texel+.5)/size);
    L=species==0?v.x:v.z;
    theta=species==0?v.y:v.w;
  } else {
    int first;
    vec4 weights=scatteringBasis(s,first);
    vec2 pixel=angles.xy*vec2(63,15)+.5;
    int c=species*16+first;
    vec4 a=scatteringBoundary(pixel,c,beta);
    vec4 b=scatteringBoundary(pixel,c+2,beta);
    L=dot(weights,vec4(a.x,a.z,b.x,b.z));
    theta=dot(weights,vec4(a.y,a.w,b.y,b.w));
  }
  float r=u*scatteringEndColumns[species]*exp(clamp(L,-40.0,species==0?3.0:2.5));
  return scatteringPalette(species,r,theta);
}
// Space-entry beta or the exact layer of a pre-expanded observer/light slice.
vec4 skyAnchor(vec2 grid,int light,float beta) {
  float slice=beta<0.0?64.0+float(light):beta;
  return texture(u_hermite_sky,vec3((grid*vec2(127,31)+.5)/vec2(128,32),(slice+.5)/66.0));
}
vec3 terrainMieOcclusion(vec3 direction,float distance);
vec3 terrainMieEndValue;
// Removed share of the endpoint Mie, per channel, and its finite path length.
// m - occluded clamped between 0 and m equals m * (1 - clamp(occluded / m, 0, 1)),
// so the fraction reproduces the full-resolution result and is independent of
// the endpoint value, which makes it suitable for depth-aware upsampling.
vec3 terrainMieFraction;
float terrainMieLength;
#ifdef NIGHTLAYER_COMPOSITE_PASS
uniform sampler2D u_shadowed_mie;
// Logical sensor pixel of the composite (also valid while metering).
ivec2 terrainMiePixel;
// Joint bilateral upsampling of the reduced-resolution fraction. Sample i was
// traced from logical pixel i*step; candidates are the 3x3 neighbours of p/step.
// Weights combine screen distance with relative path-length agreement, so a
// fraction never crosses a terrain silhouette. Without any agreeing sample the
// nearest path length wins; without any occluded sample nothing is removed.
vec3 terrainMieUpsampledFraction(float length){
  int step=${MIE_SHADOW_STEP};
  if(step==1)return texelFetch(u_shadowed_mie,terrainMiePixel,0).xyz;
  ivec2 size=textureSize(u_shadowed_mie,0),base=terrainMiePixel/step;
  vec3 sum=vec3(0),nearest=vec3(0);
  float total=0.0,best=1e30;
  for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
    ivec2 cell=clamp(base+ivec2(x,y),ivec2(0),size-1);
    vec4 s=texelFetch(u_shadowed_mie,cell,0);
    if(s.w<0.0)continue;
    vec2 offset=vec2(terrainMiePixel-cell*step)/float(step);
    float relative=abs(s.w-length)/max(length,1e-3);
    float weight=exp(-.5*dot(offset,offset)/.5625-(relative/.03)*(relative/.03));
    sum+=weight*s.xyz;total+=weight;
    if(relative<best){best=relative;nearest=s.xyz;}
  }
  return total>1e-4?sum/total:nearest;
}
#endif
// occlude: subtract terrain-shadowed cumulative Mie over [0, occludeLength].
// Its endpoint value P(occludeLength) is exactly this ray's Mie value m: the
// same uv, layer and coordinate u.y; it is passed on instead of re-evaluated.
 vec3 hermiteScattering(float height,vec3 up,vec3 direction,vec3 sun,vec2 u,bool sky,int light,bool boundary,bool occlude,float occludeLength) {
  vec3 angles=scatteringAngles(height,up,direction,sun);
  // Space entry: the solar elevation varies per pixel; finite slices are uniform in
  // the smooth chart, sky anchors stay on the warped nodes.
  float finiteBeta=-1.0,skyBeta=-1.0;
  if(boundary) {
    float mapped=scatteringSunFraction(height,up,sun);
    finiteBeta=clamp(mapped,0.0,1.0)*63.0;
    skyBeta=hermiteWarp(mapped,3)*float(u_hermite_counts.w-1);
  }
  vec3 r,m;
  if(sky) {
    vec2 grid=vec2(hermiteWarp(angles.x,1),hermiteWarp(angles.y,2));
    vec4 anchor=skyAnchor(grid,light,skyBeta);
    r=scatteringPalette(0,anchor.x,anchor.y);
    m=scatteringPalette(1,anchor.z,anchor.w);
  } else {
    r=finiteColor(angles,light,0,u.x,finiteBeta);
    m=finiteColor(angles,light,1,u.y,finiteBeta);
  }
   if(occlude){
   #ifdef NIGHTLAYER_COMPOSITE_PASS
     m*=1.0-terrainMieUpsampledFraction(occludeLength);
   #else
     terrainMieEndValue=m;
     vec3 occludedMie=terrainMieOcclusion(direction,occludeLength);
     // Fitted RGB cumulative curves are not guaranteed monotone channel by
     // channel; the fraction bound keeps the signed RGB between 0 and m.
     terrainMieFraction=mix(clamp(occludedMie/m,0.0,1.0),vec3(0),equal(m,vec3(0)));
     terrainMieLength=occludeLength;
     m*=1.0-terrainMieFraction;
   #endif
   }
  float nu=clamp(dot(direction,sun),-1.0,1.0),g=u_fog_mie_g;
  float pr=3.0/(16.0*${Math.PI})*(1.0+nu*nu);
  float pm=3.0/(8.0*${Math.PI})*(1.0-g*g)/(2.0+g*g)*(1.0+nu*nu)/pow(1.0+g*g-2.0*g*nu,1.5);
  return max((pr*r+pm*m)*u_observer_radiance_scale,vec3(0));
}
uniform sampler2D u_transmission;
uniform vec2 u_transmission_domain,u_transmission_scale;
uniform float u_transmission_mie;
vec3 hermiteTransmission(vec3 columns) {
  vec2 q=clamp(log(1.0+columns.xz/u_transmission_scale)/log(1.0+u_transmission_domain/u_transmission_scale),0.0,1.0);
  vec2 size=vec2(textureSize(u_transmission,0));vec2 uv=(q.yx*(size-1.0)+.5)/size;
  return clamp(exp(-columns.y*u_transmission_mie)*texture(u_transmission,uv).rgb,vec3(0),vec3(1));
}
`;

export const hermiteOpticalGLSL: string = `
// Hardware-filtered, path-normalized endpoint tails.
uniform sampler2D u_tail;
struct TailValue { vec3 base; float height; float span; };
TailValue tailSample(float h,float mu) {
  // Physical endpoints are above ground; clamp tiny negative roundoff at sea level.
  h=max(h,0.0);
  ivec2 size=textureSize(u_tail,0);
  float band=float(size.y/4),row;
  if(h<10.0) row=h*(band/10.0);
  else if(h<25.0) row=band+(h-10.0)*(band/15.0);
  else if(h<40.0) row=2.0*band+(h-25.0)*(band/15.0);
  else row=3.0*band+(h-40.0)*((band-1.0)/60.0);
  float m=clamp(mu,0.0,1.0);
  float column=(1.2*m/(m+.2))*float(size.x-1);
  vec3 value=texture(u_tail,(vec2(column,row)+.5)/vec2(size)).rgb;
  float t=(6371.0+h)*m,delta=max(100.0-h,0.0)*(12842.0+h);
  float distance=delta/max(sqrt(t*t+delta)+t,1e-30);
  return TailValue(value,h,distance);
}
TailValue lineTail(float h,float mu,float lineOffset) {
  TailValue value=tailSample(h,mu);
  // Ozone is absent below 10 km; reuse the same spherical-line tail there.
  if(h<10.0) {
    float om=sqrt(max(lineOffset+127520.0,0.0))/6381.0;
    TailValue ozone=tailSample(10.0,om);
    value.base.z=ozone.base.z;
  }
  return value;
}
vec2 tailRM(TailValue a) {
  return exp(a.base.xy-a.height/vec2(8.0,1.2))*a.span;
}
vec3 tailDifference(TailValue a,TailValue b) {
  return vec3(tailRM(a)-tailRM(b),a.base.z-b.base.z);
}
vec3 tailColumns(float h,float mu,float distance) {
  float radius=6371.0+h,t=radius*mu;
  vec3 local=vec3(0);
  if(distance<1.0) {
    float midpoint=.5*max(distance,0.0),delta=midpoint*(2.0*t+midpoint);
    float midH=max(h+delta/(sqrt(radius*radius+delta)+radius),0.0);
    local=vec3(exp(-midH/vec2(8.0,1.2)),max(0.0,1.0-abs(midH-25.0)/15.0))*max(distance,0.0);
    if(distance<=.25) return local;
  }
  float delta=distance*(2.0*t+distance);
  float endH=h+delta/(sqrt(radius*radius+delta)+radius);
  float endMu=(t+distance)/(6371.0+endH);
  float lineOffset=t*t-h*(12742.0+h);
  TailValue first=lineTail(h,abs(mu),lineOffset),last=lineTail(endH,abs(endMu),lineOffset);
  vec3 result;
  if(t<0.0&&t+distance>0.0) {
    float tangentH=h-t*t/(sqrt(max(radius*radius-t*t,0.0))+radius);
    TailValue tangent=lineTail(tangentH,0.0,lineOffset);
    result=tailDifference(tangent,first)+tailDifference(tangent,last);
  } else result=t<0.0?tailDifference(last,first):tailDifference(first,last);
  return distance<1.0?mix(local,result,smoothstep(.25,1.0,distance)):result;
}
${refractionGLSL}
#ifdef NIGHTLAYER_COMPOSITE_PASS
uniform bool u_terrain_mie_shadows;
#else
${terrainMieShadowGLSL}
#endif
vec3 terrainMieUV;
float terrainMieHeight, terrainMieMu, terrainMieScale;
vec3 terrainMiePrefixEnd(float distance) {
  return terrainMieEndValue;
}
vec3 terrainMiePrefix(float distance) {
  if(distance<=0.0) return vec3(0);
  float coordinate=max(tailColumns(terrainMieHeight,terrainMieMu,distance).y,0.0)/terrainMieScale;
  return finiteColor(terrainMieUV,0,1,clamp(coordinate,0.0,1.0),-1.0);
}
 void atmosphereRayFrom(vec3 origin,vec3 direction,float distance,bool sky,float height,vec3 up,
                        out vec3 transmission,out vec3 scattering) {
   atmosphereBackgroundOrigin=origin;atmosphereBackgroundDirection=direction;
  transmission=vec3(1);scattering=vec3(0);
  if(u_atmosphere_strength<=0.0||distance<=0.0) return;
  bool boundary=height>=100.0;
  if(height>100.0) {
    float along=dot(origin,direction),impact=length(cross(origin,direction));
    if(along>=0.0||impact>=6471.0) return;
    float root=sqrt(max((6471.0-impact)*(6471.0+impact),0.0));
    float entry=(height-100.0)*(height+12842.0)/max(-along+root,1e-20);
    if(distance<=entry) return;
    origin+=entry*direction;
    up=normalize(origin);
    distance-=entry;
    height=100.0;
  }
  float radius=6371.0+height;
  float mu=clamp(dot(up,direction),-1.0,1.0),t=radius*mu;
  float delta=(100.0-height)*(12842.0+height),root=sqrt(max(t*t+delta,0.0));
  float exitDistance=t>=0.0?delta/max(root+t,1e-20):root-t;
  float radial=height*(12742.0+height),groundDisc=t*t-radial;
  bool ground=mu<0.0&&groundDisc>=0.0;
  float cutoff=ground?radial/max(-t+sqrt(max(groundDisc,0.0)),1e-20):min(sqrt(radial),exitDistance);
  float length=sky?(ground?cutoff:exitDistance):min(distance,cutoff);
   vec3 columns=max(tailColumns(height,mu,length),vec3(0));
   vec3 opticalColumns=columns;
    if(u_refraction&&sky&&!ground)opticalColumns=refractionSkyColumns(height,mu);
   transmission=hermiteTransmission(opticalColumns);
  // A geographic sky pixel may still hit the physical sea-level sphere.
  // Occlude the background, but retain the atmosphere in front of that sphere.
  if(sky&&ground) transmission=vec3(0);
   bool skyExit=sky&&!ground;
   if(u_refraction&&skyExit)refractionBackground(height,up,direction);
  // u=1 always means the geometric horizon, NOT the first ground collision.
  // Beyond ground the coordinate alone continues at sea-level density. This
  // virtual column never contributes to transmission or physical scattering.
  vec2 endColumns=vec2(1),u=vec2(1);
  if(!skyExit) {
    if(cutoff<=0.0) return;
    float virtualDistance=ground?max(sqrt(radial)-cutoff,0.0):0.0;
    vec2 physicalEnd=tailColumns(height,mu,cutoff).xy;
    endColumns=max(physicalEnd+vec2(virtualDistance),vec2(1e-30));
     u=clamp(columns.xy/endColumns,0.0,1.0);
       scatteringEndColumns=endColumns;
        float horizontal=sqrt(max((1.0-mu)*(1.0+mu),0.0)),horizon=sqrt(radial);
        scatteringRelativeElevation=degrees(atan(mu*6371.0+horizontal*horizon,horizontal*6371.0-mu*horizon));
  }
  // Finite surface paths have a cumulative curve. The independent sky-exit
  // anchor is not a curve beyond the finite domain and must not be differenced.
  bool occlude=u_terrain_mie_shadows && !sky && !boundary && length>0.0;
  if(occlude) {
    terrainMieUV=scatteringAngles(height,up,direction,u_solar_direction);
    terrainMieHeight=height; terrainMieMu=mu;
    terrainMieScale=endColumns.y;
  }
  scattering=hermiteScattering(height,up,direction,u_solar_direction,u,skyExit,0,boundary,occlude,length);
  if(u_moon_ratio>1e-12)
    scattering+=hermiteScattering(height,up,direction,normalize(u_moon_position-origin),u,skyExit,1,boundary,false,0.0)*u_moon_ratio;
}
void atmosphereRay(vec3 origin,vec3 direction,float distance,bool sky,out vec3 transmission,out vec3 scattering) {
  atmosphereRayFrom(origin,direction,distance,sky,max(u_ray4d_observer_height,0.0),
                    normalize(u_ray4d_observer_up),transmission,scattering);
}
`;
