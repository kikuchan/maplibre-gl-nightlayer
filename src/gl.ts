export function compileProgram(gl: WebGL2RenderingContext, fragment: string, vertex: string): WebGLProgram {
  const program = gl.createProgram();
  if (!program) throw new Error('Failed to create NightLayer program');
  const shaders: WebGLShader[] = [];
  try {
    for (const [type, text] of [
      [gl.VERTEX_SHADER, vertex],
      [gl.FRAGMENT_SHADER, fragment],
    ] as const) {
      const shader = gl.createShader(type);
      if (!shader) throw new Error('Failed to create NightLayer shader');
      shaders.push(shader);
      gl.shaderSource(shader, text);
      gl.compileShader(shader);
      gl.attachShader(program, shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
      throw new Error(
        shaders.map((shader) => gl.getShaderInfoLog(shader)).join('\n') ||
          gl.getProgramInfoLog(program) ||
          'Failed to link NightLayer program',
      );
    return program;
  } catch (error) {
    gl.deleteProgram(program);
    throw error;
  } finally {
    for (const shader of shaders) gl.deleteShader(shader);
  }
}

// Uniform locations stay valid until a program is relinked. NightLayer links
// each program once, so locations are cached per program object; programs
// recreated after a context loss are new keys and the old entries are dropped.
const locations = new WeakMap<WebGLProgram, Map<string, WebGLUniformLocation | null>>();

/** Cached `getUniformLocation`, avoiding per-draw lookups in tile loops. */
export function uniformLocation(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  name: string,
): WebGLUniformLocation | null {
  let cache = locations.get(program);
  if (!cache) locations.set(program, (cache = new Map()));
  let location = cache.get(name);
  if (location === undefined) cache.set(name, (location = gl.getUniformLocation(program, name)));
  return location;
}

/** Inverse-LUT payloads and scene reception share one stencil. */
export class SurfaceTargets {
  /** Packed direction, range and ground attributes; see inverseLookupGLSL. */
  readonly rays: WebGLTexture;
  readonly framebuffer: WebGLFramebuffer;
  readonly #receiver: WebGLFramebuffer;
  #receiverTexture?: WebGLTexture;
  readonly #textures: WebGLTexture[] = [];
  readonly #framebuffers: WebGLFramebuffer[] = [];
  readonly #renderbuffers: WebGLRenderbuffer[] = [];

  constructor(
    private readonly gl: WebGL2RenderingContext,
    readonly width: number,
    readonly height: number,
  ) {
    const maximum = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE), gl.getParameter(gl.MAX_RENDERBUFFER_SIZE));
    if (Math.max(width, height) > maximum) throw new Error('Inverse LUT exceeds GPU size limits');
    try {
      this.rays = this.#texture();
      this.framebuffer = this.#framebuffer();
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.rays, 0);
      const depth = this.#renderbuffer(gl.DEPTH24_STENCIL8);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, depth);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      this.#check();
      this.#receiver = this.#framebuffer();
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, depth);
    } catch (error) {
      this.dispose();
      throw error;
    }
  }
  #texture(): WebGLTexture {
    const gl = this.gl,
      texture = gl.createTexture();
    if (!texture) throw new Error('Failed to allocate inverse-LUT texture');
    this.#textures.push(texture);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, this.width, this.height, 0, gl.RGBA, gl.FLOAT, null);
    return texture;
  }
  #framebuffer(): WebGLFramebuffer {
    const framebuffer = this.gl.createFramebuffer();
    if (!framebuffer) throw new Error('Failed to allocate inverse-LUT framebuffer');
    this.#framebuffers.push(framebuffer);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, framebuffer);
    return framebuffer;
  }
  #renderbuffer(format: number): WebGLRenderbuffer {
    const gl = this.gl,
      renderbuffer = gl.createRenderbuffer();
    if (!renderbuffer) throw new Error('Failed to allocate inverse-LUT renderbuffer');
    this.#renderbuffers.push(renderbuffer);
    gl.bindRenderbuffer(gl.RENDERBUFFER, renderbuffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, format, this.width, this.height);
    return renderbuffer;
  }
  #check(): void {
    const status = this.gl.checkFramebufferStatus(this.gl.FRAMEBUFFER);
    if (status !== this.gl.FRAMEBUFFER_COMPLETE)
      throw new Error(`Incomplete NightLayer surface framebuffer (status 0x${status.toString(16)})`);
  }
  beginLut(): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, this.width, this.height);
    gl.disable(gl.SCISSOR_TEST);
    gl.colorMask(true, true, true, true);
    gl.depthMask(true);
    gl.stencilMask(0xff);
    gl.clearColor(0, 0, 0, 0);
    gl.clearDepth(1);
    gl.clearStencil(0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
  }
  /** Explicit picking/export only; preserve shared pixel-pack state. */
  readRays(x = 0, y = 0, width: number = this.width, height: number = this.height): Float32Array {
    const gl = this.gl;
    const framebuffer = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING),
      buffer = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING);
    const pack = [gl.PACK_ALIGNMENT, gl.PACK_ROW_LENGTH, gl.PACK_SKIP_PIXELS, gl.PACK_SKIP_ROWS].map(
      (parameter) => [parameter, gl.getParameter(parameter)] as const,
    );
    try {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, this.framebuffer);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      for (const [parameter] of pack) gl.pixelStorei(parameter, parameter === gl.PACK_ALIGNMENT ? 1 : 0);
      const pixels = new Float32Array(width * height * 4);
      gl.readPixels(x, y, width, height, gl.RGBA, gl.FLOAT, pixels);
      if (gl.isContextLost()) throw new Error('WebGL context lost during depth readback');
      return pixels;
    } finally {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, framebuffer);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buffer);
      for (const [parameter, value] of pack) gl.pixelStorei(parameter, value);
    }
  }
  beginReceiver(texture: WebGLTexture): void {
    const gl = this.gl;
    gl.disable(gl.SAMPLE_ALPHA_TO_COVERAGE);
    gl.disable(gl.SAMPLE_COVERAGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.#receiver);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    if (texture !== this.#receiverTexture) {
      // Write straight to the receiver's RAW texture while retaining the ground stencil.
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      this.#check();
      this.#receiverTexture = texture;
    }
    gl.colorMask(true, true, true, true);
    gl.clearColor(0, 0, 0, 0);
    // Keep the inverse LUT's SSOT depth/stencil.
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.STENCIL_TEST);
    gl.stencilMask(0);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
  }
  selectGround(ground: boolean): void {
    this.gl.stencilFunc(this.gl.EQUAL, ground ? 1 : 0, 0xff);
  }
  beginSolar(): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.#receiver);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    this.selectGround(false);
    // Disk coverage is already included in the radiance.
    gl.disable(gl.SAMPLE_ALPHA_TO_COVERAGE);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE);
  }
  endSolar(): void {
    this.gl.disable(this.gl.SAMPLE_ALPHA_TO_COVERAGE);
    this.gl.disable(this.gl.BLEND);
  }
  endReceiver(): void {
    const gl = this.gl;
    gl.disable(gl.STENCIL_TEST);
    gl.stencilMask(0xff);
  }
  dispose(): void {
    for (const texture of this.#textures) this.gl.deleteTexture(texture);
    for (const framebuffer of this.#framebuffers) this.gl.deleteFramebuffer(framebuffer);
    for (const renderbuffer of this.#renderbuffers) this.gl.deleteRenderbuffer(renderbuffer);
  }
}
