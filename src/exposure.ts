const SIZE = 16;
const SAMPLE_INTERVAL = 250;

/** Low-resolution, unexposed log-luminance metering with asynchronous readback. */
export class AutoExposure {
  #framebuffer?: WebGLFramebuffer;
  #color?: WebGLRenderbuffer;
  #buffer?: WebGLBuffer;
  #fence?: WebGLSync;
  #pixels = new Uint8Array(SIZE * SIZE * 4);
  #dirty = true;
  #sampleTime = -Infinity;
  #updateTime?: number;
  #target = 0;
  value = 0;

  invalidate(): void {
    this.#dirty = true;
  }

  get needsFrame(): boolean {
    return this.#dirty || !!this.#fence || Math.abs(this.#target - this.value) > 0.02;
  }

  update(gl: WebGL2RenderingContext, now: number): void {
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
        const samples: { log: number; weight: number }[] = [];
        for (let y = 0; y < SIZE; y++)
          for (let x = 0; x < SIZE; x++) {
            const i = (y * SIZE + x) * 4;
            if (this.#pixels[i + 3] === 0) continue;
            const radius = (((x + 0.5) / SIZE) * 2 - 1) ** 2 + (((y + 0.5) / SIZE) * 2 - 1) ** 2;
            samples.push({ log: (this.#pixels[i] / 255) * 24 - 20, weight: Math.exp(-2 * radius) });
          }
        if (samples.length) {
          samples.sort((a, b) => a.log - b.log);
          // Winsorize the brightest 10% so a small solar disk cannot dominate metering.
          const ceiling = samples[Math.floor((samples.length - 1) * 0.9)].log;
          const total = samples.reduce((sum, sample) => sum + sample.weight, 0);
          const mean = samples.reduce((sum, sample) => sum + Math.min(sample.log, ceiling) * sample.weight, 0) / total;
          this.#target = Math.max(-4, Math.min(8, Math.log2(0.18) - mean));
        }
      }
    }
    const elapsed = Math.min(0.1, Math.max(0, (now - (this.#updateTime ?? now)) / 1000));
    this.#updateTime = now;
    // Bright scenes reduce exposure faster than dark scenes increase it.
    const seconds = this.#target < this.value ? 0.4 : 1.2;
    this.value += (this.#target - this.value) * (1 - Math.exp(-elapsed / seconds));
    if (Math.abs(this.#target - this.value) <= 0.02) this.value = this.#target;
  }

  measure(gl: WebGL2RenderingContext, now: number, draw: () => void): void {
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
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA8, SIZE, SIZE);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, this.#color);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
          throw new Error('Incomplete exposure metering framebuffer');
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.#buffer);
        gl.bufferData(gl.PIXEL_PACK_BUFFER, this.#pixels.byteLength, gl.STREAM_READ);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.#framebuffer);
      gl.viewport(0, 0, SIZE, SIZE);
      draw();
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.#buffer!);
      for (const { parameter } of pack) gl.pixelStorei(parameter, parameter === gl.PACK_ALIGNMENT ? 1 : 0);
      gl.readPixels(0, 0, SIZE, SIZE, gl.RGBA, gl.UNSIGNED_BYTE, 0);
      this.#fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0) ?? undefined;
      if (!this.#fence) throw new Error('Failed to schedule exposure metering');
      gl.flush();
      this.#sampleTime = now;
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
