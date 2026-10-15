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
