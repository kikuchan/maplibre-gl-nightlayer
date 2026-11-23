import type { Bucket, CustomRenderMethodInput, Map as MaplibreMap, Tile } from 'maplibre-gl';
import { compileProgram, uniformLocation } from './gl.js';
import { shadowFragment, shadowVertexGLSL, type TerrainShadow } from './shadow.js';
import {
  bindRelativePosition,
  bindTerrainElevation,
  bindTileProjection,
  inverseLookupGLSL,
  projectionVertexPrelude,
  relativePositionGLSL,
  terrainElevationGLSL,
  type ProjectionCamera,
} from './surface.js';

type Painter = MaplibreMap['painter'];
type Frame = Parameters<Painter['renderLayer']>[4];
type Configuration = NonNullable<Parameters<Frame['useProgram']>[1]>;
type Terrain = NonNullable<MaplibreMap['terrain']>;
type VertexBuffer = ReturnType<Terrain['getTerrainMesh']>['vertexBuffer'];
type ExtrusionBucket = Bucket & {
  layoutVertexBuffer?: VertexBuffer;
  centroidVertexBuffer?: VertexBuffer;
  indexBuffer?: ReturnType<Terrain['getTerrainMesh']>['indexBuffer'];
  segments: ReturnType<Terrain['getTerrainMesh']>['segments'];
  programConfigurations: { get(id: string): Configuration | undefined };
};
type ExtrusionPaint = {
  get(name: 'fill-extrusion-opacity'): number;
  get(name: 'fill-extrusion-height' | 'fill-extrusion-base'): { constantOr(fallback: number): number };
  get(name: 'fill-extrusion-translate'): [number, number];
  get(name: 'fill-extrusion-translate-anchor'): 'map' | 'viewport';
};
type Extrusion = {
  tile: Tile;
  segments: ExtrusionBucket['segments'];
  indexBuffer: WebGLBuffer;
  vertexBuffers: VertexBuffer[];
  configuration: Configuration;
  paint: ExtrusionPaint;
  translate: [number, number];
  maxElevation: number;
  signature: readonly unknown[];
};
type ExtrusionProgram = {
  program: WebGLProgram;
  vao: WebGLVertexArrayObject;
  uniforms: ReturnType<Configuration['getUniforms']>;
  attributes: Map<string, number>;
};
type Pass = { shadow: TerrainShadow } | { camera: ProjectionCamera };

const TILE_EXTENT = 8192; // MapLibre's native tile vertex grid.
const extrusionUniforms = ['u_height', 'u_height_t', 'u_base', 'u_base_t'];
const extrusionAttributes = ['a_pos', 'a_normal_ed', 'a_centroid', 'a_height', 'a_base'];

const surfaceFragment = `#version 300 es
precision highp float;
precision highp int;
in vec3 v_path, v_up, v_normal;
layout(location=0) out vec4 hit;
${inverseLookupGLSL}
void main() {
  vec3 up = normalize(v_up);
  vec3 east = normalize(vec3(up.z, 0.0, -up.x));
  vec3 north = cross(up, east);
  // The packed wall perpendicular points opposite the front-face normal in
  // the east/south tile basis. Roof Z already points up.
  vec3 normal = normalize(-east * v_normal.x + north * v_normal.y + up * v_normal.z);
  hit = packGroundHit(v_path, normal, false);
}`;

function paintInput(name: 'height' | 'base', uniforms: Set<string>): [string, string] {
  if (uniforms.has(`u_${name}`)) return [`uniform float u_${name};`, `u_${name}`];
  if (uniforms.has(`u_${name}_t`))
    return [`in vec2 a_${name}; uniform float u_${name}_t;`, `mix(a_${name}.x, a_${name}.y, u_${name}_t)`];
  return [`in float a_${name};`, `a_${name}`];
}

/** Reuse native buckets and paint bindings; only vertex positioning is reproduced here. */
export class ExtrusionRenderer {
  #programs = new Map<string, ExtrusionProgram>();
  #bucketIds = new WeakMap<ExtrusionBucket, number>();
  #nextBucketId = 0;

  collect(map: MaplibreMap, layerIds: readonly string[]): Extrusion[] {
    const zoom = map.getZoom();
    const terrain = map.terrain;
    const extrusions: Extrusion[] = [];
    for (const id of layerIds) {
      const layer = map.style.getLayer(id);
      if (layer?.type !== 'fill-extrusion' || layer.isHidden(zoom)) continue;
      const paint = layer.paint as ExtrusionPaint;
      // The atmosphere passes treat the geometry as solid, regardless of its paint.
      if (paint.get('fill-extrusion-opacity') === 0) continue;
      const manager = map.style.tileManagers[layer.source];
      if (!manager) continue;
      const height = paint.get('fill-extrusion-height').constantOr(0);
      const base = paint.get('fill-extrusion-base').constantOr(0);
      const [x, y] = paint.get('fill-extrusion-translate');
      const angle =
        paint.get('fill-extrusion-translate-anchor') === 'viewport' ? (map.getBearing() * Math.PI) / 180 : 0;
      const cos = Math.cos(angle),
        sin = Math.sin(angle);
      for (const tileID of manager.getVisibleCoordinates()) {
        const tile = manager.getTile(tileID);
        const bucket = tile?.getBucket(layer) as ExtrusionBucket | undefined;
        if (!bucket?.layoutVertexBuffer?.buffer || !bucket.indexBuffer?.buffer) continue;
        if (terrain && !bucket.centroidVertexBuffer?.buffer) continue;
        const configuration = bucket.programConfigurations.get(id);
        if (!configuration) continue;
        let bucketId = this.#bucketIds.get(bucket);
        if (bucketId === undefined) {
          bucketId = ++this.#nextBucketId;
          this.#bucketIds.set(bucket, bucketId);
        }
        const scale = TILE_EXTENT / (tile.tileSize * 2 ** (zoom - tileID.overscaledZ));
        const translate: [number, number] = [(x * cos - y * sin) * scale, (x * sin + y * cos) * scale];
        const demTile = terrain?.tileManager.getSourceTile(tileID, true);
        const dem = demTile?.dem;
        const maxElevation =
          Math.max(
            0,
            height,
            base,
            configuration.getMaxValue('fill-extrusion-height'),
            configuration.getMaxValue('fill-extrusion-base'),
          ) + Math.max(0, (dem?.max ?? 0) * (terrain?.exaggeration ?? 0));
        extrusions.push({
          tile,
          segments: bucket.segments,
          indexBuffer: bucket.indexBuffer.buffer,
          vertexBuffers: [
            bucket.layoutVertexBuffer,
            ...(terrain && bucket.centroidVertexBuffer ? [bucket.centroidVertexBuffer] : []),
            ...configuration.getPaintVertexBuffers(),
          ],
          configuration,
          paint,
          translate,
          maxElevation,
          signature: [
            id,
            tileID.key,
            bucketId,
            tile.featureStateRevision,
            height,
            base,
            translate,
            demTile?.uid,
            maxElevation,
          ],
        });
      }
    }
    return extrusions;
  }

  #getProgram(
    context: Painter['context'],
    options: CustomRenderMethodInput,
    configuration: Configuration,
    terrain: boolean,
    shadow: boolean,
  ): ExtrusionProgram {
    const gl = context.gl;
    const uniformNames = new Set(configuration.getBinderUniforms().filter((name) => extrusionUniforms.includes(name)));
    const bindings = extrusionUniforms.filter((name) => uniformNames.has(name)).join('/');
    const key = `${shadow ? 'shadow' : `surface/${options.shaderData.variantName}`}/${terrain}/${bindings}`;
    const cached = this.#programs.get(key);
    if (cached) return cached;
    const [heightInput, height] = paintInput('height', uniformNames);
    const [baseInput, base] = paintInput('base', uniformNames);
    const program = compileProgram(
      gl,
      shadow ? shadowFragment : surfaceFragment,
      (shadow ? '#version 300 es\nprecision highp float;\nprecision highp int;\n' : projectionVertexPrelude(options)) +
        `
      ${terrain ? '#define HAS_DEM' : ''}
      in vec2 a_pos;
      in ivec4 a_normal_ed;
      in vec2 a_centroid;
      ${heightInput}
      ${baseInput}
      uniform vec2 u_translate, u_tile_scale;
      ${terrainElevationGLSL}
      ${relativePositionGLSL}
      ${shadow ? `uniform mat4 u_shadow_matrix;\n${shadowVertexGLSL}` : 'out vec3 v_path, v_up, v_normal;'}
      void main() {
        float base = ${base};
        float height = ${height};
        float ground = terrainElevation(a_centroid);
        // Match the native extrusion: centroid DEM for both ends and a basement
        // for ground-level walls so they do not hang above slopes.
        float basement = ${terrain ? 'base <= 0.0 ? 10.0 : 0.0' : '0.0'};
        float elevation = ground + ((a_normal_ed.x & 1) != 0 ? max(0.0,height) : max(0.0,base)-basement);
        vec2 position = a_pos + u_translate;
        vec3 delta = localSphereDelta(position, u_tile_scale);
        vec3 path = localEndpoint(delta, elevation);
        ${
          shadow
            ? `vec3 light = (u_shadow_matrix * vec4(path, 1.0)).xyz;
        gl_Position = shadowPosition(light, elevation * .001);`
            : `#ifdef GLOBE
        gl_Position = interpolateProjectionFor3D(position, projectToSphere(position, a_pos), elevation);
        #else
        gl_Position = projectTileFor3D(position, elevation);
        #endif
        v_path = path;
        v_up = u_local_to_world * (u_local_up + delta);
        v_normal = vec3(ivec3(a_normal_ed.x & ~1, a_normal_ed.yz));`
        }
      }`,
    );
    const locations: Record<string, WebGLUniformLocation> = {};
    for (const name of uniformNames) {
      const location = uniformLocation(gl, program, name);
      if (location) locations[name] = location;
    }
    const uniforms = configuration.getUniforms(context, locations);
    const attributes = new Map(extrusionAttributes.map((name) => [name, gl.getAttribLocation(program, name)]));
    const result = { program, vao: gl.createVertexArray()!, uniforms, attributes };
    this.#programs.set(key, result);
    return result;
  }

  render(map: MaplibreMap, options: CustomRenderMethodInput, extrusions: readonly Extrusion[], pass: Pass): void {
    if (extrusions.length === 0) return;
    const context = map.painter.context;
    const gl = context.gl;
    const terrain = map.terrain;
    const zoom = map.getZoom();
    const observer = 'shadow' in pass ? pass.shadow.origin : pass.camera.origin;
    // Prepare native DEM textures before switching to our own raw draw state.
    const demTiles = extrusions.map(({ tile }) => terrain?.getTerrainData(tile.tileID));
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.depthFunc(gl.LEQUAL);
    gl.depthRange(0, 1);
    gl.disable(gl.BLEND);
    if ('shadow' in pass) {
      gl.disable(gl.STENCIL_TEST);
      gl.disable(gl.CULL_FACE);
    } else {
      // Replace sky/sea coverage only where the building passes the terrain depth test.
      gl.enable(gl.STENCIL_TEST);
      gl.stencilMask(0xff);
      gl.stencilFunc(gl.ALWAYS, 1, 0xff);
      gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.BACK);
      gl.frontFace(gl.CCW);
    }
    for (const [
      i,
      { tile, segments, indexBuffer, vertexBuffers, configuration, paint, translate },
    ] of extrusions.entries()) {
      const dem = demTiles[i];
      const { program, vao, uniforms, attributes } = this.#getProgram(
        context,
        options,
        configuration,
        !!dem,
        'shadow' in pass,
      );
      const projection = options.getProjectionData({
        tileID: tile.tileID,
        applyTerrainMatrix: false,
        applyGlobeMatrix: true,
      });
      const { tileMercatorCoords } = projection;
      gl.useProgram(program);
      gl.bindVertexArray(vao);
      if ('shadow' in pass) pass.shadow.bindProjection(gl, program);
      else bindTileProjection(gl, program, projection);
      bindRelativePosition(gl, program, tileMercatorCoords, observer);
      if (dem) bindTerrainElevation(gl, program, dem);
      gl.uniform2f(uniformLocation(gl, program, 'u_tile_scale'), tileMercatorCoords[2], tileMercatorCoords[3]);
      gl.uniform2fv(uniformLocation(gl, program, 'u_translate'), translate);
      configuration.setUniforms(context, uniforms, paint, { zoom });
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
      for (const segment of segments.get()) {
        for (const buffer of vertexBuffers) {
          for (const attribute of buffer.attributes) {
            const location = attributes.get(attribute.name) ?? -1;
            if (location < 0) continue;
            const offset = attribute.offset + segment.vertexOffset * buffer.itemSize;
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer.buffer);
            gl.enableVertexAttribArray(location);
            if (attribute.name === 'a_normal_ed')
              gl.vertexAttribIPointer(location, attribute.components, gl.SHORT, buffer.itemSize, offset);
            else
              gl.vertexAttribPointer(
                location,
                attribute.components,
                attribute.type === 'Int16' ? gl.SHORT : gl.FLOAT,
                false,
                buffer.itemSize,
                offset,
              );
          }
        }
        gl.drawElements(gl.TRIANGLES, segment.primitiveLength * 3, gl.UNSIGNED_SHORT, segment.primitiveOffset * 6);
      }
    }
    context.setDirty();
  }

  dispose(gl: WebGL2RenderingContext): void {
    for (const { program, vao } of this.#programs.values()) {
      gl.deleteVertexArray(vao);
      gl.deleteProgram(program);
    }
    this.#programs.clear();
  }
}
