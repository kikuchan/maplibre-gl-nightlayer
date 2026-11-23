import type { CustomRenderMethodInput, Map as MaplibreMap } from 'maplibre-gl';
import { compileProgram, uniformLocation } from './gl.js';

// Geographic projection and the observer-local frame.
export type WorldCoordinates = NonNullable<MaplibreMap['style']['projection']>['worldCoordinateHelper'];
export type ProjectionCamera = {
  origin: number[];
  planarPosition?: number[];
  worldToPhysical: number[];
  inverseProjection: number[];
  planarView: number[];
  sphereView: number[];
};

export type DepthSample = {
  /** Straight-line range; infinity for sky. */
  distanceMeters: number;
  /** Optical-axis depth; infinity for sky or endpoints behind the camera. */
  depthMeters: number;
  /** Coverage of the nearest surface, including opaque buildings. */
  groundCoverage: number;
};

/** Physical forward axis. Mixed projections have no exact pinhole camera;
 * use the same participating geographic frames as the renderer. */
export function viewAxis(camera: ProjectionCamera): number[] {
  const axis = [0, 1, 2].map((i) => -camera.planarView[6 + i] - camera.sphereView[6 + i]);
  const length = Math.hypot(...axis);
  return axis.map((v) => v / length);
}

/** Decode the same octahedral direction and range as the GPU inverse lookup. */
export function opticalDepthMeters(ray: ArrayLike<number>, axis: ArrayLike<number>, offset = 0): number {
  const range = ray[offset + 2];
  if (range <= 0) return Infinity;
  let x = ray[offset],
    y = ray[offset + 1];
  const z = 1 - Math.abs(x) - Math.abs(y);
  if (z < 0) {
    const foldedX = (1 - Math.abs(y)) * (x < 0 ? -1 : 1);
    y = (1 - Math.abs(x)) * (y < 0 ? -1 : 1);
    x = foldedX;
  }
  const axial = x * axis[0] + y * axis[1] + z * axis[2];
  return axial > 0 ? (axial * range * 1000) / Math.hypot(x, y, z) : Infinity;
}

export function depthSample(ray: ArrayLike<number>, axis: ArrayLike<number>): DepthSample {
  const groundCoverage = ray[2] > 0 ? 1 : 0;
  if (groundCoverage <= 0) return { distanceMeters: Infinity, depthMeters: Infinity, groundCoverage };
  return {
    distanceMeters: ray[2] * 1000,
    depthMeters: opticalDepthMeters(ray, axis),
    groundCoverage,
  };
}

/** Built-in geographic projections share this grid; registered CRSs do not. */
export function supportsProjection(map: MaplibreMap): boolean {
  return map.style.projection?.worldCoordinateHelper.name === 'mercator';
}

export function transform3(matrix: ArrayLike<number>, vector: ArrayLike<number>): number[] {
  return [0, 1, 2].map((row) => [0, 1, 2].reduce((sum, col) => sum + matrix[col * 3 + row] * vector[col], 0));
}

export function inverse3(matrix: ArrayLike<number>): number[] {
  const cross = (a: number[], b: number[]) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const columns = [0, 1, 2].map((i) => Array.from({ length: 3 }, (_, j) => matrix[i * 3 + j]));
  const rows = [cross(columns[1], columns[2]), cross(columns[2], columns[0]), cross(columns[0], columns[1])];
  const determinant = columns[0].reduce((sum, n, i) => sum + n * rows[0][i], 0);
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-30)
    throw new Error('NightLayer: singular geographic camera transform');
  return Array.from({ length: 9 }, (_, i) => rows[i % 3][Math.floor(i / 3)] / determinant);
}

function geographicUp(lng: number, lat: number): number[] {
  const longitude = (lng * Math.PI) / 180,
    latitude = (lat * Math.PI) / 180;
  return [Math.sin(longitude) * Math.cos(latitude), Math.sin(latitude), Math.cos(longitude) * Math.cos(latitude)];
}

export function worldFrame(coordinates: WorldCoordinates, position: number[], kilometres: number): number[] {
  const location = coordinates.lngLatFromWorld(position[0], position[1]);
  const up = geographicUp(location.lng, location.lat);
  const longitude = (location.lng * Math.PI) / 180;
  const east = [Math.cos(longitude), 0, -Math.sin(longitude)];
  const north = [up[1] * east[2], Math.hypot(up[0], up[2]), -up[1] * east[0]];
  const horizontal = 2 * Math.PI * 6371 * Math.hypot(up[0], up[2]);
  return [...east.map((v) => v * horizontal), ...north.map((v) => -v * horizontal), ...up.map((v) => v * kilometres)];
}

export function projectionCamera(map: MaplibreMap, options: CustomRenderMethodInput): ProjectionCamera {
  const coordinates = map.style.projection?.worldCoordinateHelper;
  if (!coordinates) throw new Error('NightLayer requires MapLibre 6.18 geographic world-coordinate mapping');
  const p = options.projectionMatrix,
    projection = options.defaultProjectionData;
  const inverse = [
    1 / p[0],
    0,
    0,
    0,
    0,
    1 / p[5],
    0,
    0,
    0,
    0,
    0,
    1 / p[14],
    p[8] / p[0],
    p[9] / p[5],
    -1,
    p[10] / p[14],
  ];
  const affine = (matrix: ArrayLike<number>) => {
    const view = Array.from({ length: 16 }, (_, i) =>
      [0, 1, 2, 3].reduce((sum, k) => sum + inverse[k * 4 + (i % 4)] * matrix[Math.floor(i / 4) * 4 + k], 0),
    );
    const inverseView = inverse3([...view.slice(0, 3), ...view.slice(4, 7), ...view.slice(8, 11)]);
    return { position: transform3(inverseView, view.slice(12, 15)).map((v) => -v), inverseView };
  };
  const transition = Math.max(0, Math.min(1, projection.projectionTransition));
  const planar = transition < 1 ? affine(projection.fallbackMatrix) : undefined;
  const sphere = transition > 0 ? affine(projection.mainMatrix) : undefined;
  let worldToPhysical = Array(9).fill(0);
  let origin = sphere?.position.map((v) => v * 6371),
    planarView = Array(9).fill(0);
  if (planar) {
    const location = coordinates.lngLatFromWorld(planar.position[0], planar.position[1]);
    // Interpret the recovered camera Z at the camera, not at the moving map center.
    const altitude = planar.position[2] / coordinates.worldZFromAltitude(1000, location);
    const physical = geographicUp(location.lng, location.lat).map((v) => v * (6371 + altitude));
    const sphericalOrigin = origin;
    origin = sphericalOrigin ? physical.map((v, i) => v + (sphericalOrigin[i] - v) * transition) : physical;
    // The rendered Mercator view still uses the map center's vertical scale.
    const viewKilometres = 1 / coordinates.worldZFromAltitude(1000, map.getCenter());
    worldToPhysical = worldFrame(coordinates, planar.position, viewKilometres);
    planarView = [0, 1, 2].flatMap((i) => transform3(worldToPhysical, planar.inverseView.slice(i * 3, i * 3 + 3)));
  }
  if (!origin?.every(Number.isFinite)) throw new Error('NightLayer: invalid geographic camera');
  return {
    origin,
    planarPosition: planar?.position,
    worldToPhysical,
    inverseProjection: inverse,
    planarView,
    sphereView: sphere?.inverseView.map((v) => v * 6371) ?? Array(9).fill(0),
  };
}

export type Color3 = [number, number, number];
export type Color4 = [number, number, number, number];
export type Color = Color3 | Color4;
export type TwilightSteps = 1 | 2 | 3 | 4;
export type Twilight = {
  steps: TwilightSteps;
  /** Linear RGB illumination multipliers and application strength, all 0..1. */
  color: Color4;
  attenuation: number;
};
export type TwilightInput = TwilightSteps | { steps: TwilightSteps; color?: Color; attenuation?: number } | null;

/** Omission selects continuous lighting; RGB-only colours default to alpha 0.95. */
export function twilightSettings(input?: TwilightInput): Twilight | null {
  if (input == null) return null;
  const value = typeof input === 'number' ? { steps: input } : input;
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new TypeError('Twilight must be a step count, settings object, or null');
  if (!Number.isInteger(value.steps) || value.steps < 1 || value.steps > 4)
    throw new RangeError('Twilight steps must be 1, 2, 3, or 4');
  const color = value.color === undefined ? [0, 0, 0, 0.95] : value.color;
  if (
    !Array.isArray(color) ||
    (color.length !== 3 && color.length !== 4) ||
    Array.from(color).some((v) => typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 1)
  )
    throw new RangeError('Twilight color must have three or four components in 0..1');
  const attenuation = value.attenuation === undefined ? 0.5 : value.attenuation;
  if (!Number.isFinite(attenuation) || attenuation <= 0)
    throw new RangeError('Twilight attenuation must be finite and positive');
  return { steps: value.steps, color: [color[0], color[1], color[2], color[3] ?? 0.95], attenuation };
}

/** Geographic step lighting replaces surface illumination, never sky or fog. */
export const twilightGLSL = `
uniform float u_twilight_steps, u_twilight_attenuation;
uniform vec4 u_twilight_color;
uniform float u_thresholds[4];
vec3 twilightIllumination(float solarHeight) {
  float band=1.0;
  for(int i=1;i<4;i++)if(solarHeight<u_thresholds[i])band+=1.0;
  band=min(band,u_twilight_steps);
  if(solarHeight>u_thresholds[0])band=0.0;
  float night=pow(clamp(band/u_twilight_steps,0.0,1.0),u_twilight_attenuation);
  return mix(vec3(1.0),u_twilight_color.rgb,night*u_twilight_color.a);
}
`;

export const mercatorSphereGLSL: string = `
vec3 mercatorSphere(vec2 mercator) {
  float longitude=mercator.x*${Math.PI}*2.0+${Math.PI};
  float t=exp(${Math.PI}-(mercator.y*${Math.PI}*2.0));
  float t2=t*t;
  float denominator=t2+1.0;
  float latitudeSin=(t2-1.0)/denominator;
  float latitudeCos=2.0*t/denominator;
  return vec3(sin(longitude)*latitudeCos,latitudeSin,cos(longitude)*latitudeCos);
}
`;
export const earthIntersectionGLSL = `
float earthIntersectionRoot(vec3 origin, vec3 direction, float discriminant) {
  float h = length(origin) - 6371.0;
  float radial = h * (12742.0 + h);
  float t = dot(origin, direction);
  if (h < 0.0 || t >= 0.0 || discriminant < 0.0) return 1e30;
  return max(0.0, radial / max(-t + sqrt(discriminant), 1e-20));
}`;
export const inverseLookupGLSL = `
// RG: octahedral direction; B: range in km, -1 for sky, 0 for no hit.
// A: two 10-bit normal components and one spherical fallback bit, stored exactly as a float.
vec2 octSign(vec2 p) {
  return vec2(p.x < 0.0 ? -1.0 : 1.0, p.y < 0.0 ? -1.0 : 1.0);
}
vec2 encodeDirection(vec3 direction) {
  vec2 p = direction.xy / max(dot(abs(direction), vec3(1)), 1e-30);
  return direction.z < 0.0 ? (1.0 - abs(p.yx)) * octSign(p) : p;
}
vec3 decodeDirection(vec2 p) {
  vec3 direction = vec3(p, 1.0 - abs(p.x) - abs(p.y));
  if (direction.z < 0.0) direction.xy = (1.0 - abs(p.yx)) * octSign(p);
  return normalize(direction);
}
vec4 packGroundHit(vec3 path, vec3 normal, bool fallback) {
  uvec2 n = uvec2(round(clamp(encodeDirection(normal) * .5 + .5, 0.0, 1.0) * 1023.0));
  uint bits = n.x | (n.y << 10) | (uint(fallback) << 20);
  return vec4(encodeDirection(path), max(length(path), 1e-30), float(bits));
}
vec3 unpackNormal(uint bits) {
  return decodeDirection(vec2(bits & 1023u, (bits >> 10) & 1023u) * (2.0 / 1023.0) - 1.0);
}`;
export const inverseSkyGLSL = `
mat3 skyBasis(vec3 direction) {
  vec3 x = normalize(cross(direction, abs(direction.z) < .9 ? vec3(0,0,1) : vec3(0,1,0)));
  return mat3(x, cross(direction, x), direction);
}`;

/** Tile-local XY/DEM metres to observer-local physical geometry. */
export const relativePositionGLSL = `
uniform vec3 u_local_up, u_local_east, u_local_north, u_local_origin;
uniform vec3 u_local_mercator;
uniform mat3 u_local_to_world;
float localExpm1(float x) {
  if(abs(x)<.1) return x*(1.0+x*(.5+x*(1.0/6.0+x*(1.0/24.0+x*(1.0/120.0+x/720.0)))));
  return exp(x)-1.0;
}
vec3 localSphereDelta(vec2 tilePosition,vec2 tileScale) {
  vec2 offset=(tilePosition-vec2(4096.0))*tileScale;
  float dl=offset.x*6.28318530718;
  float em1=localExpm1(-offset.y*6.28318530718);
  float t=u_local_mercator.z;
  float halfTan=t*em1/(1.0+t*t*(1.0+em1));
  float cosm1Lat=-2.0*halfTan*halfTan/(1.0+halfTan*halfTan);
  float sp=2.0*halfTan/(1.0+halfTan*halfTan),cp=1.0+cosm1Lat;
  float s0=u_local_mercator.x,c0=u_local_mercator.y;
  float c=c0*cp-s0*sp;
  float cosm1=-2.0*sin(dl*.5)*sin(dl*.5);
  float radial=cosm1Lat+c0*c*cosm1;
  return u_local_east*(c*sin(dl))+u_local_north*(sp-s0*c*cosm1)+u_local_up*radial;
}
vec3 localNormalizeDelta(vec3 delta) {
  float k=2.0*dot(u_local_up,delta)+dot(delta,delta);
  float norm=sqrt(max(1.0+k,1e-20));
  return (delta-u_local_up*(k/(norm+1.0)))/norm;
}
vec3 localEndpoint(vec3 delta,float height) {
  vec3 relative=u_local_origin+6371.0*delta+(u_local_up+delta)*(height*.001);
  return u_local_to_world*relative;
}
vec3 localPoleDelta(float north) {
  return transpose(u_local_to_world)*vec3(0,north,0)-u_local_up;
}
`;

export function bindRelativePosition(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  tile: ArrayLike<number>,
  observer: number[],
): void {
  const mx = tile[0] + 4096 * tile[2],
    my = tile[1] + 4096 * tile[3];
  const longitude = mx * 2 * Math.PI + Math.PI;
  const t = Math.exp(Math.PI - my * 2 * Math.PI);
  const sinLat = (t * t - 1) / (t * t + 1),
    cosLat = (2 * t) / (t * t + 1);
  const up = [Math.sin(longitude) * cosLat, sinLat, Math.cos(longitude) * cosLat];
  const east = [Math.cos(longitude), 0, -Math.sin(longitude)];
  const north = [-Math.sin(longitude) * sinLat, cosLat, -Math.cos(longitude) * sinLat];
  const frame = observerFrame(observer);
  const dot = (a: number[], b: number[]) => a.reduce((sum, v, i) => sum + v * b[i], 0);
  const rotate = (v: number[]) => frame.map((axis) => dot(axis, v));
  gl.uniformMatrix3fv(uniformLocation(gl, program, 'u_local_to_world'), false, frame.flat());
  gl.uniform3fv(uniformLocation(gl, program, 'u_local_up'), rotate(up));
  gl.uniform3fv(uniformLocation(gl, program, 'u_local_east'), rotate(east));
  gl.uniform3fv(uniformLocation(gl, program, 'u_local_north'), rotate(north));
  gl.uniform3fv(uniformLocation(gl, program, 'u_local_origin'), rotate(up.map((v, i) => 6371 * v - observer[i])));
  gl.uniform3f(uniformLocation(gl, program, 'u_local_mercator'), sinLat, cosLat, t);
}

function observerFrame(observer: number[]): number[][] {
  const radius = Math.hypot(...observer);
  const up = observer.map((v) => v / radius);
  const longitude = Math.atan2(observer[0], observer[2]);
  return [
    [Math.cos(longitude), 0, -Math.sin(longitude)],
    [-up[1] * Math.sin(longitude), Math.hypot(up[0], up[2]), -up[1] * Math.cos(longitude)],
    up,
  ];
}

// Native tile mesh and DEM binding.
type NativeTerrain = NonNullable<MaplibreMap['terrain']>;
type NativeMesh = ReturnType<NativeTerrain['getTerrainMesh']>;
type Projection = ReturnType<CustomRenderMethodInput['getProjectionData']>;
type TerrainData = ReturnType<NativeTerrain['getTerrainData']>;

export function bindTileProjection(gl: WebGL2RenderingContext, p: WebGLProgram, projection: Projection): void {
  for (const [name, value] of [
    ['u_projection_matrix', projection.mainMatrix],
    ['u_projection_fallback_matrix', projection.fallbackMatrix],
  ] as const)
    gl.uniformMatrix4fv(uniformLocation(gl, p, name), false, value);
  gl.uniform4fv(uniformLocation(gl, p, 'u_projection_tile_mercator_coords'), projection.tileMercatorCoords);
  gl.uniform4fv(uniformLocation(gl, p, 'u_projection_clipping_plane'), projection.clippingPlane);
  gl.uniform1f(uniformLocation(gl, p, 'u_projection_transition'), projection.projectionTransition);
}

export const terrainElevationGLSL = `
uniform sampler2D u_dem;
uniform mat4 u_dem_matrix;
uniform vec4 u_dem_unpack;
uniform float u_dem_size,u_dem_exaggeration;
float demMetres(ivec2 cell) {
  ivec2 hi=textureSize(u_dem,0)-1;
  vec4 encoded=texelFetch(u_dem,clamp(cell,ivec2(0),hi),0)*255.0;
  return dot(encoded.rgb,u_dem_unpack.rgb)-u_dem_unpack.a;
}
float terrainElevation(vec2 tile) {
  #ifdef HAS_DEM
    #ifdef GLOBE
      if(tile.y < -32767.5 || tile.y > 32766.5) return 0.0;
    #endif
    vec2 coordinate=(u_dem_matrix*vec4(tile,0,1)).xy*u_dem_size+1.5;
    ivec2 cell=ivec2(floor(coordinate));
    vec2 weight=fract(coordinate);
    vec2 row=vec2(mix(demMetres(cell),demMetres(cell+ivec2(1,0)),weight.x),
      mix(demMetres(cell+ivec2(0,1)),demMetres(cell+ivec2(1,1)),weight.x));
    return mix(row.x,row.y,weight.y)*u_dem_exaggeration;
  #else
    return 0.0;
  #endif
}
`;

export function bindTerrainElevation(gl: WebGL2RenderingContext, p: WebGLProgram, terrain: TerrainData): void {
  gl.activeTexture(gl.TEXTURE3);
  gl.bindTexture(gl.TEXTURE_2D, terrain.texture);
  gl.uniform1i(uniformLocation(gl, p, 'u_dem'), 3);
  gl.uniformMatrix4fv(uniformLocation(gl, p, 'u_dem_matrix'), false, terrain.u_terrain_matrix);
  gl.uniform4fv(uniformLocation(gl, p, 'u_dem_unpack'), terrain.u_terrain_unpack);
  gl.uniform1f(uniformLocation(gl, p, 'u_dem_size'), terrain.u_terrain_dim);
  gl.uniform1f(uniformLocation(gl, p, 'u_dem_exaggeration'), terrain.u_terrain_exaggeration);
}

export class TileSurfaceProgram {
  readonly program: WebGLProgram;
  #vao: WebGLVertexArrayObject;

  constructor(gl: WebGL2RenderingContext, vertex: string, fragment: string) {
    this.program = compileProgram(gl, fragment, vertex);
    this.#vao = gl.createVertexArray()!;
  }
  bind(gl: WebGL2RenderingContext, projection: Projection, terrain: TerrainData | null, skirt: number): void {
    const p = this.program;
    gl.useProgram(p);
    gl.bindVertexArray(this.#vao);
    bindTileProjection(gl, p, projection);
    gl.uniform1f(uniformLocation(gl, p, 'u_skirt'), skirt);
    if (terrain) bindTerrainElevation(gl, p, terrain);
  }
  #bindVertex(gl: WebGL2RenderingContext, mesh: NativeMesh, offset: number): void {
    const vertex = mesh.vertexBuffer;
    const attribute = vertex.attributes.find((a) => a.name === 'a_pos3d' || a.name === 'a_pos');
    if (!attribute) throw new Error('NightLayer: unsupported MapLibre terrain vertex layout');
    const types = { Int16: gl.SHORT, Uint16: gl.UNSIGNED_SHORT, Float32: gl.FLOAT };
    const type = types[attribute.type as keyof typeof types];
    if (!type) throw new Error(`NightLayer: unsupported terrain attribute type ${attribute.type}`);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertex.buffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(
      0,
      attribute.components,
      type,
      false,
      vertex.itemSize,
      attribute.offset + vertex.itemSize * offset,
    );
  }
  draw(gl: WebGL2RenderingContext, mesh: NativeMesh): void {
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.indexBuffer.buffer);
    for (const segment of mesh.segments.get()) {
      this.#bindVertex(gl, mesh, segment.vertexOffset);
      gl.drawElements(gl.TRIANGLES, segment.primitiveLength * 3, gl.UNSIGNED_SHORT, segment.primitiveOffset * 6);
    }
  }
  dispose(gl: WebGL2RenderingContext): void {
    gl.deleteVertexArray(this.#vao);
    gl.deleteProgram(this.program);
  }
}

export function projectionVertexPrelude(options: CustomRenderMethodInput): string {
  return `#version 300 es
precision highp float;
precision highp int;
${options.shaderData.define}
${options.shaderData.vertexShaderPrelude}
`;
}

export function tileVertexPrelude(options: CustomRenderMethodInput, terrain: boolean): string {
  return (
    projectionVertexPrelude(options) +
    `
${terrain ? '#define HAS_DEM' : ''}
${terrainElevationGLSL}
layout(location=0) in vec3 a_tile;
uniform float u_skirt;
`
  );
}

const RADIUS = 6371;

/** Narrow native map coverage toward the spherical horizon with the atmosphere fade. */
export function groundHorizonChordSquared(radius: number, weight: number): number {
  if (weight <= 0) return 4; // The largest possible squared chord on a unit sphere.
  const horizon = (2 * Math.max(0, radius - RADIUS)) / radius;
  if (weight >= 1) return horizon;
  // Shrink the geographic range on a log scale rather than fade coverage.
  // A small floor keeps the interpolation defined for observers at sea level.
  const floor = (2 * 1e-6) / RADIUS;
  return (horizon + floor) * ((4 + floor) / (horizon + floor)) ** (1 - weight) - floor;
}
const AZIMUTHS = 256;
type Frame = {
  projection: ReturnType<CustomRenderMethodInput['getProjectionData']>;
  basis: number[];
  height: number;
  distance: number;
  dip: number;
  mercator: number[];
  anchor: number[];
};

/** A virtual tile near the observer keeps the standard forward projection
 * precise. Geography is known before rendering; no screen/CRS inverse is used. */
export function forwardFrame(
  camera: Pick<ProjectionCamera, 'origin' | 'planarPosition'>,
  options: CustomRenderMethodInput,
): Frame {
  const basis = observerFrame(camera.origin).flat();
  const height = Math.max(Math.hypot(...camera.origin) - RADIUS, 1e-6);
  const distance = Math.sqrt(height * (2 * RADIUS + height));
  const latitude = Math.asin(Math.max(-1, Math.min(1, basis[7])));
  const longitude = Math.atan2(camera.origin[0], camera.origin[2]);
  const t = Math.tan(Math.PI / 4 + Math.max(-Math.PI / 2 + 1e-8, Math.min(Math.PI / 2 - 1e-8, latitude)) / 2);
  const normalizedX = 0.5 + longitude / (2 * Math.PI);
  const mx = normalizedX + Math.round((camera.planarPosition?.[0] ?? normalizedX) - normalizedX);
  const my = 0.5 - Math.log(t) / (2 * Math.PI);
  const z = Math.max(
    0,
    Math.min(18, Math.floor(-Math.log2(distance / (2 * Math.PI * RADIUS * Math.max(Math.cos(latitude), 1e-8))))),
  );
  const scale = 2 ** z,
    wrap = Math.floor(mx);
  const x = Math.min(scale - 1, Math.floor((mx - wrap) * scale));
  const y = Math.max(0, Math.min(scale - 1, Math.floor(my * scale)));
  const projection = options.getProjectionData({
    tileID: { canonical: { x, y, z }, wrap },
    applyTerrainMatrix: false,
    applyGlobeMatrix: true,
  });
  return {
    projection,
    basis,
    height,
    distance,
    dip: Math.atan2(distance, RADIUS),
    mercator: [Math.sin(latitude), Math.cos(latitude), t],
    anchor: [(mx * scale - x - wrap * scale) * 8192, (my * scale - y) * 8192, scale * 8192],
  };
}

export function bindForwardFrame(gl: WebGL2RenderingContext, program: WebGLProgram, frame: Frame): void {
  bindTileProjection(gl, program, frame.projection);
  gl.uniformMatrix3fv(uniformLocation(gl, program, 'u_forward_basis'), false, frame.basis);
  gl.uniform4f(uniformLocation(gl, program, 'u_forward_geometry'), frame.height, frame.distance, frame.dip, RADIUS);
  gl.uniform3fv(uniformLocation(gl, program, 'u_forward_mercator'), frame.mercator);
  gl.uniform3fv(uniformLocation(gl, program, 'u_forward_anchor'), frame.anchor);
}

/** Positions are observer-footpoint ENU kilometres. Sky and sea share the
 * tangent ring and use the same native projection through the transition. */
export const forwardGeometryGLSL: string = `
uniform mat3 u_forward_basis;
uniform vec4 u_forward_geometry; // height, tangent distance, dip, Earth radius
uniform vec3 u_forward_mercator; // sin/cos observer latitude, tan(pi/4+lat/2)
uniform vec3 u_forward_anchor; // observer tile XY, tile units per world unit
vec3 forwardHorizon(vec2 azimuth) {
  float h=u_forward_geometry.x,r=u_forward_geometry.w,D=u_forward_geometry.y;
  return vec3(azimuth*(r*D/(r+h)),-r*h/(r+h));
}
vec3 forwardSky(vec2 polar) {
  vec2 azimuth=vec2(cos(polar.x),sin(polar.x));
  if(polar.y==0.0)return forwardHorizon(azimuth);
  float dip=u_forward_geometry.z;
  float angle=-dip+polar.y*(1.570796326795+(polar.y>0.0?dip:-dip));
  return vec3(azimuth*cos(angle),sin(angle))*u_forward_geometry.y+vec3(0,0,u_forward_geometry.x);
}
vec3 forwardSea(vec2 polar) {
  if(polar.y==0.0)return forwardHorizon(vec2(cos(polar.x),sin(polar.x)));
  float angle=(1.0+polar.y)*u_forward_geometry.z,r=u_forward_geometry.w;
  float halfSine=sin(angle*.5);
  return vec3(vec2(cos(polar.x),sin(polar.x))*(r*sin(angle)),-2.0*r*halfSine*halfSine);
}
float forwardLog1p(float x) {
  return abs(x)<.01?x*(1.0+x*(-.5+x*(.333333333333+x*(-.25+x*.2)))):log(1.0+x);
}
vec4 forwardPosition(vec3 point) {
  float R=u_forward_geometry.w;
  float z=R+point.z,s=u_forward_mercator.x,c=u_forward_mercator.y;
  float a=z*c-point.y*s,b=z*s+point.y*c;
  float radial=length(vec2(a,point.x));
  float dl=atan(point.x,a);
  float dy;
  if(a>0.0) {
    float difference=point.x*point.x/max(radial+a,1e-20);
    float dp=atan(point.y-difference*s,z+difference*c);
    float k=tan(dp*.5),t=u_forward_mercator.z;
    dy=-(forwardLog1p(k/t)-forwardLog1p(-t*k))/6.28318530718;
  } else {
    float latitude=clamp(atan(b,radial),-1.5707962,1.5707962);
    dy=-log(tan(.785398163397+latitude*.5)/u_forward_mercator.z)/6.28318530718;
  }
  vec2 tile=u_forward_anchor.xy+vec2(dl/6.28318530718,dy)*u_forward_anchor.z;
  // Stable altitude avoids subtracting two Earth-sized radii.
  float norm=length(vec3(point.xy,z));
  float altitude=(dot(point,point)+2.0*R*point.z)/(norm+R)*1000.0;
  #ifdef GLOBE
    // Native forward interpolation, with the known sphere position. This
    // also keeps exact poles representable without a singular Mercator inverse.
    vec4 clip=interpolateProjectionFor3D(tile,u_forward_basis*vec3(point.xy,z)/norm,altitude);
  #else
    vec4 clip=projectTileFor3D(tile,altitude);
  #endif
  // NightLayer's sky/sea supplement is not cut by the native map's near/far
  // planes. Preserve native X/Y/W and behind-camera clipping, not native Z.
  clip.z=0.0;
  return clip;
}
vec4 projectSky(vec3 direction) {
  vec3 local=transpose(u_forward_basis)*direction*u_forward_geometry.y+vec3(0,0,u_forward_geometry.x);
  return forwardPosition(local);
}`;

export function forwardVertexPrelude(options: CustomRenderMethodInput): string {
  return `#version 300 es
precision highp float;
precision highp int;
${options.shaderData.define}
${options.shaderData.vertexShaderPrelude}
${forwardGeometryGLSL}
`;
}

const LOWER_RINGS = 32;
export function forwardMesh(): { vertices: Float32Array; indices: Uint16Array } {
  const lower = LOWER_RINGS,
    upper = 64;
  const rings = lower + upper,
    stride = AZIMUTHS + 1;
  const vertices = new Float32Array((rings + 1) * stride * 2);
  const indices = new Uint16Array(rings * AZIMUTHS * 6);
  for (let ring = 0; ring <= rings; ring++) {
    const level = ring <= lower ? -(((lower - ring) / lower) ** 2) : ((ring - lower) / upper) ** 2;
    for (let azimuth = 0; azimuth <= AZIMUTHS; azimuth++) {
      const i = (ring * stride + azimuth) * 2;
      vertices[i] = azimuth === AZIMUTHS ? 0 : (azimuth / AZIMUTHS) * 2 * Math.PI;
      vertices[i + 1] = level;
    }
  }
  for (let ring = 0, i = 0; ring < rings; ring++)
    for (let azimuth = 0; azimuth < AZIMUTHS; azimuth++) {
      const a = ring * stride + azimuth,
        b = a + stride;
      indices.set([a, b, a + 1, a + 1, b, b + 1], i);
      i += 6;
    }
  return { vertices, indices };
}

/** Fill the shared inverse LUT with sky, then the visible sea-level sphere. */
export class ForwardSurface {
  #programs = new Map<string, WebGLProgram>();
  #vao: WebGLVertexArrayObject;
  #vertices: WebGLBuffer;
  #indices: WebGLBuffer;
  #count: number;
  constructor(gl: WebGL2RenderingContext) {
    const mesh = forwardMesh();
    const vao = gl.createVertexArray(),
      vertices = gl.createBuffer(),
      indices = gl.createBuffer();
    if (!vao || !vertices || !indices) {
      gl.deleteVertexArray(vao);
      gl.deleteBuffer(vertices);
      gl.deleteBuffer(indices);
      throw new Error('Failed to allocate NightLayer forward geometry');
    }
    this.#vao = vao;
    this.#vertices = vertices;
    this.#indices = indices;
    this.#count = mesh.indices.length;
    gl.bindVertexArray(this.#vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.#vertices);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.vertices, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.#indices);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);
  }
  render(gl: WebGL2RenderingContext, camera: ProjectionCamera, options: CustomRenderMethodInput): void {
    const key = options.shaderData.variantName;
    let program = this.#programs.get(key);
    if (!program) {
      program = compileProgram(
        gl,
        forwardSurfaceFragment,
        forwardVertexPrelude(options) +
          `
layout(location=0) in vec2 a_polar;
uniform bool u_sea;
out highp vec3 v_point;
void main() {
  v_point=u_sea?forwardSea(a_polar):forwardSky(a_polar);
  gl_Position=forwardPosition(v_point);
}`,
      );
      this.#programs.set(key, program);
    }
    gl.useProgram(program);
    bindForwardFrame(gl, program, forwardFrame(camera, options));
    gl.bindVertexArray(this.#vao);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.BLEND);
    gl.colorMask(true, true, true, true);
    const sea = uniformLocation(gl, program, 'u_sea');
    gl.disable(gl.STENCIL_TEST);
    gl.uniform1i(sea, 0);
    gl.drawElements(gl.TRIANGLES, this.#count, gl.UNSIGNED_SHORT, 0);
    // The sea is a fallback, not a depth occluder for the native terrain.
    gl.enable(gl.STENCIL_TEST);
    gl.stencilFunc(gl.ALWAYS, 1, 0xff);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
    gl.uniform1i(sea, 1);
    gl.drawElements(gl.TRIANGLES, LOWER_RINGS * AZIMUTHS * 6, gl.UNSIGNED_SHORT, 0);
  }
  dispose(gl: WebGL2RenderingContext): void {
    for (const program of this.#programs.values()) gl.deleteProgram(program);
    this.#programs.clear();
    gl.deleteBuffer(this.#vertices);
    gl.deleteBuffer(this.#indices);
    gl.deleteVertexArray(this.#vao);
  }
}

const forwardSurfaceFragment = `#version 300 es
precision highp float;
precision highp int;
in highp vec3 v_point;
uniform mat3 u_forward_basis;
uniform vec4 u_forward_geometry;
uniform bool u_sea;
layout(location=0) out vec4 ray;
${inverseLookupGLSL}
void main() {
  vec3 path=u_forward_basis*(v_point-vec3(0,0,u_forward_geometry.x));
  if(u_sea){
    vec3 up=normalize(u_forward_basis*(v_point+vec3(0,0,u_forward_geometry.w)));
    ray=packGroundHit(path,up,true);
  }else{
    ray=vec4(encodeDirection(path),-1.0,0.0);
  }
}`;
