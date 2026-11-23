"""Refractive geometry, solar disk and celestial optical atlas generation."""

import json
from time import perf_counter
from types import SimpleNamespace
import moderngl
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from lut_model import HERE, Atmosphere, apparent_view, dip


class Geometry:
    def __init__(self):
        self.context = moderngl.create_standalone_context(require=430, backend="egl")
        self.program = self.context.compute_shader(GEOMETRY_SHADER)
        print("GPU", self.context.info["GL_RENDERER"], flush=True)

    def evaluate(self, queries, step=0.5):
        queries = np.asarray(queries, dtype="f8")
        n = len(queries)
        source = self.context.buffer(queries.tobytes())
        result = self.context.buffer(reserve=n * 8 * 8)
        try:
            source.bind_to_storage_buffer(0)
            result.bind_to_storage_buffer(1)
            self.program["count"] = n
            self.program["step_size"] = step
            self.program.run(group_x=(n + 63) // 64)
            self.context.memory_barrier()
            out = np.frombuffer(result.read(), "f8").reshape(n, 8).copy()
            if not np.all(out[:, 7] == 1):
                raise RuntimeError("Some refractive paths did not reach a boundary")
            out[:, 4] = np.arcsin(np.clip(out[:, 4], -1, 1)) - out[:, 5]
            return out
        finally:
            source.release()
            result.release()


def build_geometry(gpu, height_count=128, angle_count=512):
    output = HERE / "geometry"
    output.mkdir(exist_ok=True)
    heights = 0.002 + (99.999 - 0.002) * np.linspace(0, 1, height_count) ** 2
    relative = 0.5 * np.sinh(
        np.linspace(np.arcsinh(-180), np.arcsinh(201), angle_count)
    )
    model = Atmosphere()
    metadata = dict(
        refractivity=model.refractivity,
        scaleHeight=model.scale_height,
        height=heights.tolist(),
        relativeElevationDegrees=relative.tolist(),
        viewAxis="local apparent elevation minus refracted sea-level horizon",
        fields=[
            "rayleighColumn",
            "mieColumn",
            "ozoneColumn",
            "length",
            "externalElevation",
            "polarDisplacement",
            "ground",
            "finished",
        ],
        precision="float64",
        stepKm=0.125,
    )
    manifest = output / "metadata.json"
    text = json.dumps(metadata, indent=2) + "\n"
    if manifest.exists() and manifest.read_text() != text:
        raise ValueError("Existing geometry has different generation parameters")
    manifest.write_text(text)
    for i, height in enumerate(heights):
        dest = output / f"height-{i:03d}.npy"
        if dest.exists():
            continue
        start = perf_counter()
        angle = np.clip(
            model.horizon(height) + np.deg2rad(relative), -np.pi / 2, np.pi / 2
        )
        query = np.column_stack([np.full(angle_count, height), np.sin(angle)])
        result = gpu.evaluate(query, 0.125)
        temporary = dest.with_suffix(".pending.npy")
        np.save(temporary, result)
        temporary.replace(dest)
        print("HEIGHT", i, "seconds", round(perf_counter() - start, 3), flush=True)


def invert(heights, relative, paths):
    model = Atmosphere()
    source_relative = np.r_[0.0, np.geomspace(1e-05, 180.0, 511)]
    output = np.zeros((len(heights), len(source_relative), 6), dtype="f8")
    horizons = []
    for i, height in enumerate(heights):
        path = paths[i]
        apparent = np.clip(
            model.horizon(height) + np.deg2rad(relative), -np.pi / 2, np.pi / 2
        )
        use = (path[:, 6] == 0) & (np.diff(np.r_[-np.inf, path[:, 4]]) > 1e-12)
        true, local = (path[use, 4], apparent[use])
        if len(true) < 4 or not np.all(np.diff(true) > 0):
            raise ValueError(f"Nonmonotone external ray map at height {height}")
        lower = float(true[0])
        horizons.append(lower)
        requested = lower + np.deg2rad(source_relative)
        valid = requested <= true[-1]
        query = np.clip(requested, true[0], true[-1])
        inverse = PchipInterpolator(true, local, extrapolate=False)
        local_query = inverse(query)
        derivative = inverse.derivative()(query)
        jacobian = np.cos(local_query) / np.maximum(np.cos(query), 1e-12) * derivative
        jacobian = np.where(
            np.abs(np.cos(query)) < 1e-07, 1 / model.index(height) ** 2, jacobian
        )
        output[i, :, :3] = PchipInterpolator(true, path[use, :3], axis=0)(query)
        output[i, :, 3] = local_query
        output[i, :, 4] = np.maximum(jacobian, 0)
        output[i, :, 5] = valid
    return (source_relative, np.array(horizons), output)


def build_solar():
    args = SimpleNamespace(refine=True)
    folder = HERE / "geometry"
    metadata = json.loads((folder / "metadata.json").read_text())
    heights = np.array(metadata["height"])
    relative = np.array(metadata["relativeElevationDegrees"])
    paths = np.stack(
        [np.load(folder / f"height-{i:03d}.npy") for i in range(len(heights))]
    )
    if args.refine:
        gpu = Geometry()
        model = Atmosphere()
        extra_relative = np.geomspace(1e-06, 3.0, 128)
        extra_paths = []
        try:
            for i, height in enumerate(heights):
                angle = model.horizon(height) + np.deg2rad(extra_relative)
                query = np.column_stack([np.full(len(angle), height), np.sin(angle)])
                values = gpu.evaluate(query, 0.125)
                if np.any(values[:, 6] != 0):
                    raise ValueError(f"Sky refinement hit ground at height {height}")
                extra_paths.append(values)
                print("SOLAR_REFINE", i, flush=True)
        finally:
            gpu.context.release()
        combined = np.r_[relative, extra_relative]
        order = np.argsort(combined)
        relative = combined[order]
        paths = np.concatenate([paths, np.stack(extra_paths)], axis=1)[:, order]
    axis, domain, values = invert(heights, relative, paths)
    assert np.isfinite(values).all() and np.all(values[..., :3] >= 0)
    destination = HERE / "solar-geometry.npz"
    np.savez(
        destination,
        height=heights,
        sourceRelativeElevationDegrees=axis,
        lowerExternalElevation=domain,
        values=values,
    )
    print("WROTE", destination, "valid", int(values[..., 5].sum()), flush=True)


class SolarGeometry:
    def __init__(self):
        with np.load(HERE / "solar-geometry.npz") as z:
            self.height = z["height"]
            self.lower = z["lowerExternalElevation"]
            self.relative = z["sourceRelativeElevationDegrees"]
            self.values = z["values"]
        exterior = self.lower[:, None] + np.deg2rad(self.relative[None, :])
        near_zenith = exterior >= np.pi / 2 - 1e-07
        self.values[..., 3] = np.where(near_zenith, np.pi / 2, self.values[..., 3])
        self.values[..., 4] = np.where(
            near_zenith,
            1 / Atmosphere().index(self.height[:, None]) ** 2,
            self.values[..., 4],
        )
        self.values[..., 3] -= np.minimum(exterior, np.pi / 2)
        self.coordinates = np.log1p(self.relative / 0.0001)
        self.table = RegularGridInterpolator(
            (self.height, self.coordinates),
            self.values,
            bounds_error=False,
            fill_value=None,
        )

    def evaluate(self, height, exterior_elevation):
        height = np.clip(height, self.height[0], self.height[-1])
        lower = np.interp(height, self.height, self.lower)
        relative = np.rad2deg(exterior_elevation - lower)
        query = np.column_stack([height, np.log1p(np.maximum(relative, 0) / 0.0001)])
        value = self.table(query)
        value[:, 3] += exterior_elevation
        visible = relative >= 0
        return (value, visible)


def disk_directions(
    center, radial_count=4, azimuth_count=16, radius=np.deg2rad(0.2666667)
):
    """Uniform-radiance disk, quadrature normalized by its exterior solid angle."""
    if radial_count == 0:
        return (center[None], np.ones(1))
    axis = (
        np.array([0.0, 0.0, 1.0]) if abs(center[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    )
    first = np.cross(center, axis)
    first /= np.linalg.norm(first)
    second = np.cross(center, first)
    node, weight = leggauss(radial_count)
    cosine = 1 - (node + 1) * 0.5 * (1 - np.cos(radius))
    azimuth = (np.arange(azimuth_count) + 0.5) * (2 * np.pi / azimuth_count)
    tangent = np.cos(azimuth)[:, None] * first + np.sin(azimuth)[:, None] * second
    direction = (
        cosine[:, None, None] * center + np.sqrt(1 - cosine**2)[:, None, None] * tangent
    )
    weights = np.broadcast_to(
        weight[:, None] / (2 * azimuth_count), (radial_count, azimuth_count)
    )
    return (direction.reshape(-1, 3), weights.ravel())


def build_celestial():
    gpu = Geometry()
    heights = np.maximum(0.002, 100 * np.linspace(0, 1, 64) ** 3)
    heights[-1] = 99.999
    below = -0.02 * np.expm1(np.linspace(np.log1p(np.sqrt(90 / 0.02)), 0, 64)) ** 2
    below[-1] = -1e-07
    above = 0.02 * np.expm1(np.linspace(0, np.log1p(100.1 / 0.02), 64))
    above[0] = 1e-06
    relative = np.r_[below, above]
    table = np.zeros((3, 64, 128, 4), "f4")
    lookup = SolarGeometry()
    for p, h in enumerate(heights):
        view = np.clip(np.deg2rad(relative) - dip(h), -np.pi / 2, np.pi / 2)
        apparent = np.array([apparent_view(h, v) for v in view])
        ends = gpu.evaluate(np.column_stack([np.full(128, h), np.sin(apparent)]), 0.125)
        np.testing.assert_array_equal(ends[:, 6], np.r_[np.ones(64), np.zeros(64)])
        table[0, p, :, :2] = np.log(np.maximum(ends[:, :2], 1e-30))
        table[0, p, :, 2] = ends[:, 2]
        table[1, p] = ends[:, [4, 5, 3, 6]]
        table[1, p, :, 0] -= view
        lower = np.interp(h, lookup.height, lookup.lower)
        solar_relative = 0.0001 * np.expm1(
            np.linspace(0, 1, 128) * np.log1p(180 / 0.0001)
        )
        solar, valid = lookup.evaluate(
            np.full(128, h), lower + np.deg2rad(solar_relative)
        )
        table[2, p] = np.column_stack(
            [
                solar[:, 3] - (lower + np.deg2rad(solar_relative)),
                solar[:, 4],
                np.full(128, lower),
                valid,
            ]
        )
        print("CELESTIAL_OPTICAL", p, flush=True)
    assert np.isfinite(table).all()
    np.save(HERE / "refraction-runtime.npy", table)
    gpu.context.release()


def top_shell_limit(table):
    """Rebake only the top escaping-sky row; retain the physical grazing anchor."""
    table = np.asarray(table, dtype="<f4").copy()
    before = table.copy()
    height = 99.999
    radius = 6371.0 + height
    geometric = -np.arccos(6371.0 / radius)
    refractive_index = 1 + 0.00029 * (np.exp(-height / 8) - np.exp(-12.5))
    sea_index = 1 + 0.00029 * (1 - np.exp(-12.5))
    physical = -np.arccos(sea_index * 6371.0 / (refractive_index * radius))
    relative = 0.02 * np.expm1(np.arange(64) / 63 * np.log(5006.0))
    relative[0] = 1e-06
    view = np.minimum(geometric + np.deg2rad(relative), np.pi / 2)
    last_negative = int(np.flatnonzero(view < 0)[-1])
    if np.all(table[1, 63, 64 + last_negative :, :] == 0):
        return table
    cutoff = view[last_negative]
    t = np.clip((view - geometric) / (cutoff - geometric), 0, 1)
    mapped = view + (physical - geometric) * (1 - t) ** 2 * (1 + 2 * t)
    gpu = Geometry()
    try:
        queries = np.column_stack(
            [np.full(last_negative, height), np.sin(mapped[:last_negative])]
        )
        ends = gpu.evaluate(queries, 0.125)
        assert np.all(ends[:, 6] == 0), "A rebaked sky ray hit the ground"
    finally:
        gpu.context.release()
    table[0, 63, 64 : 64 + last_negative, :2] = np.log(np.maximum(ends[:, :2], 1e-30))
    table[0, 63, 64 : 64 + last_negative, 2] = ends[:, 2]
    table[1, 63, 64 : 64 + last_negative] = ends[:, [4, 5, 3, 6]]
    table[1, 63, 64 : 64 + last_negative, 0] -= view[:last_negative]
    table[:, 63, 64] = before[:, 63, 64]
    table[0, 63, 64 + last_negative :, :2] = np.log(1e-30)
    table[0, 63, 64 + last_negative :, 2:] = 0
    table[1, 63, 64 + last_negative :] = 0
    assert np.isfinite(table).all()
    np.testing.assert_array_equal(table[:, :63], before[:, :63])
    np.testing.assert_array_equal(table[:, 63, :64], before[:, 63, :64])
    np.testing.assert_array_equal(table[2], before[2])
    return table


GEOMETRY_SHADER = r"""#version 430
// Double precision is used only offline, especially for the grazing rays.
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Queries { dvec2 queries[]; };
layout(std430,binding=1) writeonly buffer Results { dvec4 results[]; };
uniform int count;
uniform float step_size;
double dexp(double x) {
  int k=int(floor(x/0.6931471805599453094LF));
  double r=x-double(k)*0.6931471805599453094LF;
  double sum=1.0LF,term=1.0LF;
  for(int i=1;i<=16;i++){term*=r/double(i);sum+=term;}
  return ldexp(sum,k);
}
double indexAt(double h) {
  return 1.0LF+0.00029LF*(dexp(-clamp(h,0.0,100.0)/8.0)-dexp(-12.5LF));
}
// (height, local direction cosine, polar displacement, path length)
dvec4 derivative(dvec4 q) {
  double mu=q.y,r=6371.0+q.x;
  double grad=-0.00029LF/8.0LF*dexp(-clamp(q.x,0.0,100.0)/8.0);
  return dvec4(mu,(1.0-mu*mu)*(1.0/r+grad/indexAt(q.x)),
               sqrt(max(1.0-mu*mu,0.0))/r,1.0);
}
dvec3 density(double h) {
  return dvec3(dexp(-max(h,0.0)/8.0LF),dexp(-max(h,0.0)/1.2LF),
               max(0.0,1.0-abs(h-25.0)/15.0));
}
void advance(dvec4 q,double ds,out dvec4 next,out dvec3 columns) {
  dvec4 a=derivative(q);
  dvec4 b=derivative(q+a*(ds*.5));
  dvec4 c=derivative(q+b*(ds*.5));
  dvec4 d=derivative(q+c*ds);
  next=q+ds/6.0*(a+2.0*b+2.0*c+d);
  columns=ds/6.0*(density(q.x)+2.0*density((q+a*(ds*.5)).x)
    +2.0*density((q+b*(ds*.5)).x)+density((q+c*ds).x));
}
void main() {
  uint id=gl_GlobalInvocationID.x;
  if(id>=uint(count))return;
  dvec2 inputRay=queries[id];
  dvec4 q=dvec4(inputRay,0.0,0.0);
  dvec3 columns=dvec3(0.0);
  double impact=(6371.0LF+q.x)*indexAt(q.x)*sqrt(max(1.0LF-q.y*q.y,0.0LF));
  bool expectedGround=q.y<0.0LF&&impact<6371.0LF*indexAt(0.0LF);
  bool ground=false,finished=false;
  for(int i=0;i<65536;i++){
    double ds=double(step_size);
    dvec4 next;dvec3 increment;
    advance(q,ds,next,increment);
    if(expectedGround&&q.y<0.0LF&&next.y>=0.0LF) {
      // A grazing step can enter and leave the ground between its endpoints.
      // Bracket the radial turning point first, then the first ground hit.
      double lo=0.0LF,hi=ds;
      for(int j=0;j<40;j++) {
        double mid=(lo+hi)*.5LF;
        advance(q,mid,next,increment);
        if(next.y>=0.0LF)hi=mid;else lo=mid;
      }
      ds=(lo+hi)*.5LF;
      advance(q,ds,next,increment);
    }
    if(next.x<0.0||next.x>100.0){
      ground=next.x<0.0;
      double lo=0.0,hi=ds;
      for(int j=0;j<40;j++){
        double mid=(lo+hi)*.5;
        advance(q,mid,next,increment);
        if(ground?next.x<0.0:next.x>100.0)hi=mid;else lo=mid;
      }
      advance(q,(lo+hi)*.5,next,increment);
      columns+=increment;q=next;finished=true;break;
    }
    columns+=increment;q=next;
  }
  // angle at exit in the observer's original meridian frame:
  // asin(mu_exit) - polar displacement. Host computes asin in float64.
  results[id*2u]=dvec4(columns,q.w);
  results[id*2u+1u]=dvec4(q.y,q.z,ground?1.0:0.0,finished?1.0:0.0);
}
"""
