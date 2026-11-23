"""Order-eight spectral sources, physical references, palette and irradiance."""

import hashlib
import json
from time import perf_counter
from types import SimpleNamespace
import moderngl
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator, BSpline
from scipy.special import eval_legendre, sph_harm_y
from scipy import sparse
from scipy.sparse.linalg import spsolve
from lut_model import HERE, MODEL, META, RGB, Atmosphere, trace, apparent_view, dip, ray_ends
from lut_model import from_grid, to_grid, sample_ray, unwarp, ANGLE_MIN, ANGLE_SPAN
from lut_optics import SolarGeometry, disk_directions

AMIN, ASPAN = np.arcsinh(-180), np.arcsinh(220) - np.arcsinh(-180)
R = 6371.0


def spectral_weights():
    return np.array(MODEL["weights48"]) * np.array(MODEL["solarIrradiance48"])[:, None]


def build_curved_references():
    folder = HERE / "reference-curved"
    folder.mkdir(exist_ok=True)
    weights = spectral_weights()
    gpu = Integrator()
    configuration = json.dumps(gpu.config, sort_keys=True)
    try:
        for p in range(16):
            dest = folder / f"p{p:02d}.npz"
            if dest.exists():
                with np.load(dest) as previous:
                    if (
                        str(previous["sourceConfig"]) == configuration
                        and int(previous["steps"]) == 32
                    ):
                        continue
            start = perf_counter()
            phys, distances, u, nu = geometry(p)
            paths = np.stack(
                [
                    path_samples(
                        phys[row, 0, 0, 0], phys[row, 0, 0, 2], distances[row], 32
                    )[0]
                    for row in range(len(nu))
                ]
            )
            query = np.stack(
                [
                    phys[..., 0],
                    phys[..., 1],
                    phys[..., 3],
                    np.broadcast_to(np.arange(len(nu))[:, None, None], phys.shape[:-1]),
                ],
                axis=-1,
            )
            xyz = gpu.evaluate(query.reshape(-1, 4), paths, weights=weights).reshape(
                *phys.shape[:-1], 43, 2, 3
            )
            xyz = phase_normalize(phys, xyz)
            if not np.isfinite(xyz).all() or np.any(xyz < 0):
                raise ArithmeticError("Invalid spectral reference")
            pending = dest.with_suffix(".pending.npz")
            np.savez(
                pending,
                physical=phys,
                distances=distances,
                u=u,
                nu=nu,
                xyz=xyz,
                steps=32,
                sourceConfig=configuration,
            )
            pending.replace(dest)
            print(
                "CURVED_REFERENCE",
                p,
                "seconds",
                round(perf_counter() - start, 2),
                flush=True,
            )
    finally:
        gpu.close()


def harmonics(mu, phi, degree):
    mu, phi = np.broadcast_arrays(mu, phi)
    values, ell = ([], [])
    for l in range(degree + 1):
        for m in range(l + 1):
            values.append(
                sph_harm_y(l, m, np.arccos(np.clip(mu, -1, 1)), phi).real
                * (1 if m == 0 else np.sqrt(2))
            )
            ell.append(l)
    return (np.stack(values, -1).astype("f4"), np.array(ell))


def prepare(folder, counts, angular, steps, degree):
    nh, nb, nv, nf = counts
    heights = 0.002 + 99.997 * np.linspace(0, 1, nh) ** 2
    nm = angular * 2
    path_file = folder / "paths.npz"
    if path_file.exists():
        with np.load(path_file) as z:
            return (heights, z["paths"], z["mu"], z["weights"])
    nodes, weights = leggauss(angular)
    time, time_weight = leggauss(steps)
    paths = np.empty((nh, nm, steps, 2, 4), "f4")
    mus, ws = ([], [])
    air = Atmosphere()
    for p, h in enumerate(heights):
        tangent = np.sin(air.horizon(h))
        mu = np.r_[
            -1 + (nodes + 1) * 0.5 * (1 + tangent),
            tangent + (nodes + 1) * 0.5 * (1 - tangent),
        ]
        weight = np.r_[weights * 0.5 * (1 + tangent), weights * 0.5 * (1 - tangent)]
        mus.append(mu)
        ws.append(weight)
        for m, sine in enumerate(mu):
            solution, _ = trace(h, np.arcsin(sine), air, max_step=4.0)
            distance = (time + 1) * 0.5 * solution.t[-1]
            state = solution.sol(distance)
            x, z, angle = state[:3]
            psi = np.arctan2(x, z)
            paths[p, m, :, 0] = np.column_stack(
                [
                    np.hypot(x, z) - 6371,
                    np.sin(angle + psi),
                    psi,
                    time_weight * 0.5 * solution.t[-1],
                ]
            )
            paths[p, m, :, 1, :3] = state[3:].T
            paths[p, m, :, 1, 3] = 0
        print("PATHS", p, flush=True)
    np.savez(path_file, paths=paths, mu=mus, weights=ws)
    return (heights, paths, np.array(mus), np.array(ws))


def solar_textures(ctx):
    with np.load(HERE / "solar-geometry.npz") as z:
        raw, axis, heights, lower = (
            z["values"],
            z["sourceRelativeElevationDegrees"],
            z["height"],
            z["lowerExternalElevation"],
        )
    x = np.log1p(axis / 0.0001)
    target = np.linspace(0, np.log1p(180 / 0.0001), 512)
    a, b = (np.zeros((128, 512, 4), "f4"), np.zeros((128, 512, 4), "f4"))
    for i in range(128):
        exterior = lower[i] + np.deg2rad(axis)
        near = exterior >= np.pi / 2 - 1e-07
        raw[i, near, 3] = np.pi / 2
        raw[i, near, 4] = 1 / Atmosphere().index(heights[i]) ** 2
        raw[i, :, 3] -= np.minimum(exterior, np.pi / 2)
        vals = PchipInterpolator(x, raw[i], axis=0)(target)
        a[i] = vals[:, :4]
        b[i, :, 0] = vals[:, 4]
        b[i, :, 3] = lower[i]
        beyond = lower[i] + np.deg2rad(0.0001 * np.expm1(target)) >= np.pi / 2
        a[i, beyond, 3] = 0
        h = heights[i]
        a[i, beyond, :3] = [
            8 * (np.exp(-h / 8) - np.exp(-100 / 8)),
            1.2 * (np.exp(-h / 1.2) - np.exp(-100 / 1.2)),
            0,
        ]
        if h < 40:
            lo = max(h, 10)
            a[i, beyond, 2] = (
                15 - (lo - 10) ** 2 / 30 if lo <= 25 else (40 - lo) ** 2 / 30
            )
        b[i, beyond, 0] = 1 / Atmosphere().index(h) ** 2
    textures = []
    for values in [a, b]:
        t = ctx.texture((512, 128), 4, values.tobytes(), dtype="f4")
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = False
        textures.append(t)
    return textures


def build_sources(args):
    counts = tuple(args.counts)
    nh, nb, nv, nf = counts
    nm, na = (args.angular * 2, args.angular)
    folder = HERE / args.output
    folder.mkdir(exist_ok=True)
    configuration = dict(
        counts=counts,
        angular=args.angular,
        steps=args.steps,
        degree=args.degree,
        orders=args.orders,
        geometry="spherical-refractive-v1",
        refractivity=0.00029,
        scaleHeightKm=8.0,
        solarDiskSamples=32,
        splitDirect="density-normalized-v2",
        solarMapping="angular-correction-zenith-limit-v2",
    )
    manifest = json.dumps(configuration, indent=2) + "\n"
    if (
        (folder / "config.json").exists()
        and (folder / "config.json").read_text() != manifest
        and (not args.replace)
    ):
        raise ValueError("Output configuration differs; use another output folder")
    (folder / "config.json").write_text(manifest)
    heights, paths, mu, weights = prepare(
        folder, counts, args.angular, args.steps, args.degree
    )
    azimuth = np.pi * (np.arange(na) + 0.5) / na
    out_phi = np.pi * np.linspace(0, 1, nf) ** 2
    out_mu = np.linspace(-1, 1, nv)
    yout, ell = harmonics(out_mu[None, :], out_phi[:, None], args.degree)
    yout = yout.reshape(nf * nv, -1)
    yin = []
    for p in range(nh):
        y, _ = harmonics(mu[p, :, None], azimuth[None, :], args.degree)
        yin.append(
            (y * weights[p, :, None, None] * (2 * np.pi / na)).reshape(nm * na, -1).T
        )
    phase_mu, phase_weight = leggauss(512)
    g = 0.758
    phase = (
        3
        / (8 * np.pi)
        * (1 - g * g)
        / (2 + g * g)
        * (1 + phase_mu**2)
        / (1 + g * g - 2 * g * phase_mu) ** 1.5
    )
    mie_eigen = np.array(
        [
            2 * np.pi * np.sum(phase_weight * phase * eval_legendre(l, phase_mu))
            for l in ell
        ]
    )
    ray_eigen = np.where(ell == 0, 1.0, np.where(ell == 2, 0.1, 0.0))
    beta = np.array(MODEL["beta"])
    ctx = moderngl.create_standalone_context(require=430, backend="egl")
    print("GPU", ctx.info["GL_RENDERER"], flush=True)
    text = SOURCE_SHADER
    text = text.replace(
        "const float AMIN=-5.886111451;", f"const float AMIN={AMIN:.12f};"
    )
    text = text.replace(
        "const float ASPAN=11.972891;", f"const float ASPAN={ASPAN:.12f};"
    )
    text = text.replace(
        "float indexAt",
        "float asinh(float x){return sign(x)*log(abs(x)+sqrt(x*x+1.0));}\nfloat sinh(float x){return .5*(exp(x)-exp(-x));}\nfloat indexAt",
    )
    program = ctx.compute_shader(text)
    path_buffer = ctx.buffer(paths.tobytes())
    path_buffer.bind_to_storage_buffer(1)
    ray_buffer = ctx.buffer(np.zeros((nh * nm, 4), "f4").tobytes())
    ray_buffer.bind_to_storage_buffer(2)
    total = nh * nb * nv * nf
    ray_total = nh * nb * nm * na
    output = ctx.buffer(reserve=max(total, ray_total) * 16)
    output.bind_to_storage_buffer(0)
    tile = min(nb, 64)
    texture = ctx.texture3d((nv, nf * tile, nh * nb // tile), 3, dtype="f4")
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    texture.repeat_x = texture.repeat_y = texture.repeat_z = False
    initial_mie = ctx.texture3d((nv, nf * tile, nh * nb // tile), 3, dtype="f4")
    initial_mie.filter = (moderngl.LINEAR, moderngl.LINEAR)
    initial_mie.repeat_x = initial_mie.repeat_y = initial_mie.repeat_z = False
    initial_mie.use(3)
    solar = solar_textures(ctx)
    for i, t in enumerate(solar):
        t.use(i)
    texture.use(2)
    for key, value in dict(
        solar0=0,
        solar1=1,
        sourceTexture=2,
        initialMie=3,
        counts=counts,
        pathCount=args.steps,
        muCount=nm,
        azimuthCount=na,
    ).items():
        if key in program:
            program[key] = value

    def dispatch(mode, size):
        program["mode"] = mode
        for first in range(0, size, 1024):
            count = min(1024, size - first)
            program["first"], program["count"] = (first, count)
            program.run(group_x=(count + 63) // 64)
            ctx.memory_barrier()
            ctx.finish()
        return (
            np.frombuffer(output.read(size=size * 16), "f4")
            .reshape(size, 4)[:, :3]
            .copy()
        )

    for batch in args.batches:
        dest = folder / f"batch-{batch:02d}.npz"
        if dest.exists() and (not args.replace):
            print("SKIP", batch, flush=True)
            continue
        start = perf_counter()
        coefficients = beta[batch * 3 : batch * 3 + 3]
        for key, value in zip(["betaR", "betaM", "betaO"], coefficients.T):
            program[key] = tuple(value)
        program["directSpecies"] = 0
        directR = dispatch(0, total).reshape(nh, nb, nf, nv, 3)
        program["directSpecies"] = 1
        directM = dispatch(0, total).reshape(nh, nb, nf, nv, 3)
        program["initialOrder"] = True
        texture.write(np.ascontiguousarray(directR).tobytes())
        initial_mie.write(np.ascontiguousarray(directM).tobytes())
        radiance = dispatch(1, ray_total).reshape(nh, nb, nm * na, 3)
        accumulated = np.zeros_like(directR)
        irradiance = np.zeros((nh, nb, 3))

        def add_irradiance():
            for p in range(nh):
                flux_weights = np.repeat(
                    weights[p] * np.maximum(mu[p], 0) * (2 * np.pi / na), na
                )
                irradiance[p] += (
                    np.einsum("bqc,q->bc", radiance[p], flux_weights)
                    * Atmosphere().index(heights[p]) ** 2
                )

        add_irradiance()
        program["initialOrder"] = False
        energies = [float(radiance.max())]
        for order in range(2, args.orders + 1):
            density = np.empty_like(directR)
            for p, height in enumerate(heights):
                moments = np.matmul(yin[p], radiance[p])
                factor = (
                    ray_eigen[:, None] * coefficients[:, 0]
                    + np.exp(-height / 1.2 + height / 8)
                    * mie_eigen[:, None]
                    * coefficients[:, 1]
                )
                scattered = np.matmul(yout, moments * factor)
                density[p] = np.maximum(scattered.reshape(nb, nf, nv, 3), 0)
            accumulated += density
            texture.write(np.ascontiguousarray(density).tobytes())
            radiance = dispatch(1, ray_total).reshape(nh, nb, nm * na, 3)
            add_irradiance()
            energies.append(float(radiance.max()))
            print(
                "ORDER",
                batch,
                order,
                "peak",
                energies[-1],
                "seconds",
                round(perf_counter() - start, 2),
                flush=True,
            )
        assert np.isfinite(accumulated).all()
        pending = dest.with_suffix(".pending.npz")
        np.savez_compressed(
            pending,
            directR=directR,
            directM=directM,
            multiple=accumulated,
            beta=coefficients,
            peaks=energies,
            irradiance=irradiance,
        )
        pending.replace(dest)
        print("BATCH", batch, "seconds", round(perf_counter() - start, 2), flush=True)
    ctx.release()


def physical_path(h, elevation):
    h = np.clip(h, 0.002, 99.999)
    return trace(h, apparent_view(h, elevation), max_step=4.0)


def path_samples(h, view, distances, steps=32):
    """Gauss integration separately on each cumulative interval.

    Display distance is a parameter, not optical length: fractions of the
    display ray's terminal distance map to fractions of the physical ray.
    The two sea-level endpoints can differ horizontally near the limb; this is
    the explicit fixed-map presentation approximation, not a fitted fake ray.
    """
    h = np.clip(h, 0.002, 99.999)
    solution, ground = physical_path(h, view)
    terminal = float(ray_ends(h, view)[1])
    ends = np.clip(np.asarray(distances) / max(terminal, 1e-30), 0, 1) * solution.t[-1]
    nodes, weights = leggauss(steps)
    previous = np.r_[0, ends[:-1]]
    lengths = ends - previous
    samples = previous[:, None] + (nodes[None] + 1) * lengths[:, None] * 0.5
    state = solution.sol(samples.ravel())
    x, z, angle = state[:3]
    psi = np.arctan2(x, z)
    packed = np.zeros((len(ends), steps, 2, 4), "f4")
    packed[:, :, 0] = np.column_stack(
        [
            np.hypot(x, z) - 6371,
            np.sin(angle + psi),
            psi,
            (lengths[:, None] * weights[None] * 0.5).ravel(),
        ]
    ).reshape(len(ends), steps, 4)
    packed[:, :, 1, :3] = state[3:].T.reshape(len(ends), steps, 3)
    return (packed, solution)


BODY = r"""
layout(std430,binding=0) readonly buffer Query { vec4 queries[]; };
layout(std430,binding=1) readonly buffer Path { vec4 path[]; };
layout(std430,binding=2) writeonly buffer Result { vec4 result[]; };
uniform int queryCount,sampleCount,steps,first;
uniform sampler3D mieTexture;
void main(){
 int id=first+int(gl_GlobalInvocationID.x);if(id>=queryCount)return;
 vec4 q=queries[id];
 int ray=int(q.w);
 vec3 sun=vec3(cos(q.y)*cos(q.z),cos(q.y)*sin(q.z),sin(q.y));
 vec3 R=vec3(0),M=vec3(0);
 for(int node=0;node<sampleCount;node++){
  for(int j=0;j<steps;j++){
   int address=((ray*sampleCount+node)*steps+j)*2;
   vec4 s=path[address],col=path[address+1];
   float mus=sin(s.z)*sun.x+cos(s.z)*sun.z;
   float tx=cos(s.z)*sun.x-sin(s.z)*sun.z;
   float cp=tx/max(length(vec2(tx,sun.y)),1e-20);
   vec3 weight=exp(-(col.x*betaR+col.y*betaM+col.z*betaO)-s.x/8.0)*s.w;
   R+=weight*sourceAt(s.x,s.y,mus,cp);
   M+=weight*exp(-s.x/1.2+s.x/8.0)*sampleSource(mieTexture,s.x,s.y,mus,cp);
  }
  float reception=indexAt(q.x);reception*=reception;
  int outputIndex=(id*sampleCount+node)*2;
  result[outputIndex]=vec4(R*reception,0);
  result[outputIndex+1]=vec4(M*reception,0);
 }
}
"""


class Integrator:
    def __init__(self, folder="sources-v2"):
        self.folder = HERE / folder
        self.config = json.loads((self.folder / "config.json").read_text())
        self.counts = tuple(self.config["counts"])
        self.context = moderngl.create_standalone_context(require=430, backend="egl")
        template = SOURCE_SHADER
        helpers = template[
            template.index("const float PI=") : template.index("vec3 directSource")
        ]
        helpers = helpers.replace(
            "const float AMIN=-5.886111451;", f"const float AMIN={AMIN:.12f};"
        )
        helpers = helpers.replace(
            "const float ASPAN=11.972891;", f"const float ASPAN={ASPAN:.12f};"
        )
        header = "#version 430\nlayout(local_size_x=64) in;\nuniform sampler2D solar0,solar1;\nuniform sampler3D sourceTexture;\nuniform ivec4 counts;\nuniform vec3 betaR,betaM,betaO;\nfloat asinh(float x){return sign(x)*log(abs(x)+sqrt(x*x+1.0));}\nfloat sinh(float x){return .5*(exp(x)-exp(-x));}\nuniform sampler3D mieTexture;\n"
        self.program = self.context.compute_shader(
            header + helpers + BODY.replace("uniform sampler3D mieTexture;", "")
        )
        self.solar = solar_textures(self.context)
        for i, t in enumerate(self.solar):
            t.use(i)
        nh, nb, nv, nf = self.counts
        tile = min(nb, 64)
        self.textures = [
            self.context.texture3d((nv, nf * tile, nh * nb // tile), 3, dtype="f4")
            for _ in range(2)
        ]
        for i, t in enumerate(self.textures):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = t.repeat_z = False
            t.use(i + 2)
        for key, value in dict(
            solar1=1, sourceTexture=2, mieTexture=3, counts=self.counts
        ).items():
            self.program[key] = value

    def evaluate(
        self,
        queries,
        paths,
        batches=range(16),
        direct_only=False,
        weights=None,
        dispatch=512,
    ):
        queries = np.asarray(queries, "f4")
        paths = np.asarray(paths, "f4")
        nr, ns, steps = paths.shape[:3]
        output = np.zeros((len(queries), ns, 2, 48 if weights is None else 3), "f4")
        buffers = [
            self.context.buffer(queries.tobytes()),
            self.context.buffer(paths.tobytes()),
            self.context.buffer(reserve=len(queries) * ns * 2 * 16),
        ]
        try:
            for i, b in enumerate(buffers):
                b.bind_to_storage_buffer(i)
            for key, value in dict(
                queryCount=len(queries), sampleCount=ns, steps=steps
            ).items():
                self.program[key] = value
            for batch in batches:
                with np.load(self.folder / f"batch-{batch:02d}.npz") as z:
                    a = z["directR"] if direct_only else z["directR"] + z["multiple"]
                    self.textures[0].write(np.ascontiguousarray(a).tobytes())
                    self.textures[1].write(np.ascontiguousarray(z["directM"]).tobytes())
                    for key, value in zip(["betaR", "betaM", "betaO"], z["beta"].T):
                        self.program[key] = tuple(value)
                for first in range(0, len(queries), dispatch):
                    self.program["first"] = first
                    self.program.run(
                        group_x=(min(dispatch, len(queries) - first) + 63) // 64
                    )
                    self.context.memory_barrier()
                    self.context.finish()
                spectral = np.frombuffer(buffers[2].read(), "f4").reshape(
                    len(queries), ns, 2, 4
                )[..., :3]
                if weights is None:
                    output[..., batch * 3 : batch * 3 + 3] = spectral
                else:
                    output += spectral @ weights[batch * 3 : batch * 3 + 3]
            return output
        finally:
            for b in buffers:
                b.release()

    def close(self):
        self.context.release()


def geometry(p):
    horizon = float(to_grid(np.array([[1.0, 0.0, -dip(1.0), 0.0]]))[0, 1] * 63)
    extras = np.r_[
        24 + np.array([0.2, 0.4, 0.6, 0.8, 0.95, 0.99]) * (horizon - 24),
        horizon + np.array([0.01, 0.05, 0.15, 0.35, 0.6, 0.85]) * (25 - horizon),
    ]
    nu = np.r_[np.arange(64), extras]
    q = np.zeros((len(nu), 8, 64, 4))
    q[..., 0] = p / 15
    q[..., 1] = nu[:, None, None] / 63
    q[..., 2] = np.arange(8)[None, :, None] / 7
    q[..., 3] = np.arange(64)[None, None, :] / 63
    physical = from_grid(q)
    h0 = physical[0, 0, 0, 0]
    h = max(0.002, min(99.999, h0))
    angle = lambda axis: np.deg2rad(
        0.5 * np.sinh(ANGLE_MIN + ANGLE_SPAN * unwarp(q[..., axis], axis))
    )
    physical[..., 1] = np.clip(angle(3) - dip(h), -np.pi / 2, np.pi / 2)
    physical[..., 2] = np.clip(angle(1) - dip(h), -np.pi / 2, np.pi / 2)
    physical[..., 0] = h
    geometry = [sample_ray(h, v) for v in physical[:, 0, 0, 2]]
    d = np.stack([np.r_[g[0], g[2]] for g in geometry])
    u = np.stack([g[1] for g in geometry])
    return (physical, d, u, nu)


def density(height):
    height = np.maximum(height, 0)
    return np.stack(
        [
            np.exp(-height / 8),
            np.exp(-height / 1.2),
            np.maximum(0, 1 - np.abs(height - 25) / 15),
        ],
        -1,
    )


def straight_geometry(h, view, t):
    x = t * np.cos(view)
    z = R + h + t * np.sin(view)
    psi = np.arctan2(x, z)
    return (np.hypot(x, z) - R, np.sin(view + psi), psi)


def straight_samples(h, view, distances, steps=32):
    """Same packing as reference.path_samples, on the straight display ray."""
    h = float(np.clip(h, 0.002, 99.999))
    ends = np.asarray(distances, "f8")
    nodes, weights = leggauss(steps)
    previous = np.r_[0, ends[:-1]]
    lengths = ends - previous
    samples = previous[:, None] + (nodes[None] + 1) * lengths[:, None] * 0.5
    flat = np.sort(samples.ravel())
    sub = np.r_[0, flat]
    inner, inner_w = leggauss(16)
    a, b = (sub[:-1], sub[1:])
    points = a[:, None] + (inner[None] + 1) * (b - a)[:, None] * 0.5
    height, _, _ = straight_geometry(h, view, points)
    pieces = (
        density(height) * (inner_w[None, :, None] * (b - a)[:, None, None] * 0.5)
    ).sum(1)
    cumulative = np.cumsum(pieces, axis=0)
    order = np.argsort(samples.ravel(), kind="stable")
    columns = np.empty((samples.size, 3))
    columns[order] = cumulative
    height, mu, psi = straight_geometry(h, view, samples.ravel())
    packed = np.zeros((len(ends), steps, 2, 4), "f4")
    packed[:, :, 0] = np.column_stack(
        [height, mu, psi, (lengths[:, None] * weights[None] * 0.5).ravel()]
    ).reshape(len(ends), steps, 4)
    packed[:, :, 1, :3] = columns.reshape(len(ends), steps, 3)
    return packed


def phase_normalize(phys, xyz):
    mu = np.sin(phys[..., 2])
    ms = np.sin(phys[..., 1])
    cosine = mu * ms + np.cos(phys[..., 2]) * np.cos(phys[..., 1]) * np.cos(
        phys[..., 3]
    )
    pr = 3 / (16 * np.pi) * (1 + cosine * cosine)
    g = 0.758
    pm = (
        3
        / (8 * np.pi)
        * (1 - g * g)
        / (2 + g * g)
        * (1 + cosine * cosine)
        / (1 + g * g - 2 * g * cosine) ** 1.5
    )
    return xyz / np.stack([pr, pm], -1)[..., None, :, None]


def build_straight_references():
    args = SimpleNamespace(
        heights=list(range(16)), curved="reference-curved", output="reference", steps=32
    )
    out = HERE / args.output
    out.mkdir(exist_ok=True)
    weights = spectral_weights()
    gpu = Integrator()
    configuration = json.dumps(gpu.config, sort_keys=True)
    try:
        for p in args.heights:
            dest = out / f"p{p:02d}.npz"
            source = HERE / args.curved / f"p{p:02d}.npz"
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if dest.exists():
                with np.load(dest) as previous:
                    if (
                        str(previous.get("curvedReference", "")) == digest
                        and int(previous["steps"]) == args.steps
                    ):
                        print("SKIP", p, flush=True)
                        continue
            start = perf_counter()
            with np.load(source) as z:
                curved = {k: z[k] for k in z.files}
            if str(curved["sourceConfig"]) != configuration:
                raise ValueError(f"Curved reference {p} uses different sources")
            phys, d = (curved["physical"], curved["distances"])
            paths = np.stack(
                [
                    straight_samples(
                        phys[row, 0, 0, 0], phys[row, 0, 0, 2], d[row, :42], args.steps
                    )
                    for row in range(len(phys))
                ]
            )
            query = np.stack(
                [
                    phys[..., 0],
                    phys[..., 1],
                    phys[..., 3],
                    np.broadcast_to(
                        np.arange(len(phys))[:, None, None], phys.shape[:-1]
                    ),
                ],
                axis=-1,
            )
            finite = gpu.evaluate(query.reshape(-1, 4), paths, weights=weights).reshape(
                *phys.shape[:-1], 42, 2, 3
            )
            xyz = curved["xyz"].copy()
            xyz[..., :42, :, :] = phase_normalize(phys, finite)
            assert np.isfinite(xyz).all() and np.all(xyz >= 0)
            pending = dest.with_suffix(".pending.npz")
            np.savez(
                pending,
                physical=phys,
                distances=d,
                u=curved["u"],
                nu=curved["nu"],
                xyz=xyz,
                steps=args.steps,
                sourceConfig=curved["sourceConfig"],
                curvedReference=digest,
                finitePath="straight display ray; sky sample copied from the curved reference",
            )
            pending.replace(dest)
            print(
                "REFERENCE", p, "seconds", round(perf_counter() - start, 2), flush=True
            )
    finally:
        gpu.close()


def basis(x, n):
    knots = np.r_[[0.0] * 4, np.linspace(0, 1, n - 2)[1:-1], [1.0] * 4]
    return sparse.csr_matrix(BSpline.design_matrix(np.clip(x, 0, 1), knots, 3))


def design(x, y, counts):
    nx, ny = counts
    a, b = (basis(x, nx), basis(y, ny))
    n = len(x)
    rows = np.repeat(np.arange(n), 16)
    ai = a.indices.reshape(n, 4)
    bi = b.indices.reshape(n, 4)
    cols = (bi[:, :, None] * nx + ai[:, None, :]).ravel()
    values = (
        b.data.reshape(n, 4)[:, :, None] * a.data.reshape(n, 4)[:, None, :]
    ).ravel()
    return sparse.csr_matrix((values, (rows, cols)), shape=(n, nx * ny))


def build_palette():
    args = SimpleNamespace(controls=[24, 8], penalty=1e-05, output="palette.rgba16f")
    nx, ny = args.controls
    color = META["palette"]["color"]
    blue = META["palette"]["thetaBlue"]
    rng = np.random.default_rng(1411)
    data = [[], []]
    for p in range(16):
        with np.load(HERE / "reference" / f"p{p:02d}.npz") as z:
            xyz = z["xyz"].reshape(-1, 2, 3)
        for sp in range(2):
            q = xyz[:, sp]
            r = np.hypot(q[:, 0], q[:, 2])
            eligible = np.flatnonzero(r > max(r.max() * 1e-07, 1e-20))
            index = rng.choice(eligible, min(50000, len(eligible)), replace=False)
            data[sp].append(q[index])
    pixels = np.ones((2, 64, 256, 4), "f4")
    for sp in range(2):
        xyz = np.concatenate(data[sp])
        r = np.hypot(xyz[:, 0], xyz[:, 2])
        theta = blue - np.arctan2(xyz[:, 2], xyz[:, 0])
        scale = np.array(color["uvScale"][sp])
        bias = np.array(color["uvBias"][sp])
        x = (theta * scale[0] + bias[0] - 0.5 / 256) / (255 / 256)
        y = (
            np.log2(np.maximum(r, color["minRadius"][sp])) * scale[1]
            + bias[1]
            - 0.5 / 64
        ) / (63 / 64)
        matrix = design(x, y, args.controls)
        target = xyz[:, 1] / r
        dx = np.diff(np.eye(nx), 2, axis=0)
        dy = np.diff(np.eye(ny), 2, axis=0)
        penalty = sparse.kron(sparse.eye(ny), dx.T @ dx) + sparse.kron(
            dy.T @ dy, sparse.eye(nx)
        )
        count = len(x)
        coefficient = spsolve(
            matrix.T @ matrix / count
            + args.penalty * penalty
            + sparse.eye(nx * ny) * 1e-08,
            matrix.T @ target / count,
        )
        xx, yy = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 64))
        pred = (design(xx.ravel(), yy.ravel(), args.controls) @ coefficient).reshape(
            64, 256
        )
        angle = ((np.arange(256) + 0.5) / 256 - bias[0]) / scale[0]
        xzangle = blue - angle
        decoded = np.stack(
            [
                np.broadcast_to(np.cos(xzangle), (64, 256)),
                np.maximum(pred, 0),
                np.broadcast_to(np.sin(xzangle), (64, 256)),
            ],
            axis=-1,
        )
        pixels[sp, :, :, :3] = decoded @ RGB.T
        error = np.abs(matrix @ coefficient - target)
        print(
            "PALETTE",
            sp,
            "Y/r absolute p50,p95,p99,max",
            np.quantile(error, [0.5, 0.95, 0.99, 1]),
            flush=True,
        )
    assert np.isfinite(pixels.astype("f2")).all()
    (HERE / args.output).write_bytes(pixels.astype("<f2").tobytes())


def build_irradiance():
    meta = META
    axis = np.array(MODEL["irradianceAxis"], "f4")
    config = json.loads((HERE / "sources-v2/config.json").read_text())
    nh, nb, *_ = config["counts"]
    weights = spectral_weights()
    rgb_weights = weights @ RGB.T
    solar = rgb_weights.sum(axis=0) * meta["observer"]["radianceScale"]
    np.save(
        HERE / "solar-radiance.npy",
        solar / (np.pi * np.sin(meta["solarAngularRadius"]) ** 2),
    )
    sky = np.zeros((nh, nb, 3))
    for batch in range(16):
        with np.load(HERE / "sources-v2" / f"batch-{batch:02d}.npz") as z:
            sky += z["irradiance"] @ weights[batch * 3 : batch * 3 + 3]
    lookup = SolarGeometry()
    sampler = RegularGridInterpolator(
        (np.linspace(0, 1, nh), np.linspace(0, 1, nb)), sky
    )
    output = np.zeros((meta["irradianceHeightCount"], len(axis), 7), "f4")
    beta = np.array(MODEL["beta"])
    transmission = meta["observer"]["transmission"]
    domain = np.array(transmission["domain"])
    scale = np.array(transmission["columnScale"])
    column = scale[:, None] * np.expm1(
        np.linspace(0, 1, 256)[None] * np.log1p(domain / scale)[:, None]
    )
    spectrum = np.exp(
        -column[0, :, None, None] * beta[None, None, :, 0]
        - column[1, None, :, None] * beta[None, None, :, 2]
    )
    material = np.ones((256, 256, 4), "f4")
    material[:, :, :3] = spectrum @ rgb_weights / rgb_weights.sum(axis=0)
    material[0, 0] = 1
    assert np.isfinite(material).all()
    (HERE / "transmittance.rgba32f").write_bytes(material.astype("<f4").tobytes())
    for p in range(len(output)):
        h = 100 * (p / (len(output) - 1)) ** 2
        exterior = np.clip(np.deg2rad(axis) - dip(h), -np.pi / 2, np.pi / 2)
        lower = np.interp(np.clip(h, 0.002, 99.999), lookup.height, lookup.lower)
        source_beta = np.clip(
            (np.arcsinh(np.rad2deg(exterior - lower) / 0.5) - AMIN) / ASPAN, 0, 1
        )
        source_p = np.sqrt(np.clip((h - 0.002) / 99.997, 0, 1))
        sky_xyz = sampler(np.column_stack([np.full(len(axis), source_p), source_beta]))
        output[p, :, 0] = axis
        output[p, :, 1:4] = sky_xyz @ RGB.T * meta["observer"]["radianceScale"] / np.pi
        for b, elevation in enumerate(exterior):
            center = np.array([np.cos(elevation), 0, np.sin(elevation)])
            directions, dw = disk_directions(center)
            geom, visible = lookup.evaluate(
                np.full(len(dw), h), np.arcsin(np.clip(directions[:, 2], -1, 1))
            )
            transmission = np.exp(-(geom[:, :3] @ beta.T))
            irradiance = (
                np.sum(transmission * (dw * visible * geom[:, 4])[:, None], axis=0)
                * Atmosphere().index(h) ** 2
            )
            output[p, b, 4:] = (
                irradiance @ weights @ RGB.T * meta["observer"]["radianceScale"] / np.pi
            )
        print("IRRADIANCE", p, flush=True)
    assert np.isfinite(output).all()
    np.save(HERE / "irradiance.npy", output)


SOURCE_SHADER = r"""#version 430
layout(local_size_x=64) in;
layout(std430,binding=0) writeonly buffer Output { vec4 outputValues[]; };
layout(std430,binding=1) readonly buffer Paths { vec4 paths[]; };
layout(std430,binding=2) readonly buffer RayInfo { vec4 rays[]; };
uniform sampler2D solar0,solar1;
uniform sampler3D sourceTexture,initialMie;
uniform ivec4 counts; // height, sun, view, phi
uniform int mode,first,count,pathCount,muCount,azimuthCount;
uniform int directSpecies;
uniform bool initialOrder;
uniform vec3 betaR,betaM,betaO;
const float PI=3.141592653589793;
const float AMIN=-5.886111451;
const float ASPAN=11.972891;
float indexAt(float h){return 1.0+.00029*(exp(-h/8.0)-exp(-12.5));}
float lowerSun(float h){
  float y=sqrt(clamp((h-.002)/99.997,0.0,1.0));
  return texture(solar1,vec2(.5/512.0,(y*127.0+.5)/128.0)).w;
}
float sunAt(float h,float b){
  return clamp(lowerSun(h)+radians(.5*sinh(AMIN+ASPAN*b)),-PI*.5,PI*.5);
}
float sunCoordinate(float h,float mu){
  float angle=degrees(asin(clamp(mu,-1.0,1.0))-lowerSun(h))/.5;
  return clamp((asinh(angle)-AMIN)/ASPAN,0.0,1.0);
}
vec3 sourceNode(sampler3D field,vec2 xy,float p,float b){
  float tile=float(min(counts.y,64)),groups=float(counts.y)/tile;
  return texture(field,vec3(xy.x,(mod(b,tile)+xy.y)/tile,
    (p*groups+floor(b/tile)+.5)/(float(counts.x)*groups))).rgb;
}
vec3 sampleSource(sampler3D field,float h,float mu,float mus,float cosPhi){
  float p=sqrt(clamp((h-.002)/99.997,0.0,1.0))*float(counts.x-1);
  float b=sunCoordinate(h,mus)*float(counts.y-1);
  float f=sqrt(acos(clamp(cosPhi,-1.0,1.0))/PI);
  vec2 xy=(vec2((mu*.5+.5)*float(counts.z-1),f*float(counts.w-1))+.5)/vec2(counts.z,counts.w);
  float p0=min(floor(p),float(counts.x-2)),b0=min(floor(b),float(counts.y-2));
  vec3 a=sourceNode(field,xy,p0,b0),c=sourceNode(field,xy,p0,b0+1.0);
  vec3 d=sourceNode(field,xy,p0+1.0,b0),e=sourceNode(field,xy,p0+1.0,b0+1.0);
  return mix(mix(a,c,b-b0),mix(d,e,b-b0),p-p0);
}
vec3 sourceAt(float h,float mu,float mus,float cosPhi){
  return sampleSource(sourceTexture,h,mu,mus,cosPhi);
}
vec3 directSource(float h,float mu,float sunElevation,float phi){
  vec3 sun=vec3(cos(sunElevation),0,sin(sunElevation));
  vec3 view=vec3(sqrt(max(1.0-mu*mu,0.0))*cos(phi),
                 sqrt(max(1.0-mu*mu,0.0))*sin(phi),mu);
  vec3 tangent=vec3(-sin(sunElevation),0,cos(sunElevation));
  vec3 sum=vec3(0);
  float lower=lowerSun(h),heightUV=(sqrt(clamp((h-.002)/99.997,0.0,1.0))*127.0+.5)/128.0;
  for(int i=0;i<32;i++){
    // Equal-area disk quadrature with paired rings, no empirical solar lift.
    float az=(float(i%16)+.5)*(2.0*PI/16.0);
    float radial=sqrt((float(i/16)+.5)/2.0)*radians(.2666667);
    vec3 disk=cos(radial)*sun+sin(radial)*(cos(az)*tangent+sin(az)*vec3(0,1,0));
    float relative=degrees(asin(clamp(disk.z,-1.0,1.0))-lower);
    if(relative<0.0)continue;
    float u=log(1.0+relative/.0001)/log(1.0+180.0/.0001);
    vec2 uv=vec2((u*511.0+.5)/512.0,heightUV);
    vec4 geometry=texture(solar0,uv),extra=texture(solar1,uv);
    vec2 horizontal=normalize(disk.xy);
    float apparentElevation=asin(clamp(disk.z,-1.0,1.0))+geometry.w;
    vec3 apparent=vec3(cos(apparentElevation)*horizontal,sin(apparentElevation));
    float nu=clamp(dot(view,apparent),-1.0,1.0);
    float pr=3.0/(16.0*PI)*(1.0+nu*nu),g=.758;
    float pm=3.0/(8.0*PI)*(1.0-g*g)/(2.0+g*g)*(1.0+nu*nu)/pow(1.0+g*g-2.0*g*nu,1.5);
    vec3 T=exp(-(geometry.x*betaR+geometry.y*betaM+geometry.z*betaO));
    // Sources store radiance/n² and are divided by Rayleigh density.
    vec3 scattering=directSpecies==0?betaR*pr:
      directSpecies==1?betaM*pm:
      betaR*pr+exp(-h/1.2+h/8.0)*betaM*pm;
    sum+=T*extra.x*scattering/32.0;
  }
  return sum;
}
void main(){
  int idx=first+int(gl_GlobalInvocationID.x);if(idx>=first+count)return;
  if(mode==0){
    int v=idx%counts.z,f=(idx/counts.z)%counts.w,b=(idx/(counts.z*counts.w))%counts.y;
    int p=idx/(counts.z*counts.w*counts.y);
    float h=.002+99.997*pow(float(p)/float(counts.x-1),2.0);
    float mu=-1.0+2.0*float(v)/float(counts.z-1),phi=PI*pow(float(f)/float(counts.w-1),2.0);
    outputValues[idx]=vec4(directSource(h,mu,sunAt(h,float(b)/float(counts.y-1)),phi),0);
  }else{
    int a=idx%azimuthCount,m=(idx/azimuthCount)%muCount;
    int b=(idx/(muCount*azimuthCount))%counts.y,p=idx/(muCount*azimuthCount*counts.y);
    vec4 ray=rays[p*muCount+m];
    float h0=.002+99.997*pow(float(p)/float(counts.x-1),2.0);
    float es=sunAt(h0,float(b)/float(counts.y-1));
    float phi=PI*(float(a)+.5)/float(azimuthCount);
    vec3 sun=vec3(cos(es)*cos(phi),cos(es)*sin(phi),sin(es));
    vec3 sum=vec3(0);
    for(int j=0;j<pathCount;j++){
      int address=((p*muCount+m)*pathCount+j)*2;
      vec4 s=paths[address],col=paths[address+1];
      // s=(height,mu,polar displacement,quadrature weight), col=columns.
      float mus=sin(s.z)*sun.x+cos(s.z)*sun.z;
      float tx=cos(s.z)*sun.x-sin(s.z)*sun.z;
      float cp=tx/max(length(vec2(tx,sun.y)),1e-20);
      vec3 density=sourceAt(s.x,s.y,mus,cp)*exp(-s.x/8.0);
      if(initialOrder)density+=sampleSource(initialMie,s.x,s.y,mus,cp)*exp(-s.x/1.2);
      sum+=exp(-(col.x*betaR+col.y*betaM+col.z*betaO))*density*s.w;
    }
    outputValues[idx]=vec4(sum,0);
  }
}
"""
