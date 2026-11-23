"""Low-rank C2 tensor B-spline of finite-path scattering, fitted to physical references.

Per species s in (Rayleigh, Mie) and field in (L = log(r / species column), theta):

    C[(p, beta), (nu, s, phi)] = m + sum_k A[(p, beta), k] B[k, (nu, s, phi)]

C are the controls of a cubic tensor B-spline in a smooth chart of the physical inputs:

    p    = horizon(h) normalized over the training heights
    beta = asinh((sun elevation + dip) / 0.5 deg), nu = same for the view (angle_fraction)
    s    = asinh(u / eps) / asinh(1 / eps), u = column / endColumns,
           eps = 0.001 * 10^(0.5 + 0.5 tanh((relative elevation - 1.5 deg) / 0.75 deg))
    phi  = phi / pi, with zero end derivatives

Knot positions only carry the density of the former piecewise warps, so the model is C2
in the physical inputs. Data: all nodal references (production targets and weights) plus
off-node random rays; sky samples above any terrain (8.849 km) are down-weighted.
"""

from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline, make_interp_spline
from scipy.sparse.linalg import LinearOperator, lsmr

import lut_fit as F
from lut_model import (
    HERE,
    BLUE,
    horizon,
    dip,
    angle_fraction,
    end_columns,
    ray_ends,
    sample_ray,
    from_grid,
    unwarp,
    polar,
    _columns,
)
from lut_palette import Palette

SHAPE = (16, 64, 32, 16, 8)  # p, beta, nu, s, phi controls
NS, NC = 16 * 64, 32 * 16 * 8
RANKS = [[128, 128], [256, 128]]  # [species][L, theta]
HEIGHT_RANGE = (0.002, 99.999)
EPSILON = dict(minimum=0.001, decades=1.0, center=1.5, width=0.75)
EVEREST = 8.849  # highest terrain; sky samples above it cannot be occluders
EXCLUDED = 0.1  # weight factor for those samples
EXTRA_MASS = 0.25  # off-node rows: 25% of the grid's weighted mass
PENALTY = [1e-5, 1e-5, 1e-3, 1e-3, 1e-3]  # second-difference factors per axis
EXTRA_RAYS, EXTRA_SEED = 8192, 20261127  # per group; four groups
# Mie radius only: extra rays through the lunar/solar shadow onset (light a few
# degrees below the geometric horizon), where log(r) drops by tens within one beta
# interval; the nodal beta lattice alone lets the cubic overshoot between nodes.
SHADOW_RAYS, SHADOW_SEED = 32768, 20261122  # per group; two groups
INIT_ITERATIONS, B_ITERATIONS, A_ITERATIONS = 2000, 600, 400
FOLDER = "pca"
RADIUS_FIT = f"censored-v1-{F.LOG_RADIUS_FLOOR:g}-{F.SHADOW_WEIGHT:g}"
MIE_RADIUS_FIT = f"{RADIUS_FIT}-radiance-{F.RADIANCE_FLOOR:g}-shadow-{SHADOW_RAYS}-{SHADOW_SEED}"


# ---------------------------------------------------------------- chart and knots
def epsilon(relative_degrees):
    e = EPSILON
    x = (np.asarray(relative_degrees) - e["center"]) / e["width"]
    return e["minimum"] * 10 ** (e["decades"] * (0.5 + 0.5 * np.tanh(x)))


def chart_axes(physical):
    """physical [..., (h, sun, view, phi)] -> (p, beta, nu, phi)."""
    h, sun, view, phi = np.moveaxis(np.asarray(physical, "f8"), -1, 0)
    lo, hi = horizon(HEIGHT_RANGE[0]), horizon(HEIGHT_RANGE[1])
    return (
        (horizon(h) - lo) / (hi - lo),
        angle_fraction(sun, h),
        angle_fraction(view, h),
        phi / np.pi,
    )


def chart_distance(u, h, view):
    e = epsilon(np.rad2deg(np.asarray(view) + dip(np.asarray(h))))
    return np.arcsinh(np.asarray(u) / e) / np.arcsinh(1 / e)


def knots():
    """Clamped cubic knot vectors (p, beta, nu, s, phi)."""
    spec = F.Spec(F.CONFIG)
    heights = [
        float(np.load(HERE / "reference" / f"p{p:02d}.npz")["physical"][0, 0, 0, 0])
        for p in range(16)
    ]
    p = chart_axes(np.array([[h, 0, 0, 0] for h in heights]))[0]
    return dict(
        p=make_interp_spline(p, np.zeros(16), k=3).t,
        beta=make_interp_spline(unwarp(np.arange(64) / 63, 3), np.zeros(64), k=3).t,
        nu=unwarp(spec.tn, 1),
        s=spec.ts,
        phi=unwarp(spec.tf, 2),
    )


def basis(x, t):
    shape = np.shape(x)
    b = BSpline.design_matrix(np.clip(np.ravel(x), 0, 1 - 1e-12), t, 3).toarray()
    return b.reshape(*shape, -1)


def flat_ends(n):
    """Merge first/last basis pairs: zero end derivative."""
    m = np.zeros((n, n - 2))
    m[0, 0] = m[1, 0] = 1
    for i in range(2, n - 2):
        m[i, i - 1] = 1
    m[n - 2, n - 3] = m[n - 1, n - 3] = 1
    return m


# ---------------------------------------------------------------- operators
class Tensor:
    """Matrix-free tensor B-spline on the nodal grid (h, nu, beta, d, phi observations)."""

    def __init__(self, p, beta, nu, distance, phi):
        self.p, self.beta, self.nu, self.distance, self.phi = p, beta, nu, distance, phi
        self.controls = (p.shape[1], beta.shape[2], nu.shape[2], distance.shape[3], phi.shape[1])
        self.observations = (p.shape[0], nu.shape[1], beta.shape[1], distance.shape[2], phi.shape[0])
        self.shape = (int(np.prod(self.observations)), int(np.prod(self.controls)))

    def forward(self, x):
        c = x.reshape(self.controls)
        out = np.empty(self.observations)
        v, b, d, f = self.observations[1:]
        s = self.controls[3]
        for h in range(self.observations[0]):
            block = np.tensordot(self.p[h], c, axes=(0, 0))
            block = np.einsum("jb,bnsf->jnsf", self.beta[h], block, optimize=True)
            block = np.einsum("vn,bnsf->vbsf", self.nu[h], block, optimize=True)
            block = np.einsum("kf,vbsf->vbsk", self.phi, block, optimize=True)
            value = self.distance[h] @ block.transpose(0, 2, 1, 3).reshape(v, s, b * f)
            out[h] = value.reshape(v, d, b, f).transpose(0, 2, 1, 3)
        return out.ravel()

    def backward(self, y, squared=False):
        data = y.reshape(self.observations)
        out = np.zeros(self.controls)
        v, b, d, f = self.observations[1:]
        s = self.controls[3]
        p, beta, nu, distance, phi = self.p, self.beta, self.nu, self.distance, self.phi
        if squared:
            p, beta, nu, distance, phi = (x * x for x in (p, beta, nu, distance, phi))
        for h in range(self.observations[0]):
            packed = data[h].transpose(0, 2, 1, 3).reshape(v, d, b * f)
            block = (distance[h].transpose(0, 2, 1) @ packed).reshape(v, s, b, f).transpose(0, 2, 1, 3)
            block = np.einsum("kf,vbsk->vbsf", phi, block, optimize=True)
            block = np.einsum("vn,vbsf->bnsf", nu[h], block, optimize=True)
            block = np.einsum("jb,jnsf->bnsf", beta[h], block, optimize=True)
            out += p[h, :, None, None, None, None] * block[None]
        return out.ravel()

    def diagonal(self, w2):
        return self.backward(w2, squared=True)


class RayExtras:
    """Off-node rows grouped by ray: a = M[ray, (p,beta,nu)] * Phi[ray, f] * S[row, s]."""

    def __init__(self, q, ray, ts):
        np_, nb, nn, ns, nf = SHAPE
        rays, first, self.local = np.unique(ray, return_index=True, return_inverse=True)
        rq = q[first]
        idx = np.zeros((len(rays), 1), np.int64)
        val = np.ones((len(rays), 1))
        for i, (axis, n) in enumerate(zip(("p", "beta", "nu"), (np_, nb, nn))):
            b = BSpline.design_matrix(np.clip(rq[:, i], 0, 1 - 1e-12), ts[axis], 3)
            idx = (idx[:, :, None] * n + b.indices.reshape(-1, 4)[:, None]).reshape(len(rays), -1)
            val = (val[:, :, None] * b.data.reshape(-1, 1, 4)).reshape(len(rays), -1)
        self.M = sparse.csr_matrix(
            (val.ravel(), idx.ravel(), np.arange(len(rays) + 1) * 64), shape=(len(rays), np_ * nb * nn)
        )
        self.MT, self.M2T = self.M.T.tocsr(), self.M.power(2).T.tocsr()
        self.phi = basis(rq[:, 4], ts["phi"]) @ flat_ends(nf + 2)
        self.S = basis(q[:, 3], ts["s"])
        self.RT = sparse.csr_matrix(
            (np.ones(len(q)), (self.local, np.arange(len(q)))), shape=(len(rays), len(q))
        )
        self.rows = len(q)

    def forward(self, c):
        p = (self.M @ c.reshape(NS * 32, 128)).reshape(-1, 16, 8)
        return np.einsum("is,is->i", self.S, np.einsum("rsf,rf->rs", p, self.phi)[self.local])

    def backward(self, y):
        q = self.RT @ (y[:, None] * self.S)
        return (self.MT @ (q[:, :, None] * self.phi[:, None, :]).reshape(len(q), -1)).ravel()

    def diagonal(self, w2):
        e = self.RT @ (w2[:, None] * self.S**2)
        return (self.M2T @ (e[:, :, None] * (self.phi**2)[:, None, :]).reshape(len(e), -1)).ravel()


class Curvature:
    """Second differences of the control tensor along each axis."""

    def __init__(self, scale):
        self.factors = np.sqrt(scale * np.array(PENALTY))
        self.sizes = [int(np.prod(SHAPE) // n * (n - 2)) for n in SHAPE]
        self.size = sum(self.sizes)
        diag = np.zeros(SHAPE)
        for axis, factor in enumerate(self.factors):
            for offset, weight in [(0, 1), (1, 4), (2, 1)]:
                sl = [slice(None)] * 5
                sl[axis] = slice(offset, offset + SHAPE[axis] - 2)
                diag[tuple(sl)] += weight * factor**2
        self.diagonal = diag.ravel()

    def forward(self, x):
        c = x.reshape(SHAPE)
        return np.concatenate([f * np.diff(c, 2, axis=i).ravel() for i, f in enumerate(self.factors)])

    def backward(self, y):
        out = np.zeros(SHAPE)
        first = 0
        for axis, (factor, size) in enumerate(zip(self.factors, self.sizes)):
            shape = list(SHAPE)
            shape[axis] -= 2
            value = y[first : first + size].reshape(shape) * factor
            first += size
            for offset, weight in [(0, 1), (1, -2), (2, 1)]:
                sl = [slice(None)] * 5
                sl[axis] = slice(offset, offset + shape[axis])
                out[tuple(sl)] += weight * value
        return out.ravel()


class Weighted:
    """Weighted grid + off-node rows + curvature, acting on a full control tensor."""

    def __init__(self, grid, extra, w, we, curvature):
        self.grid, self.extra, self.w, self.we, self.curvature = grid, extra, w, we, curvature
        self.ng, self.ne = grid.shape[0], (extra.rows if extra else 0)

    def forward(self, c):
        rows = [self.w * self.grid.forward(c)]
        if self.extra:
            rows.append(self.we * self.extra.forward(c))
        return np.concatenate([*rows, self.curvature.forward(c)])

    def backward(self, y):
        ng, ne = self.ng, self.ne
        out = self.grid.backward(self.w * y[:ng]) + self.curvature.backward(y[ng + ne :])
        if self.extra:
            out += self.extra.backward(self.we * y[ng : ng + ne])
        return out


def lsq(full, mapping, adjoint, diagonal, n, target, x0, iterations, label, ridge, censored=None):
    """Jacobi-preconditioned LSMR, alternating with projection of upper-bound rows."""
    scale = np.sqrt(np.maximum(diagonal + ridge**2, 1e-30))

    def forward(x):
        z = x / scale
        return np.r_[full.forward(mapping(z)), ridge * z]

    def backward(y):
        return (adjoint(full.backward(y[:-n])) + ridge * y[-n:]) / scale

    op = LinearOperator((len(target) + n, n), matvec=forward, rmatvec=backward, dtype="f8")
    t = np.r_[target, np.zeros(n)]
    x = x0 * scale
    if censored is not None:
        # A free target at or below the bound gives residual max(pred - upper, 0).
        # Only observation rows are projected; curvature and ridge stay fixed.
        observations = t[:len(censored)]
        upper = observations[censored]

        def project(x):
            pred = full.forward(mapping(x / scale))[:len(censored)][censored]
            return np.minimum(pred, upper)

        observations[censored] = project(x)
    start = perf_counter()
    for stage in range(max(1, iterations // 200)):
        solved = lsmr(op, t, atol=1e-7, btol=1e-7, maxiter=min(200, iterations), x0=x)
        x = solved[0]
        print(
            "PCA_SOLVE", label, stage, "residual", round(float(solved[3]), 4),
            "seconds", round(perf_counter() - start), flush=True,
        )
        stable = True
        if censored is not None:
            projected = project(x)
            stable = np.allclose(projected, observations[censored], rtol=0, atol=1e-6)
            observations[censored] = projected
        if stable and solved[1] in (1, 2, 4, 5):
            break
    if not np.isfinite(x).all():
        raise FloatingPointError(f"Nonfinite PCA solution: {label}")
    return x / scale


# ---------------------------------------------------------------- data
def nodal_targets(palette, out):
    """Production targets/weights on every nodal reference, and its chart coordinates."""
    spec = F.Spec(F.CONFIG)
    P, B, N, S, occluder = [], [], [], [], []
    peaks = np.zeros((2, 16, 64))
    for p in range(16):
        with np.load(HERE / "reference" / f"p{p:02d}.npz") as z:
            phys, xyz, u, d = (z[k] for k in ("physical", "xyz", "u", "distances"))
        h = float(phys[0, 0, 0, 0])
        views = phys[:, 0, 0, 2]
        ce = np.stack([end_columns(h, v) for v in views])
        cp = np.stack([end_columns(h, v, virtual=False) for v in views])
        rel = np.rad2deg(views + np.arctan(horizon(h) / 6371))
        fields = {k: np.zeros((2, len(views), xyz.shape[2], 41, xyz.shape[1])) for k in ("L", "theta", "wL", "wT")}
        for sp in range(2):
            for b in range(xyz.shape[2]):
                # Production fixes the slice peak from the 64 main angular nodes.
                samples = xyz[:, :, b, 1:42, sp]
                r, initial = polar(samples)
                peak = max(r[:64].max(), 1e-300)
                peaks[sp, p, b] = peak
                fitted = palette.fit_theta(sp, samples, r, initial)
                for j, _, L, theta, wl, wt in F._slice_data(
                    spec, xyz[:, :, b], u, d, sp, ce, cp, rel, peak, (r, fitted), radiance=sp == 1
                ):
                    keep = u[j, 1:42, sp] > 0
                    for key, value in zip(fields, (L, theta, wl, wt)):
                        fields[key][sp, j, b, keep] = value.T
        np.savez(out / f"grid-p{p:02d}.npz", **{k: v.astype("f4") for k, v in fields.items()})
        p_, beta, _, phi = chart_axes(phys[0])
        _, _, nu, _ = chart_axes(phys[:, 0, 0])
        P.append(p_[0, 0])
        B.append(beta[0])
        N.append(nu)
        S.append(chart_distance(u[:, 1:42], h, views[:, None, None]).transpose(2, 0, 1))
        r0 = 6371 + h
        dd = d[:, 1:42]
        altitude = np.sqrt(r0 * r0 + dd * dd + 2 * r0 * dd * np.sin(views[:, None])) - 6371
        ground = np.array([bool(ray_ends(h, v)[2]) for v in views])[:, None]
        occluder.append(ground | (altitude <= EVEREST))
        print("PCA_TARGETS", p, flush=True)
    np.savez(
        out / "grid-chart.npz", p=np.array(P), beta=np.stack(B), nu=np.stack(N),
        s=np.stack(S, 1), phi=phi[:, 0], occluder=np.stack(occluder), peak=peaks,
    )


def slice_peak(chart, sp, p, beta):
    """Nodal slice peak at off-node (p, beta), log-linear between the nodes."""
    logs = np.log(np.maximum(chart["peak"][sp], 1e-300))
    nodes = chart["p"]
    i = np.clip(np.searchsorted(nodes, p) - 1, 0, len(nodes) - 2)
    t = np.clip((p - nodes[i]) / (nodes[i + 1] - nodes[i]), 0, 1)
    at = lambda k: np.array([np.interp(b, chart["beta"][j], logs[j]) for j, b in zip(k, beta)])
    return np.exp((1 - t) * at(i) + t * at(i + 1))


def off_node_references():
    """Independent random rays between the nodes (GPU spectral integration)."""
    rng = np.random.default_rng(EXTRA_SEED)
    physical = []
    for group in ("random", "twilight", "ground", "ground_twilight"):
        phys = from_grid(rng.random((EXTRA_RAYS, 4)))
        phys[:, 0] = np.clip(phys[:, 0], *HEIGHT_RANGE)
        if "ground" in group:
            phys[:, 0] = np.exp(rng.uniform(np.log(0.002), np.log(0.1), EXTRA_RAYS))
        if "twilight" in group:
            phys[:, 1] = np.deg2rad(rng.uniform(-18, 3, EXTRA_RAYS))
        near = rng.random(EXTRA_RAYS) < 0.5  # half the rays cluster around the horizon
        phys[near, 2] = -dip(phys[near, 0]) + np.deg2rad(rng.uniform(-2, 2, near.sum()))
        physical.append(phys)
    integrate_rays(np.concatenate(physical), rng, HERE / FOLDER / "off-node.npz")


def shadow_references():
    """Rays with the light near the shadow onset, for the Mie radius fit only."""
    rng = np.random.default_rng(SHADOW_SEED)
    physical = []
    for group, (low, high) in (("low", (-4.0, 0.5)), ("all", (-8.0, 1.0))):
        phys = from_grid(rng.random((SHADOW_RAYS, 4)))
        phys[:, 0] = np.clip(phys[:, 0], *HEIGHT_RANGE)
        if group == "low":
            phys[:, 0] = np.exp(rng.uniform(np.log(0.002), np.log(3.0), SHADOW_RAYS))
        phys[:, 1] = np.deg2rad(rng.uniform(low, high, SHADOW_RAYS)) - dip(phys[:, 0])
        near = rng.random(SHADOW_RAYS) < 0.75
        phys[near, 2] = np.deg2rad(rng.uniform(-2, 8, near.sum())) - dip(phys[near, 0])
        physical.append(phys)
    integrate_rays(np.concatenate(physical), rng, HERE / FOLDER / "shadow-rays.npz")


def integrate_rays(phys, rng, path):
    from lut_scattering import Integrator, straight_samples, spectral_weights, phase_normalize

    distances, columns, ends = [], [], []
    for h, _, view, _ in phys:
        d = sample_ray(h, view)[0]
        fractions = np.sort(np.r_[0, 10 ** rng.uniform(-5, 0, 10), rng.random(2), 1])
        target = fractions * d[-1]
        dense = np.linspace(0, d[-1], 16385)
        col = _columns(h, view, dense)
        columns.append(np.stack([np.interp(target, dense, col[:, sp]) for sp in range(2)], -1))
        distances.append(target)
        ends.append(end_columns(h, view))
    distances, columns, ends = np.array(distances), np.array(columns), np.array(ends)
    weights = spectral_weights()
    gpu = Integrator()
    try:
        xyz = []
        for first in range(0, len(phys), 2048):
            sl = slice(first, first + 2048)
            queries = np.column_stack([phys[sl, 0], phys[sl, 1], phys[sl, 3], np.arange(len(phys[sl]))])
            paths = np.stack([straight_samples(h, v, d, 32) for (h, _, v, _), d in zip(phys[sl], distances[sl])])
            # At most 64 rays per dispatch, finished before the next: keeps the GPU responsive.
            xyz.append(phase_normalize(phys[sl], gpu.evaluate(queries, paths, weights=weights, dispatch=64)))
            print("PCA_OFF_NODE", first + len(phys[sl]), flush=True)
    finally:
        gpu.close()
    xyz = np.concatenate(xyz)
    assert np.isfinite(xyz).all() and np.all(xyz >= 0)
    np.savez(path, physical=phys, distances=distances, columns=columns, ends=ends, xyz=xyz)


def off_node_rows(palette, chart, name="off-node.npz"):
    """Include dark off-node samples for radius; fit theta only for visible colors."""
    z = dict(np.load(HERE / FOLDER / name))
    phys = z["physical"]
    r = np.hypot(z["xyz"][..., 0], z["xyz"][..., 2])
    theta = BLUE - np.arctan2(z["xyz"][..., 2], z["xyz"][..., 0])
    u = np.clip(z["columns"] / z["ends"][:, None, :], 0, 1)
    p, beta, nu, phi = chart_axes(phys)
    _, _, ground = ray_ends(phys[:, 0], phys[:, 2])
    ground = ground[:, None]
    r0 = 6371 + phys[:, 0, None]
    d = z["distances"]
    altitude = np.sqrt(r0 * r0 + d * d + 2 * r0 * d * np.sin(phys[:, 2, None])) - 6371
    factor = np.where(ground | (altitude <= EVEREST), 1.0, EXCLUDED)
    rows = {}
    for sp in range(2):
        rs = r[:, :, sp]
        peak = rs.max(axis=1, keepdims=True)
        nonzero = rs > np.maximum(peak * 1e-12, 1e-300)
        visible = peak > 1e-6 * rs.max()
        valid = u[:, :, sp] > 0
        L = F.log_radius_target(rs, np.maximum(z["columns"][:, :, sp], 1e-300))
        if sp == 1:
            peaks = slice_peak(chart, sp, p, beta)[:, None]
            radius_weight = F.radiance_weight(rs, peaks, L == F.LOG_RADIUS_FLOOR)
        else:
            radius_weight = np.where(L == F.LOG_RADIUS_FLOOR, F.SHADOW_WEIGHT, np.where(visible, 1.0, F.DARK))
        angle_use = valid & nonzero & visible
        fitted = np.zeros_like(L)
        fitted[angle_use] = palette.fit_theta(
            sp, z["xyz"][:, :, sp][angle_use], rs[angle_use], theta[:, :, sp][angle_use]
        )
        # Relative brightness along the ray applies only to color.
        brightness = np.log(np.maximum(rs, 1e-300) / np.maximum(u[:, :, sp], 1e-300))
        brightest = np.where(nonzero, brightness, -np.inf).max(axis=1, keepdims=True)
        wt = np.where(nonzero, np.clip(np.exp(brightness - np.where(nonzero.any(axis=1, keepdims=True), brightest, 0)), 0.001, 1), 0)
        end = np.zeros_like(valid)
        end[:, -1] = True
        endpoint = np.where(end, F.END, 1.0)
        s = chart_distance(u[:, :, sp], phys[:, 0, None], phys[:, 2, None])
        q = np.stack(np.broadcast_to(np.array([p, beta, nu, np.zeros_like(p), phi])[..., None], (5, *s.shape)), -1).copy()
        q[..., 3] = s
        rows[sp] = dict(
            q=q[valid], ray=np.broadcast_to(np.arange(len(phys))[:, None], valid.shape)[valid],
            L=L[valid], theta=fitted[valid],
            wL=(endpoint * factor * radius_weight)[valid], wT=(endpoint * factor * wt * angle_use)[valid],
        )
    return rows


# ---------------------------------------------------------------- fitting
def grid_operator(chart, ts, sp):
    return Tensor(
        basis(chart["p"], ts["p"]), basis(chart["beta"], ts["beta"]), basis(chart["nu"], ts["nu"]),
        basis(chart["s"][sp], ts["s"]), basis(chart["phi"], ts["phi"]) @ flat_ends(10),
    )


def grid_field(sp, field, occluder, targets):
    key, weight = ("L", "wL") if field == 0 else ("theta", "wT")
    values, weights = [], []
    for p in range(16):
        with np.load(targets / f"grid-p{p:02d}.npz") as z:
            values.append(z[key][sp].astype("f8"))
            weights.append(z[weight][sp].astype("f8"))
    factor = np.where(occluder, 1.0, EXCLUDED)[:, :, None, :, None]
    # Production normal matrices weight squared residuals; LSMR rows take sqrt(weight).
    return np.stack(values).ravel(), np.sqrt(np.stack(weights) * factor).ravel()


def fit_field(sp, field, ts, chart, rows, targets, output, shadow=None):
    key = ("L", "theta")[field]
    K = RANKS[sp][field]
    grid = grid_operator(chart, ts, sp)
    y, w = grid_field(sp, field, chart["occluder"], targets)
    censored = (y == F.LOG_RADIUS_FLOOR) & (w > 0) if field == 0 else None
    ones = lambda z: z
    # 1) Full-rank initial controls from the nodal grid alone.
    curvature = Curvature(float(grid.diagonal(w**2).mean()))
    full = Weighted(grid, None, w, None, curvature)
    diagonal = grid.diagonal(w**2) + curvature.diagonal
    trace = float(diagonal.mean())
    c0 = lsq(full, ones, ones, diagonal, NS * NC, np.r_[w * y, np.zeros(curvature.size)],
             np.full(NS * NC, float(np.sum(y * w * w) / np.sum(w * w))), INIT_ITERATIONS,
             f"sp{sp}-{key}-init", np.sqrt(trace * 1e-9), censored).reshape(NS, NC)
    # 2) Off-node rows and the low-rank factors.
    r = rows[sp]
    if censored is None:
        use = r["wT"] > 0
        r = {k: v[use] for k, v in r.items()}
    else:
        censored = np.r_[censored, r["L"] == F.LOG_RADIUS_FLOOR]
    we = np.sqrt(r["wL" if field == 0 else "wT"])
    we *= np.sqrt(EXTRA_MASS * float(np.sum(w * w)) / max(float(np.sum(we**2)), 1e-30))
    if shadow is not None:
        # Shadow rows get the same per-row scale as the random off-node rows.
        scale = np.sqrt(EXTRA_MASS * float(np.sum(w * w)) / max(float(np.sum(r["wL"])), 1e-30))
        r = {k: np.r_[v, shadow[k] + (r["ray"].max() + 1 if k == "ray" else 0)] for k, v in r.items()}
        we = np.r_[we, scale * np.sqrt(shadow["wL"])]
        censored = np.r_[censored, shadow["L"] == F.LOG_RADIUS_FLOOR]
    extra = RayExtras(r["q"], r["ray"], ts)
    diag = (grid.diagonal(w**2) + extra.diagonal(we**2)).reshape(NS, NC)
    trace = float(diag.mean())
    curvature = Curvature(trace)
    diag = diag + curvature.diagonal.reshape(NS, NC)
    ridge = np.sqrt(trace * 1e-9)
    full = Weighted(grid, extra, w, we, curvature)
    target = np.r_[w * y, we * r[key], np.zeros(curvature.size)]
    del y
    m = c0.mean(axis=0)
    u, s, vt = np.linalg.svd(c0 - m, full_matrices=False)
    A, B = u[:, :K] * s[:K], vt[:K].copy()
    del c0, u, s, vt
    # 3) Alternating least squares: [m; B] | A, A | [m; B], [m; B] | A.
    for step in range(2):
        At = np.c_[np.ones(NS), A]
        X = lsq(full, lambda z: (At @ z.reshape(K + 1, NC)).ravel(),
                lambda g: (At.T @ g.reshape(NS, NC)).ravel(), ((At**2).T @ diag).ravel(),
                (K + 1) * NC, target, np.r_[m[None], B].ravel(), B_ITERATIONS,
                f"sp{sp}-{key}-K{K}-B{step}", ridge, censored).reshape(K + 1, NC)
        m, B = X[0], X[1:]
        if step:
            break
        offset = full.forward(np.broadcast_to(m, (NS, NC)).ravel())
        A = lsq(full, lambda z: (z.reshape(NS, K) @ B).ravel(),
                lambda g: (g.reshape(NS, NC) @ B.T).ravel(), (diag @ (B**2).T).ravel(),
                NS * K, target - offset, A.ravel(), A_ITERATIONS,
                f"sp{sp}-{key}-K{K}-A{step}", ridge, censored).reshape(NS, K)
        del offset
    np.savez(output, m=m, A=A, B=B)


def factor_path(palette, sp, key):
    radius = MIE_RADIUS_FIT if sp == 1 else RADIUS_FIT
    folder = HERE / FOLDER / (palette.cache_key if key == "theta" else radius)
    return folder / f"factors-sp{sp}-{key}.npz"


def build_pca():
    """Nodal targets, off-node references and the low-rank fits, each resumable."""
    out = HERE / FOLDER
    palette = Palette()
    pending = []
    for sp in range(2):
        for field, key in enumerate(("L", "theta")):
            path = factor_path(palette, sp, key)
            if not path.exists():
                pending.append((sp, field, key, path))
    if not pending:
        return
    targets = out / f"{palette.cache_key}-{MIE_RADIUS_FIT}"
    targets.mkdir(parents=True, exist_ok=True)
    if not (targets / "grid-chart.npz").exists():
        nodal_targets(palette, targets)
    if not (out / "off-node.npz").exists():
        off_node_references()
    if not (out / "shadow-rays.npz").exists():
        shadow_references()
    ts = knots()
    chart = dict(np.load(targets / "grid-chart.npz"))
    rows = off_node_rows(palette, chart)
    for sp, field, key, path in pending:
        start = perf_counter()
        path.parent.mkdir(exist_ok=True)
        shadow = off_node_rows(palette, chart, "shadow-rays.npz")[1] if (sp, field) == (1, 0) else None
        fit_field(sp, field, ts, chart, rows, targets, path, shadow)
        print("PCA_FIELD", sp, key, "seconds", round(perf_counter() - start), flush=True)


def packed_factors(palette):
    """Pack fields in order; Mie L's mean and B[0] require a separate float32 pair."""
    parts, ranks, precise = [], [], []
    for sp in range(2):
        for key in ("L", "theta"):
            path = factor_path(palette, sp, key)
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}; run --stage pca first")
            z = np.load(path)
            if sp == 1 and key == "L":
                precise = [z["m"].ravel(), z["B"][0].ravel()]
                parts += [z["A"].ravel(), z["B"][1:].ravel()]
            else:
                parts += [z["m"].ravel(), z["A"].ravel(), z["B"].ravel()]
            ranks.append(int(z["A"].shape[1]))
    payload = np.concatenate(parts).astype("<f2")
    pair = np.concatenate(precise).astype("<f4")
    if pair.size != 2 * NC or not np.isfinite(pair).all() or not np.isfinite(payload).all():
        raise ArithmeticError("Nonfinite PCA factors")
    ts = knots()
    return payload, pair, dict(
        ranks=ranks,
        knots={k: [float(v) for v in t] for k, t in ts.items()},
        heightRange=list(HEIGHT_RANGE),
        epsilon=EPSILON,
    )
