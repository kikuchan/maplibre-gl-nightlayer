"""Production finite-path targets, weights and B-spline layout (nodal references)."""

from types import SimpleNamespace
import numpy as np
from scipy.interpolate import BSpline
from lut_model import to_grid, polar

CONFIG = SimpleNamespace(ns=16, nn=32, nf=8, focus=0.05)
END, DARK = 4.0, 1e-2
TAU, RELEPS = 0.5, (3e-3, 3e-2, 3.0)
# Below this log radius per species column, the data only supplies an upper bound.
# Keep the bound above the shader's -40 floor.
LOG_RADIUS_FLOOR = -34.0
SHADOW_WEIGHT = 0.1
# Mie radius rows weigh their log residual by (r / slice peak)^2, at least this
# floor, so the fit approximates radiance error relative to the brightest value of
# the (height, light) slice instead of spending its terms on invisible log detail.
RADIANCE_FLOOR = 1e-6


def radiance_weight(r, peak, censored):
    rel = np.minimum(r / np.maximum(peak, 1e-300), 1.0)
    return np.where(censored, SHADOW_WEIGHT * RADIANCE_FLOOR, np.maximum(rel * rel, RADIANCE_FLOOR))


def log_radius_target(radius, column):
    """Return finite log targets; the floor marks an upper-bound row."""
    ratio = radius / column
    return np.log(
        ratio, out=np.full(ratio.shape, LOG_RADIUS_FLOOR),
        where=ratio > np.exp(LOG_RADIUS_FLOOR),
    )


def clamped(n, interior=None):
    """Clamped cubic knot vector with n basis functions on [0, 1]."""
    if interior is None:
        interior = np.linspace(0, 1, n - 2)[1:-1]
    assert len(interior) == n - 4
    return np.r_[[0.0] * 4, interior, [1.0] * 4]


def design(t, x):
    x = np.clip(np.asarray(x, "f8"), 0, 1 - 1e-12)
    return BSpline.design_matrix(x, t, 3).toarray()


def flat_ends(n):
    """Merge first/last pairs: zero end derivative for a clamped cubic B-spline."""
    m = np.zeros((n, n - 2))
    m[0, 0] = m[1, 0] = 1
    for i in range(2, n - 2):
        m[i, i - 1] = 1
    m[n - 2, n - 3] = m[n - 1, n - 3] = 1
    return m


def d2(n):
    return np.diff(np.eye(n), 2, axis=0)


def nu_knots(n, focus, width):
    """Interior knots for n functions on q_nu, denser around `focus` (horizon)."""
    density = 1 + 3 * np.exp(-0.5 * ((np.linspace(0, 1, 4097) - focus) / width) ** 2)
    cdf = np.cumsum(density)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    return np.interp(np.linspace(0, 1, n - 2)[1:-1], cdf, np.linspace(0, 1, 4097))


class Spec:
    def __init__(self, args):
        self.ns, self.nn, self.nf = (args.ns, args.nn, args.nf)
        self.ts = clamped(self.ns)
        horizon = to_grid(
            np.array([[1.0, 0.0, -np.arctan(np.sqrt(12743.0) / 6371.0), 0.0]])
        )[0, 1]
        self.tn = clamped(
            self.nn, nu_knots(self.nn, horizon, args.focus) if args.focus > 0 else None
        )
        self.tf = clamped(self.nf + 2)
        self.mf = flat_ends(self.nf + 2)

    def s(self, u, ug, rel):
        e = ray_eps(rel, ug)
        return np.arcsinh(np.asarray(u) / e) / np.arcsinh(1 / e)

    def bs(self, u, ug, rel):
        """Distance basis in horizon-normalized species column coordinates."""
        return design(self.ts, self.s(u, ug, rel))

    def bn(self, q):
        return design(self.tn, q)

    def bf(self, q):
        return design(self.tf, q) @ self.mf


def ray_eps(rel_deg, ug):
    """Per-ray warp strength, continuous across the geometric horizon."""
    lo, hi, R = RELEPS
    x = np.clip(np.asarray(rel_deg) / R, 0, 1)
    e = lo * (hi / lo) ** (x * x * (3 - 2 * x))
    return e * np.minimum(np.asarray(ug) / TAU, 1.0)


def _slice_data(spec, xyz, u, d, sp, cend, cphys, rel, slice_max=None, encoded=None, radiance=False):
    """Per-nu sample arrays: list of (j, Bs (K,ns), L (8,K), T (8,K), wl, wt).

    Judge darkness for radius by r / column, not raw r: near-origin samples
    can be small even in daylight. Retain relative brightness weights for theta."""
    if slice_max is None:
        slice_max = max(polar(xyz[:, :, 1:42, sp])[0].max(), 1e-300)
    out = []
    for j in range(len(u)):
        if d[j, 41] <= 0:
            continue
        uj = u[j, 1:42, sp]
        keep = uj > 0
        uj = uj[keep]
        if encoded is None:
            r, th = polar(xyz[j, :, 1:42, sp][:, keep])
        else:
            r, th = (field[j][:, keep] for field in encoded)
        peak = r.max(axis=1, keepdims=True)
        end = len(uj) - 1
        nonzero = r > np.maximum(peak * 1e-12, 1e-300)
        column = uj * cend[j, sp]
        L = log_radius_target(r, column)
        visible = np.where(peak > 1e-06 * slice_max, 1.0, DARK)
        if radiance:
            wl = radiance_weight(r, slice_max, L == LOG_RADIUS_FLOOR)
        else:
            wl = np.where(L == LOG_RADIUS_FLOOR, SHADOW_WEIGHT, visible)
        brightness = np.where(nonzero, r / column, 0)
        brightest = brightness.max(axis=1, keepdims=True)
        rel_w = brightness / np.where(brightest > 0, brightest, 1)
        wt = nonzero * np.clip(rel_w, 0.001, 1) * visible
        wl[:, end] *= END
        wt[:, end] *= END
        out.append(
            (
                j,
                spec.bs(
                    uj,
                    np.full(len(uj), min(cphys[j, sp] / cend[j, sp], 1.0)),
                    np.full(len(uj), rel[j]),
                ),
                L,
                th,
                wl,
                wt,
            )
        )
    return out
