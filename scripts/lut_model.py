"""Physical model, observer chart and fixed spectral quadrature."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

HERE = Path(os.environ["NIGHTLAYER_LUT_WORKDIR"])
MODEL = json.loads(Path(__file__).with_name("lut-model.json").read_text())
META = MODEL["metadata"]
LAYOUT = META["observer"]
CDF = np.array(LAYOUT["axisCDF"], "f8")
NODES = [np.array(n, "f8") for n in LAYOUT["axisNodes"]]
BLUE = float(META["palette"]["thetaBlue"])
RGB = np.array(
    [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.204, 1.057]]
)
ANGLE_MIN = np.arcsinh(-180.0)
ANGLE_SPAN = np.arcsinh(200.1717139250062) - ANGLE_MIN
R, TOP = 6371.0, 100.0


@dataclass(frozen=True)
class Atmosphere:
    refractivity: float = 0.00029
    scale_height: float = 8.0

    def index(self, height):
        density = np.exp(-np.clip(height, 0, TOP) / self.scale_height)
        return 1 + self.refractivity * (density - np.exp(-TOP / self.scale_height))

    def gradient(self, height):
        return (
            -self.refractivity
            / self.scale_height
            * np.exp(-np.clip(height, 0, TOP) / self.scale_height)
        )

    def rhs(self, _, state):
        x, z, angle = state[:3]
        radius = np.hypot(x, z)
        height = radius - R
        tangent = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-tangent[1], tangent[0]])
        curvature = (
            self.gradient(height)
            / self.index(height)
            * (normal @ np.array([x, z]) / radius)
        )
        density = np.array(
            [
                np.exp(-max(height, 0) / 8),
                np.exp(-max(height, 0) / 1.2),
                max(0, 1 - abs(height - 25) / 15),
            ]
        )
        return np.r_[tangent, curvature, density]

    def horizon(self, height):
        """Apparent elevation of a sea-level tangent ray (monotone n(r)r)."""
        ratio = self.index(0) * R / (self.index(height) * (R + height))
        return -np.arccos(np.clip(ratio, -1, 1))


def trace(height, elevation, atmosphere=Atmosphere(), *, rtol=2e-10, max_step=8.0):
    """Backward optical ray from observer to first ground/top intersection."""
    if not 0 < height < TOP:
        raise ValueError("Reference tracer expects 0 < observer height < 100 km")

    def ground(_, y):
        return np.hypot(y[0], y[1]) - R

    def top(_, y):
        return np.hypot(y[0], y[1]) - R - TOP

    def tangent(_, y):
        return y[0] * np.cos(y[2]) + y[1] * np.sin(y[2])

    ground.terminal = top.terminal = True
    ground.direction, top.direction = (-1, 1)
    expected_ground = elevation < atmosphere.horizon(height)
    tangent.terminal = expected_ground
    tangent.direction = 1
    solution = solve_ivp(
        atmosphere.rhs,
        (0, 5000),
        [0, R + height, elevation, 0, 0, 0],
        method="DOP853",
        rtol=rtol,
        atol=rtol * 0.0001,
        max_step=max_step,
        events=(ground, top, tangent),
        dense_output=True,
    )
    if not solution.success or solution.status != 1:
        raise RuntimeError(f"Ray did not reach a boundary: {solution.message}")
    if (
        expected_ground
        and (not len(solution.t_events[0]))
        and len(solution.t_events[2])
    ):
        if ground(solution.t[-1], solution.y[:, -1]) >= 0:
            raise RuntimeError("Ground tangent disagrees with the optical invariant")
        i = np.flatnonzero(np.hypot(solution.y[0], solution.y[1]) >= R)[-1]
        hit = brentq(
            lambda t: ground(t, solution.sol(t)),
            solution.t[i],
            solution.t[-1],
            xtol=1e-10,
        )
        state = solution.sol(hit)
        solution.t = np.r_[solution.t[: i + 1], hit]
        solution.y = np.column_stack([solution.y[:, : i + 1], state])
        solution.t_events[0] = np.array([hit])
        solution.y_events[0] = state[None]
    return (solution, bool(len(solution.t_events[0])))


def warp(x, axis):
    """Runtime hermiteWarp: normalized physical parameter -> grid fraction."""
    mapped = np.interp(np.clip(x, 0, 1), np.linspace(0, 1, 65), CDF[axis])
    nodes = NODES[axis]
    return np.interp(mapped, nodes, np.linspace(0, 1, len(nodes)))


def unwarp(q, axis):
    nodes = NODES[axis]
    mapped = np.interp(q, np.linspace(0, 1, len(nodes)), nodes)
    return np.interp(mapped, CDF[axis], np.linspace(0, 1, 65))


def horizon(h):
    return np.sqrt(h * (12742 + h))


def dip(h):
    return np.arctan(horizon(h) / R)


def angle_fraction(angle, h):
    a = np.rad2deg(angle + dip(h)) / 0.5
    return (np.arcsinh(a) - ANGLE_MIN) / ANGLE_SPAN


def to_grid(physical):
    """physical [h, sun, view, phi] (radians) -> grid fractions [p, nu, phi, beta] in [0,1]."""
    h, sun, view, phi = np.asarray(physical, "f8").T
    return np.stack(
        [
            warp(horizon(h) / np.sqrt(1284200.0), 0),
            warp(angle_fraction(view, h), 1),
            warp(phi / np.pi, 2),
            warp(angle_fraction(sun, h), 3),
        ],
        axis=-1,
    )


def from_grid(q):
    """Grid fractions [p, nu, phi, beta] -> physical [h, sun, view, phi]."""
    q = np.asarray(q, "f8")
    hz = unwarp(q[..., 0], 0) * np.sqrt(1284200.0)
    h = hz * hz / (np.sqrt(R * R + hz * hz) + R)
    d = np.arctan(hz / R)
    angle = lambda x: np.deg2rad(0.5 * np.sinh(ANGLE_MIN + ANGLE_SPAN * x))
    view = np.clip(angle(unwarp(q[..., 1], 1)) - d, -np.pi / 2, np.pi / 2)
    sun = np.clip(angle(unwarp(q[..., 3], 3)) - d, -np.pi / 2, np.pi / 2)
    phi = unwarp(q[..., 2], 2) * np.pi
    return np.stack([h, sun, view, phi], axis=-1)


def ray_ends(h, view):
    """(finite cut, terminal, ground) as in the runtime atmosphereRay."""
    t = (R + h) * np.sin(view)
    radial = h * (12742 + h)
    disc = t * t - radial
    ground = (t < 0) & (disc >= 0)
    delta = (100 - h) * (12842 + h)
    root = np.sqrt(np.maximum(t * t + delta, 0))
    exit_ = np.where(t >= 0, delta / np.maximum(root + t, 1e-20), root - t)
    hit = radial / np.maximum(-t + np.sqrt(np.maximum(disc, 0)), 1e-20)
    cut = np.where(ground, hit, np.minimum(np.sqrt(radial), exit_))
    terminal = np.where(ground, hit, exit_)
    return (np.maximum(cut, 0), np.maximum(terminal, 0), ground)


def _columns(h, view, distances):
    r = R + h
    delta = distances * (2 * r * np.sin(view) + distances)
    altitude = np.maximum(h + delta / (np.sqrt(r * r + delta) + r), 0)
    dens = np.exp(-altitude[..., None] / np.array([8.0, 1.2]))
    steps = np.diff(distances)[:, None] * (dens[1:] + dens[:-1]) / 2
    return np.concatenate([np.zeros((1, 2)), np.cumsum(steps, axis=0)])


def sample_ray(h, view, dense=16385):
    """42 finite distances (same targets as the production build) and u per species."""
    cut, terminal, ground = ray_ends(h, view)
    cut = float(cut)
    targets = np.unique(np.r_[np.linspace(0, 1, 13), np.geomspace(1e-05, 0.1, 8)])
    if cut <= 0:
        return (np.zeros(42), np.zeros((42, 2)), float(terminal))
    grid = np.linspace(0, cut, dense)
    col = _columns(h, view, grid)
    normalized = col / np.maximum(col[-1], np.finfo(float).tiny)
    d = np.sort(
        np.concatenate([np.interp(targets, normalized[:, s], grid) for s in (0, 1)])
    )
    cols = np.stack([np.interp(d, grid, col[:, s]) for s in (0, 1)], axis=1)
    denominator = col[-1] + (max(float(horizon(h)) - cut, 0.0) if ground else 0.0)
    u = np.clip(cols / np.maximum(denominator, 1e-30), 0, 1)
    u[0] = 0
    if not ground:
        u[-1] = 1
    return (d, u, max(float(terminal), cut))


def polar(xyz):
    v = np.asarray(xyz, "f8")[..., [0, 2]]
    return (np.linalg.norm(v, axis=-1), BLUE - np.arctan2(v[..., 1], v[..., 0]))


def end_columns(h, view, dense=16385, virtual=True):
    """Runtime endColumns (R, M): species column to the finite cut, plus (virtual=True)
    the sea-level virtual continuation to the horizon distance for ground rays."""
    cut, _, ground = ray_ends(h, view)
    cut = float(cut)
    if cut <= 0:
        return np.array([1e-30, 1e-30])
    col = _columns(h, view, np.linspace(0, cut, dense))[-1]
    return np.maximum(
        col + (max(float(horizon(h)) - cut, 0.0) if ground and virtual else 0.0), 1e-30
    )


def apparent_view(h, elevation):
    """Horizon-preserving sky/foreground presentation, with fixed poles."""
    ground = -float(dip(h))
    delta = Atmosphere().horizon(h) - ground
    span = np.pi / 2 - ground if elevation >= ground else np.pi / 2 + ground
    t = np.clip(abs(elevation - ground) / span, 0, 1)
    return elevation + delta * (1 - t) ** 2 * (1 + 2 * t)
