"""Local RGB scattering sources and density for spherical transport (km)."""

import numpy as np

RADIUS = 6371.0
TOP = 100.0


def sample(data, points):
    p = np.clip(points, 0, np.array(data.shape[:-1]) - 1)
    lo = np.floor(p).astype(int)
    hi = np.minimum(lo + 1, np.array(data.shape[:-1]) - 1)
    f = p - lo
    result = np.zeros(p.shape[:-1] + (3,))
    for mask in range(1 << p.shape[-1]):
        index, weight = [], np.ones(p.shape[:-1])
        for axis in range(p.shape[-1]):
            upper = (mask >> axis) & 1
            index.append(hi[..., axis] if upper else lo[..., axis])
            weight *= f[..., axis] if upper else 1 - f[..., axis]
        result += data[tuple(index)] * weight[..., None]
    return result


def density(height):
    h = np.maximum(height, 0)
    values = np.stack((np.exp(-h / 8), np.exp(-h / 1.2), np.maximum(0, 1 - np.abs(h - 25) / 15)), axis=-1)
    return np.where((np.asarray(height) >= 0)[..., None], values, 0)


def source(multiple, beam, h, mu, sun, nu):
    above = np.asarray(h) >= 0
    h = np.clip(h, 0, TOP)
    horizon = np.sqrt(TOP * (2 * RADIUS + TOP))
    top = lambda m: -RADIUS * m + np.sqrt((RADIUS * m) ** 2 + horizon ** 2)
    a = (top(sun) - TOP) / (horizon - TOP)
    A = (top(-.5) - TOP) / (horizon - TOP)
    sx = np.clip(np.maximum(1 - a / A, 0) / (1 + a), 0, 1) * (beam.shape[1] - 1)
    z = np.sqrt(h * (2 * RADIUS + h)) / horizon * (multiple.shape[0] - 1)
    y = (mu + 1) * .5 * (multiple.shape[1] - 1)
    angle = np.clip((nu + 1) * .5, 0, 1) * 7
    i = np.minimum(np.floor(angle), 6)
    f = angle - i
    width = beam.shape[1]
    ms = sample(multiple, np.stack(np.broadcast_arrays(z, y, i * width + sx), axis=-1)) * (1 - f)[..., None]
    ms += sample(multiple, np.stack(np.broadcast_arrays(z, y, (i + 1) * width + sx), axis=-1)) * f[..., None]
    direct = sample(beam, np.stack(np.broadcast_arrays(z, sx), axis=-1))
    mie = sample(beam, np.stack(np.broadcast_arrays(z + multiple.shape[0], sx), axis=-1))
    phase = 3 / (16 * np.pi) * (1 + nu * nu)
    dr, dm, _ = np.moveaxis(density(h), -1, 0)
    values = np.stack((dr[..., None] * (direct + ms / phase[..., None]), dm[..., None] * mie), axis=-2)
    return np.where(above[..., None, None], values, 0)
