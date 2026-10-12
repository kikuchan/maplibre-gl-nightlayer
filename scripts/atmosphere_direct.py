"""Build phase-separated mean camera-to-endpoint transport."""

import numpy as np

from atmosphere_transport import RADIUS, TOP, density, source


def solar_boundary(irradiance, radius_degrees):
    """Uniform solar disk in the same linear RGB units as the transport sources."""
    radius = np.deg2rad(radius_degrees)
    if not 0 < radius < np.pi / 2:
        raise ValueError('Solar angular radius must lie between zero and pi/2')
    irradiance = np.asarray(irradiance)
    if irradiance.shape != (3,) or not np.isfinite(irradiance).all():
        raise ValueError('Solar irradiance must be finite RGB')
    return irradiance / (np.pi * np.sin(radius) ** 2), float(radius)


def geometry(height, view, ground):
    """Separate ground/escape branches; never interpolate across the horizon."""
    r = RADIUS + height
    c = height * (2 * RADIUS + height)
    horizon = -np.sqrt(c) / r
    mu = horizon + (-1 - horizon if ground else 1 - horizon) * view ** 2
    t = r * mu
    if ground:
        distance = c / max(-t + np.sqrt(max(0, t * t - c)), 1e-30)
    else:
        distance = -t + np.sqrt(t * t + (TOP - height) * (2 * RADIUS + TOP + height))
    return mu, max(0, distance)


def integrate(multiple, beam, extinction, height, mu, distances, sun, azimuth, cells=256):
    """Precompute phase-separated sources along straight camera rays."""
    distances = np.asarray(distances)
    maximum = max(float(distances[-1]), 0)
    horizontal = np.sqrt(max(0, 1 - mu * mu))
    nu = mu * sun + horizontal * np.sqrt(np.maximum(0, 1 - sun * sun)) * np.cos(azimuth)
    if maximum == 0:
        return np.zeros((len(distances), 3)), np.zeros((len(distances), *nu.shape, 2, 3))
    # Resolve both observer and tangent ends independently of the LUT knots.
    uniform = np.linspace(0, maximum, cells + 1)
    logarithmic = np.expm1(np.linspace(0, np.log1p(maximum), cells + 1))
    closest = -(RADIUS + height) * mu
    tangent = np.clip(closest + np.linspace(-maximum, maximum, cells + 1) ** 3 / maximum ** 2, 0, maximum)
    edges = np.unique(np.r_[uniform, logarithmic, tangent, distances])
    ds = np.diff(edges)
    d = (edges[:-1] + edges[1:]) * .5
    t = (RADIUS + height) * mu
    radial = height * (2 * RADIUS + height) + d * (2 * t + d)
    radius = np.sqrt(np.maximum(0, RADIUS ** 2 + radial))
    h = radial / (radius + RADIUS)
    beta = density(h) @ extinction
    tau = beta * ds[:, None]
    cumulative_tau = np.vstack((np.zeros(3), np.cumsum(tau, axis=0)))
    weight = np.exp(-cumulative_tau[:-1]) * (-np.expm1(-tau)) / np.maximum(beta, 1e-30)
    angular_shape = (1,) * nu.ndim
    r = radius.reshape((-1, *angular_shape))
    solar = ((RADIUS + height) * sun + d.reshape((-1, *angular_shape)) * nu) / r
    values = source(multiple, beam, h.reshape((-1, *angular_shape)),
                    (t + d).reshape((-1, *angular_shape)) / r, solar, nu)
    increments = weight.reshape((-1, *angular_shape, 1, 3)) * values
    cumulative = np.concatenate((np.zeros((1, *nu.shape, 2, 3)), np.cumsum(increments, axis=0)))
    indices = np.searchsorted(edges, distances)
    return cumulative_tau[indices], cumulative[indices]


def build(multiple, beam, extinction, size=(9, 9, 13, 13, 5), cells=256):
    """Mean optical coefficients and phase-separated mean scattering sources."""
    axes = [np.linspace(0, 1, n) for n in size]
    height, view, distance, sun, azimuth = axes
    heights = np.expm1(height * np.log(11)) ** 2
    fractions = np.expm1(distance * np.log(256)) / 255
    suns = np.sinh((sun * 2 - 1) * 4) / np.sinh(4)
    azimuths = azimuth * np.pi
    radiance = np.zeros((2, *size, 2, 3), dtype=np.float32)
    optical = np.zeros((2, *size[:3], 3), dtype=np.float32)
    sun, azimuth = np.meshgrid(suns, azimuths, indexing='ij')
    for branch in range(2):
        for ih, h in enumerate(heights):
            for iv, v in enumerate(view):
                mu, maximum = geometry(h, v, branch == 0)
                distances = fractions * maximum
                tau, values = integrate(multiple, beam, extinction, h, mu, distances, sun, azimuth, cells)
                nu = mu * sun + np.sqrt(max(0, 1 - mu * mu)) * np.sqrt(1 - sun * sun) * np.cos(azimuth)
                beta = density(h) @ extinction
                local = source(multiple, beam, h, mu, sun, nu)
                tau = np.divide(tau, distances[:, None], out=np.broadcast_to(beta, tau.shape).copy(), where=distances[:, None] > 0)
                divisor = distances.reshape((-1, *([1] * (values.ndim - 1))))
                values = np.divide(values, divisor, out=np.broadcast_to(local, values.shape).copy(), where=divisor > 0)
                optical[branch, ih, iv] = tau
                radiance[branch, ih, iv] = values
            print(f'Direct transport: branch {branch}, height {ih + 1}/{len(heights)}', flush=True)
    return radiance, optical, axes
