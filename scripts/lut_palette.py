"""Fit scattering angles against the same half-float palette the shader samples."""

from hashlib import sha256
import json

import numpy as np

from lut_model import HERE, META, RGB


class Palette:
    def __init__(self, filename="palette.rgba16f"):
        self.payload = (HERE / filename).read_bytes()
        meta = META["palette"]
        self.cache_key = sha256(
            b"theta-rgb-original-column-targets-v1\0"
            + json.dumps(meta, sort_keys=True).encode()
            + self.payload
        ).hexdigest()[:16]
        color = meta["color"]
        self.width, self.height, layers = color["size"]
        self.pixels = np.frombuffer(self.payload, "<f2").reshape(
            layers, self.height, self.width, 4
        )[..., :3].astype("f8")
        # GLSL receives these uniforms as float32.
        self.scale = np.asarray(color["uvScale"], "f4").astype("f8")
        self.bias = np.asarray(color["uvBias"], "f4").astype("f8")
        self.min_radius = np.asarray(color["minRadius"], "f4").astype("f8")

    def fit_theta(self, species, xyz, radius, theta):
        """Keep radius fixed; project RGB/r onto the nearby palette segments."""
        shape = theta.shape
        radius = radius.ravel()
        theta = theta.ravel().copy()
        active = np.flatnonzero(radius > 0)
        if active.size == 0:
            return theta.reshape(shape)
        r = radius[active]
        target = (xyz.reshape(-1, 3)[active] / r[:, None]) @ RGB.T
        sx, sy = self.scale[species]
        bx, by = self.bias[species]
        x = np.clip((theta[active] * sx + bx) * self.width - 0.5, 0, self.width - 1)
        y = (np.log2(np.maximum(r, self.min_radius[species])) * sy + by) * self.height - 0.5
        y = np.clip(y, 0, self.height - 1)
        row = np.minimum(y.astype(int), self.height - 2)
        fy = (y - row)[:, None]
        pixels = self.pixels[species]

        def project(segment, samples):
            j, f = row[samples], fy[samples]
            a = (1 - f) * pixels[j, segment] + f * pixels[j + 1, segment]
            b = (1 - f) * pixels[j, segment + 1] + f * pixels[j + 1, segment + 1]
            delta = b - a
            offset = target[samples] - a
            norm = np.sum(delta * delta, axis=1)
            t = np.clip(np.sum(delta * offset, axis=1) / np.maximum(norm, 1e-30), 0, 1)
            t = np.where(norm > 0, t, np.clip(x[samples] - segment, 0, 1))
            residual = offset - t[:, None] * delta
            return segment + t, np.sum(residual * residual, axis=1)

        # At a fixed log-radius, bilinear filtering is piecewise linear in theta.
        # Start at the XZ angle and walk to lower-error neighboring segments.
        samples = np.arange(len(active))
        segment = np.minimum(x.astype(int), self.width - 2)
        best_x, best_error = project(segment, samples)
        while samples.size:
            left = np.maximum(segment - 1, 0)
            right = np.minimum(segment + 1, self.width - 2)
            lx, le = project(left, samples)
            rx, re = project(right, samples)
            go_left = le <= re
            error = np.where(go_left, le, re)
            improved = error < best_error[samples]
            if not improved.any():
                break
            samples = samples[improved]
            segment = np.where(go_left, left, right)[improved]
            best_x[samples] = np.where(go_left, lx, rx)[improved]
            best_error[samples] = error[improved]

        theta[active] = ((best_x + 0.5) / self.width - bx) / sx
        return theta.reshape(shape)
