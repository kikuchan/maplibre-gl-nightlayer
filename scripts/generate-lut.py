# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy>=1.15", "moderngl"]
# ///
"""Generate atmosphere-lut.zip from fixed physical inputs or resume its cache.

No existing LUT or historical work/ research scripts are inputs.
Run with uv. A desktop EGL/OpenGL 4.3 GPU with float64 shaders is required.
The full bake is expensive; --work-dir keeps restartable intermediates.
"""

import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent


def initialize():
    import numpy as np
    from numpy.polynomial.legendre import leggauss
    from lut_model import HERE

    # 128×128 path-normalized straight-ray tails, split at ozone density corners.
    n = 128
    heights = []
    for a, b in [(0, 10), (10, 25), (25, 40), (40, 100)]:
        intervals = n // 4 - int(a == 40)
        heights.extend((a + (b - a) * np.linspace(0, 1, intervals + 1))[:-1])
    heights.append(100.0)
    v = np.linspace(0, 1, n)
    mu = 0.2 * v / (1.2 - v)
    nodes, weights = leggauss(96)
    table = np.zeros((n, n, 4))
    for i, h in enumerate(heights[:-1]):
        t = (6371 + h) * mu
        cuts = [np.zeros(n)]
        for level in [10, 25, 40, 100]:
            delta = max((level - h) * (12742 + level + h), 0)
            cuts.append(delta / np.maximum(np.sqrt(t * t + delta) + t, 1e-30))
        for a, b in zip(cuts[:-1], cuts[1:]):
            s = (a[:, None] + b[:, None]) / 2 + (b - a)[:, None] * nodes / 2
            delta = s * (2 * t[:, None] + s)
            altitude = h + delta / (np.sqrt((6371 + h) ** 2 + delta) + 6371 + h)
            table[i, :, :2] += (
                np.sum(
                    np.exp(-(altitude[:, :, None] - h) / np.array([8.0, 1.2]))
                    * weights[None, :, None],
                    axis=1,
                )
                * (b - a)[:, None]
                / 2
            )
            table[i, :, 2] += (
                (np.maximum(0, 1 - abs(altitude - 25) / 15) @ weights) * (b - a) / 2
            )
        table[i, :, :2] = np.log(table[i, :, :2] / cuts[-1][:, None])
    (HERE / "density-integrals.rgba32f").write_bytes(table.astype("<f4").tobytes())


def build_optics():
    from lut_optics import (
        Geometry,
        build_geometry,
        build_solar,
        build_celestial,
    )

    gpu = Geometry()
    try:
        build_geometry(gpu)
    finally:
        gpu.context.release()
    build_solar()
    build_celestial()


def pack_archive(output):
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    from lut_model import BLUE, HERE, META
    from lut_optics import top_shell_limit
    from lut_palette import Palette

    meta = copy.deepcopy(META)
    configuration = json.loads((HERE / "sources-v2/config.json").read_text())
    source_configuration = json.dumps(configuration, sort_keys=True)
    from lut_pca import MIE_RADIUS_FIT, RADIUS_FIT, packed_factors

    palette = Palette()
    finite, mie_pair, layout = packed_factors(palette)
    meta["scattering"].update(layout)
    planes = []
    for p in range(16):
        reference = HERE / "reference" / f"p{p:02d}.npz"
        with np.load(reference) as z:
            if str(z["sourceConfig"]) != source_configuration:
                raise ValueError(f"Scattering source/reference mismatch at height {p}")
            xyz, nu = z["xyz"][..., -1, :, :], z["nu"]
        order = np.argsort(nu)
        nu, xyz = nu[order], xyz[order]
        use = nu > 24.239197702372184
        x, values = nu[use] / 63, xyz[use]
        radius = np.hypot(values[..., 0], values[..., 2])
        angle = BLUE - np.arctan2(values[..., 2], values[..., 0])
        for sp in range(2):
            angle[..., sp] = palette.fit_theta(
                sp, values[..., sp, :], radius[..., sp], angle[..., sp]
            )
        radius = np.maximum(radius, 1e-30)
        encoded = np.stack(
            [
                np.log(radius[..., 0]),
                angle[..., 0],
                np.log(radius[..., 1]),
                angle[..., 1],
            ],
            axis=-1,
        )
        x = np.r_[0, x]
        encoded = np.concatenate([encoded[:1], encoded], axis=0)
        planes.append(
            PchipInterpolator(x, encoded, axis=0)(np.linspace(0, 1, 32)).transpose(
                2, 1, 0, 3
            )
        )
    sky = np.stack(planes).astype("<f2")
    if not np.isfinite(sky).all():
        raise ArithmeticError("Nonfinite packed scattering values")
    # Keep the cached physical atlas untouched; apply the presentation limit once.
    physical = top_shell_limit(np.load(HERE / "refraction-runtime.npy"))
    irradiance = np.load(HERE / "irradiance.npy")
    meta["solarRadiance"] = np.load(HERE / "solar-radiance.npy").tolist()
    zenith = np.array(
        [np.interp(90, irradiance[0, :, 0], irradiance[0, :, i]) for i in range(1, 4)]
    )
    meta["skyReference"] = float(zenith @ [0.2126, 0.7152, 0.0722])
    meta["generation"] = dict(
        model="low-rank C2 tensor cubic B-spline in a smooth physical chart, log(r / species-column) and theta",
        physicalModel="spherical achromatic refraction; 48 wavelengths; scattering orders 1..8; finite solar disk",
        sourceConfiguration=configuration,
        finiteFit="all nodal references plus off-node random rays; alternating least squares on PCA factors",
        radiusFit=f"rayleigh {RADIUS_FIT}; mie {MIE_RADIUS_FIT}",
        thetaFit="fixed XZ radius; local linear RGB projection onto the packed palette",
        finitePath="straight display ray through refractive source fields",
        sky="PCHIP nu32 phi8 from physical curved sky rays; one-sided support extension",
        topShell="zero-path presentation limit before the tangent interval",
    )
    entries = {
        "metadata.json": json.dumps(meta, indent=2).encode(),
        "scattering-pca.f16": finite.tobytes(),
        "scattering-mie-pair.f32": mie_pair.tobytes(),
        "sky-scattering.f16": sky.tobytes(),
        "palette.rgba16f": palette.payload,
        "irradiance.f32": irradiance.astype("<f4").tobytes(),
        "density-integrals.rgba32f": (HERE / "density-integrals.rgba32f").read_bytes(),
        "transmittance.rgba32f": (HERE / "transmittance.rgba32f").read_bytes(),
        "refraction.rgba32f": physical.astype("<f4").tobytes(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(".pending.zip")
    with zipfile.ZipFile(
        pending, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    pending.replace(output)
    print("WROTE", output, output.stat().st_size, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "work/generate-lut",
        help="Intermediate cache directory (default: work/generate-lut); reused on subsequent runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "src/atmosphere-lut.zip",
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="CPU threads for the tensor fits"
    )
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "initialize",
            "optics",
            "sources",
            "references",
            "pca",
            "palette",
            "pack",
        ],
        default="all",
        help="Normally all; individual stages use this same work directory",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.environ["NIGHTLAYER_LUT_WORKDIR"] = str(work)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.threads))
    print("WORK", work, "\nOUTPUT", args.output.resolve(), flush=True)
    # Imports are deferred so --help does not install/import the numerical stack.
    import lut_scattering as scattering

    def sources():
        scattering.build_sources(
            SimpleNamespace(
                counts=[32, 128, 64, 32],
                angular=24,
                degree=16,
                steps=256,
                orders=8,
                batches=list(range(16)),
                output="sources-v2",
                replace=False,
            )
        )

    def references():
        scattering.build_curved_references()
        scattering.build_straight_references()

    def pca():
        from lut_pca import build_pca

        build_pca()

    def palette():
        scattering.build_palette()
        scattering.build_irradiance()

    stages = {
        "initialize": initialize,
        "optics": build_optics,
        "sources": sources,
        "references": references,
        "palette": palette,
        "pca": pca,
        "pack": lambda: pack_archive(args.output.resolve()),
    }
    for name, run in stages.items():
        if args.stage not in ["all", name]:
            continue
        done = work / f"{name}.done"
        # PCA checks its own factors against the current palette.
        if args.stage == "all" and name not in ("pca", "pack") and done.exists():
            print("RESUME", name, flush=True)
            continue
        print("STAGE", name, flush=True)
        run()
        done.touch()


if __name__ == "__main__":
    main()
