# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.4", "colour-science==0.4.7", "pvlib==0.16.1",
#                 "moderngl==5.12.0", "PyOpenGL==3.1.10"]
# ///
"""Offline spherical spectral multiple scattering, driven by Python/EGL.

The OpenGL adapter integrates spectral local scattering sources into linear RGB.
The CPU builds direct camera-to-endpoint transport, without display baking.
Direct ground illuminance is integrated spectrally here as well.
"""

import argparse
import hashlib
import importlib
import io
import os
import re
import tarfile
import urllib.request
from pathlib import Path

import colour
import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parent.parent
REVISION = "34f14e745cff948f4ca3157d1b62a445ffa7286f"
ARCHIVE_SHA256 = "a514554dcefb39cdff6e4448bf3fc169402357a2445ea05a51a9ee84b49b1d06"
MODEL = {
    "planetRadius": 6371000,
    "atmosphereHeight": 100000,
    "observerHeight": 1,
    "rayleighScaleHeight": 8000,
    "rayleighOpticalDepth": "0.008735 * wavelengthMicrometers ** -4.08 (SPECTRL2)",
    "aerosolOpticalDepth": 0.0252,
    "aerosolScaleHeight": 1200,
    "aerosolSingleScatteringAlbedo": 1,
    "aerosolPhaseParameter": 0.758,
    "ozoneDobsonUnits": 300,
    "ozoneProfile": "Triangular: zero at 10/40 km, peak at 25 km; integral 15 km",
    "surfaceAlbedo": 0,
    "sunAngularRadiusDegrees": 0.2666667,
    "refraction": False,
    "solarParallax": False,
}


def load_sources(cache, resolution):
    source = cache / f"precomputed_atmospheric_scattering-{REVISION}"
    if not source.exists():
        url = f"https://codeload.github.com/ebruneton/precomputed_atmospheric_scattering/tar.gz/{REVISION}"
        archive = urllib.request.urlopen(url, timeout=120).read()
        if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Upstream archive checksum mismatch")
        cache.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(cache, filter="data")
    sizes = {
        "TRANSMITTANCE_TEXTURE_WIDTH": 256 * resolution,
        "TRANSMITTANCE_TEXTURE_HEIGHT": 64 * resolution,
        "SCATTERING_TEXTURE_R_SIZE": 32 * resolution,
        "SCATTERING_TEXTURE_MU_SIZE": 128,
        "SCATTERING_TEXTURE_MU_S_SIZE": 32 * resolution,
        "SCATTERING_TEXTURE_NU_SIZE": 8,
        "IRRADIANCE_TEXTURE_WIDTH": 1024 * resolution,
        "IRRADIANCE_TEXTURE_HEIGHT": 16 * resolution,
    }
    return source / "atmosphere", sizes


def create_context():
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import moderngl

    context = moderngl.create_context(standalone=True, backend="egl", require=330)
    print(f"Renderer: {context.info['GL_RENDERER']}", flush=True)
    return context


def diffuse(context, source, sizes, altitudes, wavelength_count, orders):
    """Port of upstream Model::Init/Precompute; numerical GLSL stays unchanged.

    The BSD-3-Clause attribution is in THIRD-PARTY-NOTICES. Preserve upstream's
    six-decimal atmosphere constants, float32 textures and spectral quadrature.
    """
    from OpenGL import GL

    definitions = (source / "definitions.glsl").read_text()
    functions = (source / "functions.glsl").read_text()
    model = (source / "model.cc").read_text()
    shaders = dict(
        re.findall(r'const char (k\w+)\[\] = R"\((.*?)\)";', model, re.DOTALL)
    )
    constants = (source / "constants.h").read_text()
    cie_text = re.search(
        r"CIE_2_DEG_COLOR_MATCHING_FUNCTIONS\[380\] = \{(.*?)\};", constants, re.DOTALL
    )[1]
    cie = np.array([float(v) for v in cie_text.split(",") if v.strip()]).reshape(-1, 4)
    vertex = """#version 330
      void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0, 1);
      }"""
    prefix = "#version 330\n#define IN(x) const in x\n#define OUT(x) out x\n#define TEMPLATE(x)\n#define TEMPLATE_ARGUMENT(x)\n#define assert(x)\n"
    prefix += (
        "".join(f"const int {key} = {value};\n" for key, value in sizes.items())
        + definitions
    )

    def vector(values):
        return "vec3(" + ",".join(f"{v:.6f}" for v in values) + ")"

    def header(lambdas):
        def spectrum(values, scale=1):
            return vector(np.interp(lambdas, wavelengths, values) * scale)

        def density(height):
            return f"DensityProfile(DensityProfileLayer[2](DensityProfileLayer(0.,0.,0.,0.,0.),DensityProfileLayer(0.,1.,{-1000 / height:.6f},0.,0.)))"

        aerosol = MODEL["aerosolOpticalDepth"] / MODEL["aerosolScaleHeight"]
        parameters = [
            spectrum(solar),
            f"{np.deg2rad(MODEL['sunAngularRadiusDegrees']):.6f}",
            f"{MODEL['planetRadius'] / 1000:.6f}",
            f"{(MODEL['planetRadius'] + MODEL['atmosphereHeight']) / 1000:.6f}",
            density(MODEL["rayleighScaleHeight"]),
            spectrum(rayleigh_tau, 1000 / MODEL["rayleighScaleHeight"]),
            density(MODEL["aerosolScaleHeight"]),
            vector([aerosol * MODEL["aerosolSingleScatteringAlbedo"] * 1000] * 3),
            vector([aerosol * 1000] * 3),
            f"{MODEL['aerosolPhaseParameter']:.6f}",
            "DensityProfile(DensityProfileLayer[2](DensityProfileLayer(25.,0.,0.,0.066667,-0.666667),DensityProfileLayer(0.,0.,0.,-0.066667,2.666667)))",
            spectrum(ozone_tau, 1000 / 15000),
            "vec3(0.)",
            "-0.500000",
        ]
        return (
            prefix
            + "\nconst AtmosphereParameters ATMOSPHERE = AtmosphereParameters("
            + ",".join(parameters)
            + ");\n"
            + functions
        )

    textures, programs, vaos = [], [], []
    framebuffer = GL.glGenFramebuffers(1)

    def texture(size):
        target = GL.GL_TEXTURE_3D if len(size) == 3 else GL.GL_TEXTURE_2D
        handle = GL.glGenTextures(1)
        textures.append(handle)
        GL.glBindTexture(target, handle)
        for key in [GL.GL_TEXTURE_MIN_FILTER, GL.GL_TEXTURE_MAG_FILTER]:
            GL.glTexParameteri(target, key, GL.GL_LINEAR)
        for key in [GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T, GL.GL_TEXTURE_WRAP_R]:
            GL.glTexParameteri(target, key, GL.GL_CLAMP_TO_EDGE)
        if len(size) == 3:
            GL.glTexImage3D(
                target, 0, GL.GL_RGBA32F, *size, 0, GL.GL_RGBA, GL.GL_FLOAT, None
            )
        else:
            GL.glTexImage2D(
                target, 0, GL.GL_RGBA32F, *size, 0, GL.GL_RGBA, GL.GL_FLOAT, None
            )
        return handle, target

    trans_size = (
        sizes["TRANSMITTANCE_TEXTURE_WIDTH"],
        sizes["TRANSMITTANCE_TEXTURE_HEIGHT"],
    )
    irrad_size = (sizes["IRRADIANCE_TEXTURE_WIDTH"], sizes["IRRADIANCE_TEXTURE_HEIGHT"])
    scatter_size = (
        sizes["SCATTERING_TEXTURE_NU_SIZE"] * sizes["SCATTERING_TEXTURE_MU_S_SIZE"],
        sizes["SCATTERING_TEXTURE_MU_SIZE"],
        sizes["SCATTERING_TEXTURE_R_SIZE"],
    )
    try:
        trans, irradiance, delta_irrad = (
            texture(trans_size),
            texture(irrad_size),
            texture(irrad_size),
        )
        scattering, mie, rayleigh, delta_mie, density = [
            texture(scatter_size) for _ in range(5)
        ]
        density_sum = texture(scatter_size)
        source_size = (scatter_size[0], 32, scatter_size[2])
        local_source = texture(source_size)
        direct_size = (sizes['SCATTERING_TEXTURE_MU_S_SIZE'], scatter_size[2])
        source_rayleigh = texture(direct_size)

        def compile_program(text, geometry=False):
            program = context.program(
                vertex_shader=vertex,
                fragment_shader=text,
                **({"geometry_shader": shaders["kGeometryShader"]} if geometry else {}),
            )
            vao = context.vertex_array(program, [])
            programs.append(program)
            vaos.append(vao)
            return program, vao

        def draw(program_vao, outputs, size, inputs=(), uniforms=None, blend=()):
            program, vao = program_vao
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
            for i in range(4):
                GL.glFramebufferTexture(
                    GL.GL_FRAMEBUFFER,
                    GL.GL_COLOR_ATTACHMENT0 + i,
                    outputs[i][0] if i < len(outputs) else 0,
                    0,
                )
            GL.glDrawBuffers(
                len(outputs), [GL.GL_COLOR_ATTACHMENT0 + i for i in range(len(outputs))]
            )
            if (
                GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
                != GL.GL_FRAMEBUFFER_COMPLETE
            ):
                raise RuntimeError("Incomplete atmosphere framebuffer")
            GL.glViewport(0, 0, *size[:2])
            GL.glUseProgram(program.glo)
            GL.glBindVertexArray(vao.glo)
            for unit, (name, tex) in enumerate(inputs):
                GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                GL.glBindTexture(tex[1], tex[0])
                if name in program:
                    program[name].value = unit
            for name, value in (uniforms or {}).items():
                if name in program:
                    if isinstance(value, np.ndarray):
                        program[name].write(value.astype("f4").T.tobytes())
                    else:
                        program[name].value = value
            GL.glBlendEquation(GL.GL_FUNC_ADD)
            GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
            for i in range(4):
                (GL.glEnablei if i < len(blend) and blend[i] else GL.glDisablei)(
                    GL.GL_BLEND, i
                )
            for layer in range(size[2] if len(size) == 3 else 1):
                if "layer" in program:
                    program["layer"].value = layer
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
                # Bound driver batches, as in the former C++ adapter.
                GL.glFinish()

        iterations = wavelength_count // 3
        dlambda = (830 - 360) / wavelength_count
        spectral = {'multiple': [], 'beam': [], 'wavelengths': [], 'rgb_weights': []}
        for iteration in range(iterations):
            lambdas = 360 + (3 * iteration + np.array([0.5, 1.5, 2.5])) * dlambda
            cmf = np.array(
                [np.interp(lambdas, cie[:, 0], cie[:, i]) for i in range(1, 4)]
            )
            matrix = (XYZ_TO_RGB @ cmf * dlambda).astype("f4")
            spectral['wavelengths'].extend(lambdas.tolist())
            spectral['rgb_weights'].extend(matrix.T.tolist())
            h = header(lambdas)
            density_shader = shaders['kComputeScatteringDensityShader'].replace(
                'layout(location = 0) out vec3 scattering_density;',
                'layout(location = 0) out vec3 scattering_density;\n'
                'layout(location = 1) out vec3 source_density;\n'
                'uniform mat3 luminance_from_radiance;'
            ).replace('scattering_order);',
                       'scattering_order);\nsource_density = scattering_density;')
            steps = [
                compile_program(h + (density_shader if name == 'ScatteringDensity'
                                     else shaders[f"kCompute{name}Shader"]), geometry)
                for name, geometry in [
                    ("Transmittance", False),
                    ("DirectIrradiance", False),
                    ("SingleScattering", True),
                    ("ScatteringDensity", True),
                    ("IndirectIrradiance", False),
                    ("MultipleScattering", True),
                ]
            ]
            accumulated = iteration > 0
            draw(steps[0], [trans], trans_size)
            draw(
                steps[1],
                [delta_irrad, irradiance],
                irrad_size,
                [("transmittance_texture", trans)],
                blend=[False, accumulated],
            )
            draw(
                steps[2],
                [rayleigh, delta_mie, scattering, mie],
                scatter_size,
                [("transmittance_texture", trans)],
                {"luminance_from_radiance": matrix},
                [False, False, accumulated, accumulated],
            )
            for order in range(2, orders + 1):
                draw(
                    steps[3],
                    [density, density_sum],
                    scatter_size,
                    [
                        ("transmittance_texture", trans),
                        ("single_rayleigh_scattering_texture", rayleigh),
                        ("single_mie_scattering_texture", delta_mie),
                        ("multiple_scattering_texture", rayleigh),
                        ("irradiance_texture", delta_irrad),
                    ],
                    {"scattering_order": order, 'luminance_from_radiance': matrix},
                    [False, order > 2],
                )
                draw(
                    steps[4],
                    [delta_irrad, irradiance],
                    irrad_size,
                    [
                        ("single_rayleigh_scattering_texture", rayleigh),
                        ("single_mie_scattering_texture", delta_mie),
                        ("multiple_scattering_texture", rayleigh),
                    ],
                    {"scattering_order": order - 1, "luminance_from_radiance": matrix},
                    [False, True],
                )
                draw(
                    steps[5],
                    [rayleigh, scattering],
                    scatter_size,
                    [
                        ("transmittance_texture", trans),
                        ("scattering_density_texture", density),
                    ],
                    {"luminance_from_radiance": matrix},
                    [False, True],
                )
            local = compile_program(h + """
              uniform sampler3D density_texture;
              uniform int layer;
              layout(location=0) out vec3 value;
              void main() {
                float sun_index = mod(gl_FragCoord.x - 0.5, float(SCATTERING_TEXTURE_MU_S_SIZE));
                float nu_index = floor(gl_FragCoord.x / float(SCATTERING_TEXTURE_MU_S_SIZE));
                vec4 uvwz = vec4(nu_index / float(SCATTERING_TEXTURE_NU_SIZE - 1),
                  (sun_index + 0.5) / float(SCATTERING_TEXTURE_MU_S_SIZE), 0.75,
                  (float(layer) + 0.5) / float(SCATTERING_TEXTURE_R_SIZE));
                float r, unused_mu, mu_s, nu; bool unused_ground;
                GetRMuMuSNuFromScatteringTextureUvwz(ATMOSPHERE, uvwz,
                  r, unused_mu, mu_s, nu, unused_ground);
                float mu = 2.0 * (gl_FragCoord.y - 0.5) / 31.0 - 1.0;
                value = GetScattering(ATMOSPHERE, density_texture, r, mu, mu_s,
                  uvwz.x * 2.0 - 1.0, RayIntersectsGround(ATMOSPHERE, r, mu)) /
                  GetProfileDensity(ATMOSPHERE.rayleigh_density, r - ATMOSPHERE.bottom_radius);
              }""", True)
            draw(local, [local_source], source_size, [('density_texture', density_sum)])
            values = np.empty(tuple(reversed(source_size)) + (4,), dtype='f4')
            GL.glBindTexture(GL.GL_TEXTURE_3D, local_source[0])
            GL.glGetTexImage(GL.GL_TEXTURE_3D, 0, GL.GL_RGBA, GL.GL_FLOAT, values)
            spectral['multiple'].append(values[..., :3].copy())
            beam = compile_program(h + """
              uniform sampler2D transmittance_texture;
              layout(location=0) out vec3 value;
              void main() {
                vec4 uvwz = vec4(0.0, gl_FragCoord.x / float(SCATTERING_TEXTURE_MU_S_SIZE),
                  0.75, gl_FragCoord.y / float(SCATTERING_TEXTURE_R_SIZE));
                float r, mu, mu_s, nu; bool ground;
                GetRMuMuSNuFromScatteringTextureUvwz(ATMOSPHERE, uvwz, r, mu, mu_s, nu, ground);
                value = ATMOSPHERE.solar_irradiance * GetTransmittanceToSun(
                  ATMOSPHERE, transmittance_texture, r, mu_s);
              }""")
            draw(beam, [source_rayleigh], direct_size, [('transmittance_texture', trans)])
            values = np.empty(tuple(reversed(direct_size)) + (4,), dtype='f4')
            GL.glBindTexture(GL.GL_TEXTURE_2D, source_rayleigh[0])
            GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_FLOAT, values)
            spectral['beam'].append(values[..., :3].copy())
            for program, vao in steps:
                vao.release()
                program.release()
                vaos.remove(vao)
                programs.remove(program)
            print(f"Spectral batch {iteration + 1}/{iterations}", flush=True)


        sample = compile_program(
            header([680, 550, 440])
            + """
          uniform sampler2D irradiance_texture;
          uniform float altitude;
          layout(location=0) out vec4 value;
          void main() {
            value = vec4(GetIrradiance(ATMOSPHERE, irradiance_texture,
              ATMOSPHERE.bottom_radius + 0.001, sin(radians(altitude))) * 683.0, 1.0);
          }"""
        )
        output = texture((1, 1))
        result = []
        for altitude in altitudes:
            draw(
                sample,
                [output],
                (1, 1),
                [("irradiance_texture", irradiance)],
                {"altitude": float(altitude)},
            )
            pixel = np.empty(4, dtype="f4")
            GL.glReadPixels(0, 0, 1, 1, GL.GL_RGBA, GL.GL_FLOAT, pixel)
            result.append(pixel[:3].copy())
        spectral['multiple'] = np.concatenate(spectral['multiple'], axis=1)
        spectral['beam'] = np.concatenate(spectral['beam'], axis=0)
        return np.array(result), spectral
    finally:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glDeleteFramebuffers(1, [framebuffer])
        if textures:
            GL.glDeleteTextures(textures)
        for vao in vaos:
            vao.release()
        for program in programs:
            program.release()


coefficients = importlib.import_module("pvlib.spectrum.spectrl2")._SPECTRL2_COEFFS
wavelengths = np.arange(360, 831)
solar = np.interp(
    wavelengths, coefficients["wavelength"], coefficients["spectral_irradiance_et"]
)
ozone_tau = np.interp(
    wavelengths, coefficients["wavelength"], coefficients["ozone_absorption"]
) * (MODEL["ozoneDobsonUnits"] / 1000)
rayleigh_tau = 0.008735 * (wavelengths / 1000) ** -4.08
observer = (
    colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    .copy()
    .align(colour.SpectralShape(360, 830, 1))
)


def direct(altitude, normal=False):
    # A uniformly bright solar disk, integrated in solid angle. No refraction
    # or parallax is applied to the geometric solar-centre altitude.
    nodes, weights = leggauss(12)
    radius = np.deg2rad(MODEL["sunAngularRadiusDegrees"])
    cos_delta = 1 - (nodes + 1) / 2 * (1 - np.cos(radius))
    phi = (np.arange(24) + 0.5) * 2 * np.pi / 24
    angle = np.deg2rad(altitude)
    mu = (
        np.sin(angle) * cos_delta[:, None]
        + np.cos(angle) * np.sqrt(1 - cos_delta[:, None] ** 2) * np.cos(phi)
    ).ravel()
    disk_weights = np.broadcast_to(weights[:, None] / 48, (12, 24)).ravel()
    visible = mu > 0  # horizontal receiver; rays below its plane contribute zero
    mu, disk_weights = mu[visible], disk_weights[visible]
    if not len(mu):
        return np.zeros(3)
    r = MODEL["planetRadius"] + MODEL["observerHeight"]
    optical = np.zeros((len(mu), 3))
    nodes, weights = leggauss(32)
    start = np.zeros_like(mu)
    # Split at ozone profile corners; each segment is smooth.
    for height in [10000, 25000, 40000, MODEL["atmosphereHeight"]]:
        end = -r * mu + np.sqrt(
            (r * mu) ** 2 + (MODEL["planetRadius"] + height) ** 2 - r**2
        )
        t = start[:, None] + (end - start)[:, None] * (nodes + 1) / 2
        h = np.sqrt(r**2 + 2 * r * mu[:, None] * t + t**2) - MODEL["planetRadius"]
        ds = (end - start)[:, None] * weights / 2
        optical[:, 0] += (
            np.sum(ds * np.exp(-h / MODEL["rayleighScaleHeight"]), axis=1)
            / MODEL["rayleighScaleHeight"]
        )
        optical[:, 1] += (
            np.sum(ds * np.exp(-h / MODEL["aerosolScaleHeight"]), axis=1)
            / MODEL["aerosolScaleHeight"]
        )
        optical[:, 2] += (
            np.sum(ds * np.maximum(0, 1 - np.abs(h - 25000) / 15000), axis=1) / 15000
        )
        start = end
    transmission = np.exp(
        -optical[:, :1] * rayleigh_tau
        - optical[:, 1:2] * MODEL["aerosolOpticalDepth"]
        - optical[:, 2:] * ozone_tau
    )
    spectrum = solar * ((disk_weights * (1 if normal else mu)) @ transmission)
    xyz = np.trapezoid(spectrum[:, None] * observer.values, wavelengths, axis=0) * 683
    # Use the same matrix as upstream, avoiding mismatched RGB/XYZ conventions.
    return XYZ_TO_RGB @ xyz


XYZ_TO_RGB = np.array(
    [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]
)


def adaptive_samples(angles, rgb, max_error, relative=False):
    """Split at the largest RGB interpolation error, not only the midpoint."""
    selected = {0, len(angles) - 1}
    pending = [(0, len(angles) - 1)]
    while pending:
        low, high = pending.pop()
        if high - low < 2:
            continue
        fraction = (angles[low + 1 : high] - angles[low]) / (angles[high] - angles[low])
        interpolated = rgb[low] + fraction[:, None] * (rgb[high] - rgb[low])
        difference = np.abs(rgb[low + 1 : high] - interpolated)
        if relative:
            difference /= np.maximum(rgb[low + 1 : high], 1e-6)
        errors = np.max(difference, axis=1)
        worst = int(np.argmax(errors))
        if errors[worst] > max_error * 0.9:
            middle = low + 1 + worst
            selected.add(middle)
            pending.extend([(low, middle), (middle, high)])
    return np.column_stack((angles, rgb))[sorted(selected)]


def surface_irradiance(angles, sky, sun, args):
    values = np.maximum(0, np.column_stack((sky, sun)))
    knots = adaptive_samples(angles, values, args.max_error / 2, relative=True)[:, 0]
    sampled = np.column_stack([np.interp(knots, angles, values[:, i]) for i in range(6)])
    return np.column_stack((knots, sampled)).astype(np.float32)


def atmosphere_sources(spectral, transform):
    lambdas = np.array(spectral['wavelengths'])
    beta_r = np.interp(lambdas, wavelengths, rayleigh_tau) * 1000 / MODEL['rayleighScaleHeight']
    beta_m = np.full(len(lambdas), MODEL['aerosolOpticalDepth'] * 1000 / MODEL['aerosolScaleHeight'])
    beta_o = np.interp(lambdas, wavelengths, ozone_tau) * 1000 / 15000
    weights = np.array(spectral['rgb_weights']) @ transform.T * (683 * np.pi)
    batches = len(lambdas) // 3
    # Integrate local radiance sources before storage, retaining the spectral solver.
    multiple = sum(spectral['multiple'][:, i * 32:(i + 1) * 32] @ weights[i * 3:(i + 1) * 3]
                   for i in range(batches))
    rows = spectral['beam'].shape[0] // batches
    beam = spectral['beam'].reshape(batches, rows, -1, 3)
    # Last row/column is top-of-atmosphere, sun-up irradiance.
    solar_irradiance = sum(beam[i, -1, -1] @ weights[i * 3:(i + 1) * 3]
                           for i in range(batches))
    direct_r = sum((beam[i] * beta_r[i * 3:(i + 1) * 3]) @ weights[i * 3:(i + 1) * 3]
                   for i in range(batches))
    direct_m = sum((beam[i] * (beta_m[i * 3:(i + 1) * 3] * MODEL['aerosolSingleScatteringAlbedo']))
                   @ weights[i * 3:(i + 1) * 3] for i in range(batches))
    response = np.abs(weights)
    response /= response.sum(axis=0)
    extinction = lambda beta: (beta @ response).tolist()
    beam = np.concatenate((direct_r, direct_m), axis=0)
    extinction_rgb = np.array([extinction(beta_r), extinction(beta_m), extinction(beta_o)])
    return multiple, beam, extinction_rgb, solar_irradiance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("./nightlayer-bruneton"))
    parser.add_argument("--wavelengths", type=int, default=48)
    parser.add_argument("--orders", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[1, 2], default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "src/lighting-lut.zip")
    parser.add_argument('--transport-size', type=int, nargs=5, default=[9, 9, 13, 13, 5], help='Height, view, distance, sun and azimuth knots')
    parser.add_argument('--integration-cells', type=int, default=256)
    parser.add_argument(
        "--probe", action="store_true", help="Test Python/EGL without precomputing"
    )
    parser.add_argument("--max-error", type=float, default=1 / 512)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate and compare the stored ZIP",
    )
    args = parser.parse_args()
    if any(not 3 <= n <= 256 for n in args.transport_size) or args.integration_cells < 16:
        parser.error('Use 3..256 transport knots and at least 16 integration cells')
    if not np.isfinite(args.max_error) or args.max_error <= 0:
        parser.error('max-error must be finite and positive')
    context = create_context()
    if args.probe:
        context.release()
        return
    if not 6 <= args.wavelengths <= 96 or args.wavelengths % 3 or args.orders < 2:
        parser.error(
            "Use 6..96 wavelengths (a multiple of three), and at least two orders"
        )
    source, sizes = load_sources(args.cache, args.resolution)
    altitudes = np.concatenate(
        (np.arange(-24, 10, 0.03125), np.arange(10, 90.25, 0.25))
    )
    try:
        sky, spectral = diffuse(context, source, sizes, altitudes, args.wavelengths, args.orders)
    finally:
        context.release()
    sun = np.array([direct(altitude) for altitude in altitudes])
    total = sun + sky
    if not np.all(np.isfinite(total)):
        raise ValueError("Nonfinite irradiance")
    reference_xyz = np.linalg.solve(XYZ_TO_RGB, total[-1])
    d65 = colour.xy_to_XYZ(
        colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    )
    adaptation = colour.adaptation.matrix_chromatic_adaptation_VonKries(
        reference_xyz / reference_xyz[1], d65, transform="Bradford"
    )
    transform = XYZ_TO_RGB @ adaptation @ np.linalg.inv(XYZ_TO_RGB) / reference_xyz[1]
    # The rounded upstream RGB matrix does not map D65 to exactly (1,1,1).
    transform = transform / (transform @ total[-1])[:, None]
    direct_normal = np.array([direct(altitude, normal=True) for altitude in altitudes]) @ transform.T
    irradiance = surface_irradiance(altitudes, sky @ transform.T, direct_normal, args)
    from atmosphere_binary import lighting_zip
    from atmosphere_direct import build, solar_boundary
    multiple, beam, extinction, solar_irradiance = atmosphere_sources(spectral, transform)
    radiance, optical, _ = build(multiple, beam, extinction, tuple(args.transport_size), args.integration_cells)
    solar_radiance, solar_radius = solar_boundary(solar_irradiance, MODEL['sunAngularRadiusDegrees'])
    parameters = {name: getattr(args, name) for name in
                  ['wavelengths', 'orders', 'resolution', 'transport_size', 'integration_cells', 'max_error']}
    metadata = {'gridSize': args.transport_size, 'solarRadiance': solar_radiance.tolist(),
                'solarAngularRadius': solar_radius, 'mieG': MODEL['aerosolPhaseParameter'],
                'skyReference': float((sky[-1] @ transform.T) @ [0.2126, 0.7152, 0.0722]),
                'generation': {'parameters': parameters, 'model': MODEL}}
    archive = lighting_zip(radiance, optical, irradiance, metadata)
    if args.check:
        if args.output.read_bytes() != archive:
            raise ValueError('LUT ZIP does not match regenerated data')
    else:
        args.output.write_bytes(archive)
    print(
        f"{'PASS' if args.check else 'Wrote'}: {args.output} ({args.output.stat().st_size} bytes)",
        flush=True,
    )


if __name__ == "__main__":
    main()
