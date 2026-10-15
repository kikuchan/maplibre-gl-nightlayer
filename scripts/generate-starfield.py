# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.3", "OpenEXR==3.4.15", "Pillow==12.3.0",
#                 "colour-science==0.4.7", "pvlib==0.16.1"]
# ///
"""Build a photometric point-star catalogue and a separately calibrated Milky Way.

HYG v4.1 (David Nash/Astronexus), derived data CC BY-SA 4.0.
NASA Deep Star Maps 2020: NASA/GSFC/SVS; ESA/Gaia/DPAC.
Johnson V energy response: SVO Generic/Johnson.V.
The code is MIT; the derived HYG dataset retains its own licence.
"""

import argparse
import csv
import gzip
import hashlib
import importlib
import struct
import urllib.request
from pathlib import Path

import colour
import numpy as np
import OpenEXR
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
NASA = "https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/"
HYG = "https://raw.githubusercontent.com/astronexus/HYG-Database/c7f7f883fe678cc7680169a50ccd7dcc49b060ce/hyg/CURRENT/"
SOURCES = {
    "hygdata_v41.csv": (HYG, "d9f69fd86bbf90a4e4d52b4c5c53eacfa6dfc0bfdef85bfd94f095e0bebe4ebd"),
    "starmap_2020_4k.exr": (NASA, "69b841cd048c2ef35543eb1c5819b322530e78aa4b0cab97ec1de3ef56a4ddcd"),
    "milkyway_2020_4k.exr": (NASA, "2eb802d6e68d170b410f766c7fec07f7518619f6b6708fdc81e9302d93e74fdb"),
}
MAGNITUDE_LIMIT = 6.5
SUN_V = -26.74
Y = np.array([0.2126, 0.7152, 0.0722])
XYZ_TO_RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])


def read_exr(path):
    with OpenEXR.File(str(path)) as image:
        pixels = image.channels()["RGB"].pixels.astype(np.float32)
    if pixels.shape != (2048, 4096, 3) or not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError(f"Unexpected NASA EXR data: {path}")
    return pixels


def photometry(rows):
    """Blackbody colour and Johnson-V -> CIE-Y correction, relative to the Sun.

    B-V is treated as an effective temperature, not a full stellar spectrum.
    The same SPECTRL2 extraterrestrial solar spectrum as the atmosphere is used.
    """
    wavelength = np.arange(360, 831)
    cmf = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(colour.SpectralShape(360, 830, 1)).values
    # SVO Generic/Johnson.V: energy-counting response, 20 nm spacing, 460..740 nm.
    response = np.interp(wavelength, np.arange(460, 741, 20),
                         [0, .02, .38, .91, .98, .72, .62, .4, .2, .08, .02, .01, .01, .01, 0], left=0, right=0)
    solar = importlib.import_module("pvlib.spectrum.spectrl2")._SPECTRL2_COEFFS
    solar = np.interp(wavelength, solar["wavelength"], solar["spectral_irradiance_et"])
    solar_y_per_v = np.trapezoid(solar * cmf[:, 1]) / np.trapezoid(solar * response)
    ci = np.array([float(row["ci"]) if row["ci"] else .65 for row in rows])
    ci = np.clip(ci, -.4, 2.0)
    table_ci = np.linspace(-.4, 2.0, 481)
    temperature = 4600 * (1 / (.92 * table_ci + 1.7) + 1 / (.92 * table_ci + .62))
    meters = wavelength * 1e-9
    spectrum = 1 / (meters[None, :] ** 5 * np.expm1(.01438776877 / (temperature[:, None] * meters)))
    xyz = np.trapezoid(spectrum[:, :, None] * cmf[None, :, :], axis=1)
    v_flux = np.trapezoid(spectrum * response[None, :], axis=1)
    correction = np.interp(ci, table_ci, xyz[:, 1] / v_flux / solar_y_per_v)
    rgb = np.maximum(xyz @ XYZ_TO_RGB.T, 0)
    rgb /= (rgb @ Y)[:, None]
    magnitude = np.array([float(row["mag"]) for row in rows])
    relative_y = 10 ** (-.4 * (magnitude - SUN_V)) * correction
    rgb = np.stack([np.interp(ci, table_ci, rgb[:, channel]) for channel in range(3)], axis=1)
    return rgb * relative_y[:, None], relative_y


def background_scale(combined, milkyway, rows, directions, relative_y):
    """Use isolated, unsaturated stars to estimate the diffuse image's zero point.

    This is empirical image photometry, not a claim of calibrated NASA pixels.
    Half-float/PSF/filter differences remain; report the scatter explicitly.
    """
    full_y, mw_y = combined @ Y, milkyway @ Y
    # A low quantile rejects positive stellar contributions. These two NASA
    # products use different normalizations (the measured ratio is about 0.5).
    mask = (mw_y > .003) & (mw_y < .3)
    background_ratio = np.quantile(full_y[mask] / mw_y[mask], .2)
    foreground = full_y - mw_y * background_ratio
    tree = cKDTree(directions)
    scales = []
    for i, row in enumerate(rows):
        magnitude, dec = float(row["mag"]), float(row["dec"])
        if not (4.5 <= magnitude <= 5.5 and abs(dec) < 60):
            continue
        if len(tree.query_ball_point(directions[i], 2 * np.sin(np.deg2rad(.5) / 2))) != 1:
            continue
        x = ((.5 - float(row["ra"]) / 24) % 1) * 4096 - .5
        y = (.5 - dec / 180) * 2048 - .5
        xs, ys = np.arange(round(x) - 7, round(x) + 8), np.arange(round(y) - 7, round(y) + 8)
        radius2 = ((xs - x)[None, :] * np.cos(np.deg2rad(dec))) ** 2 + (ys - y)[:, None] ** 2
        aperture, annulus = radius2 < 9, (radius2 > 16) & (radius2 < 36)
        values = foreground[ys[:, None], xs[None, :] % 4096]
        if full_y[ys[:, None], xs[None, :] % 4096][aperture].max() > .8:
            continue
        solid_angle = (2 * np.pi / 4096) * (np.pi / 2048) * np.cos(np.deg2rad(dec))
        flux = (values - np.median(values[annulus]))[aperture].sum() * solid_angle
        if flux > 0:
            scales.append(relative_y[i] / flux)
    if len(scales) < 50:
        raise ValueError("Not enough unsaturated photometric reference stars")
    scale = np.median(scales)
    scatter = np.quantile(2.5 * np.log10(np.array(scales) / scale), [.16, .84])
    print(f"Background calibration: {len(scales)} stars, image ratio={background_ratio:.6g}, "
          f"scale={scale * background_ratio:.9g} / sr per solar Y irradiance, 16..84% scatter={scatter} mag")
    return float(scale * background_ratio)


def catalog_bytes(rows, directions, flux, background):
    ra = np.deg2rad(np.array([float(row["ra"]) * 15 for row in rows]))
    dec = np.deg2rad(np.array([float(row["dec"]) for row in rows]))
    east = np.stack([-np.sin(ra), np.cos(ra), np.zeros_like(ra)], axis=1)
    north = np.stack([-np.cos(ra) * np.sin(dec), -np.sin(ra) * np.sin(dec), np.cos(dec)], axis=1)
    pmra = np.array([float(row["pmra"] or 0) for row in rows])
    pmdec = np.array([float(row["pmdec"] or 0) for row in rows])
    motion = (east * pmra[:, None] + north * pmdec[:, None]) * np.deg2rad(1 / 3600000)
    records = np.concatenate([directions, flux, motion], axis=1).astype('<f4')
    print(f"Catalogue: {len(rows)} stars through V={MAGNITUDE_LIMIT}")
    return struct.pack('<IIfI', 0x53545054, len(rows), background, 0) + records.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    for name, (base, checksum) in SOURCES.items():
        path = args.cache / name
        if not path.exists():
            urllib.request.urlretrieve(base + name, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Unexpected source checksum: {name}')
    # Include fainter neighbours in the isolation test, not just the drawn stars.
    with (args.cache / 'hygdata_v41.csv').open() as file:
        rows = [row for row in csv.DictReader(file) if row['id'] != '0' and float(row['mag']) <= 9]
    ra = np.deg2rad(np.array([float(row['ra']) * 15 for row in rows]))
    dec = np.deg2rad(np.array([float(row['dec']) for row in rows]))
    directions = np.stack([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)], axis=1)
    flux, relative_y = photometry(rows)
    combined = read_exr(args.cache / 'starmap_2020_4k.exr')
    milkyway = read_exr(args.cache / 'milkyway_2020_4k.exr')
    scale = background_scale(combined, milkyway, rows, directions, relative_y)
    selected = np.array([float(row['mag']) <= MAGNITUDE_LIMIT for row in rows])
    data = catalog_bytes([row for row, keep in zip(rows, selected) if keep], directions[selected], flux[selected], scale)
    target = ROOT / 'src/star-catalog.bin.gz'
    target.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    # Only diffuse structure is stored here; point-star sharpness is independent
    # of this small 1K panorama.
    linear = milkyway.reshape(512, 4, 1024, 4, 3).mean(axis=(1, 3))
    srgb = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
    pixels = np.round(np.clip(srgb, 0, 1) * 255).astype(np.uint8)
    image = ROOT / 'src/starfield.webp'
    Image.fromarray(pixels).save(image, format='WEBP', quality=90, method=6)
    print(f"Embedded data: catalogue {target.stat().st_size:,} + background {image.stat().st_size:,} bytes")


if __name__ == '__main__':
    main()
