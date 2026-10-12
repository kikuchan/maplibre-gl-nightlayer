"""The fixed NightLayer ZIP layout; all numeric arrays are little-endian Float32."""

import io
import json
import zipfile

import numpy as np


def lighting_zip(radiance, optical, irradiance, metadata):
    # Source order: branch, sun, azimuth, height, view, distance, RGB.
    sources = radiance.transpose(6, 0, 4, 5, 1, 2, 3, 7)
    arrays = {
        'irradiance.bin': irradiance,
        'rayleigh.bin': sources[0],
        'mie.bin': sources[1],
        'optical.bin': optical,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=False) as archive:
        def entry(name, values):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, values, compresslevel=9)

        entry('metadata.json', json.dumps(metadata, separators=(',', ':'), sort_keys=True).encode())
        for name, values in arrays.items():
            values = np.ascontiguousarray(values, dtype='<f4')
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite LUT: {name}')
            entry(name, values.tobytes())
    return output.getvalue()
