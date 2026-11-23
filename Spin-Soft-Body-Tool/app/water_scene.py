# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Write optional water-height reference geometry."""
import os
from pathlib import Path
import numpy as np


def write_water_plane(spec, written, temporary):
    settings = spec.get('water_plane')
    if settings is None:
        return None
    height, x, z, extent = (float(settings[key]) for key in ('height', 'x', 'z', 'extent'))
    if not np.isfinite([height, x, z, extent]).all() or extent <= 0:
        raise ValueError('Invalid water-height plane geometry.')
    target = Path(spec['water_return_path'])
    temp = target.with_name(target.name + '.tmp')
    temporary.append(temp)
    points = [(x-extent,height,z-extent), (x+extent,height,z-extent),
              (x+extent,height,z+extent), (x-extent,height,z+extent)]
    with temp.open('w', encoding='utf-8', newline='\n') as output:
        output.write('o Water\n')
        for point in points:
            output.write('v %.9g %.9g %.9g\n' % point)
        output.write('vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl water_reference\nf 1/1 4/4 3/3 2/2\n')
    os.replace(temp, target)
    written.append(target)
    return dict(path=str(target), height=height, vertex_count=4)
