# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Thread-safe wind source settings, independent of the DAZ bridge."""
import math
import threading
import numpy as np


DEFAULT_WIND = dict(enabled=False, x=0., y=100., z=100., heading=0., elevation=0.,
                    speed=5., axial=False, distance=300., radial=True, radius=50., softness=50.,
                    shielding=True, gusts=0., frequency=.5, turbulence=0., mode='parallel', spread=45.)


def wind_direction(heading, elevation):
    h, e = math.radians(heading), math.radians(elevation)
    return np.array([math.sin(h) * math.cos(e), math.sin(e), -math.cos(h) * math.cos(e)], dtype=np.float32)


def validate_wind(settings):
    result = dict(DEFAULT_WIND, **settings)
    if result['mode'] not in ('parallel', 'spreading'):
        raise ValueError('Unknown wind mode')
    for key, low, high in (('x', -100000, 100000), ('y', -100000, 100000), ('z', -100000, 100000),
                           ('heading', 0, 360), ('elevation', -90, 90), ('speed', 0, 30),
                           ('distance', 1, 100000), ('radius', 1, 100000), ('softness', 0, 100),
                           ('gusts', 0, 100), ('frequency', .05, 5), ('turbulence', 0, 100), ('spread', 0, 90)):
        value = float(result[key])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Wind setting is outside its supported range: ' + key)
        result[key] = value
    return result


class WindControl:
    def __init__(self, settings=None):
        self.lock = threading.Lock()
        self.settings = validate_wind(settings or {})
        self.revision = 0

    def set(self, settings):
        settings = validate_wind(settings)
        with self.lock:
            self.settings = settings
            self.revision += 1

    def snapshot(self):
        with self.lock:
            return self.revision, self.settings.copy()


def pack_wind(settings):
    s = validate_wind(settings)
    angle = math.radians(s['spread']) if s['mode'] == 'spreading' else 0.
    cosine = 0. if s['spread'] == 90 and s['mode'] == 'spreading' else math.cos(angle)
    return np.array([float(s['enabled']), s['x'] * .01, s['y'] * .01, s['z'] * .01,
                     *wind_direction(s['heading'], s['elevation']), s['speed'],
                     float(s['axial']), s['distance'] * .01, float(s['radial']), s['radius'] * .01,
                     s['softness'] * .01, float(s['shielding']), s['gusts'] * .01,
                     s['frequency'], s['turbulence'] * .01, float(s['mode'] == 'spreading'),
                     cosine, math.sin(angle)], dtype=np.float32)
