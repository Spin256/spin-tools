# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Still-water settings and thread-safe simulation snapshots."""
import math
import threading
import numpy as np

DEFAULT_WATER = dict(enabled=False, height=100., buoyancy=110., drag=8., entry=35., transition=5.)


def validate_water(settings):
    result = dict(DEFAULT_WATER, **settings)
    for key, low, high in (('height', -100000, 100000), ('buoyancy', 0, 300),
                           ('drag', 0, 50), ('entry', 0, 100), ('transition', .1, 50)):
        value = float(result[key])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Water setting outside its supported range: ' + key)
        result[key] = value
    return result


class WaterControl:
    def __init__(self, settings=None):
        self.lock = threading.Lock()
        self.settings = validate_water(settings or {})
        self.revision = 0

    def set(self, settings):
        settings = validate_water(settings)
        with self.lock:
            self.settings = settings
            self.revision += 1

    def snapshot(self):
        with self.lock:
            return self.revision, self.settings.copy()


def pack_water(settings):
    s = validate_water(settings)
    return np.array([float(s['enabled']), s['height'] * .01, s['buoyancy'] * .01,
                     s['drag'], s['entry'], s['transition'] * .01], dtype=np.float32)
