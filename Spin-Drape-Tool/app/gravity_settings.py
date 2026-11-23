# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Uniform gravity direction and strength, exchanged safely with the solver."""
import math
import threading
import numpy as np
from wind_settings import wind_direction

STANDARD_GRAVITY = 9.80665


class GravityControl:
    def __init__(self, heading=0., elevation=-90., strength=1.):
        self.lock = threading.Lock()
        self.revision = -1
        self.set(dict(heading=heading, elevation=elevation, strength=strength))

    def set(self, settings):
        heading, elevation, strength = (float(settings[key]) for key in ('heading', 'elevation', 'strength'))
        if not all(math.isfinite(v) for v in (heading, elevation, strength)) or not (
                0 <= heading <= 360 and -90 <= elevation <= 90 and 0 <= strength <= 5):
            raise ValueError('Gravity direction or strength is outside its supported range.')
        acceleration = wind_direction(heading, elevation) * (STANDARD_GRAVITY * strength)
        acceleration[abs(acceleration) < 1.e-12] = 0.
        with self.lock:
            self.acceleration = np.asarray(acceleration, dtype=np.float32)
            self.revision += 1

    def snapshot(self):
        with self.lock:
            return self.revision, self.acceleration.copy()
