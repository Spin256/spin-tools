# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Wind anchored to a reference head pose; settings and GPU/preview use world cm."""
import math
import numpy as np
from wind_settings import wind_direction


def wind_at_pose(settings, pose):
    result = dict(settings)
    if not result.get('track_avatar', False):
        return result
    rotation, translation = pose
    reference = np.asarray(result.get('reference_rotation', np.eye(3)), float)
    origin = np.asarray(result.get('reference_translation', np.zeros(3)), float)
    turn = np.asarray(rotation) @ reference.T
    source = turn @ (np.array([result[k] for k in ('x','y','z')]) - origin) + translation
    direction = turn @ wind_direction(result['heading'], result['elevation'])
    result.update(zip(('x','y','z'), map(float, source)))
    result['heading'] = math.degrees(math.atan2(direction[0], -direction[2])) % 360
    result['elevation'] = math.degrees(math.atan2(direction[1], math.hypot(direction[0], direction[2])))
    return result


class RelativeWind:
    def __init__(self, control, pose):
        self.control, self.pose = control, pose

    def snapshot(self):
        revision, settings = self.control.snapshot()
        if not settings.get('track_avatar', False):
            return revision, settings
        rotation, translation = self.pose()
        key = (revision, *np.asarray(rotation).ravel(), *np.asarray(translation))
        return key, wind_at_pose(settings, (rotation, translation))
