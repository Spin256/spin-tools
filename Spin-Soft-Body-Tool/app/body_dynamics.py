# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Time-based offset dynamics about prescribed poses, in meters and seconds."""
import math
import numpy as np
from body_geometry import connected_labels


class VerticalBands:
    """Couple vertical offsets across each painted patch without changing its mesh."""

    def __init__(self, points, mobility, neighbours):
        labels, _ = connected_labels(neighbours, mobility > 0)
        active = np.flatnonzero(labels >= 0)
        ordered = active[np.argsort(labels[active], kind='stable')]
        splits = np.flatnonzero(np.diff(labels[ordered])) + 1
        self.patches = []
        for indices in np.split(ordered, splits):
            if len(indices) < 4:
                continue
            heights = points[indices, 1]
            span = float(np.ptp(heights))
            if span < 1e-6:
                continue
            centers = np.linspace(heights.min(), heights.max(), 8)
            spacing = centers[1] - centers[0]
            taper = mobility[indices] / mobility[indices].max()
            shape = np.maximum(0., 1. - np.abs(heights[:, None] - centers) / spacing) * taper[:, None]
            populated = np.any(shape > 0, axis=0)
            fit = np.linalg.pinv(shape[:, populated])
            self.patches.append((indices, centers, taper, shape, populated, fit))

    def project(self, offset):
        for indices, centers, taper, shape, populated, fit in self.patches:
            vertical = offset[indices, 1]
            # A shift that already follows the bands survives this fit intact.
            shifts = np.interp(centers, centers[populated], fit @ vertical)
            max_change = .35 * (centers[1] - centers[0])
            # Let slices compress and stretch while bounding the change in their spacing.
            for index in range(1, len(shifts)):
                excess = np.clip(shifts[index] - shifts[index - 1], -max_change, max_change)
                shifts[index] = shifts[index - 1] + excess
            target = shape @ shifts
            # Keep the common translation when limiting changes in ring spacing.
            target += taper * (np.dot(taper, vertical - target) / np.dot(taper, taper))
            offset[indices, 1] = vertical + .65 * (target - vertical)


def predict_offsets(offset,velocity,pose_velocity,previous_pose_velocity,mobility,dt,frequency,damping,inertia,
                    *,volume=None,directional_lag=False,gravity=0.):
    omega=2*math.pi*frequency
    decay = math.exp(-damping * dt)
    impulse = (pose_velocity - previous_pose_velocity) * decay
    if directional_lag:
        # Split damping between motion relative to the body and motion in world
        # space. This maintains a lag even as the prescribed body slows down.
        impulse += pose_velocity * (.5 * (1. - decay))
    if volume is not None:
        impulse = volume.shared_impulse(impulse)
    velocity = (velocity - offset * (omega * omega * dt)) * decay - impulse * inertia * mobility[:, None]
    if gravity:
        velocity[:, 1] -= gravity * dt * decay * mobility
    speed=np.linalg.norm(velocity,axis=1)
    velocity*=np.minimum(1.,1./np.maximum(speed,1e-20))[:,None]
    predicted=offset+velocity*dt
    predicted[mobility<=0]=0;velocity[mobility<=0]=0
    return predicted,velocity
