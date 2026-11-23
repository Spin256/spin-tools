# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Approximate cloth buoyancy and still-water/entry resistance below a waterline."""
import numpy as np
import warp as wp
from water_settings import pack_water


@wp.func
def clipped_fraction(a: float, b: float, c: float):
    # Signed depths at a triangle's vertices. Edge clipping gives area ratios.
    if a <= 0. and b <= 0. and c <= 0.:
        return 0.
    if a > 0. and b > 0. and c > 0.:
        return 1.
    if a > 0. and b <= 0. and c <= 0.:
        return a * a / ((a - b) * (a - c))
    if b > 0. and a <= 0. and c <= 0.:
        return b * b / ((b - a) * (b - c))
    if c > 0. and a <= 0. and b <= 0.:
        return c * c / ((c - a) * (c - b))
    if a <= 0.:
        return 1. - a * a / ((a - b) * (a - c))
    if b <= 0.:
        return 1. - b * b / ((b - a) * (b - c))
    return 1. - c * c / ((c - a) * (c - b))


@wp.kernel
def water_triangles(faces: wp.array2d(dtype=int), positions: wp.array(dtype=wp.vec3),
                    settings: wp.array(dtype=float), area_sum: wp.array(dtype=float),
                    wet_sum: wp.array(dtype=float)):
    i = wp.tid()
    if settings[0] > .5:
        a, b, c = faces[i, 0], faces[i, 1], faces[i, 2]
        p, q, r = positions[a], positions[b], positions[c]
        area = wp.length(wp.cross(q - p, r - p)) / 6.
        wet = float(0.)
        # Average clipped areas across a shallow transition below the surface.
        for sample in range(4):
            level = settings[1] - settings[5] * (float(sample) + .5) / 4.
            wet += clipped_fraction(level - p[1], level - q[1], level - r[1]) * .25
        wp.atomic_add(area_sum, a, area)
        wp.atomic_add(area_sum, b, area)
        wp.atomic_add(area_sum, c, area)
        wp.atomic_add(wet_sum, a, area * wet)
        wp.atomic_add(wet_sum, b, area * wet)
        wp.atomic_add(wet_sum, c, area * wet)


@wp.kernel
def apply_water(positions: wp.array(dtype=wp.vec3), velocities: wp.array(dtype=wp.vec3),
                masses: wp.array(dtype=float), inverse_mass: wp.array(dtype=float),
                forces: wp.array(dtype=wp.vec3), settings: wp.array(dtype=float),
                gravity: wp.array(dtype=wp.vec3), area_sum: wp.array(dtype=float),
                wet_sum: wp.array(dtype=float), dt: float):
    i = wp.tid()
    if settings[0] > .5 and inverse_mass[i] > 0. and masses[i] > 0. and area_sum[i] > 1.e-14:
        wet = wp.clamp(wet_sum[i] / area_sum[i], 0., 1.)
        velocity = velocities[i]
        coefficient = settings[3] * wet
        vertical_coefficient = coefficient
        if velocity[1] < 0.:
            depth = wp.max(settings[1] - positions[i][1], 0.)
            surface = wp.clamp(1. - depth / (settings[5] * 2.), 0., 1.)
            vertical_coefficient += settings[4] * wet * surface
        drag = wp.vec3(-velocity[0] * coefficient / (1. + coefficient * dt),
                      -velocity[1] * vertical_coefficient / (1. + vertical_coefficient * dt),
                      -velocity[2] * coefficient / (1. + coefficient * dt))
        acceleration = drag - gravity[0] * (settings[2] * wet)
        forces[i] = forces[i] + acceleration * masses[i]


class WaterForces:
    def __init__(self, faces, count):
        self.faces = wp.array(faces, dtype=int)
        self.settings = wp.zeros(6, dtype=float)
        self.gravity = wp.zeros(1, dtype=wp.vec3)
        self.area_sum = wp.zeros(count, dtype=float)
        self.wet_sum = wp.zeros(count, dtype=float)
        self.revision = -1

    def update(self, control, gravity):
        revision, settings = control.snapshot()
        if revision != self.revision:
            self.settings.assign(pack_water(settings))
            self.revision = revision
        self.gravity.assign(np.asarray(gravity, dtype=np.float32).reshape(1, 3))

    def apply(self, model, state, inverse_mass, dt):
        self.area_sum.zero_()
        self.wet_sum.zero_()
        wp.launch(water_triangles, dim=len(self.faces), inputs=[self.faces, state.particle_q,
                  self.settings, self.area_sum, self.wet_sum])
        wp.launch(apply_water, dim=len(model.particle_mass), inputs=[state.particle_q, state.particle_qd,
                  model.particle_mass, inverse_mass, state.particle_f, self.settings, self.gravity,
                  self.area_sum, self.wet_sum, dt])
