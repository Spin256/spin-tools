# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Approximate two-sided aerodynamic pressure with beam falloff and ray shielding."""
import numpy as np
import warp as wp
from wind_settings import pack_wind


@wp.kernel
def wind_triangles(faces: wp.array2d(dtype=int), positions: wp.array(dtype=wp.vec3),
                   velocities: wp.array(dtype=wp.vec3), inverse_mass: wp.array(dtype=float),
                   forces: wp.array(dtype=wp.vec3), settings: wp.array(dtype=float),
                   clock: wp.array(dtype=float), colliders: wp.array(dtype=wp.uint64)):
    face = wp.tid()
    if settings[0] > .5 and settings[7] > 0.:
        a, b, c = faces[face, 0], faces[face, 1], faces[face, 2]
        center = (positions[a] + positions[b] + positions[c]) / 3.
        direction = wp.vec3(settings[4], settings[5], settings[6])
        source = wp.vec3(settings[1], settings[2], settings[3])
        delta = center - source
        axial = wp.dot(delta, direction)
        radial_offset = delta - direction * axial
        radial_distance = wp.length(radial_offset)
        axis = direction
        if settings[17] > .5:
            direction = axis * settings[18]
            if radial_distance > 1.e-8:
                direction += radial_offset * (settings[19] / radial_distance)
        factor = float(1.)
        if (settings[8] > .5 or settings[10] > .5 or settings[19] > 1.e-8) and axial < 0.:
            factor = 0.
        if settings[8] > .5:
            factor *= wp.clamp(1. - axial / settings[9], 0., 1.)
        if settings[10] > .5:
            radius = radial_distance
            inner = settings[11] * (1. - settings[12])
            edge = wp.clamp((settings[11] - radius) / wp.max(settings[11] - inner, 1.e-6), 0., 1.)
            factor *= edge * edge * (3. - 2. * edge)
        if factor > 0.:
            if settings[13] > .5:
                distance = float(10000.)
                if settings[8] > .5 or settings[10] > .5:
                    distance = wp.max(axial - .001, 0.)
                if settings[17] > .5 and settings[19] > 1.e-8:
                    distance = float(10000.)
                    if settings[18] > 1.e-8:
                        distance = wp.min(distance, wp.max(axial / settings[18] - .001, 0.))
                    if radial_distance > 1.e-8:
                        distance = wp.min(distance, wp.max(radial_distance / settings[19] - .001, 0.))
                ray_direction = axis
                if wp.length(direction) > 1.e-8:
                    ray_direction = wp.normalize(direction)
                for i in range(colliders.shape[0]):
                    hit = wp.mesh_query_ray(colliders[i], center - ray_direction * .001, -ray_direction, distance)
                    if hit.result:
                        factor *= .1
                        break
            time = clock[0]
            gust = wp.max(0., 1. + settings[14] * (.65 * wp.sin(6.2831853 * settings[15] * time)
                            + .35 * wp.sin(3.883222 * settings[15] * time)))
            air = direction * (settings[7] * gust)
            turbulence = wp.vec3(wp.sin(center[1] * 8. + time * 3.),
                                 wp.sin(center[2] * 7. + time * 2.3), wp.sin(center[0] * 9. + time * 2.7))
            air += turbulence * (settings[7] * settings[16] * .3)
            relative = air - (velocities[a] + velocities[b] + velocities[c]) / 3.
            cross = wp.cross(positions[b] - positions[a], positions[c] - positions[a])
            twice_area = wp.length(cross)
            if twice_area > 1.e-12:
                normal = cross / twice_area
                speed = wp.dot(relative, normal)
                # Air density 1.2 kg/m3, drag coefficient 1; 2% tangential drag.
                force = (normal * (speed * wp.abs(speed)) + relative * (wp.length(relative) * .02))
                force *= .6 * (.5 * twice_area) * factor / 3.
                if inverse_mass[a] > 0.: wp.atomic_add(forces, a, force)
                if inverse_mass[b] > 0.: wp.atomic_add(forces, b, force)
                if inverse_mass[c] > 0.: wp.atomic_add(forces, c, force)


@wp.kernel
def limit_wind(forces: wp.array(dtype=wp.vec3), masses: wp.array(dtype=float)):
    i = wp.tid()
    maximum = masses[i] * 80.
    length = wp.length(forces[i])
    if length > maximum and length > 1.e-12:
        forces[i] *= maximum / length


class WindForces:
    def __init__(self, faces, sources):
        self.faces = wp.array(faces, dtype=int)
        self.settings = wp.zeros(20, dtype=float)
        self.clock = wp.zeros(1, dtype=float)
        self.colliders = wp.array(np.asarray([s.mesh.id for s in sources], dtype=np.uint64), dtype=wp.uint64)
        self.revision = -1

    def update(self, control, time):
        revision, settings = control.snapshot()
        if revision != self.revision:
            self.settings.assign(pack_wind(settings))
            self.revision = revision
        self.clock.assign(np.array([time], dtype=np.float32))

    def apply(self, model, state, inverse_mass):
        wp.launch(wind_triangles, dim=len(self.faces), inputs=[self.faces, state.particle_q,
                  state.particle_qd, inverse_mass, state.particle_f, self.settings, self.clock, self.colliders])
        wp.launch(limit_wind, dim=len(model.particle_mass), inputs=[state.particle_f, model.particle_mass])
