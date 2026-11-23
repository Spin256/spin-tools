# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Mouse handles whose strength does not depend on mesh density or patch size."""
import numpy as np
import warp as wp


@wp.kernel
def project_grab(positions: wp.array(dtype=wp.vec3), masses: wp.array(dtype=float),
                 targets: wp.array(dtype=wp.vec3), weights: wp.array(dtype=float), maximum_step: float):
    i = wp.tid()
    if weights[i] > 0. and masses[i] > 0.:
        delta = (targets[i] - positions[i]) * wp.min(weights[i], 1.)
        distance = wp.length(delta)
        limit = maximum_step * wp.min(weights[i], 1.)
        if distance > limit:
            delta = delta * (limit / distance)
        positions[i] = positions[i] + delta


@wp.kernel
def protect_grab_from_collider(positions: wp.array(dtype=wp.vec3), weights: wp.array(dtype=float),
                              masses: wp.array(dtype=float), mesh: wp.uint64, radius: float):
    i = wp.tid()
    if weights[i] > 0. and masses[i] > 0.:
        query = wp.mesh_query_point(mesh, positions[i], 1.e6)
        if query.result:
            surface = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            delta = positions[i] - surface
            distance = wp.length(delta)
            if distance * query.sign < radius:
                if distance > 1.e-9:
                    outward = delta * (query.sign / distance)
                    positions[i] = surface + outward * radius


@wp.kernel
def grab_inverse_mass(original: wp.array(dtype=float), weights: wp.array(dtype=float),
                      frozen: wp.array(dtype=int), inverse_mass: wp.array(dtype=float)):
    i = wp.tid()
    remaining = 1. - wp.clamp(weights[i], 0., 1.)
    inverse_mass[i] = original[i] * remaining * remaining
    if frozen[i] != 0:
        inverse_mass[i] = 0.


@wp.kernel
def grab_velocity(previous: wp.array(dtype=wp.vec3), positions: wp.array(dtype=wp.vec3),
                  velocities: wp.array(dtype=wp.vec3), masses: wp.array(dtype=float),
                  weights: wp.array(dtype=float), dt: float, damping: float):
    i = wp.tid()
    if weights[i] > 0. and masses[i] > 0.:
        velocity = (positions[i] - previous[i]) * (wp.exp(-damping * dt) / dt)
        speed = wp.length(velocity)
        if speed > 1.5:
            velocity = velocity * (1.5 / speed)
        velocities[i] = velocity


class LiveGrab:
    def __init__(self, count, inverse_mass):
        self.targets = wp.zeros(count, dtype=wp.vec3)
        self.weights = wp.zeros(count, dtype=float)
        self.count, self.revision = count, -1
        self.inverse_mass = inverse_mass
        self.original_inverse_mass = wp.clone(inverse_mass)
        self.frozen = wp.zeros(count, dtype=int)

    def update(self, control, freeze=None):
        revision, grab = control.grab_snapshot()
        revision = (revision, freeze.revision if freeze is not None else -1)
        if revision == self.revision:
            return
        weights = np.zeros(self.count, dtype=np.float32)
        targets = np.zeros((self.count, 3), dtype=np.float32)
        if grab is not None:
            indices, positions, strength_weights, strength = grab
            targets[indices] = positions
            weights[indices] = strength_weights * strength
        if freeze is not None:
            wp.copy(self.frozen, freeze.mask)
            weights[freeze.host_mask] = 0.
        self.targets.assign(targets)
        self.weights.assign(weights)
        wp.launch(grab_inverse_mass, dim=self.count,
                  inputs=[self.original_inverse_mass, self.weights, self.frozen, self.inverse_mass])
        self.revision = revision

    def project(self, model, positions, sources, maximum_step, radius):
        wp.launch(project_grab, dim=self.count, inputs=[positions, model.particle_mass,
                  self.targets, self.weights, maximum_step])
        self.protect(model, positions, sources, radius)

    def protect(self, model, positions, sources, radius):
        for source in sources:
            wp.launch(protect_grab_from_collider, dim=self.count,
                      inputs=[positions, self.weights, model.particle_mass, source.mesh.id, radius])

    def velocity(self, model, previous, following, dt, damping):
        wp.launch(grab_velocity, dim=self.count, inputs=[previous.particle_q, following.particle_q,
                  following.particle_qd, model.particle_mass, self.weights, dt, damping])
