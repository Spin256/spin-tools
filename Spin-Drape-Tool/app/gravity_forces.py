# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Uniform gravity forces with live GPU-buffer updates between frames."""
import warp as wp


@wp.kernel
def apply_gravity(forces: wp.array(dtype=wp.vec3), masses: wp.array(dtype=float),
                  inverse_mass: wp.array(dtype=float), acceleration: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if inverse_mass[i] > 0. and masses[i] > 0.:
        forces[i] = forces[i] + acceleration[0] * masses[i]


class GravityForces:
    def __init__(self):
        self.acceleration = wp.zeros(1, dtype=wp.vec3)
        self.revision = -1

    def update(self, control):
        revision, acceleration = control.snapshot()
        if revision != self.revision:
            self.acceleration.assign(acceleration.reshape(1, 3))
            self.revision = revision

    def apply(self, model, state, inverse_mass):
        wp.launch(apply_gravity, dim=len(model.particle_mass), inputs=[state.particle_f,
                  model.particle_mass, inverse_mass, self.acceleration])
