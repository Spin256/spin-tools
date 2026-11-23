# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""World-position freezing of selected cloth during a live static drape."""
import numpy as np
import warp as wp


@wp.kernel
def freeze_positions(mask: wp.array(dtype=int), targets: wp.array(dtype=wp.vec3),
                     positions: wp.array(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if mask[i] != 0:
        positions[i] = targets[i]
        velocity[i] = wp.vec3(0.)


class ClothFreeze:
    def __init__(self, count):
        self.mask = wp.zeros(count, dtype=int)
        self.targets = wp.zeros(count, dtype=wp.vec3)
        self.host_mask = np.zeros(count, dtype=bool)
        self.revision = -1

    def update(self, control):
        revision, snapshot = control.freeze_snapshot()
        if revision == self.revision:
            return
        self.host_mask[:] = False
        targets = np.zeros((len(self.host_mask), 3), dtype=np.float32)
        if snapshot is not None:
            indices, positions = snapshot
            self.host_mask[indices] = True
            targets[indices] = positions
        self.mask.assign(self.host_mask.astype(np.int32))
        self.targets.assign(targets)
        self.revision = revision

    def project(self, state):
        wp.launch(freeze_positions, dim=len(self.host_mask),
                  inputs=[self.mask, self.targets, state.particle_q, state.particle_qd])
