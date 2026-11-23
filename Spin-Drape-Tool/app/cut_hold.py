# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Hold the two sides of a garment cut together until its release frame (Spin256 · spin256.com/tools).

garment_cut.py splits the vertices along a cut; each (original, copy) pair is pulled to a shared point
every substep while the hold is on, so a clasp stays closed until the chosen frame and then lets go. The on/off flag is a device value, so it can change between captured frames."""
import numpy as np
import warp as wp


@wp.kernel
def join_pairs(positions: wp.array(dtype=wp.vec3), inverse_mass: wp.array(dtype=float),
               first: wp.array(dtype=int), second: wp.array(dtype=int), active: wp.array(dtype=int)):
    i = wp.tid()
    if active[0] == 0:
        return
    a = first[i]
    b = second[i]
    total = inverse_mass[a] + inverse_mass[b]
    if total <= 0.:
        return
    delta = positions[b] - positions[a]
    positions[a] = positions[a] + delta * (inverse_mass[a] / total)
    positions[b] = positions[b] - delta * (inverse_mass[b] / total)


class CutHold:
    def __init__(self, pairs, release_interval):
        pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        self.count = len(pairs)
        self.release = release_interval
        self.first = wp.array(pairs[:, 0].copy(), dtype=int)
        self.second = wp.array(pairs[:, 1].copy(), dtype=int)
        self.active = wp.array([1], dtype=int)
        self.state = 1

    def update(self, interval):
        state = int(self.release is not None and interval < self.release)
        if state != self.state:
            self.active.assign(np.array([state], dtype=np.int32))
            self.state = state

    def project(self, positions, inverse_mass):
        if self.count:
            wp.launch(join_pairs, self.count, inputs=[positions, inverse_mass, self.first, self.second, self.active])
