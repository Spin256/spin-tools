# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Hair-to-hair collision on the GPU: strands keep their spacing so hair keeps its volume.

Captured hair layers often sit closer than any sensible collision gap, so each point
gets a "clearance": its distance at rest to the nearest point of another strand. A pair
is pushed apart only when it comes closer than the smaller of the spacing setting and
the pair's own captured clearance, so layers never sink further into each other than
they were styled, and nothing puffs up at the start.
"""
import numpy as np
import warp as wp


@wp.kernel
def rest_clearance(grid: wp.uint64, points: wp.array(dtype=wp.vec3), owner: wp.array(dtype=int),
                   radius: float, clearance: wp.array(dtype=float)):
    i = wp.tid()
    p = points[i]
    best = radius
    query = wp.hash_grid_query(grid, p, radius)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if owner[j] != owner[i]:
            d = wp.length(points[j] - p)
            if d < best:
                best = d
    clearance[i] = best


@wp.kernel
def gather_repulsion(grid: wp.uint64, pos: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
                     owner: wp.array(dtype=int), clearance: wp.array(dtype=float), spacing: float,
                     corrections: wp.array(dtype=wp.vec3), counts: wp.array(dtype=float)):
    i = wp.tid()
    if inv_mass[i] > 0.0:
        p = pos[i]
        query = wp.hash_grid_query(grid, p, spacing)
        j = int(0)
        while wp.hash_grid_query_next(query, j):
            if owner[j] != owner[i]:
                gap = wp.min(spacing, wp.max(wp.min(clearance[i], clearance[j]), 0.02))
                d = p - pos[j]
                length = wp.length(d)
                if length < gap and length > 1.0e-6:
                    w = inv_mass[i] + inv_mass[j]
                    if w > 0.0:
                        # Half of the overlap is resolved by each side of the pair.
                        push = d * ((gap - length) / length * inv_mass[i] / w)
                        wp.atomic_add(corrections, i, push)
                        wp.atomic_add(counts, i, 1.0)


@wp.kernel
def apply_repulsion(pos: wp.array(dtype=wp.vec3), corrections: wp.array(dtype=wp.vec3),
                    counts: wp.array(dtype=float), strength: float, max_step: float):
    i = wp.tid()
    if counts[i] > 0.0:
        c = corrections[i] * (strength / counts[i])
        length = wp.length(c)
        if length > max_step:
            c = c * (max_step / length)
        pos[i] = pos[i] + c


class HairCollision:
    """Strand-strand repulsion with per-point captured clearance (all lengths in cm)."""

    def __init__(self, rest_points, particle_count, spacing=0.6, strength=0.6):
        n = len(rest_points)
        self.spacing, self.strength = float(spacing), float(strength)
        self.owner = wp.array(np.repeat(np.arange(len(particle_count)), particle_count).astype(np.int32), dtype=int)
        bounds = np.ptp(rest_points, axis=0)
        cells = np.clip((bounds / max(spacing, 0.05)).astype(int) + 1, 16, 256)
        self.grid = wp.HashGrid(int(cells[0]), int(cells[1]), int(cells[2]))
        rest = wp.array(np.asarray(rest_points, np.float32), dtype=wp.vec3)
        self.grid.build(rest, self.spacing)
        self.clearance = wp.zeros(n, dtype=float)
        wp.launch(rest_clearance, dim=n, inputs=[self.grid.id, rest, self.owner, self.spacing, self.clearance])
        self.corrections = wp.zeros(n, dtype=wp.vec3)
        self.counts = wp.zeros(n, dtype=float)
        self.count = n

    def project(self, pos, inv_mass):
        self.grid.build(pos, self.spacing)
        self.corrections.zero_()
        self.counts.zero_()
        wp.launch(gather_repulsion, dim=self.count, inputs=[self.grid.id, pos, inv_mass, self.owner,
                  self.clearance, self.spacing, self.corrections, self.counts])
        wp.launch(apply_repulsion, dim=self.count, inputs=[pos, self.corrections, self.counts, self.strength,
                  self.spacing * 0.5])
