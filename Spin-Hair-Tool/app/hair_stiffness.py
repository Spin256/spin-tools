# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Strand stiffness by shape matching, and a gravity-balanced rest shape.

Stiffness: every strand is covered by short overlapping sections (5 points, starting
every 2). Each section is pulled toward its rest shape, rotated to best fit how it
currently lies (the best-fit rotation), so the strand resists bending as real hair does
without being pinned anywhere in space. The first section also contains the fixed
root and a virtual follicle point behind it, so stiffness reaches into the scalp (only for roots
that are fixed; hair hanging from other hair has no follicle).

Gravity balance: a captured hairstyle already hangs under gravity. Using it directly
as the rest shape makes hair sag twice and lie flat. ``balance`` settles the hair under
gravity, measures how far it drifts from the capture and moves the rest shape the
opposite way, a few times over, so the capture becomes the hair's balance point. With
gravity off, hair then relaxes toward its natural (weightless) shape and stands out.
"""
import numpy as np
import warp as wp

# Section sizes: short ones keep local curls, long ones keep the strand's overall shape.
SECTIONS = (5, 9, 17)


@wp.kernel
def match_sections(pos: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=float), rest: wp.array(dtype=wp.vec3),
                   first: wp.array(dtype=int), size: wp.array(dtype=int), with_anchor: wp.array(dtype=int),
                   anchor: wp.array(dtype=wp.vec3), rest_anchor: wp.array(dtype=wp.vec3), strand: wp.array(dtype=int),
                   strength: float, taper: wp.array(dtype=float), turn: wp.array(dtype=wp.quat), corrections: wp.array(dtype=wp.vec3),
                   counts: wp.array(dtype=float)):
    c = wp.tid()
    a = first[c]
    n = size[c]
    use_anchor = with_anchor[c]
    s = strand[c]
    # Centroids (the anchor counts as one more, fixed, member).
    cx = wp.vec3(0.0)
    cr = wp.vec3(0.0)
    for k in range(n):
        cx += pos[a + k]
        cr += rest[a + k]
    total = float(n)
    if use_anchor != 0:
        cx += anchor[s]
        cr += rest_anchor[s]
        total += 1.0
    cx = cx / total
    cr = cr / total
    m = wp.mat33(0.0)
    for k in range(n):
        m += wp.outer(pos[a + k] - cx, rest[a + k] - cr)
    if use_anchor != 0:
        m += wp.outer(anchor[s] - cx, rest_anchor[s] - cr)
    # Rotation by the robust iterative method (Mueller et al. 2016), warm-started from the
    # section's last rotation: exactly identity at rest, so it never adds energy by itself
    # (float32 SVD would, from round-off on nearly straight sections).
    q = turn[c]
    for _ in range(8):
        r = wp.quat_to_matrix(q)
        r0 = wp.vec3(r[0, 0], r[1, 0], r[2, 0])
        r1 = wp.vec3(r[0, 1], r[1, 1], r[2, 1])
        r2 = wp.vec3(r[0, 2], r[1, 2], r[2, 2])
        m0 = wp.vec3(m[0, 0], m[1, 0], m[2, 0])
        m1 = wp.vec3(m[0, 1], m[1, 1], m[2, 1])
        m2 = wp.vec3(m[0, 2], m[1, 2], m[2, 2])
        omega = (wp.cross(r0, m0) + wp.cross(r1, m1) + wp.cross(r2, m2)) / (
            wp.abs(wp.dot(r0, m0) + wp.dot(r1, m1) + wp.dot(r2, m2)) + 1.0e-9)
        w = wp.length(omega)
        if w < 1.0e-7:
            break
        q = wp.normalize(wp.quat_from_axis_angle(omega / w, w) * q)
    turn[c] = q
    r = wp.quat_to_matrix(q)
    for k in range(n):
        i = a + k
        if inv_mass[i] > 0.0:
            goal = cx + r * (rest[i] - cr)
            wp.atomic_add(corrections, i, (goal - pos[i]) * (strength * taper[i]))
            wp.atomic_add(counts, i, 1.0)


@wp.kernel
def momentum_give(vel: wp.array(dtype=wp.vec3), pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
                  root: wp.array(dtype=int), base: wp.array(dtype=float), dt: float, speed: float,
                  out: wp.array(dtype=float)):
    """Momentum: a point moving fast relative to its strand's root eases its stiffness, so a
    flicked strand carries on; it springs back to its style once it slows down."""
    i = wp.tid()
    r = root[i]
    root_velocity = (pos[r] - prev[r]) / dt
    relative = wp.length(vel[i] - root_velocity) / speed
    out[i] = base[i] / (1.0 + relative * relative)


@wp.kernel
def apply_sections(pos: wp.array(dtype=wp.vec3), corrections: wp.array(dtype=wp.vec3), counts: wp.array(dtype=float)):
    i = wp.tid()
    if counts[i] > 0.0:
        pos[i] = pos[i] + corrections[i] / counts[i]


class StrandStiffness:
    """Shape-matching sections for every strand; ``rest`` can be replaced (gravity balance)."""

    def __init__(self, particle_start, particle_count, rest_points, anchors, strength, fixed_root, taper=0.0,
                 swing=False):
        firsts, sizes, owners, anchored = [], [], [], []
        for s, (start, count) in enumerate(zip(particle_start, particle_count)):
            for section in SECTIONS:
                if section > 5 and count <= section // 2 + 2:
                    break
                k = 0
                while True:
                    size = min(section, count - k)
                    if size < 3:
                        break
                    # swing: only the short root section holds the follicle angle, so long sections
                    # (and the whole strand) can pivot about the root and trail behind movement.
                    anchor_here = k == 0 and fixed_root[s] and (not swing or section == SECTIONS[0])
                    firsts.append(start + k); sizes.append(size); owners.append(s); anchored.append(int(anchor_here))
                    if k + size >= count:
                        break
                    k += section // 2
            if count > SECTIONS[-1]:
                # One section over the whole strand: its overall shape, turning about the root.
                firsts.append(start); sizes.append(count); owners.append(s); anchored.append(int(fixed_root[s] and not swing))
        self.sections = len(firsts)
        self.first = wp.array(np.asarray(firsts, np.int32), dtype=int)
        self.size = wp.array(np.asarray(sizes, np.int32), dtype=int)
        self.strand = wp.array(np.asarray(owners, np.int32), dtype=int)
        self.with_anchor = wp.array(np.asarray(anchored, np.int32), dtype=int)
        self.rest = wp.array(np.asarray(rest_points, np.float32), dtype=wp.vec3)
        self.anchor = wp.array(np.asarray(anchors, np.float32), dtype=wp.vec3)
        self.rest_anchor = wp.array(np.asarray(anchors, np.float32), dtype=wp.vec3)
        self.strength = float(strength)
        # Taper: real hair is stiff near the root and flexible toward the tip, so the tips trail
        # behind movement. Weight (1 - position along the strand) ** taper; 0 = even stiffness.
        along = np.concatenate([np.linspace(0.0, 1.0, c) for c in particle_count])
        self.taper = wp.array(((1.0 - along) ** taper if taper > 0 else np.ones_like(along)).astype(np.float32), dtype=float)
        self.base_taper = wp.clone(self.taper)
        self.root_of = wp.array(np.repeat(np.asarray(particle_start), particle_count).astype(np.int32), dtype=int)
        self.turn = wp.array(np.tile(np.array([0, 0, 0, 1], np.float32), (self.sections, 1)), dtype=wp.quat)
        n = len(rest_points)
        self.corrections = wp.zeros(n, dtype=wp.vec3)
        self.counts = wp.zeros(n, dtype=float)
        self.count = n

    def set_rest(self, rest_points):
        self.rest.assign(np.asarray(rest_points, np.float32))

    def reset(self):
        self.turn.assign(np.tile(np.array([0, 0, 0, 1], np.float32), (self.sections, 1)))

    def ease(self, vel, pos, prev, dt, speed):
        """Momentum for this substep (speed in cm/s at which stiffness halves); None = off."""
        if speed is None:
            wp.copy(self.taper, self.base_taper)
        else:
            wp.launch(momentum_give, dim=self.count, inputs=[vel, pos, prev, self.root_of, self.base_taper, dt,
                                                              float(speed), self.taper])

    def project(self, pos, inv_mass):
        if self.strength <= 0.0 or not self.sections:
            return
        self.corrections.zero_()
        self.counts.zero_()
        wp.launch(match_sections, dim=self.sections, inputs=[pos, inv_mass, self.rest, self.first, self.size,
                  self.with_anchor, self.anchor, self.rest_anchor, self.strand, self.strength,
                  self.taper, self.turn, self.corrections, self.counts])
        wp.launch(apply_sections, dim=self.count, inputs=[pos, self.corrections, self.counts])
