# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Conservative discrete vertex/triangle self-contact with topology exclusions."""
import numpy as np
import warp as wp
from warp.sim.collide import triangle_closest_point_barycentric


def excluded_neighbors(count, faces, fasteners=()):
    neighbors = [set([i]) for i in range(count)]
    for a, b, c in faces:
        for first, second in ((a, b), (b, c), (c, a)):
            neighbors[first].add(int(second))
            neighbors[second].add(int(first))
    for a, b in fasteners:
        neighbors[a].add(int(b))
        neighbors[b].add(int(a))
    offsets, values = [0], []
    for index in range(count):
        excluded = set(neighbors[index])
        for neighbor in neighbors[index]:
            excluded.update(neighbors[neighbor])
        values.extend(sorted(excluded))
        offsets.append(len(values))
    return np.asarray(offsets, dtype=np.int32), np.asarray(values, dtype=np.int32)


@wp.kernel
def gather_contacts(mesh: wp.uint64, faces: wp.array2d(dtype=int), snapshot: wp.array(dtype=wp.vec3),
                    previous: wp.array(dtype=wp.vec3), inverse_mass: wp.array(dtype=float),
                    active: wp.array(dtype=int), offsets: wp.array(dtype=int), excluded: wp.array(dtype=int),
                    gap: float, corrections: wp.array(dtype=wp.vec3), counts: wp.array(dtype=float)):
    i = wp.tid()
    if active[i] != 0:
        p = snapshot[i]
        query = wp.mesh_query_aabb(mesh, p - wp.vec3(gap), p + wp.vec3(gap))
        face = int(0)
        visited = int(0)
        while wp.mesh_query_aabb_next(query, face):
            visited += 1
            if visited > 256:
                break
            a, b, c = faces[face, 0], faces[face, 1], faces[face, 2]
            skip = bool(False)
            for slot in range(offsets[i], offsets[i + 1]):
                neighbor = excluded[slot]
                if neighbor == a or neighbor == b or neighbor == c:
                    skip = True
                    break
            if not skip:
                x, y, z = snapshot[a], snapshot[b], snapshot[c]
                cross = wp.cross(y - x, z - x)
                if wp.length(cross) > 1.e-12:
                    bary = triangle_closest_point_barycentric(x, y, z, p)
                    closest = x * bary[0] + y * bary[1] + z * bary[2]
                    delta = p - closest
                    distance = wp.length(delta)
                    if distance < gap:
                        normal = wp.normalize(cross)
                        if distance > 1.e-9:
                            normal = delta / distance
                        else:
                            if i > a:
                                normal = -normal
                        # Interior-face contacts retain their preceding side when
                        # a small step crosses the face within the contact shell.
                        if bary[0] > 1.e-4 and bary[1] > 1.e-4 and bary[2] > 1.e-4:
                            old_normal = wp.cross(previous[b] - previous[a], previous[c] - previous[a])
                            if wp.length(old_normal) > 1.e-12:
                                old_normal = wp.normalize(old_normal)
                                old_q = previous[a] * bary[0] + previous[b] * bary[1] + previous[c] * bary[2]
                                side = wp.dot(previous[i] - old_q, old_normal)
                                if wp.abs(side) > 1.e-8:
                                    normal = wp.normalize(cross)
                                    if side < 0.:
                                        normal = -normal
                        total = inverse_mass[i] + bary[0] * bary[0] * inverse_mass[a] + bary[1] * bary[1] * inverse_mass[b] + bary[2] * bary[2] * inverse_mass[c]
                        if total > 0.:
                            correction = normal * ((gap - wp.dot(delta, normal)) * .8 / total)
                            if inverse_mass[i] > 0.:
                                wp.atomic_add(corrections, i, correction * inverse_mass[i])
                                wp.atomic_add(counts, i, 1.)
                            for corner in range(3):
                                vertex = faces[face, corner]
                                if inverse_mass[vertex] > 0. and bary[corner] > 1.e-6:
                                    wp.atomic_add(corrections, vertex, -correction * (inverse_mass[vertex] * bary[corner]))
                                    wp.atomic_add(counts, vertex, 1.)


@wp.kernel
def apply_contacts(positions: wp.array(dtype=wp.vec3), corrections: wp.array(dtype=wp.vec3),
                   counts: wp.array(dtype=float), maximum: float):
    i = wp.tid()
    if counts[i] > 0.:
        correction = corrections[i] / counts[i]
        length = wp.length(correction)
        if length > maximum:
            correction *= maximum / length
        positions[i] += correction


class ClothSelfCollision:
    def __init__(self, points, faces, inverse_mass, gap=.001, passes=2, fasteners=()):
        self.faces = wp.array(faces, dtype=int)
        self.mesh = wp.Mesh(points=wp.array(points, dtype=wp.vec3), indices=wp.array(faces.ravel(), dtype=int))
        active = np.zeros(len(points), dtype=np.int32)
        active[np.unique(faces)] = 1
        self.active = wp.array(active, dtype=int)
        offsets, excluded = excluded_neighbors(len(points), faces, fasteners)
        self.offsets = wp.array(offsets, dtype=int)
        self.excluded = wp.array(excluded, dtype=int)
        self.corrections = wp.zeros(len(points), dtype=wp.vec3)
        self.counts = wp.zeros(len(points), dtype=float)
        self.inverse_mass = inverse_mass
        self.gap, self.passes, self.count = gap, passes, len(points)

    def project(self, positions, previous):
        for _ in range(self.passes):
            wp.copy(self.mesh.points, positions)
            self.mesh.refit()
            self.corrections.zero_()
            self.counts.zero_()
            wp.launch(gather_contacts, dim=self.count, inputs=[self.mesh.id, self.faces, self.mesh.points,
                      previous, self.inverse_mass, self.active, self.offsets, self.excluded, self.gap,
                      self.corrections, self.counts])
            wp.launch(apply_contacts, dim=self.count, inputs=[positions, self.corrections, self.counts, self.gap * .5])
