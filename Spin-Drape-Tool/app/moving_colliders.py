# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""GPU updates for deforming collision meshes and surface-attached cloth pins."""

import numpy as np
import warp as wp

WRAP = -1          # load_interval: the step from the last frame into frame 0 (seamless loops)


@wp.func
def surface_basis(mesh: wp.uint64, face: int):
    a = wp.mesh_get_point(mesh, face * 3)
    b = wp.mesh_get_point(mesh, face * 3 + 1)
    c = wp.mesh_get_point(mesh, face * 3 + 2)
    tangent = wp.normalize(b - a)
    normal = wp.normalize(wp.cross(b - a, c - a))
    bitangent = wp.cross(normal, tangent)
    return tangent, bitangent, normal


@wp.kernel
def bind_pins(
    mesh: wp.uint64, collider: int,
    points: wp.array(dtype=wp.vec3), pinned: wp.array(dtype=int),
    best_distance: wp.array(dtype=float), bound_collider: wp.array(dtype=int),
    bound_face: wp.array(dtype=int), barycentric: wp.array(dtype=wp.vec2),
    local_offset: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    if pinned[index] != 0:
        query = wp.mesh_query_point_no_sign(mesh, points[index], 1.0e6)
        if query.result:
            surface = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            delta = points[index] - surface
            distance = wp.length(delta)
            if distance < best_distance[index]:
                tangent, bitangent, normal = surface_basis(mesh, query.face)
                best_distance[index] = distance
                bound_collider[index] = collider
                bound_face[index] = query.face
                barycentric[index] = wp.vec2(query.u, query.v)
                local_offset[index] = wp.vec3(
                    wp.dot(delta, tangent), wp.dot(delta, bitangent), wp.dot(delta, normal)
                )


@wp.kernel
def deform_mesh(
    first: wp.array(dtype=wp.vec3), second: wp.array(dtype=wp.vec3),
    alpha: float, frame_dt: float,
    positions: wp.array(dtype=wp.vec3), velocities: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    positions[index] = first[index] + (second[index] - first[index]) * alpha
    velocities[index] = (second[index] - first[index]) / frame_dt


@wp.kernel
def move_pins(
    mesh: wp.uint64, collider: int,
    bound_collider: wp.array(dtype=int), bound_face: wp.array(dtype=int),
    barycentric: wp.array(dtype=wp.vec2), local_offset: wp.array(dtype=wp.vec3),
    positions: wp.array(dtype=wp.vec3), anchors: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    if bound_collider[index] == collider:
        face = bound_face[index]
        uv = barycentric[index]
        tangent, bitangent, normal = surface_basis(mesh, face)
        offset = local_offset[index]
        target = wp.mesh_eval_position(mesh, face, uv[0], uv[1])
        target += tangent * offset[0] + bitangent * offset[1] + normal * offset[2]
        positions[index] = target
        anchors[index] = target


class MovingColliders:
    def __init__(self, sources, motion, points, pinned):
        self.motion = motion
        self.resources = {}
        count = len(points)
        self.bound_collider = wp.full(count, -1, dtype=int)
        self.bound_face = wp.zeros(count, dtype=int)
        self.barycentric = wp.zeros(count, dtype=wp.vec2)
        self.local_offset = wp.zeros(count, dtype=wp.vec3)
        best = wp.full(count, 1.0e20, dtype=float)
        gpu_points = wp.array(points, dtype=wp.vec3)
        gpu_pinned = wp.array(pinned.astype(np.int32), dtype=int)
        for index, source in enumerate(sources):
            mesh = source.mesh
            if index in motion.samples:
                first = wp.array(motion.colliders[index][0], dtype=wp.vec3)
                second = wp.clone(first)
                self.resources[index] = (mesh, first, second)
            wp.launch(
                bind_pins, dim=count,
                inputs=[
                    mesh.id, index, gpu_points, gpu_pinned, best, self.bound_collider,
                    self.bound_face, self.barycentric, self.local_offset,
                ],
            )
        if np.any(self.bound_collider.numpy()[pinned] < 0):
            raise ValueError("Some pins could not be attached to the moving collision meshes.")
        self.count = count

    def load_interval(self, interval, hold=False, part=0, parts=1):
        """Load the move for one simulated frame: a whole timeline interval, or part ``part`` of
        ``parts`` equal pieces of it (fast frames are split so the cloth can follow)."""
        for index, (_, first, second) in self.resources.items():
            if interval == WRAP:
                # Loop join: one real step from the last frame into frame 0 (straight moves: the rigid-limb paths
                # only join neighbouring frames).
                left, right = self.motion.positions(index, self.motion.intervals), self.motion.positions(index, 0)
                if parts > 1:
                    start, end = part / parts, (part + 1) / parts
                    left, right = left + (right - left) * start, left + (right - left) * end
            elif hold:
                left = self.motion.positions(index, min(interval, self.motion.intervals))
                right = left
            elif getattr(self.motion, "rigid", None) is not None and interval < self.motion.intervals:
                # Rigid limbs: in-between poses rotate parts about their joints (rigid_motion).
                rigid = self.motion.rigid
                left = rigid.positions(index, interval, part / parts)
                right = rigid.positions(index, interval, (part + 1) / parts)
            else:
                left, right = self.motion.pair(index, interval)
                if parts > 1:
                    start, end = part / parts, (part + 1) / parts
                    left, right = left + (right - left) * start, left + (right - left) * end
            first.assign(np.ascontiguousarray(left, dtype=np.float32))
            second.assign(np.ascontiguousarray(right, dtype=np.float32))

    def update(self, alpha, frame_dt, positions, anchors):
        for index, (mesh, first, second) in self.resources.items():
            wp.launch(
                deform_mesh, dim=len(first),
                inputs=[first, second, alpha, frame_dt, mesh.points, mesh.velocities],
            )
            mesh.refit()
            wp.launch(
                move_pins, dim=self.count,
                inputs=[
                    mesh.id, index, self.bound_collider, self.bound_face,
                    self.barycentric, self.local_offset, positions, anchors,
                ],
            )
