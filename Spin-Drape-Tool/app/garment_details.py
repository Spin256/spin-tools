# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Move detached pockets/buttons with the simulated fabric under them."""
import numpy as np
import warp as wp
from moving_colliders import surface_basis


@wp.kernel
def bind_details(mesh: wp.uint64, indices: wp.array(dtype=int), rest: wp.array(dtype=wp.vec3),
                 binding_points: wp.array(dtype=wp.vec3), faces: wp.array(dtype=int),
                 uv: wp.array(dtype=wp.vec2), offsets: wp.array(dtype=wp.vec3)):
    slot = wp.tid()
    point = rest[indices[slot]]
    query = wp.mesh_query_point_no_sign(mesh, binding_points[indices[slot]], 1.0e6)
    if query.result:
        faces[slot] = query.face
        uv[slot] = wp.vec2(query.u, query.v)
        surface = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        tangent, bitangent, normal = surface_basis(mesh, query.face)
        delta = point - surface
        offsets[slot] = wp.vec3(wp.dot(delta, tangent), wp.dot(delta, bitangent), wp.dot(delta, normal))


@wp.kernel
def move_details(mesh: wp.uint64, indices: wp.array(dtype=int), faces: wp.array(dtype=int),
                 uv: wp.array(dtype=wp.vec2), offsets: wp.array(dtype=wp.vec3), positions: wp.array(dtype=wp.vec3)):
    slot = wp.tid()
    face = faces[slot]
    if face >= 0:
        bary = uv[slot]
        tangent, bitangent, normal = surface_basis(mesh, face)
        offset = offsets[slot]
        positions[indices[slot]] = (wp.mesh_eval_position(mesh, face, bary[0], bary[1])
                                   + tangent * offset[0] + bitangent * offset[1] + normal * offset[2])


class GarmentDetails:
    def __init__(self, points, structure):
        self.resources = []
        rest = wp.array(points, dtype=wp.vec3)
        binding_points = points.copy()
        rigid = {int(group[0]) for group in structure.button_groups}
        # Details of each garment bind only to that garment's main fabric.
        groups = {}
        for indices, faces in structure.detail_groups:
            if int(indices[0]) in rigid:
                # A button follows one surface frame as a rigid piece. It must
                # not stretch between two panels when the front is left open.
                binding_points[indices] = points[indices].mean(axis=0)
            key = (int(faces[0, 0]), len(faces))
            if key not in groups:
                groups[key] = ([], faces)
            groups[key][0].extend(indices.tolist())
        binding_points = wp.array(binding_points, dtype=wp.vec3)
        for indices, triangles in groups.values():
            mesh = wp.Mesh(points=wp.clone(rest), indices=wp.array(triangles.ravel(), dtype=int))
            indices = wp.array(np.asarray(indices, dtype=np.int32), dtype=int)
            faces = wp.full(len(indices), -1, dtype=int)
            uv = wp.zeros(len(indices), dtype=wp.vec2)
            offsets = wp.zeros(len(indices), dtype=wp.vec3)
            wp.launch(bind_details, dim=len(indices), inputs=[mesh.id, indices, rest, binding_points, faces, uv, offsets])
            if np.any(faces.numpy() < 0):
                raise ValueError("Could not attach all garment details to the underlying fabric.")
            self.resources.append((mesh, indices, faces, uv, offsets))

    def update(self, positions):
        for mesh, indices, faces, uv, offsets in self.resources:
            wp.copy(mesh.points, positions)
            mesh.refit()
            wp.launch(move_details, dim=len(indices), inputs=[mesh.id, indices, faces, uv, offsets, positions])
