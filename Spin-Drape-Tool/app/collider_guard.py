# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Keep cloth outside colliders after the fabric constraints (Spin256 · spin256.com/tools).

Each substep Warp pushes cloth particles out of colliders, then Drape's fabric constraints run.
An inextensible skirt held at the waist can pull particles back through a lifting leg in that
second step, after collision has been handled. This final projection moves every free particle
that ended closer than the collision radius (or inside) back to the outside of each collider.
"""
import warp as wp


@wp.kernel
def keep_outside(positions: wp.array(dtype=wp.vec3), inverse_mass: wp.array(dtype=float),
                 mesh: wp.uint64, radius: float, reach: float):
    i = wp.tid()
    if inverse_mass[i] > 0.:
        query = wp.mesh_query_point_sign_normal(mesh, positions[i], reach)
        if query.result:
            surface = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            delta = positions[i] - surface
            distance = wp.length(delta)
            if distance * query.sign < radius and distance > 1.e-9:
                outward = delta * (query.sign / distance)
                positions[i] = surface + outward * radius


@wp.kernel
def keep_above_ground(positions: wp.array(dtype=wp.vec3), previous: wp.array(dtype=wp.vec3),
                      inverse_mass: wp.array(dtype=float), height: float, radius: float, grip: float):
    """Infinite floor at ``height`` (Y up). Points pushed up to it lose ``grip`` of their sliding
    along the floor this substep (0: slide freely, 1: stay where they touched down)."""
    i = wp.tid()
    if inverse_mass[i] > 0.:
        p = positions[i]
        floor = height + radius
        if p[1] < floor:
            q = previous[i]
            slide_x = (p[0] - q[0]) * (1. - grip)
            slide_z = (p[2] - q[2]) * (1. - grip)
            positions[i] = wp.vec3(q[0] + slide_x, floor, q[2] + slide_z)


class ColliderGuard:
    """``ground``: None or dict(height=meters, grip=0-1)."""

    def __init__(self, sources, count, radius, reach=.05, ground=None):
        self.sources, self.count, self.radius, self.reach = list(sources), count, radius, reach
        self.ground = ground

    def project(self, positions, inverse_mass, previous=None):
        for source in self.sources:
            wp.launch(keep_outside, dim=self.count,
                      inputs=[positions, inverse_mass, source.mesh.id, self.radius, self.reach])
        if self.ground is not None:
            wp.launch(keep_above_ground, dim=self.count,
                      inputs=[positions, positions if previous is None else previous, inverse_mass,
                              float(self.ground["height"]), self.radius, float(self.ground.get("grip", .5))])


class FullMeshGuard:
    """The same projection for a full garment carried on a proxy: run on the full-resolution
    vertices of each shown, recorded or sent frame, against the colliders' pose for that frame."""

    def __init__(self, colliders, radius, reach=.05, ground=None):
        import numpy as np
        self.np = np
        self.radius, self.reach, self.ground = radius, reach, ground
        self.meshes = [wp.Mesh(wp.array(np.asarray(points, np.float32), dtype=wp.vec3),
                               wp.array(np.asarray(faces, np.int32).ravel(), dtype=int))
                       for points, faces in colliders]
        self.masses = None

    def project(self, vertices_cm, moving_cm=None):
        np = self.np
        for index, positions in (moving_cm or {}).items():
            if index < len(self.meshes):
                self.meshes[index].points.assign(np.ascontiguousarray(np.asarray(positions) * .01, dtype=np.float32))
                self.meshes[index].refit()
        points = wp.array(np.ascontiguousarray(np.asarray(vertices_cm) * .01, dtype=np.float32), dtype=wp.vec3)
        if self.masses is None or self.masses.shape[0] != points.shape[0]:
            self.masses = wp.ones(points.shape[0], dtype=float)
        for mesh in self.meshes:
            wp.launch(keep_outside, dim=points.shape[0], inputs=[points, self.masses, mesh.id, self.radius, self.reach])
        if self.ground is not None:
            wp.launch(keep_above_ground, dim=points.shape[0],
                      inputs=[points, points, self.masses, float(self.ground["height"]), self.radius, 0.])
        return points.numpy() * 100.
