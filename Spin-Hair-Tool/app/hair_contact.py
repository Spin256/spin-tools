# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Persistent, releasable contacts between two opposing collider surfaces (cm).

Keep the selected material coordinates, rather than choosing a new nearest side
on every solver pass. No scalp classification or asset-specific rules.
"""
import itertools
import warp as wp


@wp.func
def basis(mesh: wp.uint64, face: int):
    # Warp's mesh_get_point indexes face vertices (it already dereferences indices).
    a = wp.mesh_get_point(mesh, face * 3)
    b = wp.mesh_get_point(mesh, face * 3 + 1)
    c = wp.mesh_get_point(mesh, face * 3 + 2)
    x = wp.normalize(b - a)
    z = wp.normalize(wp.cross(x, c - a))
    y = wp.cross(z, x)
    return wp.transpose(wp.mat33(x, y, z))


@wp.kernel
def contact(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
            vel: wp.array(dtype=wp.vec3), inv: wp.array(dtype=float),
            base: wp.array(dtype=float), owner: wp.array(dtype=int),
            faces: wp.array(dtype=wp.vec2i), uv: wp.array(dtype=wp.vec4),
            offset: wp.array(dtype=wp.vec3), width: wp.array(dtype=float), share: wp.array(dtype=float),
            targets: wp.array(dtype=wp.vec3),
            first: wp.uint64, second: wp.uint64, pair: int, distance: float):
    i = wp.tid()
    if base[i] > 0.0 and ((owner[i] == -1 and inv[i] > 0.0) or owner[i] == pair):
        p = pos[i]
        if owner[i] == pair:
            f = faces[i]
            t = uv[i]
            a = wp.mesh_eval_position(first, f[0], t[0], t[1])
            b = wp.mesh_eval_position(second, f[1], t[2], t[3])
            target = a + (b - a) * share[i] + basis(first, f[0]) * offset[i]
            # Evaluate the retained material contacts directly. Re-running a
            # nearest-surface search here would allow the chosen contact to flip.
            if (wp.length(b - a) > width[i] + wp.max(0.2, distance * 0.3)
                or wp.length(target - a) > distance * 1.5 or wp.length(target - b) > distance * 1.5):
                owner[i] = -1
                inv[i] = base[i]
                prev[i] = p
                vel[i] = wp.vec3(0.0)
            else:
                pos[i] = target
                targets[i] = target
                prev[i] = target
                vel[i] = wp.vec3(0.0)
                inv[i] = 0.0
        else:
            qa = wp.mesh_query_point(first, p, distance)
            qb = wp.mesh_query_point(second, p, distance)
            if qa.result and qb.result:
                a = wp.mesh_eval_position(first, qa.face, qa.u, qa.v)
                b = wp.mesh_eval_position(second, qb.face, qb.u, qb.v)
                da = a - p
                db = b - p
                # Two nearby surfaces must face one another across the hair.
                # Mere proximity at a neck/shoulder corner does not qualify.
                if wp.length(da) > 0.0001 and wp.length(db) > 0.0001:
                    if wp.dot(wp.normalize(da), wp.normalize(db)) < -0.7 and wp.abs(wp.determinant(basis(first, qa.face))) > 0.99:
                        owner[i] = pair
                        faces[i] = wp.vec2i(qa.face, qb.face)
                        uv[i] = wp.vec4(qa.u, qa.v, qb.u, qb.v)
                        share[i] = wp.clamp(wp.dot(p - a, b - a) / wp.dot(b - a, b - a), 0.0, 1.0)
                        offset[i] = wp.transpose(basis(first, qa.face)) * (p - a - (b - a) * share[i])
                        width[i] = wp.length(b - a)
                        inv[i] = 0.0
                        targets[i] = p
                        prev[i] = p
                        vel[i] = wp.vec3(0.0)


@wp.kernel
def retain(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
           vel: wp.array(dtype=wp.vec3), owner: wp.array(dtype=int), targets: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if owner[i] >= 0:
        pos[i] = targets[i]
        prev[i] = targets[i]
        vel[i] = wp.vec3(0.0)


class TrappedContact:
    def __init__(self, meshes, inv, distance):
        self.meshes = meshes
        self.pairs = list(itertools.combinations(range(len(meshes)), 2))
        self.distance = distance
        self.base = wp.clone(inv)
        n = len(inv)
        self.owner = wp.full(n, -1, dtype=int)
        self.faces = wp.zeros(n, dtype=wp.vec2i)
        self.uv = wp.zeros(n, dtype=wp.vec4)
        self.offset = wp.zeros(n, dtype=wp.vec3)
        self.width = wp.zeros(n, dtype=float)
        self.share = wp.zeros(n, dtype=float)
        self.targets = wp.zeros(n, dtype=wp.vec3)

    def update(self, pos, prev, vel, inv):
        for pair, (a, b) in enumerate(self.pairs):
            wp.launch(contact, len(pos), inputs=[pos, prev, vel, inv, self.base,
                      self.owner, self.faces, self.uv, self.offset, self.width, self.share, self.targets,
                      self.meshes[a].id, self.meshes[b].id, pair, self.distance])

    def retain(self, pos, prev, vel):
        wp.launch(retain, len(pos), inputs=[pos, prev, vel, self.owner, self.targets])

    def reset(self, inv):
        self.owner.fill_(-1)
        wp.copy(inv, self.base)
