# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Rigid-limb interpolation between two captured avatar poses (Spin256 · spin256.com/tools).

Moving every vertex in a straight line between DAZ frames would shorten a leg swinging through a
large angle and cut corners. Here, without any rig:

1. the avatar (frame A) is split into small surface patches (drape_proxy clustering, along the
   mesh, never across gaps);
2. each patch gets the rigid motion (rotation + translation) that best maps it from A to B;
3. neighboring patches that move as one rigid piece are merged into parts (thigh, shin, foot,
   forearm, ...), accepting a merge only while the merged part still fits within TOLERANCE;
4. parts form a tree from the largest part (the torso); each part's motion relative to its
   parent becomes a rotation about the joint where they meet;
5. at time s every part rotates by s of its relative rotation about its joint (shortest arc),
   on top of its parent, so limbs keep their length and swing in arcs; the root moves along a
   screw (rotation about an axis plus slide along it);
6. vertices near a joint blend their part with the neighboring part; the small remainder that
   is not rigid (fit residual, morphs between the frames) is added linearly, so s=0 is exactly
   A and s=1 exactly B.
"""
import math

import numpy as np

from drape_proxy import ProxyMesh, _edges

TOLERANCE = .0025            # metres: RMS fit error allowed when merging patches into one rigid part
PATCHES = 600                # surface patches per avatar
JOINT_BLEND = .015           # metres: how far from a joint vertices blend the two parts


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


def axis_angle(matrix):
    angle = math.acos(max(-1., min(1., (np.trace(matrix) - 1) / 2)))
    if angle < 1e-9:
        return np.array([0., 0., 1.]), 0.
    if math.pi - angle < 1e-6:                    # 180°: axis from the symmetric part
        values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
        return vectors[:, np.argmax(values)], angle
    axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    return axis / (2 * math.sin(angle)), angle


class Fit:
    """Additive sums for a best-fit rigid motion A -> B of a set of points (Kabsch from sums)."""

    def __init__(self, n=0, sa=None, sb=None, saa=0., sbb=0., m=None):
        self.n, self.saa, self.sbb = n, saa, sbb
        self.sa = np.zeros(3) if sa is None else sa
        self.sb = np.zeros(3) if sb is None else sb
        self.m = np.zeros((3, 3)) if m is None else m

    def __add__(self, other):
        return Fit(self.n + other.n, self.sa + other.sa, self.sb + other.sb,
                   self.saa + other.saa, self.sbb + other.sbb, self.m + other.m)

    def solve(self):
        """(R, t, rms) with B ≈ R A + t."""
        ca, cb = self.sa / self.n, self.sb / self.n
        h = self.m - self.n * np.outer(ca, cb)
        u, s, vt = np.linalg.svd(h)
        d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.
        r = vt.T @ np.diag([1, 1, d]) @ u.T
        residual = (self.saa - self.n * ca @ ca) + (self.sbb - self.n * cb @ cb) - 2 * np.trace(r @ h)
        return r, cb - r @ ca, math.sqrt(max(residual, 0.) / self.n)


def patch_fits(a, b, labels, count):
    fits = []
    n = np.bincount(labels, minlength=count).astype(float)
    sa, sb = np.zeros((count, 3)), np.zeros((count, 3))
    np.add.at(sa, labels, a)
    np.add.at(sb, labels, b)
    saa, sbb = np.bincount(labels, (a * a).sum(1), count), np.bincount(labels, (b * b).sum(1), count)
    m = np.zeros((count, 3, 3))
    np.add.at(m, labels, a[:, :, None] * b[:, None, :])
    for k in range(count):
        fits.append(Fit(n[k], sa[k], sb[k], saa[k], sbb[k], m[k]))
    return fits


class RigidPath:
    """Positions between pose A (s=0) and pose B (s=1) moving rigid parts in arcs."""

    def __init__(self, a, b, faces, patches=None, tolerance=TOLERANCE):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        self.a, self.b = a, b
        count = len(a)
        edges = _edges(faces)
        if patches is None:
            patches = ProxyMesh(a, faces, PATCHES).owner
        patch_count = int(patches.max()) + 1
        fits = patch_fits(a, b, patches, patch_count)
        # ----- merge patches into rigid parts (greedy, best agreements first) -----------
        pa, pb = patches[edges[:, 0]], patches[edges[:, 1]]
        cross = pa != pb
        pairs = np.unique(np.sort(np.column_stack([pa[cross], pb[cross]]), axis=1), axis=0)
        parent = np.arange(patch_count)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        part_fit = {k: fits[k] for k in range(patch_count)}
        scored = sorted(((fits[i] + fits[j]).solve()[2], i, j) for i, j in pairs.tolist())
        for _, i, j in scored:
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            merged = part_fit[ri] + part_fit[rj]
            if merged.solve()[2] <= tolerance:
                parent[rj] = ri
                part_fit[ri] = merged
                del part_fit[rj]
        roots = np.array([find(k) for k in range(patch_count)])
        _, part_of_patch = np.unique(roots, return_inverse=True)
        self.part = part_of_patch[patches]
        parts = int(self.part.max()) + 1
        self.parts = parts
        part_fits = patch_fits(a, b, self.part, parts)
        transforms = [f.solve() for f in part_fits]
        sizes = np.bincount(self.part, minlength=parts)
        # ----- part tree and joints -----------------------------------------------------
        qa, qb = self.part[edges[:, 0]], self.part[edges[:, 1]]
        between = qa != qb
        part_edges = np.column_stack([qa[between], qb[between]])
        neighbours = {k: {} for k in range(parts)}
        for i, j in part_edges.tolist():
            neighbours[i][j] = neighbours[i].get(j, 0) + 1
            neighbours[j][i] = neighbours[j].get(i, 0) + 1
        self.root = int(np.argmax(sizes))
        self.parent = np.full(parts, -1)
        self.order = []
        seen = np.zeros(parts, bool)
        for start in [self.root] + [int(k) for k in np.argsort(-sizes)]:
            if seen[start]:
                continue
            seen[start] = True                          # separate pieces each get their own root
            queue = [start]
            while queue:
                k = queue.pop(0)
                self.order.append(k)
                for other in sorted(neighbours[k], key=lambda o: -neighbours[k][o]):
                    if not seen[other]:
                        seen[other] = True
                        self.parent[other] = k
                        queue.append(other)
        self.motion = [None] * parts             # per part: ('root', screw) or ('child', axis, angle, pivot, slide)
        for k in self.order:
            r_k, t_k, _ = transforms[k]
            q = self.parent[k]
            if q < 0:
                axis, angle = axis_angle(r_k)
                slide = float(t_k @ axis) if angle > 1e-9 else 0.
                if angle > 1e-9:
                    lhs = np.eye(3) - r_k
                    pivot = np.linalg.lstsq(lhs, t_k - slide * axis, rcond=None)[0]
                    self.motion[k] = ("root", axis, angle, pivot, slide * axis)
                else:
                    self.motion[k] = ("root", axis, 0., np.zeros(3), t_k)
                continue
            r_q, t_q, _ = transforms[q]
            r_x = r_q.T @ r_k                          # motion relative to the parent, in A's frame
            t_x = r_q.T @ (t_k - t_q)
            boundary = np.unique(np.concatenate([edges[between][(qa[between] == k) & (qb[between] == q)].ravel(),
                                                 edges[between][(qa[between] == q) & (qb[between] == k)].ravel()]))
            centre = a[boundary].mean(0) if len(boundary) else a[self.part == k].mean(0)
            lhs = r_x - np.eye(3)
            weight = 1e-3
            pivot = np.linalg.solve(lhs.T @ lhs + weight * np.eye(3), lhs.T @ (-t_x) + weight * centre)
            axis, angle = axis_angle(r_x)
            self.motion[k] = ("child", axis, angle, pivot, r_x @ pivot + t_x - pivot)
        # ----- joint blending: vertices near a part boundary share the neighbour's motion --
        length = float(np.linalg.norm(a[edges[:, 0]] - a[edges[:, 1]], axis=1).mean()) if len(edges) else 1.
        rings = max(1, int(round(JOINT_BLEND / max(length, 1e-9))))
        seeds = np.unique(edges[between].ravel())
        same = edges[~between]
        other = np.full(count, -1)
        # nearest other part across the boundary, for each boundary vertex
        for u, v in ((edges[between][:, 0], edges[between][:, 1]), (edges[between][:, 1], edges[between][:, 0])):
            other[u] = self.part[v]
        distance = np.full(count, -1)
        distance[seeds] = 0
        frontier_other = other.copy()
        for step in range(1, rings + 1):
            grow_b = (distance[same[:, 0]] >= 0) & (distance[same[:, 1]] < 0)
            grow_a = (distance[same[:, 1]] >= 0) & (distance[same[:, 0]] < 0)
            if not grow_a.any() and not grow_b.any():
                break
            distance[same[grow_b, 1]] = step
            frontier_other[same[grow_b, 1]] = frontier_other[same[grow_b, 0]]
            distance[same[grow_a, 0]] = step
            frontier_other[same[grow_a, 0]] = frontier_other[same[grow_a, 1]]
        near = (distance >= 0) & (frontier_other >= 0)
        self.other = np.where(near, frontier_other, self.part)
        self.blend = np.where(near, .5 * np.clip(1 - distance / (rings + 1), 0, 1), 0.)
        self.residual = b - self._rigid(1.)

    def _transforms(self, s):
        world = [None] * self.parts
        for k in self.order:
            kind, axis, angle, pivot, slide = self.motion[k]
            r = rotation(axis, s * angle)
            t = pivot - r @ pivot + s * slide              # rotate about the pivot, slide by s
            if kind == "child":
                r_q, t_q = world[self.parent[k]]
                r, t = r_q @ r, r_q @ t + t_q
            world[k] = (r, t)
        return world

    def _rigid(self, s):
        world = self._transforms(s)
        r = np.stack([w[0] for w in world])
        t = np.stack([w[1] for w in world])
        own = np.einsum("nij,nj->ni", r[self.part], self.a) + t[self.part]
        neighbour = np.einsum("nij,nj->ni", r[self.other], self.a) + t[self.other]
        return own * (1 - self.blend)[:, None] + neighbour * self.blend[:, None]

    def at(self, s):
        """Positions at s in [0, 1] (exactly A at 0 and B at 1)."""
        if s <= 0:
            return self.a.astype(np.float32)
        if s >= 1:
            return self.b.astype(np.float32)
        return (self._rigid(s) + s * self.residual).astype(np.float32)

    def length(self, samples=8):
        """Longest path any vertex travels from A to B (meters), along the arcs."""
        total = np.zeros(len(self.a))
        previous = self.a
        for i in range(1, samples + 1):
            current = self.at(i / samples).astype(float)
            total += np.linalg.norm(current - previous, axis=1)
            previous = current
        return float(total.max())


class RigidPaths:
    """Rigid-limb paths for every moving collider and timeline interval of a ColliderMotion,
    built on first use; patches are clustered once per collider (topology does not change)."""

    def __init__(self, motion, faces):
        self.motion, self.faces = motion, faces
        self.paths, self.patches = {}, {}

    def path(self, index, interval):
        key = (index, interval)
        if key not in self.paths:
            first = self.motion.positions(index, min(interval, self.motion.intervals))
            second = self.motion.positions(index, min(interval + 1, self.motion.intervals))
            if index not in self.patches:
                self.patches[index] = ProxyMesh(first, self.faces[index], PATCHES).owner
            self.paths[key] = RigidPath(first, second, self.faces[index], self.patches[index])
        return self.paths[key]

    def positions(self, index, interval, s):
        return self.path(index, interval).at(s)

    def length(self, index, interval):
        return self.path(index, interval).length()
