# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Distance constraints with disjoint vertex groups for stable GPU projection."""

import heapq

import numpy as np
import warp as wp
from fabric_material import edge_distances


@wp.kernel
def project_bends(positions: wp.array(dtype=wp.vec3), inverse_mass: wp.array(dtype=float),
                  edges: wp.array2d(dtype=int), lengths: wp.array(dtype=float), strengths: wp.array(dtype=float)):
    i = wp.tid()
    a, b = edges[i, 0], edges[i, 1]
    delta = positions[b] - positions[a]
    length = wp.length(delta)
    total = inverse_mass[a] + inverse_mass[b]
    if length > 1.e-9 and total > 0.:
        correction = delta * ((length - lengths[i]) * strengths[i] / (length * total))
        positions[a] = positions[a] + correction * inverse_mass[a]
        positions[b] = positions[b] - correction * inverse_mass[b]


@wp.kernel
def project_edges(
    positions: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    edges: wp.array2d(dtype=int),
    lengths: wp.array(dtype=float),
    stretch: wp.array(dtype=float),
    strength: float,
):
    constraint = wp.tid()
    first = edges[constraint, 0]
    second = edges[constraint, 1]
    delta = positions[second] - positions[first]
    length = wp.length(delta)
    total = inverse_mass[first] + inverse_mass[second]
    if length > 1.0e-9 and total > 0.0:
        target = wp.clamp(length, lengths[constraint] * (1.0 - stretch[constraint]), lengths[constraint] * (1.0 + stretch[constraint]))
        correction = delta * ((length - target) * strength / (length * total))
        positions[first] = positions[first] + correction * inverse_mass[first]
        positions[second] = positions[second] - correction * inverse_mass[second]


@wp.kernel
def project_tethers(
    positions: wp.array(dtype=wp.vec3),
    anchors: wp.array(dtype=wp.vec3),
    anchor_index: wp.array(dtype=int),
    maximum_length: wp.array(dtype=float),
    inverse_mass: wp.array(dtype=float),
):
    index = wp.tid()
    anchor = anchor_index[index]
    if anchor >= 0 and inverse_mass[index] > 0.0:
        delta = positions[index] - anchors[anchor]
        length = wp.length(delta)
        if length > maximum_length[index] and length > 1.0e-9:
            positions[index] = anchors[anchor] + delta * (maximum_length[index] / length)


@wp.kernel
def update_velocity(
    previous: wp.array(dtype=wp.vec3),
    current: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    inverse_mass: wp.array(dtype=float),
    dt: float,
    damping: float,
):
    index = wp.tid()
    if inverse_mass[index] > 0.0:
        velocity[index] = (current[index] - previous[index]) * (wp.exp(-damping * dt) / dt)


# Bend settings were tuned at 8 solver iterations x 8 substeps (static 60 fps).
REFERENCE_PASSES = 64


def per_pass_strength(strength, iterations, substeps):
    """Per-pass bend strength giving the same correction per simulated frame.

    Each pass removes the fraction s of the remaining error, so N passes leave
    (1 - s) ** N. Matching the reference pass count makes bend resistance
    independent of solver iterations and of the timeline frame rate. Without
    substeps (unit tests of a single projection) the raw strength is used.
    """
    strength = np.clip(np.asarray(strength, dtype=np.float64), 0., 1.)
    if substeps is None:
        return strength.astype(np.float32)
    passes = max(1, int(iterations) * int(substeps))
    scaled = 1. - np.power(1. - strength, REFERENCE_PASSES / passes)
    return np.where(strength >= 1., 1., scaled).astype(np.float32)


class ClothConstraints:
    def __init__(self, points, faces, masses, pinned, *, stretch=0.0, iterations=8, fasteners=(), bending=0.0,
                 fastener_gap=None, edge_width=0., edge_stiffness=0., crease_angle=30., crease_stiffness=0.,
                 substeps=None):
        self.stretch = stretch
        self.iterations = iterations
        self.bending = bending
        self.crease_count = 0
        self.edge_distance = edge_distances(points, faces, edge_width)
        self.edge_band = self.edge_distance <= edge_width if edge_width > 0 else np.zeros(len(points), dtype=bool)
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        mesh_edges = np.unique(np.sort(edges, axis=1), axis=0)
        # A fastening on an existing fabric edge (or on one vertex) would
        # override that edge's material length and stretch, so it is ignored.
        mesh_edge_set = {tuple(map(int, edge)) for edge in mesh_edges}
        requested = {tuple(sorted(map(int, pair))) for pair in np.asarray(fasteners, dtype=np.int32).reshape(-1, 2)}
        fasteners = sorted(pair for pair in requested if pair[0] != pair[1] and pair not in mesh_edge_set)
        self.ignored_fasteners = len(requested) - len(fasteners)
        if len(fasteners):
            edges = np.concatenate((edges, np.asarray(fasteners, dtype=np.int32).reshape(-1, 2)))
        self.edges = np.unique(np.sort(edges, axis=1), axis=0)
        self.rest_lengths = np.linalg.norm(
            points[self.edges[:, 0]] - points[self.edges[:, 1]], axis=1
        )
        self.inverse_mass = wp.array(
            np.divide(1.0, masses, out=np.zeros_like(masses), where=masses > 0),
            dtype=float,
        )

        # Each group touches a vertex at most once, so full corrections do not race.
        colors = np.zeros(len(points), dtype=object)
        groups = []
        for edge in self.edges:
            used = colors[edge[0]] | colors[edge[1]]
            color = 0
            while used & (1 << color):
                color += 1
            while len(groups) <= color:
                groups.append([])
            groups[color].append(edge)
            colors[edge[0]] |= 1 << color
            colors[edge[1]] |= 1 << color
        self.groups = []
        fastening_pairs = {tuple(sorted(pair)) for pair in fasteners}
        self.fastening_groups = []
        for group in groups:
            pairs = np.asarray(group, dtype=np.int32)
            lengths = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
            fastening = np.asarray([tuple(pair) in fastening_pairs for pair in pairs])
            if fastener_gap is not None:
                lengths[fastening] = np.minimum(lengths[fastening], fastener_gap)
            allowances = np.asarray([0.0 if tuple(pair) in fastening_pairs else stretch for pair in pairs], dtype=np.float32)
            self.groups.append((wp.array(pairs, dtype=int), wp.array(lengths, dtype=float), wp.array(allowances, dtype=float)))
            if np.any(fastening):
                self.fastening_groups.append((wp.array(pairs[fastening], dtype=int),
                                              wp.array(lengths[fastening], dtype=float), wp.zeros(int(fastening.sum()), dtype=float)))

        # Opposite vertices across a shared triangle edge encode local rest
        # curvature. Gentle distance-based bending resists tiny checkerboard
        # folds without changing the garment's starting pleats or triangulation.
        self.bending_groups = []
        self.bending_strengths = []
        if bending > 0 or (edge_width > 0 and edge_stiffness > 0) or crease_stiffness > 0:
            adjacent = {}
            for a, b, c in faces:
                for first, second, opposite in ((a, b, c), (b, c, a), (c, a, b)):
                    adjacent.setdefault(tuple(sorted((int(first), int(second)))), []).append(int(opposite))
            bend_edges = sorted({tuple(sorted(opposite)) for opposite in adjacent.values()
                                 if len(opposite) == 2 and opposite[0] != opposite[1]})
            strength_by_pair = {pair: bending for pair in bend_edges}
            if edge_width > 0 and edge_stiffness > 0:
                for edge, opposite in adjacent.items():
                    if len(opposite) == 2 and opposite[0] != opposite[1] and all(self.edge_band[list(edge)]):
                        pair = tuple(sorted(opposite))
                        strength_by_pair[pair] = bending + (1. - bending) * edge_stiffness
            if crease_stiffness > 0:
                for edge, opposite in adjacent.items():
                    if len(opposite) != 2 or opposite[0] == opposite[1]:
                        continue
                    a, b = points[list(edge)]
                    first = np.cross(b - a, points[opposite[0]] - a)
                    second = np.cross(points[opposite[1]] - a, b - a)
                    length = np.linalg.norm(first) * np.linalg.norm(second)
                    if length > 1.e-20:
                        angle = np.degrees(np.arccos(np.clip(np.dot(first, second) / length, -1., 1.)))
                        if angle > crease_angle + 1.e-5:
                            pair = tuple(sorted(opposite))
                            strength_by_pair[pair] = max(strength_by_pair[pair], bending + (1. - bending) * crease_stiffness)
                            self.crease_count += 1
            colors = np.zeros(len(points), dtype=object)
            groups = []
            for edge in bend_edges:
                used = colors[edge[0]] | colors[edge[1]]
                color = 0
                while used & (1 << color):
                    color += 1
                while len(groups) <= color:
                    groups.append([])
                groups[color].append(edge)
                colors[edge[0]] |= 1 << color
                colors[edge[1]] |= 1 << color
            for group in groups:
                pairs = np.asarray(group, dtype=np.int32)
                lengths = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
                self.bending_groups.append((wp.array(pairs, dtype=int), wp.array(lengths, dtype=float),
                                            wp.zeros(len(pairs), dtype=float)))
                strengths = np.asarray([strength_by_pair[tuple(pair)] for pair in pairs], dtype=np.float64)
                self.bending_strengths.append(wp.array(per_pass_strength(strengths, iterations, substeps), dtype=float))

        # Limit total extension along the cloth to its nearest held vertex.
        # Paths follow real fabric only: a fastening is not a cloth path, so
        # it must not shorten the tether limit across an opening.
        adjacency = [[] for _ in points]
        mesh_lengths = np.linalg.norm(points[mesh_edges[:, 0]] - points[mesh_edges[:, 1]], axis=1)
        for (first, second), distance in zip(mesh_edges, mesh_lengths):
            adjacency[first].append((second, float(distance)))
            adjacency[second].append((first, float(distance)))
        distance = np.full(len(points), np.inf)
        anchor_index = np.full(len(points), -1, dtype=np.int32)
        queue = []
        for index in np.flatnonzero(pinned):
            distance[index] = 0.0
            anchor_index[index] = index
            heapq.heappush(queue, (0.0, int(index)))
        while queue:
            length, index = heapq.heappop(queue)
            if length != distance[index]:
                continue
            for neighbor, edge_length in adjacency[index]:
                candidate = length + edge_length
                if candidate < distance[neighbor]:
                    distance[neighbor] = candidate
                    anchor_index[neighbor] = anchor_index[index]
                    heapq.heappush(queue, (candidate, neighbor))
        self.unattached = int(np.count_nonzero((anchor_index < 0) & (masses > 0)))
        self.anchors = wp.array(points, dtype=wp.vec3)
        self.anchor_index = wp.array(anchor_index, dtype=int)
        self.maximum_length = wp.array(distance * (1.0 + max(0.02, stretch)), dtype=float)
        self.count = len(points)

    def project(self, positions):
        for _ in range(self.iterations):
            for edges, lengths, allowances in self.groups:
                wp.launch(
                    project_edges, dim=len(lengths),
                    inputs=[positions, self.inverse_mass, edges, lengths, allowances, 1.0],
                )
            for (edges, lengths, _), strengths in zip(self.bending_groups, self.bending_strengths):
                wp.launch(project_bends, dim=len(lengths),
                          inputs=[positions, self.inverse_mass, edges, lengths, strengths])
            wp.launch(
                project_tethers, dim=self.count,
                inputs=[
                    positions, self.anchors, self.anchor_index,
                    self.maximum_length, self.inverse_mass,
                ],
            )
            # Bending and long-range tethers must not undo a fastening at the
            # end of a projection pass.
            for edges, lengths, allowances in self.fastening_groups:
                wp.launch(project_edges, dim=len(lengths),
                          inputs=[positions, self.inverse_mass, edges, lengths, allowances, 1.0])
