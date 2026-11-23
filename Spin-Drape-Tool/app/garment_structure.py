# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Identify detached garment details and optional button fastenings in the rest mesh."""
from collections import Counter
import heapq
import numpy as np


class GarmentStructure:
    def __init__(self, points, faces, ranges=None, button_vertices=()):
        self.points = points
        self.detail_groups = []
        self.button_groups = []
        self.body_faces = []
        self.fasteners = []
        self.component_count = 0
        ranges = ranges or [(0, len(points))]
        buttons = set(button_vertices)
        parent = np.arange(len(points))
        sizes = np.ones(len(points), dtype=np.int32)

        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for first, second, third in faces:
            for a, b in ((first, second), (first, third)):
                a, b = root(a), root(b)
                if a != b:
                    if sizes[a] < sizes[b]:
                        a, b = b, a
                    parent[b] = a
                    sizes[a] += sizes[b]
        labels = np.array([root(i) for i in range(len(points))])
        triangle_area = .5 * np.linalg.norm(
            np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]]), axis=1)
        face_labels = labels[faces[:, 0]]
        self.attached = np.zeros(len(points), dtype=bool)
        for start, end in ranges:
            mask = (faces[:, 0] >= start) & (faces[:, 0] < end)
            local_faces = faces[mask]
            areas = Counter()
            for label, area in zip(face_labels[mask], triangle_area[mask]):
                areas[int(label)] += float(area)
            self.component_count += len(areas)
            if not areas:
                continue
            largest = max(areas, key=areas.get)
            main_vertices = np.flatnonzero((labels == largest) & (np.arange(len(points)) >= start) & (np.arange(len(points)) < end))
            details = set()
            detached = []
            # Large disconnected panels remain cloth. Only small satellite pieces
            # bind to this garment's own largest fabric component.
            for label, area in areas.items():
                if label == largest:
                    continue
                group = np.flatnonzero((labels == label) & (np.arange(len(points)) >= start) & (np.arange(len(points)) < end))
                if len(group) <= len(main_vertices) * .2 and area <= areas[largest] * .15:
                    details.add(label)
                    detached.append(group)
                    self.attached[group] = True
            body = local_faces[~np.isin(labels[local_faces[:, 0]], list(details))]
            named_buttons = any(start <= v < end for v in buttons)
            for group in detached:
                self.detail_groups.append((group, body))
                if any(int(v) in buttons for v in group) or (
                        not named_buttons and self._compact_button(points, local_faces, group)):
                    self.button_groups.append(group)
                    self.fasteners.extend(self._button_fasteners(points, body, group))
            self.body_faces.extend(body)
        self.body_faces = np.asarray(self.body_faces, dtype=np.int32).reshape(-1, 3)
        self.fasteners = sorted(set(tuple(sorted(pair)) for pair in self.fasteners))

    @staticmethod
    def _compact_button(points, faces, group):
        # Re-imported garments can have generic material names. Recognize only
        # small, rounded, densely meshed details; large/open pockets stay flexible.
        if len(group) < 24:
            return False
        vertices = points[group]
        if np.linalg.norm(np.ptp(vertices, axis=0)) > .04:
            return False
        singular = np.linalg.svd(vertices - vertices.mean(axis=0), compute_uv=False)
        if singular[0] < 1.e-8 or singular[1] / singular[0] < .5:
            return False
        triangles = faces[np.isin(faces[:, 0], group)]
        edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = np.unique(edges[counts == 1])
        return bool(np.all(counts <= 2) and len(boundary) / len(group) <= .1)

    @staticmethod
    def _button_fasteners(points, faces, button):
        edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = np.unique(edges[counts == 1])
        if not len(boundary):
            return []
        center = points[button].mean(axis=0)
        diameter = float(np.linalg.norm(np.ptp(points[button], axis=0)))
        radius = min(.05, max(.015, diameter * 1.5))
        distances = np.linalg.norm(points[boundary] - center, axis=1)
        nearby = boundary[distances <= radius]
        if len(nearby) < 2:
            return []
        first = int(nearby[np.argmin(np.linalg.norm(points[nearby] - center, axis=1))])
        # Reject neighbours on the same edge/panel by walking a short distance
        # through the fabric. Opposite front edges are close in space but far
        # apart through the vest's back. Armholes/hem outside the button area
        # never enter the candidate set.
        adjacency = {}
        for a, b in edges:
            length = float(np.linalg.norm(points[b] - points[a]))
            adjacency.setdefault(int(a), []).append((int(b), length))
            adjacency.setdefault(int(b), []).append((int(a), length))
        cutoff = max(.1, radius * 4)
        distance = {first: 0.0}
        queue = [(0.0, first)]
        while queue:
            length, vertex = heapq.heappop(queue)
            if length != distance[vertex]:
                continue
            for neighbor, edge_length in adjacency.get(vertex, ()):
                candidate = length + edge_length
                if candidate <= cutoff and candidate < distance.get(neighbor, float("inf")):
                    distance[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        other = np.array([v for v in nearby if int(v) not in distance], dtype=np.int32)
        if not len(other):
            return []
        second = int(other[np.argmin(np.linalg.norm(points[other] - points[first], axis=1))])
        if np.linalg.norm(points[second] - points[first]) > radius:
            return []
        return [(first, second)]
