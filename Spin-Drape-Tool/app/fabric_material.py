# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Approximate fabric presets and reinforcement distance along open mesh edges."""
import heapq
import numpy as np

# Adjustable starting points, not measured material calibrations.
FABRICS = {
    'Cotton': dict(stretch=1., bending=12., weight=150., damping=2.4, friction=.6),
    'Denim': dict(stretch=.5, bending=45., weight=400., damping=3., friction=.7),
    'Leather': dict(stretch=.25, bending=75., weight=600., damping=4., friction=.7),
    'Silk / lightweight': dict(stretch=1.5, bending=5., weight=80., damping=1.5, friction=.35),
}


def edge_distances(points, faces, width):
    """Shortest edge-path distance to an open boundary, without crossing layers."""
    distances = np.full(len(points), np.inf)
    if width <= 0 or not len(faces):
        return distances
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    boundary = np.unique(edges[counts == 1])
    adjacency = [[] for _ in points]
    for (a, b), length in zip(edges, np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)):
        adjacency[a].append((int(b), float(length)))
        adjacency[b].append((int(a), float(length)))
    queue = [(0., int(v)) for v in boundary]
    heapq.heapify(queue)
    distances[boundary] = 0.
    while queue:
        distance, vertex = heapq.heappop(queue)
        if distance != distances[vertex]:
            continue
        for neighbor, length in adjacency[vertex]:
            candidate = distance + length
            if candidate <= width and candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances
