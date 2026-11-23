# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Connected surface painting and one-ring selection growth/erosion."""
import numpy as np


class FabricSelection:
    def __init__(self, count, faces):
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        neighbors = [set() for _ in range(count)]
        for a, b in edges:
            neighbors[a].add(int(b))
            neighbors[b].add(int(a))
        self.neighbors = [np.asarray(sorted(values), dtype=np.int32) for values in neighbors]
        self.boundary = np.zeros(count, dtype=bool)
        self.boundary[np.unique(edges[counts == 1])] = True

    def brush(self, points, seed, radius, allowed):
        inside = (np.linalg.norm(points - points[seed], axis=1) <= radius) & allowed
        found = np.zeros(len(points), dtype=bool)
        if not inside[seed]:
            return found
        pending = [int(seed)]
        found[seed] = True
        while pending:
            neighbors = self.neighbors[pending.pop()]
            for vertex in neighbors[inside[neighbors] & ~found[neighbors]]:
                found[vertex] = True
                pending.append(int(vertex))
        return found

    def resize(self, selection, grow, allowed):
        result = selection.copy()
        for vertex in np.flatnonzero(selection):
            neighbors = self.neighbors[vertex]
            if grow:
                result[neighbors] = True
            elif self.boundary[vertex] or np.any(~selection[neighbors]):
                result[vertex] = False
        return result & allowed
