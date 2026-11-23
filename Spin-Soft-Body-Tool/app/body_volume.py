# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Shared inertia and approximate volume preservation for painted jiggle regions."""
import numpy as np
from body_geometry import connected_labels


class PaintedVolume:
    def __init__(self, mobility, neighbours, faces, source_cross):
        labels, count = connected_labels(neighbours, mobility > 0)
        self.active = np.flatnonzero(labels >= 0)
        self.regions = labels[self.active]
        self.count = count
        self.mobility = mobility[self.active]
        face_regions = labels[faces].max(axis=1)
        self.face_mask = face_regions >= 0
        self.face_regions = face_regions[self.face_mask]
        # Include triangles along the rim, even when some of their vertices are fixed.
        patch_faces = faces[self.face_mask]
        # Isolated vertices can still have inertia but have no enclosed surface.
        self.patch_vertices = np.union1d(patch_faces, self.active)
        self.faces = np.searchsorted(self.patch_vertices, patch_faces)
        self.patch_active = np.searchsorted(self.patch_vertices, self.active)
        area = np.linalg.norm(source_cross[self.face_mask], axis=1)
        vertex_area = np.zeros(len(self.patch_vertices), dtype=np.float32)
        np.add.at(vertex_area, self.faces.ravel(), np.repeat(area, 3))
        self.weights = np.maximum(vertex_area[self.patch_active] * self.mobility, 1e-20)
        self.total_weight = self._sum(self.regions, self.weights)

    def _sum(self, labels, values):
        return np.bincount(labels, weights=values, minlength=self.count)

    def shared_impulse(self, impulse):
        result = np.zeros_like(impulse)
        for axis in range(3):
            totals = self._sum(self.regions, impulse[self.active, axis] * self.weights)
            result[self.active, axis] = (totals / self.total_weight)[self.regions]
        return result

    def restore(self, offset, pose, pose_cross, limit):
        """Redistribute swept volume before the contact solve, with a fixed rim."""
        reference = pose_cross[self.face_mask]
        posed = pose[self.patch_vertices][self.faces]
        edge_a = posed[:, 1] - posed[:, 0]
        edge_b = posed[:, 2] - posed[:, 0]
        for _ in range(2):
            delta = offset[self.patch_vertices][self.faces]
            change_a = delta[:, 1] - delta[:, 0]
            change_b = delta[:, 2] - delta[:, 0]
            # The average area vector during a linear sweep is the mean of its
            # two endpoint values minus one sixth of the quadratic term.
            quadratic = np.cross(change_a, change_b)
            current = np.cross(edge_a + change_a, edge_b + change_b)
            swept = np.einsum('ij,ij->i', (reference + current) * .5 - quadratic / 6., delta.sum(axis=1))
            error = self._sum(self.face_regions, swept)
            gradients = np.zeros((len(self.patch_vertices), 3), dtype=np.float32)
            np.add.at(gradients, self.faces.ravel(), np.repeat(current, 3, axis=0))
            normals = gradients[self.patch_active]
            area = np.linalg.norm(normals, axis=1)
            support = self._sum(self.regions, area * self.mobility)
            pressure = np.clip(-error / np.maximum(support, 1e-20), -limit, limit)
            scale = pressure[self.regions] * self.mobility / np.maximum(area, 1e-20)
            offset[self.active] += normals * scale[:, None]
