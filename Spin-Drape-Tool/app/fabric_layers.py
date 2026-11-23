# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Find the nearest separate fabric layer along a picked triangle's normal."""
import numpy as np


def layer_pair(points, faces, cell_id, location, reach, excluded=None):
    triangle = faces[cell_id]
    a, b, c = points[triangle]
    normal = np.cross(b - a, c - a)
    length = np.linalg.norm(normal)
    if length < 1e-10:
        return None
    normal /= length
    source = int(triangle[np.argmin(np.linalg.norm(points[triangle] - location, axis=1))])
    # Cast both ways: exported winding need not point towards the other layer.
    origin = np.asarray(location)
    triangles = points[faces]
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    h = np.cross(np.broadcast_to(normal, e2.shape), e2)
    det = np.einsum('ij,ij->i', e1, h)
    valid = abs(det) > 1e-10
    inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
    s = origin - triangles[:, 0]
    u = inv * np.einsum('ij,ij->i', s, h)
    q = np.cross(s, e1)
    v = inv * (q @ normal)
    t = inv * np.einsum('ij,ij->i', e2, q)
    valid &= (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.000001)
    valid &= (abs(t) > 1e-4) & (abs(t) <= reach)
    valid &= ~np.any(np.isin(faces, triangle), axis=1)
    if excluded is not None:
        valid &= ~np.any(excluded[faces], axis=1)
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        return None
    hit = candidates[np.argmin(abs(t[candidates]))]
    hit_point = origin + normal * t[hit]
    target = int(faces[hit][np.argmin(np.linalg.norm(points[faces[hit]] - hit_point, axis=1))])
    return tuple(sorted((source, target)))
