# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Suggested paint areas from the DAZ example frames, and named regions.

The body may travel, turn or swing its limbs between frames; that is rigid motion of each small part of the
surface and is ignored. What is left after fitting a rigid move to each small patch (a breast reshaped, the
crotch deforming as the legs move, a bending joint) is suggested for painting. Suggestions are only offered:
the user clicks the ones they want."""
import numpy as np

from body_geometry import connected_labels


def _rigid_residual(rest, posed, cell):
    """Distance of each vertex from the best rigid move of its grid cell (rest -> posed)."""
    keys = np.floor(rest / cell).astype(np.int64)
    _, owner = np.unique(keys, axis=0, return_inverse=True)
    owner = owner.ravel()
    order = np.argsort(owner, kind='stable')
    bounds = np.flatnonzero(np.diff(owner[order])) + 1
    residual = np.zeros(len(rest))
    for members in np.split(order, bounds):
        if len(members) < 4:
            continue
        a, b = rest[members], posed[members]
        weights = np.ones(len(members))
        # Robust fit: vertices that deform are down-weighted, so the move follows the part that stays rigid.
        for _ in range(4):
            w = weights / weights.sum()
            ca, cb = w @ a, w @ b
            u, _, vt = np.linalg.svd(((a - ca) * w[:, None]).T @ (b - cb))
            d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.
            rotation = vt.T @ np.diag([1., 1., d]) @ u.T
            error = np.linalg.norm((a - ca) @ rotation.T + cb - b, axis=1)
            weights = 1. / (1. + (error / .001) ** 2)
        residual[members] = error
    return residual


def deformation_score(reference, examples, rows, cell=.15):
    """Per-vertex non-rigid movement (metres) between the reference pose and the example poses: the
    largest over the examples, each measured on two offset grids (the smaller wins, so cell borders do
    not show), then averaged once with the neighbours."""
    reference = np.asarray(reference, dtype=float)
    score = np.zeros(len(reference))
    for posed in examples:
        posed = np.asarray(posed, dtype=float)
        first = _rigid_residual(reference, posed, cell)
        second = _rigid_residual(reference + cell * .5, posed, cell)
        score = np.maximum(score, np.minimum(first, second))
    smoothed = score.copy()
    for i, near in enumerate(rows):
        if near:
            smoothed[i] = (score[i] + score[list(near)].sum()) / (len(near) + 1)
    return smoothed


def suggest_patches(score, rows, threshold, minimum=12, hole_fraction=.02):
    """Connected areas moving more than threshold (metres), as labels (-1 = none). Small unmarked islands
    surrounded by a single area are filled, so a breast whose base deforms while its tip moves rigidly is
    suggested whole."""
    active = np.asarray(score) > threshold
    labels, count = connected_labels(rows, active)
    sizes = np.bincount(labels[labels >= 0], minlength=count)
    for label in np.flatnonzero(sizes < minimum):
        labels[labels == label] = -1
    holes, hole_count = connected_labels(rows, labels < 0)
    limit = max(minimum, int(len(score) * hole_fraction))
    hole_sizes = np.bincount(holes[holes >= 0], minlength=hole_count)
    for hole in np.flatnonzero(hole_sizes <= limit):
        members = np.flatnonzero(holes == hole)
        around = {int(labels[j]) for i in members for j in rows[i] if labels[j] >= 0}
        if len(around) == 1:
            labels[members] = around.pop()
    kept = np.unique(labels[labels >= 0])
    remap = np.full(max(int(labels.max()) + 1, 1), -1)
    remap[kept] = np.arange(len(kept))
    return np.where(labels >= 0, remap[np.maximum(labels, 0)], -1), len(kept)


class NamedRegions:
    """Saved selections by name. Regions may overlap; selecting one puts it in the current selection, so
    every action that works on the selection (softness, rigid, ignore collider, motion examples) can use it."""

    def __init__(self, count):
        self.count = count
        self.masks = {}

    def save(self, name, mask):
        name = name.strip()
        if not name:
            raise ValueError('Give the region a name.')
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != self.count or not mask.any():
            raise ValueError('Select some vertices first.')
        self.masks[name] = mask.copy()
        return name

    def get(self, name):
        return self.masks[name].copy()

    def remove(self, name):
        self.masks.pop(name, None)

    def names(self):
        return list(self.masks)
