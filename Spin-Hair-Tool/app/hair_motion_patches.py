# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Mesh patches for rigid-limb inference (Spin256 · spin256.com/tools).

Surface patches clustered along the mesh, used to find the avatar's rigid parts.
"""
import numpy as np

SMALL_PIECE = 64
DEFAULT_SIZE = 25_000


def _edges(faces):
    f = np.asarray(faces, dtype=np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def _components(count, edges):
    """Connected piece label per vertex (union-find by repeated min-label propagation)."""
    label = np.arange(count, dtype=np.int64)
    a, b = edges[:, 0], edges[:, 1]
    while True:
        low = np.minimum(label[a], label[b])
        changed = False
        for side in (a, b):
            before = label[side]
            np.minimum.at(label, side, low)
            changed |= bool(np.any(label[side] != before))
        label = label[label]            # pointer jumping
        if not changed:
            break
    _, label = np.unique(label, return_inverse=True)
    return label


def _grow(count, edges, seeds, owner=None, steps=None):
    """Multi-source BFS along mesh edges: cluster index per vertex (-1 if not reached within
    ``steps`` rings). New seeds get the next cluster indices."""
    if owner is None:
        owner = np.full(count, -1, dtype=np.int64)
        first = 0
    else:
        owner = owner.copy()
        first = int(owner.max()) + 1
    owner[seeds] = first + np.arange(len(seeds))
    a, b = edges[:, 0], edges[:, 1]
    step = 0
    while steps is None or step < steps:
        grow_b = (owner[a] >= 0) & (owner[b] < 0)
        grow_a = (owner[b] >= 0) & (owner[a] < 0)
        if not grow_a.any() and not grow_b.any():
            break
        new = owner.copy()
        new[b[grow_b]] = owner[a[grow_b]]
        new[a[grow_a]] = owner[b[grow_a]]
        owner = new
        step += 1
    return owner


class ProxyMesh:
    """Mesh-connected patch labels for rigid-part inference."""

    def __init__(self, points, faces, size=DEFAULT_SIZE, ranges=None, progress=None):
        points = np.asarray(points, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        count = len(points)
        self.full_count = count
        self.size = size
        edges = _edges(faces)
        piece = _components(count, edges)
        piece_size = np.bincount(piece)
        small = piece_size[piece] < SMALL_PIECE
        area = 0.5 * np.linalg.norm(np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                                             points[faces[:, 2]] - points[faces[:, 0]]), axis=1).sum()
        large_target = max(size - int(small.sum()), 64)
        cell = max(np.sqrt(max(area, 1e-12) / large_target) * 1.15, 1e-6)
        for _ in range(4):                       # adjust the cell size toward the requested size
            seeds = self._seeds(points, piece, small, cell)
            if abs(len(seeds) - size) < .15 * size or count <= size:
                break
            cell *= np.sqrt(len(seeds) / size)
        if progress:
            progress("growing proxy regions", .3)
        seeds = np.unique(seeds)
        # Clusters may grow only about one cell along the mesh. Stacked layers (a fold, a hem)
        # share grid cells, so seeds alone would let one layer's clusters swallow the other;
        # with capped growth, any area left unclaimed gets its own seeds and clusters stay compact.
        edge_length = float(np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1).mean()) if len(edges) else cell
        reach = max(1, int(np.ceil(cell / max(edge_length, 1e-12))))
        owner = _grow(count, edges, seeds, steps=reach)
        while (owner < 0).any():
            left = np.flatnonzero(owner < 0)
            keys = np.column_stack([piece[left], np.floor(points[left] / cell).astype(np.int64)])
            _, first = np.unique(keys, axis=0, return_index=True)
            extra = left[first]
            owner = _grow(count, edges, extra, owner=owner, steps=reach)
            seeds = np.concatenate([seeds, extra])
        # Order proxy vertices by their seed's index so garment ranges stay contiguous.
        order = np.argsort(seeds, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        self.owner = rank[owner]

    @staticmethod
    def _seeds(points, piece, small, cell):
        large = np.flatnonzero(~small)
        keys = np.floor(points[large] / cell).astype(np.int64)
        table = np.column_stack([piece[large], keys])
        _, first = np.unique(table, axis=0, return_index=True)
        return np.concatenate([large[first], np.flatnonzero(small)])

