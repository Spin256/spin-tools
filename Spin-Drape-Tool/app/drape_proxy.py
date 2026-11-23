# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Proxy simulation for heavy garments (Spin256 · https://www.spin256.com/tools).

A lighter copy of the garment is simulated and its motion carried onto the full mesh, so a
300k-vertex dress prepares and runs like a ~25k one. The full mesh, its vertex order and
topology are unchanged (the DAZ return is untouched).

Building the proxy:
- seeds: one vertex per occupied grid cell, chosen separately for each connected piece, so
  pockets, buttons and separate panels keep their own proxy; pieces under SMALL_PIECE vertices
  are kept whole;
- clusters: every vertex joins the nearest seed *along the mesh edges* (multi-source BFS), never
  across empty space, so touching layers (front/back of a skirt) never merge;
- proxy vertices sit at their cluster's centroid; proxy triangles are the full faces whose three
  corners fall in three different clusters (duplicates removed, source winding kept).

Carrying motion back: each full vertex is embedded in a proxy triangle next to its own cluster
(barycentric coordinates of its projection, unclamped, plus an offset along the triangle normal),
which reproduces the captured shape exactly at rest and keeps fine modelled detail.
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
    """Proxy of ``points`` (N, 3) / ``faces`` (F, 3) with about ``size`` vertices."""

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
        self.seeds = seeds[order]
        self.count = len(self.seeds)
        sums = np.zeros((self.count, 3))
        np.add.at(sums, self.owner, points)
        self.rest = (sums / np.bincount(self.owner, minlength=self.count)[:, None]).astype(np.float32)
        self.faces = self._proxy_faces(faces)
        self.ranges = None
        if ranges is not None:
            self.ranges = [(int(np.searchsorted(self.seeds, start)), int(np.searchsorted(self.seeds, end)))
                           for start, end in ranges]
        if progress:
            progress("embedding the full mesh in the proxy", .6)
        self._embed(points)

    @staticmethod
    def _seeds(points, piece, small, cell):
        large = np.flatnonzero(~small)
        keys = np.floor(points[large] / cell).astype(np.int64)
        table = np.column_stack([piece[large], keys])
        _, first = np.unique(table, axis=0, return_index=True)
        return np.concatenate([large[first], np.flatnonzero(small)])

    def _proxy_faces(self, faces):
        labels = self.owner[faces]
        distinct = (labels[:, 0] != labels[:, 1]) & (labels[:, 1] != labels[:, 2]) & (labels[:, 0] != labels[:, 2])
        tris = labels[distinct]
        _, first = np.unique(np.sort(tris, axis=1), axis=0, return_index=True)
        return tris[np.sort(first)].astype(np.int32)

    def _embed(self, points):
        """Pick, for each full vertex, a proxy triangle around its own proxy vertex and store
        unclamped barycentric coordinates plus the normal offset."""
        rest = self.rest.astype(np.float64)
        tris = self.faces.astype(np.int64)
        count = len(points)
        corner_v = tris.ravel()
        corner_t = np.repeat(np.arange(len(tris)), 3)
        order = np.argsort(corner_v, kind="stable")
        corner_v, corner_t = corner_v[order], corner_t[order]
        starts = np.searchsorted(corner_v, np.arange(self.count + 1))
        best_t = np.full(count, -1, dtype=np.int64)
        best_cost = np.full(count, np.inf)
        best_bary = np.zeros((count, 3))
        best_d = np.zeros(count)
        degree = starts[1:] - starts[:-1]
        # Test up to the first MAX candidate triangles of each vertex's proxy vertex, one round each.
        for k in range(int(min(degree.max(initial=0), 12))):
            has = degree[self.owner] > k
            idx = np.flatnonzero(has)
            if not len(idx):
                break
            t = corner_t[starts[self.owner[idx]] + k]
            bary, d = self._project(points[idx], rest[tris[t]])
            outside = np.clip(-bary, 0, None).sum(1) + np.clip(bary - 1, 0, None).sum(1)
            cost = outside * 10 + np.abs(d) / max(self._scale(rest), 1e-12)
            better = cost < best_cost[idx]
            sel = idx[better]
            best_t[sel], best_cost[sel] = t[better], cost[better]
            best_bary[sel], best_d[sel] = bary[better], d[better]
        loose = best_t < 0                       # proxy vertex on no triangle: follow it rigidly
        self.tri, self.bary, self.offset = best_t, best_bary, best_d
        self.loose = np.flatnonzero(loose)
        self.loose_delta = (points[loose] - rest[self.owner[loose]])

    @staticmethod
    def _scale(rest):
        return float(np.linalg.norm(rest.max(0) - rest.min(0)))

    @staticmethod
    def _project(p, tri):
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        e1, e2 = b - a, c - a
        n = np.cross(e1, e2)
        length = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.maximum(length, 1e-20)
        rel = p - a
        d = np.einsum("ij,ij->i", rel, n)
        q = rel - d[:, None] * n
        d00, d01, d11 = (np.einsum("ij,ij->i", e1, e1), np.einsum("ij,ij->i", e1, e2), np.einsum("ij,ij->i", e2, e2))
        d20, d21 = np.einsum("ij,ij->i", q, e1), np.einsum("ij,ij->i", q, e2)
        den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-30, 1e-30, d00 * d11 - d01 * d01)
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        return np.column_stack([1 - v - w, v, w]), d

    # ----- mapping ----------------------------------------------------------------------
    def expand(self, proxy_points):
        """Full-mesh positions for proxy positions, in the units the proxy was built in (meters)."""
        q = np.asarray(proxy_points, dtype=np.float64)
        out = np.empty((self.full_count, 3))
        ok = self.tri >= 0
        tri = self.faces[self.tri[ok]]
        a, b, c = q[tri[:, 0]], q[tri[:, 1]], q[tri[:, 2]]
        n = np.cross(b - a, c - a)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
        bary = self.bary[ok]
        out[ok] = a * bary[:, :1] + b * bary[:, 1:2] + c * bary[:, 2:] + n * self.offset[ok, None]
        if len(self.loose):
            out[self.loose] = q[self.owner[self.loose]] + self.loose_delta
        return out.astype(np.float32)

    def to_proxy(self, indices):
        """Proxy vertex indices for full-mesh vertex indices (duplicates removed)."""
        return np.unique(self.owner[np.asarray(indices, dtype=np.int64)])

    def gather(self, indices, values, reduce="mean"):
        """Proxy indices and per-proxy values (mean or max) for full-mesh indices and values."""
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        owners = self.owner[indices]
        unique, inverse = np.unique(owners, return_inverse=True)
        shape = (len(unique),) + values.shape[1:]
        if reduce == "max":
            out = np.full(shape, -np.inf)
            np.maximum.at(out, inverse, values)
        else:
            out = np.zeros(shape)
            np.add.at(out, inverse, values)
            out /= np.bincount(inverse)[(...,) + (None,) * (values.ndim - 1)]
        return unique, out


def proxy_live(base):
    """A LiveDrape whose grab/freeze calls (full-mesh indices) reach the proxy simulation."""
    class ProxyLiveDrape(base):
        proxy = None

        def set_grab(self, indices=None, targets=None, weights=None, strength=1.):
            if indices is None or self.proxy is None:
                return super().set_grab()
            proxy_indices, proxy_targets = self.proxy.gather(indices, targets)
            _, proxy_weights = self.proxy.gather(indices, weights, reduce="max")
            return super().set_grab(proxy_indices, proxy_targets, proxy_weights, strength)

        def set_frozen(self, indices, positions):
            if self.proxy is None or not len(indices):
                return super().set_frozen(np.empty(0, np.int32), np.empty((0, 3), np.float32))
            proxy_indices, proxy_positions = self.proxy.gather(indices, positions)
            return super().set_frozen(proxy_indices, proxy_positions)
    return ProxyLiveDrape()
