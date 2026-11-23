# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Grow hair from painted areas of the avatar (Spin256 · spin256.com/tools).

For short hair that fills gaps: a scalp, a beard, armpits, pubic hair. The artist paints polygons of the avatar into
named areas; each area grows straight strands (90 degrees to the skin) at a density, length, segment count, width and
color. Each strand is a thin quad strip (wide at the root, narrowing to the tip) so the existing hair solver takes it as
an ordinary card, and it returns to DAZ as a plain-colored prop named after its area.

Edges: "polygon" keeps roots on the painted polygons; "smooth" cuts the polygon staircase with a smooth line through
the boundary polygons (each vertex's value is the share of its polygons that are painted; roots go where the value
blended over a polygon is at least one half); a shape (circle, ellipse, rectangle, triangle) fitted to the painted
polygons in their best-fitting plane. Feather thins the density toward the edge. Scatter: even (a minimum spacing, as
follicles grow), random or grid. Everything is seeded, so growing again gives the same strands."""
from dataclasses import dataclass, field

import numpy as np

EDGES = ("polygon", "smooth", "circle", "ellipse", "rectangle", "triangle")
SCATTERS = ("even", "random", "grid")
MIN_LENGTH = .05                   # cm: strands shorter than this (feathered to nothing at the edge) are not grown
TIP_WIDTH = .05                    # the tip's share of the root width (a strip never collapses to one point)


@dataclass
class GrowArea:
    """One named area and its settings (lengths in cm, width in mm, density in strands per cm²)."""
    name: str
    faces: set = field(default_factory=set)          # painted avatar triangles
    edge: str = "smooth"
    flip: bool = False                               # triangle points the other way
    extend: bool = False                             # a shape may reach past the painted polygons
    feather: float = 0.                              # 0..1 share of the way from the edge to the center that blends
    edge_density: float = .2                         # 0..1 density at the very edge, relative
    edge_length: float = .5                          # 0..1 length at the very edge, relative
    min_length: float = .1                           # cm: no strand is shorter
    scatter: str = "even"
    density: float = 60.
    length: float = 1.5
    length_variation: float = .2                     # 0..1, random share of the length
    segments: int = 4
    width: float = .15
    color: tuple = (60, 42, 30)
    color2: tuple = (165, 160, 155)                  # secondary color (gray hints, dark patches)
    mix: float = 0.                                  # 0..1 share of strands in the secondary color (0: one material)
    material: str = "Hair"                           # named materials: shared by areas;
    material2: str = "Gray"                          # color/color2 are filled from them when growing
    stiffness: float = 0.                            # 0..1 share of the simulation's stiffness (0: falls and lies)
    root_lift: float = .5                            # 0..1 how firmly strands keep leaving the skin along its normal
                                                     # (a center part on the crown)
    seed: int = 1

    def to_json(self):
        return dict(name=self.name, faces=sorted(int(f) for f in self.faces), edge=self.edge, flip=self.flip,
                    extend=self.extend,
                    feather=self.feather, edge_density=self.edge_density, edge_length=self.edge_length,
                    min_length=self.min_length,
                    scatter=self.scatter, density=self.density, length=self.length,
                    length_variation=self.length_variation, segments=self.segments, width=self.width,
                    stiffness=self.stiffness, root_lift=self.root_lift,
                    color=list(self.color), color2=list(self.color2), mix=self.mix, material=self.material,
                    material2=self.material2, seed=self.seed)

    @classmethod
    def from_json(cls, data):
        area = cls(str(data["name"]))
        for key, value in data.items():
            if key == "faces":
                area.faces = {int(f) for f in value}
            elif key in ("color", "color2"):
                setattr(area, key, tuple(int(c) for c in value))
            elif hasattr(area, key) and key != "name":
                setattr(area, key, type(getattr(area, key))(value))
        return area


def face_normals(vertices, triangles):
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1) * .5
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12), area


def vertex_normals(vertices, triangles):
    n = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]], vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    out = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(out, triangles[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def vertex_values(triangles, painted, vertex_count):
    """Share of each vertex's triangles that are painted (1 inside, 0 outside, between on the edge)."""
    painted_mask = np.zeros(len(triangles), bool)
    painted_mask[list(painted)] = True
    total = np.zeros(vertex_count)
    hit = np.zeros(vertex_count)
    for k in range(3):
        np.add.at(total, triangles[:, k], 1.)
        np.add.at(hit, triangles[:, k], painted_mask.astype(float))
    return hit / np.maximum(total, 1.)


def candidate_faces(triangles, painted, edge):
    """Triangles that may hold roots: the painted ones, plus their neighbors for smooth edges and shapes."""
    painted = np.array(sorted(painted), dtype=np.int64)
    if edge == "polygon" or not len(painted):
        return painted
    touched = np.zeros(int(triangles.max()) + 1, bool)
    touched[triangles[painted].ravel()] = True
    return np.flatnonzero(touched[triangles].any(axis=1))


def sample_points(vertices, triangles, faces, count, scatter, rng):
    """``count``-ish points on the faces, area-weighted; returns (points, face of each, barycentrics)."""
    a, b, c = (vertices[triangles[faces, k]] for k in range(3))
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) * .5
    total = float(area.sum())
    if total <= 0 or count <= 0:
        return np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros((0, 3))
    if scatter == "grid":
        # Regular: each triangle gets its share of the count, placed on an even (R2) sequence, with no randomness.
        share = area / total * count
        per = np.diff(np.round(np.concatenate([[0.], np.cumsum(share)]))).astype(int)
        g = 1.32471795724474602596
        pts, owner, bary = [], [], []
        for i, f in enumerate(faces):
            for j in range(per[i]):
                x, y = (.5 + (j + 1) / g) % 1., (.5 + (j + 1) / (g * g)) % 1.
                if x + y > 1:
                    x, y = 1 - x, 1 - y
                w = np.array([1 - x - y, x, y])
                pts.append(w @ np.stack([a[i], b[i], c[i]]))
                owner.append(f)
                bary.append(w)
        return np.array(pts).reshape(-1, 3), np.array(owner, np.int64), np.array(bary).reshape(-1, 3)
    draws = count * (6 if scatter == "even" else 1)
    pick = rng.choice(len(faces), size=draws, p=area / total)
    r1, r2 = np.sqrt(rng.random(draws)), rng.random(draws)
    bary = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], axis=1)
    points = bary[:, :1] * a[pick] + bary[:, 1:2] * b[pick] + bary[:, 2:] * c[pick]
    if scatter == "even":
        keep = _spaced(points, np.sqrt(total / count) * .68)[:count]
        points, pick, bary = points[keep], pick[keep], bary[keep]
    return points, faces[pick], bary


def _spaced(points, spacing):
    """Indices of points kept by dart throwing: none closer than ``spacing``."""
    cells = {}
    keep = []
    for i, p in enumerate(points):
        key = tuple(np.floor(p / spacing).astype(int))
        near = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in cells.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        if np.sum((points[j] - p) ** 2) < spacing * spacing:
                            near = True
                            break
                    if near:
                        break
                if near:
                    break
            if near:
                break
        if not near:
            cells.setdefault(key, []).append(i)
            keep.append(i)
    return np.array(keep, np.int64)


def plane(points):
    """Center and two in-plane axes (largest spread first) of the points' best-fitting plane."""
    centre = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - centre, full_matrices=False)
    return centre, axes[0], axes[1]


def shape_inside(edge, painted_points, points, flip=False, area=None):
    """Depth (>0 inside, about 1 at the center) of ``points`` in the shape fitted to the painted points, in the
    painted area's plane: centered on them, turned and stretched like them, and covering ``area`` (the painted area)."""
    centre, u_axis, v_axis = plane(painted_points)
    pu, pv = (painted_points - centre) @ u_axis, (painted_points - centre) @ v_axis
    qu, qv = (points - centre) @ u_axis, (points - centre) @ v_axis
    su, sv = max(np.sqrt(np.mean(pu ** 2)), 1e-9), max(np.sqrt(np.mean(pv ** 2)), 1e-9)
    if area is None:
        area = 12. * su * sv                                # a uniform rectangle with this spread
    if edge == "circle":
        radius = np.sqrt(area / np.pi)
        return 1. - np.hypot(qu, qv) / radius
    if edge == "ellipse":
        k = np.sqrt(area / (np.pi * su * sv))
        return 1. - np.hypot(qu / (k * su), qv / (k * sv))
    if edge == "rectangle":
        k = np.sqrt(area / (4. * su * sv))
        return np.minimum(1. - np.abs(qu) / (k * su), 1. - np.abs(qv) / (k * sv))
    if edge == "triangle":
        # Pointing along the area's longest direction (flip: the other way). Height h along u, base 2w across.
        k = np.sqrt(area / (su * sv * 2.))
        h, w = k * su * 2., k * sv
        x = qu * (-1. if flip else 1.) + h / 3.               # 0 at the base, h at the point
        width = w * (1. - x / h)
        return np.minimum(np.minimum(x / (h / 3.), (h - x) / (h / 3.)),
                          (width - np.abs(qv)) / np.maximum(width, 1e-9)) * np.where(x >= 0, 1., 1.)
    raise ValueError(f"Unknown edge shape: {edge}")


def edge_points(vertices, triangles, painted, per_edge=4):
    """Points along the painted area's outline (edges of exactly one painted triangle)."""
    tris = triangles[np.asarray(painted, np.int64)]
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    outline = unique[counts == 1]
    if not len(outline):
        return np.zeros((0, 3))
    t = np.linspace(0., 1., per_edge, endpoint=False)
    a, b = vertices[outline[:, 0]], vertices[outline[:, 1]]
    return (a[:, None, :] * (1. - t)[None, :, None] + b[:, None, :] * t[None, :, None]).reshape(-1, 3)


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x * x * (3. - 2. * x)


def grow_roots(vertices, triangles, area):
    """Root positions, root normals, their triangles and each root's feather blend (0 at the edge, 1 past the
    feather band) for one area."""
    if not area.faces:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0)
    vertices = np.asarray(vertices, float)
    triangles = np.asarray(triangles, np.int64)
    rng = np.random.default_rng(area.seed)
    painted = np.array(sorted(area.faces), np.int64)
    _, areas = face_normals(vertices, triangles[painted])
    count = int(round(float(areas.sum()) * area.density))
    faces = candidate_faces(triangles, area.faces, area.edge)
    if area.edge not in ("polygon", "smooth") and not area.extend:
        faces = painted                              # the shape trims the painted polygons
    elif area.edge not in ("polygon", "smooth"):
        # A shape may reach past the painted polygons: any nearby skin facing the same way that lies inside it.
        centres = vertices[triangles[painted]].mean(axis=1)
        centre, u_axis, v_axis = plane(centres)
        facing = np.cross(u_axis, v_axis)
        normals_all, _ = face_normals(vertices, triangles)
        if (normals_all[painted] @ facing).mean() < 0:
            facing = -facing
        all_centres = vertices[triangles].mean(axis=1)
        reach = 3. * float(np.sqrt(((centres - centre) ** 2).sum(axis=1).max()))
        near = (np.linalg.norm(all_centres - centre, axis=1) <= reach) & (normals_all @ facing > .3)
        inside = shape_inside(area.edge, centres, all_centres, area.flip, area=float(areas.sum())) > -.25
        faces = np.union1d(faces, np.flatnonzero(near & inside))
    extra =float(face_normals(vertices, triangles[faces])[1].sum()) / max(float(areas.sum()), 1e-12)
    points, owner, bary = sample_points(vertices, triangles, faces, int(round(count * extra)), area.scatter, rng)
    if not len(points):
        return points, points, owner, np.zeros(0)
    if area.edge == "polygon":
        depth = np.ones(len(points))
    elif area.edge == "smooth":
        values = vertex_values(triangles, area.faces, len(vertices))
        value = (bary * values[triangles[owner]]).sum(axis=1)
        depth = (value - .5) * 2.                          # 0 on the smooth edge, 1 well inside
    else:
        centres = vertices[triangles[painted]].mean(axis=1)
        depth = shape_inside(area.edge, centres, points, area.flip, area=float(areas.sum()))
    keep = depth >= 0.
    # Feather: over this share of the way from the edge to the center the hair blends from
    # the edge density and length to the area's own. Distance is measured from the outline: the painted polygons' edge
    # (polygon and smooth edges) or the shape's.
    blend = np.ones(len(points))
    if area.feather > 0 and keep.any():
        if area.edge in ("polygon", "smooth"):
            outline = edge_points(vertices, triangles, painted)
            if len(outline):
                from hair_strands import nearest_distance
                distance = nearest_distance(points, outline)
                inner = float(distance[keep].max()) if keep.any() else 0.
                relative = distance / inner if inner > 0 else np.ones(len(points))
            else:
                relative = np.ones(len(points))
        else:
            relative = np.clip(depth, 0., 1.)                  # a shape's depth is already edge 0 .. center 1
        blend = smoothstep(relative / max(area.feather, 1e-6))
        keep &= rng.random(len(points)) < area.edge_density + (1. - area.edge_density) * blend
    normals = vertex_normals(vertices, triangles)
    root_normals = (bary[:, :, None] * normals[triangles[owner]]).sum(axis=1)
    root_normals /= np.maximum(np.linalg.norm(root_normals, axis=1, keepdims=True), 1e-12)
    return points[keep], root_normals[keep], owner[keep], blend[keep]


# Segment length warning: a strip bends only at its rungs (30 cm with 4 segments bends every 7.5 cm), while the
# simulated strand is followed every 2.5 cm. Segments is used as set; the panel warns and suggests a number.
SEGMENT_SPACING = 2.5


def suggested_segments(area):
    """Segments for no segment longer than SEGMENT_SPACING, or None when the area's own are short enough."""
    needed = int(np.ceil(float(area.length) / SEGMENT_SPACING - 1e-9))
    return needed if needed > int(area.segments) else None


def strand_strips(roots, normals, area, blend=None):
    """Quad strips (cm), one per root: vertices, faces (F, 4), per-vertex UVs (u across, v root 0 to tip 1)."""
    roots, normals = np.asarray(roots, float), np.asarray(normals, float)
    n, segments = len(roots), max(1, int(area.segments))
    if not n:
        return np.zeros((0, 3)), np.zeros((0, 4), np.int64), np.zeros((0, 2))
    rng = np.random.default_rng(area.seed + 7919)
    lengths = area.length * (1. - area.length_variation * rng.random(n))
    if blend is not None:                               # shorter toward the edge (feather)
        lengths = lengths * (area.edge_length + (1. - area.edge_length) * np.asarray(blend, float))
    # Never shorter than the area's minimum (and never a zero-length strip).
    lengths = np.maximum(lengths, max(float(getattr(area, "min_length", MIN_LENGTH)), MIN_LENGTH))
    # Strip width lies along the skin, at a random turn around the normal for each strand.
    helper = np.where(np.abs(normals[:, :1]) < .9, [[1., 0., 0.]], [[0., 1., 0.]])
    t = np.cross(normals, helper)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(normals, t)
    angle = rng.random(n) * 2 * np.pi
    across = t * np.cos(angle)[:, None] + b * np.sin(angle)[:, None]
    k = np.arange(segments + 1) / segments                                     # 0 root .. 1 tip
    centre = roots[:, None, :] + normals[:, None, :] * (lengths[:, None] * k[None, :])[:, :, None]
    half = (area.width * .1 * .5) * (1. - (1. - TIP_WIDTH) * k)                  # mm to cm, narrowing to the tip
    offset = across[:, None, :] * half[None, :, None]
    vertices = np.stack([centre - offset, centre + offset], axis=2).reshape(-1, 3)   # strand, rung, side
    base = (np.arange(n)[:, None] * (segments + 1) + np.arange(segments)[None, :]) * 2
    faces = np.stack([base, base + 1, base + 3, base + 2], axis=-1).reshape(-1, 4)
    uvs = np.stack([np.tile([0., 1.], n * (segments + 1)), np.repeat(np.tile(k, n), 2)], axis=1)
    return vertices, faces, uvs


def grow_area(avatar_vertices, avatar_triangles, area):
    """Everything an area grows: dict(vertices, faces, uvs, roots, strands)."""
    roots, normals, owner, blend = grow_roots(avatar_vertices, avatar_triangles, area)
    vertices, faces, uvs = strand_strips(roots, normals, area, blend if area.feather > 0 else None)
    # Each strand takes the secondary color at random with the area's mix (seeded: the same strands every time).
    secondary = np.random.default_rng(area.seed + 104729).random(len(roots)) < area.mix
    face_secondary = np.repeat(secondary, max(1, int(area.segments)))
    return dict(vertices=vertices, faces=faces, uvs=uvs, roots=roots, root_faces=owner, strands=len(roots),
                secondary=secondary, face_secondary=face_secondary)
