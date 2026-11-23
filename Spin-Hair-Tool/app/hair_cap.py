# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Scalp cap: short extra hair cards on scalp the artist left bare.

Hair assets are made to be seen upright. Some leave part of the scalp without strand roots
(it is covered by hair falling over it), so a figure bending over shows a bald patch. With
the cap enabled, the tool covers the asset's whole scalp surface with short cards, whether
or not real strands are rooted there (roots often fan out from a crown swirl and leave the
skin between them bare once the hair falls away). Each is a copy of the root end of the nearest scalp-rooted card, laid close
along the scalp the same way, so it keeps that card's UVs and surface (and so its material,
root shading included). Cap cards are fixed to the scalp, outside the simulation, and are added after the original
polygons in the returned hair; with the cap off nothing changes.
"""
import numpy as np

from hair_strands import _Grid, nearest_distance


def head_sphere(roots):
    """Center and radius of a sphere fitted to scalp root positions."""
    fit = np.linalg.lstsq(np.c_[2 * roots, np.ones(len(roots))], (roots ** 2).sum(1), rcond=None)[0]
    centre = fit[:3]
    return centre, float(np.sqrt(max(fit[3] + centre @ centre, 1e-9)))


def _tangent(direction, normal):
    t = direction - normal * (direction @ normal)
    length = np.linalg.norm(t)
    if length < 1e-6:
        t = np.cross(normal, [1.0, 0.0, 0.0])
        if np.linalg.norm(t) < 1e-6:
            t = np.cross(normal, [0.0, 0.0, 1.0])
        length = np.linalg.norm(t)
    return t / length


def _frame(direction, normal):
    t = _tangent(direction, normal)
    b = np.cross(t, normal)
    return np.stack([t, normal, b])           # rows


def build_cap(inputs, model, density=1.0, spacing=0.6, cover_length=8.0, bare_gap=0.0, covered=8.0, limit=20000,
              lie=0.3, clearance=0.25, ear_top=-0.05, ear_side=1.0, ear_down=-0.35, ear_angle=24.0, nape=-0.8,
              face_front=0.5, hairline=0.45):
    """Cap cards for bare scalp. Returns None when nothing is bare, else a dict with
    vertices (M, 3), faces (F, 4) indexed from the end of the original mesh, face_surface,
    source (M) original vertex each cap vertex copies its UV from, and source_face (F)."""
    vertices, faces, face_surface = inputs["vertices"], inputs["faces"], inputs["face_surface"]
    spacing = spacing / np.sqrt(max(density, 0.05))
    scalp_strands = np.flatnonzero(model.root_kind == 0)
    if len(scalp_strands) < 8:
        return None
    start, count = model.particle_start, model.particle_count
    roots = model.particles[start[scalp_strands]]
    # Head: fitted to the roots; with few roots (they may lie on one ring, which cannot
    # define a sphere) to the asset's scalp surface instead.
    scalp_points = vertices[model.static_vertices]
    shape = roots if len(roots) >= 50 or len(scalp_points) < 20 else scalp_points
    centre, radius = head_sphere(shape)
    up = shape.mean(0) - centre
    up /= max(np.linalg.norm(up), 1e-9)
    body = np.concatenate([np.asarray(c["vertices"], float) for c in inputs["colliders"]])
    hair_points = vertices[model.strand_of_vertex >= 0]
    # Spots: the avatar's head skin (the asset's scalp surface may not reach everywhere a
    # bending figure shows), limited to the nominal hair area below.
    offset = body - centre
    distance = np.linalg.norm(offset, axis=1)
    spots = body[(distance < radius * 1.6) & (offset @ up > -radius)]
    if not len(spots):
        return None
    # Even spacing: one spot per grid cell.
    _, keep = np.unique(np.floor(spots / spacing).astype(np.int64), axis=0, return_index=True)
    spots = spots[np.sort(keep)][:limit]

    # Root-end section of every scalp-rooted card: vertices within cover_length of the root.
    seg = np.zeros(len(model.particles))
    has_next = np.ones(len(model.particles), bool)
    has_next[start + count - 1] = False
    seg[has_next] = np.linalg.norm(model.particles[1:][has_next[:-1]] - model.particles[:-1][has_next[:-1]], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    owner = np.repeat(np.arange(model.strand_count), count)
    particle_arc = cumulative[:-1] - cumulative[start][owner]
    moving = model.strand_of_vertex >= 0
    vertex_arc = np.full(len(vertices), np.inf)
    j = model.bind_segment[moving]
    vertex_arc[moving] = particle_arc[j] + model.bind_t[moving] * seg[j]
    face_strand = np.where(moving[faces[:, 0]], model.strand_of_vertex[faces[:, 0]], -1)
    face_reach = np.where(faces >= 0, vertex_arc[np.maximum(faces, 0)], -np.inf).max(1)
    section_faces = {}
    usable = np.flatnonzero((face_strand >= 0) & (face_reach <= cover_length))
    for f in usable:
        section_faces.setdefault(int(face_strand[f]), []).append(f)
    donors = np.array([s for s in scalp_strands if len(section_faces.get(int(s), ())) >= 1], int)
    if not len(donors):
        return None
    grid = _Grid(model.particles[start[donors]], 2.0)

    out_vertices, out_faces, out_surface, out_source, out_source_face = [], [], [], [], []
    base = len(vertices)
    donor_roots = model.particles[start[donors]]
    # Cap cards lie close on the skin (they are fixed, so nothing pushes them out).
    lift = np.minimum(nearest_distance(donor_roots, body), lie)
    # Scalp-surface spots sit just above the skin already; measure heights from the skin.
    scalp_lift = 0.0
    # Ears: the sides of the head below about the ear tops (head frame from the roots, so a
    # leaning figure works too; figures face +z, so x is sideways). Non-standard heads are
    # the user's choice; this is the nominal area.
    side = np.array([1.0, 0.0, 0.0]) - up * up[0]
    side /= max(np.linalg.norm(side), 1e-9)

    forward = np.cross(side, up)

    # Ears: a rounded zone around each ear (sideways and a little below the head's middle),
    # so the cap's edge curves behind them instead of turning a corner. Face: the front below
    # the hairline, and the front half below ear-top level.
    ear_dirs = [d / np.linalg.norm(d) for d in (side * ear_side + up * ear_down, -side * ear_side + up * ear_down)]

    def off_area(points):
        o = points - centre
        o = o / np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-9)
        ears = np.zeros(len(o), bool)
        for e in ear_dirs:
            ears |= o @ e > np.cos(np.radians(ear_angle))
        height, front = o @ up, o @ forward
        face = (front > face_front) & (height < hairline)
        jaw = (front > 0.0) & (height < ear_top)
        return ears | face | jaw | (height < nape)

    spots = spots[~off_area(spots)]
    for spot in spots:
        hit, _ = grid.nearest(spot, 10.0)
        if hit < 0:
            hit = int(np.argmin(np.linalg.norm(donor_roots - spot, axis=1)))
        s = int(donors[hit])
        root = model.particles[start[s]]
        first = model.particles[start[s] + 1] - root
        donor_normal = (root - centre) / max(np.linalg.norm(root - centre), 1e-9)
        normal = (spot - centre) / max(np.linalg.norm(spot - centre), 1e-9)
        spot = spot - normal * scalp_lift
        # Same lie on the scalp: the donor's first direction, turned into the new tangent plane.
        old = _frame(first, donor_normal)
        # Combed down the head: real roots fan out from crown swirls, and cards following them
        # converge there and leave gaps; cards combed down overlap evenly.
        new = _frame(-up, normal)
        turn = new.T @ old
        # Same height over the scalp as the donor's root (the head is not a true sphere).
        new_root = spot + normal * lift[hit]
        card_faces = section_faces[s]
        used = np.unique(faces[card_faces][faces[card_faces] >= 0])
        # Lay the card along the scalp: its centre line follows the head (at the donor root's
        # height) in the combed direction; the card's width keeps its shape around it.
        arc = vertex_arc[used]
        j = model.bind_segment[used]
        on_line = model.particles[j] * (1 - model.bind_t[used])[:, None] + model.particles[j + 1] * model.bind_t[used][:, None]
        across = (vertices[used] - on_line) @ turn.T
        height = np.linalg.norm(new_root - centre)
        ahead = new[0]
        path = new_root - centre + ahead[None] * arc[:, None]
        path = centre + path / np.linalg.norm(path, axis=1, keepdims=True) * height
        # Lay the card's width flat on the scalp (a card standing on edge covers nothing).
        out_dir = (path - centre) / height
        size = np.linalg.norm(across, axis=1, keepdims=True)
        flat = across - out_dir * (across * out_dir).sum(1, keepdims=True)
        flat_size = np.linalg.norm(flat, axis=1, keepdims=True)
        sideways = np.cross(out_dir, ahead)
        sideways_sign = np.sign((across * sideways).sum(1, keepdims=True) + 1e-12)
        flat = np.where(flat_size > 1e-6, flat / np.maximum(flat_size, 1e-9), sideways * sideways_sign)
        across = flat * size
        card = path + across
        # Keep to the nominal hair area: drop the faces that would lie on an ear or the neck.
        outside = dict(zip(used.tolist(), (~off_area(card)).tolist()))
        kept = [f for f in card_faces if all(outside[int(v)] for v in faces[f] if v >= 0)]
        if not kept:
            continue
        keep_vertices = np.unique(faces[kept][faces[kept] >= 0])
        keep_index = np.searchsorted(used, keep_vertices)
        card, used = card[keep_index], keep_vertices
        local = {int(v): base + sum(len(x) for x in out_vertices) + k for k, v in enumerate(used)}
        out_vertices.append(card)
        out_source.append(used)
        for f in kept:
            out_faces.append([local[int(v)] if v >= 0 else -1 for v in faces[f]])
            out_surface.append(int(face_surface[f]))
            out_source_face.append(int(f))
    cap_vertices = np.concatenate(out_vertices)
    # Over the avatar's skin and over the asset's scalp surface (both are drawn over anything
    # beneath them).
    head = body[np.linalg.norm(body - centre, axis=1) < radius * 2.0]
    cap_vertices = above(cap_vertices, head, centre, lie, exact=True)
    if model.static_vertices.any():
        cap_vertices = above(cap_vertices, vertices[model.static_vertices], centre, clearance)
    cap_faces = np.asarray(out_faces, np.int64)
    return dict(vertices=cap_vertices, normals=vertex_normals(cap_vertices, cap_faces - base), faces=np.asarray(out_faces, np.int64),
                face_surface=np.asarray(out_surface, int), source=np.concatenate(out_source),
                source_face=np.asarray(out_source_face, int), cards=len(out_vertices))


def above(points, surface, centre, clearance, chunk=2048, exact=False):
    """Push points out from the head center until they sit ``clearance`` over the nearest point
    of ``surface`` (the asset's scalp mesh, which is drawn over anything beneath it). ``exact``
    also pulls points in, so they lie at that height (used for the skin)."""
    out = points.copy()
    surface_radius = np.linalg.norm(surface - centre, axis=1)
    for a in range(0, len(points), chunk):
        p = points[a:a + chunk]
        nearest = np.argmin(((p[:, None, :] - surface[None]) ** 2).sum(2), axis=1)
        offset = p - centre
        radius = np.linalg.norm(offset, axis=1)
        wanted = surface_radius[nearest] + clearance
        if not exact:
            wanted = np.maximum(radius, wanted)
        out[a:a + chunk] = centre + offset * (wanted / np.maximum(radius, 1e-9))[:, None]
    return out


def vertex_normals(points, faces):
    """Area-weighted vertex normals (quads as two triangles)."""
    normals = np.zeros_like(points)
    for tri in (faces[:, [0, 1, 2]], faces[faces[:, 3] >= 0][:, [0, 2, 3]]):
        n = np.cross(points[tri[:, 1]] - points[tri[:, 0]], points[tri[:, 2]] - points[tri[:, 0]])
        for k in range(3):
            np.add.at(normals, tri[:, k], n)
    return (normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def with_cap(inputs, cap):
    """Session inputs with the cap cards appended (original vertex and face order kept)."""
    if cap is None:
        return inputs
    extended = dict(inputs)
    extended["vertices"] = np.concatenate([inputs["vertices"], cap["vertices"]])
    extended["faces"] = np.concatenate([inputs["faces"], cap["faces"]])
    extended["face_surface"] = np.concatenate([inputs["face_surface"], cap["face_surface"]])
    extended["cap"] = cap
    return extended
