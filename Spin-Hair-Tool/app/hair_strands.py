# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Turn hair cards into simulation strands, find roots and anchors, and bind every card vertex.

A DAZ hair card is a ribbon of quads, w quads wide and h long. Each card becomes
one strand: a chain of particles along its center line. Every card vertex is
stored relative to that center line (arc position + offset in a local frame),
so the full mesh can be rebuilt from the simulated chain with its topology, UVs
and vertex order unchanged.

Anchoring (decided once, before a run):
- scalp-rooted: the root end lies on the scalp (a "scalp"/"cap" surface, or the
  avatar's head when the hair has none) and is fixed there;
- parent-rooted: otherwise the root attaches to the closest point on another
  strand, preferring a scalp-rooted strand within reach (layered and tied hair);
- frozen: whole cards the user freezes do not simulate; strand ends near a frozen
  card attach to it (a tie or bun holding both ends of swept-up hair);
- unanchored: anything left is reported so the user can freeze or accept it.
"""
from collections import deque
import heapq
from dataclasses import dataclass, field
import numpy as np

# Surfaces held still by name (the scalp mesh under the hair). Not "base": hair assets use it
# for real hair layers (cards in some assets are named "... Base"). A
# base/scalp shell that is not made of cards is held still anyway (not card-like).
STATIC_SURFACE_WORDS = ("scalp", "cap", "skull")
SHELL_WIDTH = 3.0     # cm: a "card" wider than this lying on the skin all over is a scalp mesh, not hair
SHELL_GAP = 1.5       # cm: ... all of it within this of the avatar's skin


def connected_components(face_vertices, vertex_count):
    """Component label per vertex for faces given as an (F, k) array with -1 padding."""
    labels = np.arange(vertex_count)
    a = face_vertices[:, 0]
    pairs = []
    for column in range(1, face_vertices.shape[1]):
        b = face_vertices[:, column]
        valid = b >= 0
        pairs.append((a[valid], b[valid]))
    first = np.concatenate([p[0] for p in pairs])
    second = np.concatenate([p[1] for p in pairs])
    while True:
        low = np.minimum(labels[first], labels[second])
        before = labels.copy()
        np.minimum.at(labels, first, low)
        np.minimum.at(labels, second, low)
        labels = labels[labels]          # pointer jumping
        if np.array_equal(labels, before):
            break
    _, compact = np.unique(labels, return_inverse=True)
    return compact


@dataclass
class Card:
    vertices: np.ndarray        # global vertex indices of the card
    rung: np.ndarray            # rung index per card vertex (0 at the first end)
    rungs: int                  # number of rungs (h + 1)
    surface: int


@dataclass
class HairModel:
    rest: np.ndarray                      # (N, 3) all hair vertices, cm
    strand_of_vertex: np.ndarray          # strand id, -1 for static/unsupported vertices
    particle_start: np.ndarray            # first particle per strand
    particle_count: np.ndarray
    particles: np.ndarray                 # (P, 3) rest particle positions, root first
    strand_surface: np.ndarray
    root_kind: np.ndarray                 # 0 scalp, 1 parent, 2 none
    parent_particle: np.ndarray           # for parent-rooted strands, else -1
    root_distance: np.ndarray             # root end distance to the scalp, cm
    # vertex binding
    bind_segment: np.ndarray              # particle index of the segment start
    bind_t: np.ndarray
    bind_offset: np.ndarray               # (N, 3) offset in the rest frame (T, N, B)
    rest_frames: np.ndarray               # (P, 3, 3) rows T, N, B per particle
    static_vertices: np.ndarray           # bool, never move (scalp, unsupported pieces)
    surface_names: list = field(default_factory=list)
    unsupported_cards: int = 0
    card_vertices: list = field(default_factory=list)   # per strand: global vertex ids

    @property
    def strand_count(self):
        return len(self.particle_start)


def _adjacency(card_faces, local):
    neighbours = [set() for _ in range(len(local))]
    for face in card_faces:
        ids = [local[v] for v in face if v >= 0]
        for i, a in enumerate(ids):
            b = ids[(i + 1) % len(ids)]
            neighbours[a].add(b)
            neighbours[b].add(a)
    return [list(n) for n in neighbours]


def _bfs(neighbours, sources):
    distance = np.full(len(neighbours), -1)
    queue = deque(sources)
    for s in sources:
        distance[s] = 0
    while queue:
        v = queue.popleft()
        for n in neighbours[v]:
            if distance[n] < 0:
                distance[n] = distance[v] + 1
                queue.append(n)
    return distance


def card_rungs(card_faces, vertices):
    """Rung index per card vertex for a w x h quad grid card, or None if not a clean grid."""
    local = {v: i for i, v in enumerate(vertices)}
    quads = len(card_faces)
    s = len(vertices) - quads - 1                       # w + h for a grid
    disc = s * s - 4 * quads
    if disc < 0 or (card_faces < 0).any():
        return None
    w = (s - np.sqrt(disc)) / 2
    if abs(w - round(w)) > 1e-6 or round(w) < 1:
        return None
    w = int(round(w))
    h = quads // w
    neighbours = _adjacency(card_faces, local)
    face_count = np.zeros(len(vertices), int)
    for face in card_faces:
        for v in face:
            face_count[local[v]] += 1
    corners = np.flatnonzero(face_count == 1)
    if len(corners) != 4:
        return None
    from_first = _bfs(neighbours, [corners[0]])
    partner = min(corners[1:], key=lambda c: from_first[c])
    if from_first[partner] != w:
        return None
    from_partner = _bfs(neighbours, [partner])
    end_rung = np.flatnonzero(from_first + from_partner == w)
    if len(end_rung) != w + 1:
        return None
    rung = _bfs(neighbours, list(end_rung))
    if rung.min() < 0 or rung.max() != h or np.any(np.bincount(rung) != w + 1):
        return None
    return rung, h + 1


def ribbon_pieces(faces):
    """Separate ribbons touching at a vertex, while keeping edge-joined faces together."""
    edges, owners = [], []
    for index, face in enumerate(faces):
        valid = face[face >= 0]
        edges.extend(tuple(sorted((a, b))) for a, b in zip(valid, np.roll(valid, -1)))
        owners.extend([index] * len(valid))
    _, labels = np.unique(edges, axis=0, return_inverse=True)
    linked = [set() for _ in range(int(labels.max()) + 1)]
    for edge, owner in zip(labels, owners):
        linked[edge].add(owner)
    adjacency = [[] for _ in faces]
    for group in linked:
        indices = list(group)
        if len(indices) > 1:
            for index in indices[1:]:
                adjacency[indices[0]].append(index)
                adjacency[index].append(indices[0])
    seen = np.zeros(len(faces), dtype=bool)
    for seed in range(len(faces)):
        if seen[seed]:
            continue
        group = [seed]
        seen[seed] = True
        for index in group:
            for other in adjacency[index]:
                if not seen[other]:
                    seen[other] = True
                    group.append(other)
        yield faces[group]


def tapered_rungs(faces, indices, vertices, scalp_points=None):
    """Recognize an elongated open ribbon with tapered or irregular quad topology."""
    if len(faces) < 3 or np.mean(faces[:, 3] >= 0) < .8:
        return None
    local = {vertex: i for i, vertex in enumerate(indices)}
    neighbours = _adjacency(faces, local)
    points = vertices[indices]
    edges = []
    for face in faces:
        valid = face[face >= 0]
        edges.extend(tuple(sorted((a, b))) for a, b in zip(valid, np.roll(valid, -1)))
    _, incidence = np.unique(edges, axis=0, return_counts=True)
    if not (incidence == 1).any() or (incidence > 2).any():
        return None

    def distances(seed):
        distance = np.full(len(indices), np.inf)
        seeds = np.atleast_1d(seed).astype(int)
        distance[seeds] = 0.
        pending = [(0., int(s)) for s in seeds]
        heapq.heapify(pending)
        while pending:
            length, index = heapq.heappop(pending)
            if length != distance[index]:
                continue
            for other in neighbours[index]:
                new = length + float(np.linalg.norm(points[index] - points[other]))
                if new < distance[other]:
                    distance[other] = new
                    heapq.heappush(pending, (new, other))
        return distance

    first = int(np.argmax(distances(0)))
    along = distances(first)
    last = int(np.argmax(along))
    length = along[last]
    back = distances(last)
    width = np.percentile(np.maximum(0., along + back - length), 90)
    if not np.isfinite(length) or length < .5:
        return None
    # Narrow ribbons can branch at a few welded tips. Arc coordinates follow
    # their topology instead of projecting curls onto a straight world axis.
    coordinate = np.clip((along - back + length) * .5, 0., length)
    if length < width * 5.:
        # Welded return strips can make a long detour through the topology.
        # Use their major axis only when the piece is geometrically narrow.
        centered = points - points.mean(axis=0)
        _, spread, axes = np.linalg.svd(centered, full_matrices=False)
        if spread[0] < 5. * spread[1]:
            return None
        coordinate = centered @ axes[0]
        coordinate -= coordinate.min()
        length = coordinate.max()
    if scalp_points is not None and len(scalp_points):
        scalp_distance = nearest_distance(points, scalp_points)
        root = int(np.argmin(scalp_distance))
        # A connected return ribbon can run tip -> scalp -> tip. Its graph
        # diameter therefore has two distal ends; reversing it cannot root it.
        # Measure outward from the scalp-facing interior instead.
        ends = coordinate <= .1 * length
        ends |= coordinate >= .9 * length
        if (.1 * length < coordinate[root] < .9 * length
                and scalp_distance[ends].min() > scalp_distance[root] + 2.):
            near_root = scalp_distance <= scalp_distance[root] + .25
            from_root = distances(root)
            seeds = np.flatnonzero(near_root & (from_root < 2.))
            coordinate = distances(seeds)
            length = coordinate.max()
    bins = max(2, min(128, int(np.ceil(length / .5))))
    _, rung = np.unique(np.rint(coordinate / length * bins).astype(int), return_inverse=True)
    if rung.max() < 2:
        return None
    return rung, int(rung.max()) + 1


class _Grid:
    """Uniform hash grid for nearest-point queries (cm)."""
    def __init__(self, points, cell=1.0):
        self.points, self.cell = points, cell
        keys = np.floor(points / cell).astype(np.int64)
        order = np.lexsort(keys.T)
        self.order = order
        sorted_keys = keys[order]
        change = np.any(np.diff(sorted_keys, axis=0) != 0, axis=1)
        starts = np.concatenate([[0], np.flatnonzero(change) + 1])
        ends = np.concatenate([starts[1:], [len(order)]])
        self.cells = {tuple(sorted_keys[s]): (s, e) for s, e in zip(starts, ends)}

    def nearest(self, point, radius, exclude=None, priority=None):
        """Point within reach, ranked by optional priority then distance; (-1, inf) if absent."""
        reach = int(np.ceil(radius / self.cell))
        base = np.floor(point / self.cell).astype(np.int64)
        best, best_d = -1, np.inf
        best_rank = (np.inf, np.inf)
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for dz in range(-reach, reach + 1):
                    span = self.cells.get((base[0] + dx, base[1] + dy, base[2] + dz))
                    if span is None:
                        continue
                    ids = self.order[span[0]:span[1]]
                    if exclude is not None:
                        ids = ids[~exclude(ids)]
                    if not len(ids):
                        continue
                    d = np.linalg.norm(self.points[ids] - point, axis=1)
                    valid = d <= radius
                    ids, d = ids[valid], d[valid]
                    if not len(ids):
                        continue
                    k = int(np.argmin(d) if priority is None else np.lexsort((d, priority[ids]))[0])
                    rank = (0. if priority is None else priority[ids[k]], d[k])
                    if rank < best_rank:
                        best, best_d = int(ids[k]), float(d[k])
                        best_rank = rank
        return (best, best_d) if best_d <= radius else (-1, np.inf)


def nearest_distance(points, cloud, chunk=512):
    """Distance from each point to the closest cloud point (batched brute force)."""
    if not len(cloud):
        return np.full(len(points), np.inf)
    cloud = np.asarray(cloud, np.float32)
    out = np.empty(len(points))
    for i in range(0, len(points), chunk):
        block = np.asarray(points[i:i + chunk], np.float32)
        d2 = (np.square(block).sum(1)[:, None] - 2 * block @ cloud.T + np.square(cloud).sum(1)[None])
        out[i:i + chunk] = np.sqrt(np.maximum(d2.min(1), 0))
    return out


def _frames(points):
    """Parallel-transport frames (rows T, N, B) along a polyline."""
    n = len(points)
    tangents = np.diff(points, axis=0)
    tangents = np.vstack([tangents, tangents[-1:]])
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    frames = np.zeros((n, 3, 3))
    t = tangents[0]
    helper = np.array([0., 1., 0.]) if abs(t[1]) < .9 else np.array([1., 0., 0.])
    normal = np.cross(t, helper); normal /= np.linalg.norm(normal)
    for i in range(n):
        t_new = tangents[i]
        if i:
            normal = normal - np.dot(normal, t_new) * t_new
            length = np.linalg.norm(normal)
            if length < 1e-9:
                helper = np.array([0., 1., 0.]) if abs(t_new[1]) < .9 else np.array([1., 0., 0.])
                normal = np.cross(t_new, helper); length = np.linalg.norm(normal)
            normal /= length
        frames[i] = (t_new, normal, np.cross(t_new, normal))
    return frames


GROWN_MAX_PARTICLES = 128       # grown strands: up to 3 m at 2.5 cm (captured hair keeps max_particles)


def build_hair(vertices, faces, face_surface, surface_names, scalp_points=None,
               segment_length=2.5, max_particles=32, root_radius=2.0, attach_radius=1.5,
               progress=lambda text: None, vertex_uvs=None, extra_roots=None, extra_from=None, skin_points=None,
               objects=None):
    """Analyze captured hair (cm). ``faces`` is (F, 4) with -1 for triangles. Grown hair: cards made
    of vertices from ``extra_from`` on are rooted on ``extra_roots`` (their roots on the avatar's skin) only; the
    captured hair before them is rooted on its scalp exactly as without grown hair.
    Hair without a scalp (a beard, or grown hair captured again): ``objects`` are the captured
    hair objects' vertex ranges; an object none of whose strands reaches the scalp is rooted on ``skin_points`` (the
    avatar) instead, at the end close enough to the skin. Objects with scalp-rooted hair are unchanged."""
    vertices = np.asarray(vertices, float).reshape(-1, 3)
    faces = np.asarray(faces, int).reshape(-1, 4)
    if not len(faces):
        # No hair yet (an avatar captured on its own, to grow hair on it).
        return empty_model(vertices, surface_names)
    progress("Finding hair cards…")
    component = connected_components(faces, len(vertices))
    face_component = component[faces[:, 0]]
    order = np.argsort(face_component, kind="stable")
    bounds = np.searchsorted(face_component[order], np.arange(component.max() + 2))
    vorder = np.argsort(component, kind="stable")
    vbounds = np.searchsorted(component[vorder], np.arange(component.max() + 2))
    static_surface = np.array([any(w in name.lower() for w in STATIC_SURFACE_WORDS) for name in surface_names])
    root_cloud = scalp_points
    if root_cloud is None or not len(root_cloud):
        scalp_indices = np.unique(faces[static_surface[np.asarray(face_surface)]])
        root_cloud = vertices[scalp_indices[scalp_indices >= 0]]
    static = np.zeros(len(vertices), bool)
    cards, unsupported = [], 0
    for c in range(component.max() + 1):
        card_faces = faces[order[bounds[c]:bounds[c + 1]]]
        card_vertices = vorder[vbounds[c]:vbounds[c + 1]]
        if not len(card_faces):
            static[card_vertices] = True
            continue
        surfaces = face_surface[order[bounds[c]:bounds[c + 1]]]
        surface = int(np.bincount(surfaces).argmax())
        if static_surface[surface]:
            static[card_vertices] = True
            continue
        found = card_rungs(card_faces, card_vertices)
        if found is None:
            for piece in ribbon_pieces(card_faces):
                indices = np.unique(piece)
                indices = indices[indices >= 0]
                rungs = card_rungs(piece, indices)
                if rungs is None:
                    rungs = tapered_rungs(piece, indices, vertices, root_cloud)
                if rungs is None:
                    static[indices] = True
                    unsupported += 1
                else:
                    cards.append(Card(indices, rungs[0], rungs[1], surface))
            continue
        cards.append(Card(card_vertices, found[0], found[1], surface))
    if skin_points is not None and len(skin_points) and cards:
        # A scalp mesh not named as one: a "card" lying on the skin all over and far wider than a
        # hair card is the scalp, held still.
        skin = np.asarray(skin_points, float)
        kept = []
        for card in cards:
            points = vertices[card.vertices]
            mids = np.zeros((card.rungs, 3))
            np.add.at(mids, card.rung, points)
            mids /= np.bincount(card.rung, minlength=card.rungs)[:, None]
            width = float(np.linalg.norm(points - mids[card.rung], axis=1).max()) * 2.
            if width > SHELL_WIDTH and nearest_distance(points, skin).max() <= SHELL_GAP:
                static[card.vertices] = True
            else:
                kept.append(card)
        cards = kept
    for card in cards:
        static[card.vertices] = False
    if scalp_points is None or not len(scalp_points):
        # Unsupported ribbons may lie anywhere along the hair. They are held
        # still, but must not masquerade as scalp and reverse nearby strands.
        scalp_indices = np.unique(faces[static_surface[np.asarray(face_surface)]])
        scalp_indices = scalp_indices[scalp_indices >= 0]
        scalp_points = vertices[scalp_indices] if len(scalp_indices) else vertices[static]
    scalp_points = np.asarray(scalp_points, float)

    progress("Building strands…")
    rung_mids, ends = [], []
    for card in cards:
        mids = np.zeros((card.rungs, 3))
        np.add.at(mids, card.rung, vertices[card.vertices])
        mids /= np.bincount(card.rung, minlength=card.rungs)[:, None]
        rung_mids.append(mids)

    # Orient each strand root-first: the end nearer the scalp is the root.
    first_ends = np.array([mids[0] for mids in rung_mids]).reshape(-1, 3)
    last_ends = np.array([mids[-1] for mids in rung_mids]).reshape(-1, 3)
    grown = np.zeros(len(cards), bool)
    if extra_roots is not None and len(extra_roots) and extra_from is not None:
        grown = np.array([card.vertices.min() >= extra_from for card in cards], bool)
    captured = ~grown
    first_d, last_d = np.full(len(cards), np.inf), np.full(len(cards), np.inf)
    if captured.any() and len(scalp_points):
        first_d[captured] = nearest_distance(first_ends[captured], scalp_points)
        last_d[captured] = nearest_distance(last_ends[captured], scalp_points)
    on_skin = np.zeros(len(cards), bool)
    if skin_points is not None and len(skin_points) and objects:
        skin = np.asarray(skin_points, float)
        lowest = np.array([card.vertices.min() for card in cards]) if cards else np.zeros(0, int)
        for begin, end in objects:
            members = captured & (lowest >= begin) & (lowest < end)
            if not members.any() or (np.minimum(first_d[members], last_d[members]) <= root_radius).any():
                continue                                     # empty, or rooted on the scalp
            first_d[members] = nearest_distance(first_ends[members], skin)
            last_d[members] = nearest_distance(last_ends[members], skin)
            on_skin |= members
    if grown.any():
        extra = np.asarray(extra_roots, float)
        first_d[grown] = nearest_distance(first_ends[grown], extra)
        last_d[grown] = nearest_distance(last_ends[grown], extra)
    root_distance = np.zeros(len(cards))
    # Both ends on the scalp (a fringe lying on the forehead, hair swept over the head): the
    # nearer end can be the tip. Pick the end nearer the roots of strands whose root is clear.
    flip_votes = np.zeros(len(cards))
    ambiguous = np.maximum(first_d, last_d) <= root_radius * 1.5
    if ambiguous.any() and (~ambiguous).any():
        clear = np.flatnonzero(~ambiguous)
        clear_roots = np.where((first_d[clear] <= last_d[clear])[:, None], first_ends[clear], last_ends[clear])
        a = np.flatnonzero(ambiguous)
        flip_votes[a] = nearest_distance(first_ends[a], clear_roots) - nearest_distance(last_ends[a], clear_roots)
    uv_vote = np.zeros(len(cards))
    if vertex_uvs is not None and np.shape(vertex_uvs) == (len(vertices), 2):
        # Learn texture direction separately per surface from unambiguous
        # scalp-to-free-tip cards. Never assume V=0 (or V=1) means "root".
        uv_delta = np.full((len(cards), 2), np.nan)
        for i, card in enumerate(cards):
            uv = np.asarray(vertex_uvs)[card.vertices]
            valid = np.isfinite(uv).all(axis=1)
            a = uv[valid & (card.rung == 0)]
            b = uv[valid & (card.rung == card.rungs - 1)]
            if len(a) and len(b):
                uv_delta[i] = b.mean(axis=0) - a.mean(axis=0)
        teachers = ((np.minimum(first_d, last_d) <= root_radius)
                    & (np.maximum(first_d, last_d) > root_radius * 1.5))
        close_ends = np.maximum(first_d, last_d) <= root_radius * 2.5
        for surface in np.unique([c.surface for c in cards]):
            group = np.array([c.surface == surface for c in cards])
            known = np.flatnonzero(group & teachers & np.isfinite(uv_delta).all(axis=1))
            if len(known) < 8:
                continue
            direction = uv_delta[known] * np.where(first_d[known] <= last_d[known], 1., -1.)[:, None]
            axis = int(np.argmax(np.median(np.abs(direction), axis=0)))
            significant = ((np.abs(direction[:, axis]) > 1.e-4)
                           & (np.argmax(np.abs(direction), axis=1) == axis))
            signs = np.sign(direction[significant, axis])
            if len(signs) < 8 or abs(signs.mean()) < .8:  # At least 90% agree.
                continue
            convention = np.sign(signs.mean())
            candidates = (group & close_ends & np.isfinite(uv_delta).all(axis=1)
                          & (np.argmax(np.abs(uv_delta), axis=1) == axis)
                          & (np.abs(uv_delta[:, axis]) > 1.e-4))
            uv_vote[candidates] = -convention * uv_delta[candidates, axis]
    for i, (card, mids) in enumerate(zip(cards, rung_mids)):
        da, db = first_d[i], last_d[i]
        flip = (db < da) if not ambiguous[i] or flip_votes[i] == 0 else flip_votes[i] > 0
        if uv_vote[i] != 0:
            flip = uv_vote[i] > 0
        if on_skin[i] and uv_vote[i] == 0:
            flip = db < da          # rooted on the skin: the end nearer the skin, not a vote from other hair
        if grown[i]:
            # Grown hair is written root first: the end holding the strip's first vertex is the root. No guessing from
            # distances, neighbours or UVs.
            flip = int(card.rung[int(np.argmin(card.vertices))]) != 0
        if flip:
            rung_mids[i] = mids[::-1]
            card.rung = card.rungs - 1 - card.rung
            da, db = db, da
        root_distance[i] = da

    # Particles: resample each centre line by arc length.
    starts, counts, particles, frames_all = [], [], [], []
    rung_arc = []
    for i, mids in enumerate(rung_mids):
        seg = np.linalg.norm(np.diff(mids, axis=0), axis=1)
        arc = np.concatenate([[0.], np.cumsum(seg)])
        # Grown hair keeps a particle every segment_length however long it is.
        cap = max(max_particles, GROWN_MAX_PARTICLES) if grown[i] else max_particles
        count = int(np.clip(np.ceil(arc[-1] / segment_length) + 1, 3, cap))
        samples = np.linspace(0, arc[-1], count)
        points = np.stack([np.interp(samples, arc, mids[:, k]) for k in range(3)], axis=1)
        starts.append(sum(counts)); counts.append(count)
        particles.append(points); frames_all.append(_frames(points)); rung_arc.append((arc, samples))
    particles = np.concatenate(particles) if particles else np.empty((0, 3))
    rest_frames = np.concatenate(frames_all) if frames_all else np.empty((0, 3, 3))
    starts, counts = np.array(starts, int), np.array(counts, int)

    progress("Binding card vertices…")
    n = len(vertices)
    strand_of_vertex = np.full(n, -1)
    bind_segment = np.zeros(n, int); bind_t = np.zeros(n); bind_offset = np.zeros((n, 3))
    for s, (card, mids) in enumerate(zip(cards, rung_mids)):
        arc, samples = rung_arc[s]
        at = arc[card.rung]                                      # arc position of each vertex's rung
        j = np.clip(np.searchsorted(samples, at, side="right") - 1, 0, counts[s] - 2)
        t = (at - samples[j]) / np.maximum(samples[j + 1] - samples[j], 1e-9)
        base = starts[s]
        p = particles[base + j] * (1 - t)[:, None] + particles[base + j + 1] * t[:, None]
        offset = vertices[card.vertices] - p
        frame = rest_frames[base + j]
        strand_of_vertex[card.vertices] = s
        bind_segment[card.vertices] = base + j
        bind_t[card.vertices] = t
        bind_offset[card.vertices] = np.einsum("nij,nj->ni", frame, offset)

    progress("Anchoring strands…")
    root_kind = np.full(len(cards), 2)
    root_kind[root_distance <= root_radius] = 0
    parent = np.full(len(cards), -1)
    owner = np.repeat(np.arange(len(cards)), counts) if len(cards) else np.empty(0, int)
    primary_mask = root_kind[owner] == 0 if len(cards) else np.empty(0, bool)
    if len(particles):
        grid = _Grid(particles, 1.0)
        # Nearby hair tips can curl back over the scalp. They are geometrically
        # close but are poor supports: prefer the scalp-side point within reach.
        parent_reach = np.concatenate([np.r_[0., np.cumsum(np.linalg.norm(
            np.diff(particles[start:start + count], axis=0), axis=1))]
            for start, count in zip(starts, counts)])
        parent_reach += root_distance[owner]
        for s in np.flatnonzero(root_kind != 0):
            root = particles[starts[s]]
            mine = lambda ids, s=s: owner[ids] == s
            # Prefer a scalp-rooted strand, then any other strand.
            hit, _ = grid.nearest(root, attach_radius, exclude=lambda ids, s=s: (owner[ids] == s) | ~primary_mask[ids], priority=parent_reach)
            if hit < 0:
                hit, _ = grid.nearest(root, attach_radius, exclude=mine, priority=parent_reach)
            if hit >= 0:
                parent[s] = hit
                root_kind[s] = 1
        _break_parent_cycles(root_kind, parent, owner)

    return HairModel(
        rest=vertices, strand_of_vertex=strand_of_vertex, particle_start=starts, particle_count=counts,
        particles=particles, strand_surface=np.array([c.surface for c in cards], int), root_kind=root_kind,
        parent_particle=parent, root_distance=root_distance, bind_segment=bind_segment, bind_t=bind_t,
        bind_offset=bind_offset, rest_frames=rest_frames, static_vertices=static,
        surface_names=list(surface_names), unsupported_cards=unsupported,
        card_vertices=[c.vertices for c in cards])


def _break_parent_cycles(root_kind, parent, owner):
    """A strand must eventually hang from the scalp; cycles of parent links are cut."""
    for s in np.flatnonzero(root_kind == 1):
        seen, current = set(), s
        while root_kind[current] == 1:
            if current in seen:
                root_kind[s], parent[s] = 2, -1
                break
            seen.add(current)
            current = owner[parent[current]]


def tip_anchors(model, frozen_strands, radius=1.5):
    """(tip particle, frozen particle) pairs: free strand tips touching a frozen strand."""
    frozen = np.zeros(model.strand_count, bool)
    frozen[list(frozen_strands)] = True
    if not frozen.any():
        return np.empty((0, 2), int)
    owner = np.repeat(np.arange(model.strand_count), model.particle_count)
    grid = _Grid(model.particles, 1.0)
    pairs = []
    for s in np.flatnonzero(~frozen):
        tip = model.particle_start[s] + model.particle_count[s] - 1
        hit, _ = grid.nearest(model.particles[tip], radius, exclude=lambda ids: ~frozen[owner[ids]])
        if hit >= 0:
            pairs.append((tip, hit))
    return np.array(pairs, int).reshape(-1, 2)


def unanchored(model, frozen_strands=()):
    """Strands with a free root that are not frozen."""
    frozen = set(int(s) for s in frozen_strands)
    return [int(s) for s in np.flatnonzero(model.root_kind == 2) if int(s) not in frozen]


def empty_model(vertices=np.zeros((0, 3)), surface_names=()):
    """A hair model with no strands (all vertices held still)."""
    n = len(vertices)
    return HairModel(rest=np.asarray(vertices, float).reshape(-1, 3), strand_of_vertex=np.full(n, -1),
                     particle_start=np.zeros(0, int), particle_count=np.zeros(0, int), particles=np.zeros((0, 3)),
                     strand_surface=np.zeros(0, int), root_kind=np.zeros(0, int), parent_particle=np.zeros(0, int),
                     root_distance=np.zeros(0), bind_segment=np.zeros(n, int), bind_t=np.zeros(n),
                     bind_offset=np.zeros((n, 3)), rest_frames=np.zeros((0, 3, 3)), static_vertices=np.ones(n, bool),
                     surface_names=list(surface_names))


def merge_models(captured, grown, vertex_offset):
    """Captured hair's model followed by a grown hair model built on its own vertices, so a change to grown hair
    analyzes only the grown strips while the captured hair's analysis is kept. Grown strands are rooted on the skin,
    and the captured hair is analyzed exactly as without
    them: no captured strand hangs from grown hair (a full joint analysis could let one pick a grown strand as its
    parent; grown hair should not hold up the hairstyle)."""
    particle_offset = len(captured.particles)
    strand_offset = captured.strand_count

    def shifted(values, offset):
        values = np.asarray(values)
        return np.where(values >= 0, values + offset, values)
    return HairModel(
        rest=np.concatenate([captured.rest, grown.rest]),
        strand_of_vertex=np.concatenate([captured.strand_of_vertex, shifted(grown.strand_of_vertex, strand_offset)]),
        particle_start=np.concatenate([captured.particle_start, np.asarray(grown.particle_start) + particle_offset]),
        particle_count=np.concatenate([captured.particle_count, grown.particle_count]),
        particles=np.concatenate([captured.particles, grown.particles]),
        strand_surface=np.concatenate([captured.strand_surface, grown.strand_surface]),
        root_kind=np.concatenate([captured.root_kind, grown.root_kind]),
        parent_particle=np.concatenate([captured.parent_particle, shifted(grown.parent_particle, particle_offset)]),
        root_distance=np.concatenate([captured.root_distance, grown.root_distance]),
        bind_segment=np.concatenate([captured.bind_segment, np.asarray(grown.bind_segment) + particle_offset]),
        bind_t=np.concatenate([captured.bind_t, grown.bind_t]),
        bind_offset=np.concatenate([captured.bind_offset, grown.bind_offset]),
        rest_frames=np.concatenate([captured.rest_frames, grown.rest_frames]),
        static_vertices=np.concatenate([captured.static_vertices, grown.static_vertices]),
        surface_names=list(grown.surface_names),
        unsupported_cards=captured.unsupported_cards + grown.unsupported_cards,
        card_vertices=list(captured.card_vertices) + [np.asarray(c) + vertex_offset for c in grown.card_vertices])
