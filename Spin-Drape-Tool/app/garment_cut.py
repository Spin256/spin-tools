# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Cut a garment along mesh edges so it can fall apart.

A closed garment (a bra whose clasp is one piece of mesh) is split along a path of polygon edges: every
vertex on the path gets one copy per side, and the polygons on each side use their own copy. The returned
static mesh then has more vertices than the capture; DAZ imports it as a new prop and checks it against the
helper-written counts and samples. An optional release frame holds the two sides together until then."""
import heapq
from pathlib import Path

import numpy as np


def read_polygons(path):
    """Vertex indices of every "f" line, in file order (OBJ polygons, not the helper's triangles)."""
    polygons = []
    with open(path, encoding="utf-8-sig") as source:
        for line in source:
            fields = line.split()
            if fields and fields[0] == "f":
                polygons.append([int(corner.split("/")[0]) - 1 for corner in fields[1:]])
    return polygons


def polygon_edges(polygons):
    edges = {}
    for polygon in polygons:
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            edges.setdefault((min(a, b), max(a, b)), 0)
            edges[(min(a, b), max(a, b))] += 1
    return edges


def edge_path(points, polygons, start, end):
    """Shortest path between two vertices along polygon edges (never across a quad's diagonal)."""
    neighbours = {}
    for a, b in polygon_edges(polygons):
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    if start not in neighbours or end not in neighbours:
        raise ValueError("Both cut points must be on the garment's polygon edges.")
    distance = {start: 0.}
    previous = {}
    queue = [(0., start)]
    while queue:
        cost, vertex = heapq.heappop(queue)
        if vertex == end:
            break
        if cost > distance[vertex]:
            continue
        for other in neighbours[vertex]:
            step = cost + float(np.linalg.norm(points[other] - points[vertex]))
            if step < distance.get(other, np.inf):
                distance[other] = step
                previous[other] = vertex
                heapq.heappush(queue, (step, other))
    if end not in distance:
        raise ValueError("The cut points are on separate pieces of the garment.")
    path = [end]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return path[::-1]


def split_polygons(polygons, vertex_count, cut_edges):
    """Split vertices along the cut edges. Returns new polygons, the (original, copy) pairs and the new
    vertex count. Around each cut vertex the polygons are grouped into fans joined by uncut edges; the first
    fan keeps the original vertex, every other fan gets a copy. An interior path end has one fan (no split),
    so the opening starts there."""
    cut = {(min(a, b), max(a, b)) for a, b in cut_edges}
    cut_vertices = sorted({v for edge in cut for v in edge})
    around = {v: [] for v in cut_vertices}
    for index, polygon in enumerate(polygons):
        for v in polygon:
            if v in around:
                around[v].append(index)
    polygons = [list(polygon) for polygon in polygons]
    pairs = []
    count = vertex_count
    for vertex in cut_vertices:
        faces = around[vertex]
        group = {face: face for face in faces}

        def root(face):
            while group[face] != face:
                group[face] = group[group[face]]
                face = group[face]
            return face
        by_edge = {}
        for face in faces:
            polygon = polygons[face]
            k = polygon.index(vertex)
            for other in (polygon[k - 1], polygon[(k + 1) % len(polygon)]):
                edge = (min(vertex, other), max(vertex, other))
                if edge not in cut:
                    by_edge.setdefault(other, []).append(face)
        for joined in by_edge.values():
            for face in joined[1:]:
                group[root(face)] = root(joined[0])
        fans = {}
        for face in faces:
            fans.setdefault(root(face), []).append(face)
        for fan in list(fans.values())[1:]:
            for face in fan:
                polygon = polygons[face]
                polygon[polygon.index(vertex)] = count
            pairs.append((vertex, count))
            count += 1
    return polygons, pairs, count


def write_cut_obj(source, target, pairs, polygons):
    """Copy the captured OBJ with the copies appended after the last "v" line and the "f" lines pointing at
    them; UV, normal, group and material lines are kept as they are."""
    lines = Path(source).read_text(encoding="utf-8-sig").splitlines(keepends=True)
    vertices = [i for i, line in enumerate(lines) if line.split()[:1] == ["v"]]
    if not vertices:
        raise ValueError("The captured garment has no vertices.")
    originals = [lines[i] for i in vertices]
    output = []
    face = 0
    for i, line in enumerate(lines):
        fields = line.split()
        if fields[:1] == ["f"]:
            corners = [corner.split("/") for corner in fields[1:]]
            for corner, vertex in zip(corners, polygons[face]):
                corner[0] = str(vertex + 1)
            output.append("f " + " ".join("/".join(corner) for corner in corners) + "\n")
            face += 1
        else:
            output.append(line if line.endswith("\n") else line + "\n")
        if i == vertices[-1]:
            output.extend(originals[original] if originals[original].endswith("\n") else originals[original] + "\n"
                          for original, _ in pairs)
    if face != len(polygons):
        raise ValueError("The captured garment changed while it was being cut.")
    Path(target).write_text("".join(output), encoding="utf-8", newline="\n")


def cut_window_inputs(inputs, paths, picks, temp_folder, release_frame=None):
    """New DrapeWindow arguments with the garment cut along the picked vertices (global indices, in order).

    ``inputs``: the window's own arguments; ``paths``: each garment's template OBJ. Every garment is read back
    from its (possibly cut) template, so ranges, buttons and earlier cut pairs move to the new numbering. The
    return spec's garment entry points at the cut template with its new vertex count; original_vertex_count
    keeps the DAZ original's count (for information)."""
    from warp_drape import read_mesh
    ranges = list(inputs.get("garment_ranges") or [(0, len(inputs["garment"][0]))])
    owner = [g for g, (start, end) in enumerate(ranges) if all(start <= p < end for p in picks)]
    if len(picks) < 2 or not owner:
        raise ValueError("Click at least two points on the same garment.")
    g = owner[0]
    start = ranges[g][0]
    local = [p - start for p in picks]
    polygons = read_polygons(paths[g])
    template = read_mesh(paths[g], keep_degenerate=True)[0]
    edges = set()
    for a, b in zip(local, local[1:]):
        path = edge_path(template, polygons, a, b)
        edges.update(zip(path, path[1:]))
    polygons, pairs, count = split_polygons(polygons, len(template), edges)
    if not pairs:
        raise ValueError("Nothing to split: run the cut to a garment edge, or around a loop.")
    folder = Path(temp_folder)
    name = Path(paths[g]).stem
    number = 1
    while (folder / f"{name}-cut{number}.obj").exists():
        number += 1
    target = folder / f"{name}-cut{number}.obj"
    write_cut_obj(paths[g], target, pairs, polygons)
    paths = list(paths)
    paths[g] = str(target)

    meshes = [read_mesh(path) for path in paths]
    new_ranges, offset = [], 0
    for points, _ in meshes:
        new_ranges.append((offset, offset + len(points)))
        offset += len(points)

    def move(index):
        for (a, b), (c, _) in zip(ranges, new_ranges):
            if a <= index < b:
                return c + index - a
        raise ValueError("A vertex is outside every garment.")
    new = dict(inputs)
    new["garment"] = (np.concatenate([m[0] for m in meshes]),
                      np.concatenate([m[1] + r[0] for m, r in zip(meshes, new_ranges)]))
    new["garment_ranges"] = new_ranges
    new["button_vertices"] = [move(v) for v in inputs.get("button_vertices") or []]
    new["cut_pairs"] = ([(move(a), move(b)) for a, b in inputs.get("cut_pairs") or []] +
                        [(new_ranges[g][0] + a, new_ranges[g][0] + b) for a, b in pairs])
    new["cut_release"] = release_frame
    new["garment_paths"] = paths
    new["uncut"] = inputs.get("uncut") or dict(inputs)
    spec = inputs.get("return_spec")
    if spec is not None:
        garments = [dict(entry) for entry in spec["garments"]]
        entry = garments[g]
        entry.setdefault("original_vertex_count", entry["vertex_count"])
        entry["path"] = str(target)
        entry["vertex_count"] = count
        new["return_spec"] = dict(spec, garments=garments)
    return new, len(pairs)
