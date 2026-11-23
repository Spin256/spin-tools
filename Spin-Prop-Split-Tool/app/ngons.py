# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Find and split polygons with more than four corners in an OBJ.

The DAZ bridge verifies pieces against DAZ's own facets (triangles and quads),
so a DAZ capture splits each n-gon into exactly the facets DAZ uses for it.
Without DAZ facets (standalone), n-gons are fan-triangulated.
"""
import os
from pathlib import Path


def _corner(token):
    return int(token.split("/")[0])


def ngon_report(path):
    """Return (count, surfaces) for polygons with more than four corners."""
    count, surfaces, material = 0, [], ""
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        fields = line.partition("#")[0].split()
        if not fields:
            continue
        if fields[0] == "usemtl" and len(fields) > 1:
            material = fields[1]
        elif fields[0] == "f" and len(fields) > 5:
            count += 1
            if material not in surfaces:
                surfaces.append(material)
    return count, surfaces


def _daz_split(vertices, facets_by_vertex):
    """Facets lying wholly inside this polygon, if they exactly tile it."""
    polygon = set(vertices)
    candidates = {facet for v in vertices for facet in facets_by_vertex.get(v, ()) if set(facet) <= polygon}
    if sum(len(facet) - 2 for facet in candidates) != len(vertices) - 2:
        return None
    position = {v: i for i, v in enumerate(vertices)}
    return [tuple(sorted(facet, key=lambda v: (position[v] - position[facet[0]]) % len(vertices)))
            for facet in sorted(candidates)]


def split_ngons(path, destination=None, facets=None):
    """Rewrite the OBJ with every n-gon split; returns (split, used_daz_facets).

    ``facets`` are DAZ facets as zero-based vertex-index lists in OBJ vertex order.
    Vertex/UV tokens are reused unchanged, so UVs, surfaces and vertex order stay.
    """
    path = Path(path)
    destination = Path(destination or path)
    facets_by_vertex = {}
    for facet in facets or ():
        facet = tuple(int(v) for v in facet)
        for v in facet:
            facets_by_vertex.setdefault(v, []).append(facet)
    lines, split, exact, vertex_count = [], 0, 0, 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.partition("#")[0].split()
        if fields and fields[0] == "v":
            vertex_count += 1
        if not fields or fields[0] != "f" or len(fields) <= 5:
            lines.append(line)
            continue
        tokens = fields[1:]
        indices = [(_corner(t) - 1) if _corner(t) > 0 else vertex_count + _corner(t) for t in tokens]
        token_for = dict(zip(indices, tokens))
        pieces = _daz_split(indices, facets_by_vertex) if facets_by_vertex else None
        if pieces is None:
            pieces = [(indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)]
        else:
            exact += 1
        lines.extend("f " + " ".join(token_for[v] for v in piece) for piece in pieces)
        split += 1
    if split or destination != path:
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    return split, exact == split and split > 0
