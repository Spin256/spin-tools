# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Split an OBJ into several props without changing the faces' world positions.

The per-piece records follow ``split_obj`` exactly (same OBJ writing, vertex
mapping, polygons and pivot fields) and add each piece's ``name``. Faces not
assigned to any piece form a final "Remainder" piece.
"""

import json
from pathlib import Path
import re

from mesh import load_obj


def piece_file_name(index, name):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "piece"
    return f"{index + 1:02d}-{slug}"


def split_pieces(source, result, pieces, allow_ngons=False):
    """``pieces`` is a list of (name, face_ids, pivot or None); returns the written records.

    ``allow_ngons`` permits polygons with more than four corners, for bridges that
    accept them. The DAZ bridge never sets it: DAZ verification needs triangles/quads.
    """
    source, result = Path(source), Path(result)
    mesh = load_obj(source)
    all_faces = set(range(len(mesh.faces)))
    assigned, used_names = set(), set()
    for name, face_ids, _ in pieces:
        face_ids = set(face_ids)
        if not face_ids or not face_ids <= all_faces:
            raise ValueError(f"Piece '{name}' has no polygons or refers to polygons outside the mesh.")
        if face_ids & assigned:
            raise ValueError(f"Piece '{name}' shares polygons with another piece.")
        if name.lower() in used_names:
            raise ValueError(f"Two pieces are named '{name}'. Give each piece a unique name.")
        assigned |= face_ids
        used_names.add(name.lower())
    remainder = all_faces - assigned
    pieces = [(name, set(faces), pivot) for name, faces, pivot in pieces]
    if remainder:
        pieces.append(("Remainder", remainder, None))
    if len(pieces) < 2:
        raise ValueError("Split off at least one piece and leave something else in another piece.")
    if not allow_ngons and any(len(face) not in (3, 4) for face in mesh.faces):
        raise ValueError("The DAZ bridge supports triangles and quads. Use Split n-gons first.")
    records = [line.partition("#")[0].split()
               for line in source.read_text(encoding="utf-8-sig").splitlines()]
    if any(fields and fields[0] in {"l", "p", "curv", "surf"} for fields in records):
        raise ValueError("The DAZ bridge supports polygon meshes only.")

    written = []
    for index, (name, face_ids, pivot) in enumerate(pieces):
        vertices = sorted({vertex for face in face_ids for vertex in mesh.faces[face]})
        indices = {old: new for new, old in enumerate(vertices, 1)}
        stem = piece_file_name(index, name)
        destination = result.with_name(result.stem + "-" + stem + ".obj")
        lines = [f"o {stem}"]
        lines.extend("v " + " ".join(map(str, mesh.vertices[vertex])) for vertex in vertices)
        face_id = 0
        texture_count = normal_count = 0
        for fields in records:
            if not fields:
                continue
            record = fields[0]
            if record in {"vt", "vn", "usemtl", "mtllib", "g", "s"}:
                lines.append(" ".join(fields))
                texture_count += record == "vt"
                normal_count += record == "vn"
            elif record == "f":
                if face_id in face_ids:
                    tokens = []
                    for vertex, token in zip(mesh.faces[face_id], fields[1:]):
                        parts = token.split("/")
                        parts[0] = str(indices[vertex])
                        for slot, count in enumerate((texture_count, normal_count), 1):
                            if len(parts) > slot and parts[slot].startswith("-"):
                                parts[slot] = str(count + int(parts[slot]) + 1)
                        tokens.append("/".join(parts))
                    lines.append("f " + " ".join(tokens))
                face_id += 1
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        points = [mesh.vertices[vertex] for vertex in vertices]
        center = [(min(axis) + max(axis)) / 2 for axis in zip(*points)]
        written.append({
            "name": name,
            "file": destination.name,
            "vertices": points,
            "faces": len(face_ids),
            "source_vertices": vertices,
            "polygons": [[indices[vertex] - 1 for vertex in mesh.faces[face]] for face in sorted(face_ids)],
            "pivot": list(pivot) if pivot is not None else center,
        })
    temporary = result.with_suffix(".tmp")
    temporary.write_text(json.dumps(written), encoding="utf-8")
    temporary.replace(result)
    return written
