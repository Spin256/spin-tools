# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""OBJ topology and selection data for the polygon selector."""

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass
class Mesh:
    vertices: list
    faces: list

    def __post_init__(self):
        edge_faces = defaultdict(list)
        for face_id, face in enumerate(self.faces):
            for a, b in zip(face, face[1:] + face[:1]):
                edge_faces[tuple(sorted((a, b)))].append(face_id)
        self.face_edges = [[] for face in self.faces]
        for adjacent in edge_faces.values():
            for face_id in adjacent:
                self.face_edges[face_id].append(adjacent)

    def connected(self, seed):
        seen = {seed}
        visited_edges = set()
        pending = [seed]
        while pending:
            face_id = pending.pop()
            for adjacent in self.face_edges[face_id]:
                key = id(adjacent)
                if key in visited_edges:
                    continue
                visited_edges.add(key)
                for neighbor in adjacent:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        pending.append(neighbor)
        return seen


def load_obj(path):
    vertices, faces = [], []
    with Path(path).open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            fields = line.partition("#")[0].split()
            if not fields:
                continue
            try:
                if fields[0] == "v":
                    point = tuple(map(float, fields[1:4]))
                    if len(point) != 3 or not all(map(math.isfinite, point)):
                        raise ValueError("invalid vertex")
                    vertices.append(point)
                elif fields[0] == "f":
                    face = []
                    for token in fields[1:]:
                        index = int(token.split("/")[0])
                        resolved = index - 1 if index > 0 else len(vertices) + index
                        if not 0 <= resolved < len(vertices):
                            raise ValueError("vertex reference out of range")
                        face.append(resolved)
                    if len(face) < 3 or len(set(face)) != len(face):
                        raise ValueError("invalid polygon")
                    faces.append(tuple(face))
            except ValueError as error:
                raise ValueError(f"OBJ line {number}: {error}") from error
    if not faces:
        raise ValueError("The OBJ contains no polygons.")
    return Mesh(vertices, faces)


def demo_mesh():
    vertices = [
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    ]
    faces = [
        (3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]
    return Mesh(vertices + [(x * .55 + 2.2, y * .55, z * .55)
                            for x, y, z in vertices],
                faces + [tuple(i + 8 for i in face) for face in faces])


def save_selections(path, source, selection, groups, pivot):
    payload = {
        "version": 1,
        "source_obj": str(source) if source else None,
        "face_index_base": 0,
        "selected_faces": sorted(selection),
        "groups": {name: sorted(faces) for name, faces in groups.items()},
        "pivot": pivot,
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

