# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Load a Spin Tools hair capture: the hair mesh (raw quads, in vertex order) and the colliders."""
import json
import math
from pathlib import Path
from collider_roles import ROLES

import numpy as np


def read_obj(path):
    """Vertices (cm), faces as (F, 4) with -1 for triangles, and the material token per face.

    Raw topology: nothing is dropped or reordered, so vertex order matches DAZ exactly.
    Polygons with more than four corners are fan-split (they are rare in hair and only
    used for collision or static surfaces).
    """
    vertices, faces, face_tokens = [], [], []
    token = ""
    with open(path, encoding="utf-8-sig") as source:
        for line in source:
            if line.startswith("v "):
                vertices.append(line.split()[1:4])
            elif line.startswith("f "):
                corners = [int(part.split("/")[0]) for part in line.split()[1:]]
                corners = [c - 1 if c > 0 else len(vertices) + c for c in corners]
                if len(corners) <= 4:
                    faces.append(corners + [-1] * (4 - len(corners)))
                    face_tokens.append(token)
                else:
                    for i in range(1, len(corners) - 1):
                        faces.append([corners[0], corners[i], corners[i + 1], -1])
                        face_tokens.append(token)
            elif line.startswith("usemtl"):
                fields = line.split()
                token = fields[1] if len(fields) > 1 else ""
    points = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 4)
    if not len(points) or not len(faces) or faces.max() >= len(points) or faces[:, :3].min() < 0:
        raise ValueError(f"Invalid OBJ mesh: {path}")
    if not np.isfinite(points).all():
        raise ValueError(f"The OBJ contains invalid coordinates: {path}")
    return points, faces, face_tokens


def triangles(faces):
    """(T, 3) triangles from (F, 4) quads/triangles."""
    tri = faces[:, :3]
    quads = faces[faces[:, 3] >= 0]
    return np.concatenate([tri, quads[:, [0, 2, 3]]])


def read_vertex_uvs(path, vertex_count):
    """Optional UV hints for root inference; missing/invalid hints stay unknown."""
    texcoords, vertices, textures = [], [], []
    seen_vertices = 0
    with open(path, encoding="utf-8-sig") as source:
        for line in source:
            if line.startswith('v '):
                seen_vertices += 1
            elif line.startswith('vt '):
                fields = line.split()
                try:
                    texcoords.append([float(fields[1]), float(fields[2]) if len(fields) > 2 else 0.])
                except (ValueError, IndexError):
                    texcoords.append([np.nan, np.nan])
            elif line.startswith('f '):
                for token in line.split()[1:]:
                    fields = token.split('/')
                    if len(fields) < 2 or not fields[1]:
                        continue
                    vertex, texture = int(fields[0]), int(fields[1])
                    vertex = vertex - 1 if vertex > 0 else seen_vertices + vertex
                    texture = texture - 1 if texture > 0 else len(texcoords) + texture
                    if 0 <= vertex < vertex_count and 0 <= texture < len(texcoords):
                        vertices.append(vertex); textures.append(texture)
    result = np.full((vertex_count, 2), np.nan)
    if vertices:
        sums = np.zeros_like(result)
        counts = np.zeros(vertex_count)
        np.add.at(sums, vertices, np.asarray(texcoords)[textures])
        np.add.at(counts, vertices, 1)
        valid = counts > 0
        result[valid] = sums[valid] / counts[valid, None]
    return result


def load_session(session_path):
    session_path = Path(session_path)
    data = json.loads(session_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("version") not in (1, 2):
        raise ValueError("This is not a Spin Hair capture.")
    if data.get("mode") != "current_frame":
        raise ValueError("The hair tool simulates the current frame. Capture without timeline movement.")
    if not data.get("live_return"):
        raise ValueError("This capture predates the Spin Tools receive window. Recapture it from DAZ.")
    entries = data.get("meshes")
    if not isinstance(entries, list):
        raise ValueError("The capture lists no meshes.")
    hair = [e for e in entries if isinstance(e, dict) and e.get("role") == "garment"]
    colliders = [e for e in entries if isinstance(e, dict) and e.get("role") in ROLES]
    # Hair is optional: an avatar on its own is captured to grow hair on it.
    if not colliders:
        raise ValueError("Assign the avatar as a collision object.")
    from collider_roles import primary_index
    primary_index(colliders)
    for e in hair + colliders:
        if not isinstance(e.get("path"), str) or not Path(e["path"]).is_file():
            raise ValueError(f"A captured mesh is missing: {e.get('label', '?')}")
    vertex_parts, face_parts, surface_parts, surface_names, ranges, uv_parts = [], [], [], [], [], []
    offset = 0
    for entry in hair:
        points, polys, tokens = read_obj(entry["path"])
        if entry.get("vertex_count") != len(points):
            raise ValueError("The captured hair does not match its vertex count.")
        if not isinstance(entry.get("return_path"), str):
            raise ValueError("The captured hair has no return path.")
        names_by_token = {m["token"]: m["name"] for m in entry.get("materials", []) if isinstance(m, dict)}
        token_names = sorted(set(tokens))
        index = {t: i + len(surface_names) for i, t in enumerate(token_names)}
        names = [names_by_token.get(t, t) for t in token_names]
        if len(hair) > 1:
            names = [f"{entry.get('label', 'Hair')}: {name}" for name in names]
        surface_names.extend(names)
        vertex_parts.append(points)
        uv_parts.append(read_vertex_uvs(entry["path"], len(points)))
        face_parts.append(np.where(polys >= 0, polys + offset, -1))
        surface_parts.append(np.array([index[t] for t in tokens], dtype=int))
        ranges.append((offset, offset + len(points)))
        offset += len(points)
    vertices = np.concatenate(vertex_parts) if vertex_parts else np.zeros((0, 3))
    faces = np.concatenate(face_parts) if face_parts else np.zeros((0, 4), int)
    face_surface = np.concatenate(surface_parts) if surface_parts else np.zeros(0, int)
    collision = []
    for e in colliders:
        points, polys, _ = read_obj(e["path"])
        collision.append(dict(label=e.get("label", "Collider"), role=e["role"], vertices=points,
                              triangles=triangles(polys)))
    if not isinstance(data.get("session_id"), str) or any(not isinstance(e.get("return_path"), str) for e in hair):
        raise ValueError("The capture has no return information.")
    frame = data.get("frame")
    if not isinstance(frame, (int, float)) or not math.isfinite(frame):
        raise ValueError("The capture has an invalid frame.")
    return dict(
        hair_label=" + ".join(e.get("label", "Hair") for e in hair) or "No hair (grow hair on the avatar)",
        vertices=vertices, faces=faces, face_surface=face_surface,
        surface_names=surface_names, colliders=collision,
        vertex_uvs=np.concatenate(uv_parts) if uv_parts else np.zeros((0, 2)),
        return_spec=dict(session_id=data["session_id"], result_path=data["result_path"], garments=hair,
                         return_frame=frame,
                         water_return_path=data.get("water_return_path", data["session_id"] + "-water.obj")),
        ranges=ranges, live_session=session_path)
