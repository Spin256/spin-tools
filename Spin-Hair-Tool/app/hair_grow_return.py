# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Grown hair in a still return (Spin256 · spin256.com/tools).

Each grown area returns as its own new prop: an OBJ of its strips (as simulated) with UVs from root (v 0) to tip
(v 1) and one surface named after the area, listed in the result under "grown" with its color and check samples.
Called after the captured hair's writer has written the result; the result is rewritten atomically with the
"grown" list added, before the revision is published."""
import json
import os
import re
from pathlib import Path

import numpy as np

CHECK_SAMPLES = 256


def surface_token(name, index):
    token = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "Grown"
    return f"{token}_{index + 1}"


def grown_stem(session_id):
    """The last part of the session id (DAZ captures use the capture's path as the id)."""
    return str(session_id).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def write_grown_obj(path, vertices, faces, uvs, token, secondary=None, token2=None):
    """``secondary``: per face, True for the secondary color's surface ``token2`` (its faces follow the primary ones)."""
    lines = ["# Spin Hair grown hair"]
    lines += ["v %.9g %.9g %.9g" % tuple(v) for v in vertices]
    lines += ["vt %.6g %.6g" % tuple(uv) for uv in uvs]
    secondary = np.zeros(len(faces), bool) if secondary is None else np.asarray(secondary, bool)
    for surface, chosen in ((token, ~secondary), (token2, secondary)):
        if not chosen.any():
            continue
        lines.append("usemtl " + surface)
        for face in np.asarray(faces)[chosen]:
            corners = [int(c) + 1 for c in face if c >= 0]
            lines.append("f " + " ".join(f"{c}/{c}" for c in corners))
    temporary = Path(str(path) + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def add_grown(spec, vertices, inputs):
    """Write each grown area's OBJ next to the result and add the "grown" list to the result JSON."""
    result_path = Path(spec["result_path"])
    folder = result_path.parent
    vertices = np.asarray(vertices, float)
    uvs = np.asarray(inputs["vertex_uvs"], float) if inputs.get("vertex_uvs") is not None else None
    faces = np.asarray(inputs["faces"])
    entries = []
    for index, part in enumerate(inputs.get("grown") or []):
        start, end = part["start"], part["end"]
        points = vertices[start:end]
        if len(points) != end - start or not np.isfinite(points).all():
            raise ValueError(f"The grown hair '{part['name']}' does not match the simulated hair.")
        area_faces = faces[part["face_start"]:part["face_end"]]
        local = np.where(area_faces >= 0, area_faces - start, -1)
        area_uvs = uvs[start:end] if uvs is not None else np.zeros((len(points), 2))
        # Surfaces are named after the area's named materials.
        material = part.get("material") or part["name"]
        material2 = part.get("material2") or material + " secondary"
        token = surface_token(material, index)
        # The session id can be a full path (the capture's own); the file goes next to the result either way.
        path = folder / f"{grown_stem(spec['session_id'])}-grown-{index + 1}.obj"
        second = np.asarray(part.get("face_secondary", np.zeros(len(local), bool)), bool)
        token2 = surface_token(material2, index)
        if token2 == token:
            token2 = token + "_secondary"
        write_grown_obj(path, points, local, area_uvs, token, second, token2)
        # Each surface with its color and label: one material per color (two colors, no gradient).
        surfaces = [dict(token=token, label=material, color=[int(c) for c in part["color"]])]
        if second.any():
            surfaces.append(dict(token=token2, label=material2,
                                 color=[int(c) for c in part.get("color2", part["color"])]))
        check = sorted(set(np.linspace(0, len(points) - 1, min(len(points), CHECK_SAMPLES)).astype(int).tolist()))
        entries.append(dict(name=part["name"], path=str(path), token=token, color=[int(c) for c in part["color"]],
                            surfaces=surfaces,
                            vertex_count=int(len(points)), facet_count=int(len(local)), strands=int(part["strands"]),
                            check_indices=check, check_vertices=[[float(x) for x in points[i]] for i in check]))
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["grown"] = entries
    temporary = result_path.with_name(result_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, result_path)
    return entries
