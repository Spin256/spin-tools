# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Load exact scene-role assignments from the DAZ selection dialog."""

import filecmp
import json
import math
import re
from pathlib import Path

import numpy as np

from mesh_io import read_mesh


def load_inputs(garment_entries, collision_entries, capture_frame=None, water_source=None, detect_water=True):
    if not garment_entries:
        raise ValueError("Select at least one garment and one collision mesh.")
    for garment in garment_entries:
        for collider in collision_entries:
            if filecmp.cmp(garment["path"], collider["path"], shallow=False):
                raise ValueError(
                    f'The garment {garment["label"]} matches collision mesh {collider["label"]}. '
                    "Check the scene assignments."
                )

    vertices = []
    faces = []
    ranges = []
    offset = 0
    button_vertices = []
    for entry in garment_entries:
        points, triangles = read_mesh(entry["path"])
        if entry.get("materials"):
            from drape_fbx import read_obj
            _, _, polygons, material_tokens = read_obj(entry["path"])
            button_tokens = {m["token"] for m in entry["materials"]
                             if re.search(r"button|btn", m["name"], re.IGNORECASE)}
            button_vertices.extend(offset + index for index in sorted({
                v for polygon, _, material in polygons if material_tokens[material] in button_tokens for v in polygon
            }))
        vertices.append(points)
        faces.append(triangles + offset)
        ranges.append((offset, offset + len(points)))
        offset += len(points)

    return dict(
        garment=(np.concatenate(vertices), np.concatenate(faces)),
        colliders=[read_mesh(entry["path"]) for entry in collision_entries],
        garment_label=", ".join(entry["label"] for entry in garment_entries),
        collision_labels=[entry["label"] for entry in collision_entries],
        collision_roles=[entry.get("role", "fixed") for entry in collision_entries],
        garment_ranges=ranges,
        capture_frame=capture_frame,
        button_vertices=button_vertices,
        water_source=water_source,
    )


def load_session(path):
    session_path = Path(path)
    data = json.loads(session_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("version") not in (1, 2) or data.get("mode") not in ("current_frame", "timeline"):
        raise ValueError("This helper requires a current-frame or timeline Drape session.")
    output_mode = data.get("output_mode", "static")
    if output_mode not in ("static", "animation"):
        raise ValueError("The Drape session contains an invalid output preference.")
    entries = data.get("meshes")
    if not isinstance(entries, list):
        raise ValueError("The Drape session has no scene assignments.")
    seen = set()
    garments = []
    colliders = []
    samples = {}
    body_samples = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("role") not in ("fixed", "moving", "garment"):
            raise ValueError("The Drape session contains an invalid role.")
        if entry.get("id") is None or entry["id"] in seen:
            raise ValueError("A scene object has more than one assignment.")
        seen.add(entry["id"])
        if not isinstance(entry.get("path"), str) or not isinstance(entry.get("label"), str):
            raise ValueError("The Drape session contains an invalid mesh entry.")
        entry = entry.copy()
        mesh_path = Path(entry["path"])
        if not mesh_path.is_absolute():
            mesh_path = session_path.parent / mesh_path
        entry["path"] = str(mesh_path)
        if entry["role"] == "garment":
            if data["mode"] == "timeline" and entry.get("samples") is not None:
                paths=entry["samples"]
                if not isinstance(paths,list) or not paths or not all(isinstance(path,str) for path in paths):
                    raise ValueError("An animated body has invalid timeline samples.")
                resolved=[str(Path(path) if Path(path).is_absolute() else session_path.parent/path) for path in paths]
                if Path(resolved[0]).resolve()!=Path(entry["path"]).resolve():
                    raise ValueError("An animated body's first sample does not match its captured mesh.")
                body_samples[len(garments)]=resolved
            garments.append(entry)
        else:
            if data["mode"] == "timeline" and entry["role"] == "moving":
                paths = entry.get("samples")
                if not isinstance(paths, list) or not paths or not all(isinstance(path, str) for path in paths):
                    raise ValueError("A moving collider has no timeline samples.")
                resolved = [str(Path(path) if Path(path).is_absolute() else session_path.parent / path) for path in paths]
                if Path(resolved[0]).resolve() != Path(entry["path"]).resolve():
                    raise ValueError("A moving collider's starting mesh does not match its first sample.")
                samples[len(colliders)] = resolved
            colliders.append(entry)
    inputs = load_inputs(garments, colliders, data.get("frame"), None, detect_water=False)
    inputs["output_mode"] = output_mode
    if data.get("result_path"):
        if not isinstance(data.get("session_id"), str):
            raise ValueError("The return session has no identifier.")
        returned = []
        for entry, (start, end) in zip(garments, inputs["garment_ranges"]):
            if not isinstance(entry.get("return_path"), str) or entry.get("vertex_count") != end - start:
                raise ValueError("The return session does not match its garment geometry.")
            returned.append(entry)
        inputs["return_spec"] = dict(
            session_id=data["session_id"], result_path=data["result_path"], garments=returned,
            return_frame=data.get("end_frame") if data["mode"] == "timeline" else data["frame"],
            water_return_path=data.get('water_return_path', data['session_id'] + '-water.obj'),
        )
    if data["mode"] == "timeline":
        from drape_motion import ColliderMotion

        fps = data.get("fps")
        if not isinstance(fps, (int, float)) or not math.isfinite(fps):
            raise ValueError("The captured timeline has an invalid frame rate.")
        from body_motion import SceneMotion
        bodies=[read_mesh(entry["path"]) for entry in garments]
        colliders_motion=ColliderMotion(inputs["colliders"],samples,fps,data.get("start_frame"),data.get("end_frame"),labels=[e['label'] for e in colliders]) if samples else None
        bodies_motion=ColliderMotion(bodies,body_samples,fps,data.get("start_frame"),data.get("end_frame"),role='Animated body',labels=[e['label'] for e in garments]) if body_samples else None
        inputs["motion"]=SceneMotion(bodies,bodies_motion,colliders_motion)
    return inputs
