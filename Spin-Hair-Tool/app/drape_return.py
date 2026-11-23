# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Write a static bridge response while preserving each captured OBJ's topology and UVs."""

import json
import os
from pathlib import Path

import numpy as np

from drape_fbx import write_fbx
from hair_duf import write_duf
from water_scene import write_water_plane


def write_animation_result(spec, frames, ranges):
    if frames is None or len(ranges) != len(spec["garments"]):
        raise ValueError("Run a timeline drape before returning an animation.")
    offset = 0
    for entry, (start, end) in zip(spec["garments"], ranges):
        if start != offset or end - start != entry["vertex_count"] or end <= start:
            raise ValueError("The animation vertex ranges do not match the captured garments.")
        offset = end
    if offset != frames.vertex_count:
        raise ValueError("The animation contains unexpected garment vertices.")
    response = Path(spec["result_path"])
    written, temporary = [], []
    try:
        returned = []
        for entry, vertex_range in zip(spec["garments"], ranges):
            target = Path(entry.get("animation_path", str(Path(entry["return_path"]).with_suffix(".fbx"))))
            # DAZ native return: DAZ's FBX importer slows with the square of the mesh size for
            # blend shapes; the same animation as a DUF scene loads in seconds. FBX stays for compatibility.
            native = spec.get("animation_format", "fbx") == "duf"
            animation = target.with_suffix(".duf") if native else target
            temp = animation.with_name(animation.name + ".tmp")
            temporary.append(temp)
            (write_duf if native else write_fbx)(temp, entry, frames, vertex_range)
            os.replace(temp, animation)
            written.append(animation)
            # One initial OBJ preserves the existing topology/UV validation path.
            base_target = Path(entry["return_path"])
            base_temp = base_target.with_name(base_target.name + ".tmp")
            temporary.append(base_temp)
            start, end = vertex_range
            base = frames.get(0)[start:end]
            index = 0
            facet_count = 0
            with open(entry["path"], encoding="utf-8-sig") as source, base_temp.open("w", encoding="utf-8") as output:
                for line in source:
                    if line.split()[:1] == ["f"]:
                        facet_count += 1
                    if line.split()[:1] == ["v"]:
                        if index >= len(base):
                            raise ValueError("The captured garment has an unexpected vertex count.")
                        output.write("v %.9g %.9g %.9g\n" % tuple(base[index]))
                        index += 1
                    else:
                        output.write(line)
            if index != len(base):
                raise ValueError("The captured garment has an unexpected vertex count.")
            os.replace(base_temp, base_target)
            written.append(base_target)
            indices = np.unique(np.linspace(0, end - start - 1, min(12, end - start), dtype=int))
            checks = [dict(frame=frames.start_frame + sample,
                           vertices=frames.get(sample)[start:end][indices].tolist())
                      for sample in sorted({0, frames.count // 2, frames.count - 1})]
            returned.append(dict(id=entry["id"], path=str(target), vertex_count=entry["vertex_count"],
                                 facet_count=facet_count,
                                 check_indices=indices.tolist(), checks=checks,
                                 **(dict(format="duf", duf_path=str(animation)) if native else {})))
        payload = dict(version=1, session_id=spec["session_id"], output_mode="animation",
                       start_frame=frames.start_frame, end_frame=frames.end_frame, fps=frames.fps,
                       take="DrapeAnimation", garments=returned)
        water = write_water_plane(spec, written, temporary)
        if water is not None:
            payload['water_plane'] = water
        temp = response.with_name(response.name + ".tmp")
        temporary.append(temp)
        temp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temp, response)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def write_static_result(spec, vertices, ranges, cap=None):
    """``cap`` (hair tool's optional scalp cap): extra cards appended after the original
    polygons, copying each source face's UV and surface; its vertices follow the original ones."""
    points = np.asarray(vertices)
    cap_count = 0 if cap is None else len(cap["vertices"])
    if cap_count:
        if len(garments := spec["garments"]) != 1 or len(points) != ranges[-1][1] + cap_count:
            raise ValueError("The scalp cap does not match the captured hair.")
        cap_points = points[ranges[-1][1]:]
        points = points[:ranges[-1][1]]
    garments = spec["garments"]
    if (
        points.ndim != 2 or points.shape[1] != 3
        or len(garments) != len(ranges) or not np.isfinite(points).all()
    ):
        raise ValueError("The static result does not match the captured garments.")
    offset = 0
    for entry, (start, end) in zip(garments, ranges):
        if start != offset or end <= start or end > len(points) or end - start != entry["vertex_count"]:
            raise ValueError("The garment vertex ranges do not match the captured meshes.")
        offset = end
    if offset != len(points):
        raise ValueError("The result contains unexpected garment vertices.")
    response_path = Path(spec["result_path"])
    temporary_paths = []
    written = []
    try:
        returned = []
        for entry, (start, end) in zip(garments, ranges):
            expected = entry["vertex_count"]
            target = Path(entry["return_path"])
            temporary = target.with_name(target.name + ".tmp")
            temporary_paths.append(temporary)
            index = start
            wanted_faces = {} if not cap_count else {int(f): None for f in cap["source_face"]}
            face_index, token = 0, ""
            facet_count = 0
            with (
                open(entry["path"], encoding="utf-8-sig") as source,
                temporary.open("w", encoding="utf-8", newline="\n") as output,
            ):
                for line in source:
                    fields = line.split()
                    if fields and fields[0] == "f":
                        facet_count += 1
                    if fields and fields[0] == "v":
                        if index >= end:
                            raise ValueError("The captured garment contains more vertices than expected.")
                        x, y, z = points[index]
                        output.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
                        index += 1
                    else:
                        output.write(line)
                        if wanted_faces and fields:
                            if fields[0] == "usemtl":
                                token = fields[1] if len(fields) > 1 else ""
                            elif fields[0] == "f":
                                corners = fields[1:]
                                split = [corners] if len(corners) <= 4 else [
                                    [corners[0], corners[i], corners[i + 1]] for i in range(1, len(corners) - 1)]
                                for corner_set in split:
                                    if face_index in wanted_faces:
                                        wanted_faces[face_index] = (token, corner_set)
                                    face_index += 1
                if cap_count:
                    # Cap cards: new vertices, the source faces' UV/normal references and surfaces.
                    output.write("# Spin Hair scalp cap\n")
                    for x, y, z in cap_points:
                        output.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
                    current = None
                    for local, source in zip(cap["faces"], cap["source_face"]):
                        found = wanted_faces.get(int(source))
                        if found is None:
                            raise ValueError("A scalp cap card has no source polygon.")
                        face_token, corner_set = found
                        if face_token != current:
                            output.write(f"usemtl {face_token}\n")
                            current = face_token
                        refs = []
                        for vertex, corner in zip([v for v in local if v >= 0], corner_set):
                            rest = corner.split("/", 1)
                            refs.append(str(int(vertex) + 1) + ("/" + rest[1] if len(rest) > 1 else ""))
                        output.write("f " + " ".join(refs) + "\n")
            if index != end:
                raise ValueError("The captured garment contains fewer vertices than expected.")
            os.replace(temporary, target)
            written.append(target)
            # Hair meshes are large: DAZ verifies counts plus these evenly spaced
            # samples instead of re-reading the OBJ. Topology, UVs and surfaces are
            # unchanged by construction (every non-vertex line is copied verbatim).
            check_indices = sorted(set(np.linspace(0, expected - 1, min(expected, 2048)).astype(int).tolist()))
            check_vertices = [[float(value) for value in points[start + i]] for i in check_indices]
            item = dict(id=entry["id"], path=str(target), vertex_count=expected + cap_count,
                        facet_count=facet_count + (int(len(cap["faces"])) if cap_count else 0),
                        check_indices=check_indices, check_vertices=check_vertices)
            if cap_count:
                item.update(cap_vertices=cap_count, cap_facets=int(len(cap["faces"])))
            returned.append(item)
        payload = dict(
            version=1, session_id=spec["session_id"], output_mode="static",
            frame=spec["return_frame"], garments=returned,
        )
        water = write_water_plane(spec, written, temporary_paths)
        if water is not None:
            payload['water_plane'] = water
        temporary_response = response_path.with_name(response_path.name + ".tmp")
        temporary_paths.append(temporary_response)
        temporary_response.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        # The DAZ bridge only imports after this complete response becomes visible.
        os.replace(temporary_response, response_path)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
