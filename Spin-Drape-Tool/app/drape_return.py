# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Write a static bridge response while preserving each captured OBJ's topology and UVs."""

import json
import os
from pathlib import Path

import numpy as np

from drape_fbx import write_fbx
from duf_return import write_duf
from water_scene import write_water_plane
from ground_scene import write_ground_plane

CHECK_SAMPLES = 2048


def check_samples(points):
    """Evenly spaced vertex indices and positions DAZ checks after import."""
    indices = sorted(set(np.linspace(0, len(points) - 1, min(len(points), CHECK_SAMPLES)).astype(int).tolist()))
    return indices, [[float(value) for value in points[i]] for i in indices]


class TemplateCounts:
    """Polygons (one per captured "f" line: DAZ captures triangles and quads) and surfaces
    (distinct "usemtl" tokens with polygons), counted while the template is copied."""

    def __init__(self):
        self.facets, self.groups, self.token = 0, set(), ""

    def line(self, fields):
        if fields and fields[0] == "usemtl":
            self.token = fields[1] if len(fields) > 1 else ""
        elif fields and fields[0] == "f":
            self.facets += 1
            self.groups.add(self.token)


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
            # DAZ native return (as Hair's): DAZ's FBX importer slows with the square of the
            # mesh size for blend shapes; the same animation as a DUF scene loads in seconds. FBX stays selectable.
            native = spec.get("animation_format", "fbx") == "duf"
            animation = target.with_suffix(".duf") if native else target
            temp = animation.with_name(animation.name + ".tmp")
            temporary.append(temp)
            if native:
                write_duf(temp, entry, frames, vertex_range, label='Simulated garment')
            else:
                write_fbx(temp, entry, frames, vertex_range)
            os.replace(temp, animation)
            written.append(animation)
            # One initial OBJ preserves the existing topology/UV validation path.
            base_target = Path(entry["return_path"])
            base_temp = base_target.with_name(base_target.name + ".tmp")
            temporary.append(base_temp)
            start, end = vertex_range
            base = frames.get(0)[start:end]
            index = 0
            counts = TemplateCounts()
            with open(entry["path"], encoding="utf-8-sig") as source, base_temp.open("w", encoding="utf-8") as output:
                for line in source:
                    if line.split()[:1] == ["v"]:
                        if index >= len(base):
                            raise ValueError("The captured garment has an unexpected vertex count.")
                        output.write("v %.9g %.9g %.9g\n" % tuple(base[index]))
                        index += 1
                    else:
                        counts.line(line.split())
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
                                 check_indices=indices.tolist(), checks=checks,
                                 facet_count=counts.facets, group_count=len(counts.groups),
                                 **(dict(format="duf", duf_path=str(animation)) if native else {})))
        payload = dict(version=1, session_id=spec["session_id"], output_mode="animation",
                       start_frame=frames.start_frame, end_frame=frames.end_frame, fps=frames.fps,
                       take="DrapeAnimation", garments=returned)
        water = write_water_plane(spec, written, temporary)
        if water is not None:
            payload['water_plane'] = water
        ground = write_ground_plane(spec, written, temporary)
        if ground is not None:
            payload['ground_plane'] = ground
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


def write_static_result(spec, vertices, ranges):
    points = np.asarray(vertices)
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
            counts = TemplateCounts()
            with (
                open(entry["path"], encoding="utf-8-sig") as source,
                temporary.open("w", encoding="utf-8", newline="\n") as output,
            ):
                for line in source:
                    fields = line.split()
                    if fields and fields[0] == "v":
                        if index >= end:
                            raise ValueError("The captured garment contains more vertices than expected.")
                        x, y, z = points[index]
                        output.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
                        index += 1
                    else:
                        counts.line(fields)
                        output.write(line)
            if index != end:
                raise ValueError("The captured garment contains fewer vertices than expected.")
            os.replace(temporary, target)
            written.append(target)
            # DAZ verifies counts plus evenly spaced samples instead of re-reading the OBJ; topology,
            # UVs and surfaces are unchanged by construction (every non-vertex line is copied verbatim).
            check_indices, check_vertices = check_samples(points[start:end])
            returned.append(dict(id=entry["id"], path=str(target), vertex_count=expected,
                                 check_indices=check_indices, check_vertices=check_vertices,
                                 facet_count=counts.facets, group_count=len(counts.groups)))
        payload = dict(
            version=1, session_id=spec["session_id"], output_mode="static",
            frame=spec["return_frame"], garments=returned,
        )
        water = write_water_plane(spec, written, temporary_paths)
        if water is not None:
            payload['water_plane'] = water
        ground = write_ground_plane(spec, written, temporary_paths)
        if ground is not None:
            payload['ground_plane'] = ground
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
