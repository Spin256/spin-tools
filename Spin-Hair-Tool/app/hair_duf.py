# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""DAZ-native animation return for Hair (Spin256 · spin256.com/tools).

This writes the
returned hair as a DAZ scene file (gzip-compressed DSON): the captured mesh with its UVs and material groups at the
first frame, and one morph per later frame holding only the vertices that move. The receive script merges it into
the open scene, keys each morph on its own frame, copies the original's materials and verifies samples."""
import gzip
import json
import uuid

import numpy as np

from drape_fbx import read_obj

THRESHOLD = 1e-5            # cm: vertices that move less are left out of a frame's morph (the check allows 0.001)


def _rows(values, fmt):
    return ",".join(fmt % tuple(row) for row in values)


def _uv_set(vertex_count, uvs, faces):
    """DSON UVs: one entry per vertex (its first corner's UV), extra entries where a corner differs (seams)."""
    uvs = np.asarray(uvs, dtype=np.float64)
    first = np.full(vertex_count, -1, dtype=np.int64)
    for vertices, corners, _ in faces:
        for v, uv in zip(vertices, corners):
            if first[v] < 0 and uv >= 0:
                first[v] = uv
    values = np.zeros((vertex_count, 2))
    values[first >= 0] = uvs[first[first >= 0]]
    extra, exceptions = {}, []
    for polygon, (vertices, corners, _) in enumerate(faces):
        for v, uv in zip(vertices, corners):
            if uv >= 0 and uv != first[v]:
                index = extra.setdefault(int(uv), vertex_count + len(extra))
                exceptions.append((polygon, v, index))
    if extra:
        values = np.concatenate([values, uvs[list(extra)]])
    return values, exceptions


def write_duf(path, entry, frames, vertex_range):
    """Write the animation of one returned hair (frames: AnimationFrames, positions in cm) as a DUF scene."""
    start, end = vertex_range
    source, uvs, faces, materials = read_obj(entry["path"])
    base = frames.get(0)[start:end].astype(np.float64)
    if source.shape != base.shape or len(base) != entry["vertex_count"] or not faces:
        raise ValueError("The animation does not match its captured hair.")
    if frames.seen != set(range(frames.count)):
        raise ValueError("The hair animation is incomplete.")
    # Every return gets its own IDs: DAZ refuses a file whose IDs match ones already loaded ("duplicate IDs"), e.g.
    # a second return merged into the same scene. Morph names stay "frame_<n>" so the receive script can key them.
    tag = uuid.uuid4().hex[:12]
    name, geometry, uv_set = "spin_hair_" + tag, "geometry_" + tag, "uvs_" + tag
    parts = ['{"file_version":"0.6.0.0","asset_info":{"id":"/spin_hair_return_%s.duf","type":"scene",' % tag +
             '"contributor":{"author":"Spin256","website":"https://www.spin256.com/tools"},"revision":"1.0"},',
             '"geometry_library":[{"id":"%s","name":"%s","type":"polygon_mesh",' % (geometry, geometry),
             '"vertices":{"count":%d,"values":[%s]},' % (len(base), _rows(base, "[%.6f,%.6f,%.6f]")),
             '"polygon_groups":{"count":1,"values":["hair"]},',
             '"polygon_material_groups":{"count":%d,"values":%s},' % (len(materials), json.dumps(materials)),
             '"polylist":{"count":%d,"values":[%s]}' % (len(faces), ",".join(
                 "[0,%d,%s]" % (material, ",".join(map(str, vertices))) for vertices, _, material in faces))]
    has_uv = bool(uvs) and all(min(corners) >= 0 for _, corners, _ in faces)
    if has_uv:
        parts.append(',"default_uv_set":"#%s"' % uv_set)
    parts.append('}],')
    if has_uv:
        values, exceptions = _uv_set(len(base), uvs, faces)
        parts.append('"uv_set_library":[{"id":"%s","name":"default","label":"Default UVs","vertex_count":%d,'
                     '"uvs":{"count":%d,"values":[%s]},"polygon_vertex_indices":[%s]}],'
                     % (uv_set, len(base), len(values), _rows(values, "[%.7f,%.7f]"), _rows(exceptions, "[%d,%d,%d]")))
    parts.append('"node_library":[{"id":"%s","name":"%s","type":"node","label":"Simulated hair"}],' % (name, name))
    modifiers = []
    for sample in range(1, frames.count):
        delta = frames.get(sample)[start:end].astype(np.float64) - base
        moving = np.flatnonzero(np.abs(delta).max(axis=1) > THRESHOLD)
        rows = np.column_stack([moving, delta[moving]])
        label = "frame_%d" % (frames.start_frame + sample)
        modifiers.append('{"id":"%s_%s","name":"%s","parent":"#%s","channel":{"id":"value","type":"float",'
                         '"name":"%s","value":0,"min":0,"max":1,"clamped":true},'
                         '"morph":{"vertex_count":%d,"deltas":{"count":%d,"values":[%s]}}}'
                         % (label, tag, label, geometry, label, len(base), len(moving),
                            _rows(rows, "[%d,%.6f,%.6f,%.6f]")))
    parts.append('"modifier_library":[' + ",".join(modifiers) + '],')
    # Scene instances get IDs of their own ("-1"): reusing the library IDs makes DAZ warn "Duplicate id found in file"
    # for the geometry and every morph. Morph instances belong to the prop's node, as in DAZ's own files.
    node = name + "-1"
    scene = dict(nodes=[{"id": node, "url": "#" + name, "label": "Simulated hair",
                         "geometries": [{"id": geometry + "-1", "url": "#" + geometry}]}],
                 modifiers=[{"id": "frame_%d_%s-1" % (frames.start_frame + k, tag),
                             "url": "#frame_%d_%s" % (frames.start_frame + k, tag), "parent": "#" + node}
                            for k in range(1, frames.count)])
    parts.append('"scene":%s}' % json.dumps(scene))
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as output:
        for part in parts:
            output.write(part)
