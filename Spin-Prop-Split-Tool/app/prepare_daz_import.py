# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Prepare a grouped OBJ and verification manifest from a selector export."""

import argparse
import json
import math
from pathlib import Path
import re

from mesh import load_obj


def prepare(selection_path):
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    if selection.get("version") != 1 or selection.get("face_index_base") != 0:
        raise ValueError("Expected a version 1 selector export with zero-based faces.")
    if not selection.get("source_obj"):
        raise ValueError("Open an OBJ in the selector before exporting.")
    source = Path(selection["source_obj"])
    mesh = load_obj(source)
    if any(len(face) not in (3, 4) for face in mesh.faces):
        raise ValueError("DAZ round-trip import supports triangles and quads.")
    if len({tuple(sorted(face)) for face in mesh.faces}) != len(mesh.faces):
        raise ValueError("Duplicate polygons make group verification ambiguous.")
    pivot = selection.get("pivot")
    if pivot is not None and (not isinstance(pivot, list) or len(pivot) != 3 or
            any(type(value) not in (int, float) or not math.isfinite(value) for value in pivot)):
        raise ValueError("Expected three finite pivot coordinates.")
    groups = selection.get("groups") or {}
    if not groups and selection.get("selected_faces"):
        groups = {"Selection": selection["selected_faces"]}
    if not groups and pivot is None:
        raise ValueError("Export at least one selected polygon or a pivot.")
    names = set()
    export_groups = {}
    membership = {}
    for label, indices in groups.items():
        name = "DPT_" + re.sub(r"[^A-Za-z0-9_]", "_", label)
        if name.lower() in names or name.lower() == "dpt_ungrouped" or not isinstance(indices, list) or not indices:
            raise ValueError(f"Invalid or conflicting group name: {label}")
        names.add(name.lower())
        export_groups[name] = indices
        for index in indices:
            if type(index) is not int or not 0 <= index < len(mesh.faces):
                raise ValueError(f"Invalid face index in {label}: {index}")
            if index in membership:
                raise ValueError("Each face can belong to only one DAZ face group.")
            membership[index] = name

    obj = selection_path.with_name(selection_path.stem + "-daz.obj")
    manifest = obj.with_suffix(".json")
    if obj.exists() or manifest.exists():
        raise ValueError("An import file already exists. Use a new selector export filename.")
    lines = []
    current_group = None
    index = 0
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if fields and fields[0] == "g" and membership:
            continue
        if fields and fields[0] == "f":
            if membership:
                target = membership.get(index, "DPT_Ungrouped")
                if target != current_group:
                    lines.append("g " + target)
                    current_group = target
            index += 1
        lines.append(line)
    # The DAZ script uses OBJ coordinates directly and verifies every vertex.
    obj.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        manifest.write_text(json.dumps({
            "version": 1, "obj": obj.name, "vertices": mesh.vertices, "faces": mesh.faces,
            "groups": export_groups,
            "pivot": pivot,
        }, indent=2) + "\n", encoding="utf-8")
    except OSError:
        obj.unlink()
        raise
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    args = parser.parse_args()
    try:
        print(prepare(args.selection.resolve()))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")

