# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Ground for the Drape Tool: recognise a captured "Ground..." plane, find the avatar's lowest point
and write an optional ground-height plane for DAZ, like the water plane.

Transfer file: write_ground_plane is called by drape_return.py; its OBJ (one quad, cm, Y up) is
imported by Import Static Drape.dsa as "Ground - Drape Height".
"""
import os
from pathlib import Path

import numpy as np

MODES = (("off", "Off"), ("floor", "DAZ floor (height 0)"), ("lowest", "Lowest point of the avatar"),
         ("plane", 'Plane named "Ground…"'), ("custom", "Custom height"))


def ground_planes(colliders, labels):
    """Heights (cm) of captured flat props whose label starts with "Ground"."""
    heights = []
    for (points, _), label in zip(colliders, labels):
        if not str(label).strip().casefold().startswith("ground"):
            continue
        points = np.asarray(points)
        extent = np.ptp(points, axis=0)
        if extent[1] <= .001 and extent[0] > 1e-6 and extent[2] > 1e-6:
            heights.append(float(points[:, 1].mean() * 100))
    return heights


def lowest_point(colliders, roles, labels, motion=None):
    """Lowest Y (cm) of the avatar (moving colliders, or every collider when none moves) over all
    timeline frames; Ground/Water planes are ignored."""
    chosen = [i for i, role in enumerate(roles) if role == "moving"] or list(range(len(colliders)))
    chosen = [i for i in chosen if not str(labels[i]).strip().casefold().startswith(("ground", "water"))]
    if not chosen:
        return None
    lowest = min(float(np.asarray(colliders[i][0])[:, 1].min()) for i in chosen)
    if motion is not None:
        for i in chosen:
            if i in motion.samples:
                for frame in range(motion.intervals + 1):
                    lowest = min(lowest, float(motion.positions(i, frame)[:, 1].min()))
    return lowest * 100


def ground_return_path(spec):
    return spec.get("ground_return_path") or (spec["session_id"] + "-ground.obj")


def write_ground_plane(spec, written, temporary):
    settings = spec.get("ground_plane")
    if settings is None:
        return None
    height, x, z, extent = (float(settings[key]) for key in ("height", "x", "z", "extent"))
    if not np.isfinite([height, x, z, extent]).all() or extent <= 0:
        raise ValueError("Invalid ground plane geometry.")
    target = Path(ground_return_path(spec))
    temp = target.with_name(target.name + ".tmp")
    temporary.append(temp)
    points = [(x - extent, height, z - extent), (x + extent, height, z - extent),
              (x + extent, height, z + extent), (x - extent, height, z + extent)]
    with temp.open("w", encoding="utf-8", newline="\n") as output:
        output.write("o Ground\n")
        for point in points:
            output.write("v %.9g %.9g %.9g\n" % point)
        output.write("vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl ground_reference\nf 1/1 4/4 3/3 2/2\n")
    os.replace(temp, target)
    written.append(target)
    return dict(path=str(target), height=height, vertex_count=4)
