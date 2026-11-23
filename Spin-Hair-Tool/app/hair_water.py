# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Captured water planes for the hair tool (same rule as the Drape Tool's water_scene).

A collision object whose label starts with "Water" and that is a flat horizontal plane is
the water surface, not a solid: it is removed from the colliders and turns water on at its
height. Without one, the Water panel can return a plain water-height plane to DAZ.
"""
import numpy as np


def split_water(inputs):
    """Inputs without captured water planes, and the detected water source (or None)."""
    solids, found = [], []
    for collider in inputs["colliders"]:
        points = np.asarray(collider["vertices"], float)
        named = str(collider.get("label", "")).strip().casefold().startswith("water")
        extent = np.ptp(points, axis=0) if len(points) else np.zeros(3)
        if named and extent[1] <= 0.1 and extent[0] > 1e-4 and extent[2] > 1e-4:
            found.append((collider.get("label", "Water"), float(points[:, 1].mean())))
        else:
            solids.append(collider)
    if not found:
        return inputs, None
    heights = [height for _, height in found]
    if max(heights) - min(heights) > 0.1:
        raise ValueError("Selected water planes have different heights. Capture one water level at a time.")
    if not solids:
        raise ValueError("Assign the avatar as a collision object (only a water plane was captured).")
    result = dict(inputs, colliders=solids)
    motion = inputs.get("motion")
    if motion is not None:
        # Keep the timeline's moving colliders pointing at the same meshes.
        kept = [i for i, c in enumerate(inputs["colliders"]) if any(c is s for s in solids)]
        from hair_animation import AvatarMotion
        result["motion"] = AvatarMotion(solids, {kept.index(i): paths for i, paths in motion.samples.items() if i in kept},
                                        motion.fps, motion.start_frame, motion.end_frame)
    return result, dict(labels=[label for label, _ in found], height=heights[0])
