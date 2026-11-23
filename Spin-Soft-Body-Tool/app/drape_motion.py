# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Stream topology-checked moving collider samples without loading an entire timeline."""

import math

import numpy as np

from mesh_io import read_mesh


class ColliderMotion:
    def __init__(self, colliders, samples, fps, start_frame, end_frame, *, role="Moving collider", labels=None):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("The captured timeline has an invalid frame rate.")
        if not isinstance(start_frame, int) or not isinstance(end_frame, int) or end_frame <= start_frame:
            raise ValueError("The captured timeline has an invalid frame range.")
        self.fps = fps
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.intervals = end_frame - start_frame
        self.duration = self.intervals / fps
        self.colliders = colliders
        self.samples = samples
        self.cache = {}
        self.role=role;self.labels=labels or [];self.topology={}
        for index, paths in samples.items():
            if len(paths) != self.intervals + 1:
                raise ValueError("A moving collider is missing captured frames.")
            self.cache[(index, 0)] = colliders[index][0]
            # Compare source connectivity, not a pose-dependent usable-triangle
            # subset. The simulation keeps its original collision-face subset.
            first,faces=read_mesh(paths[0],preserve_topology=True)
            if first.shape!=colliders[index][0].shape:raise ValueError('The first animation sample does not match its source vertex count.')
            a,b,c=first[faces].transpose(1,0,2)
            usable=faces[np.linalg.norm(np.cross(b-a,c-a),axis=1)>1e-12]
            if not np.array_equal(usable,colliders[index][1]):raise ValueError('The first animation sample does not match its source face connectivity.')
            self.topology[index]=faces

    def positions(self, index, frame):
        key = (index, frame)
        if key not in self.cache:
            points, faces = read_mesh(self.samples[index][frame],preserve_topology=True)
            original_points = self.colliders[index][0]
            label=f"{self.role} {index+1}"+(f" ({self.labels[index]})" if index<len(self.labels) else "")
            if points.shape != original_points.shape:
                raise ValueError(
                    f"{label} changed vertex count at frame {self.start_frame + frame} "
                    f"({len(original_points)} to {len(points)}). Keep its mesh resolution consistent throughout the timeline."
                )
            if not np.array_equal(faces,self.topology[index]):
                raise ValueError(f"{label} changed face connectivity at frame {self.start_frame+frame}. Keep its geometry consistent throughout the timeline.")
            if not np.isfinite(points).all():
                raise ValueError(f"{label} contains invalid vertex positions.")
            # Retain just the current pair of samples for each collider.
            self.cache = {
                existing: value for existing, value in self.cache.items()
                if existing[0] != index or existing[1] in (frame - 1, frame)
            }
            self.cache[key] = points
        return self.cache[key]

    def pair(self, index, interval):
        first = min(interval, self.intervals)
        second = min(interval + 1, self.intervals)
        return self.positions(index, first), self.positions(index, second)

    def preview(self, frame):
        sample = min(frame, self.intervals)
        return {index: self.positions(index, sample) * 100 for index in self.samples}

    def reset(self):
        self.cache.clear()
        for index in self.samples:
            self.cache[(index, 0)] = self.colliders[index][0]
