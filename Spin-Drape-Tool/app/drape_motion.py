# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Stream topology-checked moving collider samples without loading an entire timeline."""

import math

import numpy as np

from warp_drape import read_mesh


class ColliderMotion:
    def __init__(self, colliders, samples, fps, start_frame, end_frame):
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
        self.raw_faces = {}
        for index, paths in samples.items():
            if len(paths) != self.intervals + 1:
                raise ValueError("A moving collider is missing captured frames.")
            self.cache[(index, 0)] = colliders[index][0]

    def positions(self, index, frame):
        key = (index, frame)
        if key not in self.cache:
            # Compare unfiltered triangle lists. The collider's degenerate-face
            # filter depends on positions, so a triangle collapsing at one
            # frame must not read as a topology change.
            points, faces = read_mesh(self.samples[index][frame], keep_degenerate=True)
            original_points = self.colliders[index][0]
            if index not in self.raw_faces:
                self.raw_faces[index] = read_mesh(self.samples[index][0], keep_degenerate=True)[1]
            if points.shape != original_points.shape or not np.array_equal(faces, self.raw_faces[index]):
                raise ValueError(
                    f"Moving collider {index + 1} changed topology at frame {self.start_frame + frame}. "
                    "Keep its mesh resolution and geometry consistent throughout the timeline."
                )
            if not np.isfinite(points).all():
                raise ValueError("A moving collider contains invalid vertex positions.")
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
