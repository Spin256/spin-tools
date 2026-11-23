# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Disk-backed DAZ-frame samples; internal settling overwrites the current sample."""
import tempfile
import numpy as np
from project_settings import session_directory


class AnimationFrames:
    def __init__(self, motion, vertices):
        self.start_frame, self.end_frame, self.fps = motion.start_frame, motion.end_frame, motion.fps
        self.count = self.end_frame - self.start_frame + 1
        self.vertex_count = len(vertices)
        self.stride = self.vertex_count * 3 * 4
        self.file = tempfile.TemporaryFile(dir=session_directory(), prefix=".drape-frames-")
        self.seen = set()
        self.put(0, vertices)

    def put(self, sample, vertices):
        points = np.asarray(vertices, dtype="<f4")
        if not 0 <= sample < self.count or points.shape != (self.vertex_count, 3) or not np.isfinite(points).all():
            raise ValueError("Invalid animation frame geometry.")
        self.file.seek(sample * self.stride)
        self.file.write(points.tobytes())
        self.seen.add(sample)

    def get(self, sample):
        if sample not in self.seen:
            raise ValueError("The garment animation is incomplete. Run the drape again.")
        self.file.seek(sample * self.stride)
        data = self.file.read(self.stride)
        if len(data) != self.stride:
            raise ValueError("Could not read the recorded garment frame.")
        return np.frombuffer(data, dtype="<f4").reshape(-1, 3)

    def close(self):
        self.file.close()
