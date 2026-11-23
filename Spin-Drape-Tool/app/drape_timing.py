# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Shared phase timing for the solver and its live preview."""

import math
from typing import NamedTuple


class DrapeStep(NamedTuple):
    interval: int
    hold: bool
    phase: str
    sample: int
    part: int = 0            # fast timeline frames are split into ``parts`` smaller moves
    parts: int = 1


def motion_parts(motion, max_move_cm):
    """Moves per timeline interval so the avatar travels at most ``max_move_cm`` per simulated
    frame (0 or None: one move per frame, as DAZ gives them). Cached on the motion."""
    if motion is None or not max_move_cm:
        return None
    cache = getattr(motion, "_spin_parts", {})
    rigid = getattr(motion, "rigid", None)
    key = (max_move_cm, rigid is not None)
    if key not in cache:
        import numpy as np
        largest = np.zeros(motion.intervals)
        for index in motion.samples:
            previous = motion.positions(index, 0)
            for interval in range(motion.intervals):
                if rigid is not None:
                    # Limbs swing in arcs (rigid_motion): measure the path, not the straight line.
                    largest[interval] = max(largest[interval], rigid.length(index, interval) * 100.)
                    continue
                current = motion.positions(index, interval + 1)
                largest[interval] = max(largest[interval],
                                        float(np.linalg.norm(current - previous, axis=1).max()) * 100.)
                previous = current
        cache[key] = [max(1, math.ceil(move / max_move_cm - 1e-9)) for move in largest]
        try:
            motion._spin_parts = cache
        except AttributeError:
            pass
    return cache[key]


class DrapeSchedule:
    def __init__(self, motion, duration, initial_duration=0.0, frame_settle=0.0, max_move_cm=0.0):
        if not all(math.isfinite(value) and value >= 0 for value in (duration, initial_duration, frame_settle)):
            raise ValueError("Drape and settling times must be finite, nonnegative numbers.")
        if motion is None and duration == 0:
            raise ValueError("Static drape duration must be greater than zero.")
        self.motion = motion
        self.fps = motion.fps if motion is not None else 60.0
        self.initial_frames = math.ceil(initial_duration * self.fps) if motion is not None else 0
        self.hold_frames = math.ceil(frame_settle * self.fps) if motion is not None else 0
        self.final_frames = math.ceil(duration * self.fps)
        intervals = motion.intervals if motion is not None else 0
        self.parts = motion_parts(motion, max_move_cm) or [1] * intervals
        self.frame_count = self.initial_frames + sum(self.parts) + intervals * self.hold_frames + self.final_frames
        self.duration = self.frame_count / self.fps

    def __iter__(self):
        if self.motion is None:
            for _ in range(self.final_frames):
                yield DrapeStep(0, True, "Draping", 0)
            return
        for _ in range(self.initial_frames):
            yield DrapeStep(0, True, "Initial drape", 0)
        for interval in range(self.motion.intervals):
            parts = self.parts[interval]
            for part in range(parts):
                yield DrapeStep(interval, False, "Timeline drape", interval + 1, part, parts)
            for _ in range(self.hold_frames):
                yield DrapeStep(interval + 1, True, "Settling at frame", interval + 1)
        for _ in range(self.final_frames):
            yield DrapeStep(self.motion.intervals, True, "Final settling", self.motion.intervals)
