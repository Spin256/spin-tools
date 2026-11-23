# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Thread-safe live controls and frame scheduling for static-result draping."""
import threading
import time
import numpy as np
from drape_timing import DrapeStep


class LiveDrape:
    def __init__(self):
        self.lock = threading.Lock()
        self.holding = False
        self.paused = False
        self.paused_ack = False
        self.at_end = False
        self.finished = False
        self.grab = None
        self.revision = 0
        self.frozen = None
        self.freeze_revision = 0

    def set_frozen(self, indices, positions):
        with self.lock:
            self.frozen = (np.asarray(indices, dtype=np.int32).copy(), np.asarray(positions, dtype=np.float32).copy())
            self.freeze_revision += 1

    def freeze_snapshot(self):
        with self.lock:
            return self.freeze_revision, self.frozen

    def set_hold(self, value):
        with self.lock:
            self.holding = bool(value)

    def set_pause(self, value):
        with self.lock:
            self.paused = bool(value)
            if not value:
                self.paused_ack = False

    def finish(self):
        with self.lock:
            self.finished = True
            self.grab = None
            self.revision += 1

    def set_grab(self, indices=None, targets=None, weights=None, strength=1.):
        with self.lock:
            self.grab = None if indices is None else (
                np.asarray(indices, dtype=np.int32).copy(),
                np.asarray(targets, dtype=np.float32).copy(),
                np.asarray(weights, dtype=np.float32).copy(), float(strength))
            self.revision += 1

    def grab_snapshot(self):
        with self.lock:
            return self.revision, self.grab

    def steps(self, schedule, cancelled=lambda: False, on_pause=None):
        """Extra held steps never consume a source step or change the source pose."""
        source = iter(schedule)
        previous = DrapeStep(0, True, "Holding frame", 0)
        exhausted = False
        deadline = time.monotonic()
        while not cancelled():
            with self.lock:
                done, paused, holding = self.finished, self.paused, self.holding
            if done:
                return
            if paused:
                with self.lock:
                    if not self.paused:
                        continue
                    notify = not self.paused_ack
                    self.paused_ack = True
                if notify and on_pause is not None:
                    on_pause()
                time.sleep(.01)
                continue
            if time.monotonic() < deadline:
                time.sleep(.01)
                continue
            hold_source = holding and schedule.motion is not None and previous.sample < schedule.motion.intervals
            if hold_source or exhausted:
                step = DrapeStep(previous.sample, True,
                                 "Final frame — settling" if exhausted else "Holding frame", previous.sample)
            else:
                step = next(source, None)
                if step is None:
                    exhausted = True
                    with self.lock:
                        self.at_end = True
                        self.paused = True
                    continue
            yield step
            previous = step
            # No catch-up burst after a slow GPU frame, pause or long mouse drag.
            deadline = time.monotonic() + 1. / schedule.fps


def grab_patch(points, faces, seed, radius, allowed):
    """A local connected patch, excluding a nearby but separate fabric layer."""
    distances = np.linalg.norm(points - points[seed], axis=1)
    inside = (distances <= max(radius, 1e-8)) & allowed
    if not inside[seed]:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    local = faces[np.all(inside[faces], axis=1)]
    adjacency = {}
    for triangle in local:
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            adjacency.setdefault(int(a), set()).add(int(b))
            adjacency.setdefault(int(b), set()).add(int(a))
    selected, pending = {int(seed)}, [int(seed)]
    while pending:
        for neighbor in adjacency.get(pending.pop(), ()):
            if neighbor not in selected:
                selected.add(neighbor)
                pending.append(neighbor)
    indices = np.array(sorted(selected), dtype=np.int32)
    weights = np.maximum(.05, 1 - distances[indices] / max(radius, 1e-8)).astype(np.float32)
    return indices, weights
