# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Soft target positions keyed to the source timeline."""
from collections import defaultdict
import heapq
import math

import numpy as np


def pull_weights(points, neighbours, seeds, soft, radius):
    """Fade a pull by distance along the soft surface from the selected vertices."""
    if radius <= 0:
        return seeds, np.ones(len(seeds))

    distance = np.full(len(points), np.inf)
    distance[seeds] = 0.
    pending = [(0., int(index)) for index in seeds]
    heapq.heapify(pending)
    while pending:
        length, index = heapq.heappop(pending)
        if length != distance[index]:
            continue
        for neighbour in neighbours[index]:
            if not soft[neighbour]:
                continue
            next_length = length + math.dist(points[index], points[neighbour])
            if next_length < radius and next_length < distance[neighbour]:
                distance[neighbour] = next_length
                heapq.heappush(pending, (next_length, neighbour))

    indices = np.flatnonzero(np.isfinite(distance))
    weight = 1. - distance[indices] / radius
    return indices, weight * weight * (3. - 2. * weight)


class FrameGuides:
    def __init__(self, keys, motion, mobility, lead_frames, release_frames, strength):
        vertex_keys = defaultdict(list)
        for sample, targets in sorted(keys.items()):
            indices = np.asarray(list(targets), dtype=np.intp)
            positions = np.asarray(list(targets.values()), dtype=np.float32)
            if (not 0 <= sample <= motion.intervals or positions.shape != (len(indices), 3)
                    or np.any(indices < 0) or np.any(indices >= len(mobility))
                    or not np.isfinite(positions).all()):
                raise ValueError('A frame guide does not match the source animation.')
            offsets = positions - motion.body_positions(sample)[indices]
            for index, offset in zip(indices, offsets):
                if mobility[index] > 0:
                    vertex_keys[index].append((sample, offset))

        # Regions painted on different frames have independent tracks.
        groups = defaultdict(list)
        for index, keyframes in vertex_keys.items():
            times, offsets = zip(*keyframes)
            groups[times].append((index, offsets))
        self.tracks = []
        for times, entries in groups.items():
            indices, offsets = zip(*entries)
            indices = np.asarray(indices, dtype=np.intp)
            targets = np.asarray(offsets, dtype=np.float32)
            slopes = np.zeros_like(targets)
            rates = np.diff(targets, axis=1) / np.diff(times)[None, :, None]
            before, after = rates[:, :-1], rates[:, 1:]
            # Use the gentler slope where motion continues, and ease through turns.
            slopes[:, 1:-1] = np.where(before * after > 0,
                                        np.sign(before) * np.minimum(abs(before), abs(after)), 0.)
            self.tracks.append((np.asarray(times), indices, targets, slopes, mobility[indices, None]))
        self.lead_frames = lead_frames
        self.release_frames = release_frames
        self.shape = (len(mobility), 3)
        self.strength = strength

    def offset_at(self, sample):
        offset = np.zeros(self.shape, dtype=np.float32)
        for times, indices, targets, slopes, mobility in self.tracks:
            weight = 1.
            if sample < times[0] or sample > times[-1]:
                before = sample < times[0]
                distance = times[0] - sample if before else sample - times[-1]
                duration = min(self.lead_frames, times[0]) if before else self.release_frames
                if distance >= duration:
                    continue
                weight = 1. - distance / duration
                weight = weight * weight * (3. - 2. * weight)
                target = targets[:, 0 if before else -1]
            else:
                right = int(np.searchsorted(times, sample))
                if right == 0:
                    target = targets[:, 0]
                else:
                    left = right - 1
                    interval = times[right] - times[left]
                    alpha = (sample - times[left]) / interval
                    ease = alpha * alpha * (3. - 2. * alpha)
                    target = targets[:, left] * (1. - ease) + targets[:, right] * ease
                    target += interval * alpha * (1. - alpha) * (
                        (1. - alpha) * slopes[:, left] - alpha * slopes[:, right])
            offset[indices] = target * (self.strength * weight) * mobility
        return offset
