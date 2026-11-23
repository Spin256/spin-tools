# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Optional end-of-timeline correction toward the simulated first frame."""
import numpy as np


def loop_tail(first, second, tail, soft):
    """Blend position and velocity corrections into the existing trajectory."""
    target = first[soft]
    first_step = second[soft] - target
    end = tail[-1]['vertices'][soft]
    end_step = end - tail[-2]['vertices'][soft]
    position = target - end
    velocity = first_step - end_step
    span = len(tail)
    for step, frame in enumerate(tail, 1):
        points = frame['vertices'].copy()
        if step == span:
            points[soft] = target
        else:
            t = step / span
            points[soft] += t*t*(3. - 2.*t)*position + span*t*t*(t - 1.)*velocity
        yield points


def painted_parts(soft, faces):
    """Connected painted areas, so the loop closes each one separately."""
    label = np.where(soft, np.arange(len(soft)), -1)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = edges[soft[edges].all(axis=1)]
    while len(edges):
        low = np.minimum(label[edges[:, 0]], label[edges[:, 1]])
        changed = label.copy()
        np.minimum.at(changed, edges[:, 0], low)
        np.minimum.at(changed, edges[:, 1], low)
        changed[soft] = changed[changed[soft]]                 # jump along already merged labels
        if np.array_equal(changed, label):
            break
        label = changed
    return [label == value for value in np.unique(label[soft])]
