# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Seamless animation loops for the Spin Tools simulators.

A simulation starts from no movement on its first frame, so a plain loop restarts the motion at the join.
Instead, after the last frame the simulation keeps going through frames 1, 2, ... on the same source poses (a
second pass that starts from the true end state) until it meets the first run. If the two have not met by the
middle of the timeline, the closest frame is used: the second pass is run again and, over the N frames before
that frame, steered onto the first run (the simulation continues from each steered state), so it lands there.
Frame 0 is simulated too: after the last frame the source animation takes one real step into frame 0 (the last
frame leads into frame 0). A last frame that repeats frame 0 just becomes a one-frame hold,
as the animation itself has. Frames after the meeting frame are the first run unchanged.

Each tool supplies its own solver hooks; this module only decides which frames to run and how to blend them.
Soft Body has its own version of this logic inside body_solver (with painted-area handling)."""
import math

import numpy as np

CATCH_UP = 2.e-4          # metres: the second pass has caught up with the first run (0.2 mm)


class LoopRecorder:
    """Keeps the first run's state for frames 0..middle, which the second pass is compared with."""

    def __init__(self, intervals):
        self.middle = max(1, intervals // 2)
        self.states = {}

    def record(self, sample, state):
        if sample <= self.middle:
            self.states[sample] = np.array(state, dtype=np.float32, copy=True)


def carry_loop(*, recorder, steer_frames, advance, wrap, get_state, set_state, snapshot, restore, publish,
               preview=None, wind_back=None, tolerance=CATCH_UP):
    """Run the second pass and publish frames 0..meeting frame. Returns the status text, or None if canceled.

    advance(sample) -> payload or None (canceled): steps the solver through source frame ``sample``.
    get_state() / set_state(array): the solver's positions (metres) that are compared and steered.
    snapshot() / restore(saved): everything the next frame depends on.
    publish(payload): stores a final frame. preview(payload): shows a frame without storing it (optional).
    wrap() -> payload or None (canceled): one step from the last frame into frame 0; its payload is frame 0.
    wind_back(): optional; moves any per-frame clocks back one cycle before each second-pass run."""
    end_state = snapshot()
    if wind_back is not None:
        wind_back()
    if wrap() is None:
        return None
    meet, closest_gap, met = 1, math.inf, False
    for sample in range(1, recorder.middle + 1):
        payload = advance(sample)
        if payload is None:
            return None
        gap = float(np.abs(get_state() - recorder.states[sample]).max(initial=0.))
        if preview is not None:
            preview(dict(payload, phase='Carrying motion into the loop…'))
        if gap < closest_gap:
            meet, closest_gap = sample, gap
        if gap <= tolerance:
            met = True
            break
    steer = 0 if met else min(steer_frames, meet)
    restore(end_state)
    if wind_back is not None:
        wind_back()
    frame0 = wrap()
    if frame0 is None:
        return None
    published = [dict(frame0, sample=0)]
    for sample in range(1, meet + 1):
        payload = advance(sample)
        if payload is None:
            return None
        if sample > meet - steer:
            t = (sample - (meet - steer)) / steer
            weight = t * t * (3. - 2. * t)
            payload = set_state(get_state() * (1. - weight) + recorder.states[sample] * weight, payload)
        if preview is not None:
            preview(dict(payload, phase='Landing the loop…'))
        published.append(payload)
    note = (f'caught up on frame {meet}' if met else
            f'steered onto frame {meet} over {steer} frames, {closest_gap * 1000.:.1f} mm apart there')
    for payload in published:
        publish(payload)
    return f'Loop closed with motion carried ({note}).'
