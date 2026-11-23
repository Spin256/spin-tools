# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Combine streamed prescribed body poses and collider movement on one timeline."""
import numpy as np


class SceneMotion:
    def __init__(self,bodies,body_motion=None,collider_motion=None):
        self.bodies=bodies;self.body_motion=body_motion;self.collider_motion=collider_motion
        timelines=[m for m in (body_motion,collider_motion) if m is not None]
        if not timelines:raise ValueError('Animation needs body or collider movement.')
        reference=timelines[0]
        self.fps=reference.fps;self.start_frame=reference.start_frame;self.end_frame=reference.end_frame
        self.intervals=self.end_frame-self.start_frame;self.duration=self.intervals/self.fps
        for motion in timelines:
            if (motion.start_frame,motion.end_frame,motion.fps)!=(self.start_frame,self.end_frame,self.fps):
                raise ValueError('Animated bodies and colliders must have matching frame counts and frame rates.')
        self.samples=collider_motion.samples if collider_motion else {}
    def body_positions(self,sample):
        return np.concatenate([self.body_motion.positions(i,sample) if self.body_motion and i in self.body_motion.samples else mesh[0]
                               for i,mesh in enumerate(self.bodies)])
    def positions(self,index,sample):return self.collider_motion.positions(index,sample)
    def pair(self,index,interval):return self.collider_motion.pair(index,interval)
    def preview(self,sample):return self.collider_motion.preview(sample) if self.collider_motion else {}
    def reset(self):
        for motion in (self.body_motion,self.collider_motion):
            if motion:motion.reset()
