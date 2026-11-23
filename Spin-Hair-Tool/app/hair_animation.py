# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Timeline captures for the hair tool (as the Drape Tool): the avatar moves, hair follows.

DAZ captures hair at the start frame and both the avatar and collision objects at each
frame of the chosen range. ``load_any`` reads either kind of capture;
hair_session.load_session reads the start frame and this module adds the
avatar frames (``AvatarMotion``), the return preference and the final frame to return.

``HeadMotion`` follows the head: a rigid fit (Kabsch) of the avatar's vertices around the
scalp from the start frame to each frame. Scalp roots, frozen hair, the hair's scalp mesh
and the scalp cap ride with it, so the simulated hair is carried by the head while the
rest of the body moves underneath it.

``AnimationFrames`` stores one hair frame per DAZ frame on disk for an animation return
(the interface write_animation_result and write_fbx expect).
"""
import json
import math
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

from hair_session import load_session
from collider_roles import ROLES


def load_any(session_path):
    """Inputs for a current-frame or timeline capture; timeline adds ``motion`` and
    ``output_mode`` and returns at the end frame."""
    session_path = Path(session_path)
    data = json.loads(session_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("mode") != "timeline":
        inputs = load_session(session_path)
        inputs["output_mode"] = "static"
        return inputs
    # Read the start frame with hair_session's loader (it only accepts current-frame captures).
    still = dict(data, mode="current_frame")
    handle, temporary = tempfile.mkstemp(suffix=".json", prefix=".hair-start-", dir=str(session_path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(still, output)
        inputs = load_session(temporary)
    finally:
        Path(temporary).unlink(missing_ok=True)
    inputs["live_session"] = session_path
    start, end, fps = data.get("start_frame"), data.get("end_frame"), data.get("fps")
    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
        raise ValueError("The captured timeline has an invalid frame range.")
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("The captured timeline has an invalid frame rate.")
    output_mode = data.get("output_mode", "static")
    if output_mode not in ("static", "animation"):
        raise ValueError("The capture has an invalid return preference.")
    colliders = [e for e in data["meshes"] if isinstance(e, dict) and e.get("role") in ROLES]
    samples = {}
    for index, entry in enumerate(colliders):
        if entry["role"] in ("avatar","avatar_attachment","collision","moving"):
            paths = entry.get("samples")
            if not isinstance(paths, list) or len(paths) != end - start + 1:
                raise ValueError(f"{entry.get('label', 'A moving collider')} is missing captured frames.")
            samples[index] = [str(Path(p) if Path(p).is_absolute() else session_path.parent / p) for p in paths]
    inputs["motion"] = AvatarMotion(inputs["colliders"], samples, fps, start, end)
    inputs["output_mode"] = output_mode
    inputs["return_spec"] = dict(inputs["return_spec"], return_frame=end)
    return inputs


def read_vertices(path):
    """Vertex positions only (cm), fast: later timeline samples share the first frame's faces."""
    data = Path(path).read_bytes()
    lines = [line[2:] for line in data.splitlines() if line.startswith(b"v ")]
    points = np.array(b" ".join(lines).split(), dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError(f"The captured frame has invalid coordinates: {path}")
    return points


class AvatarMotion:
    """Moving collider positions per captured frame (cm). Frames are read in the background
    (``prefetch``, started on first use) and kept as float32, so playback does not wait on disk."""

    def __init__(self, colliders, samples, fps, start_frame, end_frame):
        self.colliders, self.samples = colliders, samples
        self.fps, self.start_frame, self.end_frame = float(fps), start_frame, end_frame
        self.intervals = end_frame - start_frame
        self.duration = self.intervals / self.fps
        self.cache, self.error = {}, None
        self.lock = threading.Lock()
        self.thread = None
        self.rigid = None

    def prepare_rigid(self):
        """Use the Drape Tool's local rigid-part paths, retaining its meter units."""
        if self.rigid is None:
            self.rigid = HairRigidPaths(self)

    def prefetch(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._read_all, daemon=True)
            self.thread.start()

    def wait(self):
        """Wait for the background reader (before deleting the capture's files)."""
        if self.thread is not None:
            self.thread.join()

    def _read_all(self):
        try:
            for frame in range(1, self.intervals + 1):
                for index in self.samples:
                    self._load(index, frame)
        except Exception as error:      # reported when that frame is needed
            self.error = error

    def _load(self, index, frame):
        key = (index, frame)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        points = read_vertices(self.samples[index][frame])
        if points.shape != np.asarray(self.colliders[index]["vertices"]).shape:
            raise ValueError(f"{self.colliders[index].get('label', 'A moving collider')} changed its mesh "
                             f"at frame {self.start_frame + frame}. Keep its resolution constant.")
        points = points.astype(np.float32)
        with self.lock:
            self.cache[key] = points
        return points

    def positions(self, index, frame):
        """Collider ``index`` at sample ``frame`` (0 = start frame)."""
        frame = int(min(max(frame, 0), self.intervals))
        if index not in self.samples or frame == 0:
            return np.asarray(self.colliders[index]["vertices"], np.float32)
        self.prefetch()
        return self._load(index, frame)

    def at(self, index, time):
        """Collider ``index`` at ``time`` seconds after the start frame, between samples."""
        position = min(max(time * self.fps, 0.0), float(self.intervals))
        first = int(math.floor(position))
        blend = position - first
        a = self.positions(index, first)
        if blend <= 1e-6 or first >= self.intervals:
            return a
        if self.rigid is not None and index in self.samples:
            return self.rigid.positions(index, first, blend)
        return a * (1 - blend) + self.positions(index, first + 1) * blend


def kabsch(source, target):
    """Rotation R and translation t with target ~ source @ R.T + t."""
    cs, ct = source.mean(0), target.mean(0)
    h = (source - cs).T @ (target - ct)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, ct - cs @ r.T


class HairRigidPaths:
    """Unit adapter (cm / meters) around the rigid-limb interpolation in hair_rigid_motion."""
    def __init__(self, motion):
        from types import SimpleNamespace
        from hair_rigid_motion import RigidPaths
        meters = SimpleNamespace(intervals=motion.intervals,
            positions=lambda index, frame: motion.positions(index, frame).astype(float) * .01)
        faces = {i: np.asarray(motion.colliders[i]['triangles'], int) for i in motion.samples}
        self.paths = RigidPaths(meters, faces)
        from collider_roles import primary_index
        self.primary = primary_index(motion.colliders)
        self.motion = motion
        self.attachments = {}

    def positions(self, index, interval, blend):
        independent = self.paths.positions(index, interval, blend) * 100.
        if self.motion.colliders[index].get('role') != 'avatar_attachment':
            return independent
        key=(index,interval)
        if key not in self.attachments:
            from hair_avatar_attachments import SurfaceAttachment
            self.attachments[key]=SurfaceAttachment(self.motion.positions(index,interval),
                self.motion.positions(index,interval+1),self.motion.positions(self.primary,interval),
                self.motion.positions(self.primary,interval+1),self.motion.colliders[self.primary]['triangles'])
        body=self.paths.positions(self.primary,interval,blend)*100.
        return self.attachments[key].at(body,blend,independent)

    def length(self, index, interval):
        if self.motion.colliders[index].get('role') == 'avatar_attachment':
            total=np.zeros(len(self.motion.positions(index,interval)))
            previous=self.motion.positions(index,interval)
            for step in range(1,9):
                current=self.positions(index,interval,step/8)
                total+=np.linalg.norm(current-previous,axis=1)
                previous=current
            return float(total.max())
        return self.paths.length(index, interval) * 100.


class MotionTiming:
    """Drape-style speed limit, in cm per captured-frame-sized simulation step.

Hair solves at 60 Hz, while the source can have any frame rate. Splitting a
source interval into N parts gives it N/source_fps seconds of physical time.
Positions and animation return frame numbers remain in source time.
"""
    def __init__(self, motion, max_move_cm=0.0):
        if not math.isfinite(max_move_cm) or max_move_cm < 0:
            raise ValueError("Avatar speed limit must be a finite, nonnegative distance.")
        self.parts = np.ones(motion.intervals, dtype=int)
        if max_move_cm > 0:
            largest = np.zeros(motion.intervals)
            for index in motion.samples:
                previous = motion.positions(index, 0)
                for interval in range(motion.intervals):
                    if getattr(motion, 'rigid', None) is not None:
                        largest[interval] = max(largest[interval], motion.rigid.length(index, interval))
                        continue
                    current = motion.positions(index, interval + 1)
                    largest[interval] = max(largest[interval],
                        float(np.linalg.norm(current - previous, axis=1).max()))
                    previous = current
            self.parts = np.maximum(1, np.ceil(largest / max_move_cm - 1e-9)).astype(int)
        self.source = np.arange(motion.intervals + 1, dtype=float) / motion.fps
        self.simulation = np.concatenate(([0.], np.cumsum(self.parts / motion.fps)))
        self.duration = float(self.simulation[-1])

    def source_time(self, simulation_time):
        return float(np.interp(simulation_time, self.simulation, self.source))


class HeadMotion:
    """Rigid head transform per time, fitted to avatar vertices around the scalp roots."""

    def __init__(self, motion, roots, reach=4.0, collider_index=None):
        self.motion = motion
        self.index, self.vertices = None, None
        best = 0
        for index in motion.samples:
            if collider_index is not None and index != collider_index:
                continue
            points = np.asarray(motion.colliders[index]["vertices"], float)
            from hair_strands import nearest_distance
            near = np.flatnonzero(nearest_distance(points, roots) < reach)
            if len(near) > best:
                best, self.index, self.vertices = len(near), index, near
        if self.index is not None and len(self.vertices) > 400:
            self.vertices = self.vertices[np.linspace(0, len(self.vertices) - 1, 400).astype(int)]
        self.rest = (np.asarray(motion.colliders[self.index]["vertices"], float)[self.vertices]
                     if self.index is not None else None)

    def transform(self, time):
        if self.index is None or len(self.vertices) < 4:
            return np.eye(3), np.zeros(3)
        return kabsch(self.rest, self.motion.at(self.index, time)[self.vertices])


class AnimationFrames:
    """One hair frame per DAZ frame, on disk (the shape write_animation_result expects)."""

    def __init__(self, start_frame, end_frame, fps, vertex_count, folder=None):
        self.start_frame, self.end_frame, self.fps = start_frame, end_frame, fps
        self.count = end_frame - start_frame + 1
        self.vertex_count = vertex_count
        self.stride = vertex_count * 3 * 4
        self.file = tempfile.TemporaryFile(dir=folder, prefix=".hair-frames-")
        self.seen = set()

    def put(self, sample, vertices):
        points = np.asarray(vertices, dtype="<f4")
        if not 0 <= sample < self.count or points.shape != (self.vertex_count, 3) or not np.isfinite(points).all():
            raise ValueError("Invalid animation frame geometry.")
        self.file.seek(sample * self.stride)
        self.file.write(points.tobytes())
        self.seen.add(sample)

    def get(self, sample):
        if sample not in self.seen:
            raise ValueError("The hair animation is incomplete. Run it again.")
        self.file.seek(sample * self.stride)
        return np.frombuffer(self.file.read(self.stride), dtype="<f4").reshape(-1, 3)

    def complete(self):
        return self.seen == set(range(self.count))

    def close(self):
        self.file.close()
