# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Eye guard: a see-through oval tube coming out of the face over both eyes, that hair drapes around (Spin256 ·
spin256.com/tools).

The tube (an elliptical cylinder along the eyes' forward direction) is fitted each capture from the avatar's eye
surfaces (Cornea, Sclera, Irises, Pupils, EyeMoisture), so by default its oval covers both eyes. Width and Height are the
oval's full sizes and Depth how far the tube reaches forward from the middle of the eyes (cm); Raise moves it up or
down. It is given in the capture's pose; during an animation the solver carries it with the head. Not saved."""
import re
import threading

import numpy as np

BACK = 2.0          # cm: the tube starts this far behind the middle of the eyes (inside the head)
EYE_SURFACE = re.compile(r"cornea|sclera|iris|pupil|eye_?moisture|eyeball", re.IGNORECASE)
EYE_FRONT = re.compile(r"cornea|iris|pupil", re.IGNORECASE)   # the front of the eyes: their gaze


def eye_vertices(avatar_entry, vertex_count, pattern=EYE_SURFACE):
    """Avatar vertex indices of its eye surfaces (from the capture's look OBJ), or an empty array. ``pattern``:
    EYE_FRONT for the front of the eyes only."""
    from hair_looks import read_look_obj
    if not avatar_entry or not avatar_entry.get("look_path"):
        return np.zeros(0, np.int64)
    try:
        corners, _, sizes, tokens, _ = read_look_obj(avatar_entry["look_path"])
    except (OSError, ValueError, TypeError):
        return np.zeros(0, np.int64)
    names = {look.get("token"): look.get("name") or "" for look in avatar_entry.get("looks") or []}
    eye_faces = np.array([bool(pattern.search(names.get(t, ""))) for t in tokens], bool)
    if not eye_faces.any():
        return np.zeros(0, np.int64)
    face_of_corner = np.repeat(np.arange(len(sizes)), sizes)
    found = np.unique(corners[eye_faces[face_of_corner]])
    return found[found < vertex_count]


class EyeGuard:
    """The fitted oval: center (3,), axes (3, 3) rows = across, up, forward; sizes (width, height, depth) in cm."""

    def __init__(self, center, axes, sizes, raise_cm=0.):
        self.center = np.asarray(center, float)
        self.axes = np.asarray(axes, float)
        self.sizes = np.asarray(sizes, float)
        self.raise_cm = float(raise_cm)

    def placed_center(self):
        return self.center + self.axes[1] * self.raise_cm

    def radii(self):
        """Half width, half height, and the tube's length (forward from the middle of the eyes)."""
        return np.array([max(self.sizes[0] * .5, .05), max(self.sizes[1] * .5, .05), max(self.sizes[2], .05)])

    def copy(self, **changes):
        guard = EyeGuard(self.center, self.axes, self.sizes, self.raise_cm)
        for key, value in changes.items():
            setattr(guard, key, np.asarray(value, float) if key == "sizes" else value)
        return guard

    def inside(self, points):
        """True for points inside the tube (capture pose)."""
        local = (np.asarray(points, float) - self.placed_center()) @ self.axes.T
        a, b, length = self.radii()
        return ((local[:, 0] / a) ** 2 + (local[:, 1] / b) ** 2 < 1.) & (local[:, 2] >= -BACK) & (local[:, 2] <= length)


def fit_eye_guard(avatar_vertices, eyes, front=None):
    """An oval tube covering both eyes, or None when the eyes are not found (fewer than 8 eye vertices). ``front``:
    the cornea/iris/pupil vertices, which give the eyes' forward direction."""
    vertices = np.asarray(avatar_vertices, float)
    eyes = np.asarray(eyes, np.int64)
    if len(eyes) < 8:
        return None
    points = vertices[eyes]
    middle = points.mean(axis=0)
    # Across: the main spread of the eye points (left eye to right eye), pointing to +x.
    _, _, vt = np.linalg.svd(points - middle, full_matrices=False)
    across = vt[0] if vt[0][0] >= 0 else -vt[0]
    side = (points - middle) @ across
    left, right = points[side < 0], points[side >= 0]
    if not len(left) or not len(right):
        return None
    eye_radius = float(np.mean([np.linalg.norm(p - p.mean(0), axis=1).max() for p in (left, right)]))
    # Forward: the eyes' gaze (front of the eyes from the eyeballs' middle); without those surfaces, level and away
    # from the head's middle (the avatar around the eyes). Square to across.
    front = np.zeros(0, np.int64) if front is None else np.asarray(front, np.int64)
    if len(front) >= 4:
        forward = vertices[front].mean(axis=0) - middle
    else:
        near = vertices[np.linalg.norm(vertices - middle, axis=1) < max(8 * eye_radius, 6.)]
        forward = middle - near.mean(axis=0)
        forward[1] = 0.
    forward -= across * (forward @ across)
    if np.linalg.norm(forward) < 1e-6:
        forward = np.cross(across, [0., 1., 0.])
    forward /= np.linalg.norm(forward)
    up = np.cross(forward, across)
    up /= np.linalg.norm(up)
    if up[1] < 0:             # flip upright (and the forward direction keeps facing out)
        up = -up
    across = np.cross(up, forward)
    span = float(np.ptp(side))
    sizes = (span + eye_radius * 1.2, eye_radius * 3.6, 25.)       # depth: the tube reaches 25 cm forward
    return EyeGuard(middle, np.array([across, up, forward]), sizes)


class EyeGuardControl:
    """Thread-safe live eye guard shared by the window and the solver (None: off)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.revision = 0
        self.guard = None

    def set(self, guard):
        with self.lock:
            self.guard = None if guard is None else guard.copy()
            self.revision += 1

    def snapshot(self):
        with self.lock:
            return self.revision, self.guard
