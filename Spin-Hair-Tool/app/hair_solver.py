# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""GPU strand solver (NVIDIA Warp): position-based chains with collisions, then mesh rebuild.

Units are centimeters and seconds, Y up. Each strand is processed by one GPU
thread that walks its chain (stretch, bending and a root tether), so there are no
write races and no graph colouring. Attachments (strand roots hanging from other
strands, tips held by frozen cards) and avatar collisions run as separate passes.
Gravity (aimable), wind and water come from live controls read between frames;
live pulling moves grabbed points toward targets set from the viewport.
"""
import threading

import numpy as np
import warp as wp

GRAVITY = -981.0          # cm/s^2
MAX_SPEED = 1500.0        # cm/s: hair tips whip fast when a figure turns or leans quickly
REFERENCE_PASSES = 32     # bend settings are tuned at 8 substeps x 4 iterations


@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
              inv_mass: wp.array(dtype=float), gravity: wp.array(dtype=wp.vec3), accel: wp.array(dtype=wp.vec3),
              weight: float, dt: float, damping: float):
    i = wp.tid()
    prev[i] = pos[i]
    if inv_mass[i] > 0.0:
        v = (vel[i] + (gravity[0] * weight + accel[i]) * dt) * wp.exp(-damping * dt)
        vel[i] = v
        pos[i] = pos[i] + v * dt


@wp.kernel
def add_unsupported(accel: wp.array(dtype=wp.vec3), loose: wp.array(dtype=float), extra: wp.vec3):
    """Grown hair: the gravity the style support takes away, given back to loose points."""
    i = wp.tid()
    accel[i] = accel[i] + extra * loose[i]


@wp.kernel
def solve_strands(pos: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
                  start: wp.array(dtype=int), count: wp.array(dtype=int),
                  seg_len: wp.array(dtype=float), bend_len: wp.array(dtype=float), arc: wp.array(dtype=float),
                  span3: wp.array(dtype=float), span5: wp.array(dtype=float),
                  anchor: wp.array(dtype=wp.vec3), anchor_len: wp.array(dtype=float),
                  lift_own: wp.array(dtype=float),
                  bend_strength: float, spray_strength: float, lift_strength: float,
                  tether_slack: float, iterations: int):
    s = wp.tid()
    first = start[s]
    n = count[s]
    for it in range(iterations):
        # Stretch: keep each segment at its rest length.
        for k in range(n - 1):
            a = first + k
            b = a + 1
            w = inv_mass[a] + inv_mass[b]
            if w > 0.0:
                d = pos[b] - pos[a]
                length = wp.length(d)
                if length > 1.0e-9:
                    c = d * ((length - seg_len[a]) / (length * w))
                    pos[a] = pos[a] + c * inv_mass[a]
                    pos[b] = pos[b] - c * inv_mass[b]
        # Bending: soft distance between every second particle (a negative rest length turns it off: grown hair).
        for k in range(n - 2):
            a = first + k
            b = a + 2
            w = inv_mass[a] + inv_mass[b]
            if w > 0.0 and bend_len[a] > 0.0:
                d = pos[b] - pos[a]
                length = wp.length(d)
                if length > 1.0e-9:
                    c = d * ((length - bend_len[a]) * bend_strength / (length * w))
                    pos[a] = pos[a] + c * inv_mass[a]
                    pos[b] = pos[b] - c * inv_mass[b]
        # Hairspray: keep the strand's own shape over longer spans (3 and 5 points), so curls and
        # arcs hold while the strand still moves freely as a whole (nothing is pinned in space).
        if spray_strength > 0.0:
            for k in range(n - 3):
                a = first + k
                b = a + 3
                w = inv_mass[a] + inv_mass[b]
                if w > 0.0:
                    d = pos[b] - pos[a]
                    length = wp.length(d)
                    if length > 1.0e-9:
                        c = d * ((length - span3[a]) * spray_strength / (length * w))
                        pos[a] = pos[a] + c * inv_mass[a]
                        pos[b] = pos[b] - c * inv_mass[b]
            for k in range(n - 5):
                a = first + k
                b = a + 5
                w = inv_mass[a] + inv_mass[b]
                if w > 0.0:
                    d = pos[b] - pos[a]
                    length = wp.length(d)
                    if length > 1.0e-9:
                        c = d * ((length - span5[a]) * spray_strength / (length * w))
                        pos[a] = pos[a] + c * inv_mass[a]
                        pos[b] = pos[b] - c * inv_mass[b]
        # Root lift: the first points keep their distance from a fixed point just beneath the root,
        # so the strand leaves the scalp at its styled angle (like a follicle); beyond, it bends freely.
        # A grown area's own root lift (lift_own >= 0) replaces the simulation's.
        if inv_mass[first] == 0.0:
            v = anchor[s]
            for k in range(1, wp.min(n, 4)):
                q = first + k
                strength = lift_strength
                if lift_own[q] >= 0.0:
                    strength = lift_own[q]
                if inv_mass[q] > 0.0 and strength > 0.0:
                    d = pos[q] - v
                    length = wp.length(d)
                    if length > 1.0e-9 and anchor_len[q] > 0.0:
                        pos[q] = pos[q] - d * ((length - anchor_len[q]) * strength / length)
        # Tether: a particle never drifts further from a held root than the strand allows.
        if inv_mass[first] == 0.0:
            root = pos[first]
            for k in range(1, n):
                p = first + k
                if inv_mass[p] > 0.0:
                    d = pos[p] - root
                    length = wp.length(d)
                    limit = arc[p] * (1.0 + tether_slack)
                    if length > limit and length > 1.0e-9:
                        pos[p] = root + d * (limit / length)


@wp.kernel
def project_grab(pos: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=float), targets: wp.array(dtype=wp.vec3),
                 weights: wp.array(dtype=float), max_step: float):
    """Live pulling: move grabbed points toward their targets, at most max_step per substep."""
    i = wp.tid()
    w = weights[i]
    if w > 0.0 and inv_mass[i] > 0.0:
        d = (targets[i] - pos[i]) * wp.min(w, 1.0)
        length = wp.length(d)
        limit = max_step * wp.min(w, 1.0)
        if length > limit:
            d = d * (limit / length)
        pos[i] = pos[i] + d


@wp.kernel
def collide_eye_guard(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
                      center: wp.vec3, axes: wp.mat33, radii: wp.vec3, strength: float):
    """Eye guard: hair inside the oval tube (radii: half width, half height, length forward
    from the eyes) is pushed out sideways to its wall, without friction."""
    i = wp.tid()
    if inv_mass[i] > 0.0:
        p = pos[i]
        local = axes * (p - center)                       # rows of axes: across, up, forward
        qx = local[0] / radii[0]
        qy = local[1] / radii[1]
        d2 = qx * qx + qy * qy
        if d2 < 1.0 and local[2] >= -2.0 and local[2] <= radii[2]:   # from 2 cm behind the eyes (eye_guard.BACK)
            wall = wp.vec3(0.0, radii[1], local[2])        # straight up when on the tube's axis
            if d2 > 1.0e-12:
                scale = 1.0 / wp.sqrt(d2)
                wall = wp.vec3(local[0] * scale, local[1] * scale, local[2])
            target = center + wp.transpose(axes) * wall
            moved = p + (target - p) * strength
            push = moved - p
            size = wp.length(push)
            if size > 1.0e-9:
                normal = push / size
                movement = p - prev[i]
                pos[i] = moved
                # Keep sliding and outgoing motion, stop motion into the oval.
                remaining = movement - normal * wp.min(wp.dot(movement, normal), 0.0)
                prev[i] = moved - remaining


class GrabControl:
    """Thread-safe live pull shared by the window and the solver (positions in cm)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.revision = 0
        self.grab = None

    def set(self, indices, targets, weights):
        with self.lock:
            self.grab = (np.asarray(indices, np.int64).copy(), np.asarray(targets, np.float32).copy(),
                         np.asarray(weights, np.float32).copy())
            self.revision += 1

    def clear(self):
        with self.lock:
            self.grab = None
            self.revision += 1

    def snapshot(self):
        with self.lock:
            return self.revision, self.grab


@wp.kernel
def pin_pulls(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
              inv: wp.array(dtype=float), indices: wp.array(dtype=int), local: wp.array(dtype=wp.vec3),
              rotation: wp.mat33, translation: wp.vec3):
    k = wp.tid()
    i = indices[k]
    p = rotation * local[k] + translation
    pos[i] = p; prev[i] = p; vel[i] = wp.vec3(0.); inv[i] = 0.


@wp.kernel
def finish_edit(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
                affected: wp.array(dtype=int)):
    i = wp.tid()
    if affected[i] != 0:
        prev[i] = pos[i]
        vel[i] = wp.vec3(0.)


@wp.kernel
def style_hold(pos: wp.array(dtype=wp.vec3), rest: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
               hold: wp.array(dtype=float)):
    """Soft pull back to the styled shape, strongest at the root and fading to the tip."""
    i = wp.tid()
    if inv_mass[i] > 0.0 and hold[i] > 0.0:
        pos[i] = pos[i] + (rest[i] - pos[i]) * hold[i]


@wp.kernel
def solve_attachments(pos: wp.array(dtype=wp.vec3), source: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
                      child: wp.array(dtype=int), parent: wp.array(dtype=int), rest: wp.array(dtype=float)):
    i = wp.tid()
    c = child[i]
    p = parent[i]
    # These are inferred supports, not knots tying two independent hairs together.
    # Only the child root follows: a short detail must not pull a parent's tip.
    # Read a snapshot so several children sharing a parent cannot race.
    if inv_mass[c] > 0.0:
        d = source[p] - source[c]
        length = wp.length(d)
        if length > rest[i] and length > 1.0e-9:
            pos[c] = source[c] + d * ((length - rest[i]) / length)


@wp.kernel
def touching(pos: wp.array(dtype=wp.vec3), mesh: wp.uint64, reach: float, out: wp.array(dtype=int)):
    i = wp.tid()
    query = wp.mesh_query_point(mesh, pos[i], reach)
    if query.result:
        if query.sign < 0.0 or wp.length(pos[i] - wp.mesh_eval_position(mesh, query.face, query.u, query.v)) < reach:
            out[i] = 1


@wp.kernel
def move_kinematic(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3), index: wp.array(dtype=int),
                   base: wp.array(dtype=wp.vec3), r0: wp.mat33, t0: wp.vec3, r1: wp.mat33, t1: wp.vec3, blend: float):
    """Points carried by the head (roots, frozen hair): between two head poses."""
    k = wp.tid()
    i = index[k]
    a = r0 * base[k] + t0
    b = r1 * base[k] + t1
    prev[i] = pos[i]
    pos[i] = a + (b - a) * blend


@wp.func
def surface_basis(mesh: wp.uint64, face: int):
    a = wp.mesh_get_point(mesh, face * 3)
    b = wp.mesh_get_point(mesh, face * 3 + 1)
    c = wp.mesh_get_point(mesh, face * 3 + 2)
    tangent = wp.normalize(b - a)
    normal = wp.normalize(wp.cross(b - a, c - a))
    return tangent, wp.cross(normal, tangent), normal


@wp.kernel
def bind_surface(points: wp.array(dtype=wp.vec3), index: wp.array(dtype=int), mesh: wp.uint64,
                 face: wp.array(dtype=int), bary: wp.array(dtype=wp.vec2), local: wp.array(dtype=wp.vec3)):
    """Frozen hair and pinned ends: the nearest point of the avatar surface and the offset
    from it in that triangle's frame, so the point rides that spot of the scalp as the avatar moves and deforms."""
    k = wp.tid()
    p = points[index[k]]
    query = wp.mesh_query_point_no_sign(mesh, p, 1.0e6)
    face[k] = -1
    if query.result:
        surface = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        tangent, bitangent, normal = surface_basis(mesh, query.face)
        d = p - surface
        face[k] = query.face
        bary[k] = wp.vec2(query.u, query.v)
        local[k] = wp.vec3(wp.dot(d, tangent), wp.dot(d, bitangent), wp.dot(d, normal))


@wp.kernel
def eval_surface(mesh: wp.uint64, face: wp.array(dtype=int), bary: wp.array(dtype=wp.vec2),
                 local: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    surface = wp.mesh_eval_position(mesh, face[k], bary[k][0], bary[k][1])
    tangent, bitangent, normal = surface_basis(mesh, face[k])
    l = local[k]
    out[k] = surface + tangent * l[0] + bitangent * l[1] + normal * l[2]


@wp.kernel
def move_bound(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3), index: wp.array(dtype=int),
               start: wp.array(dtype=wp.vec3), end: wp.array(dtype=wp.vec3), blend: float):
    k = wp.tid()
    i = index[k]
    prev[i] = pos[i]
    pos[i] = start[k] + (end[k] - start[k]) * blend


@wp.kernel
def collide_avatar(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
                   mesh: wp.uint64, radius: float, friction: float):
    i = wp.tid()
    if inv_mass[i] > 0.0:
        p = pos[i]
        query = wp.mesh_query_point(mesh, p, radius + 5.0)
        if query.result:
            surface = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            d = p - surface
            distance = wp.length(d)
            normal = wp.vec3(0.0, 1.0, 0.0)
            if distance > 1.0e-9:
                normal = d / distance * query.sign
            depth = radius - distance * query.sign
            if depth > 0.0:
                movement = p - prev[i]
                p = p + normal * depth
                # Friction: remove part of the sliding motion along the surface.
                tangential = movement - normal * wp.dot(movement, normal)
                pos[i] = p - tangential * friction
                # Depenetration is a position repair, not an elastic impulse.
                # Preserve sliding and outgoing motion, and stop inward motion.
                remaining = tangential * (1.0 - friction) + normal * wp.max(wp.dot(movement, normal), 0.0)
                prev[i] = pos[i] - remaining


@wp.kernel
def shell_gaps(rest: wp.array(dtype=wp.vec3), mesh: wp.uint64, radius: float, gaps: wp.array(dtype=float)):
    i = wp.tid()
    gap = radius
    query = wp.mesh_query_point(mesh, rest[i], radius + 0.001)
    if query.result:
        point = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        gap = wp.min(radius, wp.length(rest[i] - point))
    gaps[i] = gap


@wp.func
def shell_project(p: wp.vec3, previous: wp.vec3, mesh: wp.uint64, gap: float):
    movement = p - previous
    length = wp.length(movement)
    if length > 0.0002:
        direction = movement / length
        ray = wp.mesh_query_ray(mesh, previous + direction * 0.0001, direction, length - 0.0001)
        if ray.result:
            normal = ray.normal
            if wp.dot(normal, direction) > 0.0:
                normal = -normal
            surface = wp.mesh_eval_position(mesh, ray.face, ray.u, ray.v)
            p = p + normal * wp.max(gap - wp.dot(p - surface, normal), 0.0)
    query = wp.mesh_query_point(mesh, p, gap + 0.001)
    if query.result:
        surface = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        delta = p - surface
        distance = wp.length(delta)
        if distance > 1.0e-7 and distance < gap - 0.00001:
            p = surface + delta * (gap / distance)
    return p


@wp.kernel
def shell_particles(pos: wp.array(dtype=wp.vec3), previous: wp.array(dtype=wp.vec3),
                    inv_mass: wp.array(dtype=float), mesh: wp.uint64, gaps: wp.array(dtype=float),
                    turn: wp.mat33, shift: wp.vec3):
    i = wp.tid()
    if inv_mass[i] > 0.0:
        old = pos[i]
        corrected = shell_project(old, turn * previous[i] + shift, mesh, gaps[i])
        repair = corrected - old
        pos[i] = corrected
        # Contact repair must not become an outward velocity impulse.
        size = wp.length(repair)
        if size > 1.0e-7:
            normal = repair / size
            movement = old - previous[i]
            movement = movement - normal * wp.min(wp.dot(movement, normal), 0.0)
            previous[i] = corrected - movement


@wp.kernel
def shell_segments(pos: wp.array(dtype=wp.vec3), previous: wp.array(dtype=wp.vec3),
                   inv: wp.array(dtype=float), starts: wp.array(dtype=int), counts: wp.array(dtype=int),
                   mesh: wp.uint64, gaps: wp.array(dtype=float)):
    # One thread walks one strand root-to-tip. Only its next free point changes,
    # so correcting a later link cannot undo an earlier link or race another strand.
    s=wp.tid();first=starts[s]
    for k in range(counts[s]-1):
        a=first+k;b=a+1
        if inv[b]>0.0:
            origin=pos[a];old=pos[b];delta=old-origin;length=wp.length(delta)
            if length>0.0002:
                direction=delta/length
                hit=wp.mesh_query_ray(mesh,origin+direction*.0001,direction,length-.0002)
                if hit.result:
                    normal=hit.normal
                    if wp.dot(normal,direction)>0.0:normal=-normal
                    surface=wp.mesh_eval_position(mesh,hit.face,hit.u,hit.v)
                    gap=wp.max(.001,wp.min(gaps[a],gaps[b]))
                    corrected=old+normal*wp.max(gap-wp.dot(old-surface,normal),0.0)
                    # Slide along the contacted face. At a corner, limit the link
                    # to its first obstruction instead of accepting another crossing.
                    slide=corrected-origin;size=wp.length(slide)
                    if size>0.0002:
                        travel=slide/size
                        again=wp.mesh_query_ray(mesh,origin+travel*.0001,travel,size-.0002)
                        if again.result:
                            corrected=origin+travel*wp.max(0.0,again.t+.0001-gap)
                    pos[b]=corrected
                    movement=old-previous[b]
                    movement=movement-normal*wp.min(wp.dot(movement,normal),0.0)
                    previous[b]=corrected-movement


@wp.kernel
def shell_vertices(pos: wp.array(dtype=wp.vec3), previous: wp.array(dtype=wp.vec3),
                   moving: wp.array(dtype=int), mesh: wp.uint64, gaps: wp.array(dtype=float),
                   turn: wp.mat33, shift: wp.vec3):
    i = wp.tid()
    if moving[i] != 0:
        pos[i] = shell_project(pos[i], turn * previous[i] + shift, mesh, gaps[i])


@wp.kernel
def update_velocity(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
                    inv_mass: wp.array(dtype=float), dt: float, max_speed: float):
    i = wp.tid()
    if inv_mass[i] > 0.0:
        v = (pos[i] - prev[i]) / dt
        speed = wp.length(v)
        if speed > max_speed:
            v = v * (max_speed / speed)
        vel[i] = v
    else:
        vel[i] = wp.vec3(0.0)


@wp.kernel
def transport_frames(pos: wp.array(dtype=wp.vec3), rest_frames: wp.array(dtype=wp.mat33),
                     start: wp.array(dtype=int), count: wp.array(dtype=int), frames: wp.array(dtype=wp.mat33)):
    """Carry each strand's rest frames along its current shape (minimal rotation per segment)."""
    s = wp.tid()
    first = start[s]
    n = count[s]
    for k in range(n):
        a = first + k
        b = first + wp.min(k + 1, n - 1)
        o = first + wp.max(k - 1, 0)
        rest = rest_frames[a]
        if k < n - 1:
            t = pos[b] - pos[a]
        else:
            t = pos[a] - pos[o]
        length = wp.length(t)
        rest_t = wp.vec3(rest[0, 0], rest[0, 1], rest[0, 2])
        if length < 1.0e-9:
            frames[a] = rest
        else:
            t = t / length
            # Rotation taking the rest tangent onto the current tangent.
            axis = wp.cross(rest_t, t)
            sin_angle = wp.length(axis)
            cos_angle = wp.dot(rest_t, t)
            r = wp.identity(n=3, dtype=float)
            if sin_angle > 1.0e-9:
                axis = axis / sin_angle
                kx = wp.mat33(0.0, -axis[2], axis[1], axis[2], 0.0, -axis[0], -axis[1], axis[0], 0.0)
                r = r + kx * sin_angle + (kx * kx) * (1.0 - cos_angle)
            elif cos_angle < 0.0:
                r = -r
            # Rows of the frame are T, N, B: rotate each row.
            frames[a] = rest * wp.transpose(r)


@wp.kernel
def rebuild(pos: wp.array(dtype=wp.vec3), frames: wp.array(dtype=wp.mat33), rest: wp.array(dtype=wp.vec3),
            segment: wp.array(dtype=int), t: wp.array(dtype=float), offset: wp.array(dtype=wp.vec3),
            moving: wp.array(dtype=int), out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if moving[i] == 0:
        out[i] = rest[i]
    else:
        j = segment[i]
        f = frames[j]
        base = pos[j] * (1.0 - t[i]) + pos[j + 1] * t[i]
        o = offset[i]
        out[i] = base + wp.vec3(f[0, 0], f[0, 1], f[0, 2]) * o[0] + wp.vec3(f[1, 0], f[1, 1], f[1, 2]) * o[1] \
            + wp.vec3(f[2, 0], f[2, 1], f[2, 2]) * o[2]


@wp.kernel
def face_normals(points: wp.array(dtype=wp.vec3), faces: wp.array2d(dtype=int), normals: wp.array(dtype=wp.vec3)):
    """Area-weighted face normal added to each corner (quads split along 0-2)."""
    f = wp.tid()
    a = faces[f, 0]
    b = faces[f, 1]
    c = faces[f, 2]
    n = wp.cross(points[b] - points[a], points[c] - points[a])
    d = faces[f, 3]
    if d >= 0:
        n = n + wp.cross(points[c] - points[a], points[d] - points[a])
        wp.atomic_add(normals, d, n)
    wp.atomic_add(normals, a, n)
    wp.atomic_add(normals, b, n)
    wp.atomic_add(normals, c, n)


@wp.kernel
def normalize(normals: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    length = wp.length(normals[i])
    if length > 1.0e-12:
        normals[i] = normals[i] / length
    else:
        normals[i] = wp.vec3(0.0, 1.0, 0.0)


def style_down(roots):
    """Down for the head a style was made on: from a sphere fitted to the scalp roots, away from
    the roots' center. Straight down when there are too few roots to tell."""
    roots = np.asarray(roots, float)
    if len(roots) < 8:
        return np.array([0.0, -1.0, 0.0])
    fit = np.linalg.lstsq(np.c_[2 * roots, np.ones(len(roots))], (roots ** 2).sum(1), rcond=None)[0]
    down = fit[:3] - roots.mean(0)
    length = np.linalg.norm(down)
    return down / length if length > 1e-6 else np.array([0.0, -1.0, 0.0])


class HairSimulation:
    """One hair model on the GPU. ``frozen`` strands and scalp-rooted roots are held fixed."""

    def __init__(self, model, colliders, frozen=(), tip_pairs=None, *, fps=60.0, substeps=8, iterations=4,
                 bend=0.35, damping=2.0, radius=0.25, friction=0.0, tether_slack=0.02, gravity_scale=1.0,
                 style=0.5, style_falloff=2.0, style_length=6.0, shelter=0.4, root_hold=0.0, wind_response=None,
                 hairspray=0.0, root_lift=0.8, style_weight=0.0, style_rest=None, frozen_reach=3.0, hair_collision=True, hair_spacing=0.6, hold_root_direction=False,
                 stiffness=0.0, stiffness_taper=0.0, stiffness_swing=False, momentum=0.0, weightless_lift=0.0, motion=None, gravity_control=None,
                 wind_control=None, water_control=None, grab_control=None, pull_speed=400.0, head_collider=None,
                 eye_guard_control=None,
                 trapped_distance=1.0, max_move_cm=0.0, clumping=0.0, head_drag=0.0, pinned_ends=(), loose=None,
                 device="cuda:0"):
        wp.init()
        self.model, self.device = model, device
        self.fps, self.substeps, self.iterations = fps, substeps, iterations
        self.dt = 1.0 / (fps * substeps)
        self.damping, self.radius, self.friction = damping, radius, friction
        self.tether_slack = tether_slack
        self.gravity_scale = gravity_scale
        self.max_move_cm = max_move_cm
        # Style at rest: the captured style stands as it does because a full head of hair (not just
        # these strands) holds it up. That support carries this share of the weight the style was
        # captured under (normal gravity, straight down). Gravity beyond 1 g, or turned away from
        # straight down, is supported less, so an upright figure keeps ~this share of its style while
        # leaning or stronger gravity still change it. Nothing is pinned in place.
        self.style_rest = min(max(style_rest if style_rest is not None else style_weight, 0.0), 1.0)
        self.weight = 1.0
        self.gravity_control, self.wind_control = gravity_control, wind_control
        self.water_control, self.grab_control = water_control, grab_control
        self.eye_guard_control = eye_guard_control      # eye guard
        self.eye_guard_frames = 0
        self.grab_revision = None
        self.pull_step = pull_speed / (fps * substeps)
        passes = substeps * iterations
        # Percentages mean "share of a shape change undone per frame", independent of passes.
        self.bend_strength = 1.0 - (1.0 - min(max(bend, 0.0), 0.999)) ** (1.0 / passes)
        self.spray_strength = 1.0 - (1.0 - min(max(hairspray, 0.0), 0.999)) ** (1.0 / passes)
        self.lift_strength = 1.0 - (1.0 - min(max(root_lift, 0.0), 0.999)) ** (1.0 / passes)
        start, count, particles = model.particle_start, model.particle_count, model.particles
        owner = np.repeat(np.arange(model.strand_count), count)
        index = np.arange(len(particles))
        last = np.repeat(start + count - 1, count)
        seg = np.zeros(len(particles))
        has_next = index < last
        seg[has_next] = np.linalg.norm(particles[index[has_next] + 1] - particles[index[has_next]], axis=1)
        bend = np.zeros(len(particles))
        has_two = index + 1 < last
        bend[has_two] = np.linalg.norm(particles[index[has_two] + 2] - particles[index[has_two]], axis=1)
        # Virtual follicle point 2 cm behind each root, along the strand's first direction.
        first_dir = particles[np.minimum(start + 1, len(particles) - 1)] - particles[start]
        first_dir /= np.maximum(np.linalg.norm(first_dir, axis=1, keepdims=True), 1e-9)
        anchors = particles[start] - first_dir * 2.0
        anchor_len = np.linalg.norm(particles - np.repeat(anchors, count, axis=0), axis=1)
        span3, span5 = np.zeros(len(particles)), np.zeros(len(particles))
        has3, has5 = index + 3 <= last, index + 5 <= last
        span3[has3] = np.linalg.norm(particles[index[has3] + 3] - particles[index[has3]], axis=1)
        span5[has5] = np.linalg.norm(particles[index[has5] + 5] - particles[index[has5]], axis=1)
        cumulative = np.concatenate([[0.], np.cumsum(seg)])
        arc = cumulative[index] - cumulative[np.repeat(start, count)]
        inv = np.ones(len(particles))
        roots = start[model.root_kind == 0]
        inv[roots] = 0.0
        if hold_root_direction:
            # The second particle sets the angle the strand leaves the scalp at.
            inv[roots + 1] = 0.0
        # Style hold per particle: strongest at the root, fading to zero at the tip,
        # normalised per frame like the bend strength.
        strand_length = np.maximum(np.repeat(cumulative[start + count - 1] - cumulative[start], count), 1e-6)
        # Style hold fades over the whole strand, or over style_length cm from the root when set.
        reach = strand_length if style_length is None else np.minimum(strand_length, style_length)
        along = np.clip(arc / reach, 0.0, 1.0)
        hold = min(max(style, 0.0), 0.999) * (1.0 - along) ** style_falloff
        if root_hold > 0:
            # Root firmness: within root_hold cm of every root (scalp, tie or parent strand) the styled
            # shape is held firmly, so hair leaves its root at the styled angle and bends further out.
            firm = 0.97 * np.clip(1.0 - arc / root_hold, 0.0, 1.0) ** 0.5
            hold = np.maximum(hold, firm)
        hold = 1.0 - (1.0 - hold) ** (1.0 / substeps)
        frozen = np.asarray(sorted(set(int(s) for s in frozen)), int)
        if len(frozen):
            inv[np.isin(owner, frozen)] = 0.0
        # Pinned ends: the far end of each selected strand is held too (riding the scalp like
        # frozen hair), so swept-back hair is fixed at the scalp and at the tie and simulates in between.
        pinned_ends = np.asarray(sorted(set(int(s) for s in pinned_ends) - set(frozen.tolist())), int)
        self.pinned_tips = (start[pinned_ends] + count[pinned_ends] - 1) if len(pinned_ends) else np.zeros(0, int)
        inv[self.pinned_tips] = 0.0
        # Points that ride the avatar surface rather than the rigid head: frozen hair and pinned ends.
        self.surface_points = np.union1d(np.flatnonzero(np.isin(owner, frozen)), self.pinned_tips).astype(np.int64)
        # Strands starting at frozen hair (a tie, bun or clip) hang from it: their root is held
        # there, rather than by a neighbouring strand.
        held_by_frozen = set()
        if len(frozen):
            from hair_strands import nearest_distance
            frozen_points = particles[np.isin(owner, frozen)]
            candidates = np.flatnonzero(~np.isin(np.arange(model.strand_count), frozen) & (model.root_kind != 0))
            near = nearest_distance(particles[start[candidates]], frozen_points) <= frozen_reach
            held_by_frozen = set(candidates[near].tolist())
            inv[start[list(held_by_frozen)]] = 0.0
        self.held_by_frozen = len(held_by_frozen)
        children, parents = [], []
        for s in np.flatnonzero(model.root_kind == 1):
            if s not in set(frozen.tolist()) and s not in held_by_frozen:
                children.append(start[s]); parents.append(model.parent_particle[s])
        if tip_pairs is not None and len(tip_pairs):
            children.extend(tip_pairs[:, 0].tolist()); parents.extend(tip_pairs[:, 1].tolist())
        rest_gap = np.linalg.norm(particles[children] - particles[parents], axis=1) if children else np.zeros(0)

        moving = (model.strand_of_vertex >= 0).astype(np.int32)
        if len(frozen) and motion is None:
            # Without a timeline frozen cards keep their exact captured shape. With one, their particles ride the
            # scalp, so the cards are rebuilt from them.
            moving[np.isin(model.strand_of_vertex, frozen)] = 0
        with wp.ScopedDevice(device):
            self.pos = wp.array(particles.astype(np.float32), dtype=wp.vec3)
            self.prev = wp.array(particles.astype(np.float32), dtype=wp.vec3)
            self.attachment_source = wp.clone(self.pos)
            self.vel = wp.zeros(len(particles), dtype=wp.vec3)
            self.inv = wp.array(inv.astype(np.float32), dtype=float)
            self.original_inv = wp.clone(self.inv)
            self.pull_holds = {}
            self.start = wp.array(start.astype(np.int32), dtype=int)
            self.count = wp.array(count.astype(np.int32), dtype=int)
            self.seg = wp.array(seg.astype(np.float32), dtype=float)
            self.bend = wp.array(bend.astype(np.float32), dtype=float)
            self.span3 = wp.array(span3.astype(np.float32), dtype=float)
            self.anchor = wp.array(anchors.astype(np.float32), dtype=wp.vec3)
            self.anchor_len = wp.array(anchor_len.astype(np.float32), dtype=float)
            self.lift_own = wp.array(np.full(len(particles), -1., np.float32), dtype=float)   # -1: the simulation's
            self.span5 = wp.array(span5.astype(np.float32), dtype=float)
            self.arc = wp.array(arc.astype(np.float32), dtype=float)
            self.rest_particles = wp.array(particles.astype(np.float32), dtype=wp.vec3)
            self.hold = wp.array(hold.astype(np.float32), dtype=float)
            self.base_hold = hold.astype(np.float32)
            self.child = wp.array(np.asarray(children, np.int32), dtype=int)
            self.parent = wp.array(np.asarray(parents, np.int32), dtype=int)
            self.gap = wp.array(rest_gap.astype(np.float32), dtype=float)
            self.rest_frames = wp.array(model.rest_frames.astype(np.float32), dtype=wp.mat33)
            self.frames = wp.zeros(len(particles), dtype=wp.mat33)
            self.rest = wp.array(model.rest.astype(np.float32), dtype=wp.vec3)
            self.segment = wp.array(model.bind_segment.astype(np.int32), dtype=int)
            self.t = wp.array(model.bind_t.astype(np.float32), dtype=float)
            self.offset = wp.array(model.bind_offset.astype(np.float32), dtype=wp.vec3)
            self.moving = wp.array(moving, dtype=int)
            self.out = wp.zeros(len(model.rest), dtype=wp.vec3)
            self.meshes = [wp.Mesh(points=wp.array(np.asarray(c["vertices"], np.float32), dtype=wp.vec3),
                                   indices=wp.array(np.asarray(c["triangles"], np.int32).ravel(), dtype=int))
                           for c in colliders]
            self.trapped_contact = None
            if trapped_distance > 0 and len(self.meshes) > 1:
                from hair_contact import TrappedContact
                self.trapped_contact = TrappedContact(self.meshes, self.inv, trapped_distance)
            from collider_roles import resolve_modes, primary_index
            modes = resolve_modes(colliders, head_collider)
            self.thin_surfaces = [int(mode == "surface") for mode in modes]
            self.shells = []
            self.shell_delta = {}
            self.shell_points = {}
            self.output_frame = -1
            self.previous_output = wp.clone(self.rest)
            for index, (mesh, thin) in enumerate(zip(self.meshes, self.thin_surfaces)):
                if thin:
                    self.shell_points[index] = np.asarray(colliders[index]["vertices"],float).copy()
                    self.shell_delta[mesh.id] = (np.eye(3),np.zeros(3))
                    particle_gaps = wp.zeros(len(particles),dtype=float)
                    vertex_gaps = wp.zeros(len(model.rest),dtype=float)
                    wp.launch(shell_gaps,len(particles),inputs=[self.pos,mesh.id,self.radius,particle_gaps])
                    wp.launch(shell_gaps,len(model.rest),inputs=[self.rest,mesh.id,min(self.radius,.05),vertex_gaps])
                    self.shells.append((mesh,particle_gaps,vertex_gaps))
            self.gravity = wp.array(np.array([[0.0, GRAVITY * gravity_scale, 0.0]], np.float32), dtype=wp.vec3)
            from hair_forces import HairForces
            from hair_forces import WIND_RESPONSE
            self.forces = HairForces(start, count, self.meshes, shelter,
                                     WIND_RESPONSE if wind_response is None else wind_response)
            self.collision = None
            if hair_collision and hair_spacing > 0:
                from hair_collision import HairCollision
                self.collision = HairCollision(particles, count, spacing=hair_spacing)
            # Stiffness: shape matching per strand section (share restored per frame).
            self.stiffness = None
            if stiffness > 0:
                from hair_stiffness import StrandStiffness
                strength = 1.0 - (1.0 - min(stiffness, 0.999)) ** (1.0 / passes)
                self.stiffness = StrandStiffness(start, count, particles, anchors, strength, inv[start] == 0, taper=stiffness_taper,
                                                  swing=stiffness_swing)
            self.captured = particles.astype(np.float32)
            self.targets = wp.zeros(len(particles), dtype=wp.vec3)
            self.weights = wp.zeros(len(particles), dtype=float)
        self.style_down = style_down(particles[start[model.root_kind == 0]])
        self.setup_weightless(weightless_lift)
        self.setup_motion(motion, inv, primary_index(colliders,head_collider))
        if motion is not None and wind_control is not None:
            from hair_relative_wind import RelativeWind
            self.wind_control = RelativeWind(wind_control, self.head_transform)
        # Momentum (timeline playback only): stiffness halves at this speed relative to the root.
        self.momentum_speed = 20.0 / momentum if momentum > 0 else None
        # Coherent motion: clumping draws each particle's motion toward the mean motion of the
        # hair within 2 cm, so locks move together instead of every strand on its own; head drag lets motion relative
        # to the moving head die away (moving with the head is not resisted).
        self.clumping, self.head_drag = float(clumping), float(head_drag)
        self.forces.support = wp.vec4(self.style_rest, *[float(x) for x in self.style_down])
        self.particle_count = len(particles)
        self.strands = model.strand_count
        self.frame = 0
        self.setup_loose(loose or {}, start, count)

    def setup_loose(self, loose, start, count):
        """Grown hair: ``loose`` maps strand -> stiffness
        share (0..1). Those strands feel full gravity (no style support), have no style hold toward their grown pose,
        and keep only that share of the stiffness, so short grown hair falls and lies as hair does."""
        self.loose = None
        if not loose:
            return
        mask = np.zeros(self.particle_count, np.float32)
        share = np.ones(self.particle_count, np.float32)
        own_lift = np.zeros(self.particle_count, np.float32)
        passes = max(self.substeps * self.iterations, 1)
        for strand, value in loose.items():
            # value: stiffness share, or [stiffness share, root lift] (root lift per area)
            stiffness, lift = (value[0], value[1]) if isinstance(value, (list, tuple)) else (value, 0.)
            strand = int(strand)
            if 0 <= strand < len(start):
                a, b = int(start[strand]), int(start[strand] + count[strand])
                mask[a:b] = 1.
                share[a:b] = float(min(max(stiffness, 0.), 1.))
                # The area's root lift per pass, as the simulation's own lift_strength is made from Root lift.
                own_lift[a:b] = 1.0 - (1.0 - min(max(float(lift), 0.), .999)) ** (1.0 / passes)
        mask *= (self.inv.numpy() > 0).astype(np.float32)
        with wp.ScopedDevice(self.device):
            self.loose = wp.array(mask, dtype=float)
            self.base_hold = (self.base_hold * (1. - mask)).astype(np.float32)
            self.hold.assign(self.base_hold)
            # No bending to the grown (straight) shape and no root lift: the area's Stiffness alone keeps it springy.
            strand_mask = np.zeros(self.particle_count, bool)
            for strand in loose:
                if 0 <= strand < len(start):
                    strand_mask[int(start[strand]):int(start[strand] + count[strand])] = True
            self.loose_points = strand_mask
            bend = self.bend.numpy(); bend[strand_mask] = -1.; self.bend.assign(bend)
            # Grown strands use their area's root lift (0: none).
            self.loose_unlifted = strand_mask & (own_lift <= 0)
            lift = self.anchor_len.numpy(); lift[self.loose_unlifted] = -1.; self.anchor_len.assign(lift)
            self.lift_own.assign(np.where(strand_mask, own_lift, -1.).astype(np.float32))
            if self.stiffness is not None:
                base = (self.stiffness.base_taper.numpy() * share).astype(np.float32)
                self.stiffness.base_taper.assign(base)
                self.stiffness.taper.assign(base)
        self.unsupported = wp.vec3(0., 0., 0.)

    def gravity_vector(self):
        """Gravity in cm/s^2: from the live control when given (m/s^2), else straight down."""
        if self.gravity_control is not None:
            return np.asarray(self.gravity_control.snapshot()[1], float) * 100.0
        return np.array([0.0, GRAVITY * self.gravity_scale, 0.0])

    def supported(self, gravity):
        """Gravity the hair feels after the style's support (see style_rest)."""
        if self.style_rest <= 0:
            return gravity
        # Support works along gravity (hair holds up hair in any direction), up to the capture's
        # 1 g, and fades (cos^2) as gravity turns away from "down" for the head the style was made
        # on: a figure leaning forward has its style tilted with the head, so its hair drops.
        size = float(np.linalg.norm(gravity))
        if size < 1e-6:
            return gravity
        facing = max(float(np.dot(gravity / size, self.style_down)), 0.0) ** 2
        return gravity * (1.0 - self.style_rest * facing * min(size, -GRAVITY) / size)

    def setup_motion(self, motion, inv, head_collider=None):
        """Timeline capture: the avatar moves through its captured frames while ``playing``;
        everything held still (scalp roots, frozen hair, the scalp mesh) rides with the head."""
        self.motion, self.time, self.playing = motion, 0.0, False
        self.motion_elapsed = 0.0
        self.motion_timing = None
        self.head_pose = (np.eye(3), np.zeros(3))
        self.anchor0 = self.anchor.numpy().copy()
        if motion is None:
            return
        from hair_animation import HeadMotion, MotionTiming
        motion.prepare_rigid()
        self.motion_timing = MotionTiming(motion, self.max_move_cm)
        roots = self.captured[self.model.particle_start[self.model.root_kind == 0]]
        self.head = HeadMotion(motion, roots if len(roots) else self.captured[self.model.particle_start],
                               collider_index=head_collider)
        kinematic = np.flatnonzero(inv == 0)
        self.style_down0 = self.style_down.copy()
        self.rest0 = self.captured.copy()
        self.moving_meshes = {index: self.meshes[index] for index in motion.samples}
        self.bound_count = 0
        head_mesh = self.moving_meshes.get(self.head.index)
        candidates = np.intersect1d(getattr(self, "surface_points", np.zeros(0, int)), kinematic)
        if head_mesh is not None and len(candidates):
            with wp.ScopedDevice(self.device):
                index = wp.array(candidates.astype(np.int32), dtype=int)
                face = wp.zeros(len(candidates), dtype=int)
                bary = wp.zeros(len(candidates), dtype=wp.vec2)
                local = wp.zeros(len(candidates), dtype=wp.vec3)
                wp.launch(bind_surface, len(candidates), inputs=[self.pos, index, head_mesh.id, face, bary, local])
            found = face.numpy() >= 0
            bound = candidates[found]
            if len(bound):
                with wp.ScopedDevice(self.device):
                    self.bound_index = wp.array(bound.astype(np.int32), dtype=int)
                    self.bound_face = wp.array(face.numpy()[found], dtype=int)
                    self.bound_bary = wp.array(bary.numpy()[found], dtype=wp.vec2)
                    self.bound_local = wp.array(local.numpy()[found], dtype=wp.vec3)
                    self.bound_start = wp.zeros(len(bound), dtype=wp.vec3)
                    self.bound_end = wp.zeros(len(bound), dtype=wp.vec3)
                self.bound_mesh, self.bound_count = head_mesh, len(bound)
                kinematic = np.setdiff1d(kinematic, bound)           # the rest still ride the rigid head
                self.sync_bound()
        with wp.ScopedDevice(self.device):
            self.kin_index = wp.array(kinematic.astype(np.int32), dtype=int)
            self.kin_base = wp.array(self.captured[kinematic], dtype=wp.vec3)
        self.kin_count = len(kinematic)

    def sync_bound(self):
        """Surface-bound points where the avatar surface puts them now (start and end of the next frame alike)."""
        if not getattr(self, "bound_count", 0):
            return
        with wp.ScopedDevice(self.device):
            wp.launch(eval_surface, self.bound_count, inputs=[self.bound_mesh.id, self.bound_face, self.bound_bary,
                                                              self.bound_local, self.bound_end])
            wp.copy(self.bound_start, self.bound_end)
            wp.launch(move_bound, self.bound_count, inputs=[self.pos, self.prev, self.bound_index, self.bound_start,
                                                           self.bound_end, 1.0])
            wp.copy(self.prev, self.pos)

    def head_transform(self):
        """Current head pose (rotation, translation) relative to the start frame."""
        return self.head_pose

    def begin_wrap(self):
        """Loop join: from the last frame the avatar takes one real frame step into frame 0
        (the last frame is the one before frame 0, not a copy of it); then the timeline carries on from frame 0."""
        self.wrap, self.wrap_from = 0.0, self.time

    def avatar_points(self, index):
        wrap = getattr(self, "wrap", None)
        if wrap is None:
            return self.motion.at(index, self.time)
        return self.motion.at(index, self.wrap_from) * (1.0 - wrap) + self.motion.at(index, 0.0) * wrap

    def move_avatar(self):
        """Advance the timeline one frame (when playing); returns the start and end head poses."""
        start_pose = self.head_pose
        wrapping = getattr(self, "wrap", None) is not None
        if self.playing and wrapping:
            self.wrap = min(1.0, self.wrap + self.motion.fps / self.fps)
        elif self.playing:
            self.motion_elapsed = min(self.motion_elapsed + 1.0 / self.fps, self.motion_timing.duration)
            self.time = self.motion_timing.source_time(self.motion_elapsed)
        if wrapping and self.head.index is not None and len(self.head.vertices) >= 4:
            from hair_animation import kabsch
            r, t = kabsch(self.head.rest, self.avatar_points(self.head.index)[self.head.vertices])
        elif wrapping:
            r, t = np.eye(3), np.zeros(3)
        else:
            r, t = self.head.transform(self.time)
        self.head_pose = (r, t)
        for index, mesh in self.moving_meshes.items():
            points = self.avatar_points(index)
            if index in self.shell_points:
                from hair_animation import kabsch
                old = self.shell_points[index]
                samples = np.linspace(0,len(old)-1,min(len(old),400)).astype(int)
                self.shell_delta[mesh.id] = kabsch(old[samples],points[samples])
                self.shell_points[index] = points.copy()
            mesh.points.assign(points.astype(np.float32))
            mesh.refit()
        self.anchor.assign((self.anchor0 @ r.T + t).astype(np.float32))
        if self.stiffness is not None:
            self.stiffness.anchor.assign((self.anchor0 @ r.T + t).astype(np.float32))
        self.rest_particles.assign((self.rest0 @ r.T + t).astype(np.float32))
        self.style_down = r @ self.style_down0
        self.forces.support = wp.vec4(self.style_rest, *[float(x) for x in self.style_down])
        if wrapping and self.wrap >= 1.0:
            self.wrap, self.time, self.motion_elapsed = None, 0.0, 0.0          # frame 0 reached
        return start_pose, (r, t)

    EYE_GUARD_EASE = 0.5        # seconds over which a newly shown eye guard pushes in (no snap on hair already there)

    def eye_guard_inputs(self):
        """The eye guard in the head's current pose (it was fitted in the capture's pose), or None when off."""
        if self.eye_guard_control is None:
            return None
        _, guard = self.eye_guard_control.snapshot()
        if guard is None:
            self.eye_guard_frames = 0
            return None
        self.eye_guard_frames += 1
        turn, shift = getattr(self, "head_pose", (np.eye(3), np.zeros(3)))
        center = turn @ guard.placed_center() + shift
        axes = guard.axes @ turn.T
        ease = min(1.0, self.eye_guard_frames / max(self.EYE_GUARD_EASE * self.fps, 1.0))
        return [wp.vec3(*[float(x) for x in center]), wp.mat33(*[float(x) for x in axes.ravel()]),
                wp.vec3(*[float(x) for x in guard.radii()]), float(ease)]

    def update_controls(self):
        gravity = self.gravity_vector()
        supported = self.supported(gravity)
        self.gravity.assign(supported.astype(np.float32).reshape(1, 3))
        if getattr(self, "loose", None) is not None:
            self.unsupported = wp.vec3(*[float(x) for x in (gravity - supported) * self.weight])
        self.forces.update(self.wind_control, self.water_control, gravity, self.frame / self.fps)
        if self.grab_control is not None:
            revision, grab = self.grab_control.snapshot()
            if revision != self.grab_revision:
                targets = np.zeros((self.particle_count, 3), np.float32)
                weights = np.zeros(self.particle_count, np.float32)
                if grab is not None:
                    indices, positions, strength = grab
                    targets[indices] = positions
                    weights[indices] = strength
                self.targets.assign(targets)
                self.weights.assign(weights)
                self.grab_revision = revision
                self.grabbing = grab is not None

    def step(self):
        """Advance one frame (several substeps) on the GPU."""
        n = self.particle_count
        with wp.ScopedDevice(self.device):
            poses = self.move_avatar() if self.motion is not None else None
            if poses is not None and getattr(self, "bound_count", 0):
                wp.copy(self.bound_start, self.bound_end)
                wp.launch(eval_surface, self.bound_count, inputs=[self.bound_mesh.id, self.bound_face, self.bound_bary,
                                                                  self.bound_local, self.bound_end])
            self.update_controls()
            eye_guard = self.eye_guard_inputs()
            for substep in range(self.substeps):
                self.apply_pull_holds()
                if self.trapped_contact is not None:
                    self.trapped_contact.update(self.pos, self.prev, self.vel, self.inv)
                if poses is not None and self.kin_count:
                    (r0, t0), (r1, t1) = poses
                    wp.launch(move_kinematic, dim=self.kin_count,
                              inputs=[self.pos, self.prev, self.kin_index, self.kin_base, wp.mat33(*r0.ravel()),
                                      wp.vec3(*t0), wp.mat33(*r1.ravel()), wp.vec3(*t1),
                                      (substep + 1.0) / self.substeps])
                if poses is not None and getattr(self, "bound_count", 0):
                    wp.launch(move_bound, self.bound_count, inputs=[self.pos, self.prev, self.bound_index,
                                                                   self.bound_start, self.bound_end,
                                                                   (substep + 1.0) / self.substeps])
                if self.forces.active:
                    self.forces.compute(self.pos, self.vel, self.inv, self.dt)
                else:
                    self.forces.accel.zero_()
                    self.forces.wetness.zero_()
                if getattr(self, "loose", None) is not None:
                    wp.launch(add_unsupported, dim=n, inputs=[self.forces.accel, self.loose, self.unsupported])
                wp.launch(integrate, dim=n, inputs=[self.pos, self.prev, self.vel, self.inv, self.gravity,
                                                    self.forces.accel, self.weight, self.dt, self.damping])
                wp.launch(solve_strands, dim=self.strands,
                          inputs=[self.pos, self.inv, self.start, self.count, self.seg, self.bend, self.arc,
                                  self.span3, self.span5, self.anchor, self.anchor_len, self.lift_own,
                                  self.bend_strength,
                                  self.spray_strength, self.lift_strength, self.tether_slack, self.iterations])
                if self.stiffness is not None:
                    easing = self.momentum_speed if (self.motion is not None and self.playing) else None
                    self.stiffness.ease(self.vel, self.pos, self.prev, self.dt, easing)
                    for _ in range(self.iterations):
                        self.stiffness.project(self.pos, self.inv)
                if self.collision is not None:
                    self.collision.project(self.pos, self.inv)
                wp.launch(style_hold, dim=n, inputs=[self.pos, self.rest_particles, self.inv, self.hold])
                if getattr(self, "grabbing", False):
                    wp.launch(project_grab, dim=n, inputs=[self.pos, self.inv, self.targets, self.weights,
                                                           self.pull_step])
                if len(self.child):
                    for _ in range(2):
                        wp.copy(self.attachment_source, self.pos)
                        wp.launch(solve_attachments, dim=len(self.child),
                                  inputs=[self.pos, self.attachment_source, self.inv, self.child, self.parent, self.gap])
                for mesh, thin_surface in zip(self.meshes, self.thin_surfaces):
                    if not thin_surface:
                        wp.launch(collide_avatar, dim=n, inputs=[self.pos, self.prev, self.inv, mesh.id, self.radius,
                                                                 self.friction])
                if eye_guard is not None:
                    wp.launch(collide_eye_guard, dim=n, inputs=[self.pos, self.prev, self.inv] + eye_guard)
                for mesh, gaps, _ in self.shells:
                    turn,shift = self.shell_delta[mesh.id] if substep == 0 else (np.eye(3),np.zeros(3))
                    wp.launch(shell_particles,n,inputs=[self.pos,self.prev,self.inv,mesh.id,gaps,
                                                       wp.mat33(*turn.ravel()),wp.vec3(*shift)])
                if self.trapped_contact is not None:
                    self.trapped_contact.retain(self.pos, self.prev, self.vel)
                self.apply_pull_holds()
                wp.launch(update_velocity, dim=n, inputs=[self.pos, self.prev, self.vel, self.inv, self.dt, MAX_SPEED])
        if self.clumping > 0 or self.head_drag > 0:
            self.coherent_motion(poses)
        self.frame += 1
        if self.lift > 0 and self.frame % 4 == 1:
            self.update_weightless()
        if self.shells:
            with wp.ScopedDevice(self.device):
                self.rebuild_output()

    def coherent_motion(self, poses, cell=2.0):
        """Once per frame: head drag and clumping on the velocity relative to the head (see __init__)."""
        free = getattr(self, "_free", None)
        if free is None:
            free = self._free = self.inv.numpy() > 0
        pos, vel = self.pos.numpy(), self.vel.numpy()
        if poses is not None:
            (r0, t0), (r1, t1) = poses
            local = (pos - t1) @ r1
            head = ((local @ r1.T + t1) - (local @ r0.T + t0)) * self.fps
        else:
            head = np.zeros_like(vel)
        relative = vel - head
        if self.head_drag > 0:
            relative *= np.exp(-self.head_drag / self.fps)
        if self.clumping > 0:
            keys = np.floor(pos / cell).astype(np.int64)
            _, owner = np.unique(keys, axis=0, return_inverse=True)
            owner = owner.ravel()
            sums = np.zeros((owner.max() + 1, 3)); counts = np.zeros(owner.max() + 1)
            np.add.at(sums, owner, relative); np.add.at(counts, owner, 1)
            relative += self.clumping * ((sums / counts[:, None])[owner] - relative)
        vel[free] = (head + relative)[free]
        with wp.ScopedDevice(self.device):
            self.vel.assign(vel.astype(np.float32))

    def reset(self, points=None):
        """Put every particle back (default: the capture) at rest."""
        self.wrap = None
        points = self.captured if points is None else np.asarray(points, np.float32)
        self.release_pull_holds()
        self.pos.assign(points); self.prev.assign(points); self.vel.zero_()
        if self.trapped_contact is not None:
            self.trapped_contact.reset(self.inv)
        self.previous_output.assign(self.model.rest.astype(np.float32))
        self.output_frame = -1
        if self.stiffness is not None:
            self.stiffness.reset()
        if getattr(self, "lift", 0) > 0:
            self.last_share = None
        if getattr(self, "motion", None) is not None:
            self.time, self.playing = 0.0, False
            self.motion_elapsed = 0.0
            with wp.ScopedDevice(self.device):
                self.move_avatar()
            self.sync_bound()

    def apply_pull_holds(self):
        if not self.pull_holds:
            return
        r, t = self.head_pose
        indices = np.array(list(self.pull_holds), np.int32)
        local = np.asarray(list(self.pull_holds.values()), np.float32)
        wp.launch(pin_pulls, len(indices), inputs=[self.pos,self.prev,self.vel,self.inv,
                  wp.array(indices,dtype=int),wp.array(local,dtype=wp.vec3),wp.mat33(*r.ravel()),wp.vec3(*t)])

    def release_pull_holds(self):
        self.pull_holds.clear()
        inv = self.original_inv.numpy()
        if self.trapped_contact is not None:
            inv[self.trapped_contact.owner.numpy() >= 0] = 0.
        self.inv.assign(inv)

    def hold_pull(self, indices):
        r, t = self.head_pose
        points = self.pos.numpy()
        base = self.original_inv.numpy()
        for i in indices:
            if base[i] > 0:
                self.pull_holds[int(i)] = (points[i] - t) @ r
        if self.trapped_contact is not None:
            owner = self.trapped_contact.owner.numpy(); owner[list(self.pull_holds)] = -1
            self.trapped_contact.owner.assign(owner)
        self.apply_pull_holds()

    def edit_pull(self, indices, targets, weights):
        """Relax a paused drag spatially, without forces, style restoration or time."""
        with wp.ScopedDevice(self.device):
            indices = np.asarray(indices, int)
            owner = np.repeat(np.arange(self.strands), self.model.particle_count)
            affected = np.isin(owner, np.unique(owner[indices]))
            base = self.original_inv.numpy()
            for i in indices:
                self.pull_holds.pop(int(i), None)
            edit_inv = base * affected
            edit_inv[list(self.pull_holds)] = 0.
            # A deliberate edit supersedes trapped contact on the edited strands.
            if self.trapped_contact is not None:
                contact = self.trapped_contact.owner.numpy(); contact[affected] = -1
                self.trapped_contact.owner.assign(contact)
            self.inv.assign(edit_inv)
            goals = np.zeros((self.particle_count, 3), np.float32)
            influence = np.zeros(self.particle_count, np.float32)
            goals[indices] = targets; influence[indices] = np.asarray(weights) * .2
            goal_gpu = wp.array(goals, dtype=wp.vec3)
            weight_gpu = wp.array(influence, dtype=float)
            points = self.pos.numpy()
            bend = np.zeros(self.particle_count, np.float32)
            for start, count in zip(self.model.particle_start, self.model.particle_count):
                bend[start:start+count-2] = np.linalg.norm(points[start+2:start+count] - points[start:start+count-2], axis=1)
            edit_bend = wp.array(bend, dtype=float)
            for _ in range(24):
                wp.launch(project_grab, self.particle_count, inputs=[self.pos, self.inv, goal_gpu, weight_gpu, 10.])
                wp.launch(solve_strands, self.strands, inputs=[self.pos,self.inv,self.start,self.count,self.seg,
                          edit_bend,self.arc,self.span3,self.span5,self.anchor,self.anchor_len,self.lift_own,
                          .05,0.,0.,self.tether_slack,2])
                if len(self.child):
                    wp.copy(self.attachment_source,self.pos)
                    wp.launch(solve_attachments,len(self.child),inputs=[self.pos,self.attachment_source,self.inv,
                              self.child,self.parent,self.gap])
                for mesh, thin in zip(self.meshes, self.thin_surfaces):
                    if not thin:
                        wp.launch(collide_avatar,self.particle_count,inputs=[self.pos,self.prev,self.inv,mesh.id,self.radius,0.])
                for mesh, gaps, _ in self.shells:
                    wp.launch(shell_particles,self.particle_count,inputs=[self.pos,self.prev,self.inv,mesh.id,gaps,
                              wp.mat33(*np.eye(3).ravel()),wp.vec3(0.)])
            wp.launch(finish_edit,self.particle_count,inputs=[self.pos,self.prev,self.vel,
                      wp.array(affected.astype(np.int32),dtype=int)])
            current = base.copy()
            if self.trapped_contact is not None:
                current[self.trapped_contact.owner.numpy() >= 0] = 0.
            self.inv.assign(current)
            self.apply_pull_holds()
            self.output_frame = -1

    def balance(self, rounds=6, frames=90, rate=0.6, progress=None):
        """Gravity-balanced rest shape: the captured style becomes where the hair settles under
        gravity, instead of sagging further. Needs stiffness. Returns the mean leftover drift (cm)."""
        if self.stiffness is None:
            return None
        # The capture hung under ordinary gravity: balance with that alone (no wind, water or pulling).
        saved = (self.gravity_control, self.wind_control, self.water_control, self.grab_control,
                 getattr(self, "grabbing", False), self.gravity_scale)
        self.gravity_control = self.wind_control = self.water_control = self.grab_control = None
        self.grabbing, self.gravity_scale = False, 1.0
        rest = self.captured.copy()
        best = None
        for r in range(rounds):
            self.reset()
            with wp.ScopedDevice(self.device):
                # Settle, then average the last third so a strand still swinging counts at its middle.
                settled = np.zeros_like(rest)
                tail = max(1, frames // 3)
                for f in range(frames):
                    self.step()
                    if f >= frames - tail:
                        settled += self.pos.numpy()
                settled /= tail
            error = self.captured - settled
            drift = float(np.linalg.norm(error, axis=1).mean())
            if progress is not None:
                progress(r + 1, rounds, drift)
            if best is None or drift < best[0]:
                best = (drift, rest)
            else:
                break                     # got worse: keep the best shape found
            if r == rounds - 1:
                break
            # Hair resting on the body is held up by it, not by its own stiffness: leave it be,
            # or the rest shape is pushed into the body and the strand springs off it.
            contact = wp.zeros(self.particle_count, dtype=int, device=self.device)
            for mesh in self.meshes:
                wp.launch(touching, dim=self.particle_count, inputs=[self.pos, mesh.id, self.radius * 2 + 0.3, contact],
                          device=self.device)
            error[contact.numpy() > 0] = 0.0
            rest = self._rechain(rest + rate * error)
            self.set_rest(rest)
        drift, rest = best
        self.set_rest(rest)
        (self.gravity_control, self.wind_control, self.water_control, self.grab_control,
         self.grabbing, self.gravity_scale) = saved
        self.forces.revisions = [None, None]
        self.reset()
        self.frame = 0
        return drift

    def setup_weightless(self, lift, lean_degrees=12.0, seed=7):
        """Weightless shape: without weight, hair relaxes toward leaving its root straight out
        from the head (90 degrees to the scalp), keeping its own curves. Each strand is turned
        toward "out from the head" (with a small random lean, so it is not a dandelion) by
        lift x weightless share, segment by segment, so a half-submerged strand only blooms
        where it is under water. Tied hair turns about its tie, so it stays tied."""
        self.lift = float(min(max(lift, 0.0), 1.0))
        self.last_share = None
        if self.lift <= 0:
            return
        model = self.model
        start, count = model.particle_start, model.particle_count
        roots = self.captured[start]
        scalp = roots[model.root_kind == 0]
        if len(scalp) >= 8:
            fit = np.linalg.lstsq(np.c_[2 * scalp, np.ones(len(scalp))], (scalp ** 2).sum(1), rcond=None)[0]
            centre = fit[:3]
        else:
            centre = roots.mean(0) - self.style_down * 10.0
        out = roots - centre
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        rng = np.random.default_rng(seed)
        jitter = rng.normal(size=out.shape)
        jitter -= out * (jitter * out).sum(1, keepdims=True)
        jitter /= np.maximum(np.linalg.norm(jitter, axis=1, keepdims=True), 1e-9)
        lean = np.radians(lean_degrees) * rng.random(len(out))[:, None]
        out = out * np.cos(lean) + jitter * np.sin(lean)
        first = self.captured[np.minimum(start + 1, len(self.captured) - 1)] - roots
        first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-9)
        axis = np.cross(first, out)
        size = np.linalg.norm(axis, axis=1, keepdims=True)
        angle = np.arctan2(size[:, 0], (first * out).sum(1))
        axis = np.where(size > 1e-9, axis / np.maximum(size, 1e-9), np.array([1.0, 0.0, 0.0]))
        owner = np.repeat(np.arange(model.strand_count), count)
        self.turn_axis, self.turn_angle = axis[owner], angle[owner]
        self.owner_of = owner
        self.segments = np.diff(self.captured, axis=0, append=self.captured[-1:])
        is_last = np.zeros(len(self.captured), bool)
        is_last[start + count - 1] = True
        self.segments[is_last] = 0.0
        self.strand_first = start

    def weightless_share(self):
        """Per point: 0 at full weight (1 g or more), 1 when weightless (or floating)."""
        g = float(np.linalg.norm(self.gravity_vector()))
        wet = self.forces.wetness.numpy() if self.forces.active else 0.0
        buoyancy = float(self.forces.water.numpy()[2]) if self.forces.active else 0.0
        weight = g * np.maximum(1.0 - buoyancy * wet, 0.0)
        share = np.clip(1.0 - weight / -GRAVITY, 0.0, 1.0)
        return np.broadcast_to(share, (self.particle_count,)).astype(np.float32)

    def update_weightless(self, force=False):
        share = self.weightless_share()
        if not force and self.last_share is not None and np.abs(share - self.last_share).max() < 0.02:
            return
        self.last_share = share.copy()
        # Turn each segment by its share of the strand's turn (Rodrigues), then chain from the root.
        theta = (self.turn_angle * self.lift * share)[:, None]
        k, v = self.turn_axis, self.segments
        turned = v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * (k * v).sum(1, keepdims=True) * (1 - np.cos(theta))
        cumulative = np.cumsum(turned, axis=0)
        base = np.repeat(cumulative[self.strand_first] - turned[self.strand_first], self.model.particle_count, axis=0)
        offset = np.concatenate([[np.zeros(3)], cumulative[:-1]]) - base
        roots = np.repeat(self.captured[self.strand_first], self.model.particle_count, axis=0)
        rest = (roots + offset).astype(np.float32)
        self.set_rest(rest)
        # The style hold (near the roots) gives way as hair becomes weightless.
        self.hold.assign((self.base_hold * (1.0 - share)).astype(np.float32))

    def set_rest(self, points):
        """Natural (weightless) shape: stiffness sections and the bend, hairspray and root-lift
        lengths follow it. Segment lengths, the scalp hold and the style hold stay as captured."""
        start, count = self.model.particle_start, self.model.particle_count
        points = np.asarray(points, np.float32)
        index = np.arange(len(points))
        last = np.repeat(start + count - 1, count)
        def span(k):
            out = np.zeros(len(points), np.float32)
            ok = index + k <= last
            out[ok] = np.linalg.norm(points[index[ok] + k] - points[index[ok]], axis=1)
            return out
        self.bend.assign(span(2)); self.span3.assign(span(3)); self.span5.assign(span(5))
        anchors = self.anchor0
        lift = np.linalg.norm(points - np.repeat(anchors, count, axis=0), axis=1).astype(np.float32)
        loose = getattr(self, "loose_points", None)
        if loose is not None:                       # grown hair keeps no bend, and root lift only from its area
            bend = span(2); bend[loose] = -1.; self.bend.assign(bend)
            lift[getattr(self, "loose_unlifted", loose)] = -1.
        self.anchor_len.assign(lift)
        if self.stiffness is not None:
            self.stiffness.set_rest(points)

    def _rechain(self, points):
        """Keep each strand's captured segment lengths, following the new point directions."""
        start, count = self.model.particle_start, self.model.particle_count
        seg = self.seg.numpy()
        out = points.copy()
        out[start] = self.captured[start]
        for k in range(1, int(count.max())):
            alive = count > k
            i = start[alive] + k
            d = points[i] - out[i - 1]
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            out[i] = out[i - 1] + d * seg[i - 1][:, None]
        return out.astype(np.float32)

    def rebuild_output(self):
        """Update shell contact once per simulated frame, independent of preview rate."""
        wp.launch(transport_frames, dim=self.strands,
                  inputs=[self.pos, self.rest_frames, self.start, self.count, self.frames])
        wp.launch(rebuild, dim=len(self.out), inputs=[self.pos, self.frames, self.rest, self.segment, self.t,
                                                      self.offset, self.moving, self.out])
        if self.frame > 0:
            for mesh, _, gaps in self.shells:
                turn,shift = self.shell_delta[mesh.id]
                wp.launch(shell_vertices,len(self.out),inputs=[self.out,self.previous_output,self.moving,mesh.id,gaps,
                                                             wp.mat33(*turn.ravel()),wp.vec3(*shift)])
        wp.copy(self.previous_output,self.out)
        self.output_frame = self.frame

    def vertices(self):
        """Rebuilt hair vertices (cm), same order as the captured mesh."""
        with wp.ScopedDevice(self.device):
            if not self.shells or self.output_frame != self.frame:
                self.rebuild_output()
            result = self.out.numpy()
        if self.motion is not None:
            # The hair's scalp mesh (and other unbound pieces) rides with the head.
            r, t = self.head_pose
            still = self.model.strand_of_vertex < 0
            result[still] = self.model.rest[still] @ r.T + t
        if not np.isfinite(result).all():
            raise RuntimeError("The hair became unstable. Try more damping or a larger collision thickness.")
        return result.astype(float)

    def particles(self):
        return self.pos.numpy().astype(float)

    def vertex_normals(self, faces):
        """Smooth vertex normals of the last rebuilt mesh, computed on the GPU (for the preview)."""
        with wp.ScopedDevice(self.device):
            if getattr(self, "faces", None) is None:
                self.faces = wp.array(np.asarray(faces, np.int32), dtype=int)
                self.normals = wp.zeros(len(self.out), dtype=wp.vec3)
            self.normals.zero_()
            wp.launch(face_normals, dim=self.faces.shape[0], inputs=[self.out, self.faces, self.normals])
            wp.launch(normalize, dim=len(self.normals), inputs=[self.normals])
            return self.normals.numpy()
