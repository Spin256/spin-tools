# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Extract authored deformations in the frame of the nearby attached surface."""
import heapq

import numpy as np


class MotionExample:
    def __init__(self, spec, motion, mobility, external, neighbours, edges, faces, feather, assigned):
        self.kind, mask, reference, samples = spec
        painted = np.asarray(mask, dtype=bool)
        mask = painted & (mobility > 0)
        self.indices = np.flatnonzero(mask)
        adjacent = neighbours[self.indices]
        anchors = np.unique(adjacent[adjacent >= 0])
        self.anchors = anchors[~assigned[anchors]]
        original = motion.body_positions(reference)
        last = getattr(motion, 'intervals', None)
        inside_timeline = sorted({int(s) for s in samples if s != reference and last is not None and 0 <= s <= last})
        if self.kind == 'Bounce' and inside_timeline:
            self.anchors = self.smooth_anchors(motion, last, inside_timeline, neighbours, mask, assigned)
        anchor_points = original[self.anchors]
        if len(anchor_points) < 3:
            raise ValueError('Each example region needs nearby unpainted mesh to anchor it.')
        self.center = anchor_points.mean(axis=0)
        self.base = anchor_points - self.center
        if np.linalg.matrix_rank(self.base) < 2:
            raise ValueError('Leave a wider unpainted attachment around the example region.')
        self.closed = original[self.indices] - self.center
        self.faces = faces[np.any(mask[faces], axis=1)]
        self.face_slots = mask[self.faces]
        self.face_vertices = np.searchsorted(self.indices, self.faces[self.face_slots])
        self.reference_area = self.triangle_area(original[self.faces])
        self.last_normal = None            # each triangle's facing on the previous frame, attached frame

        # Fade the override inside the paint. Unpainted positions remain exact.
        weights = mask.astype(np.float32)
        frontier = ~mask
        for ring in range(feather):
            next_ring = np.any((neighbours >= 0) & frontier[neighbours], axis=1) & ~frontier
            weights[next_ring] = (ring + 1.) / (feather + 1.)
            frontier |= next_ring
        self.weight = weights[self.indices, None]
        if self.kind in ('Bounce', 'Shaped'):
            # Motion-driven: the reference is rest, each example frame a shape to blend toward.
            # Order: rest first, then the examples as given.
            self.times = [reference] + [s for s in dict.fromkeys(samples) if s != reference]
        else:
            self.times = sorted(set((reference, *samples))) if self.kind == 'Keyed' else list(samples)
        self.shapes = []
        for sample in self.times:
            pose = motion.body_positions(sample)
            rotation, center = self.attachment(pose)
            self.shapes.append((pose[self.indices] - center) @ rotation.T)
        # Fold limits learned from the examples: each triangle may shrink to a little
        # below the smallest size it has in any example frame (0.8 of it), between 2% and 25% of its reference
        # size. Breast bases squashed by an authored overshoot pass; crotch folds the examples keep wide stay wide.
        smallest = np.ones(len(self.faces))
        reference_area = np.maximum(self.reference_area, 1.e-20)
        for sample in self.times:
            area = self.triangle_area(motion.body_positions(sample)[self.faces])
            smallest = np.minimum(smallest, area / reference_area)
        self.limit_ratio = np.clip(.8 * smallest, .02, .25)
        if self.kind == 'Opening':
            if np.max(np.linalg.norm(self.shapes[0] - self.closed, axis=1)) < 1.e-6:
                raise ValueError('The Opening example must move the painted region relative to its attachment.')
            contact = mask & (external > 0)
            self.probe_vertices = np.flatnonzero(contact[self.indices])
            contact_edges = edges[np.all(contact[edges], axis=1)]
            self.probe_edges = np.searchsorted(self.indices, contact_edges)
            if not len(self.probe_vertices):
                ignoring = int(np.count_nonzero(mask & ~(external > 0)))
                raise ValueError(
                    f'The Opening region has no soft vertex that can touch the collider: {int(painted.sum())} '
                    f'vertices in the region, {int(mask.sum())} soft, {ignoring} of those ignoring collider '
                    'contact. Choose the region, then Restore collider contact on selection and Apply softness '
                    'to selection.')
            self.amount = np.zeros(len(self.indices), dtype=np.float32)
            painted_edges = edges[np.all(mask[edges], axis=1)]
            self.links = np.searchsorted(self.indices, painted_edges).T
            left, right = self.links
            self.edge_lengths = np.linalg.norm(self.closed[right] - self.closed[left], axis=1)
            self.drag_neighbours = [[] for _ in self.indices]
            for a, b, length in zip(left.tolist(), right.tolist(), self.edge_lengths.tolist()):
                self.drag_neighbours[a].append((b, length))
                self.drag_neighbours[b].append((a, length))
            self.drag_contact = np.zeros(len(self.probe_vertices) + len(self.probe_edges), dtype=bool)
            self.lip_drag = np.zeros((len(self.indices), 3), dtype=np.float32)

    def smooth_anchors(self, motion, last, examples, neighbours, mask, assigned):
        """Bounce attachment from a skin ring that the example poses do not jolt.

        Example poses keyed into the timeline also reshape skin just outside the painted area, so the nearest
        ring can jolt on those frames, which the bounce spring would read as a kick and the shape would follow.
        Rings are taken outward (up to 10);
        the first whose center jolt around the example frames is within 15% of the smoothest farther ring is
        used. The body's real motion on those frames (a stop and reversal) is kept. Opening keeps its own."""
        frames = [motion.body_positions(frame) for frame in range(last + 1)]
        window = range(max(1, examples[0] - 1), min(last, examples[-1] + 2))
        if not len(window):
            return self.anchors
        inside = mask.copy()
        ring = self.anchors
        rings, jolts = [], []
        for _ in range(10):
            if len(ring) < 3:
                break
            centers = np.array([pose[ring].mean(axis=0) for pose in frames])
            rings.append(ring)
            jolts.append(max(np.linalg.norm(centers[f + 1] - 2. * centers[f] + centers[f - 1]) for f in window))
            inside[ring] = True
            following = neighbours[ring]
            following = np.unique(following[following >= 0])
            ring = following[~inside[following] & ~assigned[following]]
        for index, jolt in enumerate(jolts):
            if jolt <= 1.15 * min(jolts[index:]) + 1.e-9:
                return rings[index]
        return self.anchors

    def setup_bounce(self, fps, frequency, damping, inertia, motion=None):
        """Example shapes as directions: each example's offset from rest (in the attached frame) and its
        weighted mean direction. A spring in the attached frame, driven by the attachment's acceleration,
        is then expressed as a non-negative mix of those directions (so left > rest > right blends)."""
        self.fps = float(fps)
        self.rest = self.shapes[0]
        self.deltas = [shape - self.rest for shape in self.shapes[1:]]
        weight = self.weight[:, 0]
        directions = [(delta * weight[:, None]).sum(axis=0) / max(float(weight.sum()), 1.e-9) for delta in self.deltas]
        keep = [k for k, d in enumerate(directions) if np.linalg.norm(d) > 1.e-5]
        if not keep:
            raise ValueError('Bounce examples must move the painted region away from its reference frame.')
        self.deltas = [self.deltas[k] for k in keep]
        self.directions = np.array([directions[k] for k in keep]).T          # (3, K)
        self.omega = 2. * np.pi * float(frequency)
        self.damping = float(damping)
        self.inertia = float(inertia)
        self.lag = np.zeros(3)
        self.lag_velocity = np.zeros(3)
        self.track = None
        if motion is not None:
            self.fit_inertia(motion)

    def fit_inertia(self, motion):
        """Scale the swing so its strong moments reach the examples instead of resting against their limit."""
        reach = float(np.linalg.norm(self.directions, axis=0).max())
        sizes = []
        for sample in range(motion.intervals + 1):
            for part in range(1, 5) if sample else (4,):
                blend = part / 4.
                pose = motion.body_positions(sample)
                if sample:
                    pose = motion.body_positions(sample - 1) * (1. - blend) + pose * blend
                self.advance_lag(sample - 1. + blend, self.attachment(pose))
                sizes.append(float(np.linalg.norm(self.lag)) / reach)
        self.inertia /= max(1., float(np.percentile(sizes, 95)))
        self.lag, self.lag_velocity, self.track = np.zeros(3), np.zeros(3), None

    def setup_shaped(self, motion, fps):
        """Opening with several example frames: each frame's collider state
        relative to the opening (where the collider is in the opening's attached frame, and which way it
        moves) is stored with that frame's shape; at run time the closest states are blended."""
        self.fps = float(fps)
        self.motion = motion
        reference = motion.body_positions(self.times[0])
        center = reference[self.indices].mean(axis=0)
        distances = {i: float(np.linalg.norm(motion.positions(i, self.times[0]) - center, axis=1).min())
                     for i in motion.samples}
        if not distances:
            raise ValueError('An Opening with several example frames needs a moving collider.')
        self.collider = min(distances, key=distances.get)
        radius = float(np.linalg.norm(reference[self.indices] - center, axis=1).max())
        self.focus_scale = max(radius * 1.5, 1.e-3)
        self.velocity_time = .05                     # seconds: how much the in/out direction counts
        last = motion.intervals

        def state(sample):
            pose = motion.body_positions(sample)
            attachment = self.attachment(pose)
            point = self.collider_point(motion.positions(self.collider, sample), pose, attachment)
            a, b = max(0, sample - 1), min(last, sample + 1)
            span = (b - a) / self.fps
            velocity = np.zeros(3)
            if span > 0:
                velocity = (self.collider_point(motion.positions(self.collider, b), motion.body_positions(b), attachment) -
                            self.collider_point(motion.positions(self.collider, a), motion.body_positions(a), attachment)) / span
            # Example frames are often next to unrelated example poses in the timeline: a jump between
            # them is not collider motion, so only plausible speeds count.
            if np.linalg.norm(velocity) > 1.5:
                velocity = np.zeros(3)
            return np.concatenate([point, velocity * self.velocity_time])
        self.states = np.array([state(sample) for sample in self.times])
        self.states[0, 3:] = 0.                      # the reference (closed) is the collider at rest
        # Each example reaches about 0.6 of the way to its nearest other example (adaptive blending
        # radius), so a state close to one example takes that shape and states in between blend.
        gaps = np.linalg.norm(self.states[:, None] - self.states[None], axis=2)
        np.fill_diagonal(gaps, np.inf)
        nearest = gaps.min(axis=1)
        self.state_scale = np.where(np.isfinite(nearest) & (nearest > 1.e-9), nearest * .6, 1.)
        self.current_colliders = None
        self.previous_state = None

    def collider_point(self, collider, pose, attachment):
        """The part of the collider at the opening (proximity-weighted), in the opening's attached frame."""
        rotation, center = attachment
        opening = pose[self.indices].mean(axis=0)
        collider = np.asarray(collider, dtype=float)
        distance = np.linalg.norm(collider - opening, axis=1)
        weights = np.exp(-(distance / self.focus_scale) ** 2)
        if weights.sum() < 1.e-12:
            weights = (distance <= distance.min() + 1.e-9).astype(float)
        point = (collider * weights[:, None]).sum(axis=0) / weights.sum()
        return (point - center) @ rotation.T

    def shaped_target(self, pose, sample, attachment):
        rotation, center = attachment
        collider = (self.current_colliders or {}).get(self.collider)
        if collider is None:
            collider = self.motion.positions(self.collider, int(round(sample)))
        point = self.collider_point(collider, pose, attachment)
        time = sample / self.fps
        if self.previous_state is None:
            velocity = np.zeros(3)
        elif time - self.previous_state[0] > 1.e-9:
            velocity = (point - self.previous_state[1]) / (time - self.previous_state[0])
        else:
            velocity = self.previous_state[2]
        if self.previous_state is None or time > self.previous_state[0] + 1.e-9:
            self.previous_state = (time, point, velocity)
        state = np.concatenate([point, velocity * self.velocity_time])
        distance = np.linalg.norm(self.states - state, axis=1)
        weights = np.exp(-(distance / self.state_scale) ** 2)
        if weights.sum() < 1.e-12:
            weights = (distance <= distance.min() + 1.e-12).astype(float)
        weights = weights / weights.sum()
        self.last_weights = weights
        shape = sum(w * s for w, s in zip(weights, self.shapes))
        target = shape @ rotation + center
        original = pose[self.indices]
        return original + self.weight * (target - original), None

    def bounce_weights(self, lag):
        """Non-negative example weights whose directions best add up to ``lag``.

        Solved exactly as a small strictly convex problem (least squares plus a light ridge, weights >= 0), so
        the weights change smoothly with the lag."""
        import itertools
        d = self.directions
        count = d.shape[1]
        gram = d.T @ d
        ridge = .02 * float(np.trace(gram)) / max(count, 1) + 1.e-18
        gram = gram + ridge * np.eye(count)
        right = d.T @ lag
        if count > 8:
            # Many examples: projected gradient on the same problem (still a unique, smooth answer).
            weights = np.zeros(count)
            step = 1. / float(np.linalg.eigvalsh(gram).max())
            for _ in range(400):
                weights = np.maximum(0., weights - step * (gram @ weights - right))
            return weights
        best, best_cost = np.zeros(count), 0.
        for size in range(1, count + 1):
            for subset in itertools.combinations(range(count), size):
                subset = list(subset)
                solved = np.linalg.solve(gram[np.ix_(subset, subset)], right[subset])
                if (solved < 0).any():
                    continue
                weights = np.zeros(count)
                weights[subset] = solved
                cost = .5 * weights @ gram @ weights - right @ weights
                if cost < best_cost:
                    best, best_cost = weights, cost
        return best

    @staticmethod
    def within_examples(weights):
        """Keep the blend between rest and the examples: the weights' sum approaches 1 smoothly from .7, so
        together they never move the area further than the example frames do."""
        total = float(weights.sum())
        if total <= .7:
            return weights
        return weights * ((.7 + .3 * (1. - np.exp(-(total - .7) / .3))) / total)

    def limit_lag(self, weights):
        """Stop the spring at the examples' reach, so it swings back as soon as the motion turns."""
        size = float(np.linalg.norm(self.lag))
        reach = float(np.linalg.norm(self.directions, axis=0).max())
        scale = max(float(weights.sum()), size / reach)
        if scale <= 1.:
            return weights
        outward = self.lag / size
        self.lag = self.lag / scale
        self.lag_velocity = self.lag_velocity - outward * max(0., float(self.lag_velocity @ outward))
        return self.bounce_weights(self.lag)

    def advance_lag(self, sample, attachment):
        rotation, center = attachment
        time = sample / self.fps
        if self.track is None:
            self.track = (time, center.copy(), np.zeros(3))
        last_time, last_center, last_velocity = self.track
        step = time - last_time
        if step > 1.e-9:
            velocity = (center - last_center) / step
            accel_local = ((velocity - last_velocity) / step) @ rotation.T   # world -> attached frame
            # The tissue lags opposite to the attachment's acceleration (a stable sub-stepped spring).
            count = max(1, int(np.ceil(step * max(self.omega, self.damping) * 4.)))
            h = step / count
            for _ in range(count):
                self.lag_velocity += (-self.omega * self.omega * self.lag - self.damping * self.lag_velocity
                                      - self.inertia * accel_local) * h
                self.lag += self.lag_velocity * h
            self.track = (time, center.copy(), velocity)

    def bounce_target(self, pose, sample, attachment):
        rotation, center = attachment
        self.advance_lag(sample, attachment)
        weights = self.within_examples(self.limit_lag(self.bounce_weights(self.lag)))
        shape = self.rest + sum(w * delta for w, delta in zip(weights, self.deltas) if w > 0)
        target = shape @ rotation + center
        original = pose[self.indices]
        return original + self.weight * (target - original), None

    def attachment(self, pose):
        anchors = pose[self.anchors]
        center = anchors.mean(axis=0)
        u, _, vt = np.linalg.svd(self.base.T @ (anchors - center))
        if np.linalg.det(u @ vt) < 0:
            u[:, -1] *= -1
        return u @ vt, center

    def target(self, pose, sample, attachment):
        rotation, center = attachment
        original = pose[self.indices]
        if self.kind == 'Opening':
            closed = self.closed @ rotation + center
            wide = self.shapes[0] @ rotation + center
            # The captured motion is the example, so mixing it into the closed
            # baseline would start opening before the collider reaches the lips.
            return closed, self.weight * (wide - closed)
        if self.kind == 'Bounce':
            return self.bounce_target(pose, sample, attachment)
        if self.kind == 'Shaped':
            return self.shaped_target(pose, sample, attachment)
        right = np.searchsorted(self.times, sample)
        if right == 0 or right == len(self.times):
            shape = self.shapes[0 if right == 0 else -1]
        else:
            left = right - 1
            t = (sample - self.times[left]) / (self.times[right] - self.times[left])
            ease = t * t * (3. - 2. * t)
            shape = self.shapes[left] * (1. - ease) + self.shapes[right] * ease
        target = shape @ rotation + center
        return original + self.weight * (target - original), None

    def probes(self, values):
        return np.concatenate((values[self.probe_vertices], values[self.probe_edges].mean(axis=1)),
                              dtype=np.float32)

    def update_opening(self, required, release):
        self.drag_contact = required > 0.
        amount = np.zeros_like(self.amount)
        count = len(self.probe_vertices)
        amount[self.probe_vertices] = required[:count]
        for endpoints in self.probe_edges.T:
            np.maximum.at(amount, endpoints, required[count:])

        # Carry some motion into neighbouring faces without pulling the full
        # region open. Spread new contact only, so it cannot creep over time.
        left, right = self.links
        for _ in range(3):
            adjacent = np.zeros_like(amount)
            np.maximum.at(adjacent, left, amount[right] * .55)
            np.maximum.at(adjacent, right, amount[left] * .55)
            np.maximum(amount, adjacent, out=amount)
        self.amount = np.maximum(amount, self.amount * release)
        return self.amount

    def round_folds(self, points, strength):
        left, right = self.links
        edge = points[right] - points[left]
        length = np.linalg.norm(edge, axis=1)
        squeeze = np.clip(2. * (1. - length / np.maximum(self.edge_lengths, 1.e-8)), 0., 1.)
        compressed = np.zeros(len(points), dtype=np.float32)
        np.maximum.at(compressed, left, squeeze)
        np.maximum.at(compressed, right, squeeze)
        correction = np.zeros_like(points)
        count = np.zeros(len(points), dtype=np.float32)
        np.add.at(correction, left, edge)
        np.add.at(correction, right, -edge)
        np.add.at(count, left, 1.)
        np.add.at(count, right, 1.)
        correction /= np.maximum(count, 1.)[:, None]
        correction *= (strength * compressed * np.minimum(1., self.amount * 4.))[:, None] * self.weight
        step = np.linalg.norm(correction, axis=1)
        return correction * np.minimum(1., .001 / np.maximum(step, 1.e-8))[:, None]

    def drag_targets(self, velocity, displacement, reach):
        """Move each contact patch together, fading by distance along its mesh."""
        active = np.linalg.norm(velocity, axis=1) > 1.e-6
        seeds = np.flatnonzero(active).tolist()
        owners = np.full(len(velocity), -1, dtype=np.intp)
        pulls = []
        for seed in seeds:
            if owners[seed] >= 0:
                continue
            owners[seed] = len(pulls)
            group = [seed]
            for vertex in group:
                for neighbour, _ in self.drag_neighbours[vertex]:
                    if active[neighbour] and owners[neighbour] < 0:
                        owners[neighbour] = owners[seed]
                        group.append(neighbour)
            direction = velocity[group].mean(axis=0)
            pulls.append(direction * (displacement / max(.003, np.linalg.norm(direction))))

        target = np.zeros_like(velocity)
        if not seeds:
            return target
        distance = np.full(len(velocity), np.inf)
        distance[seeds] = 0.
        pending = [(0., seed) for seed in seeds]
        heapq.heapify(pending)
        while pending:
            length, vertex = heapq.heappop(pending)
            if length != distance[vertex]:
                continue
            for neighbour, edge_length in self.drag_neighbours[vertex]:
                next_length = length + edge_length
                if next_length < reach and next_length < distance[neighbour]:
                    distance[neighbour] = next_length
                    owners[neighbour] = owners[vertex]
                    heapq.heappush(pending, (next_length, neighbour))
        reached = owners >= 0
        falloff = 1. - distance[reached] / reach if reach > 0. else np.ones(len(seeds))
        target[reached] = np.asarray(pulls)[owners[reached]] * (falloff**2 * (3. - 2.*falloff))[:, None]
        return target * self.weight

    @staticmethod
    def triangle_area(triangles):
        a, b, c = triangles.transpose(1, 0, 2)
        return np.linalg.norm(np.cross(b - a, c - a), axis=1)

    def accept_target(self, pose, target, attachment):
        """Hold triangles at their fold limit instead of falling back to an older shape. Only the region vertices of triangles below their limit (or turned over relative to the body
        pose) are moved, by the least amount that restores the limit; everything else follows the example.
        Nothing is stored between frames, so there is no lag to release later. Returns the target and the
        number of triangles held."""
        triangles = pose[self.faces]
        pose_cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        pose_area = np.linalg.norm(pose_cross, axis=1)
        normal = pose_cross / np.maximum(pose_area, 1.e-20)[:, None]
        # Never ask for more than the body pose itself has (thin faces already in the DAZ pose stay usable).
        minimum = np.minimum(self.limit_ratio * self.reference_area, .9 * pose_area)
        movable = self.face_slots.astype(float)
        positions = triangles.copy()

        # Facing is judged against each triangle's own facing on the previous frame (in the attached frame),
        # not the closed DAZ pose: an Opening turns its lips far from that pose on purpose, and checking
        # against it would push openings shut. Only the facing is carried, never the shape.
        rotation = attachment[0]
        if self.last_normal is not None:       # first frame: the DAZ pose (the rest shape there)
            normal = self.last_normal @ rotation

        def signed(points, faces=slice(None)):
            return .5 * np.einsum('ij,ij->i', np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
                                  normal[faces])

        def place(local):
            positions[self.face_slots] = local[self.face_vertices]
            return signed(positions) < .5 * minimum

        local = np.array(target, dtype=float)
        bad = place(local)
        held = int(np.count_nonzero(bad))
        if not held:
            self.remember_facing(positions, normal, rotation)
            return target, 0
        for _ in range(40):
            if not bad.any():
                break
            p = positions[bad]
            n = normal[bad]
            grad = .5 * np.stack([np.cross(n, p[:, 2] - p[:, 1]), np.cross(n, p[:, 0] - p[:, 2]),
                                  np.cross(n, p[:, 1] - p[:, 0])], axis=1) * movable[bad][:, :, None]
            gap = .5 * minimum[bad] * 1.05 - signed(p, bad)
            scale = gap / np.maximum((grad * grad).sum(axis=(1, 2)), 1.e-30)
            moves = grad * scale[:, None, None]
            change = np.zeros_like(local)
            count = np.zeros(len(local))
            slots = self.face_slots[bad]
            np.add.at(change, self.face_vertices_of(bad), moves[slots])
            np.add.at(count, self.face_vertices_of(bad), 1.)
            local += change / np.maximum(count, 1.)[:, None]
            bad = place(local)
        if bad.any():
            # Too deep to open by pushing (turned far over): move those corners only as far from this frame's
            # DAZ pose toward the pushed position as stays within the limit. The fraction changes smoothly with
            # the target and starts from the current pose, so it neither snaps nor lags behind the body.
            affected = np.unique(self.face_vertices_of(bad))
            source = pose[self.indices][affected].astype(float)
            wanted = local[affected].copy()
            low, high = 0., 1.
            for _ in range(16):
                middle = (low + high) * .5
                local[affected] = source + (wanted - source) * middle
                if place(local).any():
                    high = middle
                else:
                    low = middle
            local[affected] = source + (wanted - source) * low
        place(local)
        self.remember_facing(positions, normal, rotation)
        return local.astype(np.asarray(target).dtype), held

    def remember_facing(self, positions, normal, rotation):
        own = np.cross(positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0])
        size = np.linalg.norm(own, axis=1)
        facing = np.where(size[:, None] > 1.e-12, own / np.maximum(size, 1.e-30)[:, None], normal)
        self.last_normal = facing @ rotation.T

    def face_vertices_of(self, faces):
        """Region-vertex numbers (into self.indices) of the painted corners of the given faces."""
        return np.searchsorted(self.indices, self.faces[faces][self.face_slots[faces]])
