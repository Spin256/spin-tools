# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""GPU cloth simulation starting from the DAZ world-space OBJ positions."""

from pathlib import Path
import math

import numpy as np
from drape_timing import DrapeSchedule, DrapeStep


def read_mesh(path, keep_degenerate=False):
    vertices = []
    triangles = []
    with open(path, encoding="utf-8-sig") as source:
        for line in source:
            fields = line.partition("#")[0].split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "f":
                indices = [int(field.split("/")[0]) for field in fields[1:]]
                face = [index - 1 if index > 0 else len(vertices) + index for index in indices]
                triangles.extend((face[0], a, b) for a, b in zip(face[1:], face[2:]))

    points = np.asarray(vertices, dtype=np.float32) * 0.01
    faces = np.asarray(triangles, dtype=np.int32)
    if not len(points) or not len(faces) or faces.min() < 0 or faces.max() >= len(points):
        raise ValueError(f"Invalid OBJ mesh: {path}")
    if keep_degenerate:
        # Raw triangle list: its topology does not depend on vertex positions.
        return points, faces
    a, b, c = points[faces].transpose(1, 0, 2)
    faces = faces[np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-12]
    if not len(faces):
        raise ValueError(f"No usable triangles in {path}")
    return points, faces


class PreparationCancelled(Exception):
    """Raised when the user cancels before the first frame."""


LARGE_GARMENT = 100_000          # vertices; above this the window warns that preparation is slow


def add_cloth(builder, points, indices, density, particle_radius, edge_ke=0.0):
    """Same model as warp.sim.ModelBuilder.add_cloth_mesh at the origin with no rotation, scale 1,
    zero velocity and no springs, but without Warp's per-vertex Python transform."""
    import warp as wp
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    count = len(points)
    start_vertex = len(builder.particle_q)
    start_tri = len(builder.tri_indices)
    builder.particle_q.extend(wp.vec3(*point) for point in points.tolist())
    builder.particle_qd.extend([wp.vec3(0.0)] * count)
    builder.particle_mass.extend([0.0] * count)
    builder.particle_radius.extend([particle_radius] * count)
    builder.particle_flags.extend([wp.uint32(1)] * count)          # PARTICLE_FLAG_ACTIVE, as add_particle
    inds = (start_vertex + np.asarray(indices, dtype=np.int64)).reshape(-1, 3)
    num_tris = len(inds)
    areas = builder.add_triangles(
        inds[:, 0], inds[:, 1], inds[:, 2],
        [builder.default_tri_ke] * num_tris, [builder.default_tri_ka] * num_tris, [builder.default_tri_kd] * num_tris,
        [builder.default_tri_drag] * num_tris, [builder.default_tri_lift] * num_tris,
    )
    mass = np.asarray(builder.particle_mass, dtype=np.float64)
    share = density * np.asarray(areas, dtype=np.float64) / 3.0
    for corner in range(3):
        np.add.at(mass, inds[:, corner], share)
    builder.particle_mass = mass.tolist()
    end_tri = len(builder.tri_indices)
    adjacency = wp.utils.MeshAdjacency(builder.tri_indices[start_tri:end_tri], end_tri - start_tri)
    edge_indices = np.fromiter((x for e in adjacency.edges.values() for x in (e.o0, e.o1, e.v0, e.v1)),
                               int).reshape(-1, 4)
    builder.add_edges(edge_indices[:, 0], edge_indices[:, 1], edge_indices[:, 2], edge_indices[:, 3],
                      edge_ke=[edge_ke] * len(edge_indices), edge_kd=[builder.default_edge_kd] * len(edge_indices))


def simulate(
    garment, colliders, on_frame=None, cancelled=lambda: False, *,
    hold_shoulders=False, duration=5.0, pinned_vertices=None, stretch=0.0,
    thickness_mm=4.0, friction=0.6, damping=2.4, iterations=8,
    motion=None, initial_duration=0.0, frame_settle=0.0,
    garment_ranges=None, button_vertices=(), keep_details=True, fasten_buttons=True, fasteners=(),
    bending=0.0,
    fastener_gap_mm=None,
    interactive=None, on_step=None, on_pause=None,
    edge_width_mm=0., edge_stiffness=0., fabric_weight_gsm=250.,
    wind=None,
    gravity=None,
    water=None,
    self_collision=False, self_collision_gap_mm=1., self_collision_passes=2,
    crease_angle=30., crease_stiffness=0.,
    cut_pairs=(), cut_release_interval=None,
    on_prepare=None, max_move_cm=0., ground=None,
    loop=False, loop_frames=8, on_loop_status=None,
):
    """``ground``: None or dict(height_cm=..., grip=0-1) for an infinite floor (Y up)."""
    def prepare(text, fraction):
        """Report a preparation step (0-1) and stop here if the user canceled."""
        if cancelled():
            raise PreparationCancelled()
        if on_prepare is not None:
            on_prepare(text, fraction)

    # Seamless loop: no final settling (it would stop the motion the loop carries);
    # after the last frame the simulation continues into frames 1, 2, ... until it meets the first run.
    looping = bool(loop and motion is not None and interactive is None and motion.intervals >= 3)
    if looping:
        duration = 0.
    schedule = DrapeSchedule(motion, duration, initial_duration, frame_settle, max_move_cm)
    fps = schedule.fps
    substeps = max(2, math.ceil(480.0 / fps))
    substeps += substeps % 2
    parameters = (stretch, thickness_mm, friction, damping, bending, edge_width_mm, edge_stiffness, fabric_weight_gsm, crease_angle, crease_stiffness)
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("Material settings must be finite numbers.")
    if not 0 <= stretch <= 0.2 or not 0.1 <= thickness_mm <= 20:
        raise ValueError("Stretch must be between 0 and 20%, and thickness between 0.1 and 20 mm.")
    if not 0 <= friction <= 2 or not 0 <= damping <= 20 or not 1 <= iterations <= 100:
        raise ValueError("Friction, damping or solver iterations are outside the supported range.")
    if not 0 <= bending <= 1:
        raise ValueError("Bend resistance must be between 0 and 100%.")
    if not 0 <= crease_angle <= 180 or not 0 <= crease_stiffness <= 1:
        raise ValueError("Crease angle or stiffness is outside the supported range.")
    if not 0 <= edge_width_mm <= 100 or not 0 <= edge_stiffness <= 1 or not 10 <= fabric_weight_gsm <= 2000:
        raise ValueError("Edge width, edge stiffness or fabric weight is outside the supported range.")
    if fastener_gap_mm is not None and (not math.isfinite(fastener_gap_mm) or not 0 <= fastener_gap_mm <= 50):
        raise ValueError("Fastening gap must be between 0 and 50 mm.")
    if (not math.isfinite(self_collision_gap_mm) or not .1 <= self_collision_gap_mm <= 10
            or not isinstance(self_collision_passes, int) or not 1 <= self_collision_passes <= 8):
        raise ValueError("Self-collision gap or passes are outside the supported range.")
    try:
        import warp as wp
        import warp.sim as sim
    except ImportError as error:
        raise RuntimeError("The Drape Tool requires warp-lang 1.9.1.") from error

    from cloth_constraints import ClothConstraints, update_velocity

    class DeformingMesh(sim.Mesh):
        # Warp caches identical initial meshes. Animated instances need separate
        # buffers because they may move apart later in the timeline.
        __hash__ = object.__hash__

    points, faces = garment
    prepare("analyzing the garment structure", .02)
    from garment_structure import GarmentStructure
    structure = GarmentStructure(points, faces, garment_ranges, button_vertices)
    simulation_faces = structure.body_faces if keep_details else faces
    attached = structure.attached if keep_details else np.zeros(len(points), dtype=bool)
    fasteners = np.asarray(fasteners, dtype=np.int32).reshape(-1, 2)
    if np.any(fasteners < 0) or np.any(fasteners >= len(points)):
        raise ValueError("Fabric fastening points do not match the garment.")
    if fasten_buttons and structure.fasteners:
        fasteners = np.concatenate((fasteners, np.asarray(structure.fasteners, dtype=np.int32)))
    wp.config.kernel_cache_dir = str(Path(__file__).with_name(".warp-cache"))
    # Native per-GPU binaries avoid repeating the driver's expensive PTX JIT
    # every fresh helper process. Warp falls back to PTX on unsupported GPUs.
    wp.config.cuda_output = "cubin"
    prepare("starting the GPU", .12)
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("NVIDIA Warp could not find a CUDA GPU.")

    with wp.ScopedDevice("cuda:0"):
        builder = sim.ModelBuilder()
        origin = wp.vec3(0.0)
        rotation = wp.quat_identity()
        collision_sources = []
        for index, (vertices, triangles) in enumerate(colliders):
            mesh_type = DeformingMesh if motion is not None else sim.Mesh
            source = mesh_type(vertices, triangles.flatten(), compute_inertia=False)
            collision_sources.append(source)
            builder.add_shape_mesh(
                body=-1, pos=origin, rot=rotation, scale=wp.vec3(1.0),
                mesh=source,
                density=0.0, mu=friction,
            )
        prepare("building the cloth mesh", .2)
        add_cloth(builder, points, simulation_faces.flatten(), fabric_weight_gsm / 1000.,
                  thickness_mm * 0.0005, edge_ke=0.0)
        prepare("placing pins and attached details", .4)
        masses = np.maximum(builder.particle_mass, 0.0001 * fabric_weight_gsm / 250.)
        pinned = np.zeros(len(points), dtype=bool)
        if pinned_vertices is not None:
            indices = np.asarray(pinned_vertices, dtype=np.int32)
            if np.any(indices < 0) or np.any(indices >= len(points)):
                raise ValueError("Pinned vertex indices do not match the garment.")
            pinned[indices] = True
        elif hold_shoulders:
            # DAZ uses Y-up. Keep the top 3 cm in the starting pose.
            pinned = points[:, 1] >= points[:, 1].max() - 0.03
        # A pin/fastening picked on a button means the fabric below it, rather
        # than an independent fixed point fighting the detail binding.
        if keep_details:
            for group, body_faces in structure.detail_groups:
                affected = set(np.flatnonzero(pinned[group]).tolist())
                group_set = set(group.tolist())
                selected = set(int(v) for v in fasteners.ravel() if v in group_set)
                selected.update(int(group[i]) for i in affected)
                body = np.unique(body_faces)
                for index in selected:
                    nearest = body[np.argmin(np.linalg.norm(points[body] - points[index], axis=1))]
                    if pinned[index]:
                        pinned[nearest] = True
                    fasteners[fasteners == index] = nearest
            pinned[attached] = False
        # Vertices outside every simulated triangle (stray OBJ points, or ones
        # whose faces were all degenerate) have no constraints. Hold them at
        # their captured position instead of letting them free-fall.
        loose = np.ones(len(points), dtype=bool)
        loose[np.unique(simulation_faces)] = False
        loose &= ~attached
        if np.any(loose):
            print(f"Holding {int(loose.sum())} garment vertices that belong to no simulated triangle.")
        masses[pinned | attached | loose] = 0.0
        for index in np.flatnonzero(pinned | attached | loose):
            builder.particle_flags[index] = wp.uint32(0)
        builder.particle_mass = masses.tolist()
        dt = 1.0 / (fps * substeps)
        prepare("building fabric constraints", .5)
        constraints = ClothConstraints(points, simulation_faces, masses, pinned, stretch=stretch,
                                       iterations=iterations, fasteners=fasteners, bending=bending,
                                       fastener_gap=None if fastener_gap_mm is None else fastener_gap_mm * .001,
                                       edge_width=edge_width_mm * .001, edge_stiffness=edge_stiffness,
                                       crease_angle=crease_angle, crease_stiffness=crease_stiffness,
                                       substeps=substeps)
        if constraints.ignored_fasteners:
            print(f"Ignored {constraints.ignored_fasteners} fastenings that coincide with existing fabric edges.")
        if np.any(pinned) and constraints.unattached:
            print(f"Drape mesh has {constraints.unattached} vertices disconnected from the held shoulders.")

        prepare("uploading the model to the GPU", .7)
        # Warp keeps at most soft_contact_max cloth-collider contacts (64k by default) and silently
        # drops the rest, which lets a large garment pass through the avatar wherever contacts are
        # dropped. Room for every cloth point against every collider.
        builder.soft_contact_max = max(builder.soft_contact_max, len(points) * max(1, len(colliders)))
        model = builder.finalize()
        gravity_force = None
        if gravity is not None:
            from gravity_forces import GravityForces
            model.gravity = np.zeros(3, dtype=np.float32)
            gravity_force = GravityForces()
        model.ground = False
        model.soft_contact_margin = 0.01
        model.soft_contact_mu = friction
        integrator = sim.XPBDIntegrator(iterations=2)
        current, following = model.state(), model.state()
        grab = None
        freeze = None
        if interactive is not None:
            from live_grab import LiveGrab
            from cloth_freeze import ClothFreeze
            grab = LiveGrab(len(points), constraints.inverse_mass)
            freeze = ClothFreeze(len(points))
        grab_step = 1.0 * dt
        cloth_contacts = None
        prepare("setting up collisions and forces", .82)
        from collider_guard import ColliderGuard
        floor = None if ground is None else dict(height=float(ground["height_cm"]) * .01, grip=float(ground.get("grip", .5)))
        guard = (ColliderGuard(collision_sources, len(points), thickness_mm * .0005, ground=floor)
                 if collision_sources or floor is not None else None)
        if self_collision:
            from cloth_self_collision import ClothSelfCollision
            cloth_contacts = ClothSelfCollision(points, simulation_faces, constraints.inverse_mass,
                                               self_collision_gap_mm * .001, self_collision_passes, fasteners)
        wind_force = None
        if wind is not None:
            from wind_forces import WindForces
            wind_force = WindForces(simulation_faces, collision_sources)
        water_force = None
        if water is not None:
            from water_forces import WaterForces
            water_force = WaterForces(simulation_faces, len(points))
        details = None
        if keep_details and np.any(attached):
            from garment_details import GarmentDetails
            details = GarmentDetails(points, structure)
            print(f"Keeping {len(structure.detail_groups)} detached garment details attached to fabric.")
        hold = None
        if len(cut_pairs) and cut_release_interval is not None and motion is not None:
            from cut_hold import CutHold
            hold = CutHold(cut_pairs, cut_release_interval)
        moving = None
        prepare("preparing the avatar movement" if motion is not None else "starting the simulation", .95)
        if motion is not None:
            from moving_colliders import MovingColliders, WRAP

            motion.reset()
            moving = MovingColliders(collision_sources, motion, points, pinned)

        def advance_frame():
            nonlocal current, following
            for substep in range(substeps):
                if moving is not None:
                    moving.update((substep + 1) / substeps, 1.0 / fps, current.particle_q, constraints.anchors)
                current.clear_forces()
                if wind_force is not None:
                    wind_force.apply(model, current, constraints.inverse_mass)
                if gravity_force is not None:
                    gravity_force.apply(model, current, constraints.inverse_mass)
                if water_force is not None:
                    water_force.apply(model, current, constraints.inverse_mass, dt)
                sim.collide(model, current)
                integrator.simulate(model, current, following, dt)
                if grab is not None:
                    grab.project(model, following.particle_q, collision_sources,
                                 grab_step, thickness_mm * .0005)
                    freeze.project(following)
                constraints.project(following.particle_q)
                if hold is not None:
                    hold.project(following.particle_q, constraints.inverse_mass)
                if cloth_contacts is not None:
                    cloth_contacts.project(following.particle_q, current.particle_q)
                if guard is not None:
                    guard.project(following.particle_q, constraints.inverse_mass, current.particle_q)
                if grab is not None:
                    grab.protect(model, following.particle_q, collision_sources, thickness_mm * .0005)
                    freeze.project(following)
                if details is not None:
                    details.update(following.particle_q)
                wp.launch(
                    update_velocity, dim=len(points),
                    inputs=[
                        current.particle_q, following.particle_q,
                        following.particle_qd, constraints.inverse_mass, dt, damping,
                    ],
                )
                if grab is not None:
                    grab.velocity(model, current, following, dt, damping)
                    freeze.project(following)
                current, following = following, current

        # An even number of substeps returns to the same state buffers. Replay on the GPU
        # to avoid thousands of Python launches for the constraint groups.
        if cancelled():
            return None
        with wp.ScopedCapture() as capture:
            advance_frame()
        steps = schedule if interactive is None else interactive.steps(schedule, cancelled, on_pause)
        result = points * 100.0
        recorder = None
        if looping:
            import sys
            from pathlib import Path as _Path
            suite_app = str(_Path(__file__).resolve().parents[2] / 'app')
            if suite_app not in sys.path:
                sys.path.append(suite_app)
            from spin_loop import LoopRecorder, carry_loop
            recorder = LoopRecorder(motion.intervals)
        counter = {'frame': 0}

        def prepare_step(step, frame):
            if hold is not None:
                hold.update(motion.intervals if step.interval == WRAP else step.interval)
            if moving is not None:
                moving.load_interval(step.interval, hold=step.hold, part=step.part, parts=step.parts)
            if wind_force is not None:
                wind_force.update(wind, frame / fps)
            if gravity_force is not None:
                gravity_force.update(gravity)
            if water_force is not None:
                water_force.update(water, gravity.snapshot()[1] if gravity is not None else model.gravity)
            if grab is not None:
                freeze.update(interactive)
                grab.update(interactive, freeze)
            if on_step is not None:
                on_step(step)

        def run_step(step, frame):
            prepare_step(step, frame)
            wp.capture_launch(capture.graph)
            values = current.particle_q.numpy() * 100.0
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"The garment became unstable at {(frame + 1) / fps:.2f} s. "
                    "The preview shows the last valid frame."
                )
            if on_frame is not None:
                on_frame(values)
            return values

        for frame, step in enumerate(steps):
            if cancelled():
                return None
            result = run_step(step, frame)
            counter['frame'] = frame + 1
            if recorder is not None and step.part == step.parts - 1:
                recorder.record(step.sample, result * .01)
        if recorder is not None:
            phase = "Carrying motion into the loop"

            def advance(sample):
                interval = sample - 1
                parts = schedule.parts[interval]
                loop_steps = [DrapeStep(interval, False, phase, sample, part, parts) for part in range(parts)]
                loop_steps += [DrapeStep(sample, True, phase, sample)] * schedule.hold_frames
                values = None
                for step in loop_steps:
                    if cancelled():
                        return None
                    values = run_step(step, counter['frame'])
                    counter['frame'] += 1
                return dict(sample=sample, vertices=values)

            buffers = (current, following)

            def snapshot():
                return dict(frame=counter['frame'], arrays=[(state.particle_q.numpy().copy(), state.particle_qd.numpy().copy())
                                                            for state in buffers])

            def restore(saved):
                counter['frame'] = saved['frame']
                for state, (q, qd) in zip(buffers, saved['arrays']):
                    state.particle_q.assign(q)
                    state.particle_qd.assign(qd)

            def set_state(values, payload):
                current.particle_q.assign(np.ascontiguousarray(values, dtype=np.float32))
                return dict(payload, vertices=np.asarray(values, dtype=np.float64) * 100.)

            def publish(payload):
                if on_step is not None:
                    on_step(DrapeStep(max(0, payload['sample'] - 1), True, "Loop", payload['sample']))
                if on_frame is not None:
                    on_frame(payload['vertices'])

            def wrap():
                """One real step from the last frame into frame 0 (split like a fast frame), then frame 0's
                settling, if any; the result is frame 0."""
                move = max(float(np.linalg.norm(motion.positions(index, motion.intervals) - motion.positions(index, 0),
                                                axis=1).max()) * 100. for index in motion.samples) if motion.samples else 0.
                parts = max(1, math.ceil(move / max_move_cm - 1e-9)) if max_move_cm else 1
                loop_steps = [DrapeStep(WRAP, False, phase, 0, part, parts) for part in range(parts)]
                loop_steps += [DrapeStep(0, True, phase, 0)] * schedule.hold_frames
                values = None
                for step in loop_steps:
                    if cancelled():
                        return None
                    values = run_step(step, counter['frame'])
                    counter['frame'] += 1
                return dict(sample=0, vertices=values)

            status = carry_loop(recorder=recorder, steer_frames=loop_frames, advance=advance, wrap=wrap,
                                get_state=lambda: current.particle_q.numpy(), set_state=set_state,
                                snapshot=snapshot, restore=restore, publish=publish)
            if status is None:
                return None
            if on_loop_status is not None:
                on_loop_status(status)
    return result

