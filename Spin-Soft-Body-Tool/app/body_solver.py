# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Static, topology-preserving contact deformation; no figure pose integration."""
from pathlib import Path
import math
import threading
import numpy as np
import warp as wp
from body_geometry import prepare,surface_constraints,triangle_groups,accept_surface_step,connected_labels,open_edge_vertices
from body_kernels import (relax,self_contacts,fixed_contacts,reverse_fixed,apply,limit_gradient,preserve_faces,
                          opening_contact,opening_tug,opening_finish_clearance,loop_clearance)
from body_dynamics import predict_offsets, VerticalBands
from body_volume import PaintedVolume
from body_guides import FrameGuides
from body_examples import MotionExample
import copy
from body_loop import loop_tail, painted_parts


EXAMPLE_STATE = ('last_normal', 'amount', 'drag_contact', 'lag', 'lag_velocity', 'track', 'current_colliders',
                 'previous_state', 'last_weights')
CATCH_UP_CM = .02          # 0.2 mm: the second pass has caught up with the first run


class RunControl:
    def __init__(self):
        self.condition=threading.Condition();self.paused=False;self.ack=False;self.stopped=False
    def pause(self,value=True):
        with self.condition:self.paused=value;self.ack=False;self.condition.notify_all()
    def stop(self):
        with self.condition:self.stopped=True;self.condition.notify_all()
    def wait(self,on_pause):
        with self.condition:notify=self.paused and not self.ack;self.ack=self.paused
        if notify:on_pause()
        with self.condition:
            while self.paused and not self.stopped:self.condition.wait(.05)
            return not self.stopped


def simulate(garment,colliders,on_frame,cancelled=lambda:False,*,softness,rigid,depth=25.,spread=45.,
             recovery=5.,clearance=.2,steps=40,feather=2,pin_open_edges=False,pin_edge_rings=1,
             control=None,on_pause=lambda:None,ranges=None,motion=None,
             jiggle=False,preserve_volume=False,vertical_bands=False,directional_lag=False,
             frequency=3.,damping=12.,inertia=60.,jiggle_gravity=0.,collision_passes=2,ignore_collider=None,opening_stretch=150.,
             part_soft_seal=False,
             guide_keys=None,guide_strength=65.,guide_lead=18,guide_blend=12,motion_examples=(),
             fold_rounding=25.,lip_drag=3.,drag_reach=8.,loop_frames=0,loop_motion=False,bounce_blend=None):
    if not all(math.isfinite(v) for v in [depth,spread,recovery,clearance]) or not (1<=depth<=100 and 0<=spread<=100 and 0<=recovery<=100 and 0<=clearance<=10 and 1<=steps<=500 and 0<=feather<=10):
        raise ValueError('Deformation settings are outside their supported ranges.')
    if not all(math.isfinite(v) for v in (frequency,damping,inertia)) or not (.25<=frequency<=12 and 0<=damping<=60 and 0<=inertia<=200):
        raise ValueError('Jiggle settings are outside their supported ranges.')
    if not 0<=jiggle_gravity<=100:
        raise ValueError('Jiggle gravity must be between 0 and 100 percent.')
    if not isinstance(loop_frames, int) or (loop_frames != 0 and not 2 <= loop_frames <= 60):
        raise ValueError('Use 2 to 60 frames to blend an animation loop.')
    if not isinstance(collision_passes,int) or not 1<=collision_passes<=8:raise ValueError('Collision accuracy must be between 1 and 8.')
    if not isinstance(pin_edge_rings,int) or not 1<=pin_edge_rings<=10:raise ValueError('Pinned edge rings must be between 1 and 10.')
    if bounce_blend is not None and (not isinstance(bounce_blend,int) or not 0<=bounce_blend<=10):raise ValueError('Bounce edge blend must be between 0 and 10 rings.')
    if not 50<=opening_stretch<=300:raise ValueError('Opening stretch must be between 50 and 300 percent.')
    if not (0<=fold_rounding<=100 and 0<=lip_drag<=10 and 0<=drag_reach<=50):
        raise ValueError('Opening rounding or collider drag is outside its supported range.')
    if (not math.isfinite(guide_strength) or not 0<=guide_strength<=100
            or any(not isinstance(frames, int) or not 1<=frames<=120 for frames in (guide_lead, guide_blend))):
        raise ValueError('Frame guide strength or timing is out of range.')
    points,faces=garment
    if not np.isfinite(points).all():raise ValueError('Invalid source mesh.')
    mobility,normals,exclude,neighbours=prepare(points,faces,softness,rigid,feather,
                                               pin_open_edges=pin_open_edges,pin_edge_rings=pin_edge_rings)
    external_mobility = mobility
    if ignore_collider is not None:
        ignore_collider = np.asarray(ignore_collider, dtype=bool)
        if ignore_collider.shape != mobility.shape:
            raise ValueError('The collider contact mask must match the receiving mesh.')
        external_mobility = np.where(ignore_collider, 0., mobility).astype(np.float32)
    edges,lengths,groups,minimum=surface_constraints(points,faces)
    assigned = np.zeros(len(points), dtype=bool)
    for kind, mask, reference, samples in motion_examples:
        mask = np.asarray(mask, dtype=bool)
        if (motion is None or kind not in ('Bounce', 'Keyed', 'Opening') or not samples
                or any(not 0 <= frame <= motion.intervals for frame in (reference, *samples))
                or mask.shape != assigned.shape or not (mask & (mobility > 0)).any()):
            raise ValueError('A motion example needs a soft region and frames from the captured timeline.')
        if (assigned & mask).any():
            raise ValueError('Paint separate regions for each motion example.')
        if kind == 'Opening' and (not colliders or reference in samples):
            raise ValueError('An Opening needs a collider and example poses distinct from its closed reference.')
        assigned |= mask & (mobility > 0)
    # An Opening with several example frames blends them by the collider's state (kind 'Shaped');
    # one wide frame keeps the contact-driven opening.
    motion_examples = [('Shaped', mask, reference, samples) if kind == 'Opening' and len(samples) > 1
                       else (kind, mask, reference, samples) for kind, mask, reference, samples in motion_examples]
    # Bounce fades into the unpainted skin over its own number of rings; other examples use the protected blend.
    examples = [MotionExample(spec, motion, mobility, external_mobility, neighbours, edges, faces,
                              bounce_blend if spec[0] == 'Bounce' and bounce_blend is not None else feather, assigned)
                for spec in motion_examples]
    openings = [example for example in examples if example.kind == 'Opening']
    for example in examples:
        if example.kind == 'Bounce':
            example.setup_bounce(motion.fps, frequency, damping, inertia * .01, motion)
        elif example.kind == 'Shaped':
            example.setup_shaped(motion, motion.fps)
    open_border = open_edge_vertices(len(points), faces) if colliders else np.zeros(len(points), dtype=bool)
    rim = open_border.copy()
    adjacent_valid = neighbours >= 0
    seal_rim = np.zeros(len(points), dtype=bool)
    if colliders and ignore_collider is not None:
        seal_rim = (external_mobility > 0) & np.any(adjacent_valid & ignore_collider[neighbours], axis=1)
        rim |= seal_rim
    rim &= external_mobility > 0
    stretch = np.full(len(edges), .5, dtype=np.float32)
    if rim.any() and opening_stretch > 50:
        # Let the painted rim widen, tapering back to the ordinary strain limit.
        band = rim.copy()
        for weight in (1., 2./3., 1./3.):
            stretch[np.any(band[edges], axis=1) & (stretch == .5)] += (opening_stretch * .01 - .5) * weight
            band |= (mobility > 0) & np.any(adjacent_valid & band[neighbours], axis=1)
    seal_rim &= ~assigned
    seal_enabled = part_soft_seal and motion is not None and seal_rim.any()
    source_triangles=points[faces]
    source_cross=np.cross(source_triangles[:,1]-source_triangles[:,0],source_triangles[:,2]-source_triangles[:,0])
    source_area=np.linalg.norm(source_cross,axis=1)
    if not mobility.any():raise ValueError('Paint/select a receiving region and apply softness before running.')
    loop_requested = bool(loop_frames and motion is not None)
    # Neither the finish nor the carried loop needs the body to end where it starts: the last
    # frame leads into frame 0, or the run simply finishes on the last frame.
    loop_possible = loop_requested and motion.intervals >= 3
    loop_reason = loop_note = None
    blend = min(loop_frames, motion.intervals - 1) if loop_possible else 0
    # Carry motion into the loop: after the end is closed, the first frames are simulated again starting from the
    # end's state, so the tissue enters frame 1 with the lag and swing it had at the end, then eases back into the
    # first run by frame N.
    carry = bool(blend and loop_motion)
    guides = FrameGuides(guide_keys, motion, np.where(assigned, 0., mobility), guide_lead,
                         guide_blend, guide_strength*.01) if motion is not None and guide_keys else None
    volume = PaintedVolume(mobility, neighbours, faces, source_cross) if motion is not None and jiggle and preserve_volume else None
    bands = VerticalBands(points, mobility, neighbours) if motion is not None and jiggle and vertical_bands else None
    open_animated_body = motion is not None and open_border.any()
    wp.config.kernel_cache_dir=str(Path(__file__).parent/'.warp-cache');wp.config.cuda_output='cubin';wp.init()
    if not wp.is_cuda_available():raise RuntimeError('Soft Body requires an NVIDIA CUDA GPU.')
    with wp.ScopedDevice('cuda:0'):
        x=wp.array(points,dtype=wp.vec3);rest=wp.clone(x);temp=wp.clone(x)
        f=wp.array(np.ascontiguousarray(faces,dtype=np.int32),dtype=int)
        mesh=wp.Mesh(points=x,indices=wp.array(np.ascontiguousarray(faces.ravel(),dtype=np.int32),dtype=int))
        reverse_mesh, reverse_faces = mesh, f
        if colliders and ignore_collider is not None:
            contact_faces = faces[~np.all(ignore_collider[faces], axis=1)]
            if not len(contact_faces):
                reverse_mesh = None
            elif len(contact_faces) != len(faces):
                # A pass-through face must not hide a receiving face behind it.
                reverse_faces = wp.array(np.ascontiguousarray(contact_faces, dtype=np.int32), dtype=int)
                reverse_mesh = wp.Mesh(points=x, indices=wp.array(
                    np.ascontiguousarray(contact_faces.ravel(), dtype=np.int32), dtype=int))
        normal=wp.array(normals,dtype=wp.vec3);mass=wp.array(mobility,dtype=float)
        external_mass = mass if external_mobility is mobility else wp.array(external_mobility,dtype=float)
        shell = wp.empty(0, dtype=int)
        ghost = wp.empty(0, dtype=int)
        paint_regions = wp.empty(0, dtype=int)
        local_contact = bool(colliders) and (motion is None or open_animated_body)
        if volume is not None or bands is not None or local_contact:
            shell = wp.array(connected_labels(neighbours, np.ones(len(points), dtype=bool))[0], dtype=int)
            ghost = wp.array(np.asarray(rigid, dtype=np.int32), dtype=int)
        if local_contact:
            # Contact with an external object must not collapse a thin painted
            # patch against its own interior. Separate paint islands still collide.
            paint_regions = wp.array(connected_labels(neighbours, mobility > 0)[0], dtype=int)
        excluded=wp.array(exclude,dtype=int);adjacent=wp.array(neighbours,dtype=int)
        surface_edges=wp.array(edges,dtype=int);edge_lengths=wp.array(lengths,dtype=float)
        edge_stretch=wp.array(stretch,dtype=float)
        edge_groups=[wp.array(g,dtype=int) for g in groups];local_size=wp.array(minimum,dtype=float)
        face_groups=[wp.array(g,dtype=int) for g in triangle_groups(len(points),faces)]
        reference_normal=wp.array(source_cross/np.maximum(source_area[:,None],1e-20),dtype=wp.vec3)
        reference_area=wp.array(source_area,dtype=float)
        changes=wp.zeros(len(points),dtype=wp.vec3);counts=wp.zeros(len(points),dtype=float)
        fixed=[wp.Mesh(points=wp.array(p,dtype=wp.vec3),indices=wp.array(np.ascontiguousarray(t.ravel(),dtype=np.int32),dtype=int),
                       support_winding_number=bool(openings or blend)) for p,t in colliders]
        opening_mask = wp.empty(0, dtype=int)
        if openings:
            marked = np.zeros(len(points), dtype=np.int32)
            for example in openings:
                marked[example.indices] = 1
            opening_mask = wp.array(marked, dtype=int)
        seal_mask = wp.array(seal_rim.astype(np.int32), dtype=int) if seal_enabled else wp.empty(0, dtype=int)
        seal_entries = [wp.zeros(2 if seal_enabled else 0, dtype=wp.vec3) for _ in fixed]
        if seal_enabled:
            entry_poses = np.zeros((len(fixed), 2, 3), dtype=np.float32)
            entry_poses[:, 0] = [(p.min(axis=0) + p.max(axis=0)) * .5 for p, _ in colliders]
        probes = {}
        finish_buffers = {}
        for example in openings:
            count = len(example.probe_vertices) + len(example.probe_edges)
            probes[example] = (wp.empty(count, dtype=wp.vec3), wp.empty(count, dtype=wp.vec3),
                               wp.empty(count, dtype=float), wp.empty(count, dtype=float))
            if fold_rounding or lip_drag:
                count = len(example.indices) + len(example.probe_edges)
                finish_buffers[example] = (wp.empty(count, dtype=wp.vec3), wp.empty(count, dtype=wp.vec3),
                                           wp.empty(count, dtype=wp.vec3), wp.empty(count, dtype=float))
        maximum=depth*.001;margin=clearance*.001
        loop_contact = bool(blend and fixed and external_mobility.any())
        if loop_contact:
            contact_vertices = np.flatnonzero(external_mobility > 0.)
            edge_contact = np.any(external_mobility[edges] > 0., axis=1)
            if ignore_collider is not None:
                edge_contact &= ~np.any(ignore_collider[edges], axis=1)
            contact_edges = edges[edge_contact]
            count = len(contact_vertices) + len(contact_edges)
            loop_before = wp.empty(count, dtype=wp.vec3)
            loop_after = wp.empty(count, dtype=wp.vec3)
            loop_blocked = wp.zeros(1, dtype=int)

        def loop_is_clear(before, after, collider_poses):
            if not loop_contact or np.array_equal(before, after):
                return True
            for index, posed in collider_poses.items():
                fixed[index].points.assign(np.ascontiguousarray(posed * .01, dtype=np.float32))
                fixed[index].refit()
            loop_before.assign(np.concatenate((before[contact_vertices], before[contact_edges].mean(axis=1))))
            loop_after.assign(np.concatenate((after[contact_vertices], after[contact_edges].mean(axis=1))))
            loop_blocked.zero_()
            for collider in fixed:
                wp.launch(loop_clearance, count, inputs=[loop_before, loop_after, collider.id, margin, loop_blocked])
            return not bool(loop_blocked.numpy()[0])

        def finish_delta(current, correction, edges, surface, delta, weights):
            if not np.any(correction):
                return current
            surface.assign(np.concatenate((current, current[edges].mean(axis=1))))
            delta.assign(np.concatenate((correction, correction[edges].mean(axis=1))))
            weights.fill_(1.)
            for collider in fixed:
                wp.launch(opening_finish_clearance,len(surface),inputs=[surface,delta,collider.id,margin,weights])
            scales = weights.numpy()
            vertex_scales = scales[:len(current)]
            for endpoints in edges.T:
                np.minimum.at(vertex_scales, endpoints, scales[len(current):])
            return current + correction * vertex_scales[:, None]

        def finish_opening(result, pose, cross, left, right, moving, sample):
            rounded = result
            if fold_rounding:
                rounded = result.copy()
                for example, (surface, delta, _, weights) in finish_buffers.items():
                    current = result[example.indices]
                    correction = example.round_folds(current, fold_rounding * .01)
                    rounded[example.indices] = finish_delta(current, correction, example.probe_edges,
                                                            surface, delta, weights)
                if not np.array_equal(rounded, result):
                    rounded, _, _ = accept_surface_step(rounded,result,pose,faces,cross,maximum)
            if not lip_drag:
                return rounded
            finished = rounded.copy()
            collider_motion = {index: (end - start).mean(axis=0) for index, (start, end) in moving.items()}
            for example, (surface, delta, tug, weights) in finish_buffers.items():
                indices = example.indices
                current = rounded[indices]
                edges = example.probe_edges
                surface.assign(np.concatenate((current, current[edges].mean(axis=1))))
                tug.zero_()
                weights.zero_()
                if sample:
                    attached_motion = (right[example.anchors] - left[example.anchors]).mean(axis=0)
                    for index, collider in enumerate(fixed):
                        relative = collider_motion.get(index, 0.) - attached_motion
                        wp.launch(opening_tug, len(surface), inputs=[surface,collider.id,wp.vec3(*relative),
                                  max(.005, 3. * margin),weights,tug])
                velocities = tug.numpy() * motion.fps
                proximity = weights.numpy()
                contact = example.probe_vertices
                # Seed from the opening contact test or a lip already resting
                # within the surface clearance band after collision projection.
                weights_at_contact = np.concatenate((proximity[contact], proximity[len(current):]))
                on_surface = weights_at_contact >= .96
                active = ((example.drag_contact | on_surface) & (weights_at_contact > 0.))
                active[:len(contact)] &= example.amount[contact] > 1.e-4
                active[len(contact):] &= example.amount[edges].mean(axis=1) > 1.e-4
                contact_velocity = np.concatenate((velocities[contact], velocities[len(current):]))
                influence = weights_at_contact * active
                contact_velocity *= influence[:, None]
                velocity = np.zeros_like(current)
                velocity[contact] = contact_velocity[:len(contact)]
                counts = np.zeros(len(current), dtype=np.float32)
                counts[contact] = influence[:len(contact)]
                for endpoints in edges.T:
                    np.add.at(velocity, endpoints, contact_velocity[len(contact):])
                    np.add.at(counts, endpoints, influence[len(contact):])
                velocity /= np.maximum(1.e-8, counts)[:, None]
                target = example.drag_targets(velocity, lip_drag*.001, drag_reach*.001)
                easing = 1. - np.exp(-1. / (motion.fps * np.where(np.any(target != 0., axis=1), .045, .12)))
                example.lip_drag += (target - example.lip_drag) * easing[:, None]
                finished[indices] = finish_delta(current, example.lip_drag, edges, surface, delta, weights)
            protected = rounded
            if not np.array_equal(finished, rounded):
                protected, _, _ = accept_surface_step(finished,rounded,pose,faces,cross,maximum)
            for example in finish_buffers:
                example.lip_drag = protected[example.indices] - rounded[example.indices]
            return protected
        receiving=points[mobility>0]
        active_low=wp.vec3(*(receiving.min(axis=0)-2*maximum-margin))
        active_high=wp.vec3(*(receiving.max(axis=0)+2*maximum+margin))
        active_bounds=wp.array([active_low,active_high],dtype=wp.vec3)
        base_substeps=max(1,math.ceil(120./motion.fps)) if motion is not None else 1
        dt=1./(motion.fps*base_substeps) if motion is not None else 1./120.
        smoothing=wp.array([1.-math.exp(-spread*.12*dt) if motion is not None else spread*.003],dtype=float)
        if openings:
            opening_step = max(.0003, .5 * min(float(np.median(minimum[example.indices])) for example in openings))
            opening_anchors = np.unique(np.concatenate([example.anchors for example in openings]))
        def project_surface():
            for group in edge_groups:
                wp.launch(limit_gradient,len(group),inputs=[x,rest,mass,surface_edges,edge_lengths,edge_stretch,group])
            for group in face_groups:
                wp.launch(preserve_faces,len(group),inputs=[x,mass,f,reference_normal,reference_area,group])
        def physics():
            wp.launch(relax,len(x),inputs=[x,rest,mass,adjacent,smoothing,0. if motion is not None else recovery*.0005,maximum,temp]);wp.copy(x,temp)
            for _ in range(collision_passes if motion is not None else 8):
                mesh.refit();changes.zero_();counts.zero_()
                if reverse_mesh is not None and reverse_mesh is not mesh:
                    reverse_mesh.refit()
                wp.launch(self_contacts,len(x),inputs=[x,normal,f,mesh.id,mass,excluded,maximum+margin,margin,active_bounds,
                    rest,reference_normal,shell,ghost,paint_regions,changes,counts])
                for collider, entry in zip(fixed, seal_entries):
                    wp.launch(fixed_contacts,len(x),inputs=[x,collider.id,external_mass,margin,
                        seal_mask,entry,maximum,opening_mask,changes,counts])
                    if reverse_mesh is not None:
                        wp.launch(reverse_fixed,len(collider.points),inputs=[collider.points,reverse_mesh.id,reverse_faces,
                            external_mass,margin,opening_mask,changes,counts])
                wp.launch(apply,len(x),inputs=[x,rest,mass,changes,counts,local_size,maximum])
                for _ in range(1 if motion is not None else 2):
                    project_surface()
        with wp.ScopedCapture() as capture:physics()
        repair=None
        if motion is not None and (jiggle or open_animated_body or guides is not None or seal_enabled or examples):
            # Give local strain and face constraints more time to converge without
            # taking another physics step or reducing motion across the whole mesh.
            with wp.ScopedCapture() as surface_capture:
                for _ in range(4):project_surface()
            def repair(candidate):
                x.assign(np.ascontiguousarray(candidate,dtype=np.float32))
                wp.capture_launch(surface_capture.graph)
                return x.numpy()
        if motion is not None:
            motion.reset();last_pose=points.copy();relative_velocity=np.zeros_like(points);previous_pose_velocity=np.zeros_like(points)
            previous_guide = np.zeros_like(points) if guides is not None else None
            total_step=0
            loop_start = []
            tail = []
            def advance(sample, wrap=False):
                """One timeline frame: the step that follows the state left by the previous frame. ``wrap``: the loop's
                step from the last frame into frame 0, output as frame 0."""
                stepping = bool(wrap or sample)
                nonlocal dt,last_pose,relative_velocity,previous_pose_velocity,previous_guide,total_step
                if wrap:
                    # Loop join: one real step from the last frame into frame 0 (the last frame leads into frame 0).
                    left, right = motion.body_positions(motion.intervals), motion.body_positions(0)
                else:
                    left=motion.body_positions(max(0,sample-1));right=motion.body_positions(sample)
                if left.shape!=points.shape or right.shape!=points.shape:raise ValueError('Animated body vertex counts changed.')
                if wrap:
                    moving = {i: (motion.positions(i, motion.intervals), motion.positions(i, 0)) for i in motion.samples}
                else:
                    moving={i:motion.pair(i,max(0,sample-1)) if sample else (motion.positions(i,0),)*2 for i in motion.samples}
                substeps = base_substeps
                if openings and stepping:
                    # Subdivide fast approaches relative to the size of the painted opening.
                    attached_travel = np.linalg.norm(right[opening_anchors] - left[opening_anchors], axis=1).max()
                    collider_travel = max((np.linalg.norm(second - first, axis=1).max()
                                           for first, second in moving.values()), default=0.)
                    substeps = max(base_substeps, min(32, math.ceil((attached_travel + collider_travel) / opening_step)))
                next_dt = 1. / (motion.fps * substeps)
                if next_dt != dt:
                    dt = next_dt
                    smoothing.assign(np.asarray([1. - math.exp(-spread * .12 * dt)], dtype=np.float32))
                limited=False;blocked=False;example_limited=False;example_held=0
                for substep in range(substeps if stepping else 1):
                    if cancelled() or (control is not None and not control.wait(on_pause)):return
                    alpha=(substep+1)/substeps if stepping else 1.
                    pose=np.asarray(left*(1-alpha)+right*alpha,dtype=np.float32)
                    if not np.isfinite(pose).all():raise ValueError('Invalid animated body pose.')
                    for i,(first,second) in moving.items():
                        collider_pose = np.ascontiguousarray(first*(1-alpha)+second*alpha,dtype=np.float32)
                        fixed[i].points.assign(collider_pose);fixed[i].refit()
                        if seal_enabled:
                            center = (collider_pose.min(axis=0) + collider_pose.max(axis=0)) * .5
                            travel = center - entry_poses[i, 0]
                            entry_poses[i, 0] = center
                            if stepping:
                                travel -= (pose[seal_rim] - last_pose[seal_rim]).mean(axis=0)
                                distance = np.linalg.norm(travel)
                                if distance > 1.e-6:
                                    entry_poses[i, 1] = travel / distance
                            seal_entries[i].assign(entry_poses[i])
                    timeline_sample = (motion.intervals + alpha if wrap else
                                       sample - 1 + alpha if sample else 0.)
                    for example in examples:
                        if example.kind == 'Shaped':
                            example.current_colliders = {i: first * (1 - alpha) + second * alpha
                                                         for i, (first, second) in moving.items()}
                        attachment = example.attachment(pose)
                        target, travel = example.target(pose, timeline_sample, attachment)
                        if travel is not None:
                            closed, wide, previous, required = probes[example]
                            closed.assign(example.probes(target))
                            wide.assign(example.probes(target + travel))
                            previous.assign(example.probes(example.amount))
                            required.zero_()
                            for collider in fixed:
                                wp.launch(opening_contact, len(closed), inputs=[closed,wide,collider.id,
                                          margin,previous,required])
                            amount = example.update_opening(required.numpy(), math.exp(-max(1., recovery) * .8 * dt))
                            target += travel * amount[:, None]
                        target, reduced = example.accept_target(pose, target, attachment)
                        pose[example.indices] = target
                        example_limited |= bool(reduced)
                        example_held = max(example_held, int(reduced))
                    offset=x.numpy()-last_pose
                    if examples and not stepping:
                        offset[assigned] = 0.
                    residual = offset - previous_guide if guides is not None else offset
                    pose_velocity=(pose-last_pose)/dt if stepping else np.zeros_like(pose)
                    if stepping and jiggle:
                        predicted,relative_velocity=predict_offsets(residual,relative_velocity,pose_velocity,previous_pose_velocity,
                            mobility,dt,frequency,damping,inertia*.01,volume=volume,directional_lag=directional_lag,
                            gravity=9.80665 * jiggle_gravity * .01)
                    elif stepping:
                        predicted=residual*math.exp(-recovery*.8*dt);relative_velocity[:]=0
                    else:predicted=residual
                    triangle=pose[faces];cross=np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]);area=np.linalg.norm(cross,axis=1)
                    if bands is not None:
                        bands.project(predicted)
                    if volume is not None:
                        volume.restore(predicted, pose, cross, min(.002, maximum * .25))
                    if examples and jiggle:
                        predicted[assigned] = residual[assigned] * math.exp(-recovery*.8*dt)
                    if guides is not None:
                        # Carry the guide motion separately so the spring can still
                        # react to the figure and contacts around the moving target.
                        guide = guides.offset_at(timeline_sample)
                        predicted = predicted + guide
                    rest.assign(pose);x.assign(np.ascontiguousarray(pose+predicted,dtype=np.float32))
                    reference_normal.assign(cross/np.maximum(area[:,None],1e-20));reference_area.assign(area)
                    vertex_normal=np.zeros_like(pose)
                    for corner in range(3):np.add.at(vertex_normal,faces[:,corner],cross)
                    normal.assign(vertex_normal/np.maximum(np.linalg.norm(vertex_normal,axis=1,keepdims=True),1e-20))
                    lengths_now=np.linalg.norm(pose[edges[:,1]]-pose[edges[:,0]],axis=1)
                    edge_lengths.assign(lengths_now);minimum_now=np.full(len(pose),.01,dtype=np.float32)
                    np.minimum.at(minimum_now,edges[:,0],lengths_now);np.minimum.at(minimum_now,edges[:,1],lengths_now);local_size.assign(minimum_now)
                    receiving=pose[mobility>0];active_bounds.assign(np.asarray([receiving.min(0)-2*maximum-margin,receiving.max(0)+2*maximum+margin],dtype=np.float32))
                    wp.capture_launch(capture.graph);candidate=x.numpy()
                    if not np.isfinite(candidate).all():raise RuntimeError('Animation deformation became invalid; the last valid preview is retained.')
                    previous_safe,_,_=accept_surface_step(pose+offset,pose,pose,faces,cross,maximum)
                    result,step_limited,step_blocked=accept_surface_step(candidate,previous_safe,pose,faces,cross,maximum,repair=repair)
                    limited|=step_limited;blocked|=step_blocked
                    accepted = result - pose
                    if guides is not None:
                        accepted -= guide
                        previous_guide = guide
                    relative_velocity=(accepted-residual)/dt if stepping and not step_blocked else np.zeros_like(pose)
                    x.assign(np.ascontiguousarray(result,dtype=np.float32));last_pose=pose;previous_pose_velocity=pose_velocity
                    total_step+=1
                saturated=int(np.count_nonzero((np.linalg.norm(result-pose,axis=1)>=maximum*.999)&(mobility>0)))
                # Refine the published surface without feeding the cosmetic changes
                # back into the contact solve or its opening/release state.
                preview = finish_opening(result, pose, cross, left, right, moving, stepping) if finish_buffers else result
                payload = dict(vertices=preview*100.,rest_vertices=pose*100.,colliders=motion.preview(sample),sample=sample,
                    time=sample/motion.fps,step=total_step,saturated=saturated,surface_limited=limited,blocked=blocked,
                    example_limited=example_limited,example_held=example_held)
                return payload, right
            first_run = {}
            soft_mask = mobility > 0.

            def snapshot():
                """Everything the next frame depends on: positions, speeds, timing and the examples' state."""
                return dict(x=x.numpy().copy(), last_pose=last_pose.copy(), relative_velocity=relative_velocity.copy(),
                            previous_pose_velocity=previous_pose_velocity.copy(), dt=dt,
                            smoothing=smoothing.numpy().copy(),
                            previous_guide=None if previous_guide is None else previous_guide.copy(),
                            entry_poses=entry_poses.copy() if seal_enabled else None,
                            examples=[{key: copy.deepcopy(value) for key, value in vars(example).items()
                                       if key in EXAMPLE_STATE} for example in examples])

            def restore(state):
                nonlocal last_pose, relative_velocity, previous_pose_velocity, previous_guide, dt
                x.assign(state['x']); smoothing.assign(state['smoothing']); dt = state['dt']
                last_pose = state['last_pose'].copy(); relative_velocity = state['relative_velocity'].copy()
                previous_pose_velocity = state['previous_pose_velocity'].copy()
                previous_guide = None if state['previous_guide'] is None else state['previous_guide'].copy()
                if state['entry_poses'] is not None:
                    entry_poses[:] = state['entry_poses']
                for example, saved in zip(examples, state['examples']):
                    for key, value in saved.items():
                        setattr(example, key, copy.deepcopy(value))

            def wind_back_clocks():
                """The second pass replays frames 1, 2, ... after the last frame, so the examples' own clocks (Bounce
                spring, Opening collider tracking) are moved back one cycle; otherwise time would run backwards and
                the spring would stop updating."""
                cycle = (motion.intervals + 1) / motion.fps     # the cycle includes the step into frame 0
                for example in examples:
                    if getattr(example, 'track', None) is not None:
                        time, center, velocity = example.track
                        example.track = (time - cycle, center, velocity)
                    if getattr(example, 'previous_state', None) is not None:
                        time, point, velocity = example.previous_state
                        example.previous_state = (time - cycle, point, velocity)

            def catch_up(end_payload):
                """Carry motion into the loop. The end is not blended: the simulation continues past the last frame
                through frames 1, 2, ... on the same body poses (a second pass from the true end state) until it
                matches the first run, which started from no movement. If it has not caught up by the middle of
                the timeline, the closest frame is used: the second pass is run again and, over the N frames before
                that frame, steered onto the first run (the simulation continues from each steered shape), so it
                lands there. Frame 0 is simulated: one real step from the last frame into frame 0 (the last frame leads
                into frame 0). Later frames are the first run unchanged."""
                end_state = snapshot()
                if advance(0, wrap=True) is None:
                    return False
                wind_back_clocks()
                middle = max(1, motion.intervals // 2)
                meet, closest_gap, met = 1, math.inf, False
                for sample in range(1, middle + 1):
                    stepped = advance(sample)
                    if stepped is None:
                        return False
                    gap = float(np.abs(stepped[0]['vertices'][soft_mask] - first_run[sample]).max(initial=0.))
                    on_frame(dict(stepped[0], preview=True, phase='Carrying motion into the loop…'))
                    if gap < closest_gap:
                        meet, closest_gap = sample, gap
                    if gap <= CATCH_UP_CM:
                        met = True
                        break
                steer = 0 if met else min(blend, meet)
                restore(end_state)
                stepped = advance(0, wrap=True)
                if stepped is None:
                    return False
                wind_back_clocks()
                published = [stepped[0]]
                for sample in range(1, meet + 1):
                    stepped = advance(sample)
                    if stepped is None:
                        return False
                    frame = stepped[0]
                    if sample > meet - steer:
                        t = (sample - (meet - steer)) / steer
                        weight = t * t * (3. - 2. * t)
                        steered = frame['vertices'].copy()
                        steered[soft_mask] = steered[soft_mask] * (1. - weight) + first_run[sample] * weight
                        frame['vertices'] = steered
                        x.assign(np.ascontiguousarray(steered * .01, dtype=np.float32))
                    on_frame(dict(frame, preview=True, phase='Landing the loop…'))
                    published.append(frame)
                note = (f'caught up on frame {motion.start_frame + meet}' if met else
                        f'steered onto frame {motion.start_frame + meet} over {steer} frames, '
                        f'{closest_gap * 10.:.1f} mm apart there')
                published[-1]['loop_status'] = f'Loop closed with motion carried ({note}).'
                published[-1]['loop_closed'] = True
                for frame in published:
                    on_frame(frame)
                return True
            for sample in range(motion.intervals+1):
                stepped = advance(sample)
                if stepped is None:
                    return
                payload, right = stepped
                if carry and sample <= motion.intervals // 2:
                    first_run[sample] = payload['vertices'][soft_mask].astype(np.float32)
                if blend and not carry and sample < 2:
                    loop_start.append((payload['vertices'], right * 100.))   # result and captured frame
                if blend and not carry and sample > motion.intervals - blend:
                    payload['source'] = right * 100.          # the captured frame (no examples or guides)
                    tail.append(payload)
                else:
                    if loop_reason and sample == motion.intervals:
                        payload['loop_status'] = f'Loop skipped: {loop_reason}'
                    on_frame(payload)
            if tail:
                corrected = []
                reduced = 0
                status = f'Finished on the last frame over {blend} frames.'
                # Finish on last frame: over the last N frames the painted area eases onto
                # its captured DAZ shape on the last frame and comes to rest there (settled, as the run starts from a
                # still pose). The blend works on offsets from the captured frames, so the moving body is followed;
                # unpainted vertices stay exactly as captured.
                end_source = tail[-1]['source']
                soft = mobility > 0.
                parts = painted_parts(soft, faces)
                target = np.zeros_like(end_source)               # no offset: the captured last-frame shape
                target_next = target                             # at rest
                offsets = [dict(vertices=payload['vertices'] - payload['source']) for payload in tail]
                closing = (offset + payload.pop('source') for payload, offset in
                           zip(tail, loop_tail(target, target_next, offsets, mobility > 0.)))
                # What a contact-held frame could not take is carried into the next frames, so the close
                # catches up once the collider lets go.
                shortfall = np.zeros_like(points, dtype=np.float64)
                for payload, candidate in zip(tail, closing):
                    if cancelled() or (control is not None and not control.wait(on_pause)):
                        return
                    before = payload['vertices'] * .01
                    after = candidate * .01 + shortfall
                    a, b, c = before[faces].transpose(1, 0, 2)
                    reference = np.cross(b-a,c-a)
                    a, b, c = after[faces].transpose(1, 0, 2)
                    reference_after = np.cross(b-a,c-a)

                    def bad_faces(cross, ref):
                        size = np.linalg.norm(ref, axis=1)
                        return (size > 1e-10) & ((np.sum(cross*ref, axis=1) <= 0) | (np.linalg.norm(cross, axis=1) < size*.08))

                    def folds(trial):
                        # A face folds only if it is turned over or collapsed against both ends of this frame's
                        # step: closing lips turn faces from the open toward the closed frame-0 shape on purpose,
                        # and judging them against the open shape alone would stall the close short of
                        # frame 0.
                        a, b, c = trial[faces].transpose(1, 0, 2)
                        cross = np.cross(b-a,c-a)
                        return bool(np.any(bad_faces(cross, reference) & bad_faces(cross, reference_after)))

                    # Never cancel: an unsafe correction takes its largest safe share. Each connected painted
                    # area gets its own share: an area that would touch the collider
                    # does not cut another area's correction on that frame.
                    accepted = before.copy()
                    frame_reduced = False
                    for part in parts:
                        def safe(fraction):
                            trial = accepted.copy()
                            trial[part] += (after[part] - before[part]) * fraction
                            return (np.isfinite(trial).all() and not folds(trial)
                                    and loop_is_clear(before, trial, payload['colliders']))
                        fraction = 1.
                        if not safe(1.):
                            low, high = 0., 1.
                            for _ in range(6):
                                middle = (low + high) * .5
                                low, high = (middle, high) if safe(middle) else (low, middle)
                            fraction = low
                            frame_reduced = True
                        accepted[part] += (after[part] - before[part]) * fraction
                    reduced += frame_reduced
                    shortfall = after - accepted
                    corrected.append(accepted * 100.)
                closed = True
                for payload, candidate in zip(tail, corrected):
                    payload['vertices'] = candidate
                if reduced:
                    status = status[:-1] + f' (correction reduced on {reduced} frame(s) to avoid folds or colliders).'
                if loop_note:
                    status += f' Note: {loop_note[0].upper()}{loop_note[1:]} The painted area follows it.'
                tail[-1]['loop_status'] = status
                tail[-1]['loop_closed'] = False                  # a finish, not a loop: replay plays straight through
                for payload in tail:
                    on_frame(payload)
            if carry and not catch_up(payload):
                return
            return
        frame=0;end=steps;previous=points.copy()
        while True:
            if cancelled() or (control is not None and not control.wait(on_pause)):return
            if frame>=end:
                if control is None:return
                control.pause();
                if not control.wait(on_pause):return
                end+=steps
            wp.capture_launch(capture.graph);result=x.numpy()
            if not np.isfinite(result).all():raise RuntimeError('Deformation became invalid; the last valid preview is retained.')
            result,limited,blocked=accept_surface_step(result,previous,points,faces,source_cross,maximum)
            # Continue from the accepted shape, never from a rejected GPU step.
            x.assign(np.ascontiguousarray(result,dtype=np.float32));previous=result.copy()
            saturated=int(np.count_nonzero((np.linalg.norm(result-points,axis=1)>=maximum*.999)&(mobility>0)))
            on_frame(dict(vertices=result*100.,colliders={},sample=0,time=(frame+1)/60.,step=frame+1,saturated=saturated,surface_limited=limited,blocked=blocked))
            frame+=1
            if blocked:
                if control is None:return
                control.pause()
