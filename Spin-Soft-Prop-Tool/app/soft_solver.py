# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Shape-preserving volumetric cage simulation with prescribed surface contacts."""
from pathlib import Path
import math
import threading
import numpy as np
import warp as wp
from soft_cage import build_cage
import sys
_SUITE_APP = str(Path(__file__).resolve().parents[2] / 'app')
if _SUITE_APP not in sys.path:
    sys.path.append(_SUITE_APP)
from spin_loop import LoopRecorder, carry_loop
from soft_kernels import predict,edges,volumes,embed,contact,apply_contact,velocity,move_mesh,volume_floor,spread_surface,reverse_contact


class RunControl:
    def __init__(self):
        self.condition=threading.Condition()
        self.paused=False;self.ack=False;self.stopped=False;self.at_end=False
    def pause(self,value=True):
        with self.condition:
            self.paused=value;self.ack=False;self.condition.notify_all()
    def stop(self):
        with self.condition:self.stopped=True;self.condition.notify_all()
    def wait(self,on_pause):
        with self.condition:
            notify=self.paused and not self.ack
            if notify:self.ack=True
        if notify:on_pause()
        with self.condition:
            while self.paused and not self.stopped:self.condition.wait(.05)
            return not self.stopped


def simulate(garment,colliders,on_frame,cancelled=lambda:False,*,ranges=None,painted=None,
             bottom=True,resolution=10,firmness=20.,volume=35.,damping=10.,recovery=.5,density=80.,
             duration=3.,iterations=4,motion=None,control=None,on_pause=lambda:None,initial=.25,
             tension=35.,clearance=.2,loop=False,loop_frames=8):
    values=[firmness,volume,damping,recovery,density,duration,initial,tension,clearance]
    if not all(math.isfinite(v) for v in values) or not (0<=firmness<=100 and 0<=volume<=100 and 0<=damping<=50
        and 0<=recovery<=5 and 10<=density<=1500 and .05<=duration<=120 and 0<=initial<=10 and 1<=iterations<=12
        and 0<=tension<=100 and 0<=clearance<=10):
        raise ValueError('Material or simulation settings are outside their supported ranges.')
    points,faces=garment
    cage=build_cage(points,ranges or [(0,len(points))],resolution,bottom,painted)
    wp.config.kernel_cache_dir=str(Path(__file__).parent/'.warp-cache')
    wp.config.cuda_output='cubin'
    wp.init()
    if not wp.is_cuda_available():raise RuntimeError('Soft prop simulation requires an NVIDIA CUDA GPU and driver.')
    fps=motion.fps if motion is not None else 60.
    substeps=max(2,math.ceil(240/fps))
    dt=1./(fps*substeps)
    young=1000.*200.**(firmness/100.)
    bulk=1000.*300.**(volume/100.)
    with wp.ScopedDevice('cuda:0'):
        masses=np.zeros(len(cage.points),dtype=np.float32)
        for k in range(4):np.add.at(masses,cage.tets[:,k],cage.rest_volumes*density/4)
        inverse=np.divide(1.,masses,out=np.zeros_like(masses),where=masses>0)
        inverse[cage.supports]=0
        inv=wp.array(inverse,dtype=float)
        rest=wp.array(cage.points,dtype=wp.vec3);x=wp.clone(rest);previous=wp.clone(rest)
        smoothed=wp.clone(rest)
        neighbours=wp.array(cage.neighbours,dtype=int);neighbour_weights=wp.array(cage.neighbour_weights,dtype=float)
        v=wp.zeros(len(x),dtype=wp.vec3)
        pair=wp.array(cage.edges,dtype=int);tet=wp.array(cage.tets,dtype=int)
        length=wp.array(np.linalg.norm(cage.points[cage.edges[:,0]]-cage.points[cage.edges[:,1]],axis=1),dtype=float)
        restvol=wp.array(cage.rest_volumes,dtype=float)
        le=wp.zeros(len(cage.edges),dtype=float);lv=wp.zeros(len(cage.tets),dtype=float)
        eg=[wp.array(g,dtype=int) for g in cage.edge_groups];tg=[wp.array(g,dtype=int) for g in cage.tet_groups]
        mapping=wp.array(cage.embedding,dtype=int);weights=wp.array(cage.weights,dtype=float)
        surface=wp.array(points,dtype=wp.vec3)
        surface_faces=wp.array(np.ascontiguousarray(faces,dtype=np.int32),dtype=int)
        soft_mesh=wp.Mesh(points=surface,indices=wp.array(np.ascontiguousarray(faces.ravel(),dtype=np.int32),dtype=int))
        corrections=wp.zeros(len(x),dtype=wp.vec3);counts=wp.zeros(len(x),dtype=float)
        meshes=[];resources={}
        for index,(vertices,triangles) in enumerate(colliders):
            mesh=wp.Mesh(points=wp.array(vertices,dtype=wp.vec3),indices=wp.array(triangles.ravel(),dtype=int))
            meshes.append(mesh)
            if motion is not None and index in motion.samples:
                resources[index]=(wp.array(vertices,dtype=wp.vec3),wp.array(vertices,dtype=wp.vec3))
        limit=min(.003,cage.minimum_cell*.2)
        margin=clearance*.001
        def spread():
            if tension>0:
                wp.launch(spread_surface,dim=len(x),inputs=[x,rest,inv,neighbours,neighbour_weights,tension*.002,smoothed])
                wp.copy(x,smoothed)
        def physics():
            wp.copy(previous,x);le.zero_();lv.zero_()
            wp.launch(predict,dim=len(x),inputs=[x,v,rest,inv,dt,damping,recovery*2*math.pi])
            for _ in range(iterations):
                for group in eg:wp.launch(edges,dim=len(group),inputs=[x,inv,pair,length,le,group,young,dt])
                for group in tg:wp.launch(volumes,dim=len(group),inputs=[x,inv,tet,restvol,lv,group,bulk,dt])
                spread()
                wp.launch(embed,dim=len(points),inputs=[x,mapping,weights,surface])
                corrections.zero_();counts.zero_()
                for mesh in meshes:
                    wp.launch(contact,dim=len(points),inputs=[surface,mapping,weights,inv,mesh.id,margin,limit,corrections,counts])
                wp.launch(apply_contact,dim=len(x),inputs=[x,corrections,counts])
            # Contact needs priority over elastic recovery. Extra projection sweeps
            # clear overlap that the last elastic solve can leave in a firm cage.
            # Keep corrections in cage space so preview and subsequent steps agree.
            for sweep in range(16 if meshes else 0):
                if sweep%4==0:spread()
                wp.launch(embed,dim=len(points),inputs=[x,mapping,weights,surface])
                corrections.zero_();counts.zero_()
                soft_mesh.refit()
                for mesh in meshes:
                    wp.launch(reverse_contact,dim=len(mesh.points),inputs=[mesh.points,soft_mesh.id,surface_faces,
                              mapping,weights,inv,margin,limit,corrections,counts])
                for mesh in meshes:
                    wp.launch(contact,dim=len(points),inputs=[surface,mapping,weights,inv,mesh.id,margin,limit,corrections,counts])
                wp.launch(apply_contact,dim=len(x),inputs=[x,corrections,counts])
                for group in tg:wp.launch(volume_floor,dim=len(group),inputs=[x,inv,tet,restvol,group])
            wp.launch(velocity,dim=len(x),inputs=[previous,x,v,inv,dt])
            wp.launch(embed,dim=len(points),inputs=[x,mapping,weights,surface])
        with wp.ScopedCapture() as capture:physics()
        if motion is not None:motion.reset()
        frames=motion.intervals+1 if motion is not None else max(1,math.ceil(duration*fps))
        state={'elapsed':0.}
        looping=bool(loop and motion is not None and motion.intervals>=3)
        recorder=LoopRecorder(motion.intervals) if looping else None

        def check(cage_points,result):
            if not np.isfinite(result).all() or np.max(np.abs(cage_points))>1.e5:
                raise RuntimeError('The soft prop became unstable. Reduce overlap/stiffness or use more iterations. The last valid frame is retained.')
            a,b,c,d=cage_points[cage.tets].transpose(1,0,2)
            ratios=np.sum((b-a)*np.cross(c-a,d-a),axis=1)/(6*cage.rest_volumes)
            if ratios.min()<.015:
                raise RuntimeError('The prop was compressed beyond its cage limit. Reduce overlap, increase resolution, or change its supports. The last valid frame is retained.')

        def output(sample):
            wp.launch(embed,dim=len(points),inputs=[x,mapping,weights,surface])
            return dict(vertices=surface.numpy()*100.,sample=sample,time=state['elapsed'],
                        colliders=motion.preview(sample) if motion is not None else {})

        def step(frame,wrap=False):
            """One output frame: the substeps from the previous source frame to this one. ``wrap``: the loop's step
            from the last frame into frame 0."""
            sample=0 if wrap else (min(frame,motion.intervals) if motion is not None else 0)
            for index,(first,second) in resources.items():
                if wrap:
                    left,right=motion.positions(index,motion.intervals),motion.positions(index,0)
                else:
                    left,right=motion.pair(index,max(0,sample-1)) if sample else (motion.positions(index,0),)*2
                first.assign(np.ascontiguousarray(left,dtype=np.float32));second.assign(np.ascontiguousarray(right,dtype=np.float32))
            steps=max(substeps,math.ceil(initial/dt)) if motion is not None and frame==0 and not wrap else substeps
            for _ in range(steps):
                if cancelled() or (control is not None and control.stopped):return None
                alpha=1. if frame==0 and not wrap else (_+1)/steps
                for index,(first,second) in resources.items():
                    wp.launch(move_mesh,dim=len(first),inputs=[first,second,alpha,meshes[index].points]);meshes[index].refit()
                wp.capture_launch(capture.graph)
                state['elapsed']+=dt
            payload=dict(vertices=surface.numpy()*100.,sample=sample,time=state['elapsed'],
                         colliders=motion.preview(sample) if motion is not None else {})
            check(x.numpy(),payload['vertices'])
            return payload

        frame=0;payload=None
        while True:
            if cancelled() or (control is not None and not control.wait(on_pause)):return
            if frame>=frames:
                if motion is None and control is not None:
                    control.at_end=True;control.pause(True)
                    if not control.wait(on_pause):return
                    frames+=max(1,math.ceil(duration*fps))
                    control.at_end=False
                    continue
                if not looping:return
                # Seamless loop: continue past the last frame until the motion meets the first run.
                def set_state(values,_payload):
                    x.assign(np.ascontiguousarray(values,dtype=np.float32))
                    return output(_payload['sample'])
                def snapshot():
                    return dict(x=x.numpy().copy(),v=v.numpy().copy(),elapsed=state['elapsed'])
                def restore(saved):
                    x.assign(saved['x']);v.assign(saved['v']);state['elapsed']=saved['elapsed']
                status=carry_loop(recorder=recorder,steer_frames=loop_frames,advance=step,wrap=lambda:step(0,wrap=True),
                                  get_state=x.numpy,set_state=set_state,snapshot=snapshot,restore=restore,
                                  publish=on_frame,preview=lambda p:on_frame(dict(p,preview=True)))
                if status is None:return
                on_frame(dict(payload,loop_status=status,loop_closed=True))
                return
            payload=step(frame)
            if payload is None:return
            if recorder is not None:recorder.record(payload['sample'],x.numpy())
            on_frame(payload)
            frame+=1

