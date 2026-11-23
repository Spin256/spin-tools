# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Attachments follow corresponding avatar triangles, in each triangle's local frame."""
import numpy as np
import warp as wp


@wp.kernel
def closest(points: wp.array(dtype=wp.vec3), mesh: wp.uint64, radius: float,
            faces: wp.array(dtype=int), uv: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    q = wp.mesh_query_point(mesh, points[i], radius)
    faces[i] = -1
    if q.result:
        faces[i] = q.face
        uv[i] = wp.vec2(q.u, q.v)


def surface_frame(vertices, triangles, faces, uv):
    tri = np.asarray(vertices, float)[np.asarray(triangles)[np.maximum(faces, 0)]]
    x = tri[:,1]-tri[:,0]
    x /= np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    z = np.cross(x,tri[:,2]-tri[:,0])
    z /= np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    y = np.cross(z,x)
    frames = np.stack((x,y,z),axis=1)
    anchor = tri[:,0]*uv[:,0,None]+tri[:,1]*uv[:,1,None]+tri[:,2]*(1-uv.sum(1))[:,None]
    return anchor,frames


class SurfaceAttachment:
    def __init__(self, start, end, body_start, body_end, triangles, reach=3.):
        self.triangles = np.asarray(triangles,int)
        mesh = wp.Mesh(wp.array(np.asarray(body_start,np.float32),dtype=wp.vec3),
                       wp.array(self.triangles.astype(np.int32).ravel(),dtype=int))
        faces=wp.zeros(len(start),dtype=int);uv=wp.zeros(len(start),dtype=wp.vec2)
        wp.launch(closest,len(start),inputs=[wp.array(np.asarray(start,np.float32),dtype=wp.vec3),
                  mesh.id,reach,faces,uv])
        self.faces,self.uv=faces.numpy(),uv.numpy()
        a,fa=surface_frame(body_start,triangles,self.faces,self.uv)
        b,fb=surface_frame(body_end,triangles,self.faces,self.uv)
        self.start=np.einsum('nij,nj->ni',fa,np.asarray(start)-a)
        self.end=np.einsum('nij,nj->ni',fb,np.asarray(end)-b)
        self.bound=(self.faces>=0)&(np.linalg.norm(self.start,axis=1)<=reach)&(np.linalg.norm(self.end,axis=1)<=reach)
        self.bound &= (np.abs(np.linalg.det(fa))>.99)&(np.abs(np.linalg.det(fb))>.99)

    def at(self, body, blend, independent):
        anchor,frame=surface_frame(body,self.triangles,self.faces,self.uv)
        offset=self.start*(1-blend)+self.end*blend
        target=anchor+np.einsum('nji,nj->ni',frame,offset)
        result=np.asarray(independent).copy()
        result[self.bound]=target[self.bound]
        return result.astype(np.float32)
