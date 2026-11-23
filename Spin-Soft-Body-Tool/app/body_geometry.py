# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Topology-local selection operations; separate islands never grow into one another."""
import numpy as np


def adjacency(count,faces):
    rows=[set() for _ in range(count)]
    for a,b,c in faces:
        rows[a].update((int(b),int(c)));rows[b].update((int(a),int(c)));rows[c].update((int(a),int(b)))
    return rows


def grow(mask,rows):
    result=mask.copy()
    for i in np.flatnonzero(mask):result[list(rows[i])]=True
    return result


def shrink(mask,rows):
    return np.asarray([bool(mask[i] and all(mask[j] for j in row)) for i,row in enumerate(rows)])


def connected_labels(neighbours, active):
    labels = np.full(len(active), -1, dtype=np.int32)
    count = 0
    for seed in np.flatnonzero(active):
        if labels[seed] >= 0:
            continue
        labels[seed] = count
        pending = [seed]
        while pending:
            for index in neighbours[pending.pop()]:
                if index >= 0 and active[index] and labels[index] < 0:
                    labels[index] = count
                    pending.append(index)
        count += 1
    return labels, count


def triangle_edges(faces):
    return np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)


def open_edge_vertices(count, faces):
    """Mark vertices on edges that belong to only one triangle."""
    edges, incidence = np.unique(triangle_edges(faces), axis=0, return_counts=True)
    border = np.zeros(count, dtype=bool)
    border[edges[incidence == 1]] = True
    return border


def border_band(border, neighbours, rings):
    """Return the open edge and the requested number of adjacent vertex rows."""
    for _ in range(rings - 1):
        border = grow(border, neighbours)
    return border


def surface_constraints(points,faces):
    edges=np.unique(triangle_edges(faces),axis=0)
    lengths=np.linalg.norm(points[edges[:,1]]-points[edges[:,0]],axis=1)
    valid=lengths>1e-8;edges=edges[valid];lengths=lengths[valid]
    minimum=np.full(len(points),.01,dtype=np.float32)
    np.minimum.at(minimum,edges[:,0],lengths);np.minimum.at(minimum,edges[:,1],lengths)
    # Disjoint constraints within a group allow deterministic in-place projection.
    masks=[0]*len(points);groups=[]
    for index,(a,b) in enumerate(edges):
        used=masks[a]|masks[b];color=0
        while used&(1<<color):color+=1
        while len(groups)<=color:groups.append([])
        groups[color].append(index);masks[a]|=1<<color;masks[b]|=1<<color
    return edges.astype(np.int32),lengths.astype(np.float32),[np.asarray(g,dtype=np.int32) for g in groups],minimum


def triangle_groups(count,faces):
    masks=[0]*count;groups=[]
    for index,row in enumerate(faces):
        used=0
        for vertex in row:used|=masks[vertex]
        color=0
        while used&(1<<color):color+=1
        while len(groups)<=color:groups.append([])
        groups[color].append(index)
        for vertex in row:masks[vertex]|=1<<color
    return [np.asarray(g,dtype=np.int32) for g in groups]


def accept_surface_step(candidate,previous,rest,faces,source_cross,maximum,*,repair=None):
    """Backtrack a proposed step until every originally valid face stays usable."""
    source_area=np.linalg.norm(source_cross,axis=1);valid=source_area>1e-10
    def safe(points):
        if not np.isfinite(points).all():return False
        triangles=points[faces]
        cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        return not np.any(valid&((np.sum(cross*source_cross,axis=1)<=0)|(np.linalg.norm(cross,axis=1)<source_area*.08)))
    def cap_depth(points):
        displacement=points-rest;lengths=np.linalg.norm(displacement,axis=1)
        return rest+displacement*np.minimum(1.,maximum/np.maximum(lengths,1e-20))[:,None]
    candidate=cap_depth(candidate)
    if safe(candidate):return candidate,False,False
    if repair is not None:
        repaired=cap_depth(repair(candidate))
        if safe(repaired):return repaired,False,False
    factor=.5
    for attempt in range(12):
        accepted=previous+(candidate-previous)*factor
        if safe(accepted):return accepted,True,False
        factor*=.5
    return previous.copy(),True,True


def brush_region(points,rows,seed,radius):
    selected=np.zeros(len(points),dtype=bool);pending=[int(seed)];selected[seed]=True
    while pending:
        i=pending.pop()
        for j in rows[i]:
            if not selected[j] and np.linalg.norm(points[j]-points[seed])<=radius:
                selected[j]=True;pending.append(j)
    return selected


def prepare(points,faces,softness,rigid,feather=2,*,pin_open_edges=False,pin_edge_rings=1):
    count=len(points);rows=adjacency(count,faces)
    mobility=np.asarray(softness,dtype=np.float32).copy()
    rigid=np.asarray(rigid,dtype=bool)
    if mobility.shape!=(count,) or rigid.shape!=(count,) or not np.isfinite(mobility).all() or np.any((mobility<0)|(mobility>1)):
        raise ValueError('Region masks must match the source vertices and softness must be between zero and one.')
    mobility[rigid]=0
    if pin_open_edges:
        mobility[border_band(open_edge_vertices(count, faces), rows, pin_edge_rings)] = 0
    # Taper the receiving side of each protected boundary without moving it.
    protected=mobility==0;front=protected.copy()
    for ring in range(feather):
        expanded=grow(front,rows);band=expanded&~front
        mobility[band]*=float(ring+1)/(feather+1);front=expanded
    exclusions=[]
    normals=np.zeros_like(points,dtype=np.float32)
    triangles=points[faces];area=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    for corner in range(3):np.add.at(normals,faces[:,corner],area)
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    for i,row in enumerate(rows):
        local={i}|row
        for j in row:local.update(rows[j])
        exclusions.append(sorted(local))
    width=max(map(len,exclusions),default=1)
    exclude=np.full((count,width),-1,dtype=np.int32)
    for i,row in enumerate(exclusions):exclude[i,:len(row)]=row
    width=max(map(len,rows),default=1)
    neighbours=np.full((count,max(1,width)),-1,dtype=np.int32)
    for i,row in enumerate(rows):neighbours[i,:len(row)]=sorted(row)
    return mobility,normals,exclude,neighbours
