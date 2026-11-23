# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Tetrahedral box cages and topology-preserving trilinear surface embedding."""
from dataclasses import dataclass
import numpy as np


def color_constraints(indices,count):
    masks=[0]*count
    groups=[]
    for number,row in enumerate(indices):
        used=0
        for i in row:used|=masks[int(i)]
        color=0
        while used&(1<<color):color+=1
        while len(groups)<=color:groups.append([])
        groups[color].append(number)
        for i in row:masks[int(i)]|=1<<color
    return [np.asarray(g,dtype=np.int32) for g in groups]


@dataclass
class Cage:
    points:object
    tets:object
    edges:object
    edge_groups:object
    tet_groups:object
    embedding:object
    weights:object
    supports:object
    rest_volumes:object
    minimum_cell:float
    neighbours:object
    neighbour_weights:object


def build_cage(surface,ranges,resolution=10,bottom=True,painted=None):
    if not 4<=resolution<=24:raise ValueError('Cage resolution must be between 4 and 24.')
    surface=np.asarray(surface,dtype=np.float64)
    painted=np.zeros(len(surface),dtype=bool) if painted is None else np.asarray(painted,dtype=bool)
    if painted.shape!=(len(surface),):raise ValueError('Support mask does not match the surface.')
    points,tets,maps,weights,supports=[],[],[],[],[]
    neighbours,neighbour_weights=[],[]
    cellsizes=[]
    for start,end in ranges:
        part=surface[start:end]
        low,high=part.min(axis=0),part.max(axis=0)
        extent=high-low
        if np.any(extent<.001):raise ValueError('Each soft prop needs 3D thickness of at least 1 mm; a flat cloth plane is not a soft prop.')
        counts=np.maximum(2,np.rint(resolution*extent/extent.max()).astype(int))
        step=extent/counts
        cellsizes.append(float(step.min()))
        nx,ny,nz=map(int,counts)
        offset=len(points)
        def index(i,j,k):return offset+(i*(ny+1)+j)*(nz+1)+k
        for i in range(nx+1):
            for j in range(ny+1):
                for k in range(nz+1):
                    points.append(low+step*[i,j,k])
                    supports.append(bool(bottom and j==0))
                    pos=(i,j,k);bounds=(nx,ny,nz)
                    boundary=[a for a in range(3) if pos[a] in (0,bounds[a])]
                    row=[];coefficients=[]
                    for axis in range(3):
                        for direction in (-1,1):
                            other=list(pos);other[axis]+=direction
                            if not 0<=other[axis]<=bounds[axis]:continue
                            # On the cage skin, spread dents tangentially instead
                            # of averaging the top directly towards its underside.
                            if boundary and not any(other[a]==pos[a] for a in boundary):continue
                            row.append(index(*other));coefficients.append(1./step[axis]**2)
                    total=sum(coefficients)
                    coefficients=[c/total for c in coefficients]
                    neighbours.append(row+[index(i,j,k)]*(6-len(row)))
                    neighbour_weights.append(coefficients+[0.]*(6-len(row)))
        # Consistent six-tet subdivision around each cell's long diagonal.
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    a=index(i,j,k);b=index(i+1,j,k);c=index(i,j+1,k);d=index(i+1,j+1,k)
                    e=index(i,j,k+1);f=index(i+1,j,k+1);g=index(i,j+1,k+1);h=index(i+1,j+1,k+1)
                    tets.extend([[a,b,d,h],[a,d,c,h],[a,c,g,h],[a,g,e,h],[a,e,f,h],[a,f,b,h]])
        coord=np.clip((part-low)/step,0,counts)
        cell=np.minimum(np.floor(coord).astype(int),counts-1)
        frac=coord-cell
        for cellpos,fraction in zip(cell,frac):
            row,weight=[],[]
            i,j,k=map(int,cellpos)
            for di in (0,1):
                for dj in (0,1):
                    for dk in (0,1):
                        row.append(index(i+di,j+dj,k+dk))
                        weight.append((fraction[0] if di else 1-fraction[0])*
                                      (fraction[1] if dj else 1-fraction[1])*(fraction[2] if dk else 1-fraction[2]))
            maps.append(row);weights.append(weight)
    points=np.asarray(points,dtype=np.float32)
    tets=np.asarray(tets,dtype=np.int32)
    a,b,c,d=points[tets].transpose(1,0,2)
    volumes=np.sum((b-a)*np.cross(c-a,d-a),axis=1)/6
    negative=volumes<0
    tets[negative,1],tets[negative,2]=tets[negative,2].copy(),tets[negative,1].copy()
    volumes=np.abs(volumes)
    if np.any(volumes<1e-15):raise ValueError('Soft prop cage contains degenerate cells.')
    pairs=np.concatenate([tets[:,[i,j]] for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))])
    edges=np.unique(np.sort(pairs,axis=1),axis=0)
    maps=np.asarray(maps,dtype=np.int32);weights=np.asarray(weights,dtype=np.float32)
    supports=np.asarray(supports,dtype=bool)
    if painted.any():supports[np.unique(maps[painted][weights[painted]>.001])]=True
    return Cage(points,tets,edges,color_constraints(edges,len(points)),color_constraints(tets,len(points)),
                maps,weights,supports,volumes,min(cellsizes),np.asarray(neighbours,dtype=np.int32),
                np.asarray(neighbour_weights,dtype=np.float32))


def embedded(cage,points):
    return np.sum(np.asarray(points)[cage.embedding]*cage.weights[:,:,None],axis=1)
