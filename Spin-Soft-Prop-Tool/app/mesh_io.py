# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Same coordinate convention as the bridge: OBJ centimeters -> meters."""
import numpy as np


def read_mesh(path):
    vertices,triangles=[],[]
    with open(path,encoding='utf-8-sig') as source:
        for line in source:
            fields=line.partition('#')[0].split()
            if not fields:continue
            if fields[0]=='v':vertices.append([float(v) for v in fields[1:4]])
            elif fields[0]=='f':
                indices=[int(f.split('/')[0]) for f in fields[1:]]
                face=[i-1 if i>0 else len(vertices)+i for i in indices]
                triangles.extend((face[0],a,b) for a,b in zip(face[1:],face[2:]))
    points=np.asarray(vertices,dtype=np.float32)*.01
    faces=np.asarray(triangles,dtype=np.int32)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or not len(faces) or faces.min()<0 or faces.max()>=len(points):
        raise ValueError('Invalid OBJ mesh: '+str(path))
    a,b,c=points[faces].transpose(1,0,2)
    faces=faces[np.linalg.norm(np.cross(b-a,c-a),axis=1)>1e-12]
    if not len(faces):raise ValueError('No usable triangles: '+str(path))
    return points,faces
