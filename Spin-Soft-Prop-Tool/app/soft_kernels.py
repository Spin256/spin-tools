# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""GPU elastic cage constraints, surface contact and embedding. No body forces."""
import warp as wp

ContactNodes=wp.types.vector(length=24,dtype=wp.int32)
ContactWeights=wp.types.vector(length=24,dtype=wp.float32)


@wp.kernel
def predict(x:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),rest:wp.array(dtype=wp.vec3),
            inverse:wp.array(dtype=float),dt:float,damping:float,recovery:float):
    i=wp.tid()
    if inverse[i]>0.:
        velocity=(v[i]+(rest[i]-x[i])*recovery*recovery*dt)*wp.exp(-damping*dt)
        speed=wp.length(velocity)
        if speed>5.:velocity=velocity*(5./speed)
        v[i]=velocity
        x[i]=x[i]+velocity*dt
    else:x[i]=rest[i]


@wp.kernel
def spread_surface(x:wp.array(dtype=wp.vec3),rest:wp.array(dtype=wp.vec3),inverse:wp.array(dtype=float),
                   neighbours:wp.array2d(dtype=int),weights:wp.array2d(dtype=float),strength:float,
                   output:wp.array(dtype=wp.vec3)):
    i=wp.tid()
    output[i]=x[i]
    if inverse[i]>0.:
        displacement=wp.vec3(0.)
        for k in range(6):
            j=neighbours[i,k]
            displacement=displacement+(x[j]-rest[j])*weights[i,k]
        # Filter displacement, never the original shape: seams and rounded
        # source geometry survive, and an unloaded prop has no shrinkage.
        output[i]=x[i]+(displacement-(x[i]-rest[i]))*strength


@wp.kernel
def edges(x:wp.array(dtype=wp.vec3),inverse:wp.array(dtype=float),pairs:wp.array2d(dtype=int),
          lengths:wp.array(dtype=float),lambdas:wp.array(dtype=float),group:wp.array(dtype=int),
          young:float,dt:float):
    e=group[wp.tid()]
    i,j=pairs[e,0],pairs[e,1]
    delta=x[j]-x[i]
    length=wp.length(delta)
    mass=inverse[i]+inverse[j]
    if length>1.e-9 and mass>0.:
        alpha=1./(young*lengths[e]*dt*dt)
        dl=(-(length-lengths[e])-alpha*lambdas[e])/(mass+alpha)
        lambdas[e]=lambdas[e]+dl
        correction=delta*(dl/length)
        x[i]=x[i]-correction*inverse[i]
        x[j]=x[j]+correction*inverse[j]


@wp.kernel
def volumes(x:wp.array(dtype=wp.vec3),inverse:wp.array(dtype=float),tets:wp.array2d(dtype=int),
            rest:wp.array(dtype=float),lambdas:wp.array(dtype=float),group:wp.array(dtype=int),
            bulk:float,dt:float):
    e=group[wp.tid()]
    i,j,k,l=tets[e,0],tets[e,1],tets[e,2],tets[e,3]
    a,b,c,d=x[i],x[j],x[k],x[l]
    scale=1./(6.*rest[e])
    gj=wp.cross(c-a,d-a)*scale
    gk=wp.cross(d-a,b-a)*scale
    gl=wp.cross(b-a,c-a)*scale
    gi=-gj-gk-gl
    C=wp.dot(b-a,wp.cross(c-a,d-a))*scale-1.
    w=inverse[i]*wp.dot(gi,gi)+inverse[j]*wp.dot(gj,gj)+inverse[k]*wp.dot(gk,gk)+inverse[l]*wp.dot(gl,gl)
    if w>0.:
        alpha=1./(bulk*rest[e]*dt*dt)
        dl=(-C-alpha*lambdas[e])/(w+alpha)
        lambdas[e]=lambdas[e]+dl
        x[i]=a+gi*(inverse[i]*dl)
        x[j]=b+gj*(inverse[j]*dl)
        x[k]=c+gk*(inverse[k]*dl)
        x[l]=d+gl*(inverse[l]*dl)


@wp.kernel
def embed(x:wp.array(dtype=wp.vec3),indices:wp.array2d(dtype=int),weights:wp.array2d(dtype=float),
          surface:wp.array(dtype=wp.vec3)):
    i=wp.tid()
    p=wp.vec3(0.)
    for k in range(8):p=p+x[indices[i,k]]*weights[i,k]
    surface[i]=p


@wp.kernel
def volume_floor(x:wp.array(dtype=wp.vec3),inverse:wp.array(dtype=float),tets:wp.array2d(dtype=int),
                 rest:wp.array(dtype=float),group:wp.array(dtype=int)):
    # A unilateral non-collapse constraint, separate from material compressibility.
    e=group[wp.tid()]
    i,j,k,l=tets[e,0],tets[e,1],tets[e,2],tets[e,3]
    a,b,c,d=x[i],x[j],x[k],x[l]
    scale=1./(6.*rest[e])
    C=wp.dot(b-a,wp.cross(c-a,d-a))*scale-.12
    if C<0.:
        gj=wp.cross(c-a,d-a)*scale
        gk=wp.cross(d-a,b-a)*scale
        gl=wp.cross(b-a,c-a)*scale
        gi=-gj-gk-gl
        w=inverse[i]*wp.dot(gi,gi)+inverse[j]*wp.dot(gj,gj)+inverse[k]*wp.dot(gk,gk)+inverse[l]*wp.dot(gl,gl)
        if w>1.e-12:
            dl=-C/w
            x[i]=a+gi*(inverse[i]*dl)
            x[j]=b+gj*(inverse[j]*dl)
            x[k]=c+gk*(inverse[k]*dl)
            x[l]=d+gl*(inverse[l]*dl)


@wp.kernel
def contact(surface:wp.array(dtype=wp.vec3),indices:wp.array2d(dtype=int),weights:wp.array2d(dtype=float),
            inverse:wp.array(dtype=float),mesh:wp.uint64,margin:float,limit:float,
            corrections:wp.array(dtype=wp.vec3),counts:wp.array(dtype=float)):
    i=wp.tid()
    p=surface[i]
    query=wp.mesh_query_point(mesh,p,1.e6)
    if query.result:
        q=wp.mesh_eval_position(mesh,query.face,query.u,query.v)
        delta=p-q
        distance=wp.length(delta)
        signed=distance*query.sign
        if signed<margin and distance>1.e-9:
            correction=delta*(query.sign/distance)*wp.min(margin-signed,limit)
            denom=float(0.)
            for k in range(8):
                node=indices[i,k]
                denom=denom+weights[i,k]*weights[i,k]*inverse[node]
            if denom>1.e-10:
                for k in range(8):
                    node=indices[i,k]
                    w=weights[i,k]
                    if inverse[node]>0. and w>.00001:
                        wp.atomic_add(corrections,node,correction*(w*inverse[node]/denom))
                        wp.atomic_add(counts,node,1.)


@wp.kernel
def apply_contact(x:wp.array(dtype=wp.vec3),corrections:wp.array(dtype=wp.vec3),counts:wp.array(dtype=float)):
    i=wp.tid()
    if counts[i]>0.:x[i]=x[i]+corrections[i]/counts[i]


@wp.kernel
def reverse_contact(collider:wp.array(dtype=wp.vec3),soft_mesh:wp.uint64,faces:wp.array2d(dtype=int),
                    indices:wp.array2d(dtype=int),weights:wp.array2d(dtype=float),inverse:wp.array(dtype=float),
                    margin:float,limit:float,corrections:wp.array(dtype=wp.vec3),counts:wp.array(dtype=float)):
    p=collider[wp.tid()]
    query=wp.mesh_query_point(soft_mesh,p,1.e6)
    if query.result and query.sign<0.:
        q=wp.mesh_eval_position(soft_mesh,query.face,query.u,query.v)
        delta=p-q
        distance=wp.length(delta)
        if distance>1.e-9:
            # Move the cushion surface inward past the prescribed collider point.
            correction=delta*(wp.min(distance+margin,limit)/distance)
            bary=wp.vec3(query.u,query.v,1.-query.u-query.v)
            nodes=ContactNodes()
            factors=ContactWeights()
            for corner in range(3):
                vertex=faces[query.face,corner]
                for k in range(8):
                    n=corner*8+k
                    nodes[n]=indices[vertex,k]
                    factors[n]=weights[vertex,k]*bary[corner]
            # Triangle corners often share cage nodes. Aggregate before computing
            # inverse effective mass, otherwise shared nodes get over-corrected.
            for a in range(24):
                for b in range(a):
                    if nodes[a]==nodes[b]:
                        factors[b]=factors[b]+factors[a]
                        factors[a]=0.
                        break
            denom=float(0.)
            for a in range(24):
                denom=denom+factors[a]*factors[a]*inverse[nodes[a]]
            if denom>1.e-10:
                for a in range(24):
                    node=nodes[a]
                    w=factors[a]
                    if inverse[node]>0. and w>.00001:
                        wp.atomic_add(corrections,node,correction*(w*inverse[node]/denom))
                        wp.atomic_add(counts,node,1.)


@wp.kernel
def velocity(previous:wp.array(dtype=wp.vec3),x:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),
             inverse:wp.array(dtype=float),dt:float):
    i=wp.tid()
    if inverse[i]>0.:v[i]=(x[i]-previous[i])/dt
    else:v[i]=wp.vec3(0.)


@wp.kernel
def move_mesh(first:wp.array(dtype=wp.vec3),second:wp.array(dtype=wp.vec3),alpha:float,
              positions:wp.array(dtype=wp.vec3)):
    i=wp.tid()
    positions[i]=first[i]*(1.-alpha)+second[i]*alpha
