# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Small surface corrections with symmetric local-contact projection."""
import warp as wp


@wp.func
def triangle_bary(p:wp.vec3,a:wp.vec3,b:wp.vec3,c:wp.vec3):
    ab=b-a;ac=c-a;ap=p-a
    d1=wp.dot(ab,ap);d2=wp.dot(ac,ap)
    if d1<=0. and d2<=0.:return wp.vec3(1.,0.,0.)
    bp=p-b;d3=wp.dot(ab,bp);d4=wp.dot(ac,bp)
    if d3>=0. and d4<=d3:return wp.vec3(0.,1.,0.)
    vc=d1*d4-d3*d2
    if vc<=0. and d1>=0. and d3<=0.:
        v=d1/(d1-d3);return wp.vec3(1.-v,v,0.)
    cp=p-c;d5=wp.dot(ab,cp);d6=wp.dot(ac,cp)
    if d6>=0. and d5<=d6:return wp.vec3(0.,0.,1.)
    vb=d5*d2-d1*d6
    if vb<=0. and d2>=0. and d6<=0.:
        w=d2/(d2-d6);return wp.vec3(1.-w,0.,w)
    va=d3*d6-d5*d4
    if va<=0. and d4-d3>=0. and d5-d6>=0.:
        w=(d4-d3)/((d4-d3)+(d5-d6));return wp.vec3(0.,1.-w,w)
    denom=va+vb+vc
    if wp.abs(denom)<1.e-16:return wp.vec3(1.,0.,0.)
    v=vb/denom;w=vc/denom
    return wp.vec3(1.-v-w,v,w)


@wp.kernel
def relax(x:wp.array(dtype=wp.vec3),rest:wp.array(dtype=wp.vec3),mobility:wp.array(dtype=float),
          neighbours:wp.array2d(dtype=int),spread:wp.array(dtype=float),recovery:float,depth:float,out:wp.array(dtype=wp.vec3)):
    i=wp.tid();delta=x[i]-rest[i]
    if mobility[i]<=0.:out[i]=rest[i]
    else:
        average=wp.vec3(0.);count=float(0.)
        for k in range(neighbours.shape[1]):
            j=neighbours[i,k]
            if j>=0:average=average+(x[j]-rest[j]);count=count+1.
        if count>0.:delta=delta+(average/count-delta)*spread[0]
        delta=delta*(1.-recovery)
        if wp.length(delta)>depth:delta=wp.normalize(delta)*depth
        out[i]=rest[i]+delta


@wp.kernel
def self_contacts(x:wp.array(dtype=wp.vec3),normals:wp.array(dtype=wp.vec3),faces:wp.array2d(dtype=int),
                  mesh:wp.uint64,mobility:wp.array(dtype=float),exclude:wp.array2d(dtype=int),radius:float,margin:float,
                  active_bounds:wp.array(dtype=wp.vec3),
                  rest:wp.array(dtype=wp.vec3),rest_normal:wp.array(dtype=wp.vec3),
                  shell:wp.array(dtype=int),ghost:wp.array(dtype=int),paint_regions:wp.array(dtype=int),
                  changes:wp.array(dtype=wp.vec3),counts:wp.array(dtype=float)):
    i=wp.tid();p=x[i];bounds=wp.vec3(radius,radius,radius)
    active_low=active_bounds[0];active_high=active_bounds[1]
    if p[0]<active_low[0] or p[1]<active_low[1] or p[2]<active_low[2] or p[0]>active_high[0] or p[1]>active_high[1] or p[2]>active_high[2]:
        return
    query=wp.mesh_query_aabb(mesh,p-bounds,p+bounds)
    face=int(0);best=radius;chosen=int(-1);bary=wp.vec3(0.);normal=wp.vec3(0.);penetration=float(0.)
    while wp.mesh_query_aabb_next(query,face):
        a=faces[face,0];b=faces[face,1];c=faces[face,2]
        nearby=bool(False)
        for k in range(exclude.shape[1]):
            j=exclude[i,k]
            if j==a or j==b or j==c:nearby=True
        if not nearby:
            n=wp.cross(x[b]-x[a],x[c]-x[a]);length=wp.length(n)
            if length>1.e-12:
                n=n/length
                if wp.dot(normals[i],n)<-.15:
                    weights=triangle_bary(p,x[a],x[b],x[c]);q=x[a]*weights[0]+x[b]*weights[1]+x[c]*weights[2]
                    distance=wp.length(p-q);signed=wp.dot(p-q,n)
                    mass=mobility[i]+mobility[a]*weights[0]*weights[0]+mobility[b]*weights[1]*weights[1]+mobility[c]*weights[2]*weights[2]
                    if signed<margin and distance<best and mass>1.e-8:
                        posed_contact = bool(False)
                        if shell.shape[0] > 0:
                            if shell[i] == shell[a] and ghost[i] + ghost[a] + ghost[b] + ghost[c] == 0:
                                separate_regions = bool(False)
                                if paint_regions.shape[0] > 0:
                                    vertex_region = paint_regions[i]
                                    face_region = wp.max(paint_regions[a], wp.max(paint_regions[b], paint_regions[c]))
                                    separate_regions = vertex_region >= 0 and face_region >= 0 and vertex_region != face_region
                                if not separate_regions:
                                    reference_delta = rest[i] - (rest[a]*weights[0] + rest[b]*weights[1] + rest[c]*weights[2])
                                    # Keep existing contact for explicit ghosts and separate paint islands.
                                    posed_contact = wp.dot(reference_delta, rest_normal[face]) < margin
                                    if paint_regions.shape[0] > 0:
                                        # A triangle plane can cross a distant point without the
                                        # two surfaces touching. Only resolve nearby new crossings.
                                        posed_contact = posed_contact or distance > wp.max(.002, 2. * margin)
                        if not posed_contact:
                            chosen=face;best=distance;bary=weights;normal=n;penetration=margin-signed
    if chosen>=0:
        a=faces[chosen,0];b=faces[chosen,1];c=faces[chosen,2]
        denom=mobility[i]+mobility[a]*bary[0]*bary[0]+mobility[b]*bary[1]*bary[1]+mobility[c]*bary[2]*bary[2]
        correction=normal*(wp.min(penetration,.002)/denom)
        if mobility[i]>0.:
            wp.atomic_add(changes,i,correction*mobility[i]);wp.atomic_add(counts,i,1.)
        for k in range(3):
            vertex=faces[chosen,k]
            if mobility[vertex]>0. and bary[k]>.00001:
                wp.atomic_add(changes,vertex,-correction*(mobility[vertex]*bary[k]));wp.atomic_add(counts,vertex,1.)


@wp.kernel
def opening_contact(closed:wp.array(dtype=wp.vec3),wide:wp.array(dtype=wp.vec3),
                    mesh:wp.uint64,margin:float,amounts:wp.array(dtype=float),required:wp.array(dtype=float)):
    i=wp.tid()
    previous=amounts[i]
    travel=wide[i]-closed[i]
    length=wp.length(travel)
    if length<1.e-6:
        return
    direction=travel/length
    start=wp.mesh_query_point_sign_winding_number(mesh,closed[i],1000.)
    if start.result:
        q=wp.mesh_eval_position(mesh,start.face,start.u,start.v)
        delta=closed[i]-q
        distance=wp.length(delta)
        if start.sign<0.:
            # A DAZ collider may have an open back and still enclose the lip
            # locally. Require an actual exit along the authored movement.
            exit_hit=wp.mesh_query_ray(mesh,closed[i],direction,2.*length)
            if exit_hit.result and exit_hit.sign<0.:
                required[i]=wp.max(required[i],wp.min(1.,(exit_hit.t+margin)/length))
        elif distance<margin and distance>1.e-8:
            away=wp.dot(direction,delta)/distance
            if away>.1:
                required[i]=wp.max(required[i],wp.min(1.,(margin-distance)/(away*length)))
    if previous>0.:
        # The closed position can be clear while the lip's return path still
        # crosses the side of an object that is partway through the opening.
        current=closed[i]+travel*previous
        hit=wp.mesh_query_ray(mesh,current,-direction,previous*length)
        if hit.result:
            hold=previous
            if hit.sign>0.:
                hold=wp.min(previous,previous+(margin-hit.t)/length)
            required[i]=wp.max(required[i],hold)


@wp.kernel
def opening_tug(points:wp.array(dtype=wp.vec3),mesh:wp.uint64,motion:wp.vec3,band:float,
                weight:wp.array(dtype=float),tug:wp.array(dtype=wp.vec3)):
    i=wp.tid()
    query=wp.mesh_query_point(mesh,points[i],band)
    if query.result:
        q=wp.mesh_eval_position(mesh,query.face,query.u,query.v)
        distance=wp.length(points[i]-q)
        fraction=distance/band
        influence=1.-fraction*fraction
        if influence>weight[i]:
            normal=wp.mesh_eval_face_normal(mesh,query.face)
            tug[i]=motion-normal*wp.dot(motion,normal)
            weight[i]=influence


@wp.func
def opening_gap(mesh:wp.uint64,point:wp.vec3):
    query=wp.mesh_query_point_sign_winding_number(mesh,point,1000.)
    gap=float(1000.)
    if query.result:
        surface=wp.mesh_eval_position(mesh,query.face,query.u,query.v)
        gap=query.sign*wp.length(point-surface)
    return gap


@wp.kernel
def loop_clearance(before:wp.array(dtype=wp.vec3),after:wp.array(dtype=wp.vec3),
                   mesh:wp.uint64,margin:float,blocked:wp.array(dtype=int)):
    i=wp.tid()
    travel=after[i]-before[i]
    length=wp.length(travel)
    if length>1.e-7:
        gap=opening_gap(mesh,before[i])
        # Lips resting in contact may slide within the clearance band; only going deeper or crossing blocks.
        if opening_gap(mesh,after[i])<wp.min(gap,.25*margin)-1.e-5:
            wp.atomic_max(blocked,0,1)
        elif gap>=0. and length>1.e-5:
            hit=wp.mesh_query_ray(mesh,before[i],travel/length,length-1.e-5)
            if hit.result and hit.sign>0.:
                wp.atomic_max(blocked,0,1)


@wp.kernel
def opening_finish_clearance(points:wp.array(dtype=wp.vec3),delta:wp.array(dtype=wp.vec3),
                             mesh:wp.uint64,margin:float,scale:wp.array(dtype=float)):
    i=wp.tid()
    length=wp.length(delta[i])
    if length>1.e-8 and scale[i]>0.:
        gap=opening_gap(mesh,points[i])
        minimum=wp.min(gap,margin)-1.e-6
        maximum=float(1000.)
        if gap>=0. and gap<=wp.max(.001,2.*margin):
            maximum=wp.max(gap,margin)+.00035
        direction=delta[i]/length
        factor=scale[i]
        if gap>=0.:
            hit=wp.mesh_query_ray(mesh,points[i],direction,length*factor)
            if hit.result and hit.sign>0.:
                factor=wp.max(0.,(hit.t-1.e-6)/length)
        for attempt in range(4):
            adjusted=opening_gap(mesh,points[i]+delta[i]*factor)
            if minimum<=adjusted and adjusted<=maximum:
                scale[i]=factor
                return
            factor=factor*.5
        scale[i]=0.


@wp.kernel
def fixed_contacts(x:wp.array(dtype=wp.vec3),mesh:wp.uint64,mobility:wp.array(dtype=float),margin:float,
                   seal:wp.array(dtype=int),entry:wp.array(dtype=wp.vec3),depth:float,
                   opening:wp.array(dtype=int),
                   changes:wp.array(dtype=wp.vec3),counts:wp.array(dtype=float)):
    i=wp.tid()
    if mobility[i]>0.:
        query=wp.mesh_query_point(mesh,x[i],1000.)
        authored=bool(False)
        if opening.shape[0]>0:
            authored=opening[i]>0
        if authored:
            query=wp.mesh_query_point_sign_winding_number(mesh,x[i],1000.)
        if query.result:
            q=wp.mesh_eval_position(mesh,query.face,query.u,query.v);delta=x[i]-q;distance=wp.length(delta)
            sign=query.sign
            if distance*sign<margin:
                correction=wp.vec3(0.)
                if distance>1.e-9:
                    correction=delta*(sign/distance)*wp.min(margin-distance*sign,.002)
                if seal.shape[0]>0 and query.sign<0.:
                    if seal[i]>0:
                        axis=entry[1]
                        radial=x[i]-entry[0]
                        radial=radial-axis*wp.dot(radial,axis)
                        if wp.dot(axis,axis)>.5 and wp.length(radial)>1.e-7:
                            direction=wp.normalize(radial)
                            hit=wp.mesh_query_ray(mesh,x[i],direction,depth)
                            if hit.result and hit.sign<0.:
                                # Part the lips across the path of entry, even on
                                # a blunt tip whose nearest surface pushes axially.
                                correction=direction*wp.min(hit.t+margin,.002)
                if wp.dot(correction,correction)>0.:
                    wp.atomic_add(changes,i,correction);wp.atomic_add(counts,i,1.)


@wp.kernel
def reverse_fixed(collider:wp.array(dtype=wp.vec3),soft_mesh:wp.uint64,faces:wp.array2d(dtype=int),
                  mobility:wp.array(dtype=float),margin:float,opening:wp.array(dtype=int),
                  changes:wp.array(dtype=wp.vec3),counts:wp.array(dtype=float)):
    p=collider[wp.tid()];query=wp.mesh_query_point(soft_mesh,p,1000.)
    if query.result and query.sign<0.:
        q=wp.mesh_eval_position(soft_mesh,query.face,query.u,query.v);delta=p-q;distance=wp.length(delta)
        bary=wp.vec3(query.u,query.v,1.-query.u-query.v)
        denom=float(0.)
        near=distance<=wp.max(.002,2.*margin)
        for k in range(3):
            i=faces[query.face,k]
            eligible=bool(True)
            if not near and opening.shape[0]>0:
                eligible=opening[i]==0
            if eligible:
                denom=denom+mobility[i]*bary[k]*bary[k]
        if denom>1.e-9 and distance>1.e-9:
            correction=delta*(wp.min(distance+margin,.002)/(distance*denom))
            for k in range(3):
                i=faces[query.face,k]
                eligible=bool(True)
                if not near and opening.shape[0]>0:
                    eligible=opening[i]==0
                if mobility[i]>0. and bary[k]>.00001 and eligible:
                    wp.atomic_add(changes,i,correction*(mobility[i]*bary[k]));wp.atomic_add(counts,i,1.)


@wp.kernel
def apply(x:wp.array(dtype=wp.vec3),rest:wp.array(dtype=wp.vec3),mobility:wp.array(dtype=float),
          changes:wp.array(dtype=wp.vec3),counts:wp.array(dtype=float),minimum:wp.array(dtype=float),depth:float):
    i=wp.tid()
    if mobility[i]<=0.:x[i]=rest[i]
    elif counts[i]>0.:
        correction=changes[i]/counts[i]
        step=wp.min(.001,minimum[i]*.15)
        if wp.length(correction)>step:correction=wp.normalize(correction)*step
        delta=x[i]+correction-rest[i]
        if wp.length(delta)>depth:delta=wp.normalize(delta)*depth
        x[i]=rest[i]+delta


@wp.kernel
def limit_gradient(x:wp.array(dtype=wp.vec3),rest:wp.array(dtype=wp.vec3),mobility:wp.array(dtype=float),
                   edges:wp.array2d(dtype=int),lengths:wp.array(dtype=float),stretch:wp.array(dtype=float),group:wp.array(dtype=int)):
    edge=group[wp.tid()];i=edges[edge,0];j=edges[edge,1]
    difference=(x[j]-rest[j])-(x[i]-rest[i])
    length=wp.length(difference);limit=lengths[edge]*stretch[edge]
    mass=mobility[i]+mobility[j]
    if length>limit and mass>1.e-9:
        correction=difference*((length-limit)/(length*mass))
        x[i]=x[i]+correction*mobility[i]
        x[j]=x[j]-correction*mobility[j]


@wp.kernel
def preserve_faces(x:wp.array(dtype=wp.vec3),mobility:wp.array(dtype=float),faces:wp.array2d(dtype=int),
                   normal:wp.array(dtype=wp.vec3),area:wp.array(dtype=float),group:wp.array(dtype=int)):
    face=group[wp.tid()];i=faces[face,0];j=faces[face,1];k=faces[face,2]
    a=x[i];b=x[j];c=x[k];n=normal[face]
    C=wp.dot(wp.cross(b-a,c-a),n)-area[face]*.25
    if C<0. and area[face]>1.e-12:
        gj=wp.cross(c-a,n);gk=wp.cross(n,b-a);gi=-gj-gk
        denom=mobility[i]*wp.dot(gi,gi)+mobility[j]*wp.dot(gj,gj)+mobility[k]*wp.dot(gk,gk)
        if denom>1.e-16:
            correction=-C/denom
            x[i]=a+gi*(mobility[i]*correction)
            x[j]=b+gj*(mobility[j]*correction)
            x[k]=c+gk*(mobility[k]*correction)
