"""Pure private full-shape normal/tangent clearance; no Scene or executor."""
import math
import numpy as np
from research.v4.candidates.r302_horizontal_short_v_20260928.copy import shape_interval
from research.v4.candidates.r302_source_crest_clearance_saved_copy_20260928.check import radius_upper


def interval(m, d, ids, direction, origin):
    """Exact compiled boxes/meshes/cylinders plus analytic capsule support."""
    out=[]
    direction=np.asarray(direction,float);origin=np.asarray(origin,float)
    for i in ids:
        if int(m.geom_type[i])==3:
            Q=d.geom_xmat[i].reshape(3,3)
            center=float((d.geom_xpos[i]-origin)@direction)
            span=float(m.geom_size[i,0]+m.geom_size[i,1]*abs(Q[:,2]@direction))
            out.append((center-span,center+span))
        else:out.append(shape_interval(m,d,[i],direction,origin))
    if not out:raise ValueError('Empty real-solid support set')
    return min(x[0]for x in out),max(x[1]for x in out)


def front_normal_then_tangent(m,d,held_ids,eef_start,tangent_destination,
                              *,position_stop_m,rotation_stop_rad,
                              measured_forward_excursion_m,clearance_m=.005):
    """Own callers supply their real held state/solids; no can/bowl literals.

    This normal-front microwave topology is named explicitly. Unknown solid
    types or missing named panel/door fail rather than guessing extents.
    The returned two waypoints do not move data, install a weld or authorize
    controls. All-arm continuous collision checking is separately required.
    """
    oven=m.body('microwave_1_main').id;Q=d.xmat[oven].reshape(3,3)
    n=Q[:,1].copy();t=Q[:,0].copy();start=np.asarray(eef_start,float)
    dest=np.asarray(tangent_destination,float)
    held_n=interval(m,d,held_ids,n,start)
    held_t=interval(m,d,held_ids,t,start)
    radius,limiter=radius_upper(m,d,held_ids,start)
    rot=2*radius*math.sin(rotation_stop_rad/2)
    pos=max(float(position_stop_m),float(measured_forward_excursion_m))
    margin=float(clearance_m)+pos+rot
    tangent_sweep=(min(start@t,dest@t)+held_t[0]-margin,
                   max(start@t,dest@t)+held_t[1]+margin)
    rows=[];limits=[]
    for suffix in ('g3','g4','g5','g6','g7','g8','g14','g15','g16','g17'):
        i=m.geom('microwave_1_'+suffix).id
        ns=interval(m,d,[i],n,np.zeros(3));ts=interval(m,d,[i],t,np.zeros(3))
        separation=max(ts[0]-tangent_sweep[1],tangent_sweep[0]-ts[1])
        disjoint=separation>0
        bound=float(ns[0]-held_n[1]-margin)
        rows.append(dict(name=m.geom(i).name,id=i,body=m.body(int(m.geom_bodyid[i])).name,
            type=int(m.geom_type[i]),normal_interval_m=ns,tangent_interval_m=ts,
            full_swept_tangent_separation_m=float(separation),
            normal_blocker=not disjoint,required_EEF_normal_upper_m=bound))
        if not disjoint:limits.append((bound,m.geom(i).name))
    if not limits:raise ValueError('No overlapping actual device solid')
    bound,limiting=min(limits)
    normal=min(float(start@n),bound)
    retreat=start+n*(normal-float(start@n))
    tangent=retreat+t*(float(dest@t)-float(retreat@t))
    # World height is preserved exactly by the existing offline expert; the
    # actual appliance axes have microradian settling tilt, reported here.
    retreat[2]=start[2];tangent[2]=start[2]
    return dict(normal_inward=n.tolist(),tangent_axis=t.tolist(),held_normal_relative_m=held_n,
        held_tangent_relative_m=held_t,full_held_radius_m=radius,radius_limiting_geom=limiter,
        original_position_stop_m=float(position_stop_m),original_rotation_stop_rad=float(rotation_stop_rad),
        measured_forward_normal_excursion_m=float(measured_forward_excursion_m),
        position_allowance_m=pos,rotation_allowance_m=rot,geometric_clearance_m=float(clearance_m),
        total_normal_allowance_m=margin,full_swept_tangent_interval_m=tangent_sweep,
        device_solids=rows,limiting_normal_solid=limiting,
        unique_EEF_normal_upper_m=bound,starting_EEF_normal_m=float(start@n),
        normal_retreat_distance_m=float(np.linalg.norm(retreat-start)),
        retreat_EEF_world_m=retreat.tolist(),tangent_EEF_world_m=tangent.tolist(),
        unchanged_world_height_m=float(start[2]),private_offline_only=True,
        full_arm_or_control_qualification=False)
