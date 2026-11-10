"""Private authored-basket/full-bottle geometry; no online coordinates."""
import mujoco
import numpy as np

REVISION = 's3-full-bottle-authored-interior-floor-support-v1'
CONTACT_DISTANCE_M = .002  # Existing frozen support contact distance.
ARITHMETIC_EPS_M = 1e-6    # Floating-point geometry only; not a larger target.


def basket_evidence(scene):
    sim=scene.sim; objects=scene.backend.objects_dict
    bottle=objects['s3_ketchup_1']; basket=objects['s3_basket_1']
    sid=sim.model.site_name2id('s3_basket_1_contain_region')
    origin=np.asarray(sim.data.site_xpos[sid]); rotation=np.asarray(sim.data.site_xmat[sid]).reshape(3,3)
    names=set(basket.contact_geoms)
    def named(suffix):
        found=[n for n in names if n.endswith(suffix)]
        if len(found)!=1: raise ValueError('authored basket geometry missing or ambiguous: '+suffix)
        gid=sim.model.geom_name2id(found[0])
        if sim.model.geom_type[gid]!=mujoco.mjtGeom.mjGEOM_BOX: raise ValueError('basket primitive must be box')
        relative=rotation.T@np.asarray(sim.data.geom_xmat[gid]).reshape(3,3)
        if not np.allclose(relative,np.eye(3),atol=1e-7): raise ValueError('authored basket axes disagree')
        return found[0], rotation.T@(sim.data.geom_xpos[gid]-origin), np.asarray(sim.model.geom_size[gid])
    floor_name,floor_pos,floor_half=named('basket_base')
    walls={suffix:named('basket_wall_'+suffix) for suffix in ('left','right','front','back')}
    left=walls['left'][1][0]+walls['left'][2][0];right=walls['right'][1][0]-walls['right'][2][0]
    front=walls['front'][1][1]+walls['front'][2][1];back=walls['back'][1][1]-walls['back'][2][1]
    floor_top=float(floor_pos[2]+floor_half[2]);rim_height=min(float(v[1][2]+v[2][2]) for v in walls.values())
    if left>=right or front>=back or floor_top>=rim_height: raise ValueError('invalid authored interior')
    projections=[]
    for n in bottle.contact_geoms:
        gid=sim.model.geom_name2id(n);size=np.asarray(sim.model.geom_size[gid])
        center=rotation.T@(sim.data.geom_xpos[gid]-origin)
        relative=rotation.T@np.asarray(sim.data.geom_xmat[gid]).reshape(3,3)
        kind=sim.model.geom_type[gid]
        if kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
            axis=relative[:,2];extent=size[0]*np.sqrt(np.maximum(0,1-axis**2))+size[1]*np.abs(axis)
        elif kind==mujoco.mjtGeom.mjGEOM_BOX: extent=np.abs(relative)@size
        else: raise ValueError('full collision footprint cannot be projected exactly: '+n)
        projections.append(dict(geom=n,min_local_m=(center-extent).tolist(),max_local_m=(center+extent).tolist()))
    if not projections: raise ValueError('bottle collision geometry missing')
    lo=np.min([g['min_local_m']for g in projections],axis=0);hi=np.max([g['max_local_m']for g in projections],axis=0)
    full_xy=bool(lo[0]>=left-ARITHMETIC_EPS_M and hi[0]<=right+ARITHMETIC_EPS_M and lo[1]>=front-ARITHMETIC_EPS_M and hi[1]<=back+ARITHMETIC_EPS_M)
    bottom_inside=bool(floor_top-CONTACT_DISTANCE_M-ARITHMETIC_EPS_M<=lo[2]<=rim_height+ARITHMETIC_EPS_M)
    contacts=[];upward_load=0.;own=set(bottle.contact_geoms)
    for index,c in enumerate(sim.data.contact[:sim.data.ncon]):
        names=[sim.model.geom_id2name(int(c.geom1)),sim.model.geom_id2name(int(c.geom2))]
        if floor_name not in names or not any(n in own for n in names): continue
        which=0 if names[0]in own else 1;force=np.zeros(6)
        mujoco.mj_contactForce(sim.model._model,sim.data._data,index,force)
        on_bottle=np.asarray(c.frame).reshape(3,3).T@force[:3]*(1 if which==1 else -1)
        load=float(on_bottle@rotation[:,2]);qualifies=bool(c.dist<=CONTACT_DISTANCE_M)
        if qualifies: upward_load+=load
        contacts.append(dict(geom1=names[0],geom2=names[1],distance_m=float(c.dist),qualifies_original_distance=qualifies,position_world_m=c.pos.tolist(),force_on_bottle_world_n=on_bottle.tolist(),upward_load_n=load))
    floor_contact=any(c['qualifies_original_distance']for c in contacts)
    return dict(revision=REVISION,offline_only=True,frame_origin_world_m=origin.tolist(),frame_rotation_local_to_world=rotation.tolist(),interior_xy_min_m=[float(left),float(front)],interior_xy_max_m=[float(right),float(back)],floor_top_local_z_m=floor_top,rim_top_local_z_m=rim_height,full_bottle_collision_projections=projections,collision_union_min_local_m=lo.tolist(),collision_union_max_local_m=hi.tolist(),entire_bottle_footprint_inside=full_xy,bottom_inside_rim=bottom_inside,in_target=full_xy and bottom_inside,floor_geom=floor_name,floor_contacts=contacts,floor_contact_within_original_distance=floor_contact,floor_upward_load_n=upward_load,floor_supported=bool(floor_contact and upward_load>0),original_distance_tolerance_m=CONTACT_DISTANCE_M,arithmetic_eps_m=ARITHMETIC_EPS_M)
