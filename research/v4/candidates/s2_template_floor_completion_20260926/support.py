"""Private full old-LIBERO bottle/tilted scanned-basket support evidence."""
import mujoco
import numpy as np
from research.v4.accelerated.scoring import PlacementTracker
CONTACT_DISTANCE_M=.002
EPS=1e-6

def basket_evidence(scene):
    sim=scene.env.sim;m=sim.model;d=sim.data;objs=scene.env.env.objects_dict
    bottle=objs['ketchup_1'];basket=objs['basket_1'];sid=m.site_name2id('basket_1_contain_region')
    origin=np.asarray(d.site_xpos[sid]);basis=np.asarray(d.site_xmat[sid]).reshape(3,3);up=basis[:,2]
    own=set(bottle.contact_geoms);other=set(basket.contact_geoms)
    floor_name='basket_1_g1'
    if floor_name not in other or len(other)!=5:raise ValueError('qualified scanned basket changed')
    def box(n):
        j=m.geom_name2id(n)
        if m.geom_type[j]!=mujoco.mjtGeom.mjGEOM_BOX:raise ValueError('original S2 collider is not box: '+n)
        return j,np.asarray(d.geom_xpos[j]),np.asarray(d.geom_xmat[j]).reshape(3,3),np.asarray(m.geom_size[j])
    bottle_boxes=[(n,*box(n))for n in sorted(own)]
    if len(bottle_boxes)!=12:raise ValueError('qualified old S2 bottle changed')
    projections=[]
    for n,j,c,r,h in bottle_boxes:
        center=basis.T@(c-origin);ext=np.abs(basis.T@r)@h
        projections.append(dict(geom=n,min_local_m=(center-ext).tolist(),max_local_m=(center+ext).tolist()))
    lo=np.min([x['min_local_m']for x in projections],axis=0);hi=np.max([x['max_local_m']for x in projections],axis=0)
    size=np.asarray(m.site_size[sid]);full_region_xy=bool(np.all(lo[:2]>=-size[:2]-EPS)and np.all(hi[:2]<=size[:2]+EPS))
    fj,fc,fr,fh=box(floor_name);axis=int(np.argmin(fh));floor_normal=fr[:,axis].copy()
    if floor_normal@up<0:floor_normal=-floor_normal
    if floor_normal@up<.98:raise ValueError('floor normal disagrees with original basket vertical')
    floor_plane=fc+floor_normal*fh[axis]
    floor_level=float(up@(floor_plane-origin));walls=[]
    for n in sorted(other-{floor_name}):
        j,c,r,h=box(n);axis=int(np.argmin(h));normal=r[:,axis].copy()
        if normal@(c-origin)<0:normal=-normal
        point=c-normal*h[axis];bound=float(normal@point)
        bottle_edge=max(float(normal@bc+np.abs(normal@br)@bh)for _,_,bc,br,bh in bottle_boxes)
        walls.append(dict(geom=n,outward_normal_world=normal.tolist(),inner_plane_world_m=point.tolist(),bottle_clearance_m=bound-bottle_edge,inside=bottle_edge<=bound+EPS,rim_height_local_m=float(up@(c-origin)+np.abs(up@r)@h)))
    rim=min(x['rim_height_local_m']for x in walls)
    full_walls=all(x['inside']for x in walls)
    bottom_inside=bool(floor_level-CONTACT_DISTANCE_M-EPS<=lo[2]<=rim+EPS)
    contacts=[];load=0.
    for index,c in enumerate(d.contact[:d.ncon]):
        ns=[m.geom_id2name(int(c.geom1)),m.geom_id2name(int(c.geom2))]
        if floor_name not in ns or not any(n in own for n in ns):continue
        which=0 if ns[0]in own else 1;force=np.zeros(6);mujoco.mj_contactForce(m._model,d._data,index,force)
        on_bottle=np.asarray(c.frame).reshape(3,3).T@force[:3]*(1 if which==1 else -1)
        zload=float(on_bottle@up);qualifies=c.dist<=CONTACT_DISTANCE_M
        if qualifies:load+=zload
        contacts.append(dict(names=ns,distance_m=float(c.dist),qualifies=bool(qualifies),upward_load_n=zload))
    contact=any(x['qualifies']for x in contacts)
    return dict(revision='s2-old-bottle-tilted-basket-full-footprint-floor-v1',offline_only=True,frame_origin_world_m=origin.tolist(),frame_rotation_local_to_world=basis.tolist(),contain_region_half_size_m=size.tolist(),full_bottle_collision_projections=projections,collision_union_min_local_m=lo.tolist(),collision_union_max_local_m=hi.tolist(),full_footprint_in_original_region=full_region_xy,full_footprint_inside_real_walls=full_walls,wall_planes=walls,floor_geom=floor_name,floor_top_local_z_m=floor_level,rim_top_local_z_m=rim,bottom_inside_rim=bottom_inside,in_target=full_region_xy and full_walls and bottom_inside,floor_contacts=contacts,floor_upward_load_n=load,floor_supported=bool(contact and load>0),original_distance_tolerance_m=CONTACT_DISTANCE_M)

class PhysicalBasketTracker(PlacementTracker):
    def __init__(self,provider):super().__init__();self.provider=provider;self.last_geometry=None
    def observe(self,**kwargs):
        self.last_geometry=basket_evidence(self.provider());kwargs['in_target']=self.last_geometry['in_target'];kwargs['supported']=self.last_geometry['floor_supported']
        return super().observe(**kwargs)
