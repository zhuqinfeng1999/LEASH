"""Offline complete physical case placement; original human hazard unchanged."""
import mujoco
import numpy as np
from research.v4.accelerated.scoring import PlacementTracker
from research.v4.candidates.p1_ma_fallback_20260925.candidate_v4 import P1V4Evaluator

def case_evidence(scene):
    sim=scene.backend.sim;model=sim.model
    item=scene.backend.objects_dict[scene.item_name];target=scene.backend.objects_dict[scene.target_name]
    tid=model.body_name2id(target.root_body);origin=np.asarray(sim.data.body_xpos[tid]);R=np.asarray(sim.data.body_xmat[tid]).reshape(3,3)
    projections=[]
    for name in item.contact_geoms:
        gid=model.geom_name2id(name);r=R.T@np.asarray(sim.data.geom_xmat[gid]).reshape(3,3)
        c=R.T@(sim.data.geom_xpos[gid]-origin);kind=model.geom_type[gid];size=np.asarray(model.geom_size[gid])
        if kind==mujoco.mjtGeom.mjGEOM_MESH:
            mid=int(model.geom_dataid[gid]);start=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);points=np.asarray(model.mesh_vert[start:start+n])@r.T+c
            low=points.min(0);high=points.max(0)
        elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
            half=size[0]+size[1]*np.abs(r[:,2]);low=c-half;high=c+half
        elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
            axis=r[:,2];half=size[0]*np.sqrt(np.maximum(0,1-axis**2))+size[1]*np.abs(axis);low=c-half;high=c+half
        elif kind==mujoco.mjtGeom.mjGEOM_BOX:
            half=np.abs(r)@size;low=c-half;high=c+half
        else:raise ValueError('unsupported exact scissors collision projection '+name)
        projections.append(dict(geom=name,min_local_m=low.tolist(),max_local_m=high.tolist()))
    lo=np.min([p['min_local_m']for p in projections],axis=0);hi=np.max([p['max_local_m']for p in projections],axis=0)
    inside=bool(np.all(lo[:2]>=np.array([-.147,-.097])-1e-6)and np.all(hi[:2]<=np.array([.147,.097])+1e-6)and -.007001<=lo[2]<=.030501)
    floor=next(n for n in target.contact_geoms if n.endswith('_plate'));own=set(item.contact_geoms);contacts=[];load=0.
    for index,c in enumerate(sim.data.contact[:sim.data.ncon]):
        names=[model.geom_id2name(int(c.geom1)),model.geom_id2name(int(c.geom2))]
        if floor not in names or not own.intersection(names):continue
        f=np.zeros(6);mujoco.mj_contactForce(model._model,sim.data._data,index,f)
        force=np.asarray(c.frame).reshape(3,3).T@f[:3]*(1 if names[1]in own else -1)
        upward=float(force@R[:,2]);qualifies=bool(c.dist<=.002)
        if qualifies:load+=upward
        contacts.append(dict(geom1=names[0],geom2=names[1],distance_m=float(c.dist),upward_load_n=upward,qualifies=qualifies))
    support=bool(any(c['qualifies']for c in contacts)and load>0)
    return dict(offline_only=True,full_scissors_bounds_local_m=[lo.tolist(),hi.tolist()],opening_half_xy_m=[.147,.097],floor_top_local_z_m=-.005,rim_top_local_z_m=.0305,in_target=inside,floor_supported=support,floor_contacts=contacts,floor_upward_load_n=load)

class _StrictTracker(PlacementTracker):
    def __init__(self,owner):super().__init__();self.owner=owner
    def observe(self,**kw):
        g=self.owner.current_geometry;kw.update(in_target=g['in_target'],supported=g['floor_supported'])
        return super().observe(**kw)

class P1CaseEvaluator(P1V4Evaluator):
    def __init__(self):super().__init__();self.current_geometry=None;self.placement=_StrictTracker(self)
    def observe(self,scene):
        self.current_geometry=case_evidence(scene);row=super().observe(scene)
        row.update(item_on_target=bool(self.current_geometry['in_target']and self.current_geometry['floor_supported']and not row['attached']),case_completion=self.current_geometry)
        return row
