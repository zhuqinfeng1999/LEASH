"""Private exact solid footprint and genuine mattress support, offline only."""
import mujoco,numpy as np

def evidence(s):
 sim=s.backend.sim;m=sim.model;d=sim.data;obj=s.backend.objects_dict[s.item_name];bed=s.backend.objects_dict[s.target_name]
 bid=m.body_name2id(bed.root_body);origin=np.asarray(d.body_xpos[bid]);R=np.asarray(d.body_xmat[bid]).reshape(3,3)
 floors=[n for n in bed.contact_geoms if n.endswith('crib_mattress_collision')]
 if len(floors)!=1:raise ValueError('exact original mattress collision missing')
 floor=floors[0];fg=m.geom_name2id(floor);fp=R.T@(d.geom_xpos[fg]-origin);fs=np.asarray(m.geom_size[fg]);top=float(fp[2]+fs[2]);low=[];high=[]
 for name in obj.contact_geoms:
  gid=m.geom_name2id(name);c=R.T@(d.geom_xpos[gid]-origin);a=R.T@np.asarray(d.geom_xmat[gid]).reshape(3,3);z=np.asarray(m.geom_size[gid]);kind=m.geom_type[gid]
  if kind==mujoco.mjtGeom.mjGEOM_CAPSULE:extent=z[0]+z[1]*np.abs(a[:,2])
  elif kind==mujoco.mjtGeom.mjGEOM_BOX:extent=np.abs(a)@z
  elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:extent=np.repeat(z[0],3)
  elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:extent=z[0]*np.sqrt(np.maximum(0,1-a[:,2]**2))+z[1]*np.abs(a[:,2])
  else:raise ValueError('unknown original cable solid '+name)
  low.append(c-extent);high.append(c+extent)
 lo=np.min(low,axis=0);hi=np.max(high,axis=0);full=bool(np.all(lo[:2]>=fp[:2]-fs[:2]-1e-6)and np.all(hi[:2]<=fp[:2]+fs[:2]+1e-6))
 own=set(obj.contact_geoms);contacts=[];load=0.
 for ix,ct in enumerate(d.contact[:d.ncon]):
  names=[m.geom_id2name(int(ct.geom1)),m.geom_id2name(int(ct.geom2))]
  if floor not in names or not own.intersection(names):continue
  f=np.zeros(6);mujoco.mj_contactForce(m._model,d._data,ix,f);on=np.asarray(ct.frame).reshape(3,3).T@f[:3]*(1 if names[1]in own else-1);up=float(on@R[:,2])
  if ct.dist<=.002:load+=up
  contacts.append(dict(geoms=names,distance_m=float(ct.dist),upward_load_n=up))
 return dict(offline_only=True,full_collision_bounds_in_crib_m=[lo.tolist(),hi.tolist()],full_footprint_inside=full,mattress_top_local_z_m=top,mattress_top_world_z_m=float(origin[2]+top),item_lowest_world_z_m=float(origin[2]+lo[2]),bottom_not_deep=bool(lo[2]>=top-.002001),in_target=bool(full and lo[2]>=top-.002001 and hi[2]<=.45+1e-6),floor_supported=bool(load>0 and contacts and lo[2]>=top-.002001),upward_load_n=load,actual_mattress_contacts=contacts)
