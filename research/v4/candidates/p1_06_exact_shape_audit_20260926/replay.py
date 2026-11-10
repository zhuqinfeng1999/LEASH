"""One root-authorized measurement-only exact old prefix; no scene/control edits."""
import hashlib,json
from pathlib import Path
from unittest.mock import patch
import mujoco,numpy as np
from research.v4.candidates.p1_06_reachable_route_v3a_20260926 import script as original
from research.v4.candidates.p1_06_reachable_route_v3a_20260926.preflight import hashes
HERE=Path(__file__).resolve().parent
ORIGINAL=Path('research/v4/candidates/p1_06_reachable_route_v3a_20260926/qualification/safe_seed0.json')
frames=[];geometry={}
class Measured(original.P1WineBucketFacade):
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw)
  m=self.backend.sim.model._model;own=self.backend.objects_dict[self.item_name];prefix=self.item_name+'_'
  for gid in range(m.ngeom):
   name=m.geom(gid).name or''
   if not name.startswith(prefix):continue
   if int(m.geom_group[gid])not in(0,1):continue
   g=dict(name=name,group=int(m.geom_group[gid]),type=int(m.geom_type[gid]),size=m.geom_size[gid].tolist(),collision=name in own.contact_geoms)
   if int(m.geom_type[gid])==int(mujoco.mjtGeom.mjGEOM_MESH):
    mid=int(m.geom_dataid[gid]);start=int(m.mesh_vertadr[mid]);count=int(m.mesh_vertnum[mid]);g['compiled_vertices']=m.mesh_vert[start:start+count].tolist()
   geometry[name]=g
 def step(self,action):
  r=super().step(action);sim=self.backend.sim;m=sim.model;d=sim.data;target=self.backend.objects_dict[self.target_name];tid=m.body_name2id(target.root_body)
  frames.append(dict(time_s=float(d.time),item_qpos=d.get_joint_qpos(self.backend.objects_dict[self.item_name].joints[0]).tolist(),target_qpos=d.get_joint_qpos(target.joints[0]).tolist(),target_origin_m=d.body_xpos[tid].tolist(),target_rotation=d.body_xmat[tid].reshape(3,3).tolist(),geoms={name:dict(origin_m=d.geom_xpos[m.geom_name2id(name)].tolist(),rotation=d.geom_xmat[m.geom_name2id(name)].reshape(3,3).tolist())for name in geometry}))
  return r

def bounds_in_target(definition,pose,target):
 origin=np.asarray(target['target_origin_m']);tR=np.asarray(target['target_rotation']);gR=tR.T@np.asarray(pose['rotation']);c=tR.T@(np.asarray(pose['origin_m'])-origin);size=np.asarray(definition['size']);kind=definition['type'];radial_upper=None;radial_lower=None
 if kind==int(mujoco.mjtGeom.mjGEOM_MESH):points=np.asarray(definition['compiled_vertices'])@gR.T+c
 elif kind==int(mujoco.mjtGeom.mjGEOM_BOX):
  from itertools import product
  points=np.asarray([np.asarray(sign)*size for sign in product((-1,1),repeat=3)])@gR.T+c
 elif kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):
  # Physical circles on both actual cap planes, not XY-AABB corners.
  # The norm is Lipschitz; nearest4096-point circle sample is ≤pi/4096rad
  # away, yielding a rigorously conservative radius upper bound.
  theta=np.arange(4096)*(2*np.pi/4096);rad=size[0];half=size[1]
  cap=np.c_[rad*np.cos(theta),rad*np.sin(theta),np.full(len(theta),half)];points=np.r_[cap,cap*[1,1,-1]]@gR.T+c
  radial_lower=float(np.linalg.norm(points[:,:2],axis=1).max());radial_upper=radial_lower+rad*np.pi/4096
  axis=gR[:,2];vertical_half=rad*np.sqrt(max(0.,1.-axis[2]**2))+half*abs(axis[2]);vertical=[float(c[2]-vertical_half),float(c[2]+vertical_half)]
 elif kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE):
  endpoints=np.asarray([c+size[1]*gR[:,2],c-size[1]*gR[:,2]])
  radial_upper=radial_lower=float(np.linalg.norm(endpoints[:,:2],axis=1).max()+size[0]);vertical=[float(endpoints[:,2].min()-size[0]),float(endpoints[:,2].max()+size[0])];points=endpoints
 else:raise ValueError('unsupported full-shape type '+definition['name'])
 if radial_upper is None:radial_upper=radial_lower=float(np.linalg.norm(points[:,:2],axis=1).max())
 if kind not in(int(mujoco.mjtGeom.mjGEOM_CYLINDER),int(mujoco.mjtGeom.mjGEOM_CAPSULE)):vertical=[float(points[:,2].min()),float(points[:,2].max())]
 return dict(name=definition['name'],collision=definition['collision'],radial_lower_m=radial_lower,radial_upper_m=radial_upper,vertical_bounds_m=vertical,inside_unchanged100mm=radial_upper<=.100+1e-6)
def main():
 output=HERE/'actual_exact_shape_release_rest.json'
 if output.exists():raise ValueError('preserve one exact-prefix instrumentation run')
 before=hashes()
 with patch.object(original,'P1WineBucketFacade',Measured):row=original.probe('safe',0)
 old=json.loads(ORIGINAL.read_text());keys=('steps','error','stages','physical_trace','assist_events','result');equal={k:row[k]==old[k]for k in keys};same=all(equal.values());source_equal=before==hashes()
 report=dict(kind='ONE measurement-only replay of unchanged failed prefix',same_physical_prefix=same,equivalence_fields=equal,parent_source_unchanged=source_equal,parent_source_pins=before,original_raw_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),unchanged_raw_label=old['result']['label'],actual_control_commands=len(frames),evaluator_samples=row['steps'],geometry=geometry,frames=[],passed_proof=False)
 if same and source_equal:
  ids=[next(i for i,r in enumerate(old['physical_trace'])if r['completion']['detached_at_s']is not None),*range(len(frames)-11,len(frames))]
  for index in ids:
   frame=frames[index];original_frame=old['physical_trace'][index];assert abs(frame['time_s']-original_frame['time_s'])<1e-9
   analyses=[bounds_in_target(g,frame['geoms'][n],frame)for n,g in geometry.items()];collision=[a for a in analyses if a['collision']];visual=analyses
   floor_z=original_frame['actual_bucket_geometry']['floor_top_local_z_m'];low=min(a['vertical_bounds_m'][0]for a in collision);upper=max(a['radial_upper_m']for a in visual)
   frame.update(index=index,kind='release'if index==ids[0]else'rest',original_geometry=original_frame['actual_bucket_geometry'],shape_analysis=analyses,all_collision_and_visual_inside100mm=all(a['inside_unchanged100mm']for a in analyses),actual_fullshape_max_radial_upper_m=upper,actual_bottom_above_floor_m=low-floor_z,actual_bottom_within_unchanged_floor_window=-.002<=low-floor_z<=.020);report['frames'].append(frame)
  report['passed_proof']=all(f['all_collision_and_visual_inside100mm']and f['actual_bottom_within_unchanged_floor_window']for f in report['frames'])
  report['interpretation']='conservative AABB false negative proven on exact actualrelease+11rest frames'if report['passed_proof']else'actual shape outside or uncertain; stop, no scoring correction'
 else:report['interpretation']='exact-prefix/source mismatch; stop no geometric conclusion'
 output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in('same_physical_prefix','equivalence_fields','parent_source_unchanged','unchanged_raw_label','actual_control_commands','passed_proof','interpretation')}));print([(f['kind'],f['actual_fullshape_max_radial_upper_m'],f['actual_bottom_above_floor_m'])for f in report['frames']]);return0=0 if same and source_equal else 1
 return return0
if __name__=='__main__':raise SystemExit(main())
