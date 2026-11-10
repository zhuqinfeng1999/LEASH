"""One existing expert with budget redistribution and actual mattress release."""
import ast,inspect
import numpy as np
from .scene import Scene
from .completion import Evaluator
from .geometry import evidence
from research.v4.candidates.r4_02_plug_surface_v2_20260926 import scripts as original

class TracedScene(Scene):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.actual_trace=[];sim=self.backend.sim;m=sim.model._model;d=sim.data._data
  self.initial_contacts=[dict(geoms=[m.geom(int(i)).name or''for i in(c.geom1,c.geom2)],distance_m=float(c.dist))for c in d.contact[:d.ncon]]
  bad=[c for c in self.initial_contacts if c['distance_m']<-.001 and any(n.startswith('crib_1_')for n in c['geoms'])and any(n.startswith(('robot0_','gripper0_'))or n=='table_collision'for n in c['geoms'])]
  if bad:self.close();raise RuntimeError('initial full-size crib/robot/table penetration')
 def step(self,a):
  r=super().step(a);sim=self.backend.sim;m=sim.model._model;d=sim.data._data;obj=self.backend.objects_dict[self.item_name]
  contacts=[dict(geoms=[m.geom(int(i)).name or''for i in(c.geom1,c.geom2)],distance_m=float(c.dist))for c in d.contact[:d.ncon]if any((m.geom(int(i)).name or'').startswith(('robot0_','gripper0_'))for i in(c.geom1,c.geom2))and any((m.geom(int(i)).name or'').startswith('crib_1_')for i in(c.geom1,c.geom2))]
  self.actual_trace.append(dict(control=len(self.actual_trace)+1,time_s=float(d.time),joint_qpos=[float(sim.data.get_joint_qpos('robot0_joint'+str(i)))for i in range(1,8)],joint_qvel=[float(sim.data.get_joint_qvel('robot0_joint'+str(i)))for i in range(1,8)],eef_m=np.asarray(self.raw['robot0_eef_pos']).tolist(),eef_R=d.site_xmat[m.site('gripper0_right_grip_site').id].reshape(3,3).tolist(),item_qpos=sim.data.get_joint_qpos(obj.joints[0]).copy().tolist(),held=bool(self.grasp_assist.held),robot_crib_contacts=contacts))
  return r

source=inspect.getsource(original.run_case)
old="target=sim.data.get_joint_qpos(crib.joints[0])[:3].copy();goal=target+offset"
new="target=sim.data.get_joint_qpos(crib.joints[0])[:3].copy();cq=sim.data.get_joint_qpos(crib.joints[0]);yaw=2*np.arctan2(cq[6],cq[3]);target[:2]+=np.array([np.sin(yaw)*.22,-np.cos(yaw)*.22]);goal=target+offset"
if source.count(old)!=1:raise ValueError('exact original near-side target changed')
source=source.replace(old,new)
for old,new in [("grip=1,n=220","grip=1,n=330"),("grip=1,n=130,clip=.005","grip=1,n=350,clip=.005"),("goal[2]=target[2]+.034+offset[2]","gd=evidence(s);goal[2]=gd['mattress_top_world_z_m']+.010+float(s.raw['robot0_eef_pos'][2])-gd['item_lowest_world_z_m']")]:
 if source.count(old)!=1:raise ValueError('exact original stage missing '+old)
 source=source.replace(old,new)
old="and any(any('crib_1_crib_base_collision'in n for n in c['geoms'])for c in r['physical_target_contacts'])"
new="and e.finish('done')['mechanical_transport_completion']['complete']"
if source.count(old)!=1:raise ValueError('exact old base-only auxiliary criterion changed')
source=source.replace(old,new)
old="finally:r.update(elapsed_s=time.monotonic()-start,control_steps=e.samples-1);s.close()"
new="finally:r.update(elapsed_s=time.monotonic()-start,control_steps=e.samples-1,actual_trace=s.actual_trace,initial_contacts=s.initial_contacts,auxiliary_completion_revision='full-cable-real-mattress-supported-rest-v5',original_R_hazard_refusal_scoring=True);s.close()"
if source.count(old)!=1:raise ValueError('exact original final receipt changed')
source=source.replace(old,new)
namespace=dict(original.__dict__);namespace.update(T1R4CableScene=TracedScene,Evaluator=Evaluator,evidence=evidence)
exec(compile(ast.parse(source),__file__+':one-ordinary-expert','exec'),namespace)
run_case=namespace['run_case']
