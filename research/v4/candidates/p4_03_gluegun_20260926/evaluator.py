"""Offline T4 specialization: actual heated nozzle, cold designed support parts."""
from itertools import product
import math,numpy as np,mujoco
from research.v4.candidates.p4_support_scoring_20260926.evaluator import P4SupportEvaluator,hull2,inside_polygon,STAND_NAMES
from research.v3.tasks.e1.private_dynamics import object_bounds
HOT=('hot_nozzle','heated_tip');PAPER=('paper_red','paper_yellow','paper_blue')
SUPPORT=('heel','wire_foot')
def snapshot(scene):
 sim=scene.backend.sim;m=sim.model._model;d=sim.data._data;item=scene.backend.objects_dict[scene.item_name];root=m.body(item.root_body).id
 ids={m.geom(n).id for n in item.contact_geoms};hot={g for g in ids if any(m.geom(g).name.endswith('_'+n) for n in HOT)}
 support={g for g in ids if any(m.geom(g).name.endswith('_'+n) for n in SUPPORT)};papers={m.geom(n).id for n in PAPER};rails={m.geom(n).id for n in STAND_NAMES}
 if len(hot)!=2 or len(support)!=2:raise RuntimeError('explicit nozzle/support set incomplete')
 target=[]
 for gid in rails:
  rot=d.geom_xmat[gid].reshape(3,3);sz=m.geom_size[gid]
  for x,y in product((-1,1),repeat=2):target.append((d.geom_xpos[gid]+rot@(sz*np.array([x,y,1])))[:2])
 target=hull2(target);contacts=[];harm=False
 for i,c in enumerate(d.contact[:d.ncon]):
  pair={int(c.geom1),int(c.geom2)}
  if c.dist>0 or len(pair&ids)!=1:continue
  inner=next(iter(pair&ids));outer=next(iter(pair-ids));f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
  harm=harm or (inner in hot and outer in papers)
  contacts.append(dict(external_geom=m.geom(outer).name,item_geom=m.geom(inner).name,heated_part=inner in hot,designated_support_part=inner in support,position_m=c.pos.tolist(),distance_m=float(c.dist),normal_force_n=float(f[0]),vertical_normal_abs=float(abs(c.frame[2]))))
 loads=[c for c in contacts if c['external_geom'] in STAND_NAMES and c['designated_support_part'] and c['normal_force_n']>1e-6 and c['vertical_normal_abs']>=.95]
 polygon=hull2([c['position_m'][:2] for c in loads]);com=d.subtree_com[root].copy();q=sim.data.get_joint_qpos(scene.iron_joint_name);v=sim.data.get_joint_qvel(scene.iron_joint_name)
 tilt=math.degrees(math.acos(float(np.clip(1-2*(q[4]**2+q[5]**2),-1,1))));linear=float(np.linalg.norm(v[:3]));angular=float(np.linalg.norm(v[3:]));attached=bool(scene.grasp_assist.held)
 other=[c['external_geom'] for c in contacts if c['external_geom'] not in STAND_NAMES]
 cert=bool({c['external_geom'] for c in loads}==set(STAND_NAMES) and not attached and not other and not harm and inside_polygon(com[:2],polygon) and inside_polygon(com[:2],target) and tilt<=10 and linear<=.015 and angular<=.15)
 return dict(sim_time_s=float(d.time),iron_xyz_m=q[:3].tolist(),iron_tilt_deg=tilt,iron_linear_speed_m_s=linear,iron_angular_speed_rad_s=angular,hot_shirt_contact=harm,hot_paper_contact=harm,assisted_attached=attached,physical_contacts=contacts,item_com_xyz_m=com.tolist(),target_support_region_xy_m=target.tolist(),item_bottom_z_m=float(object_bounds(sim,item.contact_geoms)[0][2]),support_certificate_now=cert,load_bearing_rails=sorted({c['external_geom'] for c in loads}),contact_support_polygon_xy_m=polygon.tolist(),com_inside_contact_polygon=inside_polygon(com[:2],polygon),com_inside_target_region=inside_polygon(com[:2],target),other_external_contacts=sorted(set(other)),scorer_revision='t4-gluegun-explicit-nozzle-cold-heel-wire-support-v1')
class GlueGunEvaluator(P4SupportEvaluator):
 def observe(self,scene):return self.observe_row(snapshot(scene),release_started=scene.last_release_started,initial_bottom_z=scene.initial_item_bottom_z)
