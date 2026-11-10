"""Finite actual neutral stability; no pose reset or orientation lock."""
import time,numpy as np,mujoco

def settle(s,controls=240):
 sim=s.backend.sim;m,d=sim.model._model,sim.data._data;obj=s.backend.objects_dict[s.item_name];bid=m.body(obj.root_body).id;q0=sim.data.get_joint_qpos(obj.joints[0]).copy();R0=d.xmat[bid].reshape(3,3).copy();start=time.monotonic();rows=[];error=None
 for n in range(controls):
  if time.monotonic()-start>60: error='neutral finite60s wall';break
  a=np.zeros(7,np.float32);a[6]=-1;s.step(a);q=sim.data.get_joint_qpos(obj.joints[0]).copy();cos=np.clip((np.trace(R0.T@d.xmat[bid].reshape(3,3))-1)/2,-1,1);angle=float(np.degrees(np.arccos(cos)));translation=float(np.linalg.norm(q[:3]-q0[:3]));contacts=[];zload=0.;blocking=[]
  for i,c in enumerate(d.contact[:d.ncon]):
   names=[m.geom(int(x)).name or''for x in(c.geom1,c.geom2)];pillow=[x.startswith('adult_pillow_1_')for x in names];rack=[x.startswith('r404_upright_wood_')for x in names];robot=[x.startswith(('robot0_','gripper0_'))for x in names];baby=[x.startswith('crib_1_doll_')for x in names]
   if not any(pillow+rack):continue
   force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);world=c.frame.reshape(3,3).T@force[:3];fz=(world[2]if pillow[1]else-world[2])if any(pillow)else 0.;support=any(pillow)and any(rack)
   if support:zload+=fz
   row=dict(geoms=names,dist_m=float(c.dist),pos_m=c.pos.tolist(),normal_force_N=float(force[0]),force_on_pillow_z_N=float(fz),support=support);contacts.append(row)
   if c.dist<-.001 and((any(robot)and any(pillow+rack))or(any(pillow)and(any(baby)or 'table_collision'in names))):blocking.append(row)
  vel=sim.data.get_joint_qvel(obj.joints[0]).copy();rows.append(dict(control=n+1,sim_time_s=float(d.time),pillow_qpos=q.tolist(),qvel=vel.tolist(),translation_from_unsettled_m=translation,rotation_from_upright_deg=angle,support_load_z_N=zload,contacts=contacts,blockers=blocking))
  if translation>.02 or angle>10 or blocking or s.grasp_assist.held:error='source macro tip/translation, external contact or unexpected attach';break
 q=sim.data.get_joint_qpos(obj.joints[0]).copy();r=dict(passed=error is None and len(rows)==controls,error=error,controls=len(rows),initial_qpos=q0.tolist(),settled_qpos=q.tolist(),rows=rows,limits='stand engineering stability only: reject >20mm translation/>10deg macro tip; not benchmark scoring threshold',elapsed_s=time.monotonic()-start)
 if r['passed']:
  s.initial_item_z=float(q[2]);s.settled_reset_evidence=dict(initial_qpos=q0.tolist(),settled_qpos=q.tolist(),physical_neutral_controls=controls,lift_baseline_change_m=float(q[2]-q0[2]),reason='actual small physical settling before evaluator/planner reset; no pose rewrite');s.neutral_controls=controls
 return r
