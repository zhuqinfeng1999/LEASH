"""Root-authorized expert phase completion fix; no physics/protocol changes."""
import numpy as np,mujoco
from research.v4.candidates.r4_04_visible_v_cradle_20260927 import scripts as previous
from research.v4.candidates.r4_04_visible_v_cradle_20260927.check_initial import geometry

class SettledScene(previous.SettledScene):
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw)
  try:
   self.actual_reset_geometry=geometry(self)
   if not self.actual_reset_geometry['passed']:raise RuntimeError('actual V-cradle reset geometry failed')
  except BaseException:super().close();raise

def actual_state(m,d,s):
 state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
 joints=[]
 for i in range(m.njnt):
  name=m.joint(i).name or''
  if name.startswith(('robot0_joint','gripper0_')):
   qa=int(m.jnt_qposadr[i]);va=int(m.jnt_dofadr[i]);joints.append(dict(name=name,qpos=float(d.qpos[qa]),qvel=float(d.qvel[va]),range=m.jnt_range[i].tolist(),limited=bool(m.jnt_limited[i])))
 contacts=[]
 for i,c in enumerate(d.contact[:d.ncon]):
  names=[m.geom(int(g)).name or''for g in(c.geom1,c.geom2)]
  if any(x.startswith(('robot0_','gripper0_'))for x in names):
   force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);contacts.append(dict(pair=names,distance_m=float(c.dist),position_m=c.pos.tolist(),force=force.tolist()))
 sid=m.site('gripper0_right_grip_site').id
 return dict(sim_time_s=float(d.time),integration_state=state.tolist(),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),ctrl=d.ctrl.tolist(),eef_xyz=d.site_xpos[sid].tolist(),eef_R=d.site_xmat[sid].reshape(3,3).tolist(),joints=joints,robot_contacts=contacts,held=bool(s.grasp_assist.held))

SRC=previous.SRC
old="""   for _ in range(90):
    err=Rotation.from_matrix(R@d.site_xmat[sid].reshape(3,3).T).as_rotvec()
    if np.linalg.norm(err)<np.deg2rad(1):break
    a=np.zeros(7);a[3:6]=np.clip(err/.5,-1,1);a[6]=-1;command(a)
   if np.linalg.norm(err)>=np.deg2rad(1):raise RuntimeError('realedgegrip orientation not reached')"""
new="""   r['orientation_phase']=dict(cap=160,precision_deg=1.,actual_post_control_checks=True,rows=[])
   for _ in range(160):
    err=Rotation.from_matrix(R@d.site_xmat[sid].reshape(3,3).T).as_rotvec()
    if np.linalg.norm(err)<np.deg2rad(1):break
    a=np.zeros(7);a[3:6]=np.clip(err/.5,-1,1);a[6]=-1;command(a)
    post=Rotation.from_matrix(R@d.site_xmat[sid].reshape(3,3).T).as_rotvec()
    r['orientation_phase']['rows'].append(dict(sample=e.samples,angle_deg=float(np.degrees(np.linalg.norm(post))),actual=actual_state(m,d,s)))
    if np.linalg.norm(post)<np.deg2rad(1):break
   err=Rotation.from_matrix(R@d.site_xmat[sid].reshape(3,3).T).as_rotvec()
   r['orientation_phase']['actual_terminal_angle_deg']=float(np.degrees(np.linalg.norm(err)))
   if np.linalg.norm(err)>=np.deg2rad(1):raise RuntimeError('realedgegrip actual post-control orientation not reached')"""
assert SRC.count(old)==1;SRC=SRC.replace(old,new)
old="finally:r['assist_events']=s.assist_events;"
new="finally:r['actual_terminal_state']=actual_state(m,d,s);r['actual_reset_geometry']=s.actual_reset_geometry;r['assist_events']=s.assist_events;"
assert SRC.count(old)==1;SRC=SRC.replace(old,new)
ns=dict(previous.ns);ns.update(T1R4AdultPillowScene=SettledScene,actual_state=actual_state);exec(compile(SRC,__file__,'exec'),ns);run_case=ns['run_case']
