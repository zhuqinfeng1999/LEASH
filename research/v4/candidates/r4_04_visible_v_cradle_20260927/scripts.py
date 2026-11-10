"""Common real source-grasp/lift/first-risk after actual stable reset."""
import time,numpy as np,mujoco
from .scene import Scene
from .neutral import settle
from research.v4.candidates.r4_04_pull_loop_legal_close_20260927 import scripts as previous
from research.v4.candidates.r4_04_real_corner_zone_v2_20260926 import scripts as parent

class SettledScene(Scene):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw)
  try:
   self.neutral_reset_trace=settle(self)
   if not self.neutral_reset_trace['passed']:raise RuntimeError('Actual neutral source stability failed: '+str(self.neutral_reset_trace['error']))
  except BaseException:super().close();raise

SRC=previous.SRC
assert SRC.count('if e.samples>=MAX')==1;SRC=SRC.replace('if e.samples>=MAX','if s.neutral_controls+e.samples>=MAX')
anchor='s.step(np.asarray(a,np.float32));e.observe(s)'
addition="""s.step(np.asarray(a,np.float32));e.observe(s)
  r.setdefault('actual_control_trace',[]).append(dict(sample=e.samples,sim_time_s=float(d.time),action=np.asarray(a,np.float32).tolist(),item_qpos=sim.data.get_joint_qpos(item.joints[0]).copy().tolist(),eef_xyz=np.asarray(s.raw['robot0_eef_pos']).tolist(),held=bool(s.grasp_assist.held)))"""
assert SRC.count(anchor)==1;SRC=SRC.replace(anchor,addition)
anchor='finally:r.update(elapsed_s=time.monotonic()-start,control_steps=e.samples-1);s.close()'
addition="finally:r['assist_events']=s.assist_events;r['neutral_reset_trace']=s.neutral_reset_trace;r['physical_reset_controls']=s.neutral_controls;r.update(elapsed_s=time.monotonic()-start,control_steps=e.samples-1,total_physical_controls=s.neutral_controls+e.samples-1);s.close()"
assert SRC.count(anchor)==1;SRC=SRC.replace(anchor,addition)
ns=dict(vars(parent));ns.update(T1R4AdultPillowScene=SettledScene,geom_points=previous.geomparent.geom_points);exec(compile(SRC,__file__,'exec'),ns);run_case=ns['run_case']
