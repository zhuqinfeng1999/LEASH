"""Common all-method P105 numerical version: only existing MULTICCD bit16."""
import time
import numpy as np
import mujoco
from research.v4.candidates.p1_05_exact_clearance_v3_20260926.scene import P1DumbbellFacade
from research.v4.candidates.p1_05_exact_clearance_v3_20260926.scorer import P1BasketEvaluator
from research.v4.candidates.p1_05_multiccd_trial_20260926.trial import model_arrays,option_values
class Scene(P1DumbbellFacade):
 protocol='P105_COMMON_MULTICCD16_SAME_ORIGINAL_GEOMETRY'
 def __init__(self,*a,**kw):
  self._case_started=time.monotonic();super().__init__(*a,**kw);m=self.backend.sim.model._model;before=model_arrays(m);old=option_values(m.opt);m.opt.enableflags|=int(mujoco.mjtEnableBit.mjENBL_MULTICCD);new=option_values(m.opt)
  changes={k:[old[k],new[k]]for k in old if old[k]!=new[k]}
  if before!=model_arrays(m)or changes!={'enableflags':[0,16]}:super().close();raise RuntimeError('unexpected physical change')
  self.case_diag=dict(option_changes=changes,model_arrays_equal=True,protocol=self.protocol,steps=0,closed=False,rows=[],final_score=None,assist_events=None,elapsed_s=None)
  self._private_evaluator=P1BasketEvaluator();self._private_evaluator.observe(self)
 def step(self,action):
  if self.case_diag['steps']>=1200:raise TimeoutError('common1200controlcap')
  if time.monotonic()-self._case_started>180:raise TimeoutError('common180scasecap')
  result=super().step(action);self.case_diag['steps']+=1;state=self._private_evaluator.observe(self);sim=self.backend.sim;item=self.backend.objects_dict[self.item_name];v=sim.data.get_joint_qvel(item.joints[0]).copy()
  self.case_diag['rows'].append(dict(time_s=float(sim.data.time),source_qpos=sim.data.get_joint_qpos(item.joints[0]).tolist(),linear_m_s=float(np.linalg.norm(v[:3])),angular_rad_s=float(np.linalg.norm(v[3:])),held=bool(self.grasp_assist.held),harm=bool(state['harm']),geometry=self._private_evaluator.current_geometry))
  return result
 def close(self):
  if hasattr(self,'case_diag')and not self.case_diag['closed']:
   self.case_diag.update(closed=True,final_score=self._private_evaluator.finish('done'),assist_events=list(self.assist_events),elapsed_s=time.monotonic()-self._case_started)
  return super().close()
