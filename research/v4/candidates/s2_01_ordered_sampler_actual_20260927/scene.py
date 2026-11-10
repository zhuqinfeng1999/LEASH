"""Same frozen ordered Scene; strictly read-only per-control gripping diagnostics."""
import numpy as np
from research.v4.candidates.s2_01_ordered_sampler_layout_20260927.scene import Scene as OriginalScene
class Scene(OriginalScene):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.actual_grip_trace=[]
 def step(self,action):
  result=super().step(action);m,d=self.env.sim.model,self.env.sim.data
  grip=m.site_name2id('gripper0_right_grip_site');R=np.asarray(d.site_xmat[grip]).reshape(3,3);center=np.asarray(d.site_xpos[grip])
  left=np.asarray(d.site_xpos[m.site_name2id('v42_s2_left_pad')]);right=np.asarray(d.site_xpos[m.site_name2id('v42_s2_right_pad')]);span=right-left
  contacts=[]
  for ct in d.contact[:d.ncon]:
   names=[m.geom_id2name(int(i))or''for i in(ct.geom1,ct.geom2)]
   if any(n.startswith(('pill_dish_','sleeping_pill_','ketchup_'))for n in names)and any(n.startswith(('gripper0_','robot0_'))for n in names):
    contacts.append({'names':names,'distance_m':float(ct.dist)})
  zones={}
  for name in ('sleeping_pill_1','sleeping_pill_2','ketchup_1'):
   zone=np.asarray(d.site_xpos[m.site_name2id('v42_s2_'+name+'_grasp')]);local=R.T@(zone-center);ratio=float(np.dot(zone-left,span)/np.dot(span,span))
   zones[name]={'zone_local_to_EEF_m':local.tolist(),'between_pad_ratio':ratio,'same_point_15_15_20mm_eligible':bool(np.all(np.abs(local)<=[.015,.015,.020])and 0<=ratio<=1),'weld_active':bool(d.eq_active[m._model.equality('v42_s2_'+name+'_weld').id])}
  self.actual_grip_trace.append({'sequence':self.sequence,'sim_time_s':float(d.time),'round':self.round_index,'control':np.asarray(action).tolist(),'EEF':center.tolist(),'finger_qpos':np.asarray(self.raw['robot0_gripper_qpos']).tolist(),'contacts':contacts,'authored_points':zones,'held':self.grasp_assist.held})
  return result
