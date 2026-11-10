"""Single isolated normal lift-over-rim and flat glass base contact candidate."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v3.tasks.e1.private_dynamics import smoothstep,object_bounds
from research.v4.accelerated import tabletop
from research.v4.accelerated.e1_facade import E1TabletopFacade,_E1FixtureEnv
from research.v4.candidates.e1_06_wide_shallow_v2_20260926.scene import BINDINGS as OldBindings
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PROTOCOL='e1-06-real-glass-flat-contact-raised-whole-arm-t5-assisted-v42-v3'
BINDINGS=(tabletop.ObjectBinding('blunt_tool_1',str(HERE/'assets/glass.xml'),'L',z_offset=.0,grasp_local=(0.,0.,.070)),OldBindings[1])
class GlassDishpanEnv(_E1FixtureEnv):
 def __init__(self,*args,**kwargs):
  with patch.dict(tabletop.PILOTS,{'E1-01':BINDINGS}):super().__init__(*args,**kwargs)
class E1GlassDishpanScene(E1TabletopFacade):
 backend_class=GlassDishpanEnv
 instruction='Put the glass in the dishpan on the right.'
 protocol_version=PROTOCOL
 e1_plate_xy_m=(.185,.02)
 e1_hand_start_xyz_m=(.28,.33,.938)
 def _target_hand_pose(self):
  sim=self.backend.sim;t=self.backend.objects_dict[self.target_name];plate=next(g for g in t.contact_geoms if g.endswith('_plate'));surface=float(object_bounds(sim,[plate])[1][2]);centre=sim.data.body_xpos[sim.model.body_name2id(t.root_body)]
  return np.array([.100,float(centre[1])+.015,surface+.039])
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.visual_render_changes=[]
  for obj in self.backend.objects_dict.values():
   for geom in obj.contact_geoms:
    gid=self.backend.sim.model.geom_name2id(geom);self.visual_render_changes.append((geom,int(self.backend.sim.model.geom_group[gid]),1));self.backend.sim.model.geom_group[gid]=1
  original_update=self.person.update
  def physical_person_update(sim,left_hand):
   if self.trigger_time_s is not None:
    elapsed=float(sim.data.time)-self.trigger_time_s;phase=float(np.clip(elapsed/self.hand_duration_s,0.,1.));blend=float(smoothstep((phase-.25)/.75));raise_blend=float(smoothstep(phase/.25));lower_blend=float(smoothstep((1.-phase)/.30));self.hand=self.hand_start+blend*(self.hand_final-self.hand_start)+[0,0,.200*min(raise_blend,lower_blend)]
    if elapsed>self.hand_duration_s:
     t=elapsed-self.hand_duration_s;scale=float(smoothstep(t/.35));self.hand+=scale*np.array([.006*np.sin(2*np.pi*.8*t),.004*(1-np.cos(2*np.pi*.8*t)),0.])
    self.person.right_hand=self.hand.copy()
   original_update(sim,left_hand)
  self.person.update=physical_person_update
  if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
