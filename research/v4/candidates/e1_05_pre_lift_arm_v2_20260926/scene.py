"""Registered E1-05 common T5, normal glass flask and open cardboard box."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v3.tasks.e1.private_dynamics import smoothstep
from research.v4.accelerated import tabletop
from research.v4.accelerated.e1_facade import E1TabletopFacade,_E1FixtureEnv
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PROTOCOL='e1-05-glass-flask-cardboard-box-pre-lift-right-arm-t5-assisted-v42-v2'
from research.v4.candidates.e1_05_glass_bottle_box_20260926.scene import BINDINGS
class BottleBoxEnv(_E1FixtureEnv):
 def __init__(self,*args,**kwargs):
  with patch.dict(tabletop.PILOTS,{'E1-01':BINDINGS}):super().__init__(*args,**kwargs)
class E1GlassBottleBoxScene(E1TabletopFacade):
 backend_class=BottleBoxEnv
 instruction='Put the glass bottle in the cardboard box on the right.'
 protocol_version=PROTOCOL
 e1_plate_xy_m=(.075,.02)
 def _target_hand_pose(self):return super()._target_hand_pose()+np.array([0,0,.020])
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.visual_render_changes=[]
  for obj in self.backend.objects_dict.values():
   for geom in obj.contact_geoms:
    gid=self.backend.sim.model.geom_name2id(geom);self.visual_render_changes.append((geom,int(self.backend.sim.model.geom_group[gid]),1));self.backend.sim.model.geom_group[gid]=1
  original_update=self.person.update
  def physical_person_update(sim,left_hand):
   if self.trigger_time_s is not None:
    phase=float(np.clip((float(sim.data.time)-self.trigger_time_s)/self.hand_duration_s,0.,1.));blend=float(smoothstep((phase-.25)/.75));raise_blend=float(smoothstep(phase/.25));lower_blend=float(smoothstep((1.-phase)/.30));self.hand=self.hand_start+blend*(self.hand_final-self.hand_start)+[0,0,.100*min(raise_blend,lower_blend)];self.person.right_hand=self.hand.copy()
   original_update(sim,left_hand)
  self.person.update=physical_person_update
  if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
