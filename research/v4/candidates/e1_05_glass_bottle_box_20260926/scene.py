"""Registered E1-05 common T5, normal glass flask and open cardboard box."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v3.tasks.e1.private_dynamics import smoothstep
from research.v4.accelerated import tabletop
from research.v4.accelerated.e1_facade import E1TabletopFacade,_E1FixtureEnv
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PROTOCOL='e1-05-glass-flask-cardboard-box-natural-right-arm-t5-assisted-v42-v1'
BINDINGS=(tabletop.ObjectBinding('blunt_tool_1',str(HERE/'assets/glass_bottle.xml'),'L',z_offset=.041,grasp_local=(.073,0.,.012)),tabletop.ObjectBinding('support_plate_1',str(HERE/'assets/cardboard_box.xml'),'R',z_offset=.002),tabletop.ObjectBinding('workpad_1',str(HERE.parent/'e1_03_supported_source_v2_20260926/assets/workpad.xml'),'L',z_offset=.0075))
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
    blend=float(smoothstep((float(sim.data.time)-self.trigger_time_s)/self.hand_duration_s));self.hand=self.hand_start+blend*(self.hand_final-self.hand_start)+[0,0,.080*max(0.,float(np.sin(np.pi*blend)))];self.person.right_hand=self.hand.copy()
   original_update(sim,left_hand)
  self.person.update=physical_person_update
  if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
