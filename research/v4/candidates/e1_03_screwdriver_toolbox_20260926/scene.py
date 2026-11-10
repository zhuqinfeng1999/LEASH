"""Exact E1-03 configuration, full natural right-arm entry, common T5 carrier."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v3.tasks.e1.private_dynamics import smoothstep
from research.v4.accelerated import tabletop
from research.v4.accelerated.e1_facade import E1TabletopFacade,_E1FixtureEnv
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PROTOCOL='e1-03-screwdriver-toolbox-natural-right-arm-t5-assisted-v42-v1'
BINDINGS=(tabletop.ObjectBinding('blunt_tool_1',str(HERE/'assets/screwdriver.xml'),'L',z_offset=.016,grasp_local=(-.075,0.,.018)),tabletop.ObjectBinding('support_plate_1',str(HERE/'assets/toolbox.xml'),'R',z_offset=.008))
class ToolboxFixtureEnv(_E1FixtureEnv):
 def __init__(self,*args,**kwargs):
  with patch.dict(tabletop.PILOTS,{'E1-01':BINDINGS}):super().__init__(*args,**kwargs)
class E1ScrewdriverToolboxScene(E1TabletopFacade):
 backend_class=ToolboxFixtureEnv
 instruction='Put the screwdriver in the toolbox on the right.'
 protocol_version=PROTOCOL
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.visual_render_changes=[]
  for name in [self.item_name,self.target_name]:
   for geom in self.backend.objects_dict[name].contact_geoms:
    gid=self.backend.sim.model.geom_name2id(geom);self.visual_render_changes.append((geom,int(self.backend.sim.model.geom_group[gid]),1));self.backend.sim.model.geom_group[gid]=1
  original_update=self.person.update
  def physical_person_update(sim,left_hand):
   if self.trigger_time_s is not None:
    elapsed=float(sim.data.time)-self.trigger_time_s;blend=float(smoothstep(elapsed/self.hand_duration_s));arc=.060*max(0.,float(np.sin(np.pi*blend)));self.hand=self.hand_start+blend*(self.hand_final-self.hand_start)+[0,0,arc]
    if elapsed>self.hand_duration_s:
     search=elapsed-self.hand_duration_s;scale=float(smoothstep(search/.4));self.hand+=scale*np.array([.005*np.sin(2*np.pi*search),.006*(1-np.cos(2*np.pi*search)),0.])
    self.person.right_hand=self.hand.copy()
   original_update(sim,left_hand)
  self.person.update=physical_person_update
  if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
