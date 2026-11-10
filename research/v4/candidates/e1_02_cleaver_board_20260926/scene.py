"""Exact E1-02 assets/configuration on the established E1 common T5 carrier."""
from pathlib import Path
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.accelerated.e1_facade import E1TabletopFacade, _E1FixtureEnv
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PROTOCOL='e1-02-cleaver-cuttingboard-original-assets-t5-assisted-v42-v1'
BINDINGS=(
 tabletop.ObjectBinding('blunt_tool_1',str(HERE/'assets/cleaver.xml'),'L',z_offset=.014,grasp_local=(-.073,0.,.015)),
 tabletop.ObjectBinding('support_plate_1',str(HERE/'assets/cutting_board.xml'),'R',z_offset=.010),
)
class CleaverFixtureEnv(_E1FixtureEnv):
 def __init__(self,*args,**kwargs):
  with patch.dict(tabletop.PILOTS,{'E1-01':BINDINGS}):super().__init__(*args,**kwargs)
class E1CleaverBoardScene(E1TabletopFacade):
 backend_class=CleaverFixtureEnv
 instruction='Put the cleaver on the cutting board on the right.'
 protocol_version=PROTOCOL

 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  # Render only named, already active asset collision geoms; preserve every
  # contact parameter/mask. Never enable the global collision render group.
  self.visual_render_changes=[]
  for name in (self.item_name,self.target_name):
   for geom in self.backend.objects_dict[name].contact_geoms:
    gid=self.backend.sim.model.geom_name2id(geom)
    self.visual_render_changes.append((geom,int(self.backend.sim.model.geom_group[gid]),1))
    self.backend.sim.model.geom_group[gid]=1
  if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
