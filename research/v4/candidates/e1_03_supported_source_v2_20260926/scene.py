"""Single justified source-reach correction: an actual15mm workshop rubber mat."""
from pathlib import Path
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.e1_03_screwdriver_toolbox_20260926 import scene as previous
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PROTOCOL='e1-03-screwdriver-toolbox-supported-source-t5-assisted-v42-v2'
BINDINGS=(
 tabletop.ObjectBinding('blunt_tool_1',str(previous.HERE/'assets/screwdriver.xml'),'L',z_offset=.031,grasp_local=(-.075,0.,.018)),
 previous.BINDINGS[1],
 tabletop.ObjectBinding('workpad_1',str(HERE/'assets/workpad.xml'),'L',z_offset=.0075),
)
class SupportedToolboxEnv(previous.ToolboxFixtureEnv):
 def __init__(self,*args,**kwargs):
  with patch.object(previous,'BINDINGS',BINDINGS):super().__init__(*args,**kwargs)
class E1ScrewdriverToolboxScene(previous.E1ScrewdriverToolboxScene):
 backend_class=SupportedToolboxEnv
 protocol_version=PROTOCOL
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  for name in self.backend.objects_dict['workpad_1'].contact_geoms:
   gid=self.backend.sim.model.geom_name2id(name);self.backend.sim.model.geom_group[gid]=1
  if self.backend.use_camera_obs:self.raw=self.backend._get_observations(force_update=True)
