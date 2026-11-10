"""Own foil source/PARK/grasp with one opt-in front/tall60 device asset."""
import copy,json,math
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.candidates.r3_04_foil_sandwich_20260926 import scene as own
from research.v4.candidates.r3_v42_tabletop_20260925 import scene as carrier
from research.v4.candidates.r3_05_oven_only_prefix_20260928.scene import OwnCartEnv
from research.v4.candidates.r3_tall_cavity_asset_20260928.binding import H as TALL
H=Path(__file__).resolve().parent
ORIGINAL_CONFIG=own.configuration
LAYOUT=json.loads((H/'layout.json').read_text())
def selected_configuration():
 c=copy.deepcopy(ORIGINAL_CONFIG())
 expected='research/v4/candidates/r3_v42_tabletop_20260925/microwave_fullsize_180.xml'
 if c['id']!='R3-04' or c['microwave']['xml']!=expected:raise ValueError('Exact own registered config admission failed')
 c['microwave']['xml']=str((TALL/'microwave_tall60.xml').relative_to(H.parents[3]))
 return c
class TallFrontFoilScene(own.T1R3FoilSandwichScene):
 layout_version='R304-own-foil-board-original-PARK-grasp-front-cart-tall60-v1'
 def __init__(self,*a,**kw):
  with patch.object(own,'configuration',selected_configuration),patch.object(carrier,'_R3FixtureEnv',OwnCartEnv):super().__init__(*a,**kw)
  sim=self.backend.sim;obj=self.backend.objects_dict[self.oven_name];q=np.asarray(sim.data.get_joint_qpos(obj.joints[0]),float).copy()
  q[:3]+=np.asarray(LAYOUT['translation_from_actual_old_oven'])
  w,x,y,z=q[3:];c=math.cos(-math.pi/4);s=math.sin(-math.pi/4)
  q[3:]=[c*w-s*z,c*x-s*y,c*y+s*x,c*z+s*w]
  sim.data.set_joint_qpos(obj.joints[0],q);self._anchor_and_open_microwave()
  if kw.get('camera_observations',False):self.raw=self.backend._get_observations(force_update=True)
