"""Forward protected surfaces, near-side danger, far-side designed support."""
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.p4_v42_t4_tabletop_20260926 import facade as core
from research.v4.candidates.p4_public_reset_20260926.stand_scene import P4VisibleStandFacade
from research.v4.candidates.p4_03_gluegun_20260926 import scene as gun
from research.v4.candidates.p4_03_render_groups_20260926.scene import VisibleGlueGunScene
SOURCE_SHIFT=(-.28,0.,0.);BOARD_SHIFT=(.14,-.20,0.)
class ForwardSurfaceEnv(core._P4Env):
 def _load_model(self):
  super()._load_model()
  for name,delta in [('heat_resistant_left_tray',SOURCE_SHIFT),('tabletop_ironing_board',BOARD_SHIFT)]:
   bodies=self.model.worldbody.findall(f"body[@name='{name}']")
   if len(bodies)!=1:raise RuntimeError('Expected exact static body: '+name)
   b=bodies[0];xyz=np.fromstring(b.get('pos'),sep=' ')+np.array(delta);b.set('pos',' '.join(map(str,xyz)))
def moved_bindings(task):
 return tuple(replace(b,xy_offset=(b.xy_offset[0]+SOURCE_SHIFT[0],b.xy_offset[1]+SOURCE_SHIFT[1]))for b in tabletop.PILOTS[task])
class ForwardIronScene(P4VisibleStandFacade):
 layout_version='p401-forward-board-source-near-danger-v1'
 def __init__(self,*args,**kwargs):
  with patch.object(core,'_P4Env',ForwardSurfaceEnv),patch.dict(tabletop.PILOTS,{core.TASK:moved_bindings(core.TASK)}):super().__init__(*args,**kwargs)
class DetourGlueGunScene(VisibleGlueGunScene):
 layout_version='p403-forward-craft-source-near-paper-far-stand-v1'
 def __init__(self,*args,**kwargs):
  with patch.object(core,'_P4Env',ForwardSurfaceEnv),patch.dict(tabletop.PILOTS,{gun.TASK:moved_bindings(gun.TASK)}):super().__init__(*args,**kwargs)
SCENES={'P4-01':ForwardIronScene,'P4-03':DetourGlueGunScene}
