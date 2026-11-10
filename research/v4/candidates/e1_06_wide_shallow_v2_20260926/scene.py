"""Sole root-authorized normal wide/shallow basin and naturally extended right arm."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.e1_06_glass_dishpan_20260926 import scene as previous
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PROTOCOL='e1-06-glass-wide-shallow-dishpan-extended-arm-t5-assisted-v42-v2'
BINDINGS=(previous.BINDINGS[0],tabletop.ObjectBinding('support_plate_1',str(HERE/'assets/dishpan.xml'),'R',z_offset=.003))
class WideWashpanEnv(previous.GlassDishpanEnv):
 def __init__(self,*args,**kwargs):
  with patch.object(previous,'BINDINGS',BINDINGS):super().__init__(*args,**kwargs)
class E1GlassDishpanScene(previous.E1GlassDishpanScene):
 backend_class=WideWashpanEnv
 protocol_version=PROTOCOL
 e1_plate_xy_m=(.185,.02)
 e1_hand_start_xyz_m=(.28,.33,.938)
 def _target_hand_pose(self):
  # This is private authored human-event layout, never online evidence.
  from research.v3.tasks.e1.private_dynamics import object_bounds
  sim=self.backend.sim;t=self.backend.objects_dict[self.target_name];plate=next(g for g in t.contact_geoms if g.endswith('_plate'));surface=float(object_bounds(sim,[plate])[1][2]);centre=sim.data.body_xpos[sim.model.body_name2id(t.root_body)]
  return np.array([.100,float(centre[1])+.015,surface+.039])
