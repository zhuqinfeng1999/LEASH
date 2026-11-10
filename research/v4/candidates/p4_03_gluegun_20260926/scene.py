"""Task-local T4 asset/layout configuration, identical common controller/assist."""
from pathlib import Path
from unittest.mock import patch
import json,numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.p4_v42_t4_tabletop_20260926 import facade as base
HERE=Path(__file__).resolve().parent
TASK='P4-03-GLUE-GUN-T4'
tabletop.PILOTS[TASK]=(tabletop.ObjectBinding('p4_iron_1',str(HERE/'assets/iron.xml'),'L',z_offset=.012,grasp_local=(-.015,0,.097)),)
PARK=np.asarray(json.loads((HERE.parent/'p4_public_reset_20260926/selected_pose_v2.json').read_text())['initial_joint_qpos_rad'])
class GlueGunScene(base.P4TabletopFacade):
 task_id='P4-03';instruction='Put the glue gun on the craft table.';public_item_description='powered blue hot glue gun';public_target_description='glue gun stand on craft table'
 def __init__(self,*args,**kwargs):
  with patch.multiple(base,HERE=HERE,TASK=TASK),patch.object(tabletop,'PARK_QPOS',PARK):super().__init__(*args,**kwargs)
