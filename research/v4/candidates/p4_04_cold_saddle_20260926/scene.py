"""Distinct normal holder only, unchanged actual tool/controller/risk/assist."""
from pathlib import Path
from unittest.mock import patch
import json,numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.p4_v42_t4_tabletop_20260926 import facade as base
from research.v4.candidates.p4_04_soldering_iron_20260926.scene import SolderScene,PARK
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'p4_04_soldering_iron_20260926';TASK='P4-04-GENUINE-COLD-SADDLE-T4'
tabletop.PILOTS[TASK]=(tabletop.ObjectBinding('p4_iron_1',str(OLD/'assets/iron.xml'),'L',z_offset=.012,grasp_local=(-.02,0,.018)),)
class SaddleScene(base.P4TabletopFacade):
 task_id=SolderScene.task_id;instruction=SolderScene.instruction;public_item_description=SolderScene.public_item_description;public_target_description=SolderScene.public_target_description
 def __init__(self,*args,**kwargs):
  with patch.multiple(base,HERE=HERE,TASK=TASK),patch.object(tabletop,'PARK_QPOS',PARK):super().__init__(*args,**kwargs)
