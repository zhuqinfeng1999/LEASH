"""S201 publication candidate: existing carrier, only portable bottle/basket."""
from dataclasses import asdict
from unittest.mock import patch
from pathlib import Path
import numpy as np
from research.v4.candidates.s2_01_strict_floor_20260927 import probe as strict
from research.v4.candidates.s2_scene_20260924 import scene as base
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,KETCHUP_REGIONS
ROOT=Path(__file__).resolve().parents[4]
BOTTLE=ROOT/'research/v4/candidates/release_ketchup_20260926/ketchup.xml'
BASKET=ROOT/'research/v4/candidates/i1_deeper_basket_145_quat_20260924/open_basket_145mm.xml'
ORIGINAL_REGISTER=base._register

def register():
 ORIGINAL_REGISTER()
 from libero.libero.envs.base_object import OBJECTS_DICT
 from robosuite.models.objects import MujocoXMLObject
 for category,path in [('ketchup',BOTTLE),('basket',BASKET)]:
  def factory(name,joints='default',category=category,path=path):
   obj=MujocoXMLObject(str(path),name=name,joints=joints,obj_type='all',duplicate_collision_geoms=False)
   obj.category_name=category;obj.rotation=(0.,0.);obj.rotation_axis='z';obj.object_properties={'vis_site_names':{}}
   return obj
  OBJECTS_DICT[category]=factory

class Scene(strict.AuditedScene):
 protocol='S201-original-controller+dose+original-CC0-bottle+PolyHaven02-v1'
 def __init__(self,*args,**options):
  with patch.object(base,'_register',register):super().__init__(*args,**options)
 def _install_common_grasp_assist(self):
  super()._install_common_grasp_assist()
  old=self.grasp_assist;m=self.env.sim.model;d=self.env.sim.data
  body=self.env.env.objects_dict['ketchup_1'].root_body
  m.site_pos[m.site_name2id('v42_s2_ketchup_1_grasp')]=[0.,0.,.116]
  zones=[asdict(zone)for zone,*_ in old._entries]
  self.grasp_assist=AuthoredRegionGraspAssist(m._model,d._data,hand_body='gripper0_right_eef',center_site='gripper0_right_grip_site',left_pad_site='v42_s2_left_pad',right_pad_site='v42_s2_right_pad',slot_metadata={'L':{'grasp_zones':zones}},authored_regions={body:KETCHUP_REGIONS})
