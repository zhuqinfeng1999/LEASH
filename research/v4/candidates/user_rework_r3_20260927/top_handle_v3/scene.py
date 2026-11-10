"""Normal authored handle top grasp + real12cm wood device riser, same task."""
from dataclasses import replace
from unittest.mock import patch
import math,xml.etree.ElementTree as ET
from research.v4.accelerated import tabletop
from research.v4.candidates.r3_v42_tabletop_20260925 import scene as original
from research.v4.candidates.user_rework_r3_20260927.scene import FrontFixtureEnv
from research.v4.candidates.r3_v42_visibility_20260926.scene_lower_aim import NEUTRAL_QPOS
SOURCE_XY=(-.36,-.24);OVEN_XY=(.02,0.);RISER_H=.12;HANDLE_ZONE=(0.,.078,.020)
class RiserFixtureEnv(FrontFixtureEnv):
 def _load_model(self):
  super()._load_model()
  p,yaw=tabletop.sample_slot('R',self.layout_seed,object_index=1);p[0]+=.10;p[1]-=.24;yaw-=math.pi/2
  body=ET.SubElement(self.model.worldbody,'body',name='r3_wood_riser',pos=f'{p[0]} {p[1]} {.9+RISER_H/2}',quat=f'{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}')
  ET.SubElement(body,'geom',name='r3_wood_riser_collision',type='box',size=f'.32 .22 {RISER_H/2}',contype='1',conaffinity='1',group='0',rgba='.57 .38 .23 1',friction='.7 .005 .0001')
  ET.SubElement(body,'geom',name='r3_wood_riser_visible',type='box',size=f'.32 .22 {RISER_H/2}',contype='0',conaffinity='0',group='1',rgba='.57 .38 .23 1',mass='0')
class TopHandleR3Scene(original.R3TabletopFacade):
 task_id='R3-01';layout_version='r3-front-facing-top-real-handle-visible12cm-riser-v3'
 def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
  bindings=tuple(replace(b,xy_offset=(SOURCE_XY[0]+.08,SOURCE_XY[1]+.24),grasp_local=HANDLE_ZONE)if b.name==original.ITEM else replace(b,xy_offset=(OVEN_XY[0]+.08,OVEN_XY[1]-.24),z_offset=RISER_H)for b in original._BINDINGS)
  with patch.object(original,'_BINDINGS',bindings),patch.object(original,'_R3FixtureEnv',RiserFixtureEnv),patch.object(tabletop,'PARK_QPOS',NEUTRAL_QPOS):
   super().__init__(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
