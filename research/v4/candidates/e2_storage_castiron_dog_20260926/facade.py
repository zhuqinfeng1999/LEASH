"""Thin exact E2-04 T5 configuration; original E2-03 dog/controller/event."""
from pathlib import Path
from unittest.mock import patch
import json
from research.v4.accelerated.tabletop import ObjectBinding
from research.v4.candidates.e2_native_event_20260926 import facade as base
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def configuration():
 c=json.loads((HERE/'config_v1.json').read_text());row=next(x for x in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances'] if x['id']==c['id'])
 if row['base_task']!='E2' or row['instructions']!=[c['instruction']] or row['slots']!={'item':c['source']['display_name'],'container':c['target']['display_name'],'pet':c['pet']['display_name']}:raise ValueError('exact E2-04 definition changed')
 return c
class E2StorageCastIronDogScene(base.E2TabletopFacade):
 task_id='E2-04';instruction='Put the cast iron pot into the storage bin.'
 protocol_version='e2-storage-castiron-dog-common-assist-t5-v42-v1'
 def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
  c=configuration();token='E2-04-STORAGE-POT-DOG-T5-CANDIDATE'
  bindings=(ObjectBinding('flowerpot',str(HERE/c['source']['xml']),'L',grasp_local=tuple(c['source']['grasp_local'])),ObjectBinding('cardboard_box',str(HERE/c['target']['xml']),'R',xy_offset=tuple(c['target']['xy_offset'])))
  # Retain carrier-internal geometry names; no names/true sites are exported
  # as an online semantic oracle. Instruction and logged instance are E2-04.
  with patch.object(base,'TASK',token),patch.dict(base.PILOTS,{token:bindings}):
   super().__init__(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
  m=self.backend.sim.model;m.site_rgba[m.site_name2id('gripper0_right_grip_site_cylinder'),3]=0
  if camera_observations:self.raw=self.backend._get_observations(force_update=True)
