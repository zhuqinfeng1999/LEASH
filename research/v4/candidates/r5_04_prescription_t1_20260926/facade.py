"""Public inherited two-view R5 façade; private actual bottle dimensions."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json
from research.v4.candidates.r5_v42_label_large_20260925.facade import R5TabletopFacade
from research.v4.tasks.r5.template import load_instance
from .scene import create,INSTRUCTION,TASK,PROTOCOL
ROOT=Path(__file__).resolve().parents[4]
class R5PrescriptionFacade(R5TabletopFacade):
 instruction=INSTRUCTION;grasp_protocol=PROTOCOL
 def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
  row=next(r for r in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if r['id']==TASK)
  if row['slots']!={'item':'处方药瓶','label':'LILY · CLASS 2B'}or row['instructions']!=[INSTRUCTION]:raise ValueError('registry mismatch')
  self.backend,self.grasp_assist=create(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id);self.env=SimpleNamespace(sim=self.backend.sim,env=self.backend);base=load_instance('R5-01');item={**base.item,'display_name':'处方药瓶','category':'v4_prescription_bottle','xml':'bottle.xml','footprint_half_xy_m':[.025,.025],'bottom_offset_z_m':0.,'center_offset_z_m':.040,'initial_z_m':.901,'grasp_z_m':.934};self.instance=replace(base,id=TASK,instruction=INSTRUCTION,item=item);self.raw=self.backend._get_observations(force_update=True);self.sequence=0;self.assist_events=[]
