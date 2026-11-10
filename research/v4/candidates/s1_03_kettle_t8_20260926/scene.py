"""Kettle geometry configuration only, same qualified S1 physics carrier."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from research.v4.candidates.s1_02_saucepan_t8_20260926 import scene as carrier
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AxisSegment
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTIONS=('Move the kettle onto the wooden board.','Put the kettle in my hands.')
PAN_BINDINGS=tuple(replace(b,xml=str((HERE/'kettle.xml').relative_to(ROOT)),grasp_local=(0.,0.,.146))for b in carrier.PAN_BINDINGS)
REGIONS=(AxisSegment((-.023,0.,.146),(.023,0.,.146)),)
PROTOCOL='v4.2-assisted-authored-region-kettle-top-handle-v1'
ORIGINAL_CREATE=carrier.create
def check_registry():
    r=next(r for r in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if r['id']=='S1-03')
    if tuple(r['instructions'])!=INSTRUCTIONS:raise ValueError('registry drift S1-03')
def create(*args,**kwargs):
    with patch.multiple(carrier,PAN_BINDINGS=PAN_BINDINGS,REGIONS=REGIONS,check_registry=check_registry):return ORIGINAL_CREATE(*args,**kwargs)
class S1KettleFacade(carrier.S1SaucepanFacade):
    task_id='S1-03';grasp_protocol=PROTOCOL
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        self.backend,self.grasp_assist=create(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id);self.env=SimpleNamespace(sim=self.backend.sim,env=self.backend);self.raw=self.backend._get_observations(force_update=True);self.sequence=0;self.assist_events=[];self.round_index=0
    @property
    def instruction(self):return INSTRUCTIONS[self.round_index]
