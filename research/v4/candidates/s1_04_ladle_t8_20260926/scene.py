"""Exact S1-04 configuration; qualified S1 control/auto-off carrier unchanged."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from research.v4.candidates.s1_02_saucepan_t8_20260926 import scene as carrier
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AxisSegment
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTIONS=('Move the ladle onto the wooden board.','Put the ladle in my hand.')
PAN_BINDINGS=tuple(replace(b,xml=str((HERE/'ladle.xml').relative_to(ROOT)),grasp_local=(.1926,0.,.046))for b in carrier.PAN_BINDINGS)
REGIONS=(AxisSegment((.118,0.,.04058),(.248,0.,.05042)),)
PROTOCOL='v4.2-assisted-authored-region-ladle-handle-v1';ORIGINAL_CREATE=carrier.create
def check_registry():
    r=next(r for r in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if r['id']=='S1-04')
    if tuple(r['instructions'])!=INSTRUCTIONS:raise ValueError('S1-04 registry drift')
def create(*args,**kwargs):
    with patch.multiple(carrier,PAN_BINDINGS=PAN_BINDINGS,REGIONS=REGIONS,check_registry=check_registry):return ORIGINAL_CREATE(*args,**kwargs)
class S1LadleFacade(carrier.S1SaucepanFacade):
    task_id='S1-04';grasp_protocol=PROTOCOL
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        self.backend,self.grasp_assist=create(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id);self.env=SimpleNamespace(sim=self.backend.sim,env=self.backend);self.raw=self.backend._get_observations(force_update=True);self.sequence=0;self.assist_events=[];self.round_index=0
    @property
    def instruction(self):return INSTRUCTIONS[self.round_index]
