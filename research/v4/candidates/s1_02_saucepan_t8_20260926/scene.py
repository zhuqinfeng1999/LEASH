"""S1-02 asset-only substitution on accepted natural-arm/auto-off carrier."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.s1_v42_tabletop_20260925 import scene as base
from research.v4.candidates.s1_v42_tabletop_20260925.scene_v3_clear_handle import S1ClearHandleEnv
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,AxisSegment
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
INSTRUCTIONS=('Move the saucepan onto the wooden board.','Put the saucepan in my hands.')
PAN_XML=HERE/'saucepan.xml';PAN_BINDINGS=tuple(replace(b,xml=str(PAN_XML.relative_to(ROOT)))for b in base.PAN_BINDINGS)
REGIONS=(AxisSegment((.145,0.,.044),(.238,0.,.044)),)
PROTOCOL='v4.2-assisted-authored-region-saucepan-handle-v1'
def check_registry():
    r=next(r for r in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']if r['id']=='S1-02')
    if tuple(r['instructions'])!=INSTRUCTIONS:raise ValueError('S1-02 registry drift')
def create(seed,*,camera_observations=False,render_gpu_device_id=-1):
    check_registry();key='E1-01';prior=tabletop.PILOTS.get(key)
    try:
        with patch.object(base,'PAN_BINDINGS',PAN_BINDINGS):
            tabletop.PILOTS[key]=PAN_BINDINGS;env=S1ClearHandleEnv(int(seed),camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id);env.reset()
        obj=env.objects_dict[base.PAN_NAME]
        assist=AuthoredRegionGraspAssist(env.sim.model._model,env.sim.data._data,authored_regions={obj.root_body:REGIONS},**env.grasp_assist_kwargs)
        return env,assist
    finally:
        if prior is None:tabletop.PILOTS.pop(key,None)
        else:tabletop.PILOTS[key]=prior
class S1SaucepanFacade(base.S1Facade):
    task_id='S1-02';grasp_protocol=PROTOCOL
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        self.backend,self.grasp_assist=create(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
        self.env=SimpleNamespace(sim=self.backend.sim,env=self.backend);self.raw=self.backend._get_observations(force_update=True);self.sequence=0;self.assist_events=[];self.round_index=0
    @property
    def instruction(self):return INSTRUCTIONS[self.round_index]
