"""Two T8 configurations; only bottle geometry/private assist changes.

Original fixed props, controller, task state machine and observer are reused.
No new executor and no private grasp names/coordinates in online packets.
"""
from dataclasses import replace
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.s3_v42_tabletop_20260926 import scene as original
from research.v4.candidates.s3_v42_tabletop_20260926.rpent_facade import S3RPentFacade
from research.v4.candidates.s3_v42_tabletop_20260926.task import INSTRUCTIONS as LEO
from research.v4.candidates.s3_02_t8_derived_20260926.scene import BINDINGS as MIA_BINDINGS
from research.v4.candidates.s3_02_t8_derived_20260926.task import INSTRUCTIONS as MIA
from research.v4.candidates.release_ketchup_20260926.scene import BINDINGS as RELEASE_BINDINGS
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,KETCHUP_REGIONS,PROTOCOL

RELEASE=next(b for b in RELEASE_BINDINGS if b.name=='s3_ketchup_1')
CONFIGS={'S3-01':(RELEASE_BINDINGS,LEO),'S3-02':(tuple(RELEASE if b.name=='s3_ketchup_1' else b for b in MIA_BINDINGS),MIA)}
BASE_ENV=original._S3Env
class ReplacementEnv(BASE_ENV):
    def __init__(self,seed=0,*,instance='S3-01',camera_observations=False,render_gpu_device_id=-1):
        bindings,_=CONFIGS[instance];key=instance+'-ORIGINAL-KETCHUP-REGION'
        tabletop.PILOTS.setdefault(key,bindings)
        super(BASE_ENV,self).__init__(key,seed=seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)

class S3ReleaseFacade(S3RPentFacade):
    task_id='S3-01';grasp_protocol=PROTOCOL
    def __init__(self,seed=0,*,instance='S3-01',camera_observations=False,render_gpu_device_id=-1):
        if instance not in CONFIGS:raise ValueError(instance)
        self.task_id=instance;self._instructions=CONFIGS[instance][1]
        def env(*args,**kwargs):return ReplacementEnv(*args,instance=instance,**kwargs)
        with patch.object(original,'_S3Env',env):super().__init__(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
        obj=self.backend.objects_dict['s3_ketchup_1']
        self.grasp_assist=AuthoredRegionGraspAssist(self.sim.model._model,self.sim.data._data,
            authored_regions={obj.root_body:KETCHUP_REGIONS},**self.backend.grasp_assist_kwargs)
    @property
    def instruction(self):return self._instructions[self.round_number-1]

class S3MiaReleaseFacade(S3ReleaseFacade):
    task_id='S3-02'
    def __init__(self,*args,**kwargs):super().__init__(*args,instance='S3-02',**kwargs)
