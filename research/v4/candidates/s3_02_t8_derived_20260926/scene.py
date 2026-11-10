"""Configuration-only S3-02 on the same tabletop, assist and T8 carrier."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from research.v4.accelerated import tabletop
from research.v4.candidates.s3_v42_tabletop_20260926 import scene as original
from research.v4.candidates.s3_v42_tabletop_20260926.rpent_facade import S3RPentFacade
from research.v4.candidates.s3_public_grounding_v2_20260926.scene_neck import NECK_BINDINGS
from .task import INSTRUCTIONS

ROOT=Path(__file__).resolve().parents[4]
REL='research/v4/candidates/s3_02_t8_derived_20260926'
REPLACEMENTS={'s3_cookie_1':'cracker.xml','s3_cookie_box_1':'cracker_box.xml','s3_nameplate_1':'nameplate.xml'}
BINDINGS=tuple(replace(b,xml=f'{REL}/{REPLACEMENTS[b.name]}') if b.name in REPLACEMENTS else b for b in NECK_BINDINGS)
BASE_ENV=original._S3Env


class MiaEnv(BASE_ENV):
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        tabletop.PILOTS.setdefault('S3-02',BINDINGS)
        # Reuse the same common environment constructor and original T8
        # fixed-prop loader, bypass only the old hardcoded S3-01 carrier key.
        super(BASE_ENV,self).__init__('S3-02',seed=seed,
            camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)


class S3MiaFacade(S3RPentFacade):
    task_id='S3-02'
    def __init__(self,*args,**kwargs):
        with patch.object(original,'_S3Env',MiaEnv):super().__init__(*args,**kwargs)
    @property
    def instruction(self):return INSTRUCTIONS[self.round_number-1]
