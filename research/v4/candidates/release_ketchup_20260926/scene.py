"""Task-local neutral ketchup transport carrier; original common controller."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.s3_v42_tabletop_20260926 import scene as original
from research.v4.candidates.s3_v42_tabletop_20260926.rpent_facade import S3RPentFacade
from research.v4.candidates.s3_public_grounding_v2_20260926.scene_neck import NECK_BINDINGS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
REL='research/v4/candidates/release_ketchup_20260926'
BINDINGS=tuple(replace(b,xml=REL+'/ketchup.xml',z_offset=0.,grasp_local=(0.,0.,.116)) if b.name=='s3_ketchup_1' else b for b in NECK_BINDINGS)
BASE_ENV=original._S3Env


class ReleaseKetchupEnv(BASE_ENV):
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        tabletop.PILOTS.setdefault('KETCHUP-RELEASE',BINDINGS)
        super(BASE_ENV,self).__init__('KETCHUP-RELEASE',seed=seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)


class ReleaseKetchupFacade(S3RPentFacade):
    task_id='KETCHUP-RELEASE'
    def __init__(self,*args,**kwargs):
        with patch.object(original,'_S3Env',ReleaseKetchupEnv):super().__init__(*args,**kwargs)
