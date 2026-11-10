"""Private grasp annotation on the real bottle neck; no asset/pose change.

The x=0,z=98 mm point is inside the authored neck collision box (z90–106 mm)
and visible upright bottle mesh. The failed center-shaft52 mm candidate stays
separate; this higher point avoids the palm contacting the cap before closing.
"""
from __future__ import annotations
from dataclasses import replace
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.s3_v42_tabletop_20260926 import scene as original
from research.v4.candidates.s3_v42_tabletop_20260926.rpent_facade import S3RPentFacade

NECK_BINDINGS=tuple(replace(b,grasp_local=(0.,0.,.098)) if b.name=='s3_ketchup_1'
                    else b for b in original.BINDINGS)
GRASP_REVISION='s3-upright-bottle-neck-zone-v3'

class NeckZoneEnv(original._S3Env):
    def __init__(self,*args,**kwargs):
        with patch.dict(tabletop.PILOTS,{'S3-01':NECK_BINDINGS}):
            super().__init__(*args,**kwargs)

class S3NeckGraspFacade(S3RPentFacade):
    def __init__(self,*args,**kwargs):
        with patch.object(original,'_S3Env',NeckZoneEnv):
            super().__init__(*args,**kwargs)
        self.grasp_annotation_revision=GRASP_REVISION
