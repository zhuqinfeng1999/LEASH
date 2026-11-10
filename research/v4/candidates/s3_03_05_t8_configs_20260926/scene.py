"""Configuration-driven current S3 scene; no copied executor or truth interface."""
from dataclasses import replace
from unittest.mock import patch
from research.v4.candidates.s3_release_region_20260926 import scene as carrier
from .task import CONFIG,instructions,HERE,ROOT
REL=str(HERE.relative_to(ROOT))
BASE=carrier.CONFIGS['S3-02'][0]
FILES={'s3_cookie_1':'food.xml','s3_cookie_box_1':'food_box.xml','s3_nameplate_1':'nameplate.xml'}
CONFIGS={instance:(tuple(replace(b,xml=f'{REL}/{instance}/{FILES[b.name]}') if b.name in FILES else b for b in BASE),instructions(instance)) for instance in CONFIG}

class ConfiguredS3Facade(carrier.S3ReleaseFacade):
    def __init__(self,*args,instance='S3-03',**kwargs):
        if instance not in CONFIGS:raise ValueError(instance)
        with patch.object(carrier,'CONFIGS',{**carrier.CONFIGS,**CONFIGS}):
            super().__init__(*args,instance=instance,**kwargs)
