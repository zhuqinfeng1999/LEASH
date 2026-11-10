"""Five own registered S3 configurations, only shared real plate bottom changed."""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from research.v4.candidates.s3_release_region_20260926 import scene as common
from research.v4.candidates.s3_03_05_t8_configs_20260926 import scene as derived
H=Path(__file__).resolve().parents[1]
CONFIGS={**common.CONFIGS,**derived.CONFIGS}
CONFIGS={id:(tuple(replace(b,xml=str(H/'plate_planar.xml'))if b.name=='s3_plate_1'else b for b in bindings),instructions)for id,(bindings,instructions)in CONFIGS.items()}
class PlanarFamilyFacade(common.S3ReleaseFacade):
 def __init__(self,*args,instance='S3-02',**kw):
  if instance not in CONFIGS:raise ValueError('Unknown own registered S3 variant')
  with patch.object(common,'CONFIGS',CONFIGS):super().__init__(*args,instance=instance,**kw)
