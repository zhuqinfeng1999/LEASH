"""Only the real authored neck-grip point changes by20mm; all physical assets unchanged."""
from dataclasses import replace
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.p1_06_wine_bucket_t2_20260926.scene import P1WineBucketFacade as Previous,BINDINGS as OLD_BINDINGS,INSTRUCTION
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BINDINGS=tuple(replace(b,grasp_local=(0,0,.260))if b.name=='blunt_tool_1'else b for b in OLD_BINDINGS)
class P1WineBucketFacade(Previous):
 def __init__(self,*args,**kwargs):
  # Patch the candidate's private asset annotation, not shared physical metadata.
  from research.v4.candidates.p1_06_wine_bucket_t2_20260926 import scene as previous
  with patch.object(previous,'BINDINGS',BINDINGS):super().__init__(*args,**kwargs)
