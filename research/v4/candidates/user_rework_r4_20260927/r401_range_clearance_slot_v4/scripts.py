"""Same own650-cap bag expert, only newly qualified whole-crib slot."""
from unittest.mock import patch
from ..r401_normal_crib_first_risk_v3 import scripts as old
from .scene import Scene,HERE,ROOT
def run_case(seed,case):
 with patch.object(old,'Scene',Scene):r=old.run_case(seed,case)
 r['layout_version']='continuous-domain-normal-crib-Y-slot-v4'
 return r
