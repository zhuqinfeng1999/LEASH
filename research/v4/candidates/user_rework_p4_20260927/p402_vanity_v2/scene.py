from pathlib import Path
from unittest.mock import patch
from research.v4.candidates.user_rework_p4_20260927.p402_vanity_v1 import scene as original
HERE=Path(__file__).resolve().parent
class Scene(original.Scene):
 layout_version='p402-normal-compact-vanity-backtilted-mirror-v2'
 def __init__(self,*args,**kwargs):
  with patch.object(original,'HERE',HERE):super().__init__(*args,**kwargs)
