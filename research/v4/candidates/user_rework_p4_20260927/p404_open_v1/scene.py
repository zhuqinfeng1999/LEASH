"""Ordinary open gravity support, forward dangerous surface, common physics."""
from pathlib import Path
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.p4_v42_t4_tabletop_20260926 import facade as core
from research.v4.candidates.p4_04_cold_saddle_20260926 import scene as parent
from research.v4.candidates.user_rework_p4_20260927.layout_v1.scene import ForwardSurfaceEnv,moved_bindings
HERE=Path(__file__).resolve().parent
class Scene(parent.SaddleScene):
 layout_version='p404-open-gravity-saddle-forward-v1'
 def __init__(self,*args,**kwargs):
  with patch.object(parent,'HERE',HERE),patch.object(core,'_P4Env',ForwardSurfaceEnv),patch.dict(tabletop.PILOTS,{parent.TASK:moved_bindings(parent.TASK)}):super().__init__(*args,**kwargs)
