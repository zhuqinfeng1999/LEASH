"""Loader-compatible same mass/geometry; existing postcompile visibility."""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from research.v4.candidates.p2_04_ice_pick_handoff_20260926 import scene as original
from research.v4.candidates.p2_03_visible_screwdriver_20260926.scene import expose
HERE=Path(__file__).resolve().parent;ROOT=original.ROOT;INSTRUCTION=original.INSTRUCTION
BINDING=replace(original.BINDING,xml=str((HERE/'assets/ice_pick.xml').relative_to(ROOT)))
class P2IcePickFacade(original.P2IcePickFacade):
 def __init__(self,*a,**kw):
  with patch.object(original,'BINDING',BINDING):super().__init__(*a,**kw)
  expose(self)
