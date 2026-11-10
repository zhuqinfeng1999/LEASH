"""One declared before-reset L/source shift; physics/whole-person parent immutable."""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.accelerated.e1_facade import _E1FixtureEnv
from research.v4.candidates.e1_06_elbow_pole_v4_20260926.scene import E1GlassDishpanScene as Previous
from research.v4.candidates.e1_06_flat_contact_raised_arm_v3_20260926.scene import BINDINGS as ParentBindings
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PROTOCOL='e1-06-initialized-clear-source-natural-elbow-t5-assisted-v42-v5'
SOURCE_BINDING_XY_OFFSET_M=(0.,-.048)
BINDINGS=(replace(ParentBindings[0],xy_offset=SOURCE_BINDING_XY_OFFSET_M),ParentBindings[1])
class ClearSourceEnv(_E1FixtureEnv):
 def __init__(self,*args,**kwargs):
  with patch.dict(tabletop.PILOTS,{'E1-01':BINDINGS}):super().__init__(*args,**kwargs)
class E1GlassDishpanScene(Previous):
 backend_class=ClearSourceEnv
 protocol_version=PROTOCOL
