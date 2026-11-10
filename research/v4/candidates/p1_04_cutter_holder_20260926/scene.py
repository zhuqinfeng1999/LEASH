"""P1-04 common T2 asset configuration; natural person/control unchanged."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.p1_visible_blade_20260926.scene import P1VisibleBladeFacade
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BINDINGS=tuple(replace(b,xml=str((HERE/'assets'/('box_cutter.xml'if b.name=='blunt_tool_1'else'pen_holder.xml')).relative_to(ROOT)),grasp_local=(-.030,0.,.028)if b.name=='blunt_tool_1'else b.grasp_local)for b in tabletop.PILOTS['E1-01'])
INSTRUCTION='Put the box cutter in the pen holder on the right.'
class P1CutterFacade(P1VisibleBladeFacade):
    instance_id='P1-04';instruction=INSTRUCTION
    def __init__(self,*args,**kwargs):
        with patch.dict(tabletop.PILOTS,{'E1-01':BINDINGS}):super().__init__(*args,**kwargs)
