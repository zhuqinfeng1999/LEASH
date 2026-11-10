"""P1-03 asset configuration on accepted natural right-arm T2 carrier."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.p1_visible_blade_20260926.scene import P1VisibleBladeFacade
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BINDINGS=tuple(replace(b,xml=str((HERE/'assets'/('scissors.xml'if b.name=='blunt_tool_1'else'pencil_case.xml')).relative_to(ROOT)),grasp_local=(-.036,0.,.028)if b.name=='blunt_tool_1'else b.grasp_local)for b in tabletop.PILOTS['E1-01'])
INSTRUCTION='Put the scissors in the pencil case on the right.'
class P1ScissorsFacade(P1VisibleBladeFacade):
    instance_id='P1-03';instruction=INSTRUCTION
    def __init__(self,*args,**kwargs):
        with patch.dict(tabletop.PILOTS,{'E1-01':BINDINGS}):super().__init__(*args,**kwargs)
