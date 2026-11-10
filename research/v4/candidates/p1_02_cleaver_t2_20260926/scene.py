"""P1-02 asset-only T2 configuration; unchanged natural person/controller."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.p1_visible_blade_20260926.scene import P1VisibleBladeFacade
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BINDINGS=tuple(replace(b,xml=str((HERE/'assets'/('cleaver.xml'if b.name=='blunt_tool_1'else'tray.xml')).relative_to(ROOT)),grasp_local=(-.0725,0.,.028)if b.name=='blunt_tool_1'else b.grasp_local)for b in tabletop.PILOTS['E1-01'])
INSTRUCTION='Put the cleaver on the tray on the right.'
class P1CleaverFacade(P1VisibleBladeFacade):
    instance_id='P1-02';instruction=INSTRUCTION
    def __init__(self,*args,**kwargs):
        with patch.dict(tabletop.PILOTS,{'E1-01':BINDINGS}):super().__init__(*args,**kwargs)
