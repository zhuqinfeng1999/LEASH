"""Single ordinary-asset simplification: physical flat rectangular ceramic platter."""
from pathlib import Path
from unittest.mock import patch
from research.v4.accelerated import tabletop
from research.v4.candidates.e1_04_carving_fork_platter_20260926 import scene as previous
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PROTOCOL='e1-04-carving-fork-flat-rectangular-platter-t5-assisted-v42-v2'
BINDINGS=(previous.BINDINGS[0],tabletop.ObjectBinding('support_plate_1',str(HERE/'assets/platter.xml'),'R',z_offset=.006),previous.BINDINGS[2])
class FlatPlatterEnv(previous.ForkPlatterEnv):
 def __init__(self,*args,**kwargs):
  with patch.object(previous,'BINDINGS',BINDINGS):super().__init__(*args,**kwargs)
class E1CarvingForkPlatterScene(previous.E1CarvingForkPlatterScene):
 backend_class=FlatPlatterEnv
 protocol_version=PROTOCOL
