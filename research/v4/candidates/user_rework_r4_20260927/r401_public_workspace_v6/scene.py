"""Explicit new own public camera/calibration proposal; physics unchanged."""
import json
from types import FunctionType
from unittest.mock import patch
from pathlib import Path
from ..r401_normal_crib_native_v5 import native as old
HERE=Path(__file__).resolve().parent;ROOT=old.ROOT
FIT=json.loads((HERE/'camera_fit.json').read_text())
def cameras(world):
 old.offline_camera(world)
 cameras=world.findall(".//camera[@name='agentview']")
 if len(cameras)!=1:raise RuntimeError('Own exact public agentview identity changed')
 c=cameras[0]
 c.set('pos',' '.join(map(str,FIT['eye'])));c.set('xyaxes',' '.join(map(str,FIT['xyaxes'])));c.set('fovy',str(FIT['fovy_deg']))
 for attr in('quat','euler','axisangle','zaxis'):
  if attr in c.attrib:del c.attrib[attr]
INIT_NS={**old.NATIVE_INIT.__globals__,'_camera':cameras}
PUBLIC_INIT=FunctionType(old.NATIVE_INIT.__code__,INIT_NS,argdefs=old.NATIVE_INIT.__defaults__)
class Scene(old.scene.Scene):
 def __init__(self,*a,**kw):
  with patch.object(old.scene,'SELECTED_INIT',PUBLIC_INIT):super().__init__(*a,**kw)
  self.observation_protocol_proposal=FIT['version']
