"""One physical I layout/support revision; old parents remain unchanged."""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import xml.etree.ElementTree as ET
from research.v4.accelerated import tabletop
from research.v4.candidates.i1_legible_paper_native_v2_20260926 import scene as old
from research.v4.candidates.i_common_release_ketchup_20260926 import scene as common
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
PROTOCOL='I-common-real-bin-device-separation-planar-bottom-support-v1'
MICROWAVE_CENTER=(-.22,.19,.9);BIN_CENTER=(.25,.18,.9)
BINDINGS=tuple(replace(b,slot='F',xy_offset=(-.07,.18))if b.name=='paper_bin_1'else
 replace(b,xml=str(H/'lit_candle_planar.xml'))if b.name=='lit_candle_1'else b for b in common.BINDINGS)
class SupportEnv(old.LegiblePaperEnv):
 @staticmethod
 def person_builder(arena):
  old.LegiblePaperEnv.person_builder(arena)
  fixture=arena.root.find(".//body[@name='i1_microwave_fixture']")
  if fixture is None:raise RuntimeError('Exact fixed microwave missing')
  fixture.set('pos',' '.join(map(str,MICROWAVE_CENTER)))
def create(seed,*,camera_observations=False,render_gpu_device_id=-1):
 with patch.object(common,'I1StraightRightEnv',SupportEnv),patch.object(common,'BINDINGS',BINDINGS):
  return common.create(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
