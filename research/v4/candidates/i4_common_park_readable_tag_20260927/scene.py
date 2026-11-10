"""One scoped own I4 paper-tag appearance, original commonPARK/physics."""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
from research.v4.candidates.user_rework_support_20260927.i_candle_front_v2 import scene as candle
from research.v4.candidates.user_rework_support_20260927.i4_common_park_v3 import scene as accepted
from research.v4.candidates.user_rework_support_20260927.i_family import scene as family
H=Path(__file__).resolve().parent;IDS=accepted.IDS;NAMES={'I4-01':'blunt_tool_1','I4-02':'lit_candle_1','I4-03':'soup_can_1'}
BASE_LAYOUT=family.layout
def create(instance,seed,**options):
 if instance not in IDS:raise ValueError('Three exact own I4 instances only')
 xml=H/instance/'tagged_object.xml'
 if not xml.is_file():raise ValueError('Exact own authored tag missing')
 def layout(bindings):
  result=BASE_LAYOUT(bindings);matches=[b for b in result if b.name==NAMES[instance]]
  if len(matches)!=1:raise ValueError('Own tag carrier binding must be unique')
  return tuple(replace(b,xml=str(xml))if b.name==NAMES[instance]else b for b in result)
 with patch.object(family,'layout',layout),patch.object(candle,'ORIGINAL_LAYOUT',layout):return accepted.create(instance,seed,**options)
