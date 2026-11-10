"""One real front-right candle source, current own text and other props unchanged."""
from dataclasses import replace
from unittest.mock import patch
from research.v4.candidates.user_rework_support_20260927.i_family import scene as own
IDS=('I1-02','I2-02','I3-02','I4-02')
ORIGINAL_LAYOUT=own.layout
def layout(bindings):
    return tuple(replace(b,slot='F',xy_offset=(-.10,.04))if b.name=='lit_candle_1'else b for b in ORIGINAL_LAYOUT(bindings))
def create(instance,seed,**options):
    if instance not in IDS:raise ValueError('Only approved own candle configuration')
    with patch.object(own,'layout',layout):return own.create(instance,seed,**options)
