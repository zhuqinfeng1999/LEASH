"""Texture-only future I1 overlay on frozen201 common original bottle scene."""
from pathlib import Path
from unittest.mock import patch
from research.v4.candidates.i_common_release_ketchup_20260926 import scene as base
from research.v4.candidates.i1_straight_right_arm_20260925.scene import I1StraightRightEnv
HERE=Path(__file__).resolve().parent
PROTOCOL=base.PROTOCOL+'-same-paper-native-uv-corrected-bold-raster-v2'
class LegiblePaperEnv(I1StraightRightEnv):
    @staticmethod
    def person_builder(arena):
        I1StraightRightEnv.person_builder(arena)
        arena.root.find(".//texture[@name='v4_i1_print']").set('file',str(HERE/'note_condensed_bold.png'))
def create(seed,**kwargs):
    with patch.object(base,'I1StraightRightEnv',LegiblePaperEnv):return base.create(seed,**kwargs)
