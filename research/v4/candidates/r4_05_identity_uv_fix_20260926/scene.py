"""Compiler-time card UV-only overlay; physical card/texture/cameras unchanged."""
import copy
from pathlib import Path
from unittest.mock import patch
from research.v4.candidates.r4_05_inclined_identity_v2_20260926 import scene as original
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
ORIGINAL_CONFIGURATION=original.configuration
def configuration():
    c=copy.deepcopy(ORIGINAL_CONFIGURATION())
    c['identity_card']['xml']=str(HERE/'identity_card_uv.xml')
    return c
class ReadableIdentityScene(original.T1R4MagneticBoxScene):
    def __init__(self,*args,**kwargs):
        with patch.object(original,'configuration',configuration):super().__init__(*args,**kwargs)
