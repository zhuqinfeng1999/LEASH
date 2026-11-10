"""Scoped opt-in paths; parent scenes/control/collision semantics retained."""
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
from research.v4.candidates.user_rework_r2_supported_charcoal_20260927 import binding as parent
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PAPER=HERE/'assets/paper_bin_wire_supported.xml';CHARCOAL=HERE/'assets/charcoal_ash_ember.xml'
@contextmanager
def scope():
 with patch.multiple(parent,PAPER=PAPER,CHARCOAL=CHARCOAL):yield
class R201WirePaperScene(parent.R201SupportedPaperScene):
 def __init__(self,*args,**kwargs):
  with scope():super().__init__(*args,**kwargs)
class R204VisibleEmberScene(parent.R204FacetedCharcoalScene):
 def __init__(self,*args,**kwargs):
  with scope():super().__init__(*args,**kwargs)
def selected(instance):
 if instance=='R2-01':return R201WirePaperScene
 if instance=='R2-04':return R204VisibleEmberScene
 raise ValueError('Two explicit own instances only')
