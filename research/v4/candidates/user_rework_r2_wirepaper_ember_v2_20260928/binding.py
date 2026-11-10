"""Only the visual surface shell declaration differs from the stopped v1."""
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
from research.v4.candidates.user_rework_r2_wirepaper_ember_20260928 import binding as v1
parent=v1.parent;HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PAPER=v1.PAPER;CHARCOAL=HERE/'assets/charcoal_ash_ember_shell.xml'
@contextmanager
def scope():
 with patch.multiple(parent,PAPER=PAPER,CHARCOAL=CHARCOAL):yield
class R201WirePaperScene(parent.R201SupportedPaperScene):
 def __init__(self,*a,**kw):
  with scope():super().__init__(*a,**kw)
class R204VisibleEmberScene(parent.R204FacetedCharcoalScene):
 def __init__(self,*a,**kw):
  with scope():super().__init__(*a,**kw)
def selected(instance):
 if instance=='R2-01':return R201WirePaperScene
 if instance=='R2-04':return R204VisibleEmberScene
 raise ValueError('Two exact own instances only')
