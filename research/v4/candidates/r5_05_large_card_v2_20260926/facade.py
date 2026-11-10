from unittest.mock import patch
from research.v4.candidates.r5_05_laundry_pod_t1_20260926 import facade as old
from .scene import create
class LargeCardFacade(old.R5LaundryFacade):
 def __init__(self,*args,**kwargs):
  with patch.object(old,'create',create):super().__init__(*args,**kwargs)
