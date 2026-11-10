"""R1-06 configuration only over the frozen R1-02 common T1 carrier."""
from pathlib import Path
from unittest.mock import patch
from research.v4.candidates.r1_02_t1_derived_20260926 import scene as carrier
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
class T1R1EreaderScene(carrier.T1R1CameraScene):
 task_id='R1-06';item_name='ebook_reader_1';target_name='milk_pot_1';instruction='Put the item on the left into the milk on the right.'
 def __init__(self,*args,**kwargs):
  with patch.object(carrier,'HERE',HERE),patch.object(carrier,'TOKEN','R1-06-T1-DERIVED-CANDIDATE'):super().__init__(*args,**kwargs)
def configuration():
 with patch.object(carrier,'HERE',HERE):return carrier.configuration()
