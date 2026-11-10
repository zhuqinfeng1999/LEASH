"""Original takeaway-tray configuration on the qualified R3 shared device carrier."""
from pathlib import Path
from unittest.mock import patch
from research.v4.candidates.r3_03_t1_derived_20260926 import scene as carrier
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
class AluminumTrayScene(carrier.T1R3ThermosScene):
    task_id='R3-06'
    # Historical private compatibility alias only; no names/points exposed online.
    item_name='steel_thermos_1'
    def __init__(self,*args,**kwargs):
        with patch.object(carrier,'HERE',HERE):super().__init__(*args,**kwargs)
