"""One task-local private authored handle point; unchanged original carrier."""
import copy
from unittest.mock import patch
from research.v4.candidates.r1_05_portrait_cradle_v2_20260926 import scene as parent
from research.v4.accelerated.grasp_assist import GraspAssist
BASE_CONFIGURATION=parent.carrier.configuration
POINT=(-.061,.057,-.004)
def configuration():
    with patch.object(parent.carrier,'HERE',parent.HERE):
        cfg=copy.deepcopy(BASE_CONFIGURATION())
    cfg['source']['grasp_local']=list(POINT)
    return cfg
class TailGraspScene(parent.T1R1GamepadScene):
    def __init__(self,*args,**kwargs):
        with patch.object(parent,'configuration',configuration),patch.object(parent.carrier,'configuration',configuration):
            super().__init__(*args,**kwargs)

