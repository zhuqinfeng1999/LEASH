"""One root-approved whole vanity-body slot translation; own source unchanged."""
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.user_rework_p4_20260927.p402_mat_front_v5 import scene as original
SHIFT=.11806697694740742
class RearVanityEnv(original.FrontMatEnv):
    def _load_model(self):
        super()._load_model()
        board=self.model.worldbody.find("body[@name='tabletop_ironing_board']")
        R=Rotation.from_quat(np.roll(np.fromstring(board.get('quat'),sep=' '),-1)).as_matrix()
        p=np.fromstring(board.get('pos'),sep=' ')-R[:,1]*SHIFT
        board.set('pos',' '.join(map(str,p)))
class Scene(original.Scene):
    layout_version='p402-original-source-whole-vanity-backshift118_066977mm-v11'
    def __init__(self,*args,**kwargs):
        with patch.object(original,'FrontMatEnv',RearVanityEnv):super().__init__(*args,**kwargs)
