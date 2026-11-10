"""Selected P4 reset pitch35°; keep prior pitch20° evidence separate."""
from pathlib import Path
from unittest.mock import patch
import json,numpy as np
from . import scene

HERE=Path(__file__).resolve().parent
CLEAR_QPOS=np.asarray(json.loads((HERE/'selected_pose_v2.json').read_text())['initial_joint_qpos_rad'],float)
RESET_REVISION='p4-reset-public-view-pitch35-v2'

class P4ClearViewFacade(scene.P4PublicResetFacade):
    def __init__(self,*args,**kwargs):
        with patch.object(scene,'PARK_QPOS',CLEAR_QPOS):
            super().__init__(*args,**kwargs)
        self.initial_reset_revision=RESET_REVISION
