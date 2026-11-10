"""P4 reset-only pose; identical camera mounts/assets/controller/score."""
from pathlib import Path
from unittest.mock import patch
import json,numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.p4_v42_t4_tabletop_20260926.facade import P4TabletopFacade

HERE=Path(__file__).resolve().parent
PARK_QPOS=np.asarray(json.loads((HERE/'selected_pose.json').read_text())['initial_joint_qpos_rad'],float)
RESET_REVISION='p4-reset-public-view-pitch20-v1'

class P4PublicResetFacade(P4TabletopFacade):
    def __init__(self,*args,**kwargs):
        with patch.object(tabletop,'PARK_QPOS',PARK_QPOS):
            super().__init__(*args,**kwargs)
        self.initial_reset_revision=RESET_REVISION
