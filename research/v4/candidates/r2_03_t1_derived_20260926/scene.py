"""Config-only R2-03 binding and scoped accepted public reset; same executor."""
from pathlib import Path
from unittest.mock import patch
import json,numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.r2_02_t1_derived_20260926 import scene as existing
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
PARK_QPOS=np.asarray(json.loads((HERE/'selected_pose_v3.json').read_text())['initial_joint_qpos_rad'],float)
class T1R2TealightScene(existing.T1R2DerivedScene):
    task_id='R2-03';item_name='lit_tealight_1';target_name='paper_shopping_bag_1';instruction='Put the item on the left into the paper bag on the right.'
    def __init__(self,*args,**kwargs):
        with patch.object(existing,'HERE',HERE),patch.object(existing,'TOKEN','R2-03-T1-DERIVED-CANDIDATE'),patch.object(tabletop,'PARK_QPOS',PARK_QPOS):
            super().__init__(*args,**kwargs)
        self.initial_reset_revision='r2-03-reset35-higher1.44-v3'
