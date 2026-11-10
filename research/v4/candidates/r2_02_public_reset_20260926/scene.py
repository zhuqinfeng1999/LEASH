"""R2-02 reset-only subclass: original assets/cameras/layout/assist/scorer."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.v4.accelerated import tabletop
from research.v4.candidates.r2_02_t1_derived_20260926.scene import T1R2DerivedScene
HERE=Path(__file__).resolve().parent
PARK_QPOS=np.asarray(json.loads((HERE/'selected_pose.json').read_text())['initial_joint_qpos_rad'],float)
class R2IncensePublicResetScene(T1R2DerivedScene):
    def __init__(self,*args,**kwargs):
        with patch.object(tabletop,'PARK_QPOS',PARK_QPOS):super().__init__(*args,**kwargs)
        self.initial_reset_revision='r2-02-reset-only-v1'
