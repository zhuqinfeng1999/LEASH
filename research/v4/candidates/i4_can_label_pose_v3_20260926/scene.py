"""Can-only higher public inspection pose; frozen v2 display/physics reused."""
import json
from pathlib import Path
from unittest.mock import patch
from research.v4.candidates.i4_named_label_pose_v2_20260926 import scene as prior
HERE=Path(__file__).resolve().parent
LEFT_HAND=prior.LEFT_HAND;RIGHT_HAND=prior.RIGHT_HAND
IDS=('I4-03',)
def create(instance,seed,**options):
    if instance not in IDS:raise ValueError(instance)
    qpos=json.loads((HERE/'park_poses.json').read_text())[instance]
    with patch.object(prior.original,'HERE',prior.HERE),patch.object(prior.original,'INSPECTION_PARK_QPOS',tuple(qpos)):
        return prior.BASE_CREATE(instance,seed,**options)
