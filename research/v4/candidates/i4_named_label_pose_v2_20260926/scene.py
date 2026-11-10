"""Target-specific normal product tag + common public initial wrist pose."""
import json
from pathlib import Path
from unittest.mock import patch
from research.v4.candidates.i_t7_candle_can_derived_20260926 import scene as original
BASE_CREATE=original.create
HERE=Path(__file__).resolve().parent
LEFT_HAND=original.LEFT_HAND;RIGHT_HAND=original.RIGHT_HAND
IDS=('I4-02','I4-03')
def create(instance,seed,**options):
    if instance not in IDS:raise ValueError(instance)
    poses=HERE/'park_poses.json'
    qpos=json.loads(poses.read_text())[instance]if poses.exists()else original.INSPECTION_PARK_QPOS
    with patch.object(original,'HERE',HERE),patch.object(original,'INSPECTION_PARK_QPOS',tuple(qpos)):
        return BASE_CREATE(instance,seed,**options)
