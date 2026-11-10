"""Final higher S2 reset candidate; original scene, camera mounts, objects and controls."""
from pathlib import Path
from unittest.mock import patch
import json,numpy as np
from . import scene
HERE=Path(__file__).resolve().parent
HIGHER_POSE_PATH=HERE/'higher_pose_projection/selected_pose.json'
HIGHER_PARK_QPOS=np.asarray(json.loads(HIGHER_POSE_PATH.read_text())['initial_joint_qpos_rad'],float)
class HigherPoseScene(scene.PoseScene):
    def __init__(self,*args,**kwargs):
        with patch.object(scene,'PARK_QPOS',HIGHER_PARK_QPOS):
            super().__init__(*args,**kwargs)
def higher_cpu_factory(**kwargs):
    with patch.object(scene,'PARK_QPOS',HIGHER_PARK_QPOS):
        return scene.parked_env_factory(**kwargs)
