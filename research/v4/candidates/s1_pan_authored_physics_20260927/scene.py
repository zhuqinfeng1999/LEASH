"""Candidate-only original frying pan; original S1 runtime/rules/slots unchanged."""
from dataclasses import replace
from unittest.mock import patch
from research.v4.candidates.s1_v42_tabletop_20260925 import scene_v5_stable_pan300 as parent
from research.v4.candidates.s1_v42_tabletop_20260925.scene import PAN_NAME, LEFT_HAND,RIGHT_HAND, INSTRUCTIONS
PAN_XML='research/v4/candidates/s1_pan_authored_physics_20260927/frying_pan.xml'
PAN_BINDINGS=tuple(replace(b,xml=PAN_XML) for b in parent.PAN_BINDINGS)
def create(seed,*,camera_observations=False,render_gpu_device_id=-1):
    with patch.object(parent,'PAN_BINDINGS',PAN_BINDINGS):
        return parent.create(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
class AuthoredPanScene(parent.S1StablePan300Facade):
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        with patch.object(parent,'PAN_BINDINGS',PAN_BINDINGS):
            super().__init__(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
