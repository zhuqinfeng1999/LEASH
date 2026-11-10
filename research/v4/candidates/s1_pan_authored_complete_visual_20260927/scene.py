"""Only display the real steel neck: no change to authored pan physics."""
from dataclasses import replace
from unittest.mock import patch
from research.v4.candidates.s1_pan_authored_physics_20260927 import scene as physical
PAN_XML='research/v4/candidates/s1_pan_authored_complete_visual_20260927/complete_visual.xml'
PAN_BINDINGS=tuple(replace(b,xml=PAN_XML) for b in physical.PAN_BINDINGS)
class CompleteVisualScene(physical.AuthoredPanScene):
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        with patch.object(physical,'PAN_BINDINGS',PAN_BINDINGS):
            super().__init__(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
