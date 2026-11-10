"""Proposed future S1-01 binding; nothing applies to shared or old runs."""
from unittest.mock import patch
from research.v4.candidates.s1_pan_authored_complete_visual_20260927 import scene as approved
from research.v4.candidates.s1_pan_authored_physics_20260927 import scene as physics
from research.v4.candidates.s1_v42_tabletop_20260925.scene import PAN_NAME,LEFT_HAND,RIGHT_HAND,INSTRUCTIONS
PAN_XML=approved.PAN_XML
PAN_BINDINGS=approved.PAN_BINDINGS
S1AuthoredReleaseFacade=approved.CompleteVisualScene
ASSET_VERSION='S1-01_original_CC0_pan_173physics_232completevisual_root_native_reviewed'
GRASP_MODE='assisted'
def create(seed,*,camera_observations=False,render_gpu_device_id=-1):
    with patch.object(physics,'PAN_BINDINGS',PAN_BINDINGS):
        return physics.create(seed,camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id)
