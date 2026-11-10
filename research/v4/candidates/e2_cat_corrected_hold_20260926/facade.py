"""Corrected opaque cat exterior, unchanged source/box layout and trajectory."""
from pathlib import Path
from research.v4.candidates.e2_cat_native_orientation_v4_20260926.facade import OrientationScene

HERE=Path(__file__).resolve().parent


class E2CorrectedHoldScene(OrientationScene):
    def __init__(self,seed=0,*,camera_observations=False,render_gpu_device_id=-1):
        super().__init__(seed,orientation='A',camera_observations=camera_observations,
                         render_gpu_device_id=render_gpu_device_id)
        self.protocol_version='e2-corrected-body-winding-original-layout-v1'
