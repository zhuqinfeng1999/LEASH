"""Shared public R5 interface with a separate product identification card."""
from dataclasses import replace
from types import SimpleNamespace

from research.v4.candidates.r5_v42_label_large_20260925.facade import R5TabletopFacade
from research.v4.tasks.r5.template import load_instance
from .scene import create


class R5CardFacade(R5TabletopFacade):
    def __init__(self, seed=0, *, camera_observations=False, render_gpu_device_id=-1):
        self.backend, self.grasp_assist = create(int(seed),
            camera_observations=camera_observations,
            render_gpu_device_id=render_gpu_device_id)
        self.env = SimpleNamespace(sim=self.backend.sim, env=self.backend)
        original = load_instance("R5-01")
        item = dict(original.item)
        item["footprint_half_xy_m"] = [0.115, 0.030]
        item["bottom_offset_z_m"] = -0.015
        self.instance = replace(original, item=item)
        self.raw = self.backend._get_observations(force_update=True)
        self.sequence = 0
        self.assist_events = []
