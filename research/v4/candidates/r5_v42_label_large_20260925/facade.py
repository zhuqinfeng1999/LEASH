"""R5 public scene facade; online packets inherit the common two-view protocol."""
from __future__ import annotations

from types import SimpleNamespace

from research.v4.candidates.i1_v42_t1t7_20260925.facade import I1Facade
from research.v4.tasks.r5.template import load_instance
from .scene import INSTRUCTION, create


class R5TabletopFacade(I1Facade):
    instruction = INSTRUCTION

    def __init__(self, seed=0, *, camera_observations=False,
                 render_gpu_device_id=-1):
        self.backend, self.grasp_assist = create(
            int(seed), camera_observations=camera_observations,
            render_gpu_device_id=render_gpu_device_id,
        )
        self.env = SimpleNamespace(sim=self.backend.sim, env=self.backend)
        self.instance = load_instance("R5-01")
        self.raw = self.backend._get_observations(force_update=True)
        self.sequence = 0
        self.assist_events = []
