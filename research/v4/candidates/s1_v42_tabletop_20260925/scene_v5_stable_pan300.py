"""S1 development scene with realistic 300 g skillet inertia.

The original LIBERO visual/collision mesh is development-only and remains
license-uncleared. This version changes only its inertial parameters relative
to the rear-pan v3 layout; the seated person, slots, cameras, event, and
evaluator stay unchanged. Publication requires a cleared replacement asset.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist
from . import scene as base
from .scene_v3_clear_handle import S1ClearHandleEnv


PAN_XML = "research/v4/candidates/s1_v42_tabletop_20260925/development_pan_300g.xml"
PAN_BINDINGS = tuple(replace(binding, xml=PAN_XML) for binding in base.PAN_BINDINGS)


def create(seed: int, *, camera_observations=False, render_gpu_device_id=-1):
    key = "E1-01"
    prior_bindings = base.PAN_BINDINGS
    prior_pilot = tabletop.PILOTS.get(key)
    base.PAN_BINDINGS = PAN_BINDINGS
    tabletop.PILOTS[key] = PAN_BINDINGS
    try:
        backend = S1ClearHandleEnv(
            int(seed), camera_observations=camera_observations,
            render_gpu_device_id=render_gpu_device_id)
        backend.reset()
        assist = GraspAssist(backend.sim.model._model, backend.sim.data._data,
                             **backend.grasp_assist_kwargs)
        return backend, assist
    finally:
        base.PAN_BINDINGS = prior_bindings
        if prior_pilot is None:
            del tabletop.PILOTS[key]
        else:
            tabletop.PILOTS[key] = prior_pilot


class S1StablePan300Facade(base.S1Facade):
    def __init__(self, seed=0, *, camera_observations=False,
                 render_gpu_device_id=-1):
        self.backend, self.grasp_assist = create(
            int(seed), camera_observations=camera_observations,
            render_gpu_device_id=render_gpu_device_id)
        self.env = SimpleNamespace(sim=self.backend.sim, env=self.backend)
        self.raw = self.backend._get_observations(force_update=True)
        self.sequence = 0
        self.assist_events = []
        self.round_index = 0
