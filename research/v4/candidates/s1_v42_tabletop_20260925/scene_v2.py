"""Versioned S1 native-composition fix: visible board, forward stove slot.

No change to pan/board collision geometry, assist tolerance, evaluator rules,
public cameras, or person pose. The stove and pan share a +0.10 m X offset.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.candidates.s1_board_visibility_20260924.visual_fixture import add_board_visual
from . import scene as base


STOVE_FORWARD_OFFSET_M = .10
PAN_BINDINGS = tuple(replace(binding,
    xy_offset=(binding.xy_offset[0] + STOVE_FORWARD_OFFSET_M,
               binding.xy_offset[1])) for binding in base.PAN_BINDINGS)
LEFT_HAND = base.LEFT_HAND
RIGHT_HAND = base.RIGHT_HAND
PAN_NAME = base.PAN_NAME
INSTRUCTIONS = base.INSTRUCTIONS


class S1VisibleEnv(base._S1Env):
    def person_builder(self, arena):
        super().person_builder(arena)
        add_board_visual(arena.root)


def create(seed: int, *, camera_observations=False, render_gpu_device_id=-1):
    key = "E1-01"
    prior_bindings = base.PAN_BINDINGS
    prior_pilot = tabletop.PILOTS.get(key)
    base.PAN_BINDINGS = PAN_BINDINGS
    tabletop.PILOTS[key] = PAN_BINDINGS
    try:
        backend = S1VisibleEnv(int(seed), camera_observations=camera_observations,
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


class S1VisibleFacade(base.S1Facade):
    def __init__(self, seed=0, *, camera_observations=False, render_gpu_device_id=-1):
        self.backend, self.grasp_assist = create(
            int(seed), camera_observations=camera_observations,
            render_gpu_device_id=render_gpu_device_id)
        self.env = SimpleNamespace(sim=self.backend.sim, env=self.backend)
        self.raw = self.backend._get_observations(force_update=True)
        self.sequence = 0
        self.assist_events = []
        self.round_index = 0
