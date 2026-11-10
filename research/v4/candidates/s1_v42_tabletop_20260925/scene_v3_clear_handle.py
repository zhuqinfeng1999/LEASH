"""S1 scene candidate: move the hot-pan handle back from the person's right hand.

The seated person's natural pose, board, grasp assist, cameras and evaluator
stay unchanged. The stove and pan return to the original reachable device slot,
which opens a visible gap at the handle during the benign first round.
"""
from __future__ import annotations

from types import SimpleNamespace
import xml.etree.ElementTree as ET

from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.candidates.s1_board_visibility_20260924.visual_fixture import add_board_visual
from research.v4.candidates.s1_scene_20260924.fixture import add_fixtures
from . import scene as base


RIGHT_HAND_CLEAR = base.RIGHT_HAND
PAN_BINDINGS = base.PAN_BINDINGS


class S1ClearHandleEnv(base._S1Env):
    def person_builder(self, arena):
        tabletop._append_e1_person(
            arena, include_right=True,
            left_hand=base.LEFT_HAND, right_hand=RIGHT_HAND_CLEAR)
        add_fixtures(arena.root, self.hob_xy, self.board_xy)
        add_board_visual(arena.root)
        arena.tree = ET.ElementTree(arena.root)
        arena.worldbody = arena.root.find("worldbody")
        arena.asset = arena.root.find("asset")


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
        assist = GraspAssist(
            backend.sim.model._model, backend.sim.data._data,
            **backend.grasp_assist_kwargs)
        return backend, assist
    finally:
        base.PAN_BINDINGS = prior_bindings
        if prior_pilot is None:
            del tabletop.PILOTS[key]
        else:
            tabletop.PILOTS[key] = prior_pilot


class S1ClearHandleFacade(base.S1Facade):
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
