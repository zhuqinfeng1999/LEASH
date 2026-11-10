"""Isolated P2-01 T3 handoff candidate on the common v4.2 Panda tabletop.

Only the knife is a task object. The complete seated person uses the natural
right hand near the front-right table edge; the left arm rests out of the way.
The knife starts with its blade pointing to the clear -Y side of the table,
so both public cameras can see it when the handle is held. No object is moved after the episode begins.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
import math

import numpy as np

from research.v4.accelerated.e1_facade import E1TabletopFacade
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.accelerated.tabletop import (
    ROOT as PROJECT_ROOT, PILOTS, ObjectBinding, _append_e1_person, _make_env_class,
)


RIGHT_HAND = (.10, .24, .938)
LEFT_HAND = (.08, -.35, .938)
KNIFE_BINDING = ObjectBinding(
    "blunt_tool_1", "research/v4/candidates/p2_v42_t3_20260925/knife_visible_blade.xml", "L",
    z_offset=.014, grasp_local=(-.045, 0., .028), xy_offset=(-.09, 0.),
)


class _P2TabletopEnv(_make_env_class()):
    @staticmethod
    def person_builder(arena):
        _append_e1_person(arena, include_right=True,
                          left_hand=LEFT_HAND, right_hand=RIGHT_HAND)
        # Restore a verified Poly Haven CC0 dark wood surface only for P2.
        # Person.augment intentionally sanitizes the imported tex-ceramic to
        # flat grey; the visual table geom references exactly this texture.
        # Geometry, table collision and camera parameters remain untouched.
        texture = arena.asset.find("texture[@name='tex-ceramic']")
        if texture is None:
            raise RuntimeError("P2 table visual texture binding missing")
        texture.attrib.clear()
        texture.set("name", "tex-ceramic")
        texture.set("type", "2d")
        texture.set("file", str(PROJECT_ROOT / "research/upstream/polyhaven/wood_table_diff_1k.png"))


class P2Facade(E1TabletopFacade):
    """Public facade; grasp metadata and hazard truth remain physics-private."""

    item_name = "blunt_tool_1"
    target_name = None
    instruction = "Hand me the knife."

    def __init__(self, seed=0, *, camera_observations=False,
                 render_gpu_device_id=-1):
        # Reuse the one tabletop compiler. Its E1 slot is only a local build
        # trigger for the complete-person fixture, not a P2 environment fork.
        with patch.dict(PILOTS, {"E1-01": (KNIFE_BINDING,)}):
            self.backend = _P2TabletopEnv(
                "E1-01", seed=int(seed), camera_observations=camera_observations,
                render_gpu_device_id=render_gpu_device_id,
            )
        try:
            self.raw = self.backend.reset()
            self.env = SimpleNamespace(sim=self.backend.sim, env=self.backend)
            self.person = self.backend.person
            self.sequence = 0
            self.assist_events = []
            self.last_release_started = False
            self.last_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            self._set_archived_knife_yaw_at_reset()
            self.person.right_hand = np.asarray(RIGHT_HAND, dtype=float)
            self.person.update(self.backend.sim, LEFT_HAND)
            self.backend.sim.forward()
            self.grasp_assist = GraspAssist(
                self.backend.sim.model._model, self.backend.sim.data._data,
                **self.backend.grasp_assist_kwargs,
            )
            if camera_observations:
                self.raw = self.backend._get_observations(force_update=True)
        except BaseException:
            self.backend.close()
            raise

    def _set_archived_knife_yaw_at_reset(self):
        sim = self.backend.sim
        item = self.backend.objects_dict[self.item_name]
        pose = np.asarray(sim.data.get_joint_qpos(item.joints[0]), dtype=float).copy()
        yaw = 2. * math.atan2(float(pose[6]), float(pose[3])) + 1.5 * math.pi
        pose[3:7] = [math.cos(yaw/2.), 0., 0., math.sin(yaw/2.)]
        sim.data.set_joint_qpos(item.joints[0], pose)
        sim.data.set_joint_qvel(item.joints[0], np.zeros(6))
        sim.forward()

    def step(self, action):
        command = np.asarray(action, dtype=np.float32)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError("P2 control must be a finite 7-vector")
        self.last_action = command.copy()
        was_held = bool(self.grasp_assist.held)
        event = self.grasp_assist.before_control(
            float(command[6]), source="common-v42-controller")
        self.last_release_started = bool(was_held and event.kind == "detached")
        if event.kind not in {"unchanged", "no_match"}:
            self.assist_events.append({"sim_time_s": float(self.backend.sim.data.time),
                                       "kind": event.kind,
                                       "candidate_count": event.candidate_count})
        self.person.right_hand = np.asarray(RIGHT_HAND, dtype=float)
        self.person.update(self.backend.sim, LEFT_HAND)
        self.raw, reward, done, info = self.backend.step(command)
        self.sequence += 1
        return self.raw, reward, done, info
