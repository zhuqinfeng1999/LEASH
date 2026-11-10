"""Public S3 bridge shape for RPent; no model or reviewer starts on import.

The scripted scorer uses ``S3Scene`` directly. This narrow facade adapts its
public packet to the already-used R2V42Adapter without exposing any physical
truth, authored grasp region, evaluator predicate, or private object pose.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from research.v4.accelerated import tabletop
from .scene import S3Scene
from .task import INSTRUCTIONS


class S3RPentFacade(S3Scene):
    public_views = tabletop.PUBLIC_VIEWS
    spectator_view = tabletop.SPECTATOR_VIEW

    def __init__(self, seed=0, *, camera_observations=False,
                 render_gpu_device_id=-1):
        super().__init__(seed, camera_observations=camera_observations,
                         render_gpu_device_id=render_gpu_device_id)
        self.env = SimpleNamespace(sim=self.backend.sim, env=self.backend)

    @property
    def instruction(self):
        return INSTRUCTIONS[self.round_number - 1]

    def advance_round(self):
        self.start_round(self.round_number + 1)
        return self.instruction

    def public_grasp_latched(self):
        return bool(self.grasp_assist.held)

    def capture_agent(self):
        packet = super().capture_agent()
        # V3/RPent adapter expects these exact three public proprioceptive
        # fields at top level; the validated robot_state remains intact.
        packet.update({key: packet["robot_state"][key]
                       for key in ("eef_pos", "eef_quat", "gripper_qpos")})
        return packet

    def spectator_rgb(self):
        frame = self.backend.sim.render(
            width=640, height=480, camera_name=self.spectator_view)
        return np.ascontiguousarray(frame[::-1])
