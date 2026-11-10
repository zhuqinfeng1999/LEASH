"""E3 commit outlet: public gripper-orientation bound for held kettle tilt.

The common latch and EEF quaternion are disclosed to B0 and Ours. No hidden
object pose, authored grasp site, evaluator geometry, or spectator frame enters
the low-level gate. Cup approach still needs gate ② before transport.
"""

from __future__ import annotations

from dataclasses import replace
from math import acos, degrees

import numpy as np

from research.v4.accelerated.p3_gated_adapter import P3GatedAdapter


class E3GatedAdapter(P3GatedAdapter):
    def __init__(self, *args, **kwargs):
        self._held_reference_quat = None
        super().__init__(*args, **kwargs)
        # The Panda OSC_POSE controller maps each rotational action component
        # to at most 0.5 rad. The positive margin accounts for control error.
        self.tilt_predictor = self._predict_tilt

    def _state_from_packet(self, packet):
        state = super()._state_from_packet(packet)
        robot = packet["robot_state"]
        quat = np.asarray(robot.get("eef_quat"), dtype=float)
        if quat.shape != (4,) or not np.isfinite(quat).all():
            raise ValueError("public EEF quaternion missing for E3 tilt bound")
        length = float(np.linalg.norm(quat))
        if length < .9 or length > 1.1:
            raise ValueError("public EEF quaternion invalid")
        quat = quat / length
        if state.hold_estimate == "empty":
            self._held_reference_quat = None
            return replace(state, held_object_tiltable=False,
                           held_object_tilt_deg=None)
        if self._held_reference_quat is None:
            self._held_reference_quat = quat.copy()
        # Total EEF rotation since public latch is a conservative bound on
        # vessel pitch. Dot-product angle is independent of quaternion order.
        dot = abs(float(np.dot(quat, self._held_reference_quat)))
        tilt_upper_deg = degrees(2 * acos(float(np.clip(dot, -1., 1.))))
        return replace(state, held_object_tiltable=True,
                       held_object_tilt_deg=tilt_upper_deg)

    def public_robot_state(self):
        raw = self._p3_scene.raw
        return self._state_from_packet({"robot_state": {
            "grasp_latched": self._p3_scene.public_grasp_latched(),
            "gripper_qpos": raw["robot0_gripper_qpos"],
            "eef_quat": raw["robot0_eef_quat"],
        }})

    @staticmethod
    def _predict_tilt(action, state):
        if state.hold_estimate != "held":
            return None
        old = state.held_object_tilt_deg
        if old is None:
            return None
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("invalid public E3 controller proposal")
        return float(old + degrees(.5 * np.linalg.norm(action[3:6])) + 5.)
