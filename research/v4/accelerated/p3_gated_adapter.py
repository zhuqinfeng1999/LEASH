"""P3 v4.2 low-level LEASH outlet over the common RPent camera adapter.

Online gates consume the public simulated grasp_latched Boolean available to
B0 and Ours. No object identity, grasp zone, evaluator state, or spectator view
is read here. P3 has no heated-device door.
"""
from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
import time

import numpy as np
from PIL import Image

from research.v4.accelerated.r2_adapter import R2V42Adapter
from research.v4.candidates.leash_commit_gate_20260924.commit_gate import (
    CameraFrame, CommitLease, PublicEvidence, PublicRobotState,
)
from research.v4.candidates.leash_outlet_adapter_20260924.outlet_adapter import (
    DoorIntent, OursCommitEnvAdapter,
)


def _p3_door_event(_action: np.ndarray) -> DoorIntent | None:
    """P3's fixed tabletop has no heated-device door or door actuator."""
    return None


class P3GatedAdapter(OursCommitEnvAdapter, R2V42Adapter):
    """Gate every analytic, hold, and pi0.5 chunk command before scene.step.

    The R2V42Adapter methods retained by the MRO render public RGB-D at
    640x480 and the policy's original 256x256 views.
    """

    def __init__(self, scene, *, instruction: str, reviewer,
                 default_commit_conditions: tuple[str, ...], on_sim_step=None):
        self._p3_scene = scene
        self.commit_decisions = []
        super().__init__(
            scene, instruction=instruction,
            public_state=self.public_robot_state,
            capture=self.capture_commit_evidence,
            reviewer=reviewer,
            door_classifier=_p3_door_event,
            default_commit_conditions=default_commit_conditions,
            review_timeout_s=180.0,
            on_sim_step=on_sim_step,
        )

    def step(self, action):
        previous = self.last_outcome
        try:
            return super().step(action)
        finally:
            outcome = self.last_outcome
            if outcome is not None and outcome is not previous:
                self.commit_decisions.extend(asdict(row) for row in outcome.gate_decisions)

    @staticmethod
    def _png(frame: np.ndarray) -> bytes:
        buffer = BytesIO()
        Image.fromarray(frame, mode="RGB").save(buffer, format="PNG")
        return buffer.getvalue()

    @classmethod
    def _depth_png(cls, depth_m: np.ndarray) -> bytes:
        valid = np.isfinite(depth_m) & (depth_m > 0)
        unit = np.clip(np.nan_to_num(depth_m, nan=0., posinf=2.) / 2., 0., 1.)
        color = np.stack((255 * unit, 255 * (1. - unit),
                          np.full_like(unit, 90.)), axis=-1).astype(np.uint8)
        color[~valid] = (255, 0, 255)
        return cls._png(color)

    def _sensor_packet(self, *, refresh: bool = False):
        if refresh:
            self._activate_render_context()
        # The facade renders a fresh native public packet and advances its
        # capture id even when the physical state has not stepped.
        packet = self._p3_scene.capture_agent()
        if set(packet["cameras"]) != {"agentview", "robot0_eye_in_hand"}:
            raise ValueError("P3 online camera set is not exactly two public views")
        return packet

    def _state_from_packet(self, packet) -> PublicRobotState:
        robot = packet["robot_state"]
        latch = robot.get("grasp_latched")
        if type(latch) is not bool:
            raise ValueError("common public grasp_latched Boolean is unavailable")
        qpos = np.asarray(robot["gripper_qpos"], dtype=float)
        if qpos.size < 2 or not np.isfinite(qpos).all():
            raise ValueError("public gripper proprioception is invalid")
        return PublicRobotState(
            gripper_closed=bool(np.mean(np.abs(qpos[:2])) < .005),
            hold_estimate="held" if latch else "empty",
            # Generic latch identifier, not an object label.
            held_object_name="latched_payload" if latch else None,
            held_object_tiltable=None if latch else False,
            held_object_tilt_deg=None,
            observed_monotonic_ns=time.monotonic_ns(),
        )

    def public_robot_state(self) -> PublicRobotState:
        # This is the cheap accessor for the *same* declared latch bit and
        # gripper proprioception shipped in capture_agent(). It avoids two
        # full 640x480 depth conversions for every ordinary controller step.
        raw = self._p3_scene.raw
        latch = self._p3_scene.public_grasp_latched()
        return self._state_from_packet({"robot_state": {
            "grasp_latched": latch,
            "gripper_qpos": raw["robot0_gripper_qpos"],
        }})

    def capture_commit_evidence(self) -> PublicEvidence:
        captured_ns = time.monotonic_ns()
        packet = self._sensor_packet(refresh=True)
        state = self._state_from_packet(packet)
        frames = []
        for name in ("agentview", "robot0_eye_in_hand"):
            camera = packet["cameras"][name]
            rgb = np.asarray(camera["rgb"], dtype=np.uint8)
            depth = np.asarray(camera["depth_m"], dtype=float)
            k = np.asarray(camera["intrinsics"], dtype=float)
            e = np.asarray(camera["extrinsics"], dtype=float)
            if (camera.get("frame_id") != packet["frame_id"] or
                    rgb.shape != (480, 640, 3) or depth.shape != (480, 640) or
                    k.shape != (3, 3) or e.shape != (4, 4) or
                    not np.isfinite(depth).all() or not (depth > 0).all() or
                    not np.isfinite(k).all() or not np.isfinite(e).all()):
                raise ValueError("P3 public RGB-D/calibration is invalid")
            frames.append(CameraFrame(
                name, self._png(rgb), self._depth_png(depth), 640, 480,
                {"K": k.tolist(), "depth_color_scale_m": [0., 2.]},
                {"camera_to_world": e.tolist()}, captured_ns))
        return PublicEvidence(*frames, state, time.monotonic_ns())

    def set_commit_conditions(self, conditions: tuple[str, ...] | None) -> None:
        self.lease = CommitLease(tuple(conditions)) if conditions else None
