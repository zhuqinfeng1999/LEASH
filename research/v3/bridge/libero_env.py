"""RPent-compatible, evaluator-blind adapter for v3 LIBERO/Panda scenes.

The scene owner keeps the MuJoCo lifecycle and supplies an optional private
per-step evaluator callback. No GPT/VLA call occurs here.
"""

from __future__ import annotations

from typing import Any, Callable
import math

import numpy as np


PUBLIC_CAMERAS = ("agentview", "robot0_eye_in_hand")
RAW_KEYS = (
    "agentview_image", "agentview_depth",
    "robot0_eye_in_hand_image", "robot0_eye_in_hand_depth",
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
)

# Optional human-only video sink. Formal Root owns it; no value returns to a
# planner, registry, reviewer, physics step or VLA input.
PUBLIC_FRAME_SINK = None


class V3LiberoEnvAdapter:
    """Present a v3 Scene to unchanged RPent ``LiberoPrimitives``."""

    return_all_frames = False
    terminated = False
    truncated = False

    def __init__(
        self, scene: Any, *, instruction: str,
        on_sim_step: Callable[[Any], None] | None = None,
    ) -> None:
        self.scene = scene
        self.instruction = instruction
        self.on_sim_step = on_sim_step
        self._reset_given = False
        self._last_gripper = -1.0
        if tuple(scene.public_views) != PUBLIC_CAMERAS:
            raise ValueError("v3 scene must expose exactly agentview and robot0_eye_in_hand")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("task instruction is required")

    def _activate_render_context(self) -> None:
        """Keep robosuite's existing EGL context current before each render phase.

        The pinned MjRenderContext.render already makes its context current.
        This extra binding covers control/capture paths around renders, but
        whether it repairs E0001 remains unverified until a native probe.
        RPent also enlarges the offscreen framebuffer to 1024 pixels.
        """
        context = getattr(self.scene.env.sim, "_render_context_offscreen", None)
        if context is not None:
            context.gl_ctx.make_current()

    def _policy_obs(self) -> dict[str, Any]:
        from robosuite.utils.transform_utils import quat2axisangle

        self._activate_render_context()
        packet = self.scene.capture_agent()
        if set(packet["cameras"]) != set(PUBLIC_CAMERAS):
            raise ValueError("scene returned extra or missing agent cameras")
        for name in PUBLIC_CAMERAS:
            rgb = np.asarray(packet["cameras"][name]["rgb"])
            depth = np.asarray(packet["cameras"][name]["depth_m"])
            if (rgb.ndim != 3 or rgb.shape[-1] != 3 or
                    depth.shape != rgb.shape[:2] or not np.isfinite(depth).all() or
                    not (depth > 0).all()):
                raise ValueError(f"invalid current {name} RGB-D frame")
        eef = np.asarray(packet["eef_pos"], dtype=np.float32)
        quat = np.asarray(packet["eef_quat"], dtype=np.float64)
        grip = np.asarray(packet["gripper_qpos"], dtype=np.float32)
        if eef.shape != (3,) or quat.shape != (4,) or grip.shape != (2,):
            raise ValueError("invalid robot state shape")
        if (not np.isfinite(eef).all() or not np.isfinite(quat).all()
                or not np.isfinite(grip).all()
                or not np.isclose(np.linalg.norm(quat), 1., atol=1e-5, rtol=0.)):
            raise ValueError("invalid finite public robot pose/quaternion")
        if PUBLIC_FRAME_SINK is not None:
            PUBLIC_FRAME_SINK(self, packet)
        # RLinf Pi0.5's original LIBERO image convention is raw[::-1,::-1].
        # Scene.capture_agent gives calibrated RGB raw[::-1].
        observation = {
            "main_images": np.ascontiguousarray(packet["cameras"]["agentview"]["rgb"][:, ::-1]),
            "wrist_images": np.ascontiguousarray(packet["cameras"]["robot0_eye_in_hand"]["rgb"][:, ::-1]),
            "extra_view_images": None,
            # Keep RPent/RLinf's original xyzw axis-angle convention, including
            # negative w. A shortest-arc conversion changes the VLA's input.
            "states": np.concatenate((eef, quat2axisangle(quat.copy()), grip)).astype(np.float32),
            "task_descriptions": self.instruction,
        }
        self._last_obs = observation
        return observation

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        # The v3 Scene is already seeded, randomized and reset before construction.
        # Calling its underlying env.reset here would erase the v3 layout.
        if self._reset_given:
            raise RuntimeError("v3 episode cannot be reset")
        self._reset_given = True
        return self._policy_obs(), {}

    def step(self, action) -> tuple[dict[str, Any], float, bool, bool, dict]:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("RPent action must be a finite 7-vector")
        self._last_gripper = float(value[6])
        self._activate_render_context()
        self.scene.step(value)
        if self.on_sim_step is not None:
            self.on_sim_step(self.scene)  # Private evaluator sink; no return value enters the agent.
        return self._policy_obs(), 0.0, False, False, {}

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        chunk = np.asarray(actions, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 7 or not len(chunk):
            raise ValueError("RPent action chunk must be [N,7] with N>0")
        frames = [self.step(action)[0] for action in chunk]
        n = len(frames)
        return (frames if return_all_frames else frames[-1],
                np.zeros(n, dtype=np.float32),
                np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), {})

    def raw_obs(self) -> dict[str, np.ndarray]:
        raw = self.scene.raw
        if any(name not in raw for name in RAW_KEYS):
            raise ValueError("v3 scene raw observation lacks a required public sensor")
        for name in PUBLIC_CAMERAS:
            depth = np.asarray(raw[name + "_depth"])
            if not (np.isfinite(depth).all() and (depth >= 0).all() and
                    (depth <= 1).all()):
                raise ValueError(f"invalid current {name} normalized depth")
        return {name: np.asarray(raw[name]).copy() for name in RAW_KEYS}

    def render_camera(self, camera_name="agentview", height=1024, width=1024,
                      depth=False):
        if camera_name not in PUBLIC_CAMERAS:
            raise ValueError("spectator or unknown camera is not agent-visible")
        self._activate_render_context()
        frame = self.scene.env.sim.render(
            width=int(width), height=int(height), camera_name=camera_name,
            depth=bool(depth),
        )
        if depth:
            normalized = np.asarray(frame[1])
            if not (np.isfinite(normalized).all() and
                    (normalized >= 0).all() and (normalized <= 1).all()):
                raise ValueError(f"invalid current {camera_name} normalized depth")
        return frame

    def get_camera_meta(self, camera_name="agentview", height=256, width=256):
        if camera_name not in PUBLIC_CAMERAS:
            raise ValueError("spectator or unknown camera is not agent-visible")
        from robosuite.utils.camera_utils import (
            get_camera_intrinsic_matrix, get_camera_extrinsic_matrix,
        )

        sim = self.scene.env.sim
        height, width = int(height), int(width)
        K = get_camera_intrinsic_matrix(sim, camera_name, height, width)
        E = get_camera_extrinsic_matrix(sim, camera_name)
        extent = float(sim.model.stat.extent)
        return {
            "camera_name": camera_name, "height": height, "width": width,
            "intrinsic_K": np.asarray(K).tolist(),
            "extrinsic_cam2world": np.asarray(E).tolist(),
            "depth_near": float(sim.model.vis.map.znear) * extent,
            "depth_far": float(sim.model.vis.map.zfar) * extent,
        }

    def get_task_language(self) -> str:
        return self.instruction

    def hold(self, seconds: float) -> dict[str, Any]:
        """Advance physics with zero OSC delta and the last gripper command."""
        if not self._reset_given:
            raise RuntimeError("RPent toolkit must initialize before hold")
        backend = self.scene.env.env
        dt = getattr(backend, "control_timestep", None)
        if dt is None:
            freq = getattr(backend, "control_freq", None)
            dt = 1.0 / float(freq) if freq else None
        if dt is None or not math.isfinite(float(dt)) or float(dt) <= 0:
            raise RuntimeError("v3 control timestep is unavailable")
        n = max(1, math.ceil(float(seconds) / float(dt)))
        action = np.array([0., 0., 0., 0., 0., 0., self._last_gripper], dtype=np.float32)
        for _ in range(n):
            self.step(action)
        return {"sim_time_advanced_s": n * float(dt), "simulation_steps": n,
                "_rpent_obs": self._last_obs}
