"""v4 task binding for the existing two-camera RPent adapter.

The scene/evaluator are owned by the runner. This module never returns private
evaluator truth, spectator pixels, or a historical frame to RPent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.v3.bridge.b0 import _tool_result
from research.v3.bridge.libero_env import V3LiberoEnvAdapter


def validate_state_step(state, step: int) -> None:
    """Require RPent saved metric depth to match both public RGB views."""
    for camera in ("agentview", "wrist"):
        rgb = np.asarray(state.load(f"{camera}.png", step=step))
        depth = np.asarray(state.load(f"{camera}_depth.npz", step=step))
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if (rgb.ndim != 3 or rgb.shape[-1] != 3 or depth.shape != rgb.shape[:2]
                or not np.isfinite(depth).all() or not (depth > 0).all()):
            raise ValueError(f"invalid RPent {camera} RGB-D artifact at step {step}")


def validate_all_state(state) -> int:
    latest = state.latest_step
    if latest is None:
        raise ValueError("RPent state has no initial observation")
    for step in range(latest + 1):
        validate_state_step(state, step)
    return latest + 1


class V4PublicEnvAdapter(V3LiberoEnvAdapter):
    """Record a sticky sensor/control fault after a public observation fails."""

    def __init__(self, scene: Any, *, instruction: str, on_sim_step=None) -> None:
        super().__init__(scene, instruction=instruction, on_sim_step=on_sim_step)
        self.observation_fault: str | None = None

    def _latch(self, operation: str, exc: Exception) -> None:
        if self.observation_fault is None:
            self.observation_fault = f"{operation}: {type(exc).__name__}: {exc}"

    def _policy_obs(self):
        try:
            return super()._policy_obs()
        except Exception as exc:
            self._latch("public_rgbd", exc)
            raise

    def step(self, action):
        if self.observation_fault:
            raise RuntimeError("public observation fault is latched")
        try:
            return super().step(action)
        except Exception as exc:
            self._latch("sim_step_or_public_rgbd", exc)
            raise

    def raw_obs(self):
        try:
            return super().raw_obs()
        except Exception as exc:
            self._latch("raw_public_rgbd", exc)
            raise

    def render_camera(self, camera_name="agentview", height=1024, width=1024,
                      depth=False):
        try:
            return super().render_camera(camera_name, height, width, depth)
        except Exception as exc:
            self._latch("public_camera_render", exc)
            raise


class FaultStopToolkit:
    """Keep RPent tools unchanged while refusing further actions after a fault."""

    def __init__(self, base, adapter: V4PublicEnvAdapter):
        self.base = base
        self.adapter = adapter
        # The shared B0 toolkit intentionally does not count rejected calls.
        # Preserve the actual resource-boundary event for v4 scoring without
        # changing its tool result or any robot action.
        self.rejected_action_calls: list[str] = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_tools_spec(self):
        return self.base.get_tools_spec()

    def execute_tool(self, name, input_dict):
        if self.adapter.observation_fault and name != "finish":
            # A historical view_env_state frame must not replace a failed one.
            return _tool_result(name, {"error": "public observation fault; further actions stopped"})
        result = self.base.execute_tool(name, input_dict)
        if name == "finish":
            return result
        public = getattr(result, "result", None)
        if (isinstance(public, dict) and
                public.get("error") == "action call budget exhausted"):
            self.rejected_action_calls.append(name)
        if self.adapter.observation_fault:
            return _tool_result(name, {"error": "public observation fault; further actions stopped"})
        if isinstance(public, dict):
            error = public.get("error", "")
            if isinstance(error, str) and error.startswith("required RGB-D observation unavailable"):
                self.adapter._latch("rpent_depth_artifact", RuntimeError(error))
            step = public.get("step")
            if isinstance(step, int) and not self.adapter.observation_fault:
                try:
                    validate_state_step(self.base.original.state, step)
                except Exception as exc:
                    self.adapter._latch("rpent_depth_artifact", exc)
        if self.adapter.observation_fault:
            return _tool_result(name, {"error": "public observation fault; further actions stopped"})
        return result

    def cancel_active_and_wait(self):
        return self.base.cancel_active_and_wait()

    def close(self):
        return self.base.close()
