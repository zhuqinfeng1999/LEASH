"""R3 official LIBERO-10 kitchen scene with unmodified BDDL initialization.

The two RGB-D public cameras are the only agent views. LIBERO's birdview is
used solely for human replay and offline QA; it is never in capture_agent().
"""
from pathlib import Path
import os

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TASK_ID = "R3"
OBJECT = "moka_pot_1"
CONTAINER = "microwave_1"


class Scene:
    public_views = ("agentview", "robot0_eye_in_hand")
    spectator_view = "birdview"

    def __init__(self, seed=0, size=256):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "configs/runtime/libero"))
        from libero.libero.envs import OffScreenRenderEnv

        self.seed = int(seed)
        self.size = int(size)
        self.sequence = 0
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(HERE / "scene.bddl"),
            camera_heights=self.size, camera_widths=self.size,
            camera_depths=True,
            render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
            horizon=10000, ignore_done=True)
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        # Important: LIBERO places the fixture body, opens its door, and
        # samples the free moka pot during reset(). Reloading XML afterward
        # erases those placements; use the built-in spectator camera instead.
        self.raw = self.env.reset()
        self.env.sim.forward()
        self.raw = self.env.env._get_observations(force_update=True)

    def step(self, action):
        self.raw, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        self.sequence += 1
        return self.raw, reward, done, info

    def capture_agent(self):
        """The agent receives only two RGB-D cameras and robot proprioception."""
        from robosuite.utils.camera_utils import get_real_depth_map
        self.env.sim.forward()
        self.raw = self.env.env._get_observations(force_update=True)
        cameras = {}
        for name in self.public_views:
            cameras[name] = {
                "rgb": np.ascontiguousarray(self.raw[name + "_image"][::-1]),
                "depth_m": np.ascontiguousarray(
                    get_real_depth_map(self.env.sim, self.raw[name + "_depth"])[::-1].squeeze()),
            }
        return {"cameras": cameras,
                "eef_pos": np.asarray(self.raw["robot0_eef_pos"]).copy(),
                "eef_quat": np.asarray(self.raw["robot0_eef_quat"]).copy(),
                "gripper_qpos": np.asarray(self.raw["robot0_gripper_qpos"]).copy()}

    def spectator_rgb(self):
        return np.ascontiguousarray(self.env.sim.render(
            width=self.size, height=self.size, camera_name=self.spectator_view)[::-1])

    def close(self):
        self.env.close()
