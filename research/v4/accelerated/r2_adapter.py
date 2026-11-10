"""v4.2 R2 RPent adapter: public 640x480, frozen π0.5 square render.

RPent's legacy 'high' observation requests 1024x1024.  We serve the same
current native 640x480 public RGB-D and its true calibration instead; no extra
high-resolution online observation is introduced.  π0.5 alone receives its
original 256x256 render at the same physical state and camera pose.
"""
from __future__ import annotations

import numpy as np

from research.v4.bridge.observation import V4PublicEnvAdapter


class R2V42Adapter(V4PublicEnvAdapter):
    PUBLIC_WIDTH = 640
    PUBLIC_HEIGHT = 480
    VLA_SIZE = 256

    def _policy_obs(self):
        # V3 first validates the *native* 640x480 public RGB-D and captures
        # proprioception. Only the two π0.5 image fields are then replaced.
        observation = super()._policy_obs()
        self._activate_render_context()
        sim = self.scene.env.sim
        observation["main_images"] = np.ascontiguousarray(
            sim.render(width=self.VLA_SIZE, height=self.VLA_SIZE,
                       camera_name="agentview")[::-1, ::-1])
        observation["wrist_images"] = np.ascontiguousarray(
            sim.render(width=self.VLA_SIZE, height=self.VLA_SIZE,
                       camera_name="robot0_eye_in_hand")[::-1, ::-1])
        return observation

    def render_camera(self, camera_name="agentview", height=1024, width=1024,
                      depth=False):
        if camera_name not in ("agentview", "robot0_eye_in_hand"):
            raise ValueError("online render must use a public camera")
        self._activate_render_context()
        raw = self.scene.raw
        rgb = np.asarray(raw[camera_name + "_image"]).copy()
        if rgb.shape != (self.PUBLIC_HEIGHT, self.PUBLIC_WIDTH, 3):
            raise ValueError("public RGB is not native 640x480")
        if not depth:
            return rgb
        normalized = np.asarray(raw[camera_name + "_depth"]).copy()
        if normalized.shape[:2] != rgb.shape[:2]:
            raise ValueError("public RGB-D mismatch")
        return rgb, normalized

    def get_camera_meta(self, camera_name="agentview", height=256, width=256):
        # RPent hardcodes 256/1024 metadata requests.  The actual saved public
        # frames are 640x480; calibration must follow the pixels, not request.
        return super().get_camera_meta(camera_name,
                                       self.PUBLIC_HEIGHT, self.PUBLIC_WIDTH)
