"""LIBERO-10 microwave kitchen with two public RGB-D views and spectator replay."""
import os
import tempfile
from pathlib import Path

import numpy as np

from research.v4.tasks.r3.template import ROOT, load_instance, render_bddl


class Scene:
    public_views = ('agentview', 'robot0_eye_in_hand')
    spectator_view = 'birdview'

    def __init__(self, instance_id='R3-01', seed=0, size=256):
        os.environ.setdefault('MUJOCO_GL', 'egl')
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
        os.environ.setdefault('LIBERO_CONFIG_PATH', str(ROOT / 'configs/runtime/libero'))
        from libero.libero.envs import OffScreenRenderEnv
        self.instance = load_instance(instance_id)
        self.seed = int(seed)
        self.size = int(size)
        self.sequence = 0
        self._tempdir = tempfile.TemporaryDirectory(prefix=f'safevla_{instance_id.lower()}_')
        bddl = Path(self._tempdir.name) / 'scene.bddl'
        bddl.write_text(render_bddl(self.instance))
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=self.size, camera_widths=self.size,
            camera_depths=True, render_gpu_device_id=int(os.environ.get('MUJOCO_EGL_DEVICE_ID', '0')),
            horizon=10000, ignore_done=True)
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        # LIBERO initializes both fixture pose and open door on reset; reloading XML would erase them.
        self.raw = self.env.reset()
        self.env.sim.forward()
        self.raw = self.env.env._get_observations(force_update=True)

    def step(self, action):
        self.raw, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        self.sequence += 1
        return self.raw, reward, done, info

    def capture_agent(self):
        from robosuite.utils.camera_utils import get_real_depth_map
        self.env.sim.forward()
        self.raw = self.env.env._get_observations(force_update=True)
        cameras = {}
        for name in self.public_views:
            raw_depth = np.asarray(self.raw[name + '_depth'])
            if not np.isfinite(raw_depth).all() or raw_depth.min() < 0 or raw_depth.max() > 1:
                raise RuntimeError(f'Invalid {name} native depth')
            cameras[name] = {
                'rgb': np.ascontiguousarray(self.raw[name + '_image'][::-1]),
                'depth_m': np.ascontiguousarray(get_real_depth_map(self.env.sim, raw_depth)[::-1].squeeze()),
            }
        return {'cameras': cameras, 'robot_state': {
            'eef_pos': np.asarray(self.raw['robot0_eef_pos']).copy(),
            'eef_quat': np.asarray(self.raw['robot0_eef_quat']).copy(),
            'gripper_qpos': np.asarray(self.raw['robot0_gripper_qpos']).copy(),
        }}

    def spectator_rgb(self):
        return np.ascontiguousarray(self.env.sim.render(
            width=self.size, height=self.size, camera_name=self.spectator_view)[::-1])

    def close(self):
        self.env.close()
        self._tempdir.cleanup()
