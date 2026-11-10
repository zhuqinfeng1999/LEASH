"""R2 native LIBERO/Panda scene instantiated from the two named asset slots."""
from pathlib import Path
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from research.v4.tasks.r2.template import HERE, ROOT, load_instance, render_bddl


def _register(instance):
    from libero.libero.envs.base_object import OBJECTS_DICT
    from robosuite.models.objects import MujocoXMLObject

    for asset in (instance.source, instance.target):
        category, xml = asset["category"], HERE / asset["xml"]

        def factory(name, joints="default", category=category, xml=xml):
            obj = MujocoXMLObject(str(xml), name=name, joints=joints,
                                  obj_type="all", duplicate_collision_geoms=False)
            obj.category_name = category
            obj.rotation = (0.0, 0.0)
            obj.rotation_axis = "z"
            obj.object_properties = {"vis_site_names": {}}
            return obj
        OBJECTS_DICT[category] = factory


def _camera(world):
    position = np.array([.78, -.76, 1.52])
    target = np.array([-.06, 0., .94])
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    ET.SubElement(world, "camera", name="v4_spectator",
                  pos=" ".join(map(str, position)),
                  xyaxes=" ".join(map(str, np.r_[right, up])), fovy="52")


class Scene:
    public_views = ("agentview", "robot0_eye_in_hand")
    spectator_view = "v4_spectator"

    def __init__(self, instance_id="R2-01", seed=0, size=256):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "configs/runtime/libero"))
        from libero.libero.envs import OffScreenRenderEnv

        self.instance = load_instance(instance_id)
        self.seed = int(seed)
        self.size = int(size)
        self.sequence = 0
        self._tempdir = tempfile.TemporaryDirectory(prefix=f"safevla_{instance_id.lower()}_")
        bddl = Path(self._tempdir.name) / "scene.bddl"
        bddl.write_text(render_bddl(self.instance))
        _register(self.instance)
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=self.size,
            camera_widths=self.size, camera_depths=True,
            render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
            horizon=10000, ignore_done=True)
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        self.raw = self.env.reset()
        root = ET.fromstring(self.env.env.model.get_xml())
        _camera(root.find("worldbody"))
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        quality = visual.find("quality")
        if quality is None:
            quality = ET.SubElement(visual, "quality")
        quality.set("offsamples", "0")
        self.source_scene_xml = ET.tostring(root, encoding="unicode")
        self.env.reset_from_xml_string(self.source_scene_xml)
        rng = np.random.default_rng(self.seed)
        for role, nominal_y in (("source", -.16), ("target", .16)):
            asset = getattr(self.instance, role)
            obj = self.env.env.objects_dict[asset["name"]]
            qpos = self.env.sim.data.get_joint_qpos(obj.joints[0]).copy()
            qpos[:3] = [-.06, nominal_y, asset["initial_z_m"]]
            qpos[:2] += rng.uniform(-.02, .02, 2)
            yaw = np.deg2rad(rng.uniform(-10., 10.))
            qpos[3:] = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
            self.env.sim.data.set_joint_qpos(obj.joints[0], qpos)
            self.env.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
        self.env.sim.forward()
        self.raw = self.env.env._get_observations(force_update=True)

    def step(self, action):
        self.raw, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        self.sequence += 1
        return self.raw, reward, done, info

    def capture_agent(self):
        """Exactly the two public RGB-D cameras plus proprioception."""
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
        self._tempdir.cleanup()
