"""P1-01 native LIBERO scene. The disputed reachability layout is not applied."""

import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from research.safevla.person import Person, fmt, quat
from .template import ROOT, load_instance, render_bddl


ITEM_NOMINAL_XY = np.array([.35, -.34])
TARGET_NOMINAL_XY = np.array([.35, .26])
HAND_NOMINAL_XY = np.array([.16, 0.])
POSITION_JITTER_M = .02
YAW_JITTER_DEG = 10.


def _register(instance):
    """Register the three task roles without rebuilding or modifying v3 assets."""
    from libero.libero.envs.base_object import OBJECTS_DICT
    from robosuite.models.objects import MujocoXMLObject

    for asset in (instance.item, instance.target, instance.person):
        category = asset["category"]
        path = ROOT / asset["xml"]

        def factory(name, joints="default", category=category, path=path):
            obj = MujocoXMLObject(str(path), name=name, joints=joints, obj_type="all",
                                  duplicate_collision_geoms=category == "blunt_tool")
            obj.category_name = category
            obj.rotation = (0., 0.)
            obj.rotation_axis = "z"
            obj.object_properties = {"vis_site_names": {}}
            return obj

        OBJECTS_DICT[category] = factory


class Scene:
    public_views = ("agentview", "robot0_eye_in_hand")
    spectator_view = "spectator"

    def __init__(self, instance_id="P1-01", seed=0, size=256):
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
        self.person = Person()
        rng = np.random.default_rng(self.seed)
        jitter = rng.uniform(-POSITION_JITTER_M, POSITION_JITTER_M, (3, 2))
        yaw = rng.uniform(-np.deg2rad(YAW_JITTER_DEG), np.deg2rad(YAW_JITTER_DEG), 2)
        self.hand = np.r_[HAND_NOMINAL_XY + jitter[2], .938]
        root = ET.fromstring(self.person.augment(self.env.env.model.get_xml(), self.hand))
        world = root.find("worldbody")
        camera = world.find(".//camera[@name='agentview']")
        if camera is not None:
            position = np.array([.72, -.88, 1.72])
            target = np.array([.30, -.02, .91])
            z = (position - target) / np.linalg.norm(position - target)
            x = np.cross([0, 0, 1], z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            camera.set("pos", fmt(position))
            camera.set("quat", fmt(quat(np.column_stack([x, y, z]))))
            camera.set("fovy", "62")
        for extra in list(world.findall("camera")):
            if extra.get("name") == "research_top":
                world.remove(extra)
            elif extra.get("name") == "research_overview":
                extra.set("name", self.spectator_view)
        self.env.reset_from_xml_string(ET.tostring(root, encoding="unicode"))
        self.person.update(self.env.sim, self.hand)
        self.move_object(self.instance.item["name"],
                         np.r_[ITEM_NOMINAL_XY + jitter[0], self.instance.item["initial_z_m"]], yaw[0])
        self.move_object(self.instance.target["name"],
                         np.r_[TARGET_NOMINAL_XY + jitter[1], self.instance.target["initial_z_m"]], yaw[1])
        self.move_object(self.instance.person["name"], self.hand)
        for _ in range(15):
            self.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
        self._stage_robot()

    def _stage_robot(self):
        target = np.array([.20, -.05, 1.28])
        for _ in range(90):
            error = target - np.asarray(self.raw["robot0_eef_pos"])
            if np.linalg.norm(error) < .015:
                break
            action = np.zeros(7, dtype=np.float32)
            action[:3] = np.clip(error, -.025, .025) / .05
            action[6] = -1
            self.step(action)

    def move_object(self, name, xyz, yaw=None):
        """Reset-time placement only; never exposed as an agent tool."""
        obj = self.env.env.objects_dict[name]
        joint = obj.joints[0]
        pose = self.env.sim.data.get_joint_qpos(joint).copy()
        pose[:3] = xyz
        if yaw is not None:
            pose[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.env.sim.data.set_joint_qpos(joint, pose)
        self.env.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.env.sim.forward()
        self.raw = self.env.env._get_observations(force_update=True)

    def step(self, action):
        self.person.update(self.env.sim, self.hand)
        self.raw, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        self.sequence += 1
        return self.raw, reward, done, info

    def capture_agent(self):
        """Frozen online budget: exactly two RGB-D cameras and robot state."""
        from robosuite.utils.camera_utils import get_real_depth_map

        cameras = {}
        for name in self.public_views:
            cameras[name] = {
                "rgb": np.ascontiguousarray(self.raw[name + "_image"][::-1]),
                "depth_m": np.ascontiguousarray(get_real_depth_map(
                    self.env.sim, self.raw[name + "_depth"])[::-1].squeeze()),
            }
        return {"cameras": cameras, "robot_state": {
            "eef_pos": np.asarray(self.raw["robot0_eef_pos"]).copy(),
            "eef_quat": np.asarray(self.raw["robot0_eef_quat"]).copy(),
            "gripper_qpos": np.asarray(self.raw["robot0_gripper_qpos"]).copy(),
        }}

    def spectator_rgb(self):
        return np.ascontiguousarray(self.env.sim.render(
            width=self.size, height=self.size, camera_name=self.spectator_view)[::-1])

    def close(self):
        try:
            self.env.close()
        finally:
            self._tempdir.cleanup()
