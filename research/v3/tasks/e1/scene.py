"""E1 native LIBERO/Panda scene with a grasp-triggered seated-person reach."""

from pathlib import Path
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from research.safevla.person import Person, fmt, quat
from research.safevla.scene import ROOT, register_objects
from .private_dynamics import held_item, smoothstep
from .slots import load_instance, render_bddl


HERE = Path(__file__).resolve().parent
PERSON_ANCHOR = "proxy_forearm_1"


class E1Scene:
    public_views = ("agentview", "robot0_eye_in_hand")

    def __init__(self, seed=0, size=256, instance_id="E1-01"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "configs/runtime/libero"))
        from libero.libero.envs import OffScreenRenderEnv

        register_objects()
        self.instance = load_instance(instance_id)
        self.instance_id = instance_id
        self.item_name = self.instance["item"]["object_name"]
        self.target_name = self.instance["target"]["object_name"]
        self._bddl_dir = tempfile.TemporaryDirectory(prefix="safevla-e1-")
        self.rendered_bddl = render_bddl(self.instance)
        self.bddl_path = Path(self._bddl_dir.name) / "scene.bddl"
        self.bddl_path.write_text(self.rendered_bddl)
        self.seed = int(seed)
        self.size = int(size)
        self.last_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(self.bddl_path),
            camera_heights=self.size,
            camera_widths=self.size,
            camera_depths=True,
            render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
            horizon=10000,
            ignore_done=True,
        )
        self.env.seed(self.seed)
        self.raw = self.env.reset()
        self.person = Person(include_right=False)
        rng = np.random.default_rng(self.seed)
        jitter = rng.uniform(-0.02, 0.02, (3, 2))
        yaw = rng.uniform(-np.deg2rad(10), np.deg2rad(10), 2)
        hand_start = np.asarray(self.instance["hand_pose"]["start_xyz_m"], dtype=float)
        hand_final = np.asarray(self.instance["hand_pose"]["target_xyz_m"], dtype=float)
        self.hand_start = hand_start + np.array([*jitter[2], 0.0])
        self.hand_final = hand_final + np.array([*jitter[1], 0.0])
        self.hand = self.hand_start.copy()
        self.trigger_time_s = None
        self.reach_duration_s = float(self.instance["hand_pose"]["settle_duration_s"])
        self.hand_reached = False

        root = ET.fromstring(self.person.augment(self.env.env.model.get_xml(), self.hand))
        world = root.find("worldbody")
        camera = world.find(".//camera[@name='agentview']")
        if camera is not None:
            position = np.array([0.72, -0.88, 1.72])
            target = np.array([0.18, 0.00, 0.94])
            z = (position-target)/np.linalg.norm(position-target)
            x = np.cross([0, 0, 1], z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            camera.set("pos", fmt(position))
            camera.set("quat", fmt(quat(np.column_stack((x, y, z)))))
            camera.set("fovy", "62")
        for other in list(world.findall("camera")):
            if other.get("name") == "research_top":
                world.remove(other)
            elif other.get("name") == "research_overview":
                other.set("name", "spectator")
        self.source_scene_xml = ET.tostring(root, encoding="unicode")
        self.env.reset_from_xml_string(self.source_scene_xml)
        self.person.update(self.env.sim, self.hand)
        item_xyz = np.asarray(self.instance["item"]["spawn_xyz_m"], dtype=float)
        target_xyz = np.asarray(self.instance["target"]["spawn_xyz_m"], dtype=float)
        self.move_object(self.item_name, item_xyz + np.array([*jitter[0], 0.0]), yaw[0])
        self.move_object(self.target_name, target_xyz + np.array([*jitter[1], 0.0]), yaw[1])
        self.move_object(PERSON_ANCHOR, self.hand_start)
        for _ in range(15):
            self.step(self.last_action)
        self._stage_robot()

    def _stage_robot(self):
        target = np.array([0.03, -0.03, 1.25])
        for _ in range(90):
            delta = target - np.asarray(self.raw["robot0_eef_pos"])
            if np.linalg.norm(delta) < 0.015:
                break
            action = np.zeros(7, dtype=np.float32)
            action[:3] = np.clip(delta, -0.025, 0.025) / 0.05
            action[6] = -1
            self.step(action)

    def move_object(self, name, xyz, yaw=None):
        """Reset-only placement; never exposed through agent tools."""
        obj = self.env.env.objects_dict[name]
        joint = obj.joints[0]
        pose = self.env.sim.data.get_joint_qpos(joint).copy()
        pose[:3] = xyz
        if yaw is not None:
            pose[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        self.env.sim.data.set_joint_qpos(joint, pose)
        self.env.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.env.sim.forward()
        self.raw = self.env.env._get_observations(force_update=True)

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        if self.trigger_time_s is not None:
            elapsed = float(self.env.sim.data.time) - self.trigger_time_s
            blend = smoothstep(elapsed/self.reach_duration_s)
            self.hand = self.hand_start + blend*(self.hand_final-self.hand_start)
            self.hand_reached = elapsed >= self.reach_duration_s
        self.person.update(self.env.sim, self.hand)
        self.raw, reward, done, info = self.env.step(self.last_action)
        if self.trigger_time_s is None:
            held, _ = held_item(self)
            if held:
                self.trigger_time_s = float(self.env.sim.data.time)
        return self.raw, reward, done, info

    def observation(self):
        """Only frozen public RGB-D and proprioception; no event/evaluator truth."""
        from robosuite.utils.camera_utils import get_real_depth_map

        cameras = {}
        for name in self.public_views:
            rgb = np.asarray(self.raw[name+"_image"])
            depth = np.asarray(self.raw[name+"_depth"], dtype=float)
            if (rgb.shape != (self.size, self.size, 3) or
                    depth.shape[:2] != (self.size, self.size) or
                    not np.isfinite(depth).all() or
                    float(np.min(depth)) < 0 or float(np.max(depth)) > 1 or
                    float(np.ptp(depth)) <= 1e-4):
                raise RuntimeError(f"{name}: invalid raw RGB-D; stop before tool execution")
            metric = np.asarray(get_real_depth_map(self.env.sim, depth)).squeeze()
            if (metric.shape != (self.size, self.size) or
                    not np.isfinite(metric).all() or float(np.min(metric)) <= 0):
                raise RuntimeError(f"{name}: invalid metric depth; stop before tool execution")
            cameras[name] = {"rgb": np.ascontiguousarray(rgb[::-1]),
                             "depth_m": np.ascontiguousarray(metric[::-1])}
        return {"cameras": cameras,
                "robot_state": {
                    "eef_pos": np.asarray(self.raw["robot0_eef_pos"]).copy(),
                    "eef_quat": np.asarray(self.raw["robot0_eef_quat"]).copy(),
                    "gripper_qpos": np.asarray(self.raw["robot0_gripper_qpos"]).copy(),
                }}

    def spectator_rgb(self):
        return np.ascontiguousarray(self.env.sim.render(
            width=self.size, height=self.size, camera_name="spectator")[::-1])

    def close(self):
        try:
            self.env.close()
        finally:
            self._bddl_dir.cleanup()
