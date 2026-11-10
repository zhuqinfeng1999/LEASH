"""P3 native LIBERO/Panda scene; the evaluator is a separate private module.

The cup and laptop are provisional project-authored MIT candidates already
present in v2.  This module does not import any old D08 policy or supervisor.
"""

from pathlib import Path
import copy
import os
import xml.etree.ElementTree as ET

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CUP = "liquid_proxy_cup_1"
COASTER = "p3_coaster_1"
LAPTOP = "p3_laptop"
CUP_XML = ROOT / "research/v2/assets/electronics-native-v1/liquid_proxy_cup.xml"
CUP65_XML = ROOT / "research/v2/candidates/d08-cup-protocol-v2/native-adapter-v1/cup65-body-grasp-v2/assets/liquid_proxy_cup.xml"


def _register(cup_xml):
    from libero.libero.envs.base_object import OBJECTS_DICT
    from robosuite.models.objects import MujocoXMLObject

    for category, path in (("liquid_proxy_cup", cup_xml), ("p3_coaster", HERE / "coaster.xml")):
        def factory(name, joints="default", category=category, path=path):
            obj = MujocoXMLObject(str(path), name=name, joints=joints,
                                  obj_type="all", duplicate_collision_geoms=False)
            obj.category_name = category
            obj.rotation = (0.0, 0.0)
            obj.rotation_axis = "z"
            obj.object_properties = {"vis_site_names": {}}
            return obj
        OBJECTS_DICT[category] = factory


def _append_laptop(root):
    """Only the prior procedural asset's static laptop; no v2 scene state."""
    # Generate an XML tree from the immutable existing asset candidates.
    from research.v2.src.safevla_v2 import electronics_assets_v2

    candidate = electronics_assets_v2.augment(
        ET.fromstring("<mujoco><asset/><worldbody/></mujoco>"),
        laptop_position=(0.0, 0.0, 0.9),
    )
    existing = {child.get("name") for child in root.find("asset")}
    for child in candidate.find("asset"):
        if child.get("name") not in existing:
            root.find("asset").append(copy.deepcopy(child))
            existing.add(child.get("name"))
    laptop = copy.deepcopy(candidate.find("worldbody/body[@name='d08_laptop']"))
    if laptop is None:
        raise RuntimeError("P3 laptop candidate is missing")
    laptop.set("name", LAPTOP)
    # The base LIBERO XML compiles in radians; candidate standalone XML uses
    # degrees for its one lid Euler rotation.
    if root.find("compiler").get("angle") == "radian":
        for element in laptop.iter():
            if element.get("euler"):
                values = np.deg2rad(np.fromstring(element.get("euler"), sep=" "))
                element.set("euler", " ".join(map(str, values)))
    for geom in laptop.iter("geom"):
        geom.set("group", "1")
    root.find("worldbody").append(laptop)


def _camera(world, name, position, target, fovy):
    position = np.asarray(position, dtype=float)
    forward = np.asarray(target, dtype=float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    ET.SubElement(world, "camera", name=name,
                  pos=" ".join(map(str, position)),
                  xyaxes=" ".join(map(str, np.r_[right, up])),
                  fovy=str(fovy))


class P3Scene:
    public_views = ("agentview", "robot0_eye_in_hand")
    spectator_view = "p3_spectator"

    def __init__(self, seed=0, size=256, cup_asset="original"):
        if cup_asset not in ("original", "cup65"):
            raise ValueError("Unknown P3 cup asset")
        self.cup_asset = cup_asset
        cup_xml = CUP65_XML if cup_asset == "cup65" else CUP_XML
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "configs/runtime/libero"))
        from libero.libero.envs import OffScreenRenderEnv

        if not cup_xml.exists():
            raise FileNotFoundError(cup_xml)
        _register(cup_xml)
        self.seed = int(seed)
        self.size = int(size)
        self.sequence = 0
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(HERE / "scene.bddl"),
            camera_heights=self.size, camera_widths=self.size,
            camera_depths=True,
            render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
            horizon=10000, ignore_done=True,
        )
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        self.raw = self.env.reset()
        root = ET.fromstring(self.env.env.model.get_xml())
        _append_laptop(root)
        world = root.find("worldbody")
        # The work-area camera must contain cup, laptop, coaster and the route.
        agentview = world.find(".//camera[@name='agentview']")
        if agentview is not None:
            position = np.array([.65, -.40, 2.25])
            target = np.array([0., 0., .93])
            forward = target - position
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0., 0., 1.])
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            for orient in ("quat", "euler", "axisangle", "zaxis"):
                agentview.attrib.pop(orient, None)
            agentview.set("pos", " ".join(map(str, position)))
            agentview.set("xyaxes", " ".join(map(str, np.r_[right, up])))
            agentview.set("fovy", "54")
        _camera(world, self.spectator_view, [.25, 0., 2.20], [0., 0., .90], 45)
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
        # x is the robot-to-person depth axis; y is left-to-right.
        for name, nominal in ((CUP, [0., -.28, .9002]),
                              (COASTER, [0., .28, .9002])):
            obj = self.env.env.objects_dict[name]
            qpos = self.env.sim.data.get_joint_qpos(obj.joints[0]).copy()
            qpos[:3] = nominal
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

    def observation(self):
        """Exactly the v3 online observation; no spectator or private state."""
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
        return {
            "cameras": cameras,
            "robot_state": {
                "eef_pos": np.asarray(self.raw["robot0_eef_pos"]).copy(),
                "eef_quat": np.asarray(self.raw["robot0_eef_quat"]).copy(),
                "gripper_qpos": np.asarray(self.raw["robot0_gripper_qpos"]).copy(),
            },
        }

    def spectator_rgb(self):
        return np.ascontiguousarray(self.env.sim.render(
            width=self.size, height=self.size,
            camera_name=self.spectator_view)[::-1])

    def close(self):
        self.env.close()
