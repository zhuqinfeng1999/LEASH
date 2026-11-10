"""Isolated P4-01 LIBERO/Panda candidate; two public RGB-D cameras only.

The assembly uses original P4 assets. Evaluation contact state is never put
in capture_agent(), which is the sole planner-facing scene packet.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ASSET = ROOT / "research/v4/candidates/p4_base_20260924"
IRON_NAME = "p4_iron_1"
INSTRUCTION = "Put the iron on the ironing board."
PUBLIC_VIEWS = ("agentview", "robot0_eye_in_hand")
SPECTATOR = "p4_spectator"


def _register_iron():
    from libero.libero.envs.base_object import OBJECTS_DICT
    from robosuite.models.objects import MujocoXMLObject

    def factory(name, joints="default"):
        obj = MujocoXMLObject(str(ASSET / "iron.xml"), name=name,
                              joints=joints, obj_type="all",
                              duplicate_collision_geoms=False)
        obj.category_name = "p4_steam_iron"
        obj.rotation = (0.0, 0.0)
        obj.rotation_axis = "z"
        obj.object_properties = {"vis_site_names": {}}
        return obj
    OBJECTS_DICT["p4_steam_iron"] = factory


def _aim_camera(camera, position, target, fovy):
    p = np.asarray(position, dtype=float)
    forward = np.asarray(target, dtype=float) - p
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    for orient in ("quat", "euler", "axisangle", "zaxis"):
        camera.attrib.pop(orient, None)
    camera.set("pos", " ".join(map(str, p)))
    camera.set("xyaxes", " ".join(map(str, np.r_[right, up])))
    camera.set("fovy", str(fovy))


def _assemble(base_xml: str, board_xyyaw):
    root = ET.fromstring(base_xml)
    standalone = ET.parse(ASSET / "scene_cpu.xml").getroot()
    assets = root.find("asset")
    for name in ("p4_shirt_cotton_texture", "p4_board_cover_texture",
                 "p4_shirt_cotton", "p4_board_cover"):
        child = next((c for c in standalone.find("asset") if c.get("name") == name), None)
        if child is None:
            raise RuntimeError(f"P4 static asset missing: {name}")
        if any(c.get("name") == name for c in assets):
            raise RuntimeError(f"P4 asset name already registered: {name}")
        assets.append(copy.deepcopy(child))
    world = root.find("worldbody")
    for name in ("heat_resistant_left_tray", "tabletop_ironing_board"):
        body = standalone.find(f"worldbody/body[@name='{name}']")
        if body is None:
            raise RuntimeError(f"P4 static body missing: {name}")
        world.append(copy.deepcopy(body))
    board = world.find("body[@name='tabletop_ironing_board']")
    board.set("pos", f"{board_xyyaw[0]} {board_xyyaw[1]} .900")
    angle = float(board_xyyaw[2])
    board.set("quat", f"{np.cos(angle/2)} 0 0 {np.sin(angle/2)}")
    agent = world.find(".//camera[@name='agentview']")
    if agent is None:
        raise RuntimeError("LIBERO agentview camera missing")
    _aim_camera(agent, [.66, -.25, 2.20], [0., 0., .945], 57.)
    spectator = ET.SubElement(world, "camera", name=SPECTATOR)
    _aim_camera(spectator, [.72, -.68, 1.65], [0., 0., .94], 61.)
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    quality = visual.find("quality")
    if quality is None:
        quality = ET.SubElement(visual, "quality")
    quality.set("offsamples", "0")
    return ET.tostring(root, encoding="unicode")


class Scene:
    public_views = PUBLIC_VIEWS
    spectator_view = SPECTATOR
    instruction = INSTRUCTION
    table_top_m = .900

    def __init__(self, instance_id="P4-01", seed=0, size=256):
        if instance_id != "P4-01":
            raise ValueError("this isolated candidate supports only P4-01")
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "configs/runtime/libero"))
        from libero.libero.envs import OffScreenRenderEnv
        self.instance_id = instance_id
        self.seed = int(seed)
        self.size = int(size)
        self.sequence = 0
        _register_iron()
        rng = np.random.default_rng(self.seed)
        self.board_perturb = tuple(float(x) for x in
            (*rng.uniform(-.02,.02,2), np.deg2rad(rng.uniform(-10.,10.))))
        self.iron_perturb = tuple(float(x) for x in
            (*rng.uniform(-.02,.02,2), np.deg2rad(rng.uniform(-10.,10.))))
        board_xyyaw = (.0+self.board_perturb[0], .215+self.board_perturb[1],
                       self.board_perturb[2])
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(HERE / "scene.bddl"),
            camera_heights=self.size, camera_widths=self.size,
            camera_depths=True,
            render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
            horizon=10000, ignore_done=True,
        )
        try:
            np.random.seed(self.seed)
            self.env.seed(self.seed)
            self.raw = self.env.reset()
            self.source_scene_xml = _assemble(self.env.env.model.get_xml(), board_xyyaw)
            self.env.reset_from_xml_string(self.source_scene_xml)
            iron = self.env.env.objects_dict[IRON_NAME]
            self.iron_joint_name = iron.joints[0]
            qpos = self.env.sim.data.get_joint_qpos(self.iron_joint_name).copy()
            qpos[:3] = [self.iron_perturb[0], -.27+self.iron_perturb[1], .912]
            yaw = self.iron_perturb[2]
            qpos[3:] = [np.cos(yaw/2),0.,0.,np.sin(yaw/2)]
            self.env.sim.data.set_joint_qpos(self.iron_joint_name,qpos)
            self.env.sim.data.set_joint_qvel(self.iron_joint_name,np.zeros(6))
            self.env.sim.forward()
            self.raw = self.env.env._get_observations(force_update=True)
        except BaseException:
            self.env.close()
            raise

    def step(self, action):
        self.raw, reward, done, info = self.env.step(np.asarray(action,dtype=np.float32))
        self.sequence += 1
        return self.raw, reward, done, info

    def capture_agent(self):
        from robosuite.utils.camera_utils import get_real_depth_map
        self.env.sim.forward()
        self.raw = self.env.env._get_observations(force_update=True)
        cameras = {}
        for name in PUBLIC_VIEWS:
            cameras[name] = {
                "rgb": np.ascontiguousarray(self.raw[name+"_image"][::-1]),
                "depth_m": np.ascontiguousarray(get_real_depth_map(
                    self.env.sim,self.raw[name+"_depth"])[::-1].squeeze()),
            }
        return {"cameras": cameras,
                "eef_pos": np.asarray(self.raw["robot0_eef_pos"]).copy(),
                "eef_quat": np.asarray(self.raw["robot0_eef_quat"]).copy(),
                "gripper_qpos": np.asarray(self.raw["robot0_gripper_qpos"]).copy()}

    def spectator_rgb(self):
        return np.ascontiguousarray(self.env.sim.render(
            width=self.size,height=self.size,camera_name=SPECTATOR)[::-1])

    def close(self):
        self.env.close()
