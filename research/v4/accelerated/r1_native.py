"""R1 v4.2 thin carrier: public 640x480 views and common physical grasp assist.

The v4 LIBERO scene supplies the robot/controller until the tabletop carrier
can be migrated without rewriting those behaviours. Both methods share this
physical adapter; the simulator-only hazard evaluator stays separate.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from research.v4.tasks.r1.scene import Scene as V4Scene, _camera, _register
from research.v4.tasks.r1.template import ROOT, load_instance, render_bddl
from research.v4.accelerated.grasp_assist import GraspAssist, inactive_weld_element


class Scene(V4Scene):
    def __init__(self, instance_id: str = "R1-01", seed: int = 0):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "configs/runtime/libero"))
        from libero.libero.envs import OffScreenRenderEnv

        self.instance = load_instance(instance_id)
        self.seed = int(seed)
        self.size = 480
        self.sequence = 0
        self._tempdir = tempfile.TemporaryDirectory(prefix=f"safevla_v42_{instance_id.lower()}_")
        bddl = Path(self._tempdir.name) / "scene.bddl"
        bddl.write_text(render_bddl(self.instance))
        _register(self.instance)
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=480,
            camera_widths=640, camera_depths=True,
            render_gpu_device_id=int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")),
            horizon=10000, ignore_done=True,
        )
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        self.raw = self.env.reset()
        root = ET.fromstring(self.env.env.model.get_xml())
        _camera(root.find("worldbody"))
        hand = root.find(".//body[@name='gripper0_right_eef']")
        left = root.find(".//body[@name='gripper0_right_finger_joint1_tip']")
        right = root.find(".//body[@name='gripper0_right_finger_joint2_tip']")
        phone = root.find(".//body[@name='phone_1_main']")
        if any(body is None for body in (hand, left, right, phone)):
            raise RuntimeError("R1 grasp-assist body mapping changed")
        ET.SubElement(left, "site", name="v42_r1_left_pad", pos="0 -.005 -.015",
                      size=".001", rgba="0 0 0 0")
        ET.SubElement(right, "site", name="v42_r1_right_pad", pos="0 .005 -.015",
                      size=".001", rgba="0 0 0 0")
        ET.SubElement(phone, "site", name="v42_r1_phone_grasp", pos="0 0 0",
                      size=".002", rgba="0 0 0 0")
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        equality.append(inactive_weld_element(
            name="v42_r1_phone_weld", hand_body="gripper0_right_eef",
            object_body="phone_1_main"))
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
        self.grasp_assist = GraspAssist(
            self.env.sim.model._model, self.env.sim.data._data,
            hand_body="gripper0_right_eef", center_site="gripper0_right_grip_site",
            left_pad_site="v42_r1_left_pad", right_pad_site="v42_r1_right_pad",
            slot_metadata={"L": {"grasp_zones": [{
                "object_body": "phone_1_main",
                "zone_site": "v42_r1_phone_grasp",
                "weld_name": "v42_r1_phone_weld",
                "tolerance_xyz": [.015, .015, .020],
            }]}})
        self.assist_events = []

    def step(self, action):
        event = self.grasp_assist.before_control(float(np.asarray(action)[6]),
                                                 source="common-v42-controller")
        if event.kind != "unchanged":
            self.assist_events.append({"sim_time": float(self.env.sim.data.time),
                                       "kind": event.kind,
                                       "candidate_count": event.candidate_count})
        return super().step(action)
