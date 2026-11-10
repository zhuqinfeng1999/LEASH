"""S3-01 task-local binding on the common v4.2 tabletop.

The common tabletop, controller, cameras, and grasp assist are reused without
changing their source. Private scoring is deliberately outside this module.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist

ROOT = Path(__file__).resolve().parents[4]
REL = "research/v4/candidates/s3_v42_tabletop_20260926"

BINDINGS = (
    tabletop.ObjectBinding("s3_plate_1", f"{REL}/plate.xml", "L",
                           grasp_local=(-.069, 0., .014),
                           xy_offset=(-.02, .04)),
    tabletop.ObjectBinding("s3_placemat_1", f"{REL}/placemat.xml", "L",
                           fixed=True, xy_offset=(.18, .04)),
    tabletop.ObjectBinding("s3_ketchup_1",
        "research/v4/candidates/s2_release_props_20260924/ketchup.xml", "R",
        z_offset=.018, grasp_local=(-.034, 0., .052), xy_offset=(-.02, -.04)),
    tabletop.ObjectBinding("s3_basket_1",
        f"{REL}/basket_s3.xml", "R",
        fixed=True, xy_offset=(.18, .05)),
    tabletop.ObjectBinding("s3_cookie_1", f"{REL}/cookie.xml", "C",
                           z_offset=.005, grasp_local=(0., 0., .010),
                           xy_offset=(.10, 0.)),
    tabletop.ObjectBinding("s3_cookie_box_1", f"{REL}/cookie_box.xml", "C",
                           fixed=True, xy_offset=(.10, 0.)),
    tabletop.ObjectBinding("s3_nameplate_1", f"{REL}/nameplate.xml", "F",
                           fixed=True, xy_offset=(.02, -.18)),
)
FIXED = {binding.name for binding in BINDINGS if binding.fixed}


class _S3Env(tabletop._make_env_class()):
    """Add only task-local fixed props to the common scene class."""

    def __init__(self, seed=0, *, camera_observations=False,
                 render_gpu_device_id=-1):
        tabletop.PILOTS.setdefault("S3-01", BINDINGS)
        super().__init__("S3-01", seed=seed,
                         camera_observations=camera_observations,
                         render_gpu_device_id=render_gpu_device_id)

    def _load_model(self):
        super()._load_model()
        for name in FIXED:
            body = self.objects_dict[name].root_body
            ET.SubElement(self.model.equality, "weld", name=f"{name}_fixture",
                          body1="world", body2=body, active="false")


class S3Scene:
    """Physical scene wrapper; online callers receive only `capture_agent`."""

    task_id = "S3-01"

    def __init__(self, seed=0, *, camera_observations=False,
                 render_gpu_device_id=-1):
        self.backend = _S3Env(seed=seed,
                              camera_observations=camera_observations,
                              render_gpu_device_id=render_gpu_device_id)
        try:
            self.raw = self.backend.reset()
            self._place_cookie_inside_box()
            self._fix_props()
            self.grasp_assist = GraspAssist(
                self.backend.sim.model._model, self.backend.sim.data._data,
                **self.backend.grasp_assist_kwargs)
            self.round_number = 1
            self.sequence = 0
            self.assist_events = []
            if camera_observations:
                self.raw = self.backend._get_observations(force_update=True)
        except BaseException:
            self.backend.close()
            raise

    @property
    def sim(self):
        return self.backend.sim

    def _fix_props(self):
        sim = self.backend.sim
        model, data = sim.model._model, sim.data._data
        for name in FIXED:
            obj = self.backend.objects_dict[name]
            body_id = model.body(obj.root_body).id
            weld_id = model.equality(f"{name}_fixture").id
            model.eq_data[weld_id, 3:6] = np.asarray(data.xpos[body_id])
            quat = np.empty(4, dtype=float)
            mujoco.mju_mat2Quat(quat, np.asarray(data.xmat[body_id]).reshape(9))
            model.eq_data[weld_id, 6:10] = quat
            data.eq_active[weld_id] = 1
        sim.forward()

    def _place_cookie_inside_box(self):
        # Separate slot jitters would otherwise put a 52 mm cookie into the
        # 90 mm box wall. This is an initial-layout relation, not an online
        # grasp shortcut or a hidden planner coordinate.
        sim = self.backend.sim
        cookie = self.backend.objects_dict["s3_cookie_1"]
        box = self.backend.objects_dict["s3_cookie_box_1"]
        c = np.asarray(sim.data.get_joint_qpos(cookie.joints[0]), float).copy()
        b = np.asarray(sim.data.get_joint_qpos(box.joints[0]), float).copy()
        c[:2] = b[:2]
        c[2] = b[2] + .038
        sim.data.set_joint_qpos(cookie.joints[0], c)
        sim.data.set_joint_qvel(cookie.joints[0], np.zeros(6))
        sim.forward()

    def start_round(self, round_number: int):
        if round_number != self.round_number + 1 or round_number not in (2, 3):
            raise ValueError("S3 rounds must advance 1, 2, 3 without reset")
        self.round_number = round_number

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("S3 action must be finite 7-vector")
        event = self.grasp_assist.before_control(float(action[6]),
                                                  source="common-v42-controller")
        if event.kind not in ("unchanged", "no_match"):
            self.assist_events.append({"time_s": float(self.sim.data.time),
                                       "kind": event.kind,
                                       "candidate_count": event.candidate_count})
        self.raw, reward, done, info = self.backend.step(action)
        self.sequence += 1
        return self.raw, reward, done, info

    def capture_agent(self):
        """Real public 640x480 RGB-D plus proprioception, with no task truth."""
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix, get_camera_intrinsic_matrix,
            get_real_depth_map,
        )
        from research.v4.accelerated.observation import public_observation

        frames = {}
        sim = self.sim
        for name in tabletop.PUBLIC_VIEWS:
            rgb = np.asarray(self.raw[name + "_image"])
            depth = np.squeeze(np.asarray(self.raw[name + "_depth"]))
            if rgb.shape != (480, 640, 3) or depth.shape != (480, 640):
                raise ValueError(f"S3 {name} was not rendered at 640x480")
            frames[name] = {
                "rgb": np.ascontiguousarray(rgb[::-1]),
                "depth_m": np.ascontiguousarray(
                    np.asarray(get_real_depth_map(sim, depth)).squeeze()[::-1]),
                "intrinsics": np.asarray(get_camera_intrinsic_matrix(
                    sim, name, 480, 640)),
                "extrinsics": np.asarray(get_camera_extrinsic_matrix(sim, name)),
                "frame_id": self.sequence,
            }
        robot = {key: self.raw[key] for key in (
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")}
        packet = public_observation(frames, {
            "eef_pos": robot["robot0_eef_pos"],
            "eef_quat": robot["robot0_eef_quat"],
            "gripper_qpos": robot["robot0_gripper_qpos"],
        }, frame_id=self.sequence)
        packet["robot_state"]["grasp_latched"] = bool(self.grasp_assist.held)
        return packet

    def close(self):
        self.backend.close()
