"""R3-01 on the existing v4.2 Panda tabletop, without BDDL construction.

The temporary PILOTS registration is process-local and removed after the
common environment has constructed.  Only the microwave fixture and R3
metadata are task-specific; robot, controllers, cameras, slots and assisted
grasp all come from the shared tabletop implementation.
"""
from __future__ import annotations

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.candidates.r3_original_moka_integration.candidate import candidate_instance

ITEM = "moka_pot_1"
OVEN = "microwave_1"
TASK = "R3-01"

_BINDINGS = (
    tabletop.ObjectBinding(
        ITEM,
        "research/v4/candidates/r3_original_moka_integration/moka_libero_support.xml",
        "L", z_offset=.068, grasp_local=(0., 0., .070),
    ),
    tabletop.ObjectBinding(
        OVEN,
        "research/v4/candidates/r3_v42_tabletop_20260925/microwave_fullsize_180.xml",
        "R", fixed=True, xy_offset=(0., -.07),
    ),
)


class _R3FixtureEnv(tabletop._make_env_class()):
    def _load_model(self):
        super()._load_model()
        # Shared reset places each object through joints[0]. The articulated
        # microwave XML lists its hinge before its injected free joint.
        oven = self.objects_dict[OVEN]
        free = [name for name in oven._joints if name == "joint0"]
        if len(free) != 1:
            raise RuntimeError(f"microwave free joint missing: {oven._joints}")
        oven._joints = free + [name for name in oven._joints if name != free[0]]
        oven_body = oven.root_body
        ET.SubElement(self.model.equality, "weld", name="r3_microwave_fixture",
                      body1="world", body2=oven_body, active="false")


class R3TabletopFacade:
    instruction = candidate_instance().instruction
    public_views = tabletop.PUBLIC_VIEWS
    spectator_view = tabletop.SPECTATOR_VIEW
    item_name = ITEM
    oven_name = OVEN

    def __init__(self, seed=0, *, camera_observations=False,
                 render_gpu_device_id=-1):
        if TASK in tabletop.PILOTS:
            raise RuntimeError("R3 candidate conflicts with shared task registry")
        tabletop.PILOTS[TASK] = _BINDINGS
        try:
            self.backend = _R3FixtureEnv(
                TASK, seed=int(seed), camera_observations=camera_observations,
                render_gpu_device_id=render_gpu_device_id,
            )
        finally:
            del tabletop.PILOTS[TASK]
        try:
            self.raw = self.backend.reset()
            self.env = SimpleNamespace(sim=self.backend.sim, env=self.backend)
            self.instance = candidate_instance()
            self._anchor_and_open_microwave()
            self.grasp_assist = GraspAssist(
                self.backend.sim.model._model, self.backend.sim.data._data,
                **self.backend.grasp_assist_kwargs,
            )
            self.assist_events = []
            self.sequence = 0
            if camera_observations:
                self.raw = self.backend._get_observations(force_update=True)
        except BaseException:
            self.backend.close()
            raise

    def _anchor_and_open_microwave(self):
        sim = self.backend.sim
        model, data = sim.model._model, sim.data._data
        fixture = model.equality("r3_microwave_fixture").id
        body_id = model.body(self.backend.objects_dict[OVEN].root_body).id
        sim.forward()
        model.eq_data[fixture, 3:6] = np.asarray(data.xpos[body_id])
        q = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(q, np.asarray(data.xmat[body_id]).reshape(9))
        model.eq_data[fixture, 6:10] = q
        data.eq_active[fixture] = 1
        joints = [i for i in range(model.njnt)
                  if (model.joint(i).name or "").endswith("microjoint")]
        if len(joints) != 1:
            raise RuntimeError(f"R3 expected one microwave door hinge, got {joints}")
        jid = joints[0]
        data.qpos[model.jnt_qposadr[jid]] = -2.08
        data.qvel[model.jnt_dofadr[jid]] = 0.
        sim.forward()

    def step(self, action):
        command = np.asarray(action, dtype=np.float32)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError("R3 control must be a finite 7-vector")
        assist = self.grasp_assist.before_control(
            float(command[6]), source="common-v42-controller")
        if assist.kind not in {"unchanged", "no_match"}:
            self.assist_events.append({
                "sim_time_s": float(self.backend.sim.data.time),
                "kind": assist.kind,
                "candidate_count": assist.candidate_count,
            })
        self.raw, reward, done, info = self.backend.step(command)
        self.sequence += 1
        return self.raw, reward, done, info

    def capture_agent(self):
        """Only the two allowed native 640x480 RGB-D cameras plus proprioception."""
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix, get_camera_intrinsic_matrix,
            get_real_depth_map,
        )
        from research.v4.accelerated.observation import public_observation

        if not self.backend.use_camera_obs:
            raise RuntimeError("R3 public capture requires native camera observations")
        self.raw = self.backend._get_observations(force_update=True)
        frames = {}
        sim = self.backend.sim
        for name in tabletop.PUBLIC_VIEWS:
            rgb = np.asarray(self.raw[name + "_image"])
            depth = np.squeeze(np.asarray(self.raw[name + "_depth"]))
            if rgb.shape != (tabletop.IMAGE_HEIGHT, tabletop.IMAGE_WIDTH, 3):
                raise RuntimeError(f"R3 RGB was not truly 640x480: {name}")
            if depth.shape != (tabletop.IMAGE_HEIGHT, tabletop.IMAGE_WIDTH):
                raise RuntimeError(f"R3 depth was not truly 640x480: {name}")
            frames[name] = {
                "rgb": np.ascontiguousarray(rgb[::-1]),
                "depth_m": np.ascontiguousarray(get_real_depth_map(sim, depth)[::-1]),
                "intrinsics": np.asarray(get_camera_intrinsic_matrix(
                    sim, name, tabletop.IMAGE_HEIGHT, tabletop.IMAGE_WIDTH)),
                "extrinsics": np.asarray(get_camera_extrinsic_matrix(sim, name)),
                "frame_id": self.sequence,
            }
        robot = {
            "eef_pos": self.raw["robot0_eef_pos"],
            "eef_quat": self.raw["robot0_eef_quat"],
            "gripper_qpos": self.raw["robot0_gripper_qpos"],
        }
        packet = public_observation(frames, robot, frame_id=self.sequence)
        packet["robot_state"]["grasp_latched"] = bool(self.grasp_assist.held)
        packet.update({key: packet["robot_state"][key] for key in robot})
        return packet

    def spectator_rgb(self):
        """Human QA view; never included in online public capture."""
        frame = self.backend.sim.render(
            width=tabletop.IMAGE_WIDTH, height=tabletop.IMAGE_HEIGHT,
            camera_name=tabletop.SPECTATOR_VIEW,
        )
        return np.ascontiguousarray(frame[::-1])

    def close(self):
        self.backend.close()
