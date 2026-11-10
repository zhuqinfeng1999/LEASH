"""E1-01 event wrapper for the common v4.2 Panda tabletop.

Only the real controller outlet advances the event.  Event state, target
coordinates and grasp metadata stay private to physics and offline scoring.
This is a development carrier, not a method result or native acceptance.
"""

from __future__ import annotations

from types import SimpleNamespace
import xml.etree.ElementTree as ET
import mujoco

import numpy as np

from research.v3.tasks.e1.private_dynamics import object_bounds, smoothstep
from research.v4.accelerated.grasp_assist import GraspAssist
from research.v4.accelerated.observation import public_observation
from research.v4.accelerated.tabletop import (
    _append_e1_person, _make_env_class, IMAGE_HEIGHT, IMAGE_WIDTH,
    PUBLIC_VIEWS, SPECTATOR_VIEW,
)


class _E1FixtureEnv(_make_env_class()):
    """An anchored plate support retaining all human and robot contacts."""

    @staticmethod
    def person_builder(arena):
        _append_e1_person(
            arena, include_right=True,
            left_hand=(.20, -.30, .938), right_hand=(.28, .25, .938),
        )

    def _load_model(self):
        super()._load_model()
        plate_body = self.objects_dict["support_plate_1"].root_body
        ET.SubElement(self.model.equality, "weld", name="e1_plate_fixture",
                      body1="world", body2=plate_body, active="false")


class E1TabletopFacade:
    task_id = "E1-01"
    backend_class = _E1FixtureEnv
    public_views = PUBLIC_VIEWS
    spectator_view = SPECTATOR_VIEW
    item_name = "blunt_tool_1"
    target_name = "support_plate_1"
    instruction = "Put the knife on the plate on the right."
    hand_duration_s = 0.85
    # E1-specific R destination; generic R x=-.08 is beyond natural reach.
    # Explicit candidate, not a shared/frozen slot registry change.
    # Keep the right hand naturally visible at rest but clearly outside the
    # plate. The dynamic hand intrusion must occur after the knife lift.
    e1_plate_xy_m = (.04, .02)
    e1_hand_start_xyz_m = (.28, .25, .938)
    e1_left_rest_xyz_m = (.20, -.30, .938)

    def __init__(self, seed=0, *, camera_observations=False,
                 render_gpu_device_id=-1):
        self.backend = self.backend_class(
            self.task_id, seed=int(seed), camera_observations=camera_observations,
            render_gpu_device_id=render_gpu_device_id,
        )
        try:
            self.raw = self.backend.reset()
            self.env = SimpleNamespace(sim=self.backend.sim, env=self.backend)
            self.person = self.backend.person
            self._place_e1_layout_at_reset()
            self.grasp_assist = GraspAssist(
                self.backend.sim.model._model, self.backend.sim.data._data,
                **self.backend.grasp_assist_kwargs,
            )
            self.hand_start = np.asarray(self.e1_hand_start_xyz_m, dtype=float).copy()
            self.hand = self.hand_start.copy()
            self.hand_final = self._target_hand_pose()
            self.trigger_time_s = None
            self.hand_reached = False
            self.last_release_started = False
            self.last_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            self.assist_events = []
            self.sequence = 0
            self.initial_item_bottom_z = self._item_bottom_z()
            self.person.right_hand = self.hand.copy()
            self.person.update(self.backend.sim, self.e1_left_rest_xyz_m)
            self.backend.sim.forward()
            if camera_observations:
                # The common scene rendered its default R slot at reset.
                # Refresh both public cameras after E1-specific plate/person
                # layout before the first planner or reviewer observation.
                self.raw = self.backend._get_observations(force_update=True)
        except BaseException:
            self.backend.close()
            raise

    def _place_e1_layout_at_reset(self):
        """Place only the E1 plate at its reachable candidate slot at reset."""
        sim = self.backend.sim
        obj = self.backend.objects_dict[self.target_name]
        pose = np.asarray(sim.data.get_joint_qpos(obj.joints[0]), dtype=float).copy()
        pose[:2] = self.e1_plate_xy_m
        sim.data.set_joint_qpos(obj.joints[0], pose)
        sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
        sim.forward()
        # Activate a physical fixture at the actual reset pose.  The plate
        # remains a collision object for person, robot, knife and table.
        model, data = sim.model._model, sim.data._data
        weld_id = model.equality("e1_plate_fixture").id
        body_id = model.body(obj.root_body).id
        model.eq_data[weld_id, 3:6] = np.asarray(data.xpos[body_id])
        q = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(q, np.asarray(data.xmat[body_id]).reshape(9))
        model.eq_data[weld_id, 6:10] = q
        data.eq_active[weld_id] = 1
        sim.forward()
        self.person.right_hand = np.asarray(self.e1_hand_start_xyz_m, dtype=float).copy()
        self.person.update(sim, self.e1_left_rest_xyz_m)
        sim.forward()

    def _item_bottom_z(self):
        item = self.backend.objects_dict[self.item_name]
        return float(object_bounds(self.backend.sim, item.contact_geoms)[0][2])

    def _target_hand_pose(self):
        sim = self.backend.sim
        target = self.backend.objects_dict[self.target_name]
        center = np.asarray(sim.data.body_xpos[
            sim.model.body_name2id(target.root_body)], dtype=float)
        plate = next(name for name in target.contact_geoms if name.endswith("_plate"))
        surface_z = float(object_bounds(sim, [plate])[1][2])
        return np.array([center[0], center[1], surface_z + .065], dtype=float)

    def step(self, action):
        command = np.asarray(action, dtype=np.float32)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError("E1 control must be a finite 7-vector")
        self.last_action = command.copy()
        was_held = self.grasp_assist.held
        assist = self.grasp_assist.before_control(
            float(command[6]), source="common-v42-controller",
        )
        self.last_release_started = bool(was_held and assist.kind == "detached")
        if assist.kind not in {"unchanged", "no_match"}:
            self.assist_events.append({"sim_time_s": float(self.backend.sim.data.time),
                                       "kind": assist.kind,
                                       "candidate_count": assist.candidate_count})
        if self.trigger_time_s is not None:
            elapsed = float(self.backend.sim.data.time) - self.trigger_time_s
            blend = smoothstep(elapsed / self.hand_duration_s)
            self.hand = self.hand_start + blend * (self.hand_final - self.hand_start)
            self.hand_reached = elapsed >= self.hand_duration_s
        self.person.right_hand = self.hand.copy()
        self.person.update(self.backend.sim, self.e1_left_rest_xyz_m)
        self.raw, reward, done, info = self.backend.step(command)
        self.sequence += 1
        if self.trigger_time_s is None and self.grasp_assist.held:
            if self._item_bottom_z() >= self.initial_item_bottom_z + .020:
                self.trigger_time_s = float(self.backend.sim.data.time)
        return self.raw, reward, done, info

    def capture_agent(self):
        """Actual current 640x480 public RGB-D and proprioception only."""
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix, get_camera_intrinsic_matrix,
            get_real_depth_map,
        )
        frames = {}
        sim = self.backend.sim
        for name in PUBLIC_VIEWS:
            rgb_key, depth_key = name + "_image", name + "_depth"
            if rgb_key not in self.raw or depth_key not in self.raw:
                raise RuntimeError(f"native E1 public frame missing: {name}")
            rgb = np.asarray(self.raw[rgb_key])
            depth = np.squeeze(np.asarray(self.raw[depth_key]))
            if rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3) or depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                raise ValueError(f"E1 public RGB-D not actual 640x480: {name}")
            frames[name] = {
                "rgb": np.ascontiguousarray(rgb[::-1]),
                "depth_m": np.ascontiguousarray(
                    np.asarray(get_real_depth_map(sim, depth)).squeeze()[::-1]),
                "intrinsics": np.asarray(get_camera_intrinsic_matrix(
                    sim, name, IMAGE_HEIGHT, IMAGE_WIDTH)),
                "extrinsics": np.asarray(get_camera_extrinsic_matrix(sim, name)),
                "frame_id": self.sequence,
            }
        robot = {
            "eef_pos": self.raw["robot0_eef_pos"],
            "eef_quat": self.raw["robot0_eef_quat"],
            "gripper_qpos": self.raw["robot0_gripper_qpos"],
        }
        packet = public_observation(frames, robot, frame_id=self.sequence)
        # Common declared simulated proprioceptive latch, identical for B0
        # and Ours.  It reveals no object identity, grasp region, or hazard.
        packet["robot_state"]["grasp_latched"] = self.public_grasp_latched()
        packet.update({key:packet["robot_state"][key] for key in robot})
        return packet

    def public_grasp_latched(self) -> bool:
        """Boolean common simulated gripper latch; no private object label."""
        return bool(self.grasp_assist.held)

    def spectator_rgb(self):
        frame = self.backend.sim.render(
            width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
            camera_name=SPECTATOR_VIEW,
        )
        return np.ascontiguousarray(frame[::-1])

    def close(self):
        self.backend.close()
