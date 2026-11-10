"""I1 v4.2 common tabletop facade for RPent; public capture exposes no truth."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np

from research.v4.accelerated.observation import public_observation
from research.v4.accelerated.tabletop import IMAGE_HEIGHT, IMAGE_WIDTH, PUBLIC_VIEWS, SPECTATOR_VIEW
from .candidate import INSTRUCTION, create


class I1Facade:
    instruction = INSTRUCTION
    public_views = PUBLIC_VIEWS
    spectator_view = SPECTATOR_VIEW

    def __init__(self, seed=0, *, camera_observations=False, render_gpu_device_id=-1):
        self.backend, self.grasp_assist = create(
            int(seed), camera_observations=camera_observations,
            render_gpu_device_id=render_gpu_device_id,
        )
        self.env = SimpleNamespace(sim=self.backend.sim, env=self.backend)
        self.raw = self.backend._get_observations(force_update=True)
        self.sequence = 0
        self.assist_events = []

    def step(self, action):
        command = np.asarray(action, dtype=np.float32)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError("I1 control must be a finite 7-vector")
        event = self.grasp_assist.before_control(float(command[6]), source="common-v42-controller")
        if event.kind not in {"unchanged", "no_match"}:
            self.assist_events.append({
                "sim_time_s": float(self.backend.sim.data.time),
                "kind": event.kind,
                "candidate_count": event.candidate_count,
            })
        self.raw, reward, done, info = self.backend.step(command)
        self.sequence += 1
        return self.raw, reward, done, info

    def public_grasp_latched(self) -> bool:
        """One declared Boolean simulated latch, equal for B0 and Ours."""
        return bool(self.grasp_assist.held)

    def capture_agent(self):
        """Only current 640x480 agentview/wrist RGB-D, calibration, proprioception."""
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix, get_camera_intrinsic_matrix,
            get_real_depth_map,
        )
        sim = self.backend.sim
        frames = {}
        for name in PUBLIC_VIEWS:
            rgb = np.asarray(self.raw[name + "_image"])
            depth = np.asarray(self.raw[name + "_depth"]).squeeze()
            if rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3) or depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                raise ValueError(f"I1 public RGB-D is not native 640x480: {name}")
            frames[name] = {
                "rgb": np.ascontiguousarray(rgb[::-1]),
                "depth_m": np.ascontiguousarray(np.asarray(get_real_depth_map(sim, depth)).squeeze()[::-1]),
                "intrinsics": np.asarray(get_camera_intrinsic_matrix(sim, name, IMAGE_HEIGHT, IMAGE_WIDTH)),
                "extrinsics": np.asarray(get_camera_extrinsic_matrix(sim, name)),
                "frame_id": self.sequence,
            }
        robot = {k: self.raw[k] for k in
                 ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")}
        robot = {k.replace("robot0_", ""): v for k, v in robot.items()}
        packet = public_observation(frames, robot, frame_id=self.sequence)
        packet["robot_state"]["grasp_latched"] = self.public_grasp_latched()
        packet.update({key: packet["robot_state"][key] for key in robot})
        return packet

    def spectator_rgb(self):
        frame = self.backend.sim.render(width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
                                        camera_name=SPECTATOR_VIEW)
        return np.ascontiguousarray(frame[::-1])

    def close(self):
        self.backend.close()
