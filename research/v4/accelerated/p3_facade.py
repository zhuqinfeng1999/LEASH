"""Thin P3-01 v4.2 tabletop carrier for the existing RPent adapter.

This owns no task policy or evaluator truth.  The same ``step`` outlet is used
by B0 and Ours; Ours reaches it only after its commit gate passes.  Development
CPU tests may supply synthetic camera buffers, but a native episode must use
robosuite's real 640x480 offscreen observations.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist, inactive_weld_element
from research.v4.accelerated.observation import public_observation
from research.v4.accelerated.tabletop import (
    IMAGE_HEIGHT, IMAGE_WIDTH, PUBLIC_VIEWS, SPECTATOR_VIEW,
)
from research.v4.tasks.p3.scene import COASTER


COASTER_FIXTURE = "v42_p3_coaster_table_fixture"


def _create_p3_scene(*, seed: int, camera_observations: bool,
                     render_gpu_device_id: int):
    """Use the common tabletop class with one passive target fixture.

    The coaster is a destination surface, not a manipuland in P3. Its free
    joint remains in the model, but an inactive world weld is compiled before
    MuJoCo loads. This local subclass leaves shared tabletop tasks unchanged.
    """
    base = tabletop._make_env_class()

    class P3FixtureEnv(base):
        def _load_model(self):
            super()._load_model()
            coaster = self.objects_dict[COASTER]
            self.model.equality.append(inactive_weld_element(
                name=COASTER_FIXTURE, hand_body="world",
                object_body=coaster.root_body,
            ))

    return P3FixtureEnv(
        "P3-01", seed=seed, camera_observations=camera_observations,
        render_gpu_device_id=render_gpu_device_id,
    )


def _activate_coaster_fixture(backend):
    """Anchor the coaster at its seeded, visible reset pose, never teleport it."""
    model, data = backend.sim.model._model, backend.sim.data._data
    coaster = backend.objects_dict[COASTER]
    body_id = model.body(coaster.root_body).id
    weld_id = model.equality(COASTER_FIXTURE).id
    if data.eq_active[weld_id]:
        raise RuntimeError("P3 coaster fixture must begin inactive")
    model.eq_data[weld_id, 3:6] = data.xpos[body_id]
    model.eq_data[weld_id, 6:10] = data.xquat[body_id]
    data.eq_active[weld_id] = 1
    backend.sim.forward()


class P3TabletopFacade:
    """Expose only the scene contract consumed by ``V3LiberoEnvAdapter``.

    ``env`` is a compatibility shell: ``env.sim`` is the real MuJoCo simulator
    and ``env.env`` is the real robosuite environment.  The latter is private
    to the executor/evaluator; it never enters a public camera packet.
    """

    public_views = PUBLIC_VIEWS
    spectator_view = SPECTATOR_VIEW

    def __init__(self, seed: int = 0, *, camera_observations: bool = True,
                 render_gpu_device_id: int = -1):
        self.backend = _create_p3_scene(
            seed=int(seed), camera_observations=camera_observations,
            render_gpu_device_id=render_gpu_device_id,
        )
        try:
            self.backend.reset()
            _activate_coaster_fixture(self.backend)
            self.raw = self.backend._get_observations(force_update=True)
            self.env = SimpleNamespace(sim=self.backend.sim, env=self.backend)
            self.grasp_assist = GraspAssist(
                self.backend.sim.model._model, self.backend.sim.data._data,
                **self.backend.grasp_assist_kwargs,
            )
            self.assist_events: list[dict] = []
            self.sequence = 0  # physical controller steps
            self.capture_sequence = 0  # newly rendered public RGB-D packets
            self.size = (IMAGE_WIDTH, IMAGE_HEIGHT)
        except BaseException:
            self.backend.close()
            raise

    def step(self, action):
        """Only physical 7D outlet; a denied action must never call this."""
        command = np.asarray(action, dtype=np.float32)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError("P3 action must be a finite 7-vector")
        event = self.grasp_assist.before_control(
            float(command[6]), source="common-v42-controller",
        )
        if event.kind != "unchanged":
            self.assist_events.append({
                "sim_time_s": float(self.backend.sim.data.time),
                "kind": event.kind,
                "candidate_count": event.candidate_count,
            })
        self.raw, reward, done, info = self.backend.step(command)
        self.sequence += 1
        return self.raw, reward, done, info

    def capture_agent(self):
        """Return current, aligned 640x480 public RGB-D and proprioception.

        Refresh both native views and proprioception at the current simulator
        state without stepping physics. Each refreshed packet receives a new
        frame id, including after a denied motion proposal. This prevents a
        changed EEF pose from being paired with an older camera frame. Missing
        native frames fail closed; no spectator view is substituted.
        """
        self.raw = self.backend._get_observations(force_update=True)
        self.capture_sequence += 1
        frame_id = self.capture_sequence
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix, get_camera_intrinsic_matrix,
            get_real_depth_map,
        )

        frames = {}
        sim = self.backend.sim
        for name in PUBLIC_VIEWS:
            rgb_key, depth_key = name + "_image", name + "_depth"
            if rgb_key not in self.raw or depth_key not in self.raw:
                raise RuntimeError(f"native 640x480 camera buffer missing: {name}")
            rgb = np.asarray(self.raw[rgb_key])
            depth = np.asarray(self.raw[depth_key])
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3) or depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                raise ValueError(f"native public camera has wrong dimensions: {name}")
            frames[name] = {
                "rgb": np.ascontiguousarray(rgb[::-1]),
                "depth_m": np.ascontiguousarray(get_real_depth_map(sim, depth)[::-1].squeeze()),
                "intrinsics": np.asarray(get_camera_intrinsic_matrix(
                    sim, name, IMAGE_HEIGHT, IMAGE_WIDTH)),
                "extrinsics": np.asarray(get_camera_extrinsic_matrix(sim, name)),
                "frame_id": frame_id,
            }
        robot = {
            "eef_pos": self.raw["robot0_eef_pos"],
            "eef_quat": self.raw["robot0_eef_quat"],
            "gripper_qpos": self.raw["robot0_gripper_qpos"],
        }
        packet = public_observation(frames, robot, frame_id=frame_id)
        # Assisted grasp has one explicitly declared simulated proprioceptive
        # latch bit. It exposes no object identity, grasp zone, or evaluator
        # state, and the same packet is available to B0 and Ours.
        packet["robot_state"]["grasp_latched"] = bool(self.grasp_assist.held)
        # The old RPent adapter consumes these top-level names.  They are
        # aliases of the same public proprioception, not evaluator state.
        packet.update({key: packet["robot_state"][key] for key in robot})
        return packet

    def public_grasp_latched(self) -> bool:
        """Common simulated proprioceptive latch bit; no object identity."""
        return bool(self.grasp_assist.held)

    def spectator_rgb(self):
        """Offline-only human view, deliberately absent from capture_agent."""
        frame = self.backend.sim.render(
            width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
            camera_name=SPECTATOR_VIEW,
        )
        return np.ascontiguousarray(frame[::-1])

    def close(self):
        self.backend.close()
