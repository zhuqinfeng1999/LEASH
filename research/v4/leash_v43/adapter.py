"""v4.3 hook at the real RPent step/chunk_step exit; old adapters stay immutable.

No object/evaluator/assist access is present in state, decode or registry paths.
The robot controller's static coordinate convention and scales are read once;
both raw gripper aperture and the public body quaternion retain their meaning.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from research.v4.accelerated.r2_adapter import R2V42Adapter
from .dispatch import CommitDispatcher, V43Intercepted
from .registry import (Command, CommitEventRegistry, GripperCalibration,
                       HoldingEstimator, Proprio, axis_rotation, rotation_matrix)


@dataclass(frozen=True)
class OSCControlSemantics:
    input_min: np.ndarray
    input_max: np.ndarray
    output_min: np.ndarray
    output_max: np.ndarray
    reference_rotation: np.ndarray
    source: str

    @classmethod
    def from_scene(cls, scene):
        # Read control configuration, not simulator object state or live goals.
        robot = scene.env.env.robots[0]
        controllers = getattr(robot, "part_controllers", {})
        controller = controllers.get("right")
        if controller is None:
            controller = getattr(robot, "controller", None)
        if (controller is None or getattr(controller, "input_type", None) != "delta"
                or getattr(controller, "_goal_update_mode", None) != "achieved"
                or getattr(controller, "impedance_mode", None) != "fixed"
                or getattr(controller, "interpolator_ori", None) is not None
                or np.any(getattr(controller, "orientation_limits", [0]))):
            raise ValueError("unsupported OSC semantics; cannot guess a pose path")
        reference = getattr(controller, "input_ref_frame", None)
        if reference == "base":
            rotation = rotation_matrix(controller.origin_ori)
        elif reference == "world":
            rotation = np.eye(3)
        else:
            raise ValueError("unknown control reference frame")
        arrays = [np.broadcast_to(np.asarray(getattr(controller, k), float), (6,)).copy()
                  for k in ("input_min", "input_max", "output_min", "output_max")]
        if (not all(np.isfinite(a).all() for a in arrays)
                or not np.all(arrays[1] > arrays[0])):
            raise ValueError("OSC action scaling unavailable")
        return cls(*arrays, rotation,
                   "installed OSC_POSE/achieved/base-or-world/delta; Panda sign gripper")

    def decode(self, payload, p: Proprio) -> Command:
        a = np.asarray(payload, dtype=np.float32)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError("decoded RPent control must be a finite 7-vector")
        # Same affine clipping as Controller.scale_action, after VLA unnormalization.
        v = np.clip(a[:6], self.input_min, self.input_max)
        scale = (self.output_max - self.output_min) / (self.input_max - self.input_min)
        scaled = (v - (self.input_max + self.input_min) * .5) * scale + (self.output_max + self.output_min) * .5
        translation = self.reference_rotation @ scaled[:3]
        delta = self.reference_rotation @ scaled[3:6]
        # Preserve a missing pose until the registry determines relevance.
        # Empty grippers and own reliably non-container bindings do not need
        # that pose for POUR_ONSET; related rotations still fail UNRESOLVED.
        target_r = None if p.eef_rotation is None else axis_rotation(delta) @ p.eef_rotation
        target_xyz = None if p.eef_xyz is None else p.eef_xyz + translation
        return Command(a.copy(), "OPEN" if a[6] < 0 else "CLOSE" if a[6] > 0 else "HOLD",
                       translation, delta, target_xyz, target_r)


class V43PublicEnvAdapter(R2V42Adapter):
    def __init__(self, scene, *, instruction, method, reviewer, calibration,
                 policy, allow_draft_for_script=False, capture=None,
                 grasp_assumption="unverified", review_timeout_s=15.,
                 position_tolerance_m=.008, orientation_tolerance_rad=.06,
                 semantics=None, on_sim_step=None):
        super().__init__(scene, instruction=instruction, on_sim_step=on_sim_step)
        from .method_inputs import METHOD_PROFILES
        self.profile = METHOD_PROFILES[method]
        self.policy = policy
        self.allow_draft_for_script = allow_draft_for_script
        self.text_history = ()
        self.history = ()
        self._capture = capture or scene.capture_agent
        self._reviewer = reviewer
        self.semantics = semantics or OSCControlSemantics.from_scene(scene)
        self.estimator = HoldingEstimator(calibration, grasp_assumption=grasp_assumption)
        initial = getattr(scene, 'public_initial_occupancy', lambda: None)()
        if initial is not None:
            import os
            from pathlib import Path
            self.estimator.initialize_empty(initial, self._proprio(0),
                episode=Path(os.environ['SAFEVLA_RUN_OUTPUT']).name)
        self.registry = CommitEventRegistry(self.estimator, position_tolerance_m=position_tolerance_m,
                                             direction_tolerance_rad=orientation_tolerance_rad, method=method)
        self.dispatcher = CommitDispatcher(registry=self.registry, decode=self.semantics.decode,
                                           read_proprio=self._proprio, send=self._consume,
                                           capture=self._commit_capture, review=self._review,
                                           project=self._project, method=method,
                                           enabled=self.profile.registry_enabled,
                                           review_timeout_s=review_timeout_s)

    def _commit_capture(self):
        # B2 has no visual input. A missing unrelated camera must not block it.
        if not self.profile.reviewer_views:
            return {"frame_id": f"public-state-{self.dispatcher.sequence}", "cameras": {}}
        return self._capture()

    def _proprio(self, sequence):
        raw = self.scene.raw
        gap = None
        try:
            q = np.asarray(raw["robot0_gripper_qpos"], float)
            if q.shape == (2,) and np.isfinite(q).all():
                gap = float(np.abs(q).sum())
        except (KeyError, TypeError, ValueError):
            pass
        r = xyz = None
        try:
            q = np.asarray(raw["robot0_eef_quat"], float)
            if q.shape == (4,) and np.isfinite(q).all() and .9 < np.linalg.norm(q) < 1.1:
                r = Rotation.from_quat(q).as_matrix()  # Existing public BODY-world xyzw.
        except (KeyError, TypeError, ValueError):
            pass
        try:
            a = np.asarray(raw["robot0_eef_pos"], float)
            if a.shape == (3,) and np.isfinite(a).all():
                xyz = a.copy()
        except (KeyError, TypeError, ValueError):
            pass
        return Proprio(gap, xyz, r, sequence)

    def _consume(self, action):
        # Historical V3/R2 behavior is reused after admission, including private
        # evaluator callbacks. Its old early last-gripper write is not used by
        # the estimator: executed() is called only after this send completes.
        return R2V42Adapter.step(self, action)

    def _project(self, *, events, command, proprio, classification, evidence, context, lease):
        from .method_inputs import PublicFrame, ReviewInputRequest, project_review_input
        required = self.profile.reviewer_views
        if (not isinstance(evidence, dict) or "frame_id" not in evidence
                or not set(required).issubset(evidence.get("cameras", {}))):
            raise ValueError("required public camera acquisition unavailable")
        version = f"{context.episode}:{context.round_id}:{context.primitive}:{self.dispatcher.sequence + 1}:{evidence['frame_id']}"
        frames = {}
        for name in required:
            c = evidence["cameras"][name]
            if c.get("frame_id") != evidence["frame_id"]:
                raise ValueError("mixed acquisition packets")
            depth = np.asarray(c["depth_m"])
            k, t = np.asarray(c["intrinsics"], float), np.asarray(c["extrinsics"], float)
            if (depth.shape != (480, 640) or not np.issubdtype(depth.dtype, np.floating)
                    or not np.isfinite(depth).all() or np.any(depth <= 0.)
                    or k.shape != (3, 3) or t.shape != (4, 4)
                    or not np.isfinite(k).all() or not np.isfinite(t).all()
                    or abs(np.linalg.det(k)) < 1e-12
                    or not np.allclose(t[3], [0., 0., 0., 1.], atol=1e-8)):
                raise ValueError("invalid metric depth or public calibration")
            rotation_matrix(t[:3, :3])
            frames[name] = PublicFrame(view=name, rgb=c["rgb"], depth=c["depth_m"],
                                       K=c["intrinsics"], T=c["extrinsics"], evidence_version=version)
        robot = {"eef_pos": None if proprio.eef_xyz is None else proprio.eef_xyz.tolist(),
                 "eef_rotation": None if proprio.eef_rotation is None else proprio.eef_rotation.tolist(),
                 "gripper_aperture_m": proprio.aperture_m,
                 "holding_estimate": self.estimator.kind.value}
        if self.estimator.initial_occupancy is not None:
            robot['initial_occupancy'] = self.estimator.initial_occupancy
        event_context = {"trusted_instruction": self.instruction,
                         "proposed_action": {"gripper": command.gripper,
                                             "translation_m": command.translation.tolist(),
                                             "rotation_world_axis_angle_rad": command.rotation_delta.tolist()},
                         "event_description": list(events)}
        history = self.history
        from .visual_holding import enabled
        if enabled() and self.profile.method.value == 'Ours':
            observations = self.scene.public_grasp_observations()
            observations.observe(evidence)
            import json
            history = (*history,*observations.history())
            event_context['holding_observations'] = json.dumps(observations.metadata(),allow_nan=False)
            robot['holding_proprio'] = self.estimator.holding_proprio.value
            robot['holding_evidence'] = self.estimator.visual_record
        projected = project_review_input(ReviewInputRequest(
            method=self.profile.method, gate=3, current_workspace=frames.get("agentview"),
            current_wrist=frames.get("robot0_eye_in_hand"), history=history,
            text_history=self.text_history, event_context=event_context, robot_state=robot,
            lease=lease if lease is not None and lease.valid else None,
            evidence_version=version, events=events),
            self.policy, allow_draft_for_mock=self.allow_draft_for_script)
        projected["deadline_monotonic_ns"] = time.monotonic_ns() + int(self.dispatcher.review_timeout_s * 1e9)
        return projected

    def _review(self, request):
        from .method_inputs import validate_review_response
        raw = self._reviewer(request)
        # Script callbacks use the dispatch vocabulary; the strict method
        # validator uses the three-gate result vocabulary.
        if isinstance(raw, dict) and "decision" in raw:
            raw = {**raw, "result": raw["decision"]}
            del raw["decision"]
        checked = validate_review_response(self.profile.method, 3, raw,
                                           expected_events=request["events"],
                                           evidence_version=request["evidence_version"])
        return {**checked, "events": tuple(request["events"]), "decision": checked["result"]}

    @contextmanager
    def primitive_scope(self, name, *, parameters=None, completion_controls,
                        episode="local-script", round_id=0, binding_version="unknown", lease=None):
        with self.dispatcher.primitive(name=name, episode=episode, round_id=round_id,
                                       binding_version=binding_version, completion_controls=completion_controls,
                                       lease=lease):
            yield self

    def step(self, action):
        return self.dispatcher.step(action)

    def chunk_step(self, actions, *, return_all_frames=False):
        a = np.asarray(actions, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 7 or len(a) == 0:
            raise ValueError("RPent action chunk must be nonempty [N,7]")
        values = self.dispatcher.run_chunk(a)
        observations = [v[0] for v in values]
        n = len(values)
        return (observations if return_all_frames else observations[-1],
                np.zeros(n, np.float32), np.zeros(n, bool), np.zeros(n, bool), {})

    def invalidate_binding(self):
        self.dispatcher.lease = None
        self.registry.invalidate()


def method_env_adapter(scene, *, method, instruction, **commit_options):
    """Explicit opt-in factory; B0/B1 retain the original, registry-free path."""
    from .method_inputs import METHOD_PROFILES
    if not METHOD_PROFILES[method].registry_enabled:
        return R2V42Adapter(scene, instruction=instruction,
                            on_sim_step=commit_options.get("on_sim_step"))
    return V43PublicEnvAdapter(scene, method=method, instruction=instruction, **commit_options)
