"""Unmodified RPent primitive implementations over a measured hardware env.

The original seven-vector is decoded here, never passed to the motor driver.
The Dobot/Jodell devices are separate: a changed jaw target completes first,
then the Cartesian command. Every actual send enters the same dispatcher.
This ordering is a recorded hardware difference, not simultaneous OSC control.
"""

from copy import deepcopy
import inspect
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from robots.libero.tools import (
    LiberoPrimitives,
    TOOLS_SPEC,
    back_project,
    view_camera_meta,
)
from rpent.tools.common import TOOLS_SPEC as COMMON_SPEC
from research.v3.bridge.b0 import (
    BUDGETED_ACTIONS,
    FINISH_SPEC,
    HOLD_SPEC,
    _agent_result,
    _tool_result,
)
from research.v4.leash_v43.tool_boundary import OriginalControlBounds
from .artifacts import HardwareArtifacts
from .official_primitives import (
    OFFICIAL_ACTIONS,
    OfficialEnv,
    OfficialModel,
    load_official_primitives,
)


class CapabilityUnavailable(RuntimeError):
    pass


class TargetRejected(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(receipt.get("error", "Target preflight rejected without send"))


class MissingPolicy:
    def predict_action_batch(self, *_args, **_kwargs):
        raise CapabilityUnavailable(
            "Pi0.5 model service/checkpoint is unavailable; no VLA action was generated or sent"
        )


class MissingSegmentation:
    def segment(self, *_args, **_kwargs):
        raise ValueError(
            "SAM3 service/checkpoint is unavailable; no substituted detector"
        )


class HardwarePrimitiveEnv:
    return_all_frames = False
    terminated = False
    truncated = False

    def __init__(self, adapter):
        self.adapter = adapter
        self.last_obs = None
        self.jaw_target = None
        self.expansions = []
        self.action_chunks = []

    def observe(self):
        p = self.adapter._proprio(self.adapter.dispatcher.sequence)
        if p.eef_xyz is None or p.eef_rotation is None or p.aperture_m is None:
            raise CapabilityUnavailable(
                "Measured grip_site position, orientation and aperture are required"
            )
        packet = self.adapter._capture()
        self.last_obs = dict(
            main_images=packet["cameras"]["agentview"]["rgb"],
            wrist_images=packet["cameras"]["robot0_eye_in_hand"]["rgb"],
            extra_view_images=None,
            states=np.r_[
                p.eef_xyz,
                Rotation.from_matrix(p.eef_rotation).as_rotvec(),
                p.aperture_m / 2,
                -p.aperture_m / 2,
            ].astype(np.float32),
            task_descriptions=self.adapter.instruction,
        )
        return self.last_obs

    def raw_obs(self):
        p = self.adapter._proprio(self.adapter.dispatcher.sequence)
        return dict(
            robot0_eef_pos=p.eef_xyz,
            robot0_eef_quat=Rotation.from_matrix(p.eef_rotation).as_quat(),
        )

    def reset(self):
        # An episode starts from the actual scene; never resets physical pose.
        return self.observe(), {"physical_scene_reset": False}

    def preflight(self, motion):
        driver = self.adapter.driver
        if not np.any(motion[:6]):
            return
        if hasattr(driver, "preflight_si"):
            receipt = driver.preflight_si(motion)
        elif np.any(motion[3:6]):
            raise CapabilityUnavailable(
                "Rotation transport has not been qualified on this hardware"
            )
        elif hasattr(driver, "preflight_translation"):
            receipt = driver.preflight_translation(motion[:3].tolist())
        else:
            return  # Explicit CPU test transport, not a physical qualification.
        if receipt.get("accepted") is not True:
            if receipt.get("recovery_allowed") is True:
                raise TargetRejected(receipt)
            raise RuntimeError(
                receipt.get("error", "Hardware preflight transport failed")
            )

    def step(self, action):
        a = np.asarray(action, float)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise CapabilityUnavailable("Policy control must be a finite seven-vector")
        # Explicit reference OSC scale, also used by the original analytic
        # primitive loops. It is NOT the driver SI decoder.
        physical = np.r_[
            np.clip(a[:3], -1, 1) * 0.05, np.clip(a[3:6], -1, 1) * 0.5, 0.0
        ]
        result = self.step_physical(np.r_[physical[:6], a[6]], reference_action=a)
        return (
            result["observation"],
            0.0,
            False,
            False,
            {"hardware_action_plan": result["hardware_action_plan"]},
        )

    def step_physical(self, physical, reference_action=None):
        physical = np.asarray(physical, float).copy()
        if physical.shape != (7,) or not np.isfinite(physical).all():
            raise CapabilityUnavailable("Finite SI control required")
        self.preflight(physical)  # Reject the pose before changing either device.
        sign = float(np.sign(physical[6]))
        physical[6] = 0
        p = self.adapter._proprio(self.adapter.dispatcher.sequence)
        repeat_complete = (
            sign == self.jaw_target and p.gripper_command_motion_complete is True
        )
        plan = dict(
            reference_action=None
            if reference_action is None
            else reference_action.tolist(),
            physical_motion_si=physical.tolist(),
            gripper_sign=sign,
            order="jaw_target_then_pose",
            repeat_terminal_target_omitted=repeat_complete,
            executed_start=self.adapter.dispatcher.sequence,
        )
        self.expansions.append(plan)
        if sign and not repeat_complete:
            jaw = np.zeros(7)
            jaw[6] = sign
            self.adapter.step(jaw)
            self.jaw_target = sign
        if np.any(physical[:6]):
            self.adapter.step(physical)
        changed = self.adapter.dispatcher.sequence != plan["executed_start"]
        obs = self.observe() if changed or self.last_obs is None else self.last_obs
        plan["executed_end"] = self.adapter.dispatcher.sequence
        return dict(
            observation=obs, hardware_action_plan=plan, acknowledged_complete=True
        )

    def chunk_step(self, actions, return_all_frames=False):
        actions = np.asarray(actions, float)
        if actions.shape != (5, 7) or not np.isfinite(actions).all():
            raise CapabilityUnavailable(
                "Reference Pi0.5 must return exactly five finite seven-dimensional actions per chunk"
            )
        frames = []
        record = dict(
            actions_reference_normalized=actions.tolist(),
            executed_start=self.adapter.dispatcher.sequence,
            completed_logical_actions=0,
            complete=False,
        )
        self.action_chunks.append(record)
        try:
            for action in actions:
                obs, *_ = self.step(action)
                frames.append(obs)
                record["completed_logical_actions"] += 1
            record["complete"] = True
        except BaseException as exc:
            record["error"] = repr(exc)
            raise
        finally:
            record["executed_end"] = self.adapter.dispatcher.sequence
        return (frames if return_all_frames else frames[-1]), 0.0, False, False, {}


def completion_controls(name, args):
    # Bounds describe actual outgoing controls, including the initial jaw
    # command. G3 retains the original finite geometric permit constraints.
    if name in {"pi0_pick", "pi0_doubled"}:
        return int(args.get("max_chunks", 24 if name == "pi0_pick" else 20)) * 10
    if name in {"rotate_wrist", "rotate_pitch"}:
        return int(args.get("max_steps", 40)) * 2
    if name == "move_pose":
        return int(args.get("max_steps", 150)) * 2
    if name == "move_to":
        return int(args.get("max_steps", 80)) * 2
    return int(args.get("max_steps", args.get("steps", 20 if name == "release" else 5)))


class NativePortTools:
    """Exact original input schemas and bound LiberoPrimitives methods.

    Observations/file paths are hardware adaptations. Missing model/segment
    services return an explicit capability failure; never a scripted fallback.
    """

    def __init__(self, adapter, legacy, *, model=None):
        self.adapter, self.legacy = adapter, legacy
        self.env = HardwarePrimitiveEnv(adapter)
        self._primitives = LiberoPrimitives(
            self.env,
            model or MissingPolicy(),
            getattr(adapter.driver, "sam3_client", None) or MissingSegmentation(),
            self.check_cancelled,
        )
        self._official = load_official_primitives()(
            env=OfficialEnv(self.env),
            model=OfficialModel(self._primitives.model),
            task_description=adapter.instruction,
            check_cancelled=self.check_cancelled,
        )
        self.calls = []
        self.round_calls = 0
        self.round_finished = False
        self.finish_result = None
        root = Path(adapter.evidence_builder.output).resolve()
        self.output_dir = (
            root.parent / "rpent" if root.name == "gates" else root / "rpent"
        )
        self.memory_dir = legacy.memory_dir
        self.observations = []
        self.artifacts = HardwareArtifacts(
            adapter.driver, self.output_dir / "public_artifacts"
        )
        self.bounds = OriginalControlBounds(
            self._primitives,
            remaining_controls=lambda: 10**9,
            control_timestep=0.05,
            vla_chunk_controls=5,
        )

    def completion_controls(self, name, args):
        if name in OFFICIAL_ACTIONS:
            if name == "vla_grasp":
                n = args.get("max_chunks", 4)
                if type(n) is not int or not 1 <= n <= 20:
                    raise ValueError("Official vla_grasp max_chunks must be 1..20")
                return n * 10
            return 1
        return self.bounds.primitive_controls(name, args) * 2

    def check_cancelled(self):
        if self.adapter.closed or self.adapter.dispatcher._abort_requested.is_set():
            raise RuntimeError("Hardware primitive owner cancelled")
        self.adapter.driver.assert_stationary()

    def begin_round(self):
        self.round_calls = 0
        self.round_finished = False
        self.finish_result = None

    def get_tools_spec(self):
        specs = deepcopy(TOOLS_SPEC) + deepcopy(COMMON_SPEC)
        specs = [deepcopy(FINISH_SPEC) if s["name"] == "finish" else s for s in specs]
        specs.append(deepcopy(HOLD_SPEC))
        official = json.loads(
            (Path(__file__).parent / "official_tool_schemas.json").read_text(
                encoding="utf-8"
            )
        )
        for spec in official:
            spec["description"] = spec["description"].replace("Franka", "Dobot/Jodell")
        specs = official + specs
        for s in specs:
            if s["name"] in BUDGETED_ACTIONS:
                s["description"] += (
                    " Hardware port: logical RPent controls are decoded to Dobot SI; changed Jodell target completes before pose motion. No simulator termination oracle."
                )
            if s["name"] == "back_project":
                s["description"] = (
                    "Project a pixel or region from the selected recorded hardware RGBD. high and low both refer to that camera's native image dimensions, reported by view_camera_meta; unavailable camera depth/extrinsics returns an explicit capability error."
                )
        return specs

    def observation_result(self, name, result):
        value = self.legacy.observation_result(name, result).result
        value["step"] = len(self.observations)
        if hasattr(self.adapter.driver, "record_native_observation"):
            self.artifacts.append(
                self.adapter.driver.record_native_observation(value["step"])
            )
        self.observations.append(value)
        return _tool_result(name, value)

    def execute_tool(self, name, args):
        if name == "finish":
            result = self.legacy.execute_tool(name, args)
            self.round_finished, self.finish_result = (
                self.legacy.round_finished,
                self.legacy.finish_result,
            )
            return result
        if name in {"observe", "view_env_state"}:
            step = args.get("step", -1)
            if step != -1:
                if type(step) is not int or not 0 <= step < len(self.observations):
                    return _tool_result(
                        name,
                        dict(
                            error="Recorded hardware step is unavailable",
                            candidate_dispatched=False,
                        ),
                    )
                return _tool_result(name, self.observations[step])
            if hasattr(self.adapter.driver, 'invalidate_observation'):
                self.adapter.driver.invalidate_observation()
            result = self.observation_result(name, {})
            return result
        if name == "view_camera_meta":
            return _tool_result(name, view_camera_meta(state=self.artifacts, **args))
        if name == "back_project":
            result = back_project(state=self.artifacts, **args)
            result = self.artifacts.annotate_projection(result, **args)
            return _tool_result(name, result)
        if name == "segment":
            from .segmentation import segment_recorded
            return _tool_result(
                name, segment_recorded(self._primitives._sam3_client,
                                       self.artifacts, **args)
            )
        if name in {"read_text_file", "write_text_file", "list_dir"}:
            return self.file_tool(name, args)
        if name not in BUDGETED_ACTIONS | OFFICIAL_ACTIONS:
            return _tool_result(
                name,
                dict(error="Unknown original RPent tool", candidate_dispatched=False),
            )
        if self.round_calls >= 60 or len(self.calls) >= 180:
            return _tool_result(
                name,
                dict(
                    error="Shared primitive budget reached", candidate_dispatched=False
                ),
            )
        self.round_calls += 1
        record = dict(
            name=name,
            arguments=deepcopy(args),
            started_monotonic_s=time.monotonic(),
            executed_start=self.adapter.dispatcher.sequence,
            model_chunk_start=len(self.env.action_chunks),
        )
        previous = self.calls[-1] if self.calls else {}
        from .continuous_waypoint import enabled as continuous_enabled
        carry_closed_target = (
            continuous_enabled(self.adapter.driver)
            and name == 'move_to' and args.get('gripper', -1.) > 0
            and previous.get('name') == 'move_to'
            and previous.get('acknowledged_complete') is True
            and previous.get('executed_end') == self.adapter.dispatcher.sequence
            and self.env.jaw_target == 1.)
        self.calls.append(record)
        # Only retain an owned completed CLOSE across consecutive translations.
        # step_physical still reads fresh terminal feedback before omitting it.
        # OPEN/release and other primitives reset this cache and retain G3.
        if not carry_closed_target:
            self.env.jaw_target = None
        record['retained_close_target_candidate'] = carry_closed_target
        try:
            if name == "hold":
                seconds = args["seconds"]
                if (
                    isinstance(seconds, bool)
                    or not isinstance(seconds, (int, float))
                    or not 0 < seconds <= 10
                ):
                    raise ValueError("hold seconds must be in (0,10]")
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    self.check_cancelled()
                    time.sleep(min(0.05, max(0, deadline - time.monotonic())))
                value = dict(seconds=seconds, physical_hold=True)
            else:
                method = getattr(
                    self._official if name in OFFICIAL_ACTIONS else self._primitives,
                    name,
                )
                inspect.signature(method).bind(**args)
                self._primitives.set_obs(self.env.observe())
                from . import continuous_waypoint
                if (name == "move_to" and args.get("target_yaw") is None
                        and continuous_waypoint.enabled(self.adapter.driver)):
                    value = continuous_waypoint.execute(self.env, self._primitives, args)
                else:
                    value = method(**args)  # Frozen OSC/VLA path remains available.
                if name in OFFICIAL_ACTIONS:
                    value = value.data
                # The full VLA input/last chunk is not tool-result prose.
                if name == "vla_grasp":
                    value = dict(value)
                    value.pop("last_chunk", None)
                if isinstance(value, dict):
                    value.pop("observation", None)
            value = _agent_result(value)
            value["acknowledged_complete"] = True
            record['acknowledged_complete'] = True
            return self.observation_result(name, value)
        except TargetRejected as exc:
            executed = self.adapter.dispatcher.sequence - record["executed_start"]
            return self.observation_result(
                name,
                dict(
                    error=str(exc),
                    hardware_preflight_rejected=True,
                    candidate_dispatched=False,
                    consumption_unknown=False,
                    recovery_allowed=True,
                    partial_primitive_executed=executed > 0,
                    executed_before_rejection=executed,
                    executed_count_unit="acknowledged_device_commands",
                    completed_logical_model_actions=sum(
                        c["completed_logical_actions"]
                        for c in self.env.action_chunks[record["model_chunk_start"] :]
                    ),
                ),
            )
        except (CapabilityUnavailable, ValueError) as exc:
            return _tool_result(
                name,
                dict(
                    error=str(exc),
                    capability_unavailable=True,
                    candidate_dispatched=False
                    if self.adapter.dispatcher.sequence == record["executed_start"]
                    else None,
                    executed_before_failure=self.adapter.dispatcher.sequence
                    - record["executed_start"],
                    recovery_allowed=True,
                ),
            )
        finally:
            record["elapsed_s"] = time.monotonic() - record["started_monotonic_s"]
            record["executed_end"] = self.adapter.dispatcher.sequence

    def file_tool(self, name, args):
        root = Path(__file__).resolve().parents[2] / "upstream/RPent-reproduce-libero"
        path = Path(args.get("path") or self.output_dir)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        allowed = (
            (self.output_dir,)
            if name == "write_text_file"
            else (self.output_dir, self.memory_dir.resolve())
        )
        if not any(path == p or p in path.parents for p in allowed):
            return _tool_result(
                name, dict(error="File path outside this run or frozen Global Memory")
            )
        if name == "list_dir":
            parameters = dict(path=str(path))
        else:
            parameters = dict(args, path=str(path))
        from rpent.tools.common import TOOL_HANDLERS

        return _tool_result(name, TOOL_HANDLERS[name](**parameters))

    def cancel_active_and_wait(self):
        return self.legacy.cancel_active_and_wait()

    def close(self):
        return self.legacy.close()
