"""Opt-in S101 round lifecycle over existing admission and control consumers.

The Owner supplies its actual round-start acquisition. The existing driver owns
base.begin_round; the existing bridge/planner owns finish. No Scene, capture,
session, executor, recovery or private task state is implemented here.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import time
from types import FunctionType, MethodType, SimpleNamespace

import numpy as np

from research.v3.bridge.b0 import BUDGETED_ACTIONS
from research.v4.candidates.p102_v43_actual_consumer_20260928.runtime import (
    Consumer as CommonConsumer, _public_workspace,
)
from research.v4.candidates.v43_own_lease_lifecycle_20260928.runtime import OwnLeaseLifecycle
from research.v4.candidates.v43_task_constraints_binding_20260928.runtime import TaskConstraints
from research.v4.leash_v43.method_inputs import (
    PublicFrame, PublicHistory, ReviewInputRequest, project_review_input,
    validate_review_response,
)

INSTRUCTIONS = ("Move the pan onto the wooden board.", "Put the pan in my hands.")


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _frames(packet, version):
    if set(packet.get("cameras", {})) != {"agentview", "robot0_eye_in_hand"}:
        raise ValueError("exact two owner-acquired public views required")
    frame_id = packet.get("frame_id")
    if frame_id is None:
        raise ValueError("public acquisition identity absent")
    result = []
    for name in ("agentview", "robot0_eye_in_hand"):
        camera = packet["cameras"][name]
        if camera.get("frame_id") != frame_id:
            raise ValueError("mixed owner public acquisitions")
        arrays = []
        for key in ("rgb", "depth_m", "intrinsics", "extrinsics"):
            a = np.array(camera[key], copy=True)
            a.flags.writeable = False
            arrays.append(a)
        result.append(PublicFrame(name, *arrays, version))
    return tuple(result)


def _executed_summaries(calls):
    # Exactly the existing S1 bridge's public motion summary schema. This is
    # only validation of its seal input, never an execution or planned-action log.
    result = []
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise ValueError("actual common toolkit call schema changed")
        if call["name"] not in BUDGETED_ACTIONS:
            continue
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            raise ValueError("actual public primitive arguments malformed")
        outcome = call.get("result", {})
        error = outcome.get("error") if isinstance(outcome, dict) else None
        result.append((call["name"], {"arguments": dict(args),
                      "outcome": "error" if error else "executed"}))
    return tuple(result)


class ProvidedCommonConsumer(CommonConsumer):
    """Reuse the complete constructor code with a per-instance provider seam."""
    def __init__(self, *args, provider_factory, **kwargs):
        if not callable(provider_factory):
            raise TypeError("selected own public provider factory required")
        super().__init__(*args, provider_factory=provider_factory, **kwargs)

    def workspace(self):
        self.acquire += 1
        if self._start_packet_pending is not None:
            packet, self._start_packet_pending = self._start_packet_pending, None
            eef = tuple(float(x) for x in packet["robot_state"]["eef_pos"])
            public = _public_workspace(self.provider, packet, eef, ())
        elif self.boundary.task_approved and self._acquisition_command in {"release", "set_gripper", "hold"}:
            packet = self.scene.capture_agent()
            eef = tuple(float(x) for x in packet["robot_state"]["eef_pos"])
            public = _public_workspace(self.provider, packet, eef, ())
        elif self.boundary.task_approved:
            public = self.provider()
            packet = self.provider.current_packet
        else:
            raise RuntimeError("task review requires its already-acquired Owner start packet")
        # Validate acquisition consistency without a second capture.
        _frames(packet, "validation-only")
        return SimpleNamespace(public=public, packet=packet, version="pending",
                               acquisition="pending")

    def request(self, gate, workspace, *, action="", lease=None):
        frames = _frames(workspace.packet, workspace.acquisition)
        state = workspace.packet["robot_state"]
        from scipy.spatial.transform import Rotation
        robot = {"eef_pos": tuple(state["eef_pos"]),
                 "eef_rotation": Rotation.from_quat(state["eef_quat"]).as_matrix().tolist(),
                 "gripper_aperture_m": float(np.abs(state["gripper_qpos"]).sum()),
                 "holding_estimate": self.adapter.estimator.kind.value}
        request = ReviewInputRequest(
            method="Ours", gate=gate, current_workspace=frames[0], current_wrist=frames[1],
            history=self.sealed_history,
            event_context={"trusted_instruction": self.instruction, "proposed_action": action,
                "event_description": f"Current semantic binding_version: {workspace.version}. "
                    "Any returned lease must use that version; evidence_version identifies the fresh acquisition."},
            robot_state=robot, lease=lease, evidence_version=workspace.acquisition)
        if gate == 2:
            from research.v4.bridge.full_corner_lite import declare_public_names
            names = getattr(self.provider, "public_entity_names", None)
            if names is None:
                names = self.provider.public_prompts
            request = replace(request, event_context=declare_public_names(
                request.event_context, workspace, names))
        projected = project_review_input(request, self.policy,
                                        allow_draft_for_mock=self.allow_draft_for_mock)
        projected["deadline_monotonic_ns"] = time.monotonic_ns() + 180_000_000_000
        return projected


class S101RoundConsumer(TaskConstraints, OwnLeaseLifecycle, ProvidedCommonConsumer):
    """Owner callback contract; not yet installed in the current S101 core."""
    def __init__(self, *args, allow_draft_for_mock=False, **kwargs):
        self.allow_draft_for_mock = allow_draft_for_mock
        self.sealed_history = ()
        self.round_records = []
        self._start_packet_pending = None
        self._start_keyframes = None
        self._live_finish = None
        self._round_call_start = None
        self._validated_task = None
        super().__init__(*args, **kwargs)
        # Preserve TaskConstraints' actual finish wrapper and require a live
        # common-toolkit terminal record before accepting a finish-based seal.
        original_finish = self.boundary._finish
        def finish(boundary, status, message):
            count = len(self.base.calls)
            result = original_finish(status, message)
            added = self.base.calls[count:]
            expected = {"status": status, "message": message}
            if (self._start_keyframes is not None and not boundary.round_open
                    and len(added) == 1 and added[0].get("name") == "finish"
                    and self.base.finish_result == expected
                    and not getattr(result, "result", {}).get("error")):
                self._live_finish = {"round_id": boundary.round_id, **expected}
                self._clear_round_binding("actual boundary finish")
            return result
        self.boundary._finish = MethodType(finish, self.boundary)

    def _clear_round_binding(self, reason):
        self._expire(reason)
        self.lease_receipt = None
        self._lease_public = self._lease_holding = self._lease_phase = None
        self._public_phase = None
        self.boundary.lease = None
        self._clear_constraints()
        self._binding_revision += 1
        self.adapter.invalidate_binding()

    def _version(self, workspace):
        instruction_hash = sha256(self.instruction.encode()).hexdigest()[:16]
        self.binding_version = (f"{self.episode}:round{self.boundary.round_id}:instruction{instruction_hash}"
            f":holding{self.adapter.estimator.generation}:binding{self._binding_revision}")
        workspace.version = self.binding_version
        workspace.acquisition = (f"{workspace.version}:frame{workspace.packet['frame_id']}:capture{self.acquire}")
        return workspace

    def admit_round(self, round_id, instruction, start_packet, *, reset_owner="driver",
                    finish_owner="bridge", driver_reset_completed):
        """Return gate-1 output only; never reset base or finish ASK/REFUSE.

        Call from the actual Owner after its driver's base.begin_round and two
        public capture. Existing bridge consumes this decision and executes
        finish through self.boundary. Seal before the next driver reset.
        """
        if (reset_owner != "driver" or finish_owner != "bridge" or driver_reset_completed is not True
                or type(round_id) is not int or round_id != len(self.sealed_history) + 1
                or round_id > len(INSTRUCTIONS) or instruction != INSTRUCTIONS[round_id - 1]
                or self.boundary.round_open or self._start_keyframes is not None
                or self.base.round_finished or self.base.finish_result is not None):
            raise ValueError("external reset/finish ownership or exact S101 round order invalid")
        self._clear_round_binding("new trusted user round")
        boundary = self.boundary
        boundary.round_id, boundary.instruction = round_id, instruction
        boundary.round_open, boundary.task_approved = True, False
        boundary.first_action, boundary.after_transition, boundary.denials = True, False, 0
        self.instruction = instruction
        self.adapter.set_review_context(instruction=instruction)
        self._round_call_start = len(self.base.calls)
        self._start_packet_pending = start_packet
        workspace = self.workspace()
        self._start_keyframes = _frames(start_packet, workspace.acquisition)
        self._live_finish = None
        checked = validate_review_response("Ours", 1, self.task_review(
            instruction=instruction, workspace=workspace, episode=self.episode, round_id=round_id))
        self._validated_task = deepcopy(checked)
        boundary.logs.append({"gate": 1, "round_id": round_id, **checked})
        if checked["result"] == "PROCEED":
            boundary.task_approved = True
            boundary.lease = checked.get("lease")
        return checked

    def seal_round(self, *, primitive_summaries, boundary, resource_end=None):
        """Seal only executed bridge summaries and an actual terminal boundary.

        resource_end is a separate, explicit quota/resource boundary and is
        never reported as finish. The Owner/bridge must supply its actual seal;
        this adapter neither proves quota exhaustion nor initiates recovery.
        """
        if self._start_keyframes is None or self._round_call_start is None:
            raise ValueError("no admitted unsealed round")
        actual = _executed_summaries(self.base.calls[self._round_call_start:])
        if _json(primitive_summaries) != _json(actual):
            raise ValueError("seal summaries differ from actual common toolkit results")
        if not isinstance(boundary, str) or not boundary.strip():
            raise ValueError("actual bridge boundary required")
        if resource_end is None:
            if (self._live_finish is None or self.boundary.round_open
                    or self._live_finish["round_id"] != self.boundary.round_id
                    or not self.base.round_finished):
                raise ValueError("finish seal requires actual live Consumer boundary closure")
            terminal = {"kind": "finish", **self._live_finish}
        else:
            if (resource_end not in {"round_call_quota", "episode_call_quota",
                                    "planner_turn_quota", "planner_time_quota"}
                    or boundary != resource_end or self._live_finish is not None
                    or not self.boundary.round_open):
                raise ValueError("explicit non-finish resource boundary invalid")
            terminal = {"kind": "resource_end", "reason": resource_end}
            self.boundary.round_open = self.boundary.task_approved = False
            self._clear_round_binding("external resource boundary")
        item = PublicHistory(self.instruction, tuple(_json(s) for s in actual),
            _json({"terminal": terminal, "bridge_boundary": boundary,
                   "validated_task_review": self._validated_task}), self._start_keyframes)
        self.sealed_history += (item,)
        self.adapter.set_review_context(history=self.sealed_history)
        self.round_records.append({"round_id": self.boundary.round_id, "instruction": self.instruction,
            "terminal": terminal, "executed_primitive_count": len(actual),
            "keyframes": tuple(f.evidence_version for f in item.keyframes)})
        self._start_keyframes = self._round_call_start = None
        return item
