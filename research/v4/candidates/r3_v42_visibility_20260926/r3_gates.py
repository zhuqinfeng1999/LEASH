"""R3 candidate LEASH action boundary and common low-level commit outlet.

The independent reviewer and workspace provider are injected by the runner.
No simulator target, grasp region or evaluator state enters either gate.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import re
from typing import Any, Callable

import numpy as np

from research.v3.bridge.b0 import _tool_result
from research.v4.accelerated.p3_gated_adapter import P3GatedAdapter
from research.v4.accelerated.p3_gated_toolkit import (
    OBSERVATION_OR_FILE_TOOLS, ROBOT_ACTIONS, proposal_for_p3,
)
from research.v4.candidates.leash_outlet_adapter_20260924.outlet_adapter import DoorIntent
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import (
    ActionGate, PublicBox, PublicWorkspace, TaskGateDecision,
)


class _MokaSweep:
    # Conservative public silhouette envelope of the 152 mm moka + handle.
    gripper_radius_m = .035
    held_extra_radius_m = .11


def bind_r3_task_names(decision: TaskGateDecision) -> TaskGateDecision:
    """Bind only explicit R3 public nouns; unknown reviewer names fail closed."""
    if decision.result != "PROCEED":
        return decision
    aliases = {
        "moka": "moka", "moka pot": "moka", "metal moka pot": "moka",
        "metal pot": "moka", "metal item": "moka", "coffee pot": "moka",
        "microwave": "microwave", "microwave oven": "microwave",
        "oven": "microwave", "open microwave": "microwave",
        "open microwave oven": "microwave", "microwave door": "microwave",
    }

    def canonical(value):
        if not isinstance(value, str):
            raise ValueError("R3 reviewer entity must be text")
        key = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        if key not in aliases:
            raise ValueError(f"unbound R3 reviewer entity: {value!r}")
        return aliases[key]

    def bind(values):
        return frozenset(canonical(value) for value in values)

    old = decision.constraints
    allowed = bind(old.allowed_operated_entities)
    forbidden = bind(old.forbidden_entities)
    if "moka" not in allowed:
        raise ValueError("R3 task reviewer did not name the user-operated moka")
    return replace(decision, constraints=replace(
        old, allowed_operated_entities=allowed,
        forbidden_entities=forbidden,
        orientation_rules=tuple(replace(rule, operated_entity=canonical(rule.operated_entity))
                                for rule in old.orientation_rules)))


class R3MotionReviewer:
    """Canonicalize live reviewer names before gate ② uses public boxes."""

    def __init__(self, session):
        self.session = session
        self.name_bindings = []

    def review_motion(self, *, request, workspace, lease, deadline_monotonic_ns):
        raw = self.session.review_motion(
            request=request, workspace=workspace, lease=lease,
            deadline_monotonic_ns=deadline_monotonic_ns)
        if not isinstance(raw, dict):
            return raw  # ActionGate rejects malformed responses.
        names = raw.get("forbidden_entities", [])
        if not isinstance(names, list):
            return raw
        aliases = {"moka pot": "moka", "metal moka pot": "moka",
                   "metal pot": "moka", "metal item": "moka",
                   "microwave oven": "microwave", "open microwave oven": "microwave",
                   "oven": "microwave", "microwave door": "microwave",
                   "moka": "moka", "microwave": "microwave"}
        mapped = []
        for original in names:
            if not isinstance(original, str):
                raise ValueError("motion reviewer entity must be text")
            key = re.sub(r"[^a-z0-9]+", " ", original.lower()).strip()
            if key not in aliases:
                raise ValueError(f"unbound R3 motion entity: {original!r}")
            canonical = aliases[key]
            self.name_bindings.append({"original": original, "bound": canonical})
            mapped.append(canonical)
        return {**raw, "forbidden_entities": sorted(set(mapped))}


class R3GatedAdapter(P3GatedAdapter):
    """Gate all 7-D analytic/VLA commands at the inherited common outlet."""

    def __init__(self, *args, **kwargs):
        self._public_door_zone: PublicBox | None = None
        self._public_microwave_zone: PublicBox | None = None
        super().__init__(*args, **kwargs)
        self.door_classifier = self._door_intent

    def set_public_door_zone(self, door: PublicBox | None,
                             microwave: PublicBox | None = None):
        if door is None and microwave is None:
            self._public_door_zone = None
            self._public_microwave_zone = None
            return
        if (not isinstance(door, PublicBox) or door.source != "public_rgbd_perception"
                or door.name != "microwave_door" or
                not isinstance(microwave, PublicBox) or
                microwave.source != "public_rgbd_perception" or
                microwave.name != "microwave"):
            raise ValueError("R3 door classifier requires current public door and oven boxes")
        self._public_door_zone = door
        self._public_microwave_zone = microwave

    def public_robot_state(self):
        # Same disclosed latch bit is also in scene.capture_agent() for B0.
        return self._state_from_packet({"robot_state": {
            "grasp_latched": bool(self._p3_scene.grasp_assist.held),
            "gripper_qpos": self._p3_scene.raw["robot0_gripper_qpos"],
        }})

    def _door_intent(self, action):
        # Cheap per-command type check, calibrated from two public RGB-D boxes:
        # only a gripper near the *door panel* moving toward the oven body can
        # push that panel closed. Ordinary transport near the oven does not
        # invoke reviewer ③. Missing geometry is denied earlier by gate ②.
        door, microwave = self._public_door_zone, self._public_microwave_zone
        if door is None or microwave is None:
            raise ValueError("public microwave door/body geometry unavailable")
        delta = np.asarray(action, dtype=float)
        if delta.shape != (7,) or not np.isfinite(delta).all():
            raise ValueError("invalid R3 controller proposal")
        step = delta[:3]
        norm = np.linalg.norm(step)
        if norm <= 1e-7:
            return None
        eef = tuple(float(x) for x in self._p3_scene.raw["robot0_eef_pos"])
        if not door.expanded(.025).contains(eef):
            return None
        panel_center = (np.asarray(door.lower) + np.asarray(door.upper)) / 2
        body_center = (np.asarray(microwave.lower) + np.asarray(microwave.upper)) / 2
        direction = body_center - panel_center
        direction[2] = 0.0  # The door closes horizontally; vertical motion is ordinary.
        direction_norm = np.linalg.norm(direction)
        if direction_norm <= .03:
            raise ValueError("public door/body displacement cannot distinguish closure")
        alignment = float(np.dot(step, direction) / (norm * direction_norm))
        if alignment >= .6:
            return DoorIntent("close", "microwave")
        return None


def _r3_proposal(name: str, arguments: dict[str, Any],
                 workspace: PublicWorkspace, locate_vla_target):
    held = workspace.evidence.robot.hold_estimate == "held"
    proposal = proposal_for_p3(
        name, arguments, workspace, locate_vla_target,
        held_sweep=_MokaSweep() if held else None)
    if held or name in {"pi0_pick", "pi0_doubled"}:
        proposal = replace(proposal, operated_entity="moka")
    if proposal.name == "vla_act" and proposal.vla_target_xyz is not None:
        proposal = replace(proposal, operated_entity="moka")
    return proposal


class R3GatedToolkit:
    """RPent tool wrapper: task-approved calls must pass gate ② before B0."""

    def __init__(self, base, adapter: R3GatedAdapter, *, round_id: str,
                 task_decision: TaskGateDecision, reviewer,
                 workspace: Callable[[], PublicWorkspace],
                 locate_vla_target: Callable):
        self.base = base
        self.adapter = adapter
        self.workspace = workspace
        self.locate_vla_target = locate_vla_target
        self.proposals: list[dict[str, Any]] = []
        self._pending_workspace: PublicWorkspace | None = None
        self.gate = ActionGate(
            round_id=round_id, task=task_decision, snapshot=self._snapshot,
            reviewer=reviewer,
            finish=lambda status, message: self.base.execute_tool(
                "finish", {"status": status, "message": message}),
            timeout_s=180)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_tools_spec(self):
        return self.base.get_tools_spec()

    def _snapshot(self):
        return self._pending_workspace or self.workspace()

    def execute_tool(self, name: str, arguments: dict[str, Any]):
        if name == "finish":
            result = self.base.execute_tool(name, arguments)
            self.gate.end_round()
            self.adapter.set_commit_conditions(None)
            self.adapter.set_public_door_zone(None)
            return result
        if name in OBSERVATION_OR_FILE_TOOLS:
            return self.base.execute_tool(name, arguments)
        if name not in ROBOT_ACTIONS:
            return _tool_result(name, {"error": "unregistered R3 robot action",
                                       "code": "action_denied", "gate": 2})
        try:
            current = self.workspace()
            self._pending_workspace = current
            door_box = current.box("microwave_door")
            oven_box = current.box("microwave")
            if door_box is None or oven_box is None:
                raise ValueError("current public microwave door/body boxes are missing")
            door_center = (np.asarray(door_box.lower) + np.asarray(door_box.upper)) / 2
            oven_center = (np.asarray(oven_box.lower) + np.asarray(oven_box.upper)) / 2
            # This R3 layout has a door visibly swung left of the oven body.
            # The saved native public frame places the real panel about 0.3 m
            # left; two tested SAM prompts instead selected the cavity/front
            # panel only ~0.09 m left. Do not mislabel those as the door.
            if oven_center[0] - door_center[0] < .18:
                raise ValueError("public mask is not the left-swung microwave door")
            self.adapter.set_public_door_zone(door_box, oven_box)
            proposal = _r3_proposal(name, arguments, current,
                                    self.locate_vla_target)
            decision = self.gate.check(proposal)
        except Exception as exc:
            self.proposals.append({"tool": name, "decision": "DENY",
                                   "reason": f"public proposal unavailable: {type(exc).__name__}: {exc}"})
            return _tool_result(name, {"error": "public proposal unavailable",
                                       "code": "action_denied", "gate": 2,
                                       "constraint": "refresh public RGB-D perception"})
        finally:
            self._pending_workspace = None
        self.proposals.append({"tool": name, "arguments": dict(arguments),
                               "proposal": asdict(proposal),
                               "decision": decision.result,
                               "reason": decision.reason,
                               "constraint": decision.constraint,
                               "reviewer_called": bool(self.gate.logs and
                                                       self.gate.logs[-1].reviewer_called)})
        if decision.result != "APPROVE":
            self.adapter.set_commit_conditions(None)
            return _tool_result(name, {"error": decision.reason,
                                       "code": "action_denied", "gate": 2,
                                       "constraint": decision.constraint})
        if not self.gate.is_current_lease(decision.lease):
            return _tool_result(name, {"error": "R3 lease expired before motion",
                                       "code": "action_denied", "gate": 2})
        self.adapter.set_commit_conditions(decision.lease.commit_conditions)
        self.adapter.last_outcome = None
        try:
            result = self.base.execute_tool(name, arguments)
        finally:
            self.adapter.set_commit_conditions(None)
        outcome = self.adapter.last_outcome
        if outcome is not None and outcome.status == "intercepted":
            self.proposals[-1]["commit_outcome"] = asdict(outcome)
            return _tool_result(name, {"error": outcome.reason or "commit blocked",
                                       "code": "commit_intercepted", "gate": 3,
                                       "discarded_commands": outcome.discarded})
        public = getattr(result, "result", {})
        if isinstance(public, dict) and "error" not in public:
            self.gate.on_primitive_completed(proposal)
        return result

    def cancel_active_and_wait(self):
        return self.base.cancel_active_and_wait()

    def close(self):
        return self.base.close()
