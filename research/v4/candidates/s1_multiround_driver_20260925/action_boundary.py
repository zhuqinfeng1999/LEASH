"""S1 gate-2 RPent boundary reusing existing ActionGate and gate-3 outlet.

All motion proposals derive only from current public workspace and the exact
planner JSON. The caller supplies public RGB-D perception/grounding callbacks.
No simulator/evaluator coordinates are imported or read here.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Callable

from research.v3.bridge.b0 import MOTION_TOOLS, _tool_result
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import (
    ActionGate, PrimitiveProposal, PublicWorkspace,
)

ROBOT_ACTIONS = frozenset(MOTION_TOOLS | {"hold"})
PUBLIC_NONACTIONS = frozenset({
    "finish", "view_env_state", "view_camera_meta", "segment", "back_project",
    "read_text_file", "write_text_file", "list_dir",
})


def _point(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3 or
        any(isinstance(x, bool) or not isinstance(x, (int, float)) or
            not math.isfinite(float(x)) for x in value)):
        raise ValueError("public target is not a finite XYZ")
    return tuple(float(x) for x in value)


def s1_public_proposal(name: str, args: dict[str, Any],
                       workspace: PublicWorkspace,
                       locate_vla_target: Callable[[str, PublicWorkspace],
                                                   tuple[str, tuple[float, float, float]] | None]
                       ) -> PrimitiveProposal:
    """Conservative public-only S1 proposal for the frozen RPent tool set."""
    if name not in ROBOT_ACTIONS or not isinstance(args, dict):
        raise ValueError("unregistered S1 robot action")
    eef = _point(workspace.eef_xyz)
    hold = workspace.evidence.robot.hold_estimate
    if hold not in {"held", "empty"}:
        raise ValueError("public held state unknown")
    held = hold == "held"
    pan = workspace.box("pan")
    if held and pan is None:
        raise ValueError("held pan not grounded in public RGB-D")
    held_extra = 0.0
    if held:
        lower, upper = pan.lower, pan.upper
        center = tuple((lo + hi) / 2 for lo, hi in zip(lower, upper))
        held_extra = math.dist(eef, center) + math.dist(lower, upper) / 2
    common = dict(parameters=tuple(sorted((str(k), repr(v)) for k, v in args.items())),
                  operated_entity="pan" if held else None,
                  gripper_radius_m=.035, held_object_radius_m=held_extra)
    if name in {"move_to", "move_pose"}:
        target = _point(args.get("xyz"))
        gripper = args.get("gripper", -1)
        if gripper not in (-1, 1):
            raise ValueError("invalid RPent gripper target")
        if held and gripper == -1:
            raise ValueError("held pan transport cannot open gripper before landing; split move and release")
        # A visible target inside the pan's public RGB-D box is an intentional
        # pan approach/grasp, even while the gripper is still empty. Declaring
        # that entity lets the reviewer assess controlled contact instead of
        # treating the grasp approach as an unrelated collision. No hidden
        # object pose or grasp-zone metadata is used.
        pan_approach = bool(pan is not None and all(
            lo - .02 <= coordinate <= hi + .02
            for coordinate, lo, hi in zip(target, pan.lower, pan.upper)))
        motion_common = {**common,
                         "operated_entity": "pan" if held or pan_approach else None}
        return PrimitiveProposal(
            name, motion_waypoints=(eef, target),
            placement_xyz=target if held and gripper == -1 else None,
            changes_grasp_state=("release" if held and gripper == -1 else
                                  "grasp" if not held and gripper == 1 else None),
            geometry_ambiguous=True, **motion_common)
    if name in {"rotate_wrist", "rotate_pitch"}:
        if held and args.get("gripper", 1) == -1:
            raise ValueError("held pan rotation cannot open gripper")
        return PrimitiveProposal(name, motion_waypoints=(eef, eef),
                                 geometry_ambiguous=True, **common)
    if name in {"pi0_pick", "pi0_doubled"}:
        prompt = args.get("prompt")
        located = locate_vla_target(prompt, workspace) if isinstance(prompt, str) else None
        if located is None or not isinstance(located[0], str):
            raise ValueError("VLA transport target unavailable from public RGB-D")
        target = _point(located[1])
        return PrimitiveProposal("vla_act", vla_target_xyz=target,
                                 parameters=(("rpent_tool", name), *common["parameters"]),
                                 operated_entity=located[0],
                                 gripper_radius_m=common["gripper_radius_m"],
                                 held_object_radius_m=held_extra,
                                 changes_grasp_state="grasp" if name == "pi0_pick" else None,
                                 geometry_ambiguous=True)
    if name == "set_gripper":
        grip = args.get("gripper")
        if grip not in (-1, 1):
            raise ValueError("invalid set_gripper target")
        if not held and grip == 1:
            if pan is None:
                raise ValueError("pan is not grounded for close")
            nearest = tuple(min(max(x, lo), hi) for x, lo, hi in
                            zip(eef, pan.lower, pan.upper))
            if math.dist(eef, nearest) > .055:
                raise ValueError("gripper is not publicly near pan")
            common["operated_entity"] = "pan"
        return PrimitiveProposal(name, changes_grasp_state=(
            "release" if held and grip == -1 else
            "grasp" if not held and grip == 1 else None), **common)
    if name == "release":
        return PrimitiveProposal(name, changes_grasp_state="release" if held else None,
                                 **common)
    return PrimitiveProposal(name, **common)  # hold


class S1ActionBoundary:
    """Check every S1 primitive before delegating to the existing RPent tools."""

    def __init__(self, base, adapter, *, task_adapter, reviewer,
                 workspace: Callable[[], PublicWorkspace],
                 locate_vla_target: Callable[[str, PublicWorkspace],
                                             tuple[str, tuple[float, float, float]] | None]):
        self.base = base
        self.adapter = adapter
        self.task_adapter = task_adapter
        self.reviewer = reviewer
        self.workspace = workspace
        self.locate_vla_target = locate_vla_target
        self.gate: ActionGate | None = None
        self.gate_round = 0
        self.proposals: list[dict[str, Any]] = []
        self.gate_logs = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_tools_spec(self):
        return self.base.get_tools_spec()

    def _current_gate(self):
        round_index = len(self.task_adapter.task_decisions)
        if round_index not in (1, 2):
            raise RuntimeError("action before the S1 task gate")
        if self.gate_round != round_index:
            if self.gate is not None:
                self.gate_logs.extend(asdict(x) for x in self.gate.logs)
            decision = self.task_adapter.task_decisions[-1]
            self.gate = ActionGate(
                round_id=f"{self.task_adapter.episode_id}-round{round_index}",
                task=decision, snapshot=self.workspace, reviewer=self.reviewer,
                finish=lambda status, message: self.base.execute_tool(
                    "finish", {"status": status, "message": message}),
                timeout_s=180)
            self.gate_round = round_index
        return self.gate

    def execute_tool(self, name: str, input_dict: dict[str, Any]):
        if name == "finish":
            result = self.base.execute_tool(name, input_dict)
            if self.gate is not None:
                self.gate.end_round()
            self.adapter.set_commit_conditions(None)
            return result
        if name in PUBLIC_NONACTIONS:
            return self.base.execute_tool(name, input_dict)
        try:
            gate = self._current_gate()
        except Exception as exc:
            return _tool_result(name, {"error": f"action gate unavailable: {exc}",
                                       "code": "action_denied", "gate": 2})
        if name not in ROBOT_ACTIONS:
            decision = gate._deny("unregistered S1 tool",
                                  "use a registered RPent primitive",
                                  reviewer_called=False)
            return _tool_result(name, {"error": decision.reason,
                                       "code": "action_denied", "gate": 2,
                                       "decision": decision.result,
                                       "constraint": decision.constraint})
        try:
            public = self.workspace()
            proposal = s1_public_proposal(
                name, input_dict, public, self.locate_vla_target)
            decision = gate.check(proposal)
        except Exception as exc:
            reason = f"public proposal failed: {type(exc).__name__}: {exc}"
            decision = gate._deny(reason,
                                  "refresh public perception or correct the primitive; "
                                  "transport a held pan with a closed gripper and release separately",
                                  reviewer_called=False)
            self.proposals.append({"tool": name, "decision": decision.result,
                                   "reason": reason, "constraint": decision.constraint})
            self.adapter.set_commit_conditions(None)
            return _tool_result(name, {"error": reason,
                                       "code": "action_denied", "gate": 2,
                                       "decision": decision.result,
                                       "constraint": decision.constraint})
        self.proposals.append({"tool": name, "arguments": dict(input_dict),
                               "proposal": asdict(proposal), "decision": decision.result,
                               "reason": decision.reason, "constraint": decision.constraint})
        if decision.result != "APPROVE":
            self.adapter.set_commit_conditions(None)
            return _tool_result(name, {"error": decision.reason,
                                       "code": "action_denied", "gate": 2,
                                       "constraint": decision.constraint})
        if not gate.is_current_lease(decision.lease):
            return _tool_result(name, {"error": "expired action lease",
                                       "code": "action_denied", "gate": 2})
        self.adapter.set_commit_conditions(decision.lease.commit_conditions)
        self.adapter.last_outcome = None
        try:
            result = self.base.execute_tool(name, input_dict)
        finally:
            self.adapter.set_commit_conditions(None)
        outcome = self.adapter.last_outcome
        if outcome is not None and outcome.status == "intercepted":
            self.proposals[-1]["commit_outcome"] = asdict(outcome)
            return _tool_result(name, {"error": outcome.reason or "commit blocked",
                                       "code": "commit_intercepted", "gate": 3,
                                       "status": "intercepted",
                                       "discarded_commands": outcome.discarded})
        result_dict = getattr(result, "result", {})
        if isinstance(result_dict, dict) and "error" not in result_dict:
            gate.on_primitive_completed(proposal)
        return result
