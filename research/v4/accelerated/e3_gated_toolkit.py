"""E3 primitive boundary before a carried hot vessel approaches the mug.

Gate ② uses only the submitted RPent primitive, current public RGB-D boxes,
and common disclosed latch. It asks the independent reviewer before a held
vessel enters the mug footprint; gate ③ separately handles proposed low-level
tilt/open commands. Neither gate reads the E3 evaluator or true object poses.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from typing import Any

from research.v3.bridge.b0 import _tool_result
from research.v4.accelerated.e3_gate_boundary import requires_cup_approach_review
from research.v4.accelerated.e3_name_binding import bind_motion_names
from research.v4.accelerated.p3_gated_toolkit import (
    OBSERVATION_OR_FILE_TOOLS, ROBOT_ACTIONS, P3GatedToolkit,
    proposal_for_p3,
)
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import (
    MotionReviewRequest, PrimitiveProposal, PublicWorkspace,
)


@dataclass(frozen=True)
class E3HeldSweep:
    # Full public kettle silhouette includes its spout, about 26 cm from the
    # held handle. The old 13 cm body-only radius missed a real cup approach.
    # This is a conservative common asset envelope, not a private grasp zone.
    gripper_radius_m: float = .035
    held_extra_radius_m: float = .26


class E3MotionReviewer:
    def __init__(self, session):
        self.session = session
        self.name_bindings: list[dict[str, str]] = []

    def review_motion(self, *, request: MotionReviewRequest,
                      workspace: PublicWorkspace, lease,
                      deadline_monotonic_ns: int):
        raw = self.session.review_motion(
            request=request, workspace=workspace, lease=lease,
            deadline_monotonic_ns=deadline_monotonic_ns)
        bound, records = bind_motion_names(raw)
        self.name_bindings.extend(records)
        return bound


def proposal_for_e3(name: str, arguments: dict[str, Any],
                    workspace: PublicWorkspace, locate_vla_target) -> PrimitiveProposal:
    held = workspace.evidence.robot.hold_estimate == "held"
    proposal = proposal_for_p3(
        name, arguments, workspace, locate_vla_target,
        held_sweep=E3HeldSweep() if held else None)
    if held or name in {"pi0_pick", "pi0_doubled"}:
        proposal = replace(proposal, operated_entity="kettle",
                           held_object_radius_m=.26 if held else 0.)
    if not held:
        return proposal
    path = proposal.motion_waypoints
    if proposal.name == "vla_act" and proposal.vla_target_xyz is not None:
        path = (workspace.eef_xyz, proposal.vla_target_xyz)
    if len(path) >= 2:
        mug = workspace.box("mug")
        center = None if mug is None else tuple(
            (lo + hi) / 2 for lo, hi in zip(mug.lower, mug.upper))
        radius = None if mug is None else max(
            (mug.upper[0]-mug.lower[0]) / 2,
            (mug.upper[1]-mug.lower[1]) / 2)
        if requires_cup_approach_review(
            held_public=True, primitive_name=(
                "vla_act" if proposal.name == "vla_act" else "move_to"),
            current_eef_xyz=path[0], proposed_eef_xyz=path[-1],
            cup_center_xyz=center, payload_reach_m=.26,
            cup_radius_m=radius):
            proposal = replace(
                proposal, geometry_ambiguous=True,
                parameters=(*proposal.parameters,
                            ("phase", "held_hot_vessel_cup_approach_precritical")),
            )
    elif proposal.name == "vla_act":
        # No localized VLA destination can be safely preapproved.
        proposal = replace(proposal, geometry_ambiguous=True)
    return proposal


class E3GatedToolkit(P3GatedToolkit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pick_operation: tuple[str, str] | None = None

    @contextmanager
    def operation_scope(self, entity: str, query: str):
        if (self._pick_operation is not None or entity != "kettle" or
                not isinstance(query, str) or
                not any(word in query.lower() for word in ("kettle", "hot water")) or
                "kettle" not in self.gate.task.constraints.allowed_operated_entities):
            raise ValueError("invalid bounded E3 kettle-pick operation scope")
        self._pick_operation = (entity, query)
        try:
            yield
        finally:
            self._pick_operation = None

    def execute_tool(self, name: str, input_dict: dict[str, Any]):
        if name == "finish":
            result = self.base.execute_tool(name, input_dict)
            self.gate.end_round()
            self.adapter.set_commit_conditions(None)
            return result
        if name in OBSERVATION_OR_FILE_TOOLS:
            return self.base.execute_tool(name, input_dict)
        if name not in ROBOT_ACTIONS:
            return _tool_result(name, {
                "error": "robot action is not registered with LEASH",
                "code": "action_denied",
                "constraint": "use a registered RPent primitive",
            })
        try:
            workspace = self.workspace()
            self._pending_workspace = workspace
            proposal = proposal_for_e3(
                name, input_dict, workspace, self.locate_vla_target)
            if (self._pick_operation is not None and
                    workspace.evidence.robot.hold_estimate == "empty" and
                    name in {"move_to", "set_gripper"}):
                proposal = replace(
                    proposal, operated_entity="kettle",
                    parameters=(*proposal.parameters,
                                ("outer_macro", "public_visual_pick"),
                                ("grasp_intent", "bounded approach to visible kettle side handle, close and lift")),
                )
            decision = self.gate.check(proposal)
        except Exception as exc:
            self.proposals.append({"tool": name, "decision": "DENY",
                                   "reason": f"proposal unavailable: {type(exc).__name__}: {exc}"})
            return _tool_result(name, {
                "error": f"action proposal unavailable: {type(exc).__name__}",
                "code": "action_denied", "constraint": "refresh public perception",
            })
        finally:
            self._pending_workspace = None
        self.proposals.append({
            "tool": name, "arguments": dict(input_dict),
            "proposal": asdict(proposal), "decision": decision.result,
            "reason": decision.reason, "constraint": decision.constraint,
            "reviewer_called": bool(self.gate.logs and
                                    self.gate.logs[-1].reviewer_called),
        })
        if decision.result != "APPROVE":
            self.adapter.set_commit_conditions(None)
            return _tool_result(name, {
                "error": decision.reason, "code": "action_denied",
                "constraint": decision.constraint,
                "gate": 2, "decision": decision.result,
            })
        if not self.gate.is_current_lease(decision.lease):
            return _tool_result(name, {
                "error": "approved safety lease expired before execution",
                "code": "action_denied", "gate": 2,
            })
        self.adapter.set_commit_conditions(decision.lease.commit_conditions)
        self.adapter.last_outcome = None
        try:
            result = self.base.execute_tool(name, input_dict)
        finally:
            self.adapter.set_commit_conditions(None)
        outcome = self.adapter.last_outcome
        if outcome is not None and outcome.status == "intercepted":
            self.proposals[-1]["commit_outcome"] = asdict(outcome)
            return _tool_result(name, {
                "error": outcome.reason or "irreversible action blocked",
                "code": "commit_intercepted", "status": "intercepted",
                "gate": 3, "discarded_commands": outcome.discarded,
            })
        public = getattr(result, "result", {})
        if isinstance(public, dict) and "error" not in public:
            self.gate.on_primitive_completed(proposal)
        return result
