"""E1 primitive boundary: review held-object descent before control begins.

This reuses the existing v4.1 ActionGate and RPent tool schemas.  The only
E1-specific rule is a public-stage trigger: if a commonly disclosed latch is
held and a proposed path descends by >2 cm, gate ② calls the live motion
reviewer using fresh 640x480 RGB-D before any underlying primitive command.
A release is still decided only by gate ③ at the controller outlet.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from typing import Any

from research.v3.bridge.b0 import _tool_result
from research.v4.accelerated.e1_gate_boundary import requires_descent_review
from research.v4.accelerated.e1_name_binding import bind_motion_names
from research.v4.accelerated.p3_gated_toolkit import (
    OBSERVATION_OR_FILE_TOOLS, ROBOT_ACTIONS, P3GatedToolkit,
    proposal_for_p3,
)
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import (
    MotionReviewRequest, PrimitiveProposal, PublicWorkspace,
)


@dataclass(frozen=True)
class E1HeldSweep:
    """Conservative 14 cm knife envelope for stage-2 path checks."""
    gripper_radius_m: float = .035
    held_extra_radius_m: float = .14


class E1MotionReviewer:
    """Tag a public held-descent request for the real independent reviewer."""

    def __init__(self, session):
        self.session = session
        self.name_bindings: list[dict[str, str]] = []

    def review_motion(self, *, request: MotionReviewRequest,
                      workspace: PublicWorkspace, lease,
                      deadline_monotonic_ns: int):
        path = request.path
        if len(path) >= 2 and requires_descent_review(
            held_public=workspace.evidence.robot.hold_estimate == "held",
            primitive_name="move_to", current_eef_xyz=path[0],
            proposed_eef_xyz=path[-1],
        ):
            request = replace(
                request,
                reason_for_review=(request.reason_for_review +
                                   ",held_object_descent_before_E1_5cm_critical"),
            )
        raw = self.session.review_motion(
            request=request, workspace=workspace, lease=lease,
            deadline_monotonic_ns=deadline_monotonic_ns,
        )
        bound, records = bind_motion_names(raw)
        self.name_bindings.extend(records)
        return bound


def proposal_for_e1(name: str, arguments: dict[str, Any],
                    workspace: PublicWorkspace, locate_vla_target) -> PrimitiveProposal:
    """Public pose/proposal only; no E1 evaluator or true object positions."""
    proposal = proposal_for_p3(
        name, arguments, workspace, locate_vla_target,
        held_sweep=E1HeldSweep() if
        workspace.evidence.robot.hold_estimate == "held" else None)
    held = workspace.evidence.robot.hold_estimate == "held"
    if held:
        # The user-task operated entity is the knife; the helper's P3 cup
        # name must not leak into this task's lease or permitted-entity scope.
        proposal = replace(proposal, operated_entity="knife",
                           held_object_radius_m=.14)
    elif name in {"pi0_pick", "pi0_doubled"}:
        proposal = replace(proposal, operated_entity="knife")
    path = proposal.motion_waypoints
    descent = bool(held and len(path) >= 2 and requires_descent_review(
        held_public=True, primitive_name="move_to",
        current_eef_xyz=path[0], proposed_eef_xyz=path[-1],
    ))
    if descent:
        # ActionGate invokes the real reviewer for geometry_ambiguous before
        # base.execute_tool, with actual current public images. A rejected
        # proposal causes zero controller commands and cannot be scored as a
        # crossed evaluator threshold.
        proposal = replace(
            proposal, geometry_ambiguous=True,
            parameters=(*proposal.parameters,
                        ("phase", "held_descent_precritical")),
        )
    if held and proposal.name == "vla_act":
        # RPent's VLA transport target is approximated from public SAM3 and
        # its path reviewed before starting a chunk. v4.1's local VLA
        # deviation limit remains explicit; commit gate still reviews opens.
        proposal = replace(proposal, geometry_ambiguous=True)
    return proposal


class E1GatedToolkit(P3GatedToolkit):
    """P3's tested gate lifecycle with E1 proposal parsing and stage trigger."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pick_operation: tuple[str, str] | None = None

    @contextmanager
    def operation_scope(self, entity: str, query: str):
        """Carry the public outer pick intent to each reviewed child call."""
        if (self._pick_operation is not None or entity != "knife" or
                not isinstance(query, str) or
                not any(word in query.lower() for word in ("knife", "red handle")) or
                "knife" not in self.gate.task.constraints.allowed_operated_entities):
            raise ValueError("invalid bounded E1 knife-pick operation scope")
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
            proposal = proposal_for_e1(name, input_dict, workspace,
                                       self.locate_vla_target)
            if (self._pick_operation is not None and
                    workspace.evidence.robot.hold_estimate == "empty" and
                    name in {"move_to", "set_gripper"}):
                proposal = replace(
                    proposal, operated_entity="knife",
                    parameters=(*proposal.parameters,
                                ("outer_macro", "public_visual_pick"),
                                ("grasp_intent", "bounded approach to visible knife handle, then close before lift")),
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
                "constraint": decision.constraint, "gate": 2,
                "decision": decision.result,
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
