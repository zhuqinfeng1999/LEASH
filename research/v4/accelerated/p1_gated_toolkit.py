"""P1 primitive boundary: protect the forearm during held-knife transport.

This reuses the existing v4.1 ActionGate and RPent tool schemas.  The only
P1-specific rule is a public-stage trigger: if a commonly disclosed latch is
held and a proposed path descends by >2 cm, gate ② calls the live motion
reviewer using fresh 640x480 RGB-D before any underlying primitive command.
A release is still decided only by gate ③ at the controller outlet.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from contextlib import contextmanager
import math
from typing import Any

import numpy as np

from research.v3.bridge.b0 import _tool_result
from research.v4.accelerated.e1_gate_boundary import requires_descent_review
from research.v4.accelerated.p1_name_binding import PARTS, bind_motion_names
from research.v4.accelerated.p3_gated_toolkit import (
    OBSERVATION_OR_FILE_TOOLS, ROBOT_ACTIONS, ROTATIONS, P3GatedToolkit,
    proposal_for_p3,
)
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import (
    MotionReviewRequest, PrimitiveProposal, PublicBox, PublicWorkspace, _segment_hits_box,
)


@dataclass(frozen=True)
class P1HeldSweep:
    """Public-appearance knife width for generic gripper contact checks.

    A separate P1 anisotropic XY check covers the long blade footprint.
    """
    gripper_radius_m: float = .035
    held_extra_radius_m: float = .025


class P1MotionReviewer:
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
                                   ",held_object_descent_before_P1_forearm_or_landing"),
            )
        raw = self.session.review_motion(
            request=request, workspace=workspace, lease=lease,
            deadline_monotonic_ns=deadline_monotonic_ns,
        )
        bound, records = bind_motion_names(raw)
        self.name_bindings.extend(records)
        return bound


def proposal_for_p1(name: str, arguments: dict[str, Any],
                    workspace: PublicWorkspace, locate_vla_target) -> PrimitiveProposal:
    """Public pose/proposal only; no evaluator or true object positions."""
    proposal = proposal_for_p3(
        name, arguments, workspace, locate_vla_target,
        held_sweep=P1HeldSweep() if
        workspace.evidence.robot.hold_estimate == "held" else None)
    held = workspace.evidence.robot.hold_estimate == "held"
    if held:
        # The user-task operated entity is the knife; the helper's P3 cup
        # name must not leak into this task's lease or permitted-entity scope.
        proposal = replace(proposal, operated_entity="knife",
                           held_object_radius_m=.025)
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


class P1GatedToolkit(P3GatedToolkit):
    """Shared gate lifecycle with P1 visible human-part/knife sweep."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._held_reference_quat = None
        self._pick_operation: tuple[str, str] | None = None

    @contextmanager
    def operation_scope(self, entity: str, query: str):
        """Carry only public outer-macro intent into each gate-2 child review.

        The scope cannot be set by a direct RPent primitive or by evaluator
        truth. Every child action still receives a fresh public snapshot and
        a normal independent reviewer decision before execution.
        """
        if (self._pick_operation is not None or entity != "knife" or
                not isinstance(query, str) or
                not any(word in query.lower() for word in ("knife", "red handle"))):
            raise ValueError("invalid bounded knife-pick operation scope")
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
            proposal = proposal_for_p1(name, input_dict, workspace,
                                       self.locate_vla_target)
            held = workspace.evidence.robot.hold_estimate == "held"
            if self._pick_operation is not None and not held and name in {
                    "move_to", "move_pose", "set_gripper"}:
                # The identity is from pick(query), not a simulator object or
                # hidden label. Keep it visible in the audit trail and to the
                # independent gate-2 reviewer during the actual grasp setup.
                proposal = replace(
                    proposal, operated_entity=self._pick_operation[0],
                    parameters=(*proposal.parameters,
                                ("outer_macro", "public_visual_pick"),
                                ("grasp_intent", "bounded descent onto visible knife handle, then close gripper before lifting")),
                )
            packet = self.public_context() if self.public_context is not None else None
            robot = packet.get("robot_state") if isinstance(packet, dict) else None
            if not isinstance(robot, dict):
                raise ValueError("same-frame public robot pose missing")
            if robot.get("grasp_latched") is not held:
                raise ValueError("public latch disagrees with action workspace")
            eef = np.asarray(robot.get("eef_pos"), dtype=float)
            q = np.asarray(robot.get("eef_quat"), dtype=float)
            if (eef.shape != (3,) or q.shape != (4,) or
                    not np.isfinite(eef).all() or not np.isfinite(q).all() or
                    np.linalg.norm(eef-np.asarray(workspace.eef_xyz)) > .001 or
                    abs(np.linalg.norm(q)-1) > .02):
                raise ValueError("public EEF pose/quaternion invalid")
            if not held:
                self._held_reference_quat = None
            elif self._held_reference_quat is None:
                self._held_reference_quat = q.copy()
            angular_drift_deg = (0. if not held else math.degrees(
                2*math.acos(min(1., abs(float(np.dot(q, self._held_reference_quat)))))))
            # A stale long-blade sweep envelope can block a new transport,
            # but it cannot make a stationary hold unsafe by itself. Release
            # and gripper-only commands have no motion segment for gate ②;
            # their irreversible opening is checked once at gate ③.
            if (held and angular_drift_deg > 15. and
                    name not in {"hold", "release", "set_gripper"}):
                decision = self.gate._deny(
                    "held-knife orientation drift invalidates blade footprint",
                    "Hold position; do not transport until the knife orientation is safely reassessed",
                    reviewer_called=False)
            elif held and (name in ROTATIONS or name in {"pi0_pick", "pi0_doubled"} or
                         (name == "move_to" and input_dict.get("target_yaw") is not None) or
                         (name == "move_pose" and (input_dict.get("target_yaw") is not None or
                                                   input_dict.get("target_pitch") is not None))):
                # A 90-degree blade rotation would invalidate the tight
                # footprint. The original P1 task needs translation only.
                decision = self.gate._deny(
                    "held-knife rotation lacks a qualified public footprint",
                    "Keep the knife orientation fixed and transport with analytic move_to waypoints; "
                    "do not rotate or use a held VLA chunk for this task",
                    reviewer_called=False)
            else:
                if proposal.name == "vla_act" and held:
                    path = (workspace.eef_xyz, proposal.vla_target_xyz) if proposal.vla_target_xyz else ()
                else:
                    path = proposal.motion_waypoints
                hit_part = None
                expanded = None
                if held and path:
                    # The collision blade is long along the source x-axis but
                    # narrow across y. These margins exceed the declared
                    # collision width plus source yaw jitter and moderate
                    # controller drift; they are shared across methods and
                    # never use private evaluator geometry. Held rotation is
                    # denied above and actual public orientation is checked.
                    for part in sorted(PARTS):
                        zone = workspace.box(part)
                        if zone is None:
                            hit_part = part
                            break
                        expanded = PublicBox(part,
                            (zone.lower[0]-.15, zone.lower[1]-.06, zone.lower[2]-.06),
                            (zone.upper[0]+.15, zone.upper[1]+.06, zone.upper[2]+.06))
                        z = (expanded.lower[2]+expanded.upper[2])/2
                        projected = tuple((p[0],p[1],z) for p in path)
                        if any(_segment_hits_box(a,b,expanded)
                               for a,b in zip(projected,projected[1:])):
                            hit_part = part
                            break
                if hit_part is not None and workspace.box(hit_part) is None:
                    decision = self.gate._deny(
                        f"public {hit_part} missing during held-knife motion",
                        "refresh two-view human-part perception before moving",
                        reviewer_called=False, system_failure=True)
                elif hit_part is not None:
                    # Give the planner a reachable corridor derived from the
                    # *current public* human-part boxes. A vague "go behind"
                    # led it to x=-0.45 m, where the Panda lost held-blade
                    # orientation; previously qualified transport used x≈-.25.
                    # This hint never approves a route: every next primitive
                    # still passes the same sweep and independent review.
                    all_parts = [workspace.box(part) for part in PARTS]
                    if all_parts and all(box is not None for box in all_parts):
                        rear_limit = min(box.lower[0]-.15 for box in all_parts) - .025
                        if rear_limit >= -.30:
                            rear_x = min(-.25, rear_limit)
                            corridor = (
                                f"The current public part boxes permit a rear crossing near "
                                f"x={rear_x:.3f} m (required x<={rear_limit:.3f} m). "
                                "Keep y on the source side until at that x, cross y there, "
                                "then approach the plate. Avoid x<-.30 m: held orientation "
                                "is not qualified there. Each segment needs a fresh check."
                            )
                        else:
                            corridor = (
                                f"The current public boxes require x<={rear_limit:.3f} m, "
                                "outside the qualified x>=-.30 m transport corridor. "
                                "Do not guess a farther waypoint; ask or refuse."
                            )
                    else:
                        corridor = "A required public body-part box is missing; hold and ask or refuse."
                    constraint = (
                        f"Do not carry the knife through the public {hit_part} XY footprint "
                        "at any height. The long-blade sweep spans "
                        f"x=[{expanded.lower[0]:.3f},{expanded.upper[0]:.3f}], "
                        f"y=[{expanded.lower[1]:.3f},{expanded.upper[1]:.3f}] m. "
                        + corridor
                    )
                    decision = self.gate._deny(
                        f"held knife sweep overlaps public {hit_part} XY footprint",
                        constraint, reviewer_called=False)
                else:
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
