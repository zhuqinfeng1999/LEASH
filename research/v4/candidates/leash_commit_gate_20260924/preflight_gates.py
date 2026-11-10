"""Isolated v4.1 task/action gate candidate; no simulator or evaluator imports.

This module is intentionally not wired into RPent. The only inputs are user
channel text, agent-visible RGB-D/keyframes, public kinematics, and named
perception boxes. All controller commands remain outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from .commit_gate import CameraFrame, PublicEvidence


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class UserInstruction:
    text: str
    channel: Literal["user"] = "user"


@dataclass(frozen=True)
class TurnHistory:
    instruction: UserInstruction
    primitive_summaries: tuple[str, ...]
    starting_agentview: CameraFrame


@dataclass(frozen=True)
class PublicBox:
    name: str
    lower: Vec3
    upper: Vec3
    source: Literal["public_rgbd_perception"] = "public_rgbd_perception"

    def expanded(self, amount: float) -> PublicBox:
        return PublicBox(
            self.name,
            tuple(x - amount for x in self.lower),
            tuple(x + amount for x in self.upper),
        )

    def contains(self, point: Vec3) -> bool:
        return all(lo <= value <= hi for lo, value, hi in zip(self.lower, point, self.upper))


@dataclass(frozen=True)
class PublicWorkspace:
    evidence: PublicEvidence
    eef_xyz: Vec3
    visible_entities: tuple[PublicBox, ...]
    source: Literal["public_observation"] = "public_observation"

    def box(self, name: str) -> PublicBox | None:
        return next((item for item in self.visible_entities if item.name == name), None)


@dataclass(frozen=True)
class OrientationRule:
    operated_entity: str
    required_yaw_deg: float
    tolerance_deg: float


@dataclass(frozen=True)
class TaskConstraints:
    allowed_operated_entities: frozenset[str] = frozenset()
    forbidden_entities: frozenset[str] = frozenset()
    orientation_rules: tuple[OrientationRule, ...] = ()
    commit_conditions: tuple[str, ...] = ("default home-safety policy",)


@dataclass(frozen=True)
class TaskGateDecision:
    result: Literal["PROCEED", "REFUSE", "ASK"]
    reason: str
    constraints: TaskConstraints
    system_failure: bool


@dataclass(frozen=True)
class PreGateLog:
    gate: Literal[1, 2]
    boundary: Literal["user_to_planner", "planner_to_executor"]
    trigger: str
    reviewer_called: bool
    review_latency_s: float
    result: str
    reason: str
    system_failure: bool


@dataclass(frozen=True)
class PrimitiveProposal:
    name: str
    parameters: tuple[tuple[str, str], ...] = ()
    operated_entity: str | None = None
    # Analytic primitive: entire planned public-space path including start.
    motion_waypoints: tuple[Vec3, ...] = ()
    # vla_act: public/perceptual target-region center; transport is straight.
    vla_target_xyz: Vec3 | None = None
    gripper_radius_m: float = 0.025
    held_object_radius_m: float = 0.0
    placement_xyz: Vec3 | None = None
    placement_yaw_deg: float | None = None
    changes_grasp_state: Literal["grasp", "release"] | None = None
    geometry_ambiguous: bool = False


@dataclass(frozen=True)
class MotionReviewRequest:
    """Only the motion portion; no release/open/pour/door permission field."""

    path: tuple[Vec3, ...]
    operated_entity: str | None
    placement_xyz: Vec3 | None
    placement_yaw_deg: float | None
    reason_for_review: str
    # Public planner/macro intent is context for the independent reviewer,
    # never an automatic permission or evaluator truth.
    primitive_name: str | None = None
    parameters: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SafetyLease:
    round_id: str
    primitive_name: str
    parameters: tuple[tuple[str, str], ...]
    forbidden_zones: tuple[PublicBox, ...]
    registered_entities: frozenset[str]
    orientation_rules: tuple[OrientationRule, ...]
    commit_conditions: tuple[str, ...]


@dataclass(frozen=True)
class ActionDecision:
    result: Literal["APPROVE", "DENY", "FINISH_REFUSED"]
    reason: str
    constraint: str | None
    lease: SafetyLease | None
    system_failure: bool


class TaskReviewer(Protocol):
    def review_task(
        self,
        *,
        round_id: str,
        instruction: UserInstruction,
        current: PublicEvidence,
        history: tuple[TurnHistory, ...],
        deadline_monotonic_ns: int,
    ) -> Mapping[str, Any]: ...


class MotionReviewer(Protocol):
    def review_motion(
        self,
        *,
        request: MotionReviewRequest,
        workspace: PublicWorkspace,
        lease: SafetyLease | None,
        deadline_monotonic_ns: int,
    ) -> Mapping[str, Any]: ...


Finish = Callable[[str, str], None]
Snapshot = Callable[[], PublicWorkspace]


def _parse_constraints(
    raw: Any, *, default_commit_conditions: tuple[str, ...] = ("default home-safety policy",)
) -> TaskConstraints:
    if raw is None:
        return TaskConstraints(commit_conditions=default_commit_conditions)
    if not isinstance(raw, Mapping):
        raise ValueError("task constraints are not a JSON object")
    def names(key: str) -> frozenset[str]:
        value = raw.get(key, [])
        if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
            raise ValueError(f"{key} must be a list of names")
        return frozenset(value)
    orientation_raw = raw.get("orientation_rules", [])
    if not isinstance(orientation_raw, list):
        raise ValueError("orientation rules must be a list")
    orientation_rules: list[OrientationRule] = []
    for item in orientation_raw:
        if not isinstance(item, Mapping):
            raise ValueError("orientation rule is not an object")
        entity = item.get("operated_entity")
        yaw = item.get("required_yaw_deg")
        tolerance = item.get("tolerance_deg")
        if not isinstance(entity, str) or not entity.strip() or not isinstance(yaw, (int, float)) or not isinstance(tolerance, (int, float)) or not math.isfinite(yaw) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("orientation rule malformed")
        orientation_rules.append(OrientationRule(entity, float(yaw), float(tolerance)))
    conditions = raw.get("commit_conditions", [])
    if not isinstance(conditions, list) or any(not isinstance(x, str) or not x.strip() for x in conditions):
        raise ValueError("task commit conditions malformed")
    # v4.1 permits an empty list of additional task conditions. Preserve a
    # nonempty default lease so gate 3 still reviews every irreversible step.
    if not conditions:
        conditions = list(default_commit_conditions)
    return TaskConstraints(
        allowed_operated_entities=names("allowed_operated_entities"),
        forbidden_entities=names("forbidden_entities"),
        orientation_rules=tuple(orientation_rules),
        commit_conditions=tuple(conditions),
    )


def _valid_vec(point: Vec3) -> bool:
    return len(point) == 3 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in point)


def _segment_hits_box(start: Vec3, end: Vec3, box: PublicBox) -> bool:
    """Exact segment/AABB intersection after caller expands the swept radius."""

    t_min, t_max = 0.0, 1.0
    for p, q, lo, hi in zip(start, end, box.lower, box.upper):
        delta = q - p
        if abs(delta) < 1e-12:
            if p < lo or p > hi:
                return False
            continue
        near, far = sorted(((lo - p) / delta, (hi - p) / delta))
        t_min, t_max = max(t_min, near), min(t_max, far)
        if t_min > t_max:
            return False
    return True


def _yaw_error(actual: float, required: float) -> float:
    return abs((actual - required + 180.0) % 360.0 - 180.0)


class TaskGate:
    """Once per user turn, before any planner invocation."""

    def __init__(
        self, reviewer: TaskReviewer, *, timeout_s: float = 15.0,
        default_commit_conditions: tuple[str, ...] = ("default home-safety policy",),
    ) -> None:
        if timeout_s <= 0 or not math.isfinite(timeout_s):
            raise ValueError("task review timeout must be finite and positive")
        if not default_commit_conditions or any(
            not isinstance(x, str) or not x.strip() for x in default_commit_conditions
        ):
            raise ValueError("task default commit conditions must be nonempty")
        self.reviewer = reviewer
        self.default_commit_conditions = default_commit_conditions
        self.timeout_ns = int(timeout_s * 1e9)
        self.logs: list[PreGateLog] = []
        self._round_decisions: dict[str, tuple[str, TaskGateDecision]] = {}

    def begin_round(
        self,
        *,
        round_id: str,
        instruction: UserInstruction,
        current: PublicEvidence,
        history: Sequence[TurnHistory],
        finish: Finish,
    ) -> TaskGateDecision:
        if not isinstance(round_id, str) or not round_id.strip():
            raise ValueError("round ID missing")
        if not isinstance(instruction, UserInstruction) or instruction.channel != "user" or not instruction.text.strip():
            raise ValueError("only nonempty user-channel text can be a task instruction")
        prior = self._round_decisions.get(round_id)
        if prior is not None:
            if prior[0] != instruction.text:
                raise ValueError("round ID reused for a different user instruction")
            return prior[1]
        evidence_ok = (
            isinstance(current, PublicEvidence)
            and getattr(current.agentview, "name", None) == "agentview"
            and getattr(current.robot0_eye_in_hand, "name", None) == "robot0_eye_in_hand"
            and getattr(current.robot, "source", None) == "public_observation"
        )
        history_ok = all(
            isinstance(item, TurnHistory)
            and isinstance(item.instruction, UserInstruction)
            and item.instruction.channel == "user"
            and getattr(item.starting_agentview, "name", None) == "agentview"
            for item in history
        )
        if not evidence_ok or not history_ok:
            reason = "task evidence/history missing or malformed"
            decision = TaskGateDecision("REFUSE", reason, TaskConstraints(), True)
            self._round_decisions[round_id] = (instruction.text, decision)
            self.logs.append(PreGateLog(
                1, "user_to_planner", "new_user_turn", False, 0.0,
                "REFUSE", reason, True,
            ))
            finish("refused", reason)
            return decision
        started = time.monotonic_ns()
        deadline = started + self.timeout_ns
        failure = False
        try:
            raw = self.reviewer.review_task(
                round_id=round_id,
                instruction=instruction,
                current=current,
                history=tuple(history),
                deadline_monotonic_ns=deadline,
            )
            elapsed = (time.monotonic_ns() - started) / 1e9
            if time.monotonic_ns() > deadline:
                raise TimeoutError("task reviewer deadline exceeded")
            if not isinstance(raw, Mapping):
                raise ValueError("task reviewer response is not an object")
            result = raw.get("decision")
            reason = raw.get("reason")
            if result not in ("PROCEED", "REFUSE", "ASK") or not isinstance(reason, str) or not reason.strip():
                raise ValueError("invalid task reviewer decision/reason")
            # A refusal/ask ends this round before a lease exists.  The
            # reviewer's constraint object is meaningful only for PROCEED;
            # do not turn an otherwise valid safety refusal into an
            # infrastructure fault because of unused lease fields.
            constraints = _parse_constraints(
                raw.get("constraints"),
                default_commit_conditions=self.default_commit_conditions,
            ) if result == "PROCEED" else TaskConstraints()
            if result == "PROCEED" and not constraints.allowed_operated_entities:
                raise ValueError("PROCEED requires an explicit user-task entity scope")
        except Exception as exc:
            elapsed = (time.monotonic_ns() - started) / 1e9
            result, reason, constraints = "REFUSE", f"task review failed: {type(exc).__name__}: {exc}", TaskConstraints()
            failure = True
        decision = TaskGateDecision(result, reason, constraints, failure)
        self._round_decisions[round_id] = (instruction.text, decision)
        self.logs.append(PreGateLog(1, "user_to_planner", "new_user_turn", True, elapsed, result, reason, failure))
        if result in ("REFUSE", "ASK"):
            finish("refused" if result == "REFUSE" else "asked", reason)
        return decision


class ActionGate:
    """Checks proposed primitive movement before any executor/controller action."""

    def __init__(
        self,
        *,
        round_id: str,
        task: TaskGateDecision,
        snapshot: Snapshot,
        reviewer: MotionReviewer,
        finish: Finish,
        timeout_s: float = 15.0,
    ) -> None:
        if timeout_s <= 0 or not math.isfinite(timeout_s):
            raise ValueError("motion review timeout must be finite and positive")
        self.round_id = round_id
        self.task = task
        self.snapshot = snapshot
        self.reviewer = reviewer
        self.finish = finish
        self.timeout_ns = int(timeout_s * 1e9)
        self.denials = 0
        self.first_motion_pending = True
        self.after_grasp_or_release = False
        self.lease: SafetyLease | None = None
        self.round_open = True
        self.logs: list[PreGateLog] = []

    def is_current_lease(self, lease: SafetyLease | None) -> bool:
        """An old returned lease object never grants permission after update/expiry."""
        return self.round_open and lease is not None and self.lease is lease

    def end_round(self) -> None:
        self.round_open = False
        self.lease = None

    def on_holding_end(self) -> None:
        self.lease = None

    def on_primitive_completed(self, proposal: PrimitiveProposal) -> None:
        if proposal.changes_grasp_state in ("grasp", "release"):
            self.after_grasp_or_release = True
        if proposal.changes_grasp_state == "release":
            self.on_holding_end()

    def _deny(self, reason: str, constraint: str, *, reviewer_called: bool, latency: float = 0.0, system_failure: bool = False) -> ActionDecision:
        self.denials += 1
        if self.denials > 5:
            self.finish("refused", "action gate exceeded five DENY responses")
            result: Literal["APPROVE", "DENY", "FINISH_REFUSED"] = "FINISH_REFUSED"
            self.round_open = False
            self.lease = None
        else:
            result = "DENY"
        self.logs.append(PreGateLog(2, "planner_to_executor", "primitive_proposal", reviewer_called, latency, result, reason, system_failure))
        return ActionDecision(result, reason, constraint, None, system_failure)

    def _zones(self, workspace: PublicWorkspace) -> tuple[PublicBox, ...] | None:
        names = set(self.task.constraints.forbidden_entities)
        if self.lease is not None:
            names.update(self.lease.registered_entities)
        boxes = tuple(workspace.box(name) for name in sorted(names))
        if any(box is None for box in boxes):
            return None
        return tuple(box for box in boxes if box is not None)

    def check(self, proposal: PrimitiveProposal) -> ActionDecision:
        if not self.round_open:
            return ActionDecision("FINISH_REFUSED", "round is closed", "do not execute", None, False)
        if self.task.result != "PROCEED":
            return self._deny("round was not approved by task gate", "do not execute", reviewer_called=False)
        if not isinstance(proposal, PrimitiveProposal):
            return self._deny("primitive proposal malformed", "submit a parsed primitive", reviewer_called=False, system_failure=True)
        if not isinstance(proposal.name, str) or not proposal.name.strip():
            return self._deny("primitive name missing", "submit a parsed primitive", reviewer_called=False, system_failure=True)
        try:
            workspace = self.snapshot()
        except Exception as exc:
            return self._deny(f"public snapshot failed: {type(exc).__name__}", "refresh public observation", reviewer_called=False, system_failure=True)
        if not isinstance(workspace, PublicWorkspace) or workspace.source != "public_observation":
            return self._deny("public workspace unavailable", "refresh public observation", reviewer_called=False, system_failure=True)
        if not _valid_vec(workspace.eef_xyz):
            return self._deny("public end-effector pose invalid", "refresh proprioception", reviewer_called=False, system_failure=True)
        if not isinstance(workspace.evidence, PublicEvidence) or any(
            box.source != "public_rgbd_perception"
            or not box.name
            or not _valid_vec(box.lower)
            or not _valid_vec(box.upper)
            or any(lo > hi for lo, hi in zip(box.lower, box.upper))
            for box in workspace.visible_entities
        ):
            return self._deny("public RGB-D perception box invalid", "refresh perception", reviewer_called=False, system_failure=True)
        if proposal.operated_entity is not None and self.task.constraints.allowed_operated_entities:
            if proposal.operated_entity not in self.task.constraints.allowed_operated_entities:
                return self._deny(
                    "proposal operates entity outside user-task scope",
                    "operate only entities needed for the user instruction",
                    reviewer_called=False,
                )
        zones = self._zones(workspace)
        if zones is None:
            return self._deny("named forbidden entity not located in public RGB-D", "refresh perception before moving", reviewer_called=False, system_failure=True)
        if proposal.gripper_radius_m < 0 or proposal.held_object_radius_m < 0:
            return self._deny("invalid swept volume radius", "correct primitive geometry", reviewer_called=False, system_failure=True)
        swept_radius = proposal.gripper_radius_m + proposal.held_object_radius_m
        if proposal.name == "vla_act":
            if proposal.vla_target_xyz is None or not _valid_vec(proposal.vla_target_xyz):
                return self._deny(
                    "vla_act transport target cannot be located",
                    "locate the target region before vla_act",
                    reviewer_called=False,
                    system_failure=True,
                )
            path = (workspace.eef_xyz, proposal.vla_target_xyz)
        else:
            path = proposal.motion_waypoints
            if not path and proposal.name not in ("release", "set_gripper", "hold", "finish"):
                return self._deny("moving primitive lacks a full predicted path", "submit full predicted waypoints", reviewer_called=False, system_failure=True)
            if path and (len(path) < 2 or any(not _valid_vec(x) for x in path)):
                return self._deny("analytic path malformed", "submit full predicted waypoints", reviewer_called=False, system_failure=True)
            if path and math.dist(path[0], workspace.eef_xyz) > 0.01:
                return self._deny("analytic path does not start at current robot pose", "replan from current pose", reviewer_called=False, system_failure=True)
        for zone in zones:
            expanded = zone.expanded(swept_radius)
            for start, end in zip(path, path[1:]):
                if _segment_hits_box(start, end, expanded):
                    constraint = (
                        "First use analytic primitives to route around the forbidden zone "
                        "and move near the target; use vla_act only for the final local step."
                        if proposal.name == "vla_act"
                        else (
                            f"The public RGB-D {zone.name} forbidden box, including the "
                            f"gripper and held-object sweep, spans x=[{expanded.lower[0]:.3f}, "
                            f"{expanded.upper[0]:.3f}], y=[{expanded.lower[1]:.3f}, "
                            f"{expanded.upper[1]:.3f}], z=[{expanded.lower[2]:.3f}, "
                            f"{expanded.upper[2]:.3f}] metres. Replan the complete "
                            "straight segment, not just its endpoint. Keep the carried "
                            "center outside this box: while still before its near y edge, "
                            f"move left of x={expanded.lower[0]-.02:.3f} or right of "
                            f"x={expanded.upper[0]+.02:.3f}; cross its y interval "
                            "on that clear side, then approach the target only after "
                            f"passing y={expanded.upper[1]+.02:.3f}. Check robot "
                            "reachability of each suggested waypoint."
                        )
                    )
                    return self._deny(f"swept path intersects forbidden {zone.name}", constraint, reviewer_called=False)
            if proposal.placement_xyz is not None and expanded.contains(proposal.placement_xyz):
                return self._deny(f"placement lands inside forbidden {zone.name}", "choose a safe landing point", reviewer_called=False)
        rules = self.lease.orientation_rules if self.lease is not None else self.task.constraints.orientation_rules
        for rule in rules:
            if rule.operated_entity == proposal.operated_entity and proposal.placement_xyz is not None:
                if proposal.placement_yaw_deg is None or _yaw_error(proposal.placement_yaw_deg, rule.required_yaw_deg) > rule.tolerance_deg:
                    return self._deny("placement orientation violates task rule", "orient the grasp end toward the recipient", reviewer_called=False)
        # A zero-motion release has no movement for gate ② to review. Preserve
        # first-motion review for the next *moving* call; gate ③ alone decides
        # whether this release may open the held gripper.
        # Rotation at a stationary end-effector is still a swept movement.
        has_motion = bool(path)
        reasons: list[str] = []
        if has_motion and self.first_motion_pending:
            reasons.append("first_motion")
        if has_motion and proposal.geometry_ambiguous:
            reasons.append("geometry_ambiguous")
        if has_motion and self.after_grasp_or_release:
            reasons.append("after_grasp_or_release")
        if has_motion and self.lease is not None and proposal.operated_entity in self.lease.registered_entities:
            reasons.append("lease_entity")
        reviewer_called = bool(reasons)
        latency = 0.0
        extra_forbidden: frozenset[str] = frozenset()
        extra_conditions: tuple[str, ...] = ()
        if reviewer_called:
            started = time.monotonic_ns()
            try:
                raw = self.reviewer.review_motion(
                    request=MotionReviewRequest(
                        path, proposal.operated_entity, proposal.placement_xyz,
                        proposal.placement_yaw_deg, ",".join(reasons),
                        primitive_name=proposal.name, parameters=proposal.parameters),
                    workspace=workspace,
                    lease=self.lease,
                    deadline_monotonic_ns=started + self.timeout_ns,
                )
                latency = (time.monotonic_ns() - started) / 1e9
                if time.monotonic_ns() > started + self.timeout_ns:
                    raise TimeoutError("motion review deadline exceeded")
                if not isinstance(raw, Mapping) or raw.get("decision") not in ("APPROVE", "DENY") or not isinstance(raw.get("reason"), str) or not raw["reason"].strip():
                    raise ValueError("motion review response malformed")
                forbidden_raw = raw.get("forbidden_entities", [])
                conditions_raw = raw.get("commit_conditions", [])
                if not isinstance(forbidden_raw, list) or not isinstance(conditions_raw, list):
                    raise ValueError("motion constraints malformed")
                if any(not isinstance(x, str) or not x.strip() for x in (*forbidden_raw, *conditions_raw)):
                    raise ValueError("motion constraints malformed")
                extra_forbidden = frozenset(forbidden_raw)
                extra_conditions = tuple(conditions_raw)
                if raw["decision"] == "DENY":
                    return self._deny(raw["reason"], str(raw.get("constraint", "replan safely")), reviewer_called=True, latency=latency)
            except Exception as exc:
                latency = (time.monotonic_ns() - started) / 1e9
                return self._deny(f"motion review failed: {type(exc).__name__}", "retry after valid review", reviewer_called=True, latency=latency, system_failure=True)
        names = frozenset(self.task.constraints.forbidden_entities | extra_forbidden)
        if self.lease is not None:
            names |= self.lease.registered_entities
        all_boxes = tuple(workspace.box(name) for name in sorted(names))
        if any(box is None for box in all_boxes):
            return self._deny("reviewer-named forbidden entity not located", "refresh public perception", reviewer_called=reviewer_called, latency=latency, system_failure=True)
        # Newly added zones from the reviewer must be checked before approval.
        for zone in all_boxes:
            if zone is None or zone.name not in extra_forbidden:
                continue
            expanded = zone.expanded(swept_radius)
            if any(_segment_hits_box(a, b, expanded) for a, b in zip(path, path[1:])):
                return self._deny(f"swept path intersects newly forbidden {zone.name}", "replan around new forbidden zone", reviewer_called=reviewer_called, latency=latency)
            if proposal.placement_xyz is not None and expanded.contains(proposal.placement_xyz):
                return self._deny(f"placement lands inside newly forbidden {zone.name}", "choose a safe landing point", reviewer_called=reviewer_called, latency=latency)
        existing_conditions = self.lease.commit_conditions if self.lease is not None else self.task.constraints.commit_conditions
        lease = SafetyLease(
            round_id=self.round_id,
            primitive_name=proposal.name,
            parameters=proposal.parameters,
            forbidden_zones=tuple(box for box in all_boxes if box is not None),
            registered_entities=names,
            orientation_rules=rules,
            commit_conditions=existing_conditions + extra_conditions,
        )
        self.lease = lease
        if has_motion:
            self.first_motion_pending = False
            self.after_grasp_or_release = False
        self.logs.append(PreGateLog(2, "planner_to_executor", ",".join(reasons) or "geometry_only", reviewer_called, latency, "APPROVE", "motion permitted; commits deferred to gate 3", False))
        return ActionDecision("APPROVE", "motion permitted; commits deferred to gate 3", None, lease, False)


class GuardedExecutor:
    """Testable executor boundary: no task/action approval means zero calls."""

    def __init__(self, gate: ActionGate, execute_primitive: Callable[[PrimitiveProposal], Any]) -> None:
        self.gate = gate
        self.execute_primitive = execute_primitive

    def submit(self, proposal: PrimitiveProposal) -> ActionDecision:
        decision = self.gate.check(proposal)
        if decision.result == "APPROVE":
            self.execute_primitive(proposal)
        return decision
