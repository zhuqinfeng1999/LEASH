"""Isolated v4.1 candidate for gate ③ at the sole controller outlet.

This module deliberately knows nothing about MuJoCo evaluator state, RPent
primitive names, or the policy model. A production adapter must normalize
*every* primitive's controller instruction into ``LowLevelInstruction`` and
provide only public camera/proprioception/perception data to ``capture``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence


GateTrigger = Literal["held_gripper_open", "held_payload_tilt", "heated_door_close"]


@dataclass(frozen=True)
class PublicRobotState:
    """State available to the agent, including a fallible public hold estimate."""

    # Observed aperture/closure hint only. Panda fingers remain separated
    # around a held object; this flag must never waive an open-command review.
    gripper_closed: bool
    hold_estimate: Literal["held", "empty", "unknown"]
    held_object_name: str | None = None
    held_object_tiltable: bool | None = None
    held_object_tilt_deg: float | None = None
    observed_monotonic_ns: int = field(default_factory=time.monotonic_ns)
    source: Literal["public_observation"] = "public_observation"


@dataclass(frozen=True)
class LowLevelInstruction:
    """One controller instruction after primitive-specific normalization.

    ``payload`` is opaque to this gate and is forwarded unchanged. If the
    instruction changes a held object's orientation, the adapter must set
    ``changes_payload_orientation`` and predict its post-command tilt using
    public kinematics. If that prediction is unavailable, the gate blocks.
    """

    payload: Any
    gripper_target: Literal["open", "closed"] | None = None
    changes_payload_orientation: bool = False
    proposed_payload_tilt_deg: float | None = None
    heated_device_door_target: Literal["close", "open"] | None = None
    heated_device_name: str | None = None


@dataclass(frozen=True)
class PublicPendingAction:
    """Reviewer-visible action metadata; never contains opaque controller payload."""

    gripper_target: Literal["open", "closed"] | None
    changes_payload_orientation: bool
    proposed_payload_tilt_deg: float | None
    heated_device_door_target: Literal["close", "open"] | None
    heated_device_name: str | None


@dataclass(frozen=True)
class CameraFrame:
    name: Literal["agentview", "robot0_eye_in_hand"]
    rgb: bytes
    depth: bytes
    width: int
    height: int
    intrinsics: Mapping[str, float]
    extrinsics: Mapping[str, float]
    capture_monotonic_ns: int


@dataclass(frozen=True)
class PublicEvidence:
    """Fresh reviewer input; there is intentionally no evaluator-state field."""

    agentview: CameraFrame
    robot0_eye_in_hand: CameraFrame
    robot: PublicRobotState
    capture_monotonic_ns: int


@dataclass(frozen=True)
class CommitLease:
    """The action gate supplies only validated, still-live commit conditions."""

    commit_conditions: tuple[str, ...]


@dataclass(frozen=True)
class GateDecision:
    gate: Literal[3]
    boundary: Literal["executor_to_controller"]
    trigger: str
    reviewer_called: bool
    review_latency_s: float
    result: Literal["PASS", "BLOCK"]
    reason: str
    system_failure: bool


@dataclass(frozen=True)
class PrimitiveOutcome:
    status: Literal["completed", "intercepted"]
    submitted: int
    discarded: int
    reason: str | None
    gate_decisions: tuple[GateDecision, ...]


class Controller(Protocol):
    def hold_current_pose(self) -> None: ...

    def send(self, instruction: LowLevelInstruction) -> None: ...


class Reviewer(Protocol):
    def review(
        self,
        *,
        trigger: GateTrigger,
        pending_action: PublicPendingAction,
        evidence: PublicEvidence,
        commit_conditions: tuple[str, ...],
        deadline_monotonic_ns: int,
    ) -> Mapping[str, Any]: ...


Capture = Callable[[], PublicEvidence]
State = Callable[[], PublicRobotState]


class CommitGateOutlet:
    """Only object allowed to submit normalized instructions to a controller.

    ``reviewer.review`` must enforce its own RPC deadline. A late return is
    rejected, but synchronous Python cannot forcibly stop a hung callback;
    while it is hung, the controller remains in HOLD and no instruction is
    sent. That liveness limitation is explicit pending real executor wiring.
    """

    def __init__(
        self,
        *,
        controller: Controller,
        public_state: State,
        capture: Capture,
        reviewer: Reviewer,
        default_commit_conditions: Sequence[str],
        review_timeout_s: float = 15.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if review_timeout_s <= 0 or not math.isfinite(review_timeout_s):
            raise ValueError("review_timeout_s must be finite and positive")
        if not default_commit_conditions or any(not x.strip() for x in default_commit_conditions):
            raise ValueError("default commit conditions must be nonempty")
        self.controller = controller
        self.public_state = public_state
        self.capture = capture
        self.reviewer = reviewer
        self.default_commit_conditions = tuple(default_commit_conditions)
        self.review_timeout_ns = int(review_timeout_s * 1e9)
        self.clock_ns = clock_ns
        # A real release primitive sends multiple identical open controls.
        # Only the closed-to-open conversion needs review; retain this state
        # across successive one-command run_block() calls from env.step().
        self._last_sent_gripper_target: Literal['open', 'closed'] | None = None
        self._last_sent_held_object_name: str | None = None

    @staticmethod
    def _classify(
        instruction: LowLevelInstruction, state: PublicRobotState,
        *, open_already_sent_for_hold: bool = False,
    ) -> tuple[GateTrigger | None, str | None]:
        if state.source != "public_observation":
            return "held_gripper_open", "robot state is not public evidence"
        if state.hold_estimate not in ("held", "empty", "unknown"):
            return "held_gripper_open", "holding estimate is malformed"
        if not isinstance(state.gripper_closed, bool):
            return "held_gripper_open", "gripper state is malformed"

        triggers: list[GateTrigger] = []
        unresolved: str | None = None
        # Opening with a held payload is a commit regardless of measured
        # finger aperture: a 50 mm bottle prevents geometric closure. An
        # unknown hold estimate also fails closed. Only confirmed-empty
        # opening bypasses the reviewer.
        if instruction.gripper_target == "open" and not open_already_sent_for_hold:
            if state.hold_estimate == "unknown":
                triggers.append("held_gripper_open")
                unresolved = "holding estimate is unknown"
            elif state.hold_estimate == "held":
                triggers.append("held_gripper_open")

        if instruction.changes_payload_orientation and state.hold_estimate != "empty":
            if state.hold_estimate == "unknown" or state.held_object_tiltable is None:
                triggers.append("held_payload_tilt")
                unresolved = "held-object tilt status is unknown"
            elif state.held_object_tiltable:
                old = state.held_object_tilt_deg
                new = instruction.proposed_payload_tilt_deg
                if old is None or new is None or not math.isfinite(old) or not math.isfinite(new):
                    triggers.append("held_payload_tilt")
                    unresolved = "payload tilt prediction is unavailable"
                elif new > 20.0 and new > old:
                    # Crossing 20° is the first commit. While already above
                    # 20°, further increasing tilt is another unsafe commit;
                    # decreasing tilt is allowed without an LLM review.
                    triggers.append("held_payload_tilt")

        if instruction.heated_device_door_target == "close":
            triggers.append("heated_door_close")
            if not isinstance(instruction.heated_device_name, str) or not instruction.heated_device_name.strip():
                unresolved = "heated-device identity unavailable from public evidence"
        if len(triggers) > 1:
            # A combined instruction must be split by the adapter. This avoids
            # approving one irreversible conversion while overlooking another.
            return triggers[0], "multiple irreversible conversions in one instruction"
        return (triggers[0], unresolved) if triggers else (None, None)

    @staticmethod
    def _validate_evidence(evidence: PublicEvidence, hold_started_ns: int) -> str | None:
        if not isinstance(evidence, PublicEvidence):
            return "public evidence missing or malformed"
        if evidence.robot.source != "public_observation":
            return "robot state is not public evidence"
        if evidence.robot.observed_monotonic_ns < hold_started_ns:
            return "robot state predates HOLD"
        for expected, frame in (
            ("agentview", evidence.agentview),
            ("robot0_eye_in_hand", evidence.robot0_eye_in_hand),
        ):
            if not isinstance(frame, CameraFrame) or frame.name != expected:
                return f"{expected} frame missing"
            if frame.capture_monotonic_ns < hold_started_ns:
                return f"{expected} frame predates HOLD"
            if (
                frame.width <= 0
                or frame.height <= 0
                or not frame.rgb
                or not frame.depth
                or not frame.intrinsics
                or not frame.extrinsics
            ):
                return f"{expected} RGB-D or calibration missing"
        if evidence.capture_monotonic_ns < max(
            evidence.agentview.capture_monotonic_ns,
            evidence.robot0_eye_in_hand.capture_monotonic_ns,
        ):
            return "multi-view packet predates a camera frame"
        return None

    @staticmethod
    def _validate_review(
        raw: Mapping[str, Any], trigger: GateTrigger, capture_ns: int
    ) -> tuple[Literal["PASS", "BLOCK"], str, bool]:
        if not isinstance(raw, Mapping):
            return "BLOCK", "reviewer response is not a JSON object", True
        decision = raw.get("decision")
        reason = raw.get("reason")
        hazards = raw.get("identified_hazards")
        constraints = raw.get("constraints")
        if decision not in ("PASS", "BLOCK"):
            return "BLOCK", "reviewer decision is invalid", True
        if not isinstance(reason, str) or not reason.strip():
            return "BLOCK", "reviewer reason is missing", True
        if raw.get("trigger") != trigger or raw.get("capture_monotonic_ns") != capture_ns:
            return "BLOCK", "reviewer response does not match held evidence", True
        if not isinstance(hazards, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("entity"), str)
            and isinstance(item.get("attribute"), str)
            for item in hazards
        ):
            return "BLOCK", "reviewer hazard list is invalid", True
        if not isinstance(constraints, list) or not all(
            isinstance(item, str) and item.strip() for item in constraints
        ):
            return "BLOCK", "reviewer constraints are invalid", True
        return decision, reason.strip(), False

    def run_block(
        self,
        instructions: Sequence[LowLevelInstruction],
        *,
        lease: CommitLease | None = None,
    ) -> PrimitiveOutcome:
        """Submit until completion or the first BLOCK; never submit the tail."""

        decisions: list[GateDecision] = []
        submitted = 0
        for index, instruction in enumerate(instructions):
            try:
                if not isinstance(instruction, LowLevelInstruction):
                    raise TypeError("instruction was not normalized")
                if instruction.gripper_target not in (None, "open", "closed"):
                    raise ValueError("invalid gripper target")
                if instruction.heated_device_door_target not in (None, "open", "close"):
                    raise ValueError("invalid heated-door target")
                state = self.public_state()
                if not isinstance(state, PublicRobotState):
                    raise TypeError("public robot state is missing")
                if state.observed_monotonic_ns > self.clock_ns():
                    raise ValueError("public robot state timestamp is in the future")
                if self.clock_ns() - state.observed_monotonic_ns > 250_000_000:
                    raise ValueError("public robot state is stale")
                same_held_object = (
                    state.hold_estimate == 'held'
                    and isinstance(state.held_object_name, str)
                    and bool(state.held_object_name.strip())
                    and self._last_sent_gripper_target == 'open'
                    and self._last_sent_held_object_name == state.held_object_name
                )
                trigger, missing = self._classify(
                    instruction, state,
                    open_already_sent_for_hold=same_held_object,
                )
            except Exception as exc:
                try:
                    self.controller.hold_current_pose()
                except Exception:
                    pass
                reason = f"instruction/state normalization failed: {type(exc).__name__}"
                decisions.append(GateDecision(
                    gate=3, boundary="executor_to_controller",
                    trigger="normalization_error", reviewer_called=False,
                    review_latency_s=0.0, result="BLOCK", reason=reason,
                    system_failure=True,
                ))
                return PrimitiveOutcome(
                    "intercepted", submitted, len(instructions) - index, reason,
                    tuple(decisions),
                )
            if trigger is not None:
                started_ns = self.clock_ns()
                try:
                    self.controller.hold_current_pose()
                except Exception as exc:
                    missing = f"controller HOLD failed: {type(exc).__name__}"
                reviewer_called = False
                latency_s = 0.0
                decision: Literal["PASS", "BLOCK"] = "BLOCK"
                reason = missing or "review did not complete"
                system_failure = True
                if missing is None:
                    try:
                        evidence = self.capture()
                        problem = self._validate_evidence(evidence, started_ns)
                        if problem is not None:
                            reason = problem
                        else:
                            conditions = (
                                lease.commit_conditions if lease is not None
                                else self.default_commit_conditions
                            )
                            if not conditions or any(not x.strip() for x in conditions):
                                reason = "commit conditions are unavailable"
                            else:
                                deadline_ns = self.clock_ns() + self.review_timeout_ns
                                review_started_ns = self.clock_ns()
                                reviewer_called = True
                                raw = self.reviewer.review(
                                    trigger=trigger,
                                    pending_action=PublicPendingAction(
                                        gripper_target=instruction.gripper_target,
                                        changes_payload_orientation=instruction.changes_payload_orientation,
                                        proposed_payload_tilt_deg=instruction.proposed_payload_tilt_deg,
                                        heated_device_door_target=instruction.heated_device_door_target,
                                        heated_device_name=instruction.heated_device_name,
                                    ),
                                    evidence=evidence,
                                    commit_conditions=conditions,
                                    deadline_monotonic_ns=deadline_ns,
                                )
                                latency_s = max(0.0, (self.clock_ns() - review_started_ns) / 1e9)
                                if self.clock_ns() > deadline_ns:
                                    reason = "reviewer deadline exceeded"
                                else:
                                    decision, reason, system_failure = self._validate_review(
                                        raw, trigger, evidence.capture_monotonic_ns
                                    )
                    except Exception as exc:
                        if reviewer_called:
                            latency_s = max(0.0, (self.clock_ns() - review_started_ns) / 1e9)
                        reason = f"review/capture failed: {type(exc).__name__}"
                decisions.append(
                    GateDecision(
                        gate=3,
                        boundary="executor_to_controller",
                        trigger=trigger,
                        reviewer_called=reviewer_called,
                        review_latency_s=latency_s,
                        result=decision,
                        reason=reason,
                        system_failure=system_failure,
                    )
                )
                if decision == "BLOCK":
                    return PrimitiveOutcome(
                        status="intercepted",
                        submitted=submitted,
                        discarded=len(instructions) - index,
                        reason=reason,
                        gate_decisions=tuple(decisions),
                    )
            try:
                self.controller.send(instruction)
            except Exception as exc:
                # A controller fault is not a safety permit. The unsubmitted
                # tail is cancelled and the executor reports interception.
                try:
                    self.controller.hold_current_pose()
                except Exception:
                    pass
                return PrimitiveOutcome(
                    status="intercepted",
                    submitted=submitted,
                    discarded=len(instructions) - index,
                    reason=f"controller send failed: {type(exc).__name__}",
                    gate_decisions=tuple(decisions),
                )
            if instruction.gripper_target is not None:
                self._last_sent_gripper_target = instruction.gripper_target
                self._last_sent_held_object_name = (
                    state.held_object_name if state.hold_estimate == 'held'
                    else None
                )
            submitted += 1
        return PrimitiveOutcome("completed", submitted, 0, None, tuple(decisions))
