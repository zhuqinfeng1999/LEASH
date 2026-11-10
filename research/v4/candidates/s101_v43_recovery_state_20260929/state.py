"""S1 recovery's public state and record seam, without a motion executor.

Known-state routes retain the original open park / closed HOLD. UNKNOWN is
never converted to the old latch False. Physical admission and the actual
core/bridge caller remain separate prerequisites for this opt-in slice.
"""
from dataclasses import dataclass
from enum import Enum
import math

from research.v4.leash_v43.registry import Holding as HoldingKind


class RecoveryCapabilityGap(RuntimeError):
    """Valid public uncertainty cannot choose the unchanged recovery route."""


class RecoveryRoute(str, Enum):
    NO_MOTION_TASK = "no_motion_after_task_gate"
    NO_MOTION_START = "no_motion_already_at_public_start"
    HOLD = "safe_hold"
    PARK = "physical_open_gripper_observation_park"


@dataclass(frozen=True)
class PublicRecoveryRecord:
    kind: str
    actual_control_steps: int
    passed_common_outlet: bool
    action_gate_checked: bool
    final_holding: HoldingKind


def _kind(value):
    if type(value) is not HoldingKind:
        raise ValueError("public executed/proprio holding enum required")
    return value


def _xyz(value):
    result = tuple(value)
    if len(result) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                              or not math.isfinite(x) for x in result):
        raise ValueError("existing public EEF XYZ required")
    return result


def route(*, holding, task_outcome, executed_task_calls, sequence,
          initial_sequence, current_public_eef, initial_public_eef):
    """No image, latch, controller, reviewer or environment access."""
    holding = _kind(holding)
    if task_outcome not in {"PROCEED", "ASK", "REFUSE"}:
        raise ValueError("validated current-round task result required")
    if any(type(x) is not int or x < 0 for x in
           (executed_task_calls, sequence, initial_sequence)):
        raise ValueError("actual execution counters required")
    if task_outcome in {"ASK", "REFUSE"}:
        if executed_task_calls or sequence != initial_sequence:
            raise ValueError("task-gated recovery has an executed motion")
        # This certifies no motion, not an empty gripper. Initial UNKNOWN is
        # retained rather than manufacturing an EMPTY or a grasp baseline.
        return RecoveryRoute.NO_MOTION_TASK
    current, target = _xyz(current_public_eef), _xyz(initial_public_eef)
    if holding is HoldingKind.UNKNOWN:
        raise RecoveryCapabilityGap("public holding UNKNOWN; unchanged park/HOLD route unavailable")
    if holding is HoldingKind.HELD:
        return RecoveryRoute.HOLD
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(current, target)))
    return RecoveryRoute.NO_MOTION_START if distance <= .009 else RecoveryRoute.PARK


def certified(record, *, task_outcome, require_action_gate):
    """Narrow replacement for the legacy bridge's boolean record predicate."""
    if (type(record) is not PublicRecoveryRecord or type(record.actual_control_steps) is not int
            or record.actual_control_steps < 0
            or type(record.passed_common_outlet) is not bool
            or type(record.action_gate_checked) is not bool
            or type(require_action_gate) is not bool
            or type(record.final_holding) is not HoldingKind):
        return False
    no_motion = (not record.actual_control_steps and not record.passed_common_outlet
                 and not record.action_gate_checked)
    if record.kind == RecoveryRoute.NO_MOTION_TASK.value:
        return no_motion and task_outcome in {"ASK", "REFUSE"}
    if record.kind == RecoveryRoute.NO_MOTION_START.value:
        return no_motion and record.final_holding is HoldingKind.EMPTY
    if not record.passed_common_outlet or (require_action_gate and not record.action_gate_checked):
        return False
    if record.kind == RecoveryRoute.HOLD.value:
        return record.actual_control_steps == 1 and record.final_holding is HoldingKind.HELD
    if record.kind == RecoveryRoute.PARK.value:
        return record.actual_control_steps <= 260 and record.final_holding is HoldingKind.EMPTY
    return False


def public_holding(controller):
    """Caller binds its existing adapter.estimator.kind, never scene truth."""
    callback = getattr(controller, "public_holding", None)
    if not callable(callback):
        raise ValueError("public holding callback unavailable")
    kind = _kind(callback())
    if kind is HoldingKind.UNKNOWN:
        raise RecoveryCapabilityGap("public holding UNKNOWN during unchanged empty-gripper park")
    return kind


def park_record(kind, actual_control_steps, passed_common_outlet,
                action_gate_checked, *, final_holding):
    record = PublicRecoveryRecord(kind, actual_control_steps, passed_common_outlet,
                                  action_gate_checked, final_holding)
    if not certified(record, task_outcome="PROCEED", require_action_gate=False):
        raise RecoveryCapabilityGap("unchanged park did not retain public EMPTY")
    return record
