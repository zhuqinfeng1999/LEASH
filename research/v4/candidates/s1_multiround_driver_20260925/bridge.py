"""CPU candidate: S1 over the existing persistent RPent multi-round driver.

No model is started here. One caller-owned planner session, toolkit and scene
persist across both rounds. A task gate receives only current public evidence
and full public history. Inter-round recovery must physically use the caller's
ordinary controller outlet before round 2 is observed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Callable

from research.v3.bridge.b0 import BUDGETED_ACTIONS
from research.v4.candidates.multiround_bridge_20260924.driver import (
    PersistentSession, PublicObservation, RoundInput, SessionResult,
    capture_public_observation, run_multiround_b0,
)
from research.v4.candidates.s1_v42_tabletop_20260925.scene_v2 import INSTRUCTIONS
from research.v4.candidates.s101_v43_recovery_state_20260929.state import RecoveryCapabilityGap


@dataclass(frozen=True)
class HistoryEntry:
    index: int
    instruction: str
    start_observation: PublicObservation  # exact PNG bytes retained
    primitive_summaries: tuple[tuple[str, Any], ...]
    boundary: str


@dataclass(frozen=True)
class TaskGateDecision:
    outcome: str  # PROCEED, REFUSE or ASK
    message: str = ""


@dataclass(frozen=True)
class RecoveryRecord:
    kind: str
    actual_control_steps: int
    passed_common_outlet: bool
    action_gate_checked: bool  # required for an Ours production adapter
    final_public_grasp_latched: bool


class S1PersistentSession:
    """Use the same inner session for both instructions; never compact history."""

    def __init__(self, inner: PersistentSession, scene: Any, *,
                 park_after_round1: Callable[[Any, Any], RecoveryRecord],
                 task_gate: Callable[[RoundInput, tuple[HistoryEntry, ...]], TaskGateDecision],
                 require_action_gate_for_recovery: bool = False,
                 advance_round: Callable[[Any, Any], str] | None = None):
        self.inner = inner
        self.scene = scene
        self.park_after_round1 = park_after_round1
        self.task_gate = task_gate
        self.require_action_gate_for_recovery = require_action_gate_for_recovery
        self.advance_round = advance_round or (lambda scene, _toolkit: scene.advance_round())
        self.history: list[HistoryEntry] = []
        self.recovery: RecoveryRecord | None = None
        self.recovery_seconds = 0.0
        self.recovery_capability_stop = None
        self.task_gate_seconds: list[float] = []
        self.scene_identity = id(scene)

    async def run_round(self, turn_input: RoundInput, toolkit: Any,
                        boundary: Callable[[], str | None], *,
                        max_turns: int, timeout_s: int) -> SessionResult:
        if turn_input.index != len(self.history) + 1 or id(self.scene) != self.scene_identity:
            raise RuntimeError("S1 round or scene identity changed")
        if turn_input.instruction != INSTRUCTIONS[turn_input.index - 1]:
            raise RuntimeError("S1 instruction order changed")
        if turn_input.index == 2 and self.recovery is None:
            raise RuntimeError("Round 2 cannot start without physical recovery")
        gate_started = time.monotonic()
        decision = self.task_gate(turn_input, tuple(self.history))
        self.task_gate_seconds.append(time.monotonic() - gate_started)
        if not isinstance(decision, TaskGateDecision) or decision.outcome not in {
            "PROCEED", "REFUSE", "ASK"
        }:
            raise ValueError("Task gate must return a TaskGateDecision")
        if decision.outcome != "PROCEED" and not decision.message.strip():
            raise ValueError("Refusal or question requires the reviewer's own message")
        call_start = len(toolkit.calls)
        if decision.outcome == "PROCEED":
            result = await self.inner.run_round(turn_input, toolkit, boundary,
                                                max_turns=max_turns,
                                                timeout_s=timeout_s)
        else:
            # The gate acts before the planner and before any robot primitive.
            response = toolkit.execute_tool("finish", {
                "status": "refused" if decision.outcome == "REFUSE" else "asked",
                "message": decision.message,
            })
            if getattr(response, "result", {}).get("error"):
                raise RuntimeError("Task-gate finish failed")
            result = SessionResult(turns_used=0)
        if result.error or not result.sdk_completed:
            return result
        seal = result.resource_end or boundary()
        if seal is None and result.turns_used >= max_turns:
            seal = "planner_turn_quota"
        if seal is None:
            # No finish/quota means no authorized next instruction.
            return result
        summaries = tuple(s for call in toolkit.calls[call_start:]
                          if (s := _call_summary(call)) is not None)
        self.history.append(HistoryEntry(
            index=turn_input.index, instruction=turn_input.instruction,
            start_observation=PublicObservation(
                str(turn_input.observation.text), bytes(turn_input.observation.image_png)),
            primitive_summaries=summaries, boundary=seal))
        if turn_input.index == 1:
            started = time.monotonic()
            try:
                recovery = self.park_after_round1(self.scene, toolkit)
            except RecoveryCapabilityGap as exc:
                self.recovery_capability_stop = {
                    "kind": "public_recovery_unavailable", "round": 1,
                    "reason": str(exc), "second_round_started": False}
                return replace(result, episode_stop="public_recovery_unavailable")
            finally:
                self.recovery_seconds = time.monotonic() - started
            no_motion = (
                isinstance(recovery, RecoveryRecord) and
                ((recovery.kind == "no_motion_after_task_gate" and
                  decision.outcome in {"REFUSE", "ASK"}) or
                 recovery.kind == "no_motion_already_at_public_start") and
                recovery.actual_control_steps == 0 and
                not recovery.final_public_grasp_latched and
                not recovery.passed_common_outlet and
                not recovery.action_gate_checked)
            ordinary = (
                isinstance(recovery, RecoveryRecord) and
                recovery.passed_common_outlet and
                recovery.actual_control_steps >= 0 and
                (not recovery.final_public_grasp_latched or
                 recovery.kind == "safe_hold") and
                (recovery.kind != "safe_hold" or
                 recovery.actual_control_steps > 0) and
                (not self.require_action_gate_for_recovery or
                 recovery.action_gate_checked))
            if (not (no_motion or ordinary) or
                id(self.scene) != self.scene_identity):
                raise RuntimeError("S1 inter-round recovery was not certified")
            self.recovery = recovery
            if self.advance_round(self.scene, toolkit) != INSTRUCTIONS[1]:
                raise RuntimeError("S1 scene did not advance to exact second instruction")
        return result


def _call_summary(item: Any) -> tuple[str, Any] | None:
    """Only planner-visible motion calls, never raw result/evaluator packets."""
    if not isinstance(item, dict) or not isinstance(item.get("name"), str):
        raise TypeError("Cannot summarize nonstandard toolkit call")
    name = item["name"]
    if name not in BUDGETED_ACTIONS:
        return None
    args = item.get("arguments", {})
    if not isinstance(args, dict):
        raise TypeError("Cannot summarize non-mapping public arguments")
    result = item.get("result", {})
    error = result.get("error") if isinstance(result, dict) else None
    return (name, {"arguments": dict(args),
                   "outcome": "error" if error else "executed"})


async def run_s1_b0_candidate(*, toolkit: Any, planner_session: PersistentSession,
                               scene: Any,
                               park_after_round1: Callable[[Any, Any], RecoveryRecord],
                               task_gate: Callable[[RoundInput, tuple[HistoryEntry, ...]], TaskGateDecision],
                               observe: Callable[[Any], PublicObservation] = capture_public_observation,
                               require_action_gate_for_recovery: bool = False,
                 advance_round: Callable[[Any, Any], str] | None = None):
    """Return two round results and complete public S1 history."""
    wrapper = S1PersistentSession(
        planner_session, scene, park_after_round1=park_after_round1,
        task_gate=task_gate,
        require_action_gate_for_recovery=require_action_gate_for_recovery,
        advance_round=advance_round)
    rounds = await run_multiround_b0(
        instructions=INSTRUCTIONS, toolkit=toolkit, session=wrapper,
        observe=observe)
    stopped = (wrapper.recovery_capability_stop is not None and
               len(rounds) == len(wrapper.history) == 1 and wrapper.recovery is None)
    if not stopped and (len(rounds) != 2 or len(wrapper.history) != 2):
        raise RuntimeError("S1 must deliver both rounds in one episode")
    # The existing generic driver times the whole wrapper as planner time.
    # Split out task review and physical recovery before reporting timing.
    rounds = [replace(row, planner_seconds=max(0.0, row.planner_seconds -
              wrapper.task_gate_seconds[i] -
              (wrapper.recovery_seconds if i == 0 else 0.0)))
              for i, row in enumerate(rounds)]
    return rounds, wrapper
