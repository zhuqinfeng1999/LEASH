"""S1 candidate wiring for real RPent owner-thread B0/Ours components.

Internal callbacks are not RPent tools and are inaccessible through tool
schemas. They marshal scene/evaluator/recovery work to the simulator owner
thread while one persistent Codex planner session runs on the worker thread.
This module never starts a model or simulator on import.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import secrets
import threading
from types import SimpleNamespace
from typing import Any, Callable

from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import (
    TaskGate, TaskConstraints, TurnHistory, UserInstruction,
)
from research.v4.candidates.leash_commit_gate_20260924.commit_gate import (
    CameraFrame, PublicEvidence,
)
from .bridge import HistoryEntry, TaskGateDecision


_INTERNAL = frozenset({
    "__s1_capture_review_evidence__", "__s1_recover__",
    "__s1_advance_evaluator__",
})


class S1OwnerCallbacks:
    """Narrow owner-thread callbacks around an unchanged RPent toolkit.

    The secret is held by the runner's Python closure and never sent to the
    model as a tool specification or observation. A forged internal call
    without the secret is rejected before touching scene/evaluator state.
    """

    def __init__(self, base: Any, *, capture: Callable[[], PublicEvidence],
                 recover: Callable[[], Any], advance: Callable[[], str]):
        self.base = base
        self._callbacks = {
            "__s1_capture_review_evidence__": capture,
            "__s1_recover__": recover,
            "__s1_advance_evaluator__": advance,
        }
        self._token = secrets.token_hex(32)
        self.owner_thread_id = threading.get_ident()
        self.internal_log: list[tuple[str, int]] = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_tools_spec(self):
        specs = self.base.get_tools_spec()
        if any(spec.get("name") in _INTERNAL for spec in specs):
            raise RuntimeError("S1 internal callback exposed as an RPent tool")
        return specs

    def execute_tool(self, name, arguments):
        if name in _INTERNAL:
            if (not isinstance(arguments, dict) or
                arguments.get("_s1_owner_token") != self._token or
                set(arguments) != {"_s1_owner_token"}):
                raise PermissionError("unauthorized internal S1 callback")
            if threading.get_ident() != self.owner_thread_id:
                raise RuntimeError("S1 internal callback bypassed simulator owner thread")
            value = self._callbacks[name]()
            self.internal_log.append((name, threading.get_ident()))
            return SimpleNamespace(result=value)
        return self.base.execute_tool(name, arguments)

    def invoke(self, owner_proxy, name: str):
        if name not in _INTERNAL:
            raise ValueError("unknown S1 owner callback")
        # Always use OwnerThreadToolkit.execute_tool: direct self.execute_tool
        # from the planner thread would violate EGL / MuJoCo thread affinity.
        return owner_proxy.execute_tool(name, {
            "_s1_owner_token": self._token,
        }).result


@dataclass(frozen=True)
class S1TaskReviewRecord:
    round_index: int
    result: str
    reason: str
    system_failure: bool
    historical_keyframes: int


class S1TaskGateAdapter:
    """Convert S1 public history to the existing live reviewer TaskGate API."""

    def __init__(self, *, method: str, task_gate: TaskGate | None,
                 owner_proxy: Any, owner_callbacks: S1OwnerCallbacks,
                 episode_id: str, normalize_decision: Callable | None = None):
        if method not in {"B0", "Ours"}:
            raise ValueError("S1 supports B0/Ours only")
        if method == "Ours" and task_gate is None:
            raise ValueError("Ours requires a real independent TaskGate")
        self.method = method
        self.task_gate = task_gate
        self.owner_proxy = owner_proxy
        self.owner_callbacks = owner_callbacks
        self.episode_id = episode_id
        self.normalize_decision = normalize_decision
        self.start_frames: list[CameraFrame] = []
        self.records: list[S1TaskReviewRecord] = []
        self.task_decisions = []

    @staticmethod
    def _summaries(history: HistoryEntry) -> tuple[str, ...]:
        return tuple(json.dumps({"name": name, **details}, sort_keys=True,
                                ensure_ascii=False, default=str)
                     for name, details in history.primitive_summaries)

    def __call__(self, round_input, history: tuple[HistoryEntry, ...]) -> TaskGateDecision:
        if self.method == "B0":
            # Original RPent baseline has no external task gate. Its one
            # persistent planner session still sees all prior conversation.
            return TaskGateDecision("PROCEED")
        evidence = self.owner_callbacks.invoke(
            self.owner_proxy, "__s1_capture_review_evidence__")
        if not isinstance(evidence, PublicEvidence):
            raise RuntimeError("S1 reviewer evidence is not PublicEvidence")
        if len(history) != len(self.start_frames):
            raise RuntimeError("S1 reviewer keyframe/history count mismatch")
        previous = tuple(TurnHistory(
            UserInstruction(entry.instruction), self._summaries(entry), frame)
            for entry, frame in zip(history, self.start_frames))
        finish_requests = []
        decision = self.task_gate.begin_round(
            round_id=f"{self.episode_id}-round{round_input.index}",
            instruction=UserInstruction(round_input.instruction),
            current=evidence, history=previous,
            finish=lambda status, message: finish_requests.append((status, message)),
        )
        self.start_frames.append(evidence.agentview)
        if decision.result == "PROCEED" and self.normalize_decision is not None:
            try:
                decision = self.normalize_decision(decision)
            except Exception as exc:
                decision = replace(
                    decision, result="REFUSE",
                    reason=f"S1 public task-name binding failed: {type(exc).__name__}: {exc}",
                    constraints=TaskConstraints(), system_failure=True)
                finish_requests.append(("refused", decision.reason))
        self.task_decisions.append(decision)
        self.records.append(S1TaskReviewRecord(
            round_input.index, decision.result, decision.reason,
            decision.system_failure, len(previous)))
        if decision.result == "PROCEED":
            if finish_requests:
                raise RuntimeError("task gate requested finish on PROCEED")
        elif finish_requests != [
            ("refused" if decision.result == "REFUSE" else "asked", decision.reason)
        ]:
            raise RuntimeError("task gate finish request disagrees with decision")
        return TaskGateDecision(decision.result, decision.reason)


def callback_recovery(owner_proxy, callbacks: S1OwnerCallbacks):
    return callbacks.invoke(owner_proxy, "__s1_recover__")


def callback_advance(owner_proxy, callbacks: S1OwnerCallbacks):
    return callbacks.invoke(owner_proxy, "__s1_advance_evaluator__")


def sync_public_rpent_round_observation(b0_toolkit, adapter) -> dict[str, Any]:
    """Publish one fresh *read-only* RPent state after owner-thread recovery.

    RPent's view_env_state reads its last saved EnvState step. Inter-round
    physical recovery changes the world without an RPent primitive call, so
    that saved step would otherwise be stale. This refresh renders the actual
    current public cameras, updates π0.5's last observation and records a
    boundary state; it sends no controller instruction and spends no call.
    """
    original = b0_toolkit.original
    primitives = original._primitives
    before = original._state.latest_step
    observation = adapter._policy_obs()
    if observation.get("task_descriptions") != adapter.instruction:
        raise RuntimeError("RPent policy observation has stale S1 instruction")
    primitives.set_obs(observation)
    recorded = original.get_env_state(
        command={"action": "s1_round_observation_refresh"},
        result={"kind": "public_observation_only_no_controller_action"},
        elapsed_s=0.0)
    after = original._state.latest_step
    if not isinstance(after, int) or (before is not None and after <= before):
        raise RuntimeError("RPent public round observation was not saved")
    if recorded.get("step") != after:
        raise RuntimeError("RPent saved step and returned public step differ")
    original._publish_step(original._state.latest_record())
    return {"previous_step": before, "public_step": after,
            "instruction": adapter.instruction,
            "primitive_calls_charged": 0}
