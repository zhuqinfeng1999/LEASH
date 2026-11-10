"""Provisional Ours-only outlet for the frozen RPent 7D LIBERO control path.

Every current RPent analytic tool calls ``env.step``; π0.5 calls
``env.chunk_step``, which calls ``self.step`` for every chunk element; planner
``hold`` also calls ``self.step``. This subclass intercepts that common step
*before* V4PublicEnvAdapter invokes ``scene.step``. It does not change B0.

The public holding, tilt and door estimators are deliberately injected. They
must be implemented from agent-visible RGB-D/proprioception, never private
scene/evaluator state. This candidate is not a ready-to-run Ours policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Literal, Sequence

import numpy as np

from research.v4.bridge.observation import V4PublicEnvAdapter
from research.v4.candidates.leash_commit_gate_20260924.commit_gate import (
    Capture, CommitGateOutlet, CommitLease, GateDecision, LowLevelInstruction,
    PrimitiveOutcome, PublicRobotState, Reviewer, State,
)


class CommitIntercepted(Exception):
    """Default local exception; live RPent should inject its ToolCancelled."""

    def __init__(self, outcome: PrimitiveOutcome):
        self.outcome = outcome
        super().__init__(f"gate ③ intercepted: {outcome.reason}")


@dataclass(frozen=True)
class DoorIntent:
    """Prediction from public views immediately before a controller action.

Return ``None`` only when a calibrated classifier rules out door motion. A
potentially closing door of unknown identity uses ``target='close', name=None``
and blocks in gate ③ before the controller receives the action.
    """

    target: Literal["close", "open"]
    name: str | None


DoorClassifier = Callable[[np.ndarray], DoorIntent | None]
TiltPredictor = Callable[[np.ndarray, PublicRobotState], float | None]
CancelFactory = Callable[[PrimitiveOutcome], Exception]


class _SimStepController:
    """Bridge gate ③ to the existing V4 simulator step, without recursion."""

    def __init__(self, adapter: "OursCommitEnvAdapter") -> None:
        self.adapter = adapter
        self.last_result = None

    def hold_current_pose(self) -> None:
        # In this synchronous MuJoCo path, *not calling scene.step* is the
        # actual HOLD. V3's planner hold() advances physics and may open the
        # gripper; using it here would violate v4.1.
        self.adapter.safety_holds += 1

    def send(self, instruction: LowLevelInstruction) -> None:
        # Explicit base-class dispatch avoids re-entering the gate. The base
        # still validates the public RGB-D observation and latches real faults.
        self.last_result = V4PublicEnvAdapter.step(self.adapter, instruction.payload)


class OursCommitEnvAdapter(V4PublicEnvAdapter):
    """Ours-specific single outlet for 7D control, including π0.5 chunks.

    The `door_classifier` argument is mandatory to make a missing classifier
    visible during integration. Its implementation must classify imminent
    heated-device closure from *public* evidence; a no-op lambda is only a
    test double and would not be a valid live implementation.
    """

    def __init__(
        self, scene, *, instruction: str,
        public_state: State,
        capture: Capture,
        reviewer: Reviewer,
        door_classifier: DoorClassifier,
        default_commit_conditions: Sequence[str],
        tilt_predictor: TiltPredictor | None = None,
        cancel_exception_factory: CancelFactory | None = None,
        review_timeout_s: float = 15.0,
        on_sim_step=None,
    ) -> None:
        super().__init__(scene, instruction=instruction, on_sim_step=on_sim_step)
        if not callable(door_classifier):
            raise ValueError("a public-evidence heated-door classifier is required")
        self.door_classifier = door_classifier
        self.tilt_predictor = tilt_predictor
        self.cancel_exception_factory = cancel_exception_factory or CommitIntercepted
        self._public_state_source = public_state
        self._pending_public_state: PublicRobotState | None = None
        self._controller = _SimStepController(self)
        self.commit_gate = CommitGateOutlet(
            controller=self._controller,
            public_state=self._state_for_gate,
            capture=capture,
            reviewer=reviewer,
            default_commit_conditions=default_commit_conditions,
            review_timeout_s=review_timeout_s,
        )
        self.lease: CommitLease | None = None
        self.safety_holds = 0
        self.last_outcome: PrimitiveOutcome | None = None
        self._chunk_context: tuple[int, int] | None = None

    def _state_for_gate(self) -> PublicRobotState:
        # Normalize and classify the pending instruction against one public
        # state snapshot. Fresh post-HOLD evidence is captured separately.
        return (self._pending_public_state if self._pending_public_state is not None
                else self._public_state_source())

    @staticmethod
    def _valid_action(action) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("RPent action must be a finite 7-vector")
        return value

    def _normalization_block(self, reason: str) -> PrimitiveOutcome:
        self._controller.hold_current_pose()
        return PrimitiveOutcome(
            status="intercepted", submitted=0, discarded=1, reason=reason,
            gate_decisions=(GateDecision(
                gate=3, boundary="executor_to_controller",
                trigger="normalization_error", reviewer_called=False,
                review_latency_s=0.0, result="BLOCK", reason=reason,
                system_failure=True,
            ),),
        )

    def _normalize(self, action: np.ndarray) -> LowLevelInstruction:
        # RPent and robosuite use negative gripper commands to open, positive
        # commands to close. A negative command is reviewed even when the
        # fingers are physically separated around a held object.
        gripper_target = "open" if action[6] < 0 else "closed" if action[6] > 0 else None
        rotates = bool(np.any(np.abs(action[3:6]) > 1e-7))
        state = self._public_state_source()
        if not isinstance(state, PublicRobotState) or state.source != "public_observation":
            raise ValueError("public robot state is unavailable")
        self._pending_public_state = state
        predicted_tilt = None
        if rotates and self.tilt_predictor is not None:
            predicted_tilt = self.tilt_predictor(action.copy(), state)
            if predicted_tilt is not None and not math.isfinite(float(predicted_tilt)):
                raise ValueError("public payload tilt prediction is nonfinite")
        door = self.door_classifier(action.copy())
        if door is not None and not isinstance(door, DoorIntent):
            raise ValueError("heated-door classifier returned an invalid event")
        return LowLevelInstruction(
            payload=action.copy(),
            gripper_target=gripper_target,
            changes_payload_orientation=rotates,
            proposed_payload_tilt_deg=predicted_tilt,
            heated_device_door_target=door.target if door else None,
            heated_device_name=door.name if door else None,
        )

    def _raise_interception(self, outcome: PrimitiveOutcome) -> None:
        if self._chunk_context is not None:
            count, index = self._chunk_context
            outcome = replace(outcome, submitted=index,
                              discarded=count - index)
        self.last_outcome = outcome
        exc = self.cancel_exception_factory(outcome)
        if not isinstance(exc, Exception):
            raise TypeError("cancel_exception_factory must return an Exception")
        raise exc

    def step(self, action):
        if self.observation_fault:
            raise RuntimeError("public observation fault is latched")
        if self._chunk_context is not None:
            count, index = self._chunk_context
            self._chunk_context = (count, index + 1)
            # ``index`` for this action is the value before increment.
            current_index = index
        else:
            current_index = None
        try:
            normalized = self._normalize(self._valid_action(action))
        except Exception as exc:
            self._pending_public_state = None
            outcome = self._normalization_block(
                f"public instruction normalization failed: {type(exc).__name__}: {exc}"
            )
            if current_index is not None:
                self._chunk_context = (self._chunk_context[0], current_index)
            self._raise_interception(outcome)
        self._controller.last_result = None
        try:
            outcome = self.commit_gate.run_block([normalized], lease=self.lease)
        finally:
            self._pending_public_state = None
        if outcome.status == "intercepted":
            if current_index is not None:
                self._chunk_context = (self._chunk_context[0], current_index)
            self._raise_interception(outcome)
        self.last_outcome = outcome
        if self._controller.last_result is None:
            self._raise_interception(self._normalization_block("controller returned no observation"))
        return self._controller.last_result

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        # The inherited V3 loop dynamically dispatches every element to our
        # step(). On an exception it exits immediately, so no tail is sent.
        chunk = np.asarray(actions, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 7 or not len(chunk):
            raise ValueError("RPent action chunk must be [N,7] with N>0")
        if self._chunk_context is not None:
            raise RuntimeError("nested RPent action chunk is unsupported")
        self._chunk_context = (len(chunk), 0)
        try:
            return super().chunk_step(chunk, return_all_frames=return_all_frames)
        finally:
            self._chunk_context = None
