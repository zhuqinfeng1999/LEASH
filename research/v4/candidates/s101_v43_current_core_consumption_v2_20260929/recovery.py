"""Reuse the finite original recovery outlet, with relevant no-motion inputs."""
from dataclasses import asdict

from research.v4.candidates.s101_v43_actual_core_binding_20260929.recovery import RecoveryScope as DraftScope
from research.v4.candidates.s101_v43_recovery_state_20260929 import state


class RecoveryScope(DraftScope):
    def recover_on_owner(self, *, scene, initial_public_eef, initial_sequence):
        checked = self._validated_task or {}
        if checked.get("result") not in {"ASK", "REFUSE"}:
            # Original finite park / one closed HOLD and fresh lease context.
            # No new control loop, route or holding estimator is supplied.
            return super().recover_on_owner(scene=scene,
                initial_public_eef=initial_public_eef, initial_sequence=initial_sequence)
        if (self._recovery_context is not None or self.boundary.round_open
                or self.boundary.task_approved or len(self.sealed_history) != 1
                or self._start_keyframes is not None or self.boundary.lease is not None
                or self.lease_receipt is not None):
            raise ValueError("no-motion recovery requires the already sealed first round")
        route = state.route(holding=self.adapter.estimator.kind,
            task_outcome=checked["result"], executed_task_calls=self.base.episode_calls,
            sequence=scene.sequence, initial_sequence=initial_sequence,
            current_public_eef=None, initial_public_eef=None)
        if route is not state.RecoveryRoute.NO_MOTION_TASK:
            raise RuntimeError("ASK/REFUSE must retain the original no-motion route")
        record = state.PublicRecoveryRecord(route.value, 0, False, False,
                                            self.adapter.estimator.kind)
        self.recovery_logs.append({"kind": "recovery_no_motion", **asdict(record)})
        return record
