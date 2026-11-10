"""Selected control callbacks on the existing S1 Owner and two-round driver.

No driver, controller, perception provider or reviewer history is substituted.
The root-owned outlet reads public proprioception for the existing estimator;
decision Consumers receive only their existing method-specific public proxies.
"""
from contextlib import nullcontext
from dataclasses import asdict
from types import FunctionType

import numpy as np

from research.v4.candidates.s101_v43_current_core_consumption_v2_20260929.runtime import CoreBindings
from research.v4.candidates.s101_v43_bridge_round_binding_20260929 import bridge
from research.v4.candidates.s101_v43_recovery_state_20260929 import state, park_binding
from research.v4.candidates.s101_v43_actual_core_binding_20260929.recovery import RecoveryIntercepted
from research.v4.candidates.s1_multiround_driver_20260925.recovery import AdapterControlOutlet
from research.v4.candidates.s1_s2_deferred_production_optin_20260927 import hooks
from research.v4.familiarization.controls_preparation.consumer_callbacks import admit_driver_round
from research.v4.familiarization.controls_preparation.control_entry import make_consumer
from research.v4.leash_v43.adapter import V43PublicEnvAdapter
from research.v4.leash_v43.method_inputs import get_method_profile, validate_review_response
from research.v4.leash_v43.registry import HoldingEstimator


def bind_public_executed_holding(adapter, calibration):
    """Observe a successful original B1 send, without a registry or permission."""
    if hasattr(adapter, "dispatcher") or hasattr(adapter, "estimator"):
        raise ValueError("B1 needs its original registry-free adapter")
    adapter.estimator = HoldingEstimator(calibration, grasp_assumption="unverified")
    original = adapter.step
    def proprio():
        return V43PublicEnvAdapter._proprio(adapter, adapter.scene.sequence)
    def step(action):
        before = proprio()
        value = original(action)
        grip = "CLOSE" if action[6] > 0 else "OPEN" if action[6] < 0 else "KEEP"
        adapter.estimator.record_command(grip, before)
        adapter.estimator.observe(proprio())
        return value
    adapter.step = step
    return proprio


class S1ControlContext(CoreBindings):
    def __init__(self, parent, *, method, policy, calibration, source_hashes):
        super().__init__(parent, policy=policy, calibration=calibration, source_hashes=source_hashes)
        self.method = method
        self.profile = get_method_profile(method)
        if method not in {"B1", "B2", "B3"}:
            raise ValueError("selected control profile required")
        self._sealed = 0
        self._task = None
        self._adapter = None
        self._proprio = None
        self.recovery_logs = []

    def require_interfaces(self, method, adapter, reviewer, workspace,
                           locate_vla_target, recovery_on_owner, diagnostic_only):
        if (method != self.method or workspace is not None or locate_vla_target is not None
                or recovery_on_owner is not None):
            raise ValueError("control core must use its own selected callbacks")
        if method == "B1":
            if reviewer is not None or hasattr(adapter, "dispatcher"):
                raise ValueError("B1 has no reviewer or registry")
        elif (not isinstance(adapter, V43PublicEnvAdapter) or adapter.profile != self.profile
                or reviewer is None or adapter.policy is not self.policy):
            raise ValueError("actual selected control adapter and reviewer required")

    def make_consumer(self, public_base, adapter, scene, sam_client, reviewer, instruction, episode):
        self._adapter = adapter
        self._proprio = (bind_public_executed_holding(adapter, self.calibration)
                         if self.method == "B1" else
                         lambda: adapter.dispatcher.read_proprio(adapter.dispatcher.sequence))
        consumer = make_consumer(self.method, public_base, adapter, policy=self.policy,
            episode=episode, primitives=public_base.original._primitives,
            control_timestep=scene.backend.control_timestep,
            vla_chunk_controls=self.vla_chunk_controls, reviewer=reviewer, instruction=instruction)
        self.last_consumer = consumer
        consumer.recover_on_owner = lambda **values: self.recover_on_owner(consumer, **values)
        self.operational_calls.append("make_consumer")
        return consumer

    def sealed_rounds(self, consumer):
        return self._sealed

    @staticmethod
    def consumer_workspace(consumer):
        return consumer.workspace if hasattr(consumer, "workspace") else None

    @staticmethod
    def consumer_provider(consumer):
        return None

    def observe_boundary(self, consumer):
        return consumer.boundary if self.method == "B1" else bridge.DriverObservationBoundary(consumer)

    def bind_owner(self, owner, adapter):
        if self.method != "B1":
            return super().bind_owner(owner, adapter)

    def admit_on_owner(self, consumer, scene, pending):
        if pending["value"] is None:
            raise ValueError("Owner admission lacks trusted round input")
        index, instruction = pending["value"]
        if scene.instruction != instruction:
            raise ValueError("scene and trusted driver instruction differ")
        checked = ({"result": "PROCEED", "reason": "B1 task gate disabled"}
                   if self.method == "B1" else admit_driver_round(
                       consumer, index, instruction, driver_reset_completed=True))
        self._task = checked
        return checked

    def seal_round(self, consumer, *, primitive_summaries, boundary, resource_end):
        if self.method == "B2":
            consumer.seal_driver_round(boundary=boundary, resource_end=resource_end)
        elif self.method == "B3":
            consumer.boundary.round_open = consumer.boundary.task_approved = False
            consumer.boundary.lease = None
            consumer.adapter.invalidate_binding()
        self._sealed += 1

    async def run_rounds(self, consumer, **kwargs):
        kwargs["require_action_gate_for_recovery"] = self.profile.action_gate
        selected = bridge.selected_bridge(consumer,
            original=self.parent.optin_bridge.S1PersistentSession.original_constructor,
            seal_round=lambda **values: self.seal_round(consumer, **values))
        original = self.parent.optin_core.run_s1_b0_candidate
        namespace = dict(original.__globals__, S1PersistentSession=hooks.boundary_constructor(selected))
        helper = FunctionType(original.__code__, namespace, original.__name__,
                              original.__defaults__, original.__closure__)
        helper.__kwdefaults__ = original.__kwdefaults__
        return await helper(**kwargs)

    def session_constructor(self):
        from research.v4.candidates.s101_v43_async_policy_transport_20260929 import policy_transport
        original = self.parent.optin_core.CodexPersistentSession.original_constructor
        selected = policy_transport.session_class(original, method=self.method,
            approved_policy=self.policy, expected_policy_sha256=self.policy.sha256)
        return hooks.session_constructor(selected)

    def recover_on_owner(self, consumer, *, scene, initial_public_eef, initial_sequence):
        if self._sealed != 1 or self._task is None:
            raise ValueError("original recovery requires the first sealed round")
        if self.method != "B1" and (consumer.boundary.round_open or consumer.boundary.task_approved):
            raise ValueError("recovery must not reopen the user round")
        adapter = self._adapter
        proprio = self._proprio()
        outcome = self._task["result"]
        route = state.route(holding=adapter.estimator.kind, task_outcome=outcome,
            executed_task_calls=consumer.boundary.episode_calls,
            sequence=scene.sequence, initial_sequence=initial_sequence,
            current_public_eef=proprio.eef_xyz, initial_public_eef=initial_public_eef)
        if route in {state.RecoveryRoute.NO_MOTION_TASK, state.RecoveryRoute.NO_MOTION_START}:
            record = state.PublicRecoveryRecord(route.value, 0, False, False, adapter.estimator.kind)
            self.recovery_logs.append({"kind": "recovery_no_motion", **asdict(record)})
            return record
        controls = 1 if route is state.RecoveryRoute.HOLD else 260
        name = "hold" if route is state.RecoveryRoute.HOLD else "move_to"
        parameters = ({"seconds": float(scene.backend.control_timestep)} if controls == 1
                      else {"target_xyz": tuple(initial_public_eef), "max_steps": 260})
        binding = f"executor-recovery:round1:{scene.sequence}"
        lease = None
        action_checked = False
        try:
            if self.method == "B2":
                consumer.begin_executor_recovery()
                workspace = consumer.workspace()
                resolved = consumer.resolve(name, parameters, workspace)
                checked = validate_review_response("B2", 2, consumer.action_review(
                    name=name, parameters=parameters, resolved=resolved, workspace=workspace, lease=None))
                self.recovery_logs.append({"kind": "fresh_recovery_action_review", **checked})
                if checked["result"] != "APPROVE":
                    raise RecoveryIntercepted(checked["reason"])
                consumer.boundary.lease = checked.get("lease")
                if not consumer.lease_is_current(consumer.boundary.lease):
                    raise ValueError("recovery APPROVE requires a fresh B2 text lease")
                geometry = consumer.geometry(resolved, workspace, consumer.boundary.lease)
                if not geometry.approved:
                    raise RecoveryIntercepted(geometry.reason, system_failure=geometry.system_failure)
                binding, lease, action_checked = workspace.version, consumer.boundary._commit_lease(), True
            outlet = AdapterControlOutlet(adapter, action_gate_checked=action_checked)
            outlet.public_holding = lambda: adapter.estimator.kind
            scope = (nullcontext() if self.method == "B1" else adapter.primitive_scope(
                name, parameters=parameters, completion_controls=controls,
                episode=consumer.episode, round_id=consumer.boundary.round_id,
                binding_version=binding, lease=lease))
            with scope:
                if route is state.RecoveryRoute.HOLD:
                    action = np.zeros(7, dtype=np.float32)
                    action[6] = 1.
                    before = scene.sequence
                    outlet.execute(action)
                    if scene.sequence != before + 1 or adapter.estimator.kind is not state.HoldingKind.HELD:
                        raise state.RecoveryCapabilityGap("original CLOSED HOLD did not retain public HELD")
                    record = state.PublicRecoveryRecord(route.value, 1, True, action_checked, adapter.estimator.kind)
                else:
                    record = park_binding.selected_park()(scene, np.asarray(initial_public_eef), outlet)
                terminal = None if self.method == "B1" else adapter.dispatcher.last_outcome
                if terminal is not None:
                    raise RecoveryIntercepted(terminal["reason"],
                        system_failure=terminal.get("system_failure") is True, outcome=dict(terminal))
            if not state.certified(record, task_outcome=outcome, require_action_gate=self.profile.action_gate):
                raise state.RecoveryCapabilityGap("original recovery did not produce a public-state certificate")
            self.recovery_logs.append({"kind": "recovery_original_outlet_completed", **asdict(record)})
            return record
        finally:
            if self.method == "B2":
                consumer.end_executor_recovery()
            elif self.method == "B3":
                adapter.invalidate_binding()

    @staticmethod
    def action_fault(consumer):
        return consumer.task_fault() or any(row.get("system_failure") is True
                                           for row in consumer.action_report()["logs"])

    def commit_fault(self, adapter):
        return False if self.method == "B1" else super().commit_fault(adapter)

    @staticmethod
    def proposal_logs(consumer): return []
    @staticmethod
    def perception_logs(consumer): return []
    def recovery_report(self, consumer): return list(self.recovery_logs)

    def commit_logs(self, adapter):
        return [] if self.method == "B1" else super().commit_logs(adapter)


def core_context(parent, *, method, **kwargs):
    return S1ControlContext(parent, method=method, **kwargs)
