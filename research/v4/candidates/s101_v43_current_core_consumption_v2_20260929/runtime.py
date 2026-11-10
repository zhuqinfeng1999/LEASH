"""Actual core callback bindings over the existing Owner and deferred driver.

No runner, queue, primitive executor or planner session is implemented here.
The selected original core calls these methods at its existing seams.
"""
from research.v4.leash_v43.run_manifest import source_custody_enabled
from dataclasses import asdict
from types import FunctionType

from research.v4.candidates.p102_v43_actual_consumer_20260928 import runtime as common
from research.v4.candidates.s101_v43_own_public_consumer_geometry_v2_20260929 import runtime as own, sources as own_sources
from research.v4.candidates.s101_v43_bridge_round_binding_20260929 import bridge
from research.v4.candidates.s101_v43_async_policy_transport_20260929 import policy_transport
from research.v4.candidates.s1_s2_deferred_production_optin_20260927 import hooks
from research.v4.leash_v43.adapter import V43PublicEnvAdapter
from research.v4.leash_v43.owner_binding import bind_owner_abort
from research.v4.leash_v43.reviewer_transport import ReviewerSession
from .recovery import RecoveryScope


class S101CoreConsumer(RecoveryScope, own.S101OwnPublicConsumer):
    pass


class OwnerTaskGate:
    """Use the existing secret Owner callback, with current trusted round data."""
    def __init__(self, context, consumer, pending, owner, callbacks):
        self.context, self.consumer, self.pending = context, consumer, pending
        self.owner, self.callbacks = owner, callbacks
        self.records, self.task_decisions = [], []

    def __call__(self, round_input, history):
        if (self.pending["value"] is not None or len(history) != self.context.sealed_rounds(self.consumer)
                or round_input.index != len(history) + 1
                or round_input.instruction != self.context.parent.own_components.instructions[round_input.index - 1]):
            raise ValueError("actual driver current instruction/history mismatch")
        self.pending["value"] = (round_input.index, round_input.instruction)
        try:
            checked = self.callbacks.invoke(self.owner, "__s1_capture_review_evidence__")
        finally:
            self.pending["value"] = None
        record = self.context.parent.own_components.wiring.S1TaskReviewRecord(
            round_input.index, checked["result"], checked["reason"], False, len(history))
        self.records.append(record)
        self.task_decisions.append(checked)
        return self.context.parent.own_components.bridge.TaskGateDecision(
            checked["result"], checked["reason"])


class CoreBindings:
    def __init__(self, parent, *, policy, calibration, source_hashes):
        self.parent, self.policy, self.calibration = parent, policy, calibration
        self.source_hashes = source_hashes
        self.vla_chunk_controls = None
        self.last_consumer = None
        self.operational_calls = []

    @staticmethod
    def sealed_rounds(consumer): return len(consumer.sealed_history)
    @staticmethod
    def consumer_workspace(consumer): return consumer.workspace
    @staticmethod
    def consumer_provider(consumer): return consumer.provider
    @staticmethod
    def proposal_logs(consumer): return consumer.proposals
    @staticmethod
    def perception_logs(consumer): return consumer.provider.log
    @staticmethod
    def recovery_report(consumer): return consumer.recovery_logs

    def require_interfaces(self, method, adapter, reviewer, workspace,
                           locate_vla_target, recovery_on_owner, diagnostic_only):
        if method != "Ours" or workspace is not None or locate_vla_target is not None or recovery_on_owner is not None:
            raise ValueError("only original S101 Ours core with its own provider/recovery is selected")
        if not isinstance(adapter, V43PublicEnvAdapter) or not callable(getattr(adapter, "primitive_scope", None)):
            raise ValueError("actual current v4.3 adapter/dispatcher interface required")
        if reviewer is None or adapter.policy is not self.policy:
            raise ValueError("actual independent reviewer / selected policy identity required")
        if not diagnostic_only and (not isinstance(reviewer, ReviewerSession) or not reviewer.config.enable_live_model):
            raise ValueError("actual independent enabled v4.3 ReviewerSession required")
        if source_custody_enabled():
            self.source_hashes()

    def make_consumer(self, public_base, adapter, scene, sam_client, reviewer, instruction, episode):
        if source_custody_enabled():
            self.source_hashes()
            own_sources.source_hashes()
        primitives = public_base.original._primitives
        consumer = S101CoreConsumer(public_base, adapter,
            components=own.own_components(), scene=scene, sam_client=sam_client,
            reviewer=reviewer, policy=self.policy, instruction=instruction,
            episode=episode, primitives=primitives,
            vla_chunk_controls=self.vla_chunk_controls)
        self.last_consumer = consumer
        self.operational_calls.append("make_consumer")
        return consumer

    @staticmethod
    def observe_boundary(consumer):
        return bridge.DriverObservationBoundary(consumer)

    @staticmethod
    def admit_on_owner(consumer, scene, pending):
        if pending["value"] is None:
            raise ValueError("original Owner capture lacks its trusted current round")
        index, instruction = pending["value"]
        if scene.instruction != instruction:
            raise ValueError("actual scene instruction and driver instruction differ")
        # The existing callback now returns this same acquisition to admit_round.
        # Driver begin_round already ran; bridge owns finish and seals history.
        return consumer.admit_round(index, instruction, scene.capture_agent(),
                                    driver_reset_completed=True)

    def task_adapter(self, consumer, pending, owner, callbacks):
        return OwnerTaskGate(self, consumer, pending, owner, callbacks)

    @staticmethod
    def bind_owner(owner, adapter):
        return bind_owner_abort(owner, adapter)

    async def run_rounds(self, consumer, **kwargs):
        selected = bridge.selected_bridge(consumer,
            original=self.parent.optin_bridge.S1PersistentSession.original_constructor)
        original = self.parent.optin_core.run_s1_b0_candidate
        namespace = dict(original.__globals__, S1PersistentSession=hooks.boundary_constructor(selected))
        helper = FunctionType(original.__code__, namespace, original.__name__,
                              original.__defaults__, original.__closure__)
        helper.__kwdefaults__ = original.__kwdefaults__
        return await helper(**kwargs)

    def session_constructor(self):
        original = self.parent.optin_core.CodexPersistentSession.original_constructor
        selected = policy_transport.session_class(original, method="Ours",
            approved_policy=self.policy, expected_policy_sha256=self.policy.sha256)
        return hooks.session_constructor(selected)

    @staticmethod
    def action_fault(consumer):
        return any(row.get("system_failure") is True for row in consumer.boundary.logs)

    @staticmethod
    def commit_fault(adapter):
        return any(row.get("system_failure") is True for row in adapter.dispatcher.logs)

    @staticmethod
    def task_logs(consumer):
        return consumer.task_report()["logs"]

    @staticmethod
    def action_logs(consumer):
        return consumer.action_report()["logs"]

    @staticmethod
    def commit_logs(adapter):
        return common.commit_report(adapter)["logs"]
