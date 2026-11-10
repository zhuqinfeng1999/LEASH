"""Fresh own recovery admission at the existing private Owner callback.

UNCONNECTED DRAFT: parent own Consumer and main selection stopped their first
checks. This module has not been imported, compiled, instantiated or tested.
The user round stays sealed. There is no base reset, tool call, new controller,
planner turn or reused task APPROVE. Physical control remains in the original
selected park routine or the original one-command closed HOLD.
"""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np

from research.v4.candidates.p102_v43_actual_consumer_20260928.runtime import _lease_digest
from research.v4.candidates.s101_v43_round_consumer_20260929.runtime import ProvidedCommonConsumer
from research.v4.candidates.s101_v43_recovery_state_20260929 import state, park_binding
from research.v4.candidates.s1_multiround_driver_20260925.recovery import AdapterControlOutlet
from research.v4.candidates.v43_own_lease_lifecycle_20260928.runtime import lease_entities
from research.v4.candidates.v43_task_constraints_binding_20260928.runtime import merge_texts
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PrimitiveProposal
from research.v4.leash_v43.method_inputs import validate_review_response
from research.v4.leash_v43.tool_boundary import ResolvedPrimitive


class RecoveryIntercepted(RuntimeError):
    def __init__(self, reason, *, system_failure=False, outcome=None):
        super().__init__(reason)
        self.system_failure = system_failure
        self.outcome = outcome


class RecoveryScope:
    """Metadata scope over a selected own Consumer, never another executor."""
    def __init__(self, *args, **kwargs):
        self._recovery_context = None
        self._recovery_number = 0
        self.recovery_logs = []
        super().__init__(*args, **kwargs)

    def _active_constraints(self):
        if self._recovery_context is not None:
            return self._recovery_context["sealed_constraints"]
        return super()._active_constraints()

    def current_public_entities(self):
        names = set(super().current_public_entities())
        if self._recovery_context is not None:
            # Public semantic names are historical evidence, never a current
            # lease or permission. No text constraints/reasons are parsed.
            checked = self._validated_task or {}
            if checked.get("result") == "PROCEED" and checked.get("lease") is not None:
                names.update(lease_entities(checked["lease"]))
        return tuple(sorted(names))

    def workspace(self):
        if self._recovery_context is None:
            return super().workspace()
        self.acquire += 1
        public = self.provider()
        packet = self.provider.current_packet
        self._expire_if_holding_changed()
        workspace = SimpleNamespace(public=public, packet=packet, version="pending", acquisition="pending")
        return self._version(workspace)

    def _version(self, workspace):
        context = self._recovery_context
        if context is None:
            return super()._version(workspace)
        self.binding_version = (f"{self.episode}:round{self.boundary.round_id}:recovery{context['number']}"
                                f":holding{self.adapter.estimator.generation}:binding{self._binding_revision}")
        workspace.version = self.binding_version
        workspace.acquisition = f"{workspace.version}:frame{workspace.packet['frame_id']}:capture{self.acquire}"
        return workspace

    def request(self, gate, workspace, **kwargs):
        context = self._recovery_context
        if context is None:
            return super().request(gate, workspace, **kwargs)
        if gate != 2:
            raise ValueError("internal recovery requests only fresh action admission")
        # Avoid TaskConstraints' current-round wording after the round ended.
        # Project the same permitted current views/proprio and sealed history.
        request = ProvidedCommonConsumer.request(self, gate, workspace, **kwargs)
        event = dict(request["event_context"])
        event["event_description"] += (
            " Internal executor observation recovery after the user round was sealed. "
            "This is a fresh gate-2 proposal; no prior task decision or expired lease authorizes it. "
            "Previously validated task constraints are sealed review evidence to retain: "
            + json.dumps(context["sealed_constraints"], ensure_ascii=False))
        return {**request, "event_context": event}

    def lease_is_current(self, lease):
        context = self._recovery_context
        if context is None:
            return super().lease_is_current(lease)
        self._expire_if_holding_changed()
        return bool(not self.boundary.round_open and not self.boundary.task_approved
                    and lease is not None and lease is self.boundary.lease
                    and self.lease_receipt is not None and self._lease_holding == self._holding()
                    and lease.get("valid") is True and lease.get("owner_method") == "Ours"
                    and lease.get("binding_version") == self.binding_version
                    and _lease_digest(lease) == self.lease_receipt
                    and all(x in lease.get("task_constraints", ()) for x in context["sealed_constraints"]))

    def recover_on_owner(self, *, scene, initial_public_eef, initial_sequence):
        if (self._recovery_context is not None or self.boundary.round_open or self.boundary.task_approved
                or len(self.sealed_history) != 1 or self._start_keyframes is not None
                or self.boundary.lease is not None or self.lease_receipt is not None):
            raise ValueError("recovery requires an already-sealed first round and expired user lease")
        checked = self._validated_task or {}
        outcome = checked.get("result")
        proprio = self.adapter.dispatcher.read_proprio(self.adapter.dispatcher.sequence)
        route = state.route(
            holding=self.adapter.estimator.kind, task_outcome=outcome,
            executed_task_calls=self.base.episode_calls, sequence=scene.sequence,
            initial_sequence=initial_sequence, current_public_eef=proprio.eef_xyz,
            initial_public_eef=initial_public_eef)
        if route in {state.RecoveryRoute.NO_MOTION_TASK, state.RecoveryRoute.NO_MOTION_START}:
            record = state.PublicRecoveryRecord(route.value, 0, False, False, self.adapter.estimator.kind)
            self.recovery_logs.append({"kind": "recovery_no_motion", **asdict(record)})
            return record
        if outcome != "PROCEED":
            raise ValueError("physical recovery only follows a completed admitted first round")
        self._recovery_number += 1
        from research.v4.candidates.s101_v43_own_public_consumer_20260929.runtime import gate1_constraints
        constraints = gate1_constraints(checked)
        self._recovery_context = {"number": self._recovery_number, "sealed_constraints": constraints}
        # No task/round lifecycle field is opened here. This is only a fresh
        # internal own binding consumed by the existing project-current hook.
        self._clear_round_binding("separate executor recovery begins")
        try:
            workspace = self.workspace()
            start = tuple(float(x) for x in workspace.public.eef_xyz)
            target = tuple(float(x) for x in initial_public_eef)
            proposal = (PrimitiveProposal("hold", operated_entity="pan", changes_grasp_state=None)
                        if route is state.RecoveryRoute.HOLD else PrimitiveProposal(
                            "move_to", operated_entity=None, motion_waypoints=(start, target),
                            gripper_radius_m=.035, geometry_ambiguous=True,
                            parameters=(("recovery", "open_gripper_observation_park"),)))
            resolved = ResolvedPrimitive(workspace.version, proposal,
                                         entities=("pan",) if route is state.RecoveryRoute.HOLD else (),
                                         geometry_unknown=proposal.geometry_ambiguous)
            parameters = ({"seconds": float(scene.env.env.control_timestep)}
                          if route is state.RecoveryRoute.HOLD else {"target_xyz": target, "max_steps": 260})
            response = validate_review_response("Ours", 2, self.action_review(
                name=proposal.name, parameters=parameters, resolved=resolved, workspace=workspace,
                lease=None, instruction=self.instruction, episode=self.episode,
                round_id=self.boundary.round_id))
            self.recovery_logs.append({"kind": "fresh_recovery_action_review", **response})
            if response["result"] != "APPROVE":
                raise RecoveryIntercepted(response["reason"])
            self.boundary.lease = response.get("lease")
            if not self.lease_is_current(self.boundary.lease):
                raise ValueError("fresh recovery APPROVE lacks a current own lease")
            geometry = self.geometry(resolved, workspace, self.boundary.lease)
            if not geometry.approved:
                raise RecoveryIntercepted(geometry.reason, system_failure=geometry.system_failure)
            controls = 1 if route is state.RecoveryRoute.HOLD else 260
            adapter = self.adapter
            # Actual original control outlet CodeType; only the public holding
            # callback required by the already-reviewed park binding is added.
            outlet = AdapterControlOutlet(adapter, action_gate_checked=True)
            outlet.public_holding = lambda: adapter.estimator.kind
            with adapter.primitive_scope(
                    proposal.name, parameters=parameters, completion_controls=controls,
                    episode=self.episode, round_id=self.boundary.round_id,
                    binding_version=workspace.version, lease=self.boundary._commit_lease()):
                if route is state.RecoveryRoute.HOLD:
                    action = np.zeros(7, dtype=np.float32)
                    action[6] = 1.
                    before = scene.sequence
                    outlet.execute(action)
                    if scene.sequence != before + 1 or adapter.estimator.kind is not state.HoldingKind.HELD:
                        raise state.RecoveryCapabilityGap("public HELD was not retained by the original closed HOLD")
                    record = state.PublicRecoveryRecord(route.value, 1, True, True, adapter.estimator.kind)
                else:
                    record = park_binding.selected_park()(scene, np.asarray(target, dtype=float), outlet)
                terminal = adapter.dispatcher.last_outcome
                if terminal is not None:
                    raise RecoveryIntercepted(terminal["reason"],
                        system_failure=terminal.get("system_failure") is True, outcome=dict(terminal))
            if not state.certified(record, task_outcome=outcome, require_action_gate=True):
                raise state.RecoveryCapabilityGap("original physical recovery did not produce a public-state certificate")
            self.recovery_logs.append({"kind": "recovery_original_outlet_completed", **asdict(record)})
            return record
        finally:
            self._clear_round_binding("separate executor recovery ends")
            self._recovery_context = None
