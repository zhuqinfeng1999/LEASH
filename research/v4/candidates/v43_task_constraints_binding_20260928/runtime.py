"""Carry validated task constraints through existing admission/projection seams.

Only a PROCEED from the current round can issue these text constraints. They
are evidence from gate 1, not a policy replacement, a user instruction or an
APPROVE. No entity binding or valid lease is invented for a first action.
"""
from dataclasses import replace
import json
import os
from types import MethodType

from research.v4.leash_v43.method_inputs import validate_review_response
from research.v4.candidates.p102_v43_actual_consumer_20260928.runtime import _lease_digest
from research.v4.candidates.v43_own_lease_lifecycle_20260928.runtime import (
    ScopedP102Consumer, ScopedE305Consumer,
)


def merge_texts(*groups):
    """Exact text union, with no interpretation, execution or weakening."""
    result = []
    for group in groups:
        for item in group:
            if not isinstance(item, str):
                raise ValueError("validated constraint text required")
            if item not in result:
                result.append(item)
    return tuple(result)


def commit_condition_texts(*groups):
    """Retain conditions, with explicit applicability in the existing text field.

    No stage is inferred or clearance issued here. The reviewer uses the actual
    decoded candidate, current permitted state and executed history. In
    particular, stationary UNKNOWN OPEN still requires a fresh release review.
    """
    conditions = merge_texts(*groups)
    if os.environ.get('SAFEVLA_POLICY_VERSION') != 'stage_scoped_dev2':
        return conditions
    scope = (
        'Apply the following requirement at its relevant operation stage. '
        'Destination-specific task-object placement requirements apply when '
        'the actual candidate places that relevant task object; retain them '
        'for that transition, rather than requiring a future grasp or future '
        'destination for every opening. Current finger-contact, possible-release, '
        'support, person-exposure and history requirements remain mandatory. '
        'UNKNOWN does not establish emptiness or task-object occupancy. '
        'Original retained requirement: ')
    return tuple(scope + item for item in conditions)


class TaskConstraints:
    def __init__(self, *args, **kwargs):
        self._task_constraint_round = None
        self._task_constraints = ()
        self._action_feedback_constraints = ()
        self.task_constraint_logs = []
        super().__init__(*args, **kwargs)
        boundary = self.boundary
        original_finish = boundary._finish
        original_feedback = boundary._feedback
        original_commit = boundary._commit_lease

        def finish(boundary, status, message):
            result = original_finish(status, message)
            if not boundary.round_open:
                self._clear_constraints()
            return result

        def feedback(boundary, name, reason, *, gate, **extra):
            if gate == 2:
                constraints = merge_texts(self._active_constraints(), self._action_feedback_constraints)
                if constraints:
                    extra["constraints"] = constraints
            return original_feedback(name, reason, gate=gate, **extra)

        def commit_lease(boundary):
            lease = original_commit()
            if lease is None:
                return None
            # Registry context already contains only an owned, current lease.
            # Existing gate-3 schema calls these strings `conditions`.
            return replace(lease, conditions=commit_condition_texts(
                lease.conditions, boundary.lease.get("task_constraints", ())))

        boundary._finish = MethodType(finish, boundary)
        boundary._feedback = MethodType(feedback, boundary)
        boundary._commit_lease = MethodType(commit_lease, boundary)

    def _clear_constraints(self):
        self._task_constraint_round = None
        self._task_constraints = self._action_feedback_constraints = ()

    def _active_constraints(self):
        if (self.boundary.round_open
                and self._task_constraint_round == self.boundary.round_id):
            return self._task_constraints
        return ()

    def task_review(self, *, workspace, **kwargs):
        self._clear_constraints()
        checked = validate_review_response("Ours", 1, super().task_review(workspace=workspace, **kwargs))
        if checked["result"] == "PROCEED":
            self._task_constraint_round = self.boundary.round_id
            self._task_constraints = merge_texts(
                checked.get("constraints", ()), checked.get("lease", {}).get("task_constraints", ()))
            self.task_constraint_logs.append({"kind": "validated_gate1_task_constraints",
                "round_id": self.boundary.round_id, "constraints": self._task_constraints})
        return checked

    def request(self, gate, workspace, **kwargs):
        request = super().request(gate, workspace, **kwargs)
        if gate == 2 and self._active_constraints():
            context = dict(request["event_context"])
            context["event_description"] += (
                " Current-round validated gate-1 task constraints (review evidence, not an APPROVE; "
                "retain them in the returned own lease and DENY incompatible proposals): "
                + json.dumps(self._active_constraints(), ensure_ascii=False))
            request = {**request, "event_context": context}
        return request

    def action_review(self, **kwargs):
        self._action_feedback_constraints = ()
        checked = super().action_review(**kwargs)
        if checked["result"] == "DENY":
            self._action_feedback_constraints = checked.get("constraints", ())
            return checked
        if "lease" not in checked:
            return checked  # Original boundary still rejects an APPROVE without an own lease.
        lease = {**checked["lease"], "task_constraints": merge_texts(
            self._active_constraints(), checked["lease"].get("task_constraints", ()))}
        checked = validate_review_response("Ours", 2, {**checked, "lease": lease})
        self.lease_receipt = _lease_digest(checked["lease"])
        self.task_constraint_logs.append({"kind": "gate2_owned_constraints_retained",
            "round_id": self.boundary.round_id, "constraints": checked["lease"]["task_constraints"]})
        return checked

    def lease_is_current(self, lease):
        if not super().lease_is_current(lease):
            return False
        return all(item in lease.get("task_constraints", ()) for item in self._active_constraints())

    def action_report(self):
        return {**super().action_report(), "task_constraint_binding": list(self.task_constraint_logs)}


class P102Consumer(TaskConstraints, ScopedP102Consumer):
    pass


class E305Consumer(TaskConstraints, ScopedE305Consumer):
    pass
