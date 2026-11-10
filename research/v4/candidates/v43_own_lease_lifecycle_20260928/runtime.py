"""Expire own bindings using already-acquired public results and executed state.

No new perception, physics, trigger or model call. Raw image/acquisition changes
are not semantic changes. Unknown public boxes remain unknown; they are not
filled using object labels, assisted latch, evaluator geometry or private poses.
"""
from dataclasses import replace
import math
from types import MethodType

from research.v4.candidates.p102_v43_actual_consumer_20260928.runtime import Consumer as P102Consumer
from research.v4.candidates.e305_v43_current_consumer_20260928.runtime import Consumer as E305Consumer


def lease_entities(lease):
    names = set(lease.get("entities", ()))
    for region in lease.get("no_go_regions", ()):
        names.add(region if isinstance(region, str) else region["name"])
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("validated lease entity names required")
    return tuple(sorted(names))


def public_binding(public, names):
    """Read only existing PublicWorkspace.box results, never acquire an image."""
    result = []
    for name in names:
        box = public.box(name)
        if box is None:
            result.append((name, None))
            continue
        lo, hi = tuple(box.lower), tuple(box.upper)
        if (box.source != "public_rgbd_perception" or len(lo) != 3 or len(hi) != 3
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                       for x in (*lo, *hi)) or any(a > b for a, b in zip(lo, hi))):
            raise ValueError("existing public binding is malformed")
        result.append((name, (lo, hi)))
    return tuple(result)


class OwnLeaseLifecycle:
    """Mixin over the actual current own Consumer, not another executor."""
    def __init__(self, *args, **kwargs):
        self._lease_public = None
        self._lease_holding = None
        self._lease_phase = None
        self._public_phase = None
        self._binding_revision = 0
        self.lease_lifecycle_logs = []
        super().__init__(*args, **kwargs)
        # These are the actual selected per-command callbacks. Keep each
        # original proprio read/project call and its control/camera behavior.
        dispatcher = self.adapter.dispatcher
        read, project = dispatcher.read_proprio, dispatcher.project
        def read_proprio(sequence):
            value = read(sequence)
            self._expire_if_holding_changed(value)
            return value
        def project_current(**packet):
            if not self.lease_is_current(self.boundary.lease):
                packet["lease"] = None
            return project(**packet)
        dispatcher.read_proprio, dispatcher.project = read_proprio, project_current
        original_commit_lease = self.boundary._commit_lease
        def commit_lease(boundary):
            if not self.lease_is_current(boundary.lease):
                return None
            value = original_commit_lease()
            visual = getattr(self.adapter.estimator,'visual_record',None)
            return replace(value, holding_generation=self._lease_holding[1],
                reliable_non_container=bool(visual is not None and visual['valid']
                    and visual['category']=='non_container'))
        self.boundary._commit_lease = MethodType(commit_lease, self.boundary)

    def _holding(self):
        # The existing estimator updates only after known environment sends.
        return self.adapter.estimator.kind.value, self.adapter.estimator.generation

    def _expire(self, reason, *, commit_completion_preserved=False):
        if self.lease_receipt is None:
            return
        self.lease_lifecycle_logs.append({"kind": "own_binding_expired", "reason": reason,
            "binding_version": self.binding_version, "holding": self._holding()})
        self.lease_receipt = None
        self._lease_public = self._lease_holding = self._lease_phase = None
        self.boundary.lease = None
        self.boundary.first_action = True
        self._binding_revision += 1
        if not commit_completion_preserved:
            self.adapter.invalidate_binding()  # Existing permit invalidation only.

    def _expire_if_holding_changed(self, proprio=None):
        if self._lease_holding is not None and self._holding() != self._lease_holding:
            preserve = getattr(self.adapter.dispatcher,'preserve_expected_release_completion',None)
            kept = bool(proprio is not None and preserve is not None and preserve(proprio))
            self._expire("executed/public holding lifecycle changed",commit_completion_preserved=kept)

    def _version(self, workspace):
        base = f"{self.episode}:1:holding{self.adapter.estimator.generation}"
        # A G2 state update and its candidate share this exact acquisition.
        # Subsequent workspace acquisition receives its new holding version.
        self.binding_version = getattr(workspace,'reviewed_binding_version',
            f"{base}:binding{self._binding_revision}")
        workspace.version = self.binding_version
        workspace.acquisition = (f"{workspace.version}:frame{workspace.packet['frame_id']}"
                                 f":capture{self.acquire}")
        return workspace

    def workspace(self):
        workspace = super().workspace()
        self._expire_if_holding_changed()
        lease = self.boundary.lease
        if self.lease_receipt is not None and lease is not None:
            observed = public_binding(workspace.public, lease_entities(lease))
            if observed != self._lease_public:
                self._expire("already-acquired related public entity/location changed or became unknown")
        return self._version(workspace)

    def resolve(self, name, parameters, workspace):
        resolved = super().resolve(name, parameters, workspace)
        # Reuse only the existing own proposal's explicit phase annotation.
        # Absence is unknown, not a claim that intrusion/heat/tilt is absent.
        phases = tuple(value for key, value in resolved.proposal.parameters if key == "phase")
        if len(phases) > 1 or any(not isinstance(value, str) for value in phases):
            raise ValueError("existing public proposal phase schema changed")
        self._public_phase = phases[0] if phases else None
        if self.lease_receipt is not None and self._public_phase != self._lease_phase:
            self._expire("existing public proposal phase changed or became unknown")
        self._version(workspace)
        if (getattr(workspace,'reviewed_binding_version',None) is not None
                and self._lease_holding == self._holding() and self.boundary.lease is not None):
            self._lease_public = public_binding(workspace.public,lease_entities(self.boundary.lease))
        return replace(resolved, binding_version=workspace.version)

    def action_review(self, *, workspace, **kwargs):
        response = super().action_review(workspace=workspace, **kwargs)
        lease = response.get("lease")
        if response["result"] == "APPROVE" and lease is not None:
            self._lease_public = public_binding(workspace.public, lease_entities(lease))
            self._lease_holding = self._holding()
            self._lease_phase = self._public_phase
            self.lease_lifecycle_logs.append({"kind": "own_binding_issued",
                "binding_version": lease["binding_version"], "entities": lease_entities(lease),
                "holding": self._lease_holding, "phase": self._lease_phase})
        return response

    def lease_is_current(self, lease):
        self._expire_if_holding_changed()
        if lease is None or self._lease_holding is None:
            return False
        return super().lease_is_current(lease)

    def action_report(self):
        return {**super().action_report(), "own_binding_lifecycle": list(self.lease_lifecycle_logs)}


class ScopedP102Consumer(OwnLeaseLifecycle, P102Consumer):
    pass


class ScopedE305Consumer(OwnLeaseLifecycle, E305Consumer):
    pass
