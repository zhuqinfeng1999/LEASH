"""Synchronous pre-consumption hook shared by analytic and VLA controls.

The existing primitive executor remains responsible for making controls.  This
hook never schedules motion or calls hold(seconds).  Waiting is lockstep: the
only send callback is guarded while review is in flight.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import threading
import time
import math
from typing import Callable

from .registry import CommitEventRegistry, Context, LeaseContext, Match, RELEASE
from .run_manifest import record_gate
from .method_inputs import InvalidReviewResponse, is_model_response_failure


class V43Intercepted(Exception):
    def __init__(self, outcome):
        self.outcome = outcome
        super().__init__(outcome["reason"])


class CommitDispatcher:
    def __init__(self, *, registry: CommitEventRegistry, decode: Callable,
                 read_proprio: Callable, send: Callable, capture: Callable,
                 review: Callable, project: Callable, method: str,
                 enabled: bool = True, review_timeout_s: float = 15., clock=time.monotonic):
        if not math.isfinite(review_timeout_s) or review_timeout_s <= 0:
            raise ValueError("finite positive review deadline required")
        self.registry, self.decode = registry, decode
        self.read_proprio, self.send, self.capture = read_proprio, send, capture
        self.review, self.project, self.method = review, project, method
        self.enabled, self.review_timeout_s, self.clock = enabled, review_timeout_s, clock
        self.logs: list[dict] = []
        self.sequence = 0
        self.primitive_id = 0
        self.context: Context | None = None
        self.lease: LeaseContext | None = None
        self.last_outcome = None
        self._owner: int | None = None
        self._lock = threading.RLock()
        self._inflight = False
        self._cancelled = False
        self._chunk_remaining = 0
        self._submitted = 0
        self._expected_release_update = None
        self._release_completion_lease = False
        # Owner shutdown/timeout can arrive on a background caller while the
        # owner holds _lock during review. Never wait for that lock to record
        # cancellation: a late PASS must not release the pending command.
        self._abort_lock = threading.Lock()
        self._abort_requested = threading.Event()
        self._abort_reason = None

    def request_owner_abort(self, reason: str):
        """Terminal original-owner cancellation, not ordinary commit BLOCK.

        This performs no send, hold, state update or queue operation. It cannot
        roll back a command already being consumed by the environment.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("owner cancellation requires an explicit reason")
        with self._abort_lock:
            if not self._abort_requested.is_set():
                self._abort_reason = reason
                self._abort_requested.set()

    def _check_owner_abort(self, *, events=(), match="UNRESOLVED"):
        if self._abort_requested.is_set():
            self._stop(self._abort_reason, system_failure=True, events=events, match=match)

    def preserve_expected_release_completion(self, proprio):
        """Keep only an already-approved stationary non-container opening.

        The G2 owner still expires its binding. Expected loss of the visual
        holding record after a known OPEN is not a new release conversion.
        UNKNOWN remains UNKNOWN; no new permit, anchor or expiry is issued.
        This hook is callable only in that send's public post-state read.
        """
        pending = self._expected_release_update
        permit = self.registry.permits.get(RELEASE)
        estimator = self.registry.estimator
        if (pending is None or permit is not pending['permit'] or self.lease is None
                or not self.lease.valid or not self.lease.reliable_non_container
                or proprio.executed_sequence != self.sequence
                or proprio.aperture_m is None or not math.isfinite(proprio.aperture_m)
                or estimator.last_gripper != 'OPEN'
                or getattr(estimator,'last_visual_invalidation_reason',None) != 'actual OPEN or new CLOSE executed'
                or estimator.generation != pending['expected_generation']):
            return False
        previous_generation = permit.generation
        permit.generation = estimator.generation
        try:
            valid = self.registry._valid_permit(RELEASE,pending['command'],proprio,self.context)
        except (TypeError,ValueError):
            valid = False
        if not valid:
            permit.generation = previous_generation
            return False
        self.lease = replace(self.lease,holding_generation=estimator.generation)
        self.registry._lease_signature = (self.lease.owner,self.lease.version,self.lease.valid,
            self.lease.holding_generation,self.lease.reliable_non_container,self.lease.conditions)
        # This exception covers release completion alone, not a tilt permit.
        self.registry.permits = {RELEASE:permit}
        self._release_completion_lease = True
        self.logs.append(dict(kind='expected_release_completion',primitive=self.primitive_id,
            executed_sequence=self.sequence,issued_sequence=permit.issued_sequence,
            expires_sequence=permit.expires_sequence,G2_binding_retained=False,
            holding_estimate=estimator.kind.value))
        return True

    @contextmanager
    def primitive(self, *, name: str, episode="local", round_id=0,
                  binding_version="unknown", completion_controls: int,
                  lease: LeaseContext | None = None):
        with self._lock:
            if self._abort_requested.is_set():
                raise RuntimeError("original simulator owner stopped; no new primitive may dispatch")
            if self.context is not None or self._inflight:
                raise RuntimeError("nested/concurrent primitive cannot bypass dispatch")
            if completion_controls < 1:
                raise ValueError("original finite primitive completion control bound required")
            if lease is not None and lease.owner != self.method:
                raise ValueError("lease belongs to a different method")
            self._owner = threading.get_ident()
            self.primitive_id += 1
            self.context = Context(episode, round_id, self.primitive_id, binding_version, completion_controls)
            self.lease = lease
            self._cancelled = False
            self._submitted = 0
            self._expected_release_update = None
            self._release_completion_lease = False
            self.last_outcome = None
            self.registry.invalidate()  # Admission does not issue commit permission.
            self.logs.append({"kind": "primitive_start", "name": name, **asdict(self.context)})
        try:
            yield self
        finally:
            with self._lock:
                self.registry.invalidate()
                self.logs.append({"kind": "primitive_end", "primitive": self.primitive_id,
                                  "cancelled": self._cancelled, "sent": self._submitted})
                self.context = None
                self.lease = None
                self._expected_release_update = None
                self._release_completion_lease = False
                self._owner = None

    def _stop(self, reason: str, *, system_failure: bool, events=(), match="UNRESOLVED",
              root_cause=None, reviewer_called=None, decision=None, elapsed_s=None):
        self._cancelled = True
        self.registry.invalidate()
        self.last_outcome = {"status": "system_failure" if system_failure else "intercepted",
                             "reason": reason, "events": list(events), "match": match,
                             "submitted": self._submitted,
                             "candidate_dispatched": False,
                             "discarded_including_candidate": 1 + self._chunk_remaining,
                             "remaining_primitive_cancelled": True,
                             "system_failure": system_failure,
                             "executed_sequence": self.sequence}
        if root_cause is not None:
            self.last_outcome["root_cause"] = root_cause
        reviewed = next((row for row in reversed(self.logs) if row.get("kind") == "review"
                         and row.get("primitive") == self.primitive_id), None)
        called = reviewed is not None if reviewer_called is None else reviewer_called
        self.last_outcome['reviewer_called'] = called
        self.logs.append({"kind": "withheld", **self.last_outcome})
        record_gate(gate=3, trigger=list(events), candidate=self.sequence + 1,
                    epoch=self.primitive_id, decision=decision or ("SYSTEM_FAILURE" if system_failure else "BLOCK"),
                    reason=reason, reviewer_called=called,
                    elapsed_s=elapsed_s if elapsed_s is not None else reviewed["wait_s"] if reviewed else 0.,
                    dispatched=False, invalidated=True)
        raise V43Intercepted(self.last_outcome)

    def step(self, payload):
        with self._lock:
            if self.context is None or self._owner != threading.get_ident():
                raise RuntimeError("no owned active primitive at dispatch")
            if self._cancelled:
                raise V43Intercepted(self.last_outcome)
            self._check_owner_abort()
            if self._inflight:
                self._stop("reentrant dispatch while review is pending", system_failure=True)
            if self._submitted >= self.context.completion_controls:
                self._stop("existing primitive completion bound exhausted", system_failure=True)
            self._inflight = True
            try:
                try:
                    p = self.read_proprio(self.sequence)
                    cmd = self.decode(payload, p)
                    if (self._release_completion_lease and (cmd.gripper != 'OPEN'
                            or not self.registry._valid_permit(RELEASE,cmd,p,self.context))):
                        self.lease = None
                        self.registry.invalidate()
                        self._release_completion_lease = False
                    result = self.registry.classify(cmd, p, self.context, self.lease) if self.enabled else None
                except Exception as exc:
                    self._stop(f"command/proprio decoding failed: {type(exc).__name__}: {exc}", system_failure=True)
                if result is not None and result.state == Match.UNRESOLVED:
                    self._stop(result.reason, system_failure=False, events=result.events,
                               root_cause="technical_unresolved", reviewer_called=False, decision="UNRESOLVED")
                if result is not None and result.state == Match.MATCH:
                    started = self.clock()
                    reviewer_called = False
                    # No send/physics/event advancement occurs while these calls run.
                    try:
                        evidence = self.capture()
                        request = self.project(events=result.events, command=cmd, proprio=p,
                                               classification=result, evidence=evidence,
                                               context=self.context, lease=self.lease)
                        version = request["evidence_version"]
                        reviewer_called = True
                        response = self.review(request)
                        if self.clock() - started > self.review_timeout_s:
                            raise TimeoutError("review deadline expired")
                        if (not isinstance(response, dict) or response.get("decision") not in {"PASS", "BLOCK"}
                                or not isinstance(response.get("reason"), str) or not response["reason"].strip()
                                or tuple(response.get("events", ())) != result.events
                                or response.get("evidence_version") != version):
                            raise InvalidReviewResponse("invalid or mismatched commit response")
                    except Exception as exc:
                        model_failure = is_model_response_failure(exc)
                        self._stop(f"review/evidence failure: {type(exc).__name__}: {exc}",
                                   system_failure=not model_failure, events=result.events, match="MATCH",
                                   root_cause="model_response_failure" if model_failure else "infrastructure_failure",
                                   reviewer_called=reviewer_called,
                                   decision="INVALID" if model_failure else None,
                                   elapsed_s=self.clock() - started)
                    self.logs.append({"kind": "review", "events": list(result.events),
                                      "decision": response["decision"], "reason": response["reason"],
                                      "evidence_version": version, "primitive": self.primitive_id,
                                      "candidate_sequence": self.sequence + 1,
                                      "wait_s": self.clock() - started, "lockstep": True})
                    if response["decision"] == "BLOCK":
                        self._stop(response["reason"], system_failure=False, events=result.events, match="MATCH")
                    if self._cancelled:
                        raise V43Intercepted(self.last_outcome)
                    self._check_owner_abort(events=result.events, match="MATCH")
                    self.registry.grant(result.events, cmd, p, self.context, version)
                # Snapshot only the permission used for this exact send,
                # before execution/observation can expire it or change state.
                # A newly approved moving release is valid for this candidate;
                # its permit's stationary test governs subsequent reuse only.
                used_events = (() if result is None else tuple(dict.fromkeys(
                    (result.events if result.state == Match.MATCH else ()) + result.permit_events)))
                used_permits = [dict(event=event, context=asdict(permit.context),
                    issued_sequence=permit.issued_sequence,
                    expires_sequence=permit.expires_sequence,
                    generation=permit.generation, evidence_version=permit.evidence_version)
                    for event in used_events
                    for permit in (self.registry.permits[event],)]
                # The sole call that can consume a candidate. Whole vector unchanged.
                self._check_owner_abort(events=result.events if result else (),
                                        match=result.state.value if result else "UNRESOLVED")
                permit = self.registry.permits.get(RELEASE)
                estimator = self.registry.estimator
                expected = (self.enabled and self.lease is not None and self.lease.valid
                    and self.lease.reliable_non_container and cmd.gripper == 'OPEN'
                    and self.registry._valid_permit(RELEASE,cmd,p,self.context))
                # Capture before send; a contradictory send callback cannot be
                # reclassified as the expected OPEN update merely by returning.
                generation_before = estimator.generation
                visual_before = estimator.visual_record
                completion_update = (dict(permit=permit,command=cmd,
                    expected_generation=generation_before+int(visual_before is not None))
                    if expected else None)
                try:
                    answer = self.send(cmd.payload)
                except Exception as exc:
                    self._cancelled = True
                    self.registry.invalidate()
                    self.registry.estimator.execution_uncertain()
                    self.last_outcome = {"status": "system_failure", "system_failure": True,
                                         "reason": f"environment send/observation failed: {type(exc).__name__}: {exc}",
                                         "candidate_dispatched": None,
                                         "consumption_unknown": True,
                                         "remaining_primitive_cancelled": True,
                                         "submitted": self._submitted,
                                         "executed_sequence": self.sequence}
                    self.logs.append({"kind": "send_fault", **self.last_outcome})
                    raise V43Intercepted(self.last_outcome) from exc
                self.sequence += 1
                self._submitted += 1
                self._expected_release_update = (completion_update
                    if estimator.generation == generation_before and estimator.visual_record is visual_before else None)
                estimator.record_command(cmd.gripper, p)
                try:
                    estimator.observe(self.read_proprio(self.sequence))
                except Exception as exc:
                    self._cancelled = True
                    self.registry.invalidate()
                    self.last_outcome = {"status": "system_failure", "system_failure": True,
                                         "reason": f"post-dispatch public state failed: {type(exc).__name__}: {exc}",
                                         "candidate_dispatched": True,
                                         "remaining_primitive_cancelled": True,
                                         "submitted": self._submitted,
                                         "executed_sequence": self.sequence}
                    self.logs.append({"kind": "post_dispatch_fault", **self.last_outcome})
                    raise V43Intercepted(self.last_outcome) from exc
                finally:
                    self._expected_release_update = None
                self.logs.append({"kind": "dispatched", "primitive": self.primitive_id,
                                  "executed_sequence": self.sequence,
                                  "match": result.state.value if result else "DISABLED",
                                  "events": list(result.events) if result else [],
                                  "used_permits": used_permits,
                                  "holding_estimate": self.registry.estimator.kind.value,
                                  "initial_empty_active": self.registry.estimator._initial_empty_active,
                                  "R_grasp": (None if self.registry.estimator.grasp_rotation is None
                                              else self.registry.estimator.grasp_rotation.tolist()),
                                  "payload": cmd.payload.tolist()})
                if result is not None and result.state == Match.MATCH:
                    record_gate(gate=3, trigger=list(result.events), candidate=self.sequence,
                                epoch=self.primitive_id, decision="PASS", reason=response["reason"],
                                reviewer_called=True, elapsed_s=self.clock() - started,
                                dispatched=True, invalidated=False)
                return answer
            finally:
                self._inflight = False

    def run_chunk(self, payloads):
        # Never submit a batch to a backend. Every element goes through step().
        items = list(payloads)
        output = []
        try:
            for i, payload in enumerate(items):
                self._chunk_remaining = len(items) - i - 1
                output.append(self.step(payload))
            return output
        finally:
            self._chunk_remaining = 0
            items.clear()  # Remove local queued references after cancellation too.
