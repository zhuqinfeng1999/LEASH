"""Bounded capacity recovery policy; no SDK, robot, simulator or file access.

Input is a real error/turn-completed event from the existing collector, never a
model reply. The caller retains the thread, unfinished request and robot state.
Only the caller may submit a retry, after checking its execution receipts.
"""
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import math
import random
import time


CAPACITY_MESSAGE = "Selected model is at capacity. Please try a different model."
DEFAULT_DELAYS = (10., 20., 40., 80., 120., 120.)


def _field(obj, *names):
    return next((obj[name] for name in names if name in obj), None)


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _capacity_tag(info):
    return (info == "serverOverloaded" or isinstance(info, dict) and
            ("serverOverloaded" in info or
             _field(info, "type", "kind") == "serverOverloaded"))


def _retry_after(error, payload, event, wall_time):
    for obj in (error, payload, event):
        value = _field(obj, "retryAfter", "retry_after", "retryAfterSeconds", "retry_after_s")
        if value is None and isinstance(obj.get("headers"), dict):
            value = next((v for k, v in obj["headers"].items()
                          if str(k).lower() == "retry-after"), None)
        if value is None:
            continue
        seconds = _number(value)
        if seconds is not None:
            return seconds
        if isinstance(value, str):
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is not None:
                    return max(0., date.timestamp() - wall_time)
            except (TypeError, ValueError, OverflowError):
                pass
        return None
    return None


def capacity_event(event, *, turn_status=None, wall_time=None):
    """Classify only service error channels; missing metadata remains None.

    Supports raw App Server ``params`` and the installed SDK's ``payload``;
    snake/camel field names are the two existing transport representations.
    """
    event = event if isinstance(event, dict) else {}
    method = event.get("method")
    payload = event.get("payload", event.get("params", {}))
    payload = payload if isinstance(payload, dict) else {}
    turn = payload.get("turn", {})
    turn = turn if isinstance(turn, dict) else {}
    status = turn.get("status", turn_status)
    error = turn.get("error") if method == "turn/completed" else payload.get("error")
    channel = method == "error" or method == "turn/completed" and status == "failed"
    error = error if isinstance(error, dict) else {}
    message = error.get("message")
    info = _field(error, "codexErrorInfo", "codex_error_info")
    capacity = channel and (_capacity_tag(info) or message == CAPACITY_MESSAGE)
    variant = info.get("serverOverloaded", {}) if isinstance(info, dict) else {}
    variant = variant if isinstance(variant, dict) else {}
    http = _field(error, "httpStatus", "http_status", "httpStatusCode", "http_status_code")
    if http is None:
        http = _field(variant, "httpStatus", "http_status", "httpStatusCode", "http_status_code")
    if http is None and isinstance(info, dict):
        http = _field(info, "httpStatus", "http_status", "httpStatusCode", "http_status_code")
    will_retry = _field(payload, "willRetry", "will_retry")
    if not isinstance(will_retry, bool):
        will_retry = None
    return dict(is_capacity=bool(capacity), service_error_channel=bool(channel),
                message=message if isinstance(message, str) else None,
                codex_error_info=info, http_status=http,
                thread_id=_field(payload, "threadId", "thread_id"),
                turn_id=turn.get("id", _field(payload, "turnId", "turn_id")),
                turn_status=status, will_retry=will_retry,
                retry_after_s=_retry_after(error, payload, event,
                    time.time() if wall_time is None else wall_time) if channel else None)


@dataclass(frozen=True)
class RetryPlan:
    action: str
    reason: str
    delay_s: float = 0.
    extra_attempt: int = 0
    turn_id: object = None


class CapacityBudget:
    """One logical request's retry budget, with no request/session API.

    ``plan`` does not consume an attempt. After the caller's bounded wait,
    ``record_wait`` accounts actual elapsed time; ``start_retry`` consumes an
    attempt only immediately before its same-thread request. ``finish`` seals
    one valid logical result (including a DENY). A failed turn must be confirmed
    before application recovery; ongoing SDK retry notifications only wait.
    """
    def __init__(self, *, max_extra_attempts=6, max_wait_s=600., deadline=None,
                 clock=time.monotonic, wall_clock=time.time, jitter=None):
        if (not isinstance(max_extra_attempts, int) or isinstance(max_extra_attempts, bool)
                or not 0 <= max_extra_attempts <= len(DEFAULT_DELAYS)):
            raise ValueError("At most six extra attempts are authorized")
        wait_limit = _number(max_wait_s)
        if wait_limit is None or wait_limit > 600:
            raise ValueError("Backoff budget must be within 600 seconds")
        if deadline is not None and _number(deadline) is None:
            raise ValueError("deadline must be a finite monotonic timestamp")
        self.max_extra_attempts = max_extra_attempts
        self.max_wait_s = wait_limit
        self.deadline = deadline
        self.clock, self.wall_clock = clock, wall_clock
        self.jitter = jitter or (lambda: random.uniform(0., 1.))
        self.extra_attempts, self.waited_s = 0, 0.
        self.first_turn_id = self.final_turn_id = self.thread_id = None
        self.finished = False
        self.records = []
        self._pending = None
        self._wait_complete = False

    def _remaining(self, remaining_wall_s):
        values = []
        if self.deadline is not None:
            values.append(self.deadline - self.clock())
        if remaining_wall_s is not None:
            number = _number(remaining_wall_s)
            values.append(-1. if number is None else number)
        return min(values) if values else math.inf

    def plan(self, event, *, turn_status=None, remaining_wall_s=None):
        e = capacity_event(event, turn_status=turn_status, wall_time=self.wall_clock())
        if remaining_wall_s is not None and _number(remaining_wall_s) is not None:
            deadline = self.clock() + float(remaining_wall_s)
            self.deadline = min(self.deadline, deadline) if self.deadline is not None else deadline
        if (not self.finished and e["is_capacity"] and e["thread_id"] and e["turn_id"] and
                (self.thread_id is None or self.thread_id == e["thread_id"])):
            self.thread_id = e["thread_id"]
            if self.first_turn_id is None:
                self.first_turn_id = e["turn_id"]
            self.final_turn_id = e["turn_id"]
        action, reason, delay = "STOP", "not_capacity", 0.
        if self.finished:
            action, reason = "COMPLETE", "valid_result_already_accepted"
        elif self._pending is not None:
            action, reason = "WAIT", "application_retry_already_pending"
        elif e["is_capacity"]:
            if e["will_retry"] is True and e["turn_status"] != "failed":
                action, reason = "WAIT_SDK", "same_turn_sdk_recovery"
            elif e["turn_status"] != "failed":
                action, reason = "WAIT_TURN", "final_failed_turn_not_confirmed"
            elif not e["thread_id"] or not e["turn_id"]:
                reason = "missing_thread_or_turn_correlation"
            elif self.thread_id is not None and e["thread_id"] != self.thread_id:
                reason = "thread_changed"
            elif self.extra_attempts >= self.max_extra_attempts:
                reason = "extra_attempt_budget_exhausted"
            else:
                jitter = _number(self.jitter())
                if jitter is None or jitter > 1.:
                    raise ValueError("Small jitter must be within [0, 1] seconds")
                delay = max(DEFAULT_DELAYS[self.extra_attempts], e["retry_after_s"] or 0.) + jitter
                if delay > self.max_wait_s - self.waited_s:
                    reason = "backoff_budget_exhausted"
                elif delay >= self._remaining(remaining_wall_s):
                    reason = "episode_wall_exhausted"
                else:
                    action, reason = "RETRY", "failed_capacity_turn"
                    self.thread_id = e["thread_id"]
                    if self.first_turn_id is None:
                        self.first_turn_id = e["turn_id"]
                    self.final_turn_id = e["turn_id"]
        plan = RetryPlan(action, reason, delay, self.extra_attempts + 1 if action == "RETRY" else 0,
                         e["turn_id"])
        self.records.append(dict(e, action=action, reason=reason,
                                 delay_s=delay, extra_attempt=plan.extra_attempt))
        if action == "RETRY":
            self._pending, self._wait_complete = plan, False
        return plan

    def record_wait(self, plan, elapsed_s):
        """Record an awaited/synchronous wait; interrupted waits never retry early."""
        if plan is not self._pending or _number(elapsed_s) is None:
            raise ValueError("No matching pending retry wait")
        elapsed_s = float(elapsed_s)
        self.waited_s += elapsed_s
        self._wait_complete = elapsed_s >= plan.delay_s
        self.records.append(dict(action="WAITED", turn_id=plan.turn_id,
                                 elapsed_s=elapsed_s, total_backoff_s=self.waited_s))
        return self._wait_complete

    def wait(self, plan, *, sleep):
        """Optional synchronous adapter; tests inject a fake clock and sleep."""
        if plan is not self._pending:
            raise ValueError("No matching pending retry")
        start = self.clock()
        try:
            sleep(plan.delay_s)
        finally:
            self.record_wait(plan, max(0., self.clock() - start))

    def start_retry(self, plan, *, remaining_wall_s=None):
        if plan is not self._pending or not self._wait_complete:
            raise ValueError("Retry requires its complete bounded wait")
        if (self.finished or self.waited_s > self.max_wait_s or
                self._remaining(remaining_wall_s) <= 0):
            self._pending = None
            raise TimeoutError("Capacity recovery budget expired before request")
        self.extra_attempts += 1
        self._pending = None
        self.records.append(dict(action="RETRY_STARTED", extra_attempt=self.extra_attempts,
                                 prior_turn_id=plan.turn_id, thread_id=self.thread_id))

    def finish(self, *, thread_id, turn_id):
        """Called only after the caller accepts a complete, schema-valid result."""
        if self.finished:
            raise ValueError("Logical result already accepted")
        if self.thread_id is not None and thread_id != self.thread_id:
            raise ValueError("Recovery cannot substitute another thread")
        self.thread_id, self.final_turn_id = thread_id, turn_id
        if self.first_turn_id is None:
            self.first_turn_id = turn_id
        self.finished = True
        self._pending = None
        self.records.append(dict(action="VALID_RESULT", first_turn_id=self.first_turn_id,
                                 final_turn_id=turn_id, thread_id=thread_id))

    def summary(self):
        return dict(extra_attempts=self.extra_attempts, total_backoff_s=self.waited_s,
                    first_turn_id=self.first_turn_id, final_turn_id=self.final_turn_id,
                    thread_id=self.thread_id, valid_logical_results=int(self.finished))
