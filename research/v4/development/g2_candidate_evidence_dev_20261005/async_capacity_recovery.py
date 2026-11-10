"""Bounded same-thread capacity continuation for the existing async collector.

No simulator, task, gate or success state is read or changed. The original
collector journals every failed and recovered SDK turn and owns interruption.
"""
import asyncio
from time import monotonic
from . import capacity_runtime as existing
from .capacity import CapacityBudget, capacity_event

async_sleep = asyncio.sleep
log = existing.log


class ObservedTurn:
    def __init__(self, turn, pending):
        self.turn, self.pending = turn, pending
        self.terminal = None
        self.interrupted = False
        self.unrelated_error = False

    def __getattr__(self, name):
        return getattr(self.turn, name)

    async def interrupt(self):
        self.interrupted = True
        return await self.turn.interrupt()

    async def stream(self):
        async for event in self.turn.stream():
            row = existing.event_dict(event)
            payload = row.get('payload') or {}
            if row['method'] in ('item/started', 'item/completed'):
                item = payload.get('item', {})
                item = item.get('root', item)
                if item.get('type') in ('mcpToolCall', 'dynamicToolCall'):
                    ident = item.get('id')
                    if row['method'] == 'item/started':
                        self.pending.add(ident)
                    elif item.get('status') in ('completed', 'failed'):
                        self.pending.discard(ident)
            if row['method'] in ('error', 'fatal') and not capacity_event(row)['is_capacity']:
                self.unrelated_error = True
            if row['method'] == 'turn/completed':
                self.terminal = row
            yield event


async def drive_with_capacity(session, turn, recorder, boundary, *, max_turns,
                              timeout_s, drive=None):
    # Routing failures have their own exact classifier and retry budget. They
    # are never reclassified as capacity, quota, or a successful model result.
    from .session_routing_recovery import drive_with_routing
    if (getattr(session._thread,'model',None)=='gemini-4-argon-high'
            and getattr(session,'reasoning_effort',None)=='high'
            and getattr(session._thread,'transport_recovery_version',None) ==
            'antigravity-exact-terminal-retained-history-dev1-20261101'):
        from research.v4.integrations.antigravity.transport_recovery import drive_with_transport
        async def original_routing(session, turn, recorder, boundary, **options):
            return await drive_with_routing(session, turn, recorder, boundary,
                **options, capacity_drive=_drive_with_capacity,
                clock=monotonic, sleep=async_sleep)
        return await drive_with_transport(session, turn, recorder, boundary,
            max_turns=max_turns, timeout_s=timeout_s, drive=drive,
            capacity_drive=original_routing, clock=monotonic, sleep=async_sleep)
    return await drive_with_routing(session, turn, recorder, boundary,
        max_turns=max_turns, timeout_s=timeout_s, drive=drive,
        capacity_drive=_drive_with_capacity, clock=monotonic, sleep=async_sleep)


async def _drive_with_capacity(session, turn, recorder, boundary, *, max_turns,
                               timeout_s, drive=None):
    if drive is None:
        from research.v4.candidates.multiround_bridge_20260924.driver import _drive_sdk_turn
        drive = _drive_sdk_turn
    if not existing.enabled():
        return await drive(turn, recorder, boundary, max_turns=max_turns, timeout_s=timeout_s)
    deadline = min(monotonic() + timeout_s, existing.deadline())
    budget = CapacityBudget(deadline=deadline)
    pending = set()
    first_turn = turn.id
    while True:
        observed = ObservedTurn(turn, pending)
        result = await drive(observed, recorder, boundary, max_turns=max_turns,
                             timeout_s=max(.001, deadline - monotonic()))
        terminal = observed.terminal
        if (budget.extra_attempts and result.sdk_completed and not result.error
                and terminal and terminal['payload']['turn'].get('status') == 'completed'):
            budget.finish(thread_id=session._thread.id, turn_id=turn.id)
            log(dict(role='planner', event='async_capacity_recovered', original_turn_id=first_turn,
                     **budget.summary()))
        if (not result.sdk_completed or not result.error or terminal is None
                or observed.interrupted or observed.unrelated_error
                or result.resource_end or boundary() is not None
                or recorder.turns >= max_turns):
            return result
        event = capacity_event(terminal)
        if not event['is_capacity']:
            return result
        if event['thread_id'] != session._thread.id or event['turn_id'] != turn.id:
            log(dict(role='planner', event='async_capacity_withheld', reason='thread_or_turn_mismatch'))
            return result
        plan = budget.plan(terminal)
        if plan.action != 'RETRY':
            log(dict(role='planner', event='async_capacity_stopped', **budget.summary()))
            return result
        try:
            before = existing.safe_snapshot(session.toolkit, pending)
        except RuntimeError as exc:
            log(dict(role='planner', event='async_capacity_withheld', reason=str(exc)))
            return result
        log(dict(role='planner', event='async_capacity_failed', original_turn_id=first_turn,
                 failed_turn_id=turn.id, thread_id=session._thread.id,
                 delay_s=plan.delay_s, owner=before, error=terminal['payload']['turn'].get('error')))
        start = monotonic()
        try:
            while monotonic() - start < plan.delay_s:
                if monotonic() >= deadline or boundary() is not None:
                    return result
                await async_sleep(min(.25, plan.delay_s - (monotonic() - start)))
        finally:
            budget.record_wait(plan, max(0., monotonic() - start))
        try:
            after = existing.safe_snapshot(session.toolkit, pending)
        except RuntimeError as exc:
            log(dict(role='planner', event='async_capacity_withheld', reason=str(exc)))
            return result
        if (before != after or boundary() is not None or monotonic() >= deadline
                or not getattr(session.toolkit.base, 'round_open', True)):
            log(dict(role='planner', event='async_capacity_withheld', reason='owner_boundary_or_deadline_changed'))
            return result
        budget.start_retry(plan)
        prompt = ('The preceding turn ended with a temporary service capacity error. Continue only '
                  'the unfinished work in this same episode and thread. Previously completed tools '
                  'and their results remain in history and must not be replayed. The simulator '
                  f'was frozen during recovery; {before["completed_tools"]} Owner tool receipts have '
                  'completed. Use the existing observations, task, policy, tools and remaining budgets unchanged.')
        if existing.os.environ.get('SAFEVLA_TRANSPORT_WORKSPACE_BUDGET'):
            from research.v4.development.transport_workspace_dev_20261005.requests import record_request
            record_request(retry=True)
        # Clear only after a terminal capacity failure and a legal continuation.
        # The failed events remain in the original planner stream journal.
        recorder.error = None
        options = {k: v for k, v in session._planner._turn_options.items()
                   if k != 'base_instructions'}
        turn = await session._thread.turn(prompt, **options)
        log(dict(role='planner', event='async_capacity_continuation_started', original_turn_id=first_turn,
                 turn_id=turn.id, owner=after, **budget.summary()))
