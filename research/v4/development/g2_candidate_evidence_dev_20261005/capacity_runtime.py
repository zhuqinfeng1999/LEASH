"""Development-only same-thread capacity continuation, preserving tool receipts."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
from unittest.mock import patch
from .capacity import CapacityBudget, capacity_event

VERSION='local-candidate-capacity-dev3-20261005'
_local=threading.local()
_owners={}
_owner_lock=threading.Lock()

def enabled():return os.environ.get('SAFEVLA_CAPACITY_RECOVERY')==VERSION

def deadline():
    value=os.environ.get('SAFEVLA_CAPACITY_DEADLINE')
    return float(value) if value else time.monotonic()+2700.

def event_dict(event):
    if isinstance(event,dict):return event
    p=getattr(event,'payload',None)
    p=p.model_dump(mode='json',by_alias=True) if hasattr(p,'model_dump') else p
    return dict(method=getattr(event,'method',None),payload=p)

def log(row):
    target=os.environ.get('SAFEVLA_RUN_OUTPUT')
    if target:
        with (Path(target)/'capacity_recovery.jsonl').open('a')as f:
            f.write(json.dumps(dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**row),allow_nan=False)+'\n')

def owner_snapshot(owner):
    # Owner completion receipts are actual synchronous executions. No object,
    # assist, evaluator or simulator state is consulted here.
    return dict(queue=owner._requests.qsize(),inflight=owner._capacity_inflight,
        completed_tools=owner._capacity_completed,last_tool=owner._capacity_last_tool,aborted=owner._aborted.is_set(),
        dispatch_fault=owner.dispatch_fault,execution_exception=owner._capacity_execution_exception)

def safe_snapshot(owner,pending):
    r=owner_snapshot(owner)
    if pending or r['queue'] or r['inflight'] or r['aborted'] or r['dispatch_fault'] or r['execution_exception'] or r['last_tool']=='finish':
        raise RuntimeError('Capacity continuation withheld: unfinished tool/Owner receipt is ambiguous')
    return r

def wait(budget,plan):
    start=time.monotonic()
    try:
        while True:
            remaining=max(0.,plan.delay_s-(time.monotonic()-start))
            if remaining==0:break
            if (time.monotonic()>=budget.deadline or getattr(budget,'cancelled',lambda:False)()):
                raise TimeoutError('Episode/model deadline or interruption during capacity wait')
            time.sleep(min(.25,remaining))
    finally:
        budget.record_wait(plan,max(0.,time.monotonic()-start))
    budget.start_retry(plan)

def failed_event(error,lifecycle,thread_id):
    # The first error is retained for reporting; retry classification follows
    # the actual final error, never an earlier transient notice.
    error=lifecycle.get('terminal_error_details',lifecycle.get('later_error_details',error)) or {}
    return dict(method='turn/completed',payload=dict(threadId=thread_id,turn=dict(
        id=lifecycle.get('turn_id'),status=lifecycle.get('final_turn_status'),error=dict(
            message=error.get('message'),codexErrorInfo=error.get('codexErrorInfo'),
            retry_after_s=lifecycle.get('retry_after_s')))))

@contextmanager
def owner_scope(owner):
    previous=getattr(_local,'owner',None);_local.owner=owner
    with _owner_lock:_owners[id(owner)]=owner
    try:yield
    finally:
        _local.owner=previous
        with _owner_lock:_owners.pop(id(owner),None)

@contextmanager
def planner_scope():
    """Only the natural planner uses this stream; reviewer has its own collector."""
    import openai_codex as sdk
    from rpent.planner.codex import _Recorder
    original_turn=sdk.Thread.turn;original_stream=sdk.TurnHandle.stream;original_observe=_Recorder.observe
    original_interrupt=sdk.TurnHandle.interrupt
    turns={};recovered_ids=set();active_handles={};cancelled=set()
    def interrupt(handle):
        cancelled.add(id(handle))
        return original_interrupt(active_handles.get(id(handle),handle))
    def turn(thread,input,**kwargs):
        if os.environ.get('SAFEVLA_TRANSPORT_WORKSPACE_BUDGET'):
            from research.v4.development.transport_workspace_dev_20261005.requests import record_request
            record_request()
        handle=original_turn(thread,input,**kwargs)
        turns[id(handle)]=(thread,kwargs)
        return handle
    def observe(recorder,event):
        result=original_observe(recorder,event)
        d=event_dict(event)
        if d['method']=='turn/completed':
            t=d['payload'].get('turn',{})
            if t.get('id')in recovered_ids and t.get('status')=='completed' and not t.get('error'):
                recorder.error=None
        return result
    def stream(handle):
        if id(handle)not in turns:
            yield from original_stream(handle);return
        thread,options=turns[id(handle)];owner=getattr(_local,'owner',None)
        if owner is None:
            # RPent adds a codex-sdk child thread inside the Owner callback.
            # Exactly one active owner in this per-episode process is required.
            with _owner_lock:
                owner=next(iter(_owners.values())) if len(_owners)==1 else None
        budget=CapacityBudget(deadline=min(deadline(),time.monotonic()+1500.))
        first_handle=handle;active_handles[id(first_handle)]=handle
        budget.cancelled=lambda:id(first_handle)in cancelled
        pending=set();terminal=None;notice=None
        first=handle.id
        while True:
            terminal=None;notice=None
            for event in original_stream(handle):
                d=event_dict(event);p=d.get('payload')or{}
                if d['method'] in ('item/started','item/completed'):
                    item=p.get('item',{});item=item.get('root',item)
                    if item.get('type')in ('mcpToolCall','dynamicToolCall'):
                        ident=item.get('id')
                        if d['method']=='item/started':pending.add(ident)
                        elif item.get('status')in ('completed','failed'):pending.discard(ident)
                        else:pending.add(ident)
                if d['method']=='error':notice=d
                if d['method']=='turn/completed':
                    terminal=d
                    if p['turn'].get('status')=='completed' and notice and capacity_event(notice)['is_capacity']:
                        recovered_ids.add(p['turn']['id'])
                yield event
            if terminal is None:return
            t=terminal['payload']['turn']
            if t.get('status')!='failed':
                if budget.extra_attempts and t.get('status')=='completed' and not t.get('error'):
                    budget.finish(thread_id=thread.id,turn_id=t['id'])
                    recovered_ids.add(t['id'])
                    # The completion was already observed: clear only known
                    # prior capacity error through an explicit local callback.
                    _local.capacity_recovered=True
                    log(dict(role='planner',event='recovered',original_turn_id=first,
                        **budget.summary()))
                return
            terminal['payload']['threadId']=thread.id
            plan=budget.plan(terminal)
            if plan.action!='RETRY' or budget.cancelled():return
            if owner is None:raise RuntimeError('Capacity continuation has no Owner receipt context')
            before=safe_snapshot(owner,pending)
            log(dict(role='planner',event='capacity_failed',original_turn_id=first,
                failed_turn_id=handle.id,thread_id=thread.id,delay_s=plan.delay_s,owner=before,
                error=terminal['payload']['turn'].get('error')))
            wait(budget,plan)
            after=safe_snapshot(owner,pending)
            if before!=after:raise RuntimeError('Owner changed during capacity backoff')
            if getattr(owner.base,'round_open',True)is False:
                raise RuntimeError('Capacity continuation withheld: round already finished')
            prompt=('The preceding turn ended with a temporary service capacity error. Continue only '
                'the unfinished work in this same episode and thread. Previously completed tools and '
                'their results remain in history and must not be replayed. The simulator was frozen '
                f'during recovery; {before["completed_tools"]} Owner tool receipts have completed. '
                'Use the existing observations, task, policy, tools and remaining budgets unchanged.')
            if os.environ.get('SAFEVLA_TRANSPORT_WORKSPACE_BUDGET'):
                from research.v4.development.transport_workspace_dev_20261005.requests import record_request
                record_request(retry=True)
            handle=original_turn(thread,prompt,**options)
            active_handles[id(first_handle)]=handle
            recovered_ids.add(handle.id)
            log(dict(role='planner',event='continuation_started',original_turn_id=first,
                turn_id=handle.id,owner=after,**budget.summary()))
    # Collector emits the completed event before the following loop knows it
    # recovered. The id is marked immediately before submitting that new turn.
    with patch.object(sdk.Thread,'turn',turn),patch.object(sdk.TurnHandle,'stream',stream),\
            patch.object(sdk.TurnHandle,'interrupt',interrupt),patch.object(_Recorder,'observe',observe):yield

@contextmanager
def reviewer_continuation(worker):
    """Replace only the next turn's input with a same-thread continuation."""
    cls=type(worker.thread)
    original=cls.turn
    def followup(thread,input,**options):
        if thread is worker.thread:
            input=('The preceding review turn ended with a temporary service capacity error. '
                'Continue the same unfinished review using the existing evidence and required output schema '
                'already in this thread. Return its first complete decision; do not repeat or change the '
                'robot candidate or request new tools.')
        return original(thread,input,**options)
    # Installed Thread has slots: patch its class method, scoped to this exact
    # object, instead of trying to replace an instance's read-only method.
    with patch.object(cls,'turn',followup):yield

def reviewer_call(worker,gate,packet,invoke):
    budget=CapacityBudget(deadline=float(packet.get('_capacity_deadline',deadline())))
    history=[];continuation=False
    while True:
        try:
            from contextlib import nullcontext
            accounting = nullcontext()
            if os.environ.get('SAFEVLA_TRANSPORT_WORKSPACE_BUDGET'):
                from research.v4.development.transport_workspace_dev_20261005.requests import reviewer_submission
                accounting = reviewer_submission(worker.thread,retry=continuation)
            with accounting:
                if continuation:
                    with reviewer_continuation(worker):answer=invoke(gate,packet)
                else:answer=invoke(gate,packet)
            if budget.extra_attempts:
                budget.finish(thread_id=worker.thread.id,turn_id=worker.last_turn_id)
                worker.last_error_lifecycle=dict(worker.last_error_lifecycle or{},capacity_recovery=dict(
                    history=history,**budget.summary()))
            return answer
        except RuntimeError:
            error=worker.last_error_details or{};life=worker.last_error_lifecycle or{}
            e=failed_event(error,life,worker.thread.id);plan=budget.plan(e)
            history.append(dict(error=error,lifecycle=life,action=plan.action,delay_s=plan.delay_s))
            if plan.action!='RETRY':
                if budget.extra_attempts:
                    worker.last_error_lifecycle=dict(life,capacity_recovery=dict(history=history,**budget.summary()))
                raise
            notify=getattr(worker,'capacity_notify',None)
            if notify:notify(dict(kind='capacity_wait',delay_s=plan.delay_s,thread_id=worker.thread.id,
                prior_turn_id=worker.last_turn_id,extra_attempt=plan.extra_attempt))
            log(dict(role='reviewer',event='capacity_failed',gate=gate,thread_id=worker.thread.id,
                error=error,lifecycle=life,delay_s=plan.delay_s))
            wait(budget,plan);continuation=True
