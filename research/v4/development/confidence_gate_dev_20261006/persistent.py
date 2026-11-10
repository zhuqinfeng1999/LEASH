"""Local confidence hooks for the existing persistent RPent round driver.

No new scheduler or session retry loop is introduced. The original driver
continues to decide finish/quota sealing and whether another instruction may
start. Physical round transitions/recovery remain an explicit owner callback
from the existing task runner; this module does not invent a recovery route.
"""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import ast
import inspect
import json
from pathlib import Path
import secrets
import textwrap
import threading

from research.v4.candidates.multiround_bridge_20260924 import driver
from .runner import ARMS, SYSTEM_PROMPT, bind_policy

INTERNAL = '__confidence_round_callback__'


class OwnerRoundCallbacks:
    """Same Owner queue, with an unadvertised runner-only callback."""
    def __init__(self, base, callback):
        self.base, self.callback = base, callback
        self.owner_thread_id = threading.get_ident()
        self.token = secrets.token_hex(24)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_tools_spec(self):
        result = self.base.get_tools_spec()
        if any(row['name'] == INTERNAL for row in result):
            raise ValueError('Internal round callback must not be advertised')
        return result

    def execute_tool(self, name, arguments):
        if name != INTERNAL:
            coordinator = getattr(self.callback, '__self__', None)
            intercept = getattr(coordinator, 'intercept_tool', None)
            if intercept is not None:
                result = intercept(name, arguments)
                if result is not None:
                    return result
            return self.base.execute_tool(name, arguments)
        if (arguments.get('token') != self.token
                or threading.get_ident() != self.owner_thread_id):
            raise PermissionError('Round callback requires the actual owner and private runner token')
        if set(arguments) != {'token','operation','index'}:
            raise PermissionError('Unexpected internal round callback argument')
        return SimpleNamespace(result=self.callback(arguments['operation'], arguments['index']))

    def invoke(self, owner, operation, index):
        return owner.execute_tool(INTERNAL, dict(token=self.token,operation=operation,index=index)).result


class RoundEvidenceProjection:
    def __init__(self, original, coordinator):
        self.original, self.coordinator = original, coordinator

    def __getattr__(self, name):
        return getattr(self.original, name)

    def capture(self, *args, **kwargs):
        result = self.original.capture(*args, **kwargs)
        gate = kwargs.get('gate', args[3] if len(args)>3 else None)
        if gate == 'G1':
            # Actual prior instruction, final text, boundary and initial RGBs;
            # no reviewer internals or guessed physical success are included.
            result['round_history'] = deepcopy(self.coordinator.history)
            self.coordinator.start_keyframes = deepcopy(result['current_images'])
        return result


class RoundCoordinator:
    def __init__(self, stack, instructions, *, prepare_physical_round,
                 after_round=None, prompt_binding=None):
        if not callable(prepare_physical_round):
            raise ValueError('Explicit existing task transition/recovery callback required')
        self.stack, self.instructions = stack, tuple(instructions)
        self.prepare_physical_round = prepare_physical_round
        self.after_round = after_round
        self.prompt_binding = prompt_binding
        self.history = []
        self.prepared = 0
        self.task_row = None
        self.start_keyframes = []
        self.previous = None
        self.callbacks = None

    def _admit_existing_round(self, index, instruction):
        """The original driver already called B0Toolkit.begin_round once."""
        if self.stack.arm == 'B0':
            self.stack.adapter.instruction = instruction
            return None
        boundary, adapter, core = self.stack.boundary, self.stack.adapter, self.stack.boundary.core
        if index != boundary.round_id + 1:
            raise ValueError('Confidence round identity changed')
        # Quota seals are administrative, never fabricated finish/refusal. The
        # original driver's _begin_next_round has validated this transition.
        boundary.round_open = True
        boundary.round_id = adapter.round_id = index
        adapter.instruction = instruction
        boundary.denials = 0
        boundary.counted_denials.clear()
        boundary.last_local_denial = None
        core.begin_round()
        candidate = dict(instruction=instruction)
        try:
            evidence = adapter.evidence('G1',candidate)
            row = core.score('G1',evidence,candidate=candidate,epoch=adapter.epoch())
        except Exception as exc:
            boundary.system_failure=f'G1 public evidence unavailable: {type(exc).__name__}: {exc}'
            return dict(gate='G1',allowed=False,status='system_failure',system_failure=True,
                        decision='REFUSE',reason=boundary.system_failure)
        boundary.logs.append(row)
        if row['system_failure']:
            boundary.system_failure=core.feedback(row)
        return row

    def _refresh_public_observation(self):
        """Existing saved RPent observation refresh, with zero control sends."""
        stack=self.stack
        original=stack.base.original
        observation=stack.adapter._policy_obs()
        original._primitives.set_obs(observation)
        recorded=original.get_env_state(command=dict(action='confidence_round_observation_refresh'),
            result=dict(kind='public_observation_only_no_controller_action'),elapsed_s=0.)
        original._publish_step(original._state.latest_record())
        if recorded.get('step') != original._state.latest_step:
            raise RuntimeError('Round public observation did not match saved RPent state')
        # This read is runner setup, before the model receives an instruction.
        # A REFUSE still prevents every non-finish planner tool invocation.
        return driver.capture_public_observation(stack.base)

    def _host_final_finish_state(self, index):
        """Pure outer authority. No judge/physics truth, mutation, or recovery."""
        if (type(index) is not int or index != self.prepared
                or index != len(self.history) + 1
                or index != len(self.instructions)):
            return dict(eligible=False, reason='Not the final unsealed registered round')
        stack = self.stack
        terminal = stack.base.finish_result
        if (not isinstance(terminal, dict) or terminal.get('status') not in ('done', 'refused')
                or not isinstance(terminal.get('message'), str)
                or stack.base.round_finished is not True):
            return dict(eligible=False, reason='No actual completed final finish(done/refused)')
        if (getattr(stack.boundary, 'system_failure', None)
                or getattr(stack.owner, 'dispatch_fault', None)
                or (stack.arm != 'B0' and stack.boundary.round_open)):
            return dict(eligible=False, reason='Existing fault or unfinished safety boundary')
        # A real terminal may close at the limit, never beyond/without it.
        if any(type(getattr(stack.base, name, None)) is not int
                for name in ('round_calls', 'episode_calls', 'per_round_budget', 'episode_budget')):
            return dict(eligible=False, reason='Primitive budget state unavailable')
        if (not 0 <= stack.base.round_calls <= stack.base.per_round_budget
                or not 0 <= stack.base.episode_calls <= stack.base.episode_budget):
            return dict(eligible=False, reason='Primitive budget state outside original limits')
        # This is exactly the same pure predicate the existing recovery uses.
        query = getattr(self, '_local_denial_recovery_query', None)
        if query is not None and query(index)['continued']:
            return dict(eligible=False, reason='Existing local-denial continuation still applies')
        calls = stack.base.calls
        if (not calls or calls[-1].get('name') != 'finish'
                or calls[-1].get('arguments') != terminal
                or calls[-1].get('result') != terminal):
            return dict(eligible=False, reason='Final finish does not match original Owner record')
        return dict(eligible=True, authority='original_round_coordinator_final_finish',
            index=index, total_instructions=len(self.instructions),
            sealed_rounds=len(self.history), terminal=deepcopy(terminal),
            original_finish_call_index=len(calls)-1,
            round_calls=stack.base.round_calls, episode_calls=stack.base.episode_calls,
            no_registered_instruction_or_permission_or_local_denial_continuation=True)

    def _host_recovery_turn_finish_state(self, index):
        """Pure current-turn authority; the original session still owns re-entry."""
        if (type(index) is not int or index!=self.prepared
                or index!=len(self.history)+1 or not 1<=index<=len(self.instructions)):
            return dict(eligible=False,reason='No current unsealed registered round')
        stack=self.stack;terminal=stack.base.finish_result
        if (not isinstance(terminal,dict) or terminal.get('status') not in ('done','refused')
                or not isinstance(terminal.get('message'),str) or stack.base.round_finished is not True):
            return dict(eligible=False,reason='No actual finished current-turn terminal')
        if (getattr(stack.boundary,'system_failure',None) or getattr(stack.owner,'dispatch_fault',None)
                or stack.arm=='B0' or stack.boundary.round_open):
            return dict(eligible=False,reason='Fault, non-LEASH or unfinished boundary')
        if any(type(getattr(stack.base,name,None)) is not int
                for name in ('round_calls','episode_calls','per_round_budget','episode_budget')):
            return dict(eligible=False,reason='Original primitive budgets unavailable')
        if (not 0<=stack.base.round_calls<stack.base.per_round_budget
                or not 0<=stack.base.episode_calls<stack.base.episode_budget):
            return dict(eligible=False,reason='No original primitive budget remains')
        query=getattr(self,'_local_denial_recovery_query',None)
        recovery=None if query is None else query(index)
        if not isinstance(recovery,dict) or recovery.get('continued') is not True:
            return dict(eligible=False,reason='Original local recovery is not currently applicable')
        calls=stack.base.calls
        if (not calls or calls[-1].get('name')!='finish' or calls[-1].get('arguments')!=terminal
                or calls[-1].get('result')!=terminal):
            return dict(eligible=False,reason='Terminal mismatches original Owner finish')
        row=recovery['row']
        return dict(eligible=True,authority='original_round_coordinator_finished_turn_local_recovery',
            index=index,total_instructions=len(self.instructions),sealed_rounds=len(self.history),
            terminal=deepcopy(terminal),original_finish_call_index=len(calls)-1,
            round_calls=stack.base.round_calls,episode_calls=stack.base.episode_calls,
            gate=row['gate'],blocked_candidate_hash=row['candidate_hash'],
            executed_sequence=stack.scene.sequence,
            same_registered_round_local_recovery_currently_applicable=True,
            original_session_still_owns_turn_time_budget_and_recovery=True,
            episode_final_finish_authorized=False)

    def _host_registered_round_finish_state(self, index):
        """Only a completed ordinary round; original driver owns the next input."""
        if (type(index) is not int or index != self.prepared
                or index != len(self.history)+1 or not 1 <= index < len(self.instructions)):
            return dict(eligible=False, reason='No nonfinal unsealed registered round')
        stack=self.stack; terminal=stack.base.finish_result
        if (not isinstance(terminal,dict) or terminal.get('status')!='done'
                or not isinstance(terminal.get('message'),str) or stack.base.round_finished is not True):
            return dict(eligible=False,reason='No actual ordinary finish(done)')
        if (getattr(stack.boundary,'system_failure',None) or getattr(stack.owner,'dispatch_fault',None)
                or (stack.arm!='B0' and stack.boundary.round_open)):
            return dict(eligible=False,reason='Fault or unfinished original safety boundary')
        if any(type(getattr(stack.base,name,None)) is not int
                for name in ('round_calls','episode_calls','per_round_budget','episode_budget')):
            return dict(eligible=False,reason='Original primitive budgets unavailable')
        if (not 0 <= stack.base.round_calls <= stack.base.per_round_budget
                or not 0 <= stack.base.episode_calls < stack.base.episode_budget):
            return dict(eligible=False,reason='Budget exceeded or no original episode budget remains')
        query=getattr(self,'_local_denial_recovery_query',None)
        if query is not None and query(index)['continued']:
            return dict(eligible=False,reason='Original local-denial continuation has priority')
        calls=stack.base.calls
        if (not calls or calls[-1].get('name')!='finish' or calls[-1].get('arguments')!=terminal
                or calls[-1].get('result')!=terminal):
            return dict(eligible=False,reason='Terminal mismatches original Owner finish')
        return dict(eligible=True,authority='original_round_coordinator_finished_registered_round',
            index=index,total_instructions=len(self.instructions),sealed_rounds=len(self.history),
            terminal=deepcopy(terminal),original_finish_call_index=len(calls)-1,
            round_calls=stack.base.round_calls,episode_calls=stack.base.episode_calls,
            next_registered_index=index+1,next_registered_instruction=self.instructions[index],
            original_registered_instructions=list(self.instructions),
            current_round_completed_next_registered_instruction_pending=True,
            original_driver_still_owns_round_seal_and_next_instruction=True,
            episode_final_finish_authorized=False)

    def _host_primitive_budget_state(self, index):
        """Pure original quota authority; no synthetic finish or round mutation."""
        from research.v3.bridge.b0 import BUDGETED_ACTIONS
        if (type(index) is not int or index!=self.prepared
                or index!=len(self.history)+1 or not 1<=index<=len(self.instructions)):
            return dict(eligible=False,reason='No current unsealed registered round')
        stack=self.stack;base=stack.base
        if (base.finish_result is not None or base.round_finished is not False
                or getattr(stack.boundary,'system_failure',None)
                or getattr(stack.owner,'dispatch_fault',None)):
            return dict(eligible=False,reason='Finished round or original system/dispatch fault')
        names=('round_calls','episode_calls','per_round_budget','episode_budget')
        if any(type(getattr(base,n,None))is not int for n in names):
            return dict(eligible=False,reason='Original primitive counters unavailable')
        if (not 0<base.round_calls<=base.per_round_budget
                or not 0<base.episode_calls<=base.episode_budget):
            return dict(eligible=False,reason='Original primitive limits exceeded or not positive')
        boundary=('episode_call_quota' if base.episode_calls==base.episode_budget else
            'round_call_quota' if base.round_calls==base.per_round_budget else None)
        start=getattr(self,'_budget_round_start_episode_calls',None)
        calls=getattr(base,'calls',None)
        if (boundary is None or type(start)is not int or not isinstance(calls,list)
                or base.episode_calls-start!=base.round_calls):
            return dict(eligible=False,reason='Original quota or prepared-round prefix not established')
        counted=[]
        for call in calls:
            if not isinstance(call,dict):return dict(eligible=False,reason='Original call record malformed')
            if call.get('name')in BUDGETED_ACTIONS:
                if not isinstance(call.get('arguments'),dict) or 'result'not in call:
                    return dict(eligible=False,reason='Original counted call has no returned result')
                counted.append(dict(name=call['name'],arguments=deepcopy(call['arguments'])))
        if len(counted)!=base.episode_calls:
            return dict(eligible=False,reason='Original returned primitive records mismatch original counters')
        return dict(eligible=True,authority='original_round_coordinator_primitive_budget',
            index=index,total_instructions=len(self.instructions),sealed_rounds=len(self.history),
            resource_boundary=boundary,terminal=None,actual_finish_called=False,
            round_calls=base.round_calls,episode_calls=base.episode_calls,
            per_round_budget=base.per_round_budget,episode_budget=base.episode_budget,
            round_start_episode_calls=start,budgeted_owner_calls=counted,
            original_registered_instructions=list(self.instructions),
            next_registered_instruction=self.instructions[index]if index<len(self.instructions)else None,
            original_driver_still_owns_quota_seal_and_next_instruction=True,
            episode_final_finish_authorized=False)

    def callback(self, operation, index):
        if operation == 'host_primitive_budget_state':
            return self._host_primitive_budget_state(index)
        if operation == 'host_registered_round_finish_state':
            return self._host_registered_round_finish_state(index)
        if operation == 'host_recovery_turn_finish_state':
            return self._host_recovery_turn_finish_state(index)
        if operation == 'host_final_finish_state':
            return self._host_final_finish_state(index)
        if operation == 'prepare':
            if index != self.prepared+1 or index > len(self.instructions):
                raise ValueError('Round preparation order changed')
            instruction=self.instructions[index-1]
            self.prepare_physical_round(index,instruction,self.previous)
            self.start_keyframes=[]
            self.task_row=self._admit_existing_round(index,instruction)
            observation=self._refresh_public_observation()
            self.prepared=index
            self._budget_round_start_episode_calls=self.stack.base.episode_calls
            return observation
        if operation == 'seal':
            if index != self.prepared or index != len(self.history)+1:
                raise ValueError('Round seal order changed')
            terminal=deepcopy(self.stack.base.finish_result)
            row=dict(index=index,instruction=self.instructions[index-1],
                final_message=None if terminal is None else terminal.get('message'),
                terminal=terminal,boundary=self.previous,
                start_keyframes=deepcopy(self.start_keyframes))
            self.history.append(row)
            # Public administrative evidence only; no extra instruction or action.
            with (Path(self.stack.base.output_dir).parent/'round_coordinator_seals.jsonl').open('a') as saved:
                saved.write(json.dumps(dict(index=index,instruction=self.instructions[index-1],
                    terminal=terminal,boundary=self.previous,sealed_rounds=len(self.history),
                    total_instructions=len(self.instructions),
                    next_registered_instruction=self.instructions[index] if index<len(self.instructions) else None),
                    ensure_ascii=False)+'\n')
            if self.after_round is not None:
                self.after_round(index,terminal,self.previous)
            return row
        raise ValueError('Unknown internal round operation')

    def observe(self, toolkit):
        if toolkit is not self.stack.owner:
            raise RuntimeError('Persistent driver changed Owner identity')
        return self.callbacks.invoke(toolkit,'prepare',self.prepared+1)


class _RoundText:
    def __init__(self, original, coordinator):
        self.original,self.coordinator=original,coordinator

    def __getattr__(self,name):
        return getattr(self.original,name)

    def text(self):
        text=self.original.text()
        binding=self.coordinator.prompt_binding
        if self.original.first_round:
            prefix=driver.B0_SYSTEM_PROMPT+'\n\n'
            if not text.startswith(prefix):
                raise ValueError('Original first-round system prompt changed')
            common = SYSTEM_PROMPT if binding is None else binding.apply(SYSTEM_PROMPT)
            text=common+'\n\n'+text[len(prefix):]
        row=self.coordinator.task_row
        if row is not None and not row['allowed']:
            core=self.coordinator.stack.boundary.core
            text+='\n'+(core.feedback(row) if 'p_safe' in row else row['reason'])
        return text


def _filtered_round_function(function):
    """Keep the original Async collector and options; add bounded capacity continuation."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    rewritten=0
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or node.func.attr!='turn':
            continue
        for keyword in node.keywords:
            if keyword.arg is None and ast.unparse(keyword.value)=='self._planner._turn_options':
                keyword.value=ast.Call(func=ast.Name(id='_confidence_async_turn_options',ctx=ast.Load()),
                    args=[ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='_planner',ctx=ast.Load())],keywords=[])
                rewritten+=1
    if rewritten!=1:
        raise ValueError('Expected one original persistent Async Thread.turn call')
    drive_calls=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_drive_sdk_turn':
            node.func.id='_confidence_capacity_drive'
            node.args.insert(0,ast.Name(id='self',ctx=ast.Load()))
            drive_calls+=1
    if drive_calls!=1:
        raise ValueError('Expected one original persistent Async drive call')
    from research.v4.development.g2_candidate_evidence_dev_20261005.async_capacity_recovery import drive_with_capacity
    namespace=dict(function.__globals__)
    namespace['_confidence_capacity_drive']=drive_with_capacity
    namespace['_confidence_async_turn_options']=lambda planner:{k:v for k,v in planner._turn_options.items()
                                                               if k!='base_instructions'}
    exec(compile(ast.fix_missing_locations(tree),function.__code__.co_filename,'exec'),namespace)
    return namespace[function.__name__]


class ConfidencePersistentSession(driver.CodexPersistentSession):
    """Original persistent run_round stream/continuation and one thread."""
    _original_run_round=_filtered_round_function(driver.CodexPersistentSession.run_round)

    def __init__(self,*,coordinator,arm,toolkit,output_dir,model_id,reasoning_effort='high'):
        if arm not in ARMS:raise ValueError('Only B0/Ours-conf are enabled')
        super().__init__(toolkit=toolkit,output_dir=output_dir,model_id=model_id,reasoning_effort=reasoning_effort)
        self.coordinator,self.arm=coordinator,arm

    async def __aenter__(self):
        from research.v4.integrations.planner_routing import MODELS, open_registered_session
        if self.model_id in MODELS:
            return await open_registered_session(self)
        import openai_codex
        from research.v3.bridge.b0 import build_b0_planner
        from rpent.planner.utils.http_mcp_server import HttpMcpServer
        self._planner=build_b0_planner(output_dir=self.output_dir,toolkit=self.toolkit,
            timeout_s=driver.PLANNER_TIMEOUT_S,model_id=self.model_id,reasoning_effort=self.reasoning_effort)
        bind_policy(self._planner,self.arm)
        self._server=HttpMcpServer(self.toolkit)
        url=self._server.start()
        try:
            self._codex=openai_codex.AsyncCodex(self._planner._build_config(url))
            self._thread=await self._codex.thread_start(ephemeral=True,**self._planner._turn_options)
            self.thread_starts=1
        except BaseException:
            self._server.stop()
            raise
        return self

    async def run_round(self,turn_input,toolkit,boundary,*,max_turns,timeout_s):
        outcome=await self._original_run_round(_RoundText(turn_input,self.coordinator),toolkit,boundary,
            max_turns=max_turns,timeout_s=timeout_s)
        seal=outcome.resource_end or boundary()
        if seal is None and outcome.turns_used >= max_turns:
            seal='planner_turn_quota'
        if not outcome.error and outcome.sdk_completed and seal is not None:
            self.coordinator.previous=seal
            self.coordinator.callbacks.invoke(toolkit,'seal',turn_input.index)
        return outcome


@contextmanager
def round_callback_scope(stack,coordinator):
    original=stack.owner.base
    callbacks=OwnerRoundCallbacks(original,coordinator.callback)
    old_builder=getattr(stack.adapter,'evidence_builder',None)
    if old_builder is not None:
        stack.adapter.evidence_builder=RoundEvidenceProjection(old_builder,coordinator)
    stack.owner.base=callbacks
    coordinator.callbacks=callbacks
    try:
        yield coordinator
    finally:
        stack.owner.base=original
        if old_builder is not None:stack.adapter.evidence_builder=old_builder


def make_persistent_callback(stack,configured,*,output_dir,model_id,prepare_physical_round,
                             reasoning_effort='high',after_round=None,prompt_binding=None,
                             episode_started_at=None,session_factory=ConfidencePersistentSession,
                             coordinator_factory=RoundCoordinator):
    """Return a context owning callbacks and an async body for the Owner pump."""
    if not configured.persistent:
        raise ValueError('This entry is only for registered multi-round tasks')
    coordinator=coordinator_factory(stack,configured.instructions,
        prepare_physical_round=prepare_physical_round,after_round=after_round,prompt_binding=prompt_binding)
    async def body():
        session=session_factory(coordinator=coordinator,arm=stack.arm,toolkit=stack.owner,
            output_dir=str(output_dir),model_id=model_id,reasoning_effort=reasoning_effort)
        async with session:
            rounds=await driver.run_multiround_b0(instructions=configured.instructions,
                toolkit=stack.owner,session=session,observe=coordinator.observe,
                episode_started_at=episode_started_at)
        return dict(rounds=rounds,history=deepcopy(coordinator.history),
                    thread_starts=session.thread_starts,usage=dict(session.usage))
    return round_callback_scope(stack,coordinator),body
