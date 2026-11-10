"""One registered natural development episode through the shared RPent runner.

Private evaluation and human recording are output sinks only. They never
return information to the planner, gates, registry or command dispatch.
"""
import argparse
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import fcntl
import os
from pathlib import Path
import signal
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = 'research/v4/development/confidence_gate_dev_20261006/'


def configure_p2_outcome(configured, version):
    """Explicit private outcome version; never silently ignore a new version."""
    if version == 'p2-uncued-right-hand-table-dev9-20261014':
        from .p2_uncued_instruction_20261014 import adapt_configuration
    elif version == 'p2-front-placement-dev8-bounded-handle-touch-20261008':
        from .p2_front_placement_v8 import adapt_configuration
    elif version == 'p2-front-placement-dev7-approximate-region-20261008':
        from .p2_front_placement_v7 import adapt_configuration
    elif version == 'p2-front-placement-dev6-robot-facing-frame-20261008':
        from .p2_front_placement_v6 import adapt_configuration
    elif version in (None, 'p2-front-placement-dev5-contact-aware-20261008'):
        from .p2_front_placement_v5 import adapt_configuration
    else:
        raise ValueError('Unsupported registered P2 outcome version')
    return adapt_configuration(configured)


class DevelopmentBudgetExceeded(BaseException):
    """End the whole run; per-request technical retries must not catch it."""
    def __init__(self, message, *, resource=None):
        super().__init__(message)
        self.resource = resource
        self.episode_resource_stop = resource in ('wall_time', 'output_bytes')


def record_execution_exception(outcome, exc):
    """Separate a legal episode wall boundary from an implementation failure.

    This never invents an RPent finish. The existing private evaluator still
    evaluates the actual trace, including earlier harm or unsafe execution.
    """
    if isinstance(exc, DevelopmentBudgetExceeded) and exc.resource in ('wall_time', 'output_bytes'):
        budget = dict(outcome.get('budget_exhausted') or {})
        budget.update(resource=exc.resource, reason=str(exc))
        if exc.resource == 'output_bytes':
            budget['limit_bytes'] = 1024**3
        else:
            budget['limit_s'] = outcome.get('wall_limit_s', 2700)
        outcome.update(status='completed', error=None, budget_exhausted=budget,
                       termination_kind='episode_budget_exhausted')
    else:
        outcome.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                       traceback=traceback.format_exc())


def expire_episode_wall(outcome):
    """Keep the primary boundary even if a tool-server task catches it."""
    limit=outcome.get('wall_limit_s',2700)
    reason = f'{limit}-second development episode wall limit'
    outcome.update(budget_exhausted=dict(resource='wall_time', limit_s=limit,
                                        reason=reason),
                   termination_kind='episode_budget_exhausted')
    raise DevelopmentBudgetExceeded(reason, resource='wall_time')


def record_planner_budget_metadata(outcome, planner_result, *, limit_s=2700):
    """Recognize the SDK's configured outer timeout without inventing finish.

    A tool-server alarm can be consumed by its HTTP task while the SDK returns
    its normal timeout result, rather than reaching the outer exception path.
    Generic service/reviewer errors are not reclassified as legal budgets.
    """
    stats = planner_result.stats or {}
    elapsed = stats.get('elapsed_s')
    if (not outcome.get('budget_exhausted')
            and planner_result.error == f'Codex SDK timed out after {limit_s}s'
            and stats.get('backend') == 'codex_sdk'
            and type(elapsed) in (int, float) and elapsed >= limit_s):
        outcome.update(budget_exhausted=dict(
            resource='planner_wall_time', limit_s=limit_s,
            reason=planner_result.error, elapsed_s=elapsed),
            termination_kind='planner_budget_exhausted')


SOURCES = tuple(PACKAGE+n for n in ('native.py','runner.py','runtime.py','common_control.py',
    'core.py','backend.py','evidence.py','primitive_call_log.py','configurations.py',
    'family_cli_seam.py','persistent.py','s_round_hooks.py','s1_recovery.py','r15_configuration.py')) + (
    'research/v4/leash_v43/registry.py','research/v4/leash_v43/dispatch.py',
    'research/v4/leash_v43/adapter.py','research/v4/leash_v43/run_manifest.py',
    'research/v4/leash_v43/initial_occupancy.py','research/v4/bridge/thread_owner.py',
    'research/v4/bridge/observation.py','research/v4/bridge/formal_recording.py',
    'research/v3/bridge/b0.py','research/v3/bridge/libero_env.py',
    'research/v4/accelerated/r2_adapter.py',
    'research/v4/candidates/leash_reviewer_transport_20260924/transport.py',
    'research/v4/development/confidence_gates_20261006/config.py',
    'research/v4/development/confidence_gates_20261006/policy_v43_original.txt',
    'research/v4/development/user_review_revision_20261026/policy_revision.py',
    'research/v4/development/user_review_revision_20261026/policy_open_liquid_20261026.txt',
    'research/v4/development/confidence_gates_20261006/prompt_confidence_G1_v1.txt',
    'research/v4/development/confidence_gates_20261006/prompt_confidence_G2_v3.txt',
    'research/v4/development/confidence_gates_20261006/prompt_confidence_G3_v3.txt',
    'research/v4/development/confidence_gates_20261006/policy_prompts_primitive_v4.json',
    'research/v4/development/p102_contact_repair_20261005/binding.py',
    'research/v4/development/p1_single_right_arm_20261004/layout.py',
    'research/v4/development/common_translation_dev_20261005/common_motion.py',
    'research/v4/development/transport_workspace_dev_20261005/world_reference_controller.py',
    'research/v4/development/transport_workspace_dev_20261005/controller_intent.py',
    'research/v4/familiarization/shared_prompt_draft/binding.py',
    'upstream/RPent-reproduce-libero/rpent/planner/codex.py',
    'upstream/RPent-reproduce-libero/robots/libero/tools.py',
    'research/v4/results/confidence_gate_dev_20261006/initialization_qualification.json')
SOURCES += ('research/v4/development/decision_gate_dev_20261019/backend.py',
            PACKAGE+'p3_physical_outcome_20261019.py')


def utc():
    return datetime.now(timezone.utc).isoformat()


def append_json(path, row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def output_size(output):
    files = [p for p in Path(output).rglob('*') if p.is_file()]
    return sum(p.stat().st_size for p in files), len(files)


def serializable_run(run):
    """Keep the existing business result fields, including RPent's slot class."""
    from .evidence import plain
    planner = run.planner_result
    return plain(dict(planner_result=dict(finish_result=planner.finish_result,
        messages=planner.messages, stats=planner.stats, error=planner.error),
        terminal=run.terminal, calls=run.calls, model_id=run.model_id,
        reasoning_effort=run.reasoning_effort))


def serializable_persistent(result):
    """Write the original driver's dataclass round results as ordinary JSON."""
    from .evidence import plain
    return plain({**result, 'rounds': [asdict(row) for row in result['rounds']]})


def capture_final_private_step(evaluator, scene, private_trace, *, wall_elapsed_s):
    """Capture one real unobserved final step before Scene.close, offline only.

    This never steps/resets the scene or refreshes a public/model observation.
    Multiple absent steps cannot be reconstructed from one terminal snapshot.
    """
    from .evidence import plain
    if evaluator is None or scene is None:
        return dict(status='not_applicable', model_observations=0, control_sends=0)
    current = getattr(scene, 'sequence', None)
    if type(current) is not int or current < 0:
        raise RuntimeError('Final private capture lacks a valid actual scene sequence')
    last_line = None
    path = Path(private_trace)
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    last_line = line
    if last_line is None:
        raise RuntimeError('Final private capture cannot reconstruct missing initial history')
    previous_row = json.loads(last_line)
    previous = previous_row.get('sequence')
    if type(previous) is not int or previous < 0:
        raise RuntimeError('Final private capture lacks a valid previous sequence')
    gap = current - previous
    if gap == 0:
        return dict(status='already_observed', previous_sequence=previous,
                    sequence=current, model_observations=0, control_sends=0)
    if gap != 1:
        raise RuntimeError(f'Final private capture cannot reconstruct {gap} absent executed steps')
    row = evaluator.observe(scene)
    append_json(path, plain(dict(sequence=current,
        observed_wall_elapsed_s=wall_elapsed_s,
        private_capture_source='close_hook_unobserved_final_step', **row)))
    return dict(status='captured_one_actual_final_step', previous_sequence=previous,
                sequence=current, model_observations=0, control_sends=0)


def record_private_evaluation(outcome, evaluator, finish, output):
    """Keep a finalization failure separate from the original execution error."""
    from research.v4.leash_v43.run_manifest import atomic_json
    from .evidence import plain
    if evaluator is None:
        return None
    try:
        physical = finish()
        atomic_json(Path(output) / 'final-result.json', plain(physical))
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        outcome.update(status='failed', evaluation_error=failure,
                       evaluation_traceback=traceback.format_exc())
        if not outcome.get('error'):
            outcome['error'] = failure
        return None
    outcome.update(label=physical['label'], physical=plain(physical))
    return physical


def configure_formal_private_outcome(configured):
    """Exact formal private-scoring seam; contains no reset or model call."""
    from .private_outcome_audit_repair8_20261010 import configure as previous
    from .r_critical_holding_union_repair9 import configure as physical_r
    from .e1_receiving_scope_20261012 import configure as receiving_e1
    from .s3_actual_snack_grasp_20261012 import configure as physical_snack
    from .i_named_danger_holding_repair9 import configure as physical_i
    result=physical_i(physical_snack(receiving_e1(physical_r(previous(configured)))))
    for field in ('private_r_critical_union_revision','private_snack_grasp_revision','private_i_danger_holding_revision'):
        if getattr(result,field,None):result.private_outcome_revision=getattr(result,field)
    from .s_safe_response_dev10_20261014 import configure as safe_s_response
    result=safe_s_response(result)
    if configured.binding.instance.startswith(('I','S1-','S2-')):
        result.private_outcome_revision='user-seed0-task-outcome-dev10-20261014'
    from .e3_material_retention_20261014 import configure as e3_retention
    result=e3_retention(result)
    if (configured.binding.instance.startswith('P2-')
            and getattr(configured,'placement_task_version',None)=='p2-uncued-right-hand-table-dev9-20261014'):
        from .p2_reception_outcome_dev12_20261016 import adapt_configuration
        result=adapt_configuration(result)
    return result


@contextmanager
def robot_job(*, formal=False, gpu=1, available_slots=(1,), run_id=None):
    from research.v4.development.transport_workspace_dev_20261005.progress import HERE, PATH, available, update
    from research.v4.bridge.formal_parallel_runtime import robot_resource
    if not formal and not available(json.loads(PATH.read_text())):
        raise RuntimeError('Development is paused')
    with robot_resource(here=HERE,read_state=lambda:json.loads(PATH.read_text()),
            update_state=update,formal=formal,gpu=gpu,available_slots=available_slots,run_id=run_id):
        yield


def model_counts(output, result, rpc_counts, records):
    stats = result['run'].planner_result.stats if result is not None else {}
    started_turns = set()
    planner_usage = {}
    streams = list((Path(output)/'rpent').rglob('*.stream.jsonl'))
    # The existing persistent driver writes its one stream at episode root.
    persistent_stream = Path(output)/'planner.stream.jsonl'
    if persistent_stream.exists():
        streams.append(persistent_stream)
    for stream in streams:
        with stream.open() as saved:
            for line in saved:
                try: event = json.loads(line)
                except (ValueError, TypeError): continue
                if event.get('method') == 'turn/started':
                    payload = event.get('payload', event.get('params', {}))
                    turn = payload.get('turn', {})
                    if turn.get('id'): started_turns.add(turn['id'])
                    if payload.get('thread_id'):
                        planner_usage.setdefault(payload['thread_id'], None)
                elif event.get('method') == 'thread/tokenUsage/updated':
                    payload = event.get('payload', event.get('params', {}))
                    if payload.get('thread_id') and payload.get('token_usage') is not None:
                        planner_usage[payload['thread_id']] = payload['token_usage']
    reviewer_attempts = [a for r in records for a in r.get('attempts', [])]
    reviewer_turns = set()
    reviewer_usage = {}
    reviewer_log = Path(output)/'reviewer_scores.jsonl'
    saved_rows = []
    if reviewer_log.exists():
        with reviewer_log.open() as saved:
            for line in saved:
                try: row = json.loads(line)
                except (ValueError, TypeError): continue
                saved_rows.append(row)
                if row.get('event') == 'score_turn_started' and row.get('thread_id') and row.get('turn_id'):
                    reviewer_turns.add((row['thread_id'], row['turn_id']))
                    reviewer_usage.setdefault(row['thread_id'], None)
    http_started = {r['request_local_id'] for r in saved_rows
                    if r.get('event')=='score_http_submitted' and r.get('request_local_id')}
    if http_started:
        completed = {r['request_local_id']:r for r in saved_rows
                     if r.get('event')=='score' and r.get('request_local_id') in http_started}
        reviewer_calls = len(http_started)
        token_usage = [completed.get(rid,{}).get('tokens') for rid in sorted(http_started)]
        call_scope = 'distinct attempted stateless HTTP requests; each response usage once; missing response usage unavailable'
    elif reviewer_turns or os.environ.get('SAFEVLA_EXECUTION_PROFILE')=='confidence_formal_20261008':
        turn_threads = {turn: thread for thread, turn in reviewer_turns}
        for row in saved_rows:
            thread = row.get('thread_id') or turn_threads.get(row.get('turn_id'))
            if thread not in reviewer_usage: continue
            usage = row.get('tokens', row.get('usage'))
            if usage is not None: reviewer_usage[thread] = usage
        reviewer_calls = len(reviewer_turns)
        token_usage = list(reviewer_usage.values())
        call_scope = 'distinct submitted SDK thread/turn IDs; includes unfinished and capacity-recovery turns'
    else:
        # Older saved rows have no submission marker. Preserve their known
        # completed attempts without claiming unreturned calls were observed.
        reviewer_calls = len(reviewer_attempts)
        token_usage = [a.get('tokens') for a in reviewer_attempts]
        call_scope = 'saved scoring attempts only; unreturned SDK turns unavailable'
    # SDK usage is cumulative per thread. Capacity recovery on the same
    # thread therefore contributes multiple calls but only its last total.
    # Planner totals are likewise final cumulative session stats, counted once.
    api_fees=[r.get('estimated_cost_usd') for r in saved_rows
              if r.get('event')=='score' and r.get('request_local_id') in http_started]
    api_fee=(sum(api_fees) if http_started and len(api_fees)==len(http_started)
             and all(type(f)in(int,float)for f in api_fees) else 'unavailable')
    return dict(planner_GPT=dict(calls=len(started_turns) or None,
                    call_scope='actual SDK turn/started IDs; not provider-internal inference count',
                    provider_requests='unavailable', stats=stats,
                    token_usage=list(planner_usage.values()),
                    token_usage_scope='last observed cumulative SDK total per thread, counted once; unreported interrupted tail unavailable',
                    elapsed_s=stats.get('elapsed_s'), fee='unavailable'),
                reviewer_GPT=dict(calls=reviewer_calls,call_scope=call_scope,
                    elapsed_s=sum(a.get('latency_s', 0.) for a in reviewer_attempts),
                    elapsed_scope='observed completed scoring attempts; interrupted time unavailable',
                    token_usage=token_usage,fee=api_fee,
                    fee_scope='documented stateless API input rate; actual invoice unavailable' if http_started else 'Codex currency unavailable'),
                VLA=rpc_counts['VLA'], SAM3=rpc_counts['SAM3'])


def main(args):
    output = args.output.resolve()
    if output.exists(): raise FileExistsError(output)
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError('Registered nonnegative seed required')
    plan = json.loads(args.plan.read_text())
    score_only_review = plan.get('reviewer_output_contract', {}).get('mode') == 'score_only'
    matches = [r for r in plan['episodes'] if r['run_id'] == output.name]
    if len(matches) != 1: raise ValueError('Run must be registered before launch')
    registration = matches[0]
    from research.v4.integrations.provider_admission import validate_native_admission
    planner_transport = validate_native_admission(plan, registration, ROOT)
    from research.v4.development.user_review_revision_20261026.policy_revision import snapshot as policy_snapshot
    registered_policy = policy_snapshot(plan,registration)
    from research.v4.development.decision_gate_dev_20261019.backend import registered_backend, MODELS
    reviewer_backend = registered_backend(args, registration, plan)
    if (registration['method'] != args.method or registration['instance'] != args.instance
            or registration['seed'] != args.seed or registration['planner_model'] != args.planner_model
            or registration['thresholds'] != [args.tau_task, args.tau_action, args.tau_commit]):
        raise ValueError('Arguments differ from the fixed registration')
    formal=registration.get('phase')=='formal'
    physical_profile=registration.get('configuration_profile')
    if physical_profile not in (None,'current_formal_shared_configuration'):
        raise ValueError('Unknown registered physical/scoring configuration')
    current_configuration=formal or physical_profile=='current_formal_shared_configuration'
    if formal and (not args.formal or plan.get('authorization',{}).get('state') != 'RUNNING'):
        raise ValueError('Registered current formal authorization required')
    if args.formal and not formal: raise ValueError('Formal flag requires formal registration')
    if formal:
        from research.v4.bridge.formal_scope_controls import in_scope
        if not in_scope(plan,registration):raise ValueError('Planner/method pair paused by user')
        os.environ['SAFEVLA_FORMAL_PLAN']=str(Path(args.plan).resolve())
    from research.v4.bridge.formal_parallel_runtime import validate_native_slot
    available_slots=validate_native_slot(plan,registration,args.gpu)if formal else (args.gpu,)
    user_review=registration.get('user_review_protocol') in ('user-review-dev5-20261008',
                                                          'user-review-dev6-local-recovery-20261008')
    user_review = user_review or formal
    wall_limit=None if formal else (5400 if user_review else 2700)
    if registration.get('wall_limit_s',wall_limit)!=wall_limit:
        raise ValueError('Registered wall limit differs from this protocol')
    render_gpu=args.gpu
    resource_identity={}
    if formal and len(available_slots)==4:
        from research.v4.bridge.distributed_gpu_resources import render_device,identity
        render_gpu=render_device(plan,args.gpu);resource_identity=identity(plan,args.gpu)
    os.environ.update(MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID=str(render_gpu),
        SAFEVLA_SOURCE_CUSTODY='0', SAFEVLA_RUN_OUTPUT=str(output),
        SAFEVLA_EXECUTION_PROFILE='confidence_formal_20261008' if formal else ('user_review_development_20261008' if user_review else 'operation_development_20261004'),
        SAFEVLA_VISUAL_HOLDING='', SAFEVLA_TRANSPORT_WORKSPACE_BUDGET='transport-workspace-dev1-20261005',
        SAFEVLA_CAPACITY_RECOVERY='local-candidate-capacity-dev3-20261005')
    from research.v4.bridge.protocol import use_installed_codex_cli
    from research.v4.bridge.run_b0 import bind_rpent_checkout, start_service, selected_familiarization
    use_installed_codex_cli(); bind_rpent_checkout()
    from research.v4.leash_v43.run_manifest import run_manifest, atomic_json
    from research.v4.leash_v43.registry import GripperCalibration
    from research.v4.bridge.formal_recording import formal_video_scope, RECORDING
    from research.v3.bridge.observation_artifacts import install_initial_occupancy_state
    from research.v4.development.transport_workspace_dev_20261005.progress import claim
    from research.v4.development.g2_candidate_evidence_dev_20261005.capacity_runtime import planner_scope
    from research.v4.familiarization.shared_prompt_draft.binding import EPISODE_BUDGET_RECOVERY_INSTRUCTIONS
    from .backend import LunaBackend
    from research.v4.development.confidence_gates_20261006.config import PROMPT_VERSION
    from .core import GateCore, Thresholds
    from .evidence import EvidenceBuilder, plain
    from .runner import run_single_episode, run_persistent_episode
    from .family_cli_seam import select_configuration, make_private_evaluator, finish_private_evaluator
    from rpent.utils.vla_client import VLAClient
    from rpent.utils.sam3_client import Sam3Client

    binding = replace(selected_familiarization('shared_familiarization'),
        recovery_rules=EPISODE_BUDGET_RECOVERY_INSTRUCTIONS,
        recovery_rule_version='common-episode-budget-recovery-v2-20261007')
    from research.v4.development.user_review_revision_20261026 import profile as reviewed_profile
    if reviewed_profile.enabled(plan):
        reviewed_profile.validate(plan)
        if registration.get('initial_condition'):
            binding = replace(binding,
                recovery_rules=binding.recovery_rules+'\n\n'+reviewed_profile.INITIAL_NOTICE,
                recovery_rule_version=reviewed_profile.INITIAL_VERSION)
    if binding.sha256 != '7994c71d22d8beacf676157f14158d13185110e6e20f5817a5dcde8c6e588462':
        raise ValueError('The unchanged five shared examples are required')
    if current_configuration:
        from .formal_configurations import select as select_configuration
    configured = select_configuration(args.instance)
    if user_review:
        if args.instance.startswith('P2-'):
            configured=configure_p2_outcome(configured,registration.get('placement_task_version'))
        if args.instance.startswith(('I','P3-')):
            from .placement_outcome_dev4 import configure
            configured=configure(configured)
        if args.instance.startswith('S'):
            from .s_placement_outcome_dev4 import configure
            configured=configure(configured)
    if current_configuration:
        configured=configure_formal_private_outcome(configured)
    thresholds = Thresholds(args.tau_task, args.tau_action, args.tau_commit)
    rpc_counts = {name:dict(calls=0,elapsed_s=0.,fee='unavailable') for name in ('VLA','SAM3')}
    outcome = dict(version='confidence-natural-dev1-20261006', instance=args.instance,
        method=args.method, seed=args.seed, condition=registration['condition'],
        physical_binding_revision=getattr(configured.AcceptedScene,'contact_zone_version',None),
        placement_task_version=getattr(configured,'placement_task_version',None),
        task_instruction_revision=getattr(configured,'task_instruction_revision',None),
        shared_grasp_retention_version=getattr(configured,'shared_grasp_retention_version',None),
        physical_grasp_annotation_revision=getattr(configured,'physical_grasp_annotation_revision',None),
        trial_protocol_version=registration.get('trial_protocol_version'),
        placement_outcome_revision=getattr(configured,'placement_outcome_revision',None),
        scene_event_revision=getattr(configured,'scene_event_revision',None),
        private_outcome_revision=getattr(configured,'private_outcome_revision',None),
        private_r_grasp_witness_revision=getattr(configured,'private_r_grasp_witness_revision',None),
        private_r_critical_union_revision=getattr(configured,'private_r_critical_union_revision',None),
        private_snack_grasp_revision=getattr(configured,'private_snack_grasp_revision',None),
        private_i_danger_holding_revision=getattr(configured,'private_i_danger_holding_revision',None),
        task_instructions=list(configured.instructions),
        private_evaluator=f'{configured.Evaluator.__module__}:{configured.Evaluator.__name__}',
        physical_gpu=resource_identity.get('compute_gpu',args.gpu),concurrent_formal_slots=len(available_slots),
        **resource_identity,
        started_utc=utc(), status='running', formal=formal, planner_model=args.planner_model, formal_cell=registration.get('cell_id'), attempt_number=registration.get('attempt_number'), model_calls=None,
        reviewer_backend=reviewer_backend if args.method!='B0' else None,
        configuration_profile=physical_profile,
        wall_limit_s=wall_limit, user_review_protocol=registration.get('user_review_protocol'),
        first_blocker=None, error=None, source_custody=False)
    started = time.monotonic(); result = None; evaluator = None; core = None; backend = None
    private_trace = None
    evaluator_scene = [None]
    business_deadline = float('inf') if wall_limit is None else started + wall_limit
    os.environ['SAFEVLA_CAPACITY_DEADLINE'] = str(business_deadline)
    def expired(*_): expire_episode_wall(outcome)
    def check_budget():
        if time.monotonic() >= business_deadline: expired()
        size, _ = output_size(output)
        if size > 1024**3:
            outcome.update(budget_exhausted=dict(resource='output_bytes',
                limit_bytes=1024**3, observed_output_bytes=size,
                reason='One-GiB development episode output limit'),
                termination_kind='episode_budget_exhausted')
            raise DevelopmentBudgetExceeded(
                'One-GiB development episode output limit', resource='output_bytes')
    def observe(scene):
        # This callback runs after a real physics step. Record that already
        # executed state before terminating at the unchanged output ceiling.
        row = evaluator.observe(scene)
        append_json(private_trace, plain(dict(sequence=scene.sequence,
            observed_wall_elapsed_s=time.monotonic()-started, **row)))
        check_budget()
    def on_scene(scene):
        nonlocal evaluator
        evaluator_scene[0] = scene
        evaluator = make_private_evaluator(configured,scene)
        if not getattr(configured,'persistent',False): observe(scene)
    def on_round_prepared(index,instruction,previous):
        scene = evaluator_scene[0]
        family = args.instance.split('-')[0]
        # Accepted S1.start_round already sampled this state. S2.observe is a
        # read-only facade. S3 must not sample the same physics time twice.
        if family=='S1':
            append_json(private_trace, plain(dict(sequence=scene.sequence,
                round_number=index, **evaluator.last)))
        elif family=='S2' or index==1:
            observe(scene)
        else:
            append_json(private_trace,dict(sequence=scene.sequence,round_number=index,
                event='round_prepared_no_physics_sample'))
    def gate_log(row):
        check_budget(); append_json(output/'gates.jsonl', row)
    def save_execution_logs(snapshot):
        # Save the existing business logs before Scene.close on every exit.
        if (evaluator is not None
                and getattr(configured, 'evaluator_requires_final_scene', False)):
            evaluator.capture_final_witness(evaluator_scene[0])
        atomic_json(output/'dispatch.json', plain(snapshot['dispatch_logs']))
        atomic_json(output/'control_trace.json', plain(snapshot['common_control_traces']))
        outcome['executed_sequence'] = snapshot['executed_sequence']
        try:
            outcome['final_private_step_capture'] = capture_final_private_step(
                evaluator, evaluator_scene[0], private_trace,
                wall_elapsed_s=time.monotonic()-started)
        except Exception as exc:
            outcome['private_final_capture_error']=f'{type(exc).__name__}: {exc}'
            raise
    prior_alarm = signal.signal(signal.SIGALRM, expired)
    prior_term = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError('Episode interrupted')))
    files = list(SOURCES)
    if plan.get('planner_extension'):
        files.extend(('research/v4/bridge/formal_planner_extensions.py',
                      'research/v4/integrations/antigravity/qualification/guard.py'))
        integration = ROOT/'research/v4/integrations/antigravity'
        files.extend(str(p.relative_to(ROOT)) for p in sorted(integration.glob('*.py'))
                     if not p.name.startswith('test_'))
    from research.v4.integrations.provider_admission import manifest_source_files
    if formal and len(available_slots)==4:
        files.extend(('research/v4/bridge/remote_gpu_services.py','research/v4/bridge/distributed_gpu_resources.py','research/v4/bridge/remote_service_evidence.py'))
    files.extend(manifest_source_files(args.planner_model))
    if reviewed_profile.enabled(plan):
        reviewed_profile.validate(plan)
        files.extend(str(p.relative_to(ROOT)) for p in reviewed_profile.PACKAGE.iterdir()
                     if p.suffix in ('.py', '.txt'))
        outcome['planner_guidance_version'] = (reviewed_profile.GUIDANCE_VERSION
            if args.method != 'B0' else None)
        outcome['initial_condition'] = registration.get('initial_condition')
        outcome['preheld'] = bool(registration.get('initial_condition'))
        outcome['replaces_prior_run'] = registration.get('replaces_prior_run')
    if formal and len(available_slots)>1:
        files.append("research/v4/bridge/service_endpoint_startup.py")
    if current_configuration:
        files.extend((PACKAGE+'formal_configurations.py',PACKAGE+'phase_timing.py'))
        if args.instance.startswith(('I','S1-','S2-')):
            files.extend((PACKAGE+'basket_reception_dev10_20261014.py',PACKAGE+'s_safe_response_dev10_20261014.py',PACKAGE+'user_seed0_outcome_dev10_20261014.py'))
        if args.instance.startswith('E3-'):
            files.append(PACKAGE+'e3_material_retention_20261014.py')
        if args.instance.startswith('R'):
            files.extend((PACKAGE+'r_physical_grasp_witness_repair9.py',PACKAGE+'r_critical_holding_union_repair9.py'))
        if args.instance.startswith('E1-'):
            files.append(PACKAGE+'e1_receiving_scope_20261012.py')
        if args.instance.startswith('S3-'):
            files.append(PACKAGE+'s3_actual_snack_grasp_20261012.py')
        files.extend(('research/v4/development/transport_workspace_dev_20261005/requests.py',
                      'research/v4/development/g2_candidate_evidence_dev_20261005/capacity_runtime.py',
                      'research/v4/development/g2_candidate_evidence_dev_20261005/capacity.py',
                      'research/v4/development/g2_candidate_evidence_dev_20261005/async_capacity_recovery.py',
                      'research/v4/development/g2_candidate_evidence_dev_20261005/session_routing_recovery.py'))
    if user_review:
        files.extend((PACKAGE+'user_permission_reply.py',
                      'research/v4/candidates/multiround_bridge_20260924/driver.py',
                      'research/v4/bridge/protocol.py'))
        if args.instance.startswith('P2-'):
            files.extend((PACKAGE+'p2_front_placement_v5.py',
                'research/v4/results/confidence_gate_dev_20261006/dev5_initialization_qualification.json'))
            if configured.placement_task_version in ('p2-front-placement-dev6-robot-facing-frame-20261008','p2-front-placement-dev7-approximate-region-20261008','p2-front-placement-dev8-bounded-handle-touch-20261008','p2-uncued-right-hand-table-dev9-20261014'):
                files.append(PACKAGE+'p2_front_placement_v6.py')
            if configured.placement_task_version in ('p2-front-placement-dev7-approximate-region-20261008','p2-front-placement-dev8-bounded-handle-touch-20261008','p2-uncued-right-hand-table-dev9-20261014'):
                files.append(PACKAGE+'p2_front_placement_v7.py')
            if configured.placement_task_version in ('p2-front-placement-dev8-bounded-handle-touch-20261008','p2-uncued-right-hand-table-dev9-20261014'):
                files.append(PACKAGE+'p2_front_placement_v8.py')
            if configured.placement_task_version=='p2-uncued-right-hand-table-dev9-20261014':
                files.append(PACKAGE+'p2_uncued_instruction_20261014.py')
        if args.instance.startswith(('I','P3-')): files.append(PACKAGE+'placement_outcome_dev4.py')
        if args.instance.startswith('S'):
            files.extend((PACKAGE+'s_placement_outcome_dev4.py',PACKAGE+'placement_outcome_dev4.py',
                          'research/v4/bridge/s1_private_trace_freshness.py'))
    if args.instance.startswith('P2-'):
        files.extend((PACKAGE+'p2_front_placement.py', PACKAGE+'p2_front_placement_v4.py'))
    if args.instance == 'P3-01':
        files.append(PACKAGE+'p3_sampled_rest.py')
    if args.instance == 'P4-01':
        files.append(PACKAGE+'p4_physical_completion.py')
    if args.instance == 'E2-01':
        files.extend((PACKAGE+'e2_physical_event.py', PACKAGE+'e2_physical_lift_trigger.py'))
    if args.instance == 'E3-01':
        files.extend((PACKAGE+'e3_physical_event.py', PACKAGE+'e2_physical_lift_trigger.py'))
    if args.instance == 'P2-01':
        files.extend((PACKAGE+'p2_contact_zone.py',
                      'research/v4/candidates/user_rework_p2_20260927/scene_v3.py',
                      'research/v4/candidates/p2_v42_t3_20260925/scene_safe_slot_v2.py',
                      'research/v4/candidates/p2_v42_t3_20260925/knife_visible_blade.xml'))
    if args.instance == 'S2-01':
        files.extend(('research/v4/candidates/s2_higher_method_20260926/rpent_facade.py',
                      'research/v4/candidates/s2_01_seeded_yaw_20260927/scene.py',
                      'research/v4/candidates/s2_01_ordered_sampler_actual_20260927/scene.py',
                      'research/v4/candidates/s2_scene_20260924/scene.py',
                      'research/v4/candidates/s2_scene_20260924/private_dynamics.py',
                      'research/v4/candidates/release_ketchup_region_assist_20260926/region_assist.py'))
    if args.instance == 'P1-06':
        # This run consumes the P106 private adapter, not the R1/R5 binding.
        files.remove(PACKAGE+'r15_configuration.py')
        files.append(PACKAGE+'p106_private_data_compat.py')
    if args.instance in ('R2-01','R3-01','R4-01'):
        files.remove(PACKAGE+'r15_configuration.py')
        files.append(PACKAGE+'r234_configuration.py')
        if args.instance == 'R3-01':
            files.append(PACKAGE+'r3_attachment_observation.py')
            # R301 uses its own accepted physical reset, not the P1 reset recipe.
            for unused in ('research/v4/development/p102_contact_repair_20261005/binding.py',
                           'research/v4/development/p1_single_right_arm_20261004/layout.py',
                           'research/v4/results/confidence_gate_dev_20261006/initialization_qualification.json'):
                files.remove(unused)
            files.extend(('research/v4/candidates/r301_horizontal_u_physical_prepare_20260928/scene.py',
                          'research/v4/candidates/r3_01_horizontal_cold_bar_actual_binding_v2_20260928/scene.py'))
    if args.instance.startswith('R1-'):
        files.extend(('research/v4/tasks/r1/liquid_intersection.py',
                      'research/v4/tasks/r1/evaluator.py',
                      'research/v4/accelerated/r1_v42_evaluator.py'))
    if args.formal and args.instance.startswith('R3-') and args.instance!='R3-01':
        files.extend(('research/v4/tasks/r3/evaluator.py',
                      'research/v4/bridge/r3_family_lite.py',
                      PACKAGE+'r3_attachment_observation.py'))
    if current_configuration:
        files.extend(('research/v4/bridge/formal_parallel_runtime.py',PACKAGE+'private_outcome_audit_repair8_20261010.py'))
        if args.instance.startswith('P1-'):
            files.extend((PACKAGE+'p1_physical_completion_repair8.py',
                'research/v4/candidates/p1_ma_fallback_20260925/candidate.py'))
        if args.instance.startswith(('E','S1-')):
            files.append(PACKAGE+'e_s_outcome_repair8_20261010.py')
        if args.instance.startswith('P4-'):
            files.extend((PACKAGE+'p4_all_physical_completion_repair8.py',PACKAGE+'p4_physical_completion.py'))
        if args.instance=='P4-02':
            files.extend((PACKAGE+'p4_heatmat_support_repair8.py',
                'research/v4/candidates/p4_02_straightener_heatmat_20260926/evaluator.py'))
        if args.instance.startswith('I') and not args.instance.endswith('-01'):
            files.extend((PACKAGE+'i_human_contact_repair8.py',PACKAGE+'i_named_danger_holding_repair9.py',
                'research/v4/candidates/i_t7_candle_can_derived_20260926/evaluator.py'))
    if formal:
        files.append('research/v4/bridge/formal_scope_controls.py')
    if current_configuration and args.instance.startswith('P2-'):
        from .p2_reception_outcome_dev12_20261016 import ASSETS
        files.extend((PACKAGE+'p2_reception_outcome_dev12_20261016.py',ASSETS[args.instance]))
    if formal and args.instance=='P1-03':
        files.extend((PACKAGE+'p103_bridge_grasp_annotation_20261014.py',
            'research/v4/results/formal_confidence_20261008/user_seed0_review_20261014/root/p103_physical_qualification.json',
            'research/v4/candidates/p1_03_scissors_case_20260926/scene.py',
            'research/v4/candidates/p1_03_scissors_case_20260926/assets/scissors.xml'))
    files.append(str(Path(binding.path).relative_to(ROOT)))
    if args.formal and args.instance.startswith(('P3-', 'E2-')):
        files.append(PACKAGE+'private_outcome_repair_20261009.py')
    if args.formal and args.instance.startswith('E2-'):
        files.append('research/v4/candidates/e2_native_event_20260926/evaluator.py')
    r_evaluation = None
    if args.instance.startswith('R'):
        from .r_result_derivation import apply_confidence_r_result
        r_evaluation = json.loads((ROOT/'research/v4/configs/run_manifest.json').read_text())[
            'work_package_20261003']['formal_authorization']['result_evaluation']
        for unused in ('research/v4/development/p102_contact_repair_20261005/binding.py',
                       'research/v4/development/p1_single_right_arm_20261004/layout.py',
                       'research/v4/results/confidence_gate_dev_20261006/initialization_qualification.json'):
            if unused in files: files.remove(unused)
        files.append(PACKAGE+'r_result_derivation.py')
    try:
        from .phase_timing import timing_scope
        with robot_job(formal=formal,gpu=args.gpu,available_slots=available_slots,run_id=output.name), timing_scope(output), run_manifest(output, repo=ROOT, method=args.method,
            instance_id=args.instance, seed=args.seed, gates={g:args.method!='B0' for g in ('G1','G2','G3')},
            planner_model=args.planner_model,planner_reasoning='high',
            reviewer_model=('gpt-6-luna' if reviewer_backend=='luna' else MODELS[reviewer_backend]) if args.method!='B0' else None,
            reviewer_reasoning='high' if args.method!='B0' and reviewer_backend=='luna' else None,
            source_files=files,run_type='confidence_formal' if formal else 'confidence_natural_development',
            budgets=dict(primitives_per_round=60,primitives_per_episode=180,wall_s=wall_limit,
                output_bytes=1024**3,grasp_attempts_per_subgoal=None,
                grasp_recovery_limit_scope='common_episode_budgets',vla_chunk_controls=5),
            familiarization=binding.manifest_fields(),recording=RECORDING,
            policy=dict(**registered_policy,
                reviewer_prompt_version=PROMPT_VERSION,
                previous_calibration_validated_for_this_prompt=False,
                applied=args.method!='B0',thresholds=asdict(thresholds),include_score=False,include_comment=not score_only_review)) as manifest:
            manifest.update(physical_gpu=resource_identity.get('compute_gpu',args.gpu),concurrent_formal_slots=len(available_slots),**resource_identity)
            if planner_transport is not None:
                manifest['planner_transport'] = planner_transport
            if reviewed_profile.enabled(plan):
                manifest['user_review_revision'] = dict(version=reviewed_profile.VERSION,
                    planner_guidance_version=reviewed_profile.GUIDANCE_VERSION if args.method!='B0' else None,
                    planner_guidance_sha256=reviewed_profile.GUIDANCE_SHA if args.method!='B0' else None,
                    original_12_policy_rules_preserved=True,
                    shared_policy_revision=registration.get('policy_revision'),
                    initial_condition=registration.get('initial_condition'),
                    replaces_prior_run=registration.get('replaces_prior_run'),
                    interpretation='Selective diagnostic revision; pre-held resets differ from original task starts')
            if not formal: claim('natural_episode_starts')
            if wall_limit is not None:
                signal.setitimer(signal.ITIMER_REAL, max(.001,business_deadline-time.monotonic()))
            private_trace = output/'evaluator-only.jsonl'
            with ExitStack() as stack:
                recording_stack=ExitStack()
                stack.callback(recording_stack.close)
                # Human-only timeline starts before service/scene startup.
                # No future camera frame is backdated into that initial wait.
                recording_stack.enter_context(formal_video_scope(output,enabled=True))
                from research.v4.bridge.formal_parallel_runtime import isolated_mcp_startup,startup_lock
                from research.v4.development.transport_workspace_dev_20261005.progress import HERE
                if formal and len(available_slots)>1:stack.enter_context(isolated_mcp_startup(HERE))
                try:
                    options = SimpleNamespace(vla_url=None,sam3_url=None,vla_gpu=args.gpu,sam3_gpu=args.gpu,
                        formal_parallel_auto_port=formal and len(available_slots)>1)
                    if formal and len(available_slots)==4:
                        from research.v4.bridge.distributed_gpu_resources import pair_for_slot
                        options.remote_service_pair=pair_for_slot(plan,args.gpu,output,output.name)
                    clients = {}
                    for name, label in (('vla','VLA'),('sam3','SAM3')):
                        from .phase_timing import phase
                        with phase('service_start', service=label):
                            # Kernel-assigned model ports need no cross-GPU load lock.
                            # Keep original serial VLA/SAM3 order within each episode.
                            with (nullcontext() if options.formal_parallel_auto_port else startup_lock(HERE)):
                                rpc, daemon, url = start_service(name,options,output)
                                if daemon is not None and daemon.poll() is not None:
                                    raise RuntimeError('Episode-owned '+name+' service exited during startup')
                        if daemon is not None: stack.callback(daemon.stop)
                        original = rpc.call
                        def counted(method,*values,_original=original,_name=name,_label=label,**keywords):
                            check_budget(); count = method == ('predict' if _name=='vla' else 'segment')
                            t = time.monotonic()
                            if count:rpc_counts[_label]['calls'] += 1
                            try:return _original(method,*values,**keywords)
                            finally:
                                if count:rpc_counts[_label]['elapsed_s'] += time.monotonic()-t
                        rpc.call = counted; clients[name] = rpc
                        outcome.setdefault('service_urls',{})[name] = url
                    if args.method != 'B0':
                        if reviewer_backend=='luna':
                            backend = LunaBackend(output,timeout_s=180.)
                        else:
                            from research.v4.development.decision_gate_dev_20261019.backend import DecisionBackend, JevBackend
                            backend = (DecisionBackend if reviewer_backend=='openai_decisions' else JevBackend)(output,timeout_s=180.)
                        stack.callback(backend.close)
                        core = GateCore(backend,thresholds,on_record=gate_log,include_comment=not score_only_review)
                    install_initial_occupancy_state()
                    with planner_scope():
                        entry = run_persistent_episode if getattr(configured,'persistent',False) else run_single_episode
                        if user_review and not getattr(configured,'persistent',False):
                            from .user_permission_reply import run_single_with_permission
                            entry=run_single_with_permission
                        extra = dict(private_evaluator_reader=lambda:evaluator,
                            on_round_prepared=on_round_prepared) if getattr(configured,'persistent',False) else {}
                        if user_review and getattr(configured,'persistent',False):
                            extra['user_permission_reply']=True
                        extra['timeout_s']=5400 if wall_limit is None else wall_limit
                        with phase('planner_with_tools'):
                            result = entry(configured=configured,arm=args.method,
                                output=output,seed=args.seed,device=render_gpu,
                                vla_client=VLAClient(clients['vla']),sam3_client=Sam3Client(clients['sam3']),
                                planner_model=args.planner_model,reasoning='high',core=core,
                                evidence_builder=EvidenceBuilder(output) if core is not None else None,
                                calibration=GripperCalibration('Existing Panda calibration',0.,.003,.075,3,.001,False),
                                prompt_binding=binding,on_sim_step=observe,on_scene=on_scene,
                                on_execution_logs=save_execution_logs,**extra)
                    outcome.update(status='completed',terminal=result['run'].terminal,
                        planner_error=result['run'].planner_result.error,
                        primitives=sum(c['name'] in ('hold','move_to','move_pose','rotate_pitch','rotate_wrist',
                            'set_gripper','release','pi0_pick','pi0_doubled','vla_act') for c in result['run'].calls),
                        executed_sequence=result['executed_sequence'],system_failure=result['system_failure'])
                    record_planner_budget_metadata(outcome, result['run'].planner_result,
                                                  limit_s=5400 if wall_limit is None else wall_limit)
                    atomic_json(output/'planner_result.json', serializable_run(result['run']))
                    if 'persistent' in result:
                        atomic_json(output/'rounds.json', serializable_persistent(result['persistent']))
                        quota=next((r.boundary for r in result['persistent']['rounds']
                            if r.boundary.endswith('_quota')),None)
                        if quota:
                            outcome.update(budget_exhausted=dict(resource=quota,
                                wall_limit_s=wall_limit,primitives_per_round=60,
                                primitives_per_episode=180,planner_turns_per_round=80),
                                termination_kind='shared_resource_budget_exhausted')
                except BaseException as exc:
                    record_execution_exception(outcome, exc)
                finally:
                    signal.setitimer(signal.ITIMER_REAL,0)
                    recording_stack.close()
                    records = [] if core is None else core.records
                    outcome['model_calls'] = model_counts(output,result,rpc_counts,records)
                    if evaluator is not None:
                        terminal = outcome.get('terminal') or {}
                        status = terminal.get('status')
                        if status not in ('done','asked','refused'): status=None
                        with phase('private_evaluation'):
                            physical = record_private_evaluation(outcome, evaluator,
                                lambda: finish_private_evaluator(configured,evaluator,terminal,
                                    final_scene=evaluator_scene[0]), output)
                        if physical is not None and r_evaluation is not None:
                            try:
                                outcome = apply_confidence_r_result(outcome,outcome['physical'],r_evaluation)
                            except Exception as exc:
                                outcome.update(evaluation_status='pending_missing_evidence',
                                    error='Result derivation failed: '+type(exc).__name__+': '+str(exc),
                                    evaluation_traceback=traceback.format_exc())
                    outcome.update(ended_utc=utc(),wall_s=time.monotonic()-started,
                        gate_count=len(records),valid_scores=sum(
                            type(r.get('p_safe')) in (int,float) and 0<=r['p_safe']<=1 for r in records),
                        reviewer_refusals=sum(bool(r.get('reviewer_refusal')) for r in records))
                    outcome['included_in_formal'] = formal and outcome.get('status')=='completed' and not outcome.get('error') and not outcome.get('system_failure')
                    atomic_json(output/'episode.json', plain(outcome))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,prior_alarm);signal.signal(signal.SIGTERM,prior_term)
    size, count = output_size(output)
    outcome.update(output_bytes=size,output_files=count)
    atomic_json(output/'episode.json', plain(outcome))
    print(json.dumps({k:outcome.get(k) for k in ('status','label','error','wall_s','output_bytes','output_files')}),flush=True)
    return 0 if outcome['status']=='completed' and not outcome.get('system_failure') else 2


def argument_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=('B0','Ours-conf'),required=True)
    parser.add_argument('--instance',default='P1-02');parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--gpu',type=int,default=1);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--planner-model',choices=('gpt-6-astra','gpt-6.1-sol','gpt-6-luna','gemini-4-argon-high','grok-4.7-high-fast'),default='gpt-6-astra')
    parser.add_argument('--formal',action='store_true')
    parser.add_argument('--reviewer-backend',choices=('luna','openai_decisions','jev'),default='luna')
    for gate in ('task','action','commit'):parser.add_argument('--tau-'+gate,type=float,default=0.)
    return parser


if __name__ == '__main__':
    raise SystemExit(main(argument_parser().parse_args()))
