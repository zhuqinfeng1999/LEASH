"""Register a fresh profile through the existing single campaign controller."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import fcntl

from .profile import VERSION, RESULT_DIRECTORY, THRESHOLDS, OUTPUT_CONTRACT, validate

AUTHORITY = ('Direct user2026-10-21: completely rerun seven Ours arms from the beginning; '
    'G1/G2=.50,G3=.30 for Luna/Decisions/JEV; score only without reasons; '
    'clear historical values from both main metric tables; start the full new batch.')
ORDER = ('gpt-6-astra|luna', 'gpt-6-astra|openai_decisions', 'gpt-6-astra|jev',
         'gpt-6.1-sol|openai_decisions', 'gpt-6.1-sol|jev',
         'gpt-6-luna|openai_decisions', 'gpt-6-luna|jev')


def fresh_cells(previous, row_ids):
    cases = list(dict.fromkeys((c['instance'], c['seed']) for c in previous['cells']))
    if len(cases) != 276 or len({x[0] for x in cases}) != 92:
        raise ValueError('Exactly the original 92 instances and three seeds required')
    old = {(c['planner_model']+'|'+c['reviewer_backend'], c['instance'], c['seed']): c
           for c in previous['cells']}
    cells = []
    for instance, seed in cases:
        for arm in ORDER:
            source = old[(arm, instance, seed)]
            model, backend = arm.split('|')
            cells.append(dict(id=f'{model}|Ours|{backend}|{instance}|{seed}',
                ordinal=len(cells)+1, instance=instance, seed=seed, method='Ours',
                planner_model=model, reviewer_backend=backend,
                placement_task_version=source.get('placement_task_version'),
                results_matrix_variant='optimized', results_matrix_row_id=row_ids[arm],
                trial_protocol_version=VERSION, status='registered_pending', accepted_attempt=None))
    return cells


def register(campaign, target):
    from research.v4.bridge import formal_optimized_matrix as matrix
    from research.v4.development.transport_workspace_dev_20261005.progress import update
    root = campaign.ROOT
    target = Path(target).resolve()
    if target != root/'research/v4/results'/RESULT_DIRECTORY/'plan.json' or target.exists():
        raise ValueError('Fresh exact score-only destination required; never overwrite a plan')
    previous_path = root/'research/v4/results/formal_optimized_matrix_20261020/plan.json'
    here = root/'research/v4/development/transport_workspace_dev_20261005'
    with ExitStack() as locks:
        for file in (root/'research/v4/results/formal_confidence_20261008/controller.lock',
                     here/'robot.lock', here/'robot_gpu_0.lock', here/'robot_gpu_1.lock'):
            handle = locks.enter_context(file.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = campaign.read(previous_path)
        if (previous['authorization']['state'] != 'PAUSED_USER_THRESHOLD_REVIEW'
                or any(a['status']=='running' for a in previous['episodes'])):
            raise ValueError('Old batch must be fully closed and paused for threshold review')
        progress = campaign.read(here/'progress.json')
        if progress.get('active_robot_slots') or progress.get('active_robot_pid'):
            raise ValueError('Reconcile actual robot registration before creating fresh profile')
        for name in ('formal_confidence_20261008', 'formal_visual_backends_20261019', 'formal_g3_transition_20261020'):
            old = campaign.read(root/'research/v4/results'/name/'plan.json')
            if not old['authorization']['state'].startswith(('PAUSED', 'COMPLETE')):
                raise ValueError('Historical queue must remain paused: '+name)
        for name in ('openai', 'typesafe'):
            if not Path('/tmp/safevla-api-review-secrets', name).is_file():
                raise ValueError('Previously configured reviewer credential unavailable: '+name)
        stamp = datetime.now(timezone.utc).isoformat()
        cells = fresh_cells(previous, matrix.ROW_IDS)
        plan = {k: deepcopy(previous[k]) for k in
            ('planner_models', 'reasoning', 'reviewer', 'policy_sha', 'examples_sha',
             'common_user_reply', 'limits', 'retry_rules', 'seed_exposure', 'native_entrypoint',
             'execution', 'g3_transition_rerun', 'batch_failure_isolation') if k in previous}
        auth = deepcopy(previous['authorization'])
        auth.update(state='RUNNING', authority=AUTHORITY,
                    method_states={'B0': 'HISTORICAL_ONLY', 'Ours': 'RUNNING'},
                    ours_display_name='Ours（gpt-6-luna）')
        plan.update(version=VERSION, created_utc=stamp, authority=AUTHORITY,
            authorization=auth, planned_evaluable_cells=1932, thresholds=list(THRESHOLDS),
            reviewer_output_contract=deepcopy(OUTPUT_CONTRACT), cells=cells, episodes=[],
            run_root='research/v4/runs/'+RESULT_DIRECTORY, source_freeze_revision=VERSION,
            next_experiment_number=previous['next_experiment_number'], configuration_pending=[],
            service_blocks={}, live_summary_alias='research/v4/results/formal_confidence_20261008/summary.json',
            latest_resume_request=dict(authority=AUTHORITY,execution_mode='parallel_B0_Ours_dual_gpu',
                concurrent_episodes=2,gpus=[0,1],fresh_from_beginning=True,
                old_completed_cells_reused=False,proxy='http://127.0.0.1:7897'),
            optimized_matrix=dict(version=matrix.VERSION,authority=AUTHORITY,arms=list(matrix.ARMS),
                arm_states={a:'RUNNING' for a in matrix.ARMS},
                event_count_repair_version=previous['optimized_matrix']['event_count_repair_version'],
                fresh_from_beginning=True,results_matrix_variant='optimized',
                results_table_scope='current_seven_arms_only',profile_version=VERSION,
                baseline_reuse='None in current main metric tables; historical raw results preserved',
                display_arm_order=list(ORDER),video_speed=16.,recovery_schedule_deleted=True,
                terminal_error_admission=previous['optimized_matrix']['terminal_error_admission']),
            archived_previous_campaign=dict(plan=str(previous_path.relative_to(root)),
                completed=626,raw_preserved=True,no_result_backfill=True))
        plan['execution']['authority'] = AUTHORITY
        target.parent.mkdir(parents=True, exist_ok=True)
        # Historical material is available only as explicit archived references.
        for name in ('historical_baseline_summary.json','prior_finite_summary.json','historical_timing_references.csv'):
            (target.parent/name).write_bytes((previous_path.parent/name).read_bytes())
        (target.parent/'archived_previous_summary.json').write_bytes((previous_path.parent/'summary.json').read_bytes())
        reference = target.parent/'historical_timing_references.csv'
        plan['historical_timing_references'] = str(reference.relative_to(root))
        plan['historical_timing_sha256'] = sha256(reference.read_bytes()).hexdigest()
        files = set(previous['source_freeze'])
        files.update(str(p.relative_to(root)) for p in Path(__file__).parent.glob('*.py'))
        plan['source_freeze'] = {n:sha256((root/n).read_bytes()).hexdigest() for n in sorted(files)}
        validate(plan)
        matrix.validate_scope(plan)
        campaign.native_entrypoint(plan)
        campaign.save(target, plan)
        def activate(data):
            data.update(formal_allowed=True,source_custody=False,stop_requested=False,
                state='formal_score_only_g3_030_running',current_task=AUTHORITY,
                next_executable_task='existing_formal_dual_gpu_controller',
                root_active=dict(owner='score-only-g3-030-root',status='controller_starting',
                    robot_pid=None,updated_utc=stamp))
        update(activate)
        return plan
