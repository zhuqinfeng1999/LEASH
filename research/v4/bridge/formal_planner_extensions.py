"""Explicit admission of the user-approved Gemini B0/JEV successor stage.

This is configuration and selection for the existing controller, not another
controller, scheduler, retry policy, model alias or change to task physics.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

VERSION = 'antigravity-planner-extension-20261027'
FAMILY = 'gemini4argon'
GEMINI_MODEL = 'gemini-4-argon-high'
TRANSPORT = 'antigravity_cli'
REASONING = 'high'
STAGE = 'existing_repairs_then_gemini'
ROWS = (
    ('gemini4argon_b0', 'Gemini 4 Argon', 'B0', None, 'extension'),
    ('gemini4argon_jev', 'Gemini 4 Argon', 'Ours', 'jev', 'extension'),
)
PENDING = frozenset(('registered_pending', 'repair_pending', 'retry_pending',
                     'registered', 'running', 'configuration_pending'))


def _configuration(plan):
    """Validate constant-size identity fields for repeated scope lookups."""
    config = plan.get('planner_extension')
    if config is None:
        return None
    if (not isinstance(config, dict) or config.get('version') != VERSION
            or not isinstance(config.get('authority'), str) or not config['authority'].strip()
            or config.get('family') != FAMILY or config.get('model_id') != GEMINI_MODEL
            or config.get('transport') != TRANSPORT or config.get('reasoning') != REASONING
            or config.get('enabled_methods') != ['B0', 'Ours']
            or config.get('reviewer_backends') != ['jev'] or config.get('stage') != STAGE
            or type(config.get('enabled')) is not bool):
        raise ValueError('Exact user-approved Gemini high B0/JEV extension required')
    receipt = config.get('connectivity_receipt')
    if receipt is not None and (not isinstance(receipt, dict) or not isinstance(receipt.get('path'), str)
            or not receipt['path'].startswith('research/v4/results/')
            or Path(receipt['path']).is_absolute() or '..' in Path(receipt['path']).parts
            or not isinstance(receipt.get('sha256'), str)
            or re.fullmatch('[0-9a-f]{64}', receipt['sha256']) is None):
        raise ValueError('Saved model-connectivity receipt and SHA256 required')
    if config['enabled'] and receipt is None:
        raise ValueError('Enabled Gemini admission requires completed connectivity qualification')
    return config


def _single_preheld_7294_plan_cell(plan,cell):
    from research.v4.integrations.provider_admission import _single_preheld_7294_plan_cell as exact_cell
    return exact_cell(plan,cell)


def validate_extension(plan):
    """Full frozen-cell validation at queue and registration boundaries."""
    config = _configuration(plan)
    if config is None:
        return None
    selected = config.get('selected_cell_ids')
    existing = config.get('existing_stage_cell_ids')
    if (not isinstance(selected, list) or len(selected) != 552
            or not all(isinstance(value, str) and value for value in selected) or len(set(selected)) != 552
            or not isinstance(existing, list) or not existing
            or not all(isinstance(value, str) and value for value in existing)
            or len(set(existing)) != len(existing) or set(selected) & set(existing)
            or existing != (plan.get('user_review_revision') or {}).get('selected_cell_ids')):
        raise ValueError('Fixed old stage and exactly 552 distinct new Gemini cells required')
    cells = plan.get('cells') or []
    by_id = {cell['id']: cell for cell in cells}
    if len(by_id) != len(cells) or not (set(selected) | set(existing)).issubset(by_id):
        raise ValueError('Extension cell identities are absent or duplicated')
    original_instances = {cell['instance'] for cell in cells if cell.get('planner_model') != GEMINI_MODEL}
    if len(original_instances) != 92:
        raise ValueError('The original 92-instance scene scope must remain fixed')
    actual = set()
    for key in selected:
        cell = by_id[key]
        method = cell.get('method')
        expected_row = 'gemini4argon_b0' if method == 'B0' else 'gemini4argon_jev'
        backend = cell.get('reviewer_backend')
        if (cell.get('planner_model') != GEMINI_MODEL or cell.get('planner_family') != FAMILY
                or cell.get('planner_transport') != TRANSPORT or cell.get('reasoning_effort') != REASONING
                or method not in ('B0', 'Ours')
                or (method == 'B0' and backend not in (None, 'luna'))
                or (method == 'Ours' and backend != 'jev')
                or cell.get('results_matrix_row_id') != expected_row
                or cell.get('trial_protocol_version') != plan.get('version')
                or ((cell.get('initial_condition') or cell.get('preheld')
                     or cell.get('replaces_prior_run'))
                    and not _single_preheld_7294_plan_cell(plan,cell))):
            raise ValueError('Gemini cell differs from the fresh authorized B0/JEV high condition')
        actual.add((cell['instance'], cell['seed'], method))
    expected = {(instance, seed, method) for instance in original_instances
                for seed in (0, 1, 2) for method in ('B0', 'Ours')}
    if actual != expected:
        raise ValueError('Gemini requires exactly the original 92 instances x 3 seeds x 2 arms')
    if {cell['id'] for cell in cells if cell.get('planner_model') == GEMINI_MODEL} != set(selected):
        raise ValueError('Unregistered extra Gemini cells are forbidden')
    return config


def verify_connectivity_receipt(plan, root):
    config = validate_extension(plan)
    if config is None:
        return None
    if config.get('connectivity_receipt') is None:
        if config['enabled']:
            raise ValueError('Enabled Gemini has no qualification receipt')
        return None
    path = (Path(root) / config['connectivity_receipt']['path']).resolve()
    if not path.is_relative_to((Path(root) / 'research/v4/results').resolve()):
        raise ValueError('Connectivity receipt must remain inside project results')
    if sha256(path.read_bytes()).hexdigest() != config['connectivity_receipt']['sha256']:
        raise ValueError('Saved Gemini connectivity evidence changed')
    receipt = json.loads(path.read_text())
    if config['enabled'] and (receipt.get('model_id') != GEMINI_MODEL
            or receipt.get('transport') != TRANSPORT or receipt.get('reasoning') != REASONING
            or any((receipt.get('qualification_checks') or {}).get(key) != 'PASS'
                   for key in ('text', 'tool', 'image', 'resume', 'isolation'))):
        raise ValueError('Text, tool, image, resume and tool-isolation qualification must all pass')
    return receipt


def extension_model_ids(plan):
    return (GEMINI_MODEL,) if _configuration(plan) is not None else ()


def extension_pairs(plan):
    return [(GEMINI_MODEL, method) for method in ('B0', 'Ours')] if _configuration(plan) else []


def extension_cells(plan):
    config = validate_extension(plan)
    selected = set(config['selected_cell_ids']) if config else set()
    return [cell for cell in plan.get('cells', []) if cell['id'] in selected]


def row_id(row):
    if row.get('planner_model', row.get('model')) != GEMINI_MODEL:
        return None
    if row.get('method') == 'B0':
        return 'gemini4argon_b0'
    if row.get('reviewer_backend') == 'jev':
        return 'gemini4argon_jev'
    raise ValueError('Gemini Decision/Luna combinations are not authorized')


def stage_status(plan):
    config = validate_extension(plan)
    if config is None:
        return None
    ids = set(config['existing_stage_cell_ids'])
    old = [cell for cell in plan['cells'] if cell['id'] in ids]
    active = [attempt for attempt in plan.get('episodes', [])
              if attempt.get('cell_id') in ids and attempt.get('status') in ('registered', 'running')]
    pending = [cell for cell in old if cell.get('status') in PENDING]
    gaps = [cell for cell in old if cell.get('status') != 'completed' and cell not in pending]
    # A terminal isolated failure is a disclosed gap, never completed success.
    # Unknown/unclosed cells must not silently unlock the next stage.
    isolated = []
    isolated_details = []
    unresolved = []
    for cell in gaps:
        attempts = [attempt for attempt in plan.get('episodes', []) if attempt.get('cell_id') == cell['id']]
        latest = attempts[-1] if attempts else {}
        if (cell.get('status') == 'failure_pending_classification'
                and latest.get('status') not in ('registered', 'running')
                and (latest.get('engineering_exclusion') or {}).get('no_automatic_retry') is True
                and (latest.get('engineering_exclusion') or {}).get('evidence_path')
                and (latest.get('engineering_exclusion') or {}).get('reason')
                and latest.get('ended_utc')):
            isolated.append(cell['id'])
            proof=latest['engineering_exclusion']
            isolated_details.append(dict(cell_id=cell['id'],run_id=latest.get('run_id'),
                reason=proof.get('reason') or latest.get('failure_class') or 'isolated engineering failure',
                evidence_path=proof.get('evidence_path'),ended_utc=latest['ended_utc']))
        else:
            unresolved.append(cell['id'])
    ready = not active and not pending and not unresolved
    return dict(stage=('gemini' if config['enabled'] else 'queued_blocked_validation') if ready else 'existing_repairs',
                original_planned=len(old), original_completed=sum(cell.get('status') == 'completed' for cell in old),
                original_active=len(active), original_pending=len(pending),
                isolated_original_gap_ids=isolated, isolated_original_gaps=isolated_details,
                unresolved_original_gap_ids=unresolved,
                original_fully_completed=all(cell.get('status') == 'completed' for cell in old),
                existing_stage_closed=ready, gemini_ready=ready and config['enabled'])


def stage_candidates(plan, cells):
    config = validate_extension(plan)
    if config is None:
        return cells
    status = stage_status(plan)
    if status['existing_stage_closed'] and not status['gemini_ready']:
        return []
    ids = set(config['selected_cell_ids'] if status['gemini_ready'] else config['existing_stage_cell_ids'])
    return [cell for cell in cells if cell['id'] in ids]


def activate_successor_stage(plan, root):
    """Promote this exact planned stage at a closed boundary after qualification.

    Caller is the existing controller and persists its plan under its original
    lock. This creates no episode, retry, extra process or new experiment ID.
    """
    config = validate_extension(plan)
    if config is None or not stage_status(plan)['gemini_ready']:
        return False
    verify_connectivity_receipt(plan, root)
    if config.get('activation'):
        return False
    status = stage_status(plan)
    for cell in [cell for cell in extension_cells(plan) if cell.get('planner_model') == GEMINI_MODEL]:
        if cell.get('status') == 'queued_blocked_validation':
            cell['status'] = 'registered_pending'
    config['activation'] = dict(version=VERSION, activated_utc=datetime.now(timezone.utc).isoformat(),
        authority=config['authority'], original_stage_fully_completed=status['original_fully_completed'],
        isolated_original_gaps=status['isolated_original_gaps'],
        qualification_receipt=config['connectivity_receipt'], actual_attempts_created=0)
    return True


def registration_metadata(plan, cell):
    config = validate_extension(plan)
    if config is None or cell.get('planner_model') != GEMINI_MODEL:
        return {}
    if (cell['id'] not in config['selected_cell_ids'] or not stage_status(plan)['gemini_ready']
            or (config.get('activation') or {}).get('version') != VERSION):
        raise ValueError('Gemini cannot start before the registered existing repair stage closes')
    return dict(planner_family=FAMILY, planner_transport=TRANSPORT, reasoning_effort=REASONING,
                results_matrix_row_id=row_id(cell), planner_extension_version=VERSION,
                connectivity_receipt=config['connectivity_receipt'],
                planner_stage_activation=config['activation'])


# The existing Gemini contract stays unchanged. Cursor is a separate explicit
# successor, never a model alias, new controller, or retry of a closed gap.
GROK_VERSION = 'cursor-planner-extension-20261029'
GROK_MODEL = 'grok-4.7-high-fast'
GROK_FAMILY = 'grok47'
GROK_TRANSPORT = 'cursor_acp'
GROK_STAGE = 'existing_repairs_then_gemini_then_grok'
GROK_ROWS = (
    ('grok47_b0', 'Grok 4.7', 'B0', None, 'extension'),
    ('grok47_jev', 'Grok 4.7', 'Ours', 'jev', 'extension'),
)
_gemini_validate = validate_extension
_gemini_verify = verify_connectivity_receipt
_gemini_status = stage_status
_gemini_candidates = stage_candidates
_gemini_activate = activate_successor_stage
_gemini_metadata = registration_metadata
_gemini_row_id = row_id


def _receipt_reference(reference):
    if (not isinstance(reference,dict) or not isinstance(reference.get('path'),str)
            or not reference['path'].startswith('research/v4/results/')
            or Path(reference['path']).is_absolute() or '..' in Path(reference['path']).parts
            or re.fullmatch('[0-9a-f]{64}',str(reference.get('sha256',''))) is None):
        raise ValueError('Project-results evidence path and exact SHA256 required')
    return reference


def _read_evidence(reference,root):
    reference=_receipt_reference(reference)
    path=(Path(root)/reference['path']).resolve()
    if not path.is_relative_to((Path(root)/'research/v4/results').resolve()):
        raise ValueError('Qualification evidence left project results')
    if sha256(path.read_bytes()).hexdigest()!=reference['sha256']:
        raise ValueError('Qualification evidence changed')
    return json.loads(path.read_text())


def _grok_configuration(plan):
    config=plan.get('grok_extension')
    if config is None:return None
    if (not isinstance(config,dict) or config.get('version')!=GROK_VERSION
            or not isinstance(config.get('authority'),str) or not config['authority'].strip()
            or config.get('model_id')!=GROK_MODEL or config.get('family')!=GROK_FAMILY
            or config.get('transport')!=GROK_TRANSPORT or config.get('reasoning')!='high'
            or config.get('service_tier')!='fast' or config.get('enabled_methods')!=['B0','Ours']
            or config.get('reviewer_backends')!=['jev'] or config.get('stage')!=GROK_STAGE
            or config.get('require_full_native') is not True or type(config.get('enabled')) is not bool):
        raise ValueError('Exact Grok high/Fast B0/JEV successor and full-native qualification required')
    for key in ('connectivity_receipt','native_qualification_receipt'):
        if config.get(key) is not None:_receipt_reference(config[key])
        elif config['enabled']:raise ValueError('Enabled Grok requires '+key)
    return config


def extension_configurations(plan):
    return [config for config in (_configuration(plan),_grok_configuration(plan)) if config is not None]


def validate_grok_extension(plan):
    config=_grok_configuration(plan)
    if config is None:return None
    gemini=_gemini_validate(plan)
    if gemini is None:raise ValueError('Grok must follow the registered Gemini stage')
    previous=gemini['existing_stage_cell_ids']+gemini['selected_cell_ids']
    selected=config.get('selected_cell_ids')
    if (config.get('existing_stage_cell_ids')!=previous or not isinstance(selected,list)
            or len(selected)!=552 or not all(isinstance(key,str) and key for key in selected)
            or len(set(selected))!=552 or set(selected)&set(previous)):
        raise ValueError('Exactly 552 fresh Grok cells after the original and Gemini fixed stages required')
    cells=plan.get('cells',[]);by_id={cell['id']:cell for cell in cells}
    if len(by_id)!=len(cells) or not(set(previous)|set(selected)).issubset(by_id):
        raise ValueError('Missing or duplicate Grok/prior cell identity')
    instances={by_id[key]['instance'] for key in gemini['selected_cell_ids']}
    actual=set()
    for key in selected:
        cell=by_id[key];method=cell.get('method');backend=cell.get('reviewer_backend')
        if (cell.get('planner_model')!=GROK_MODEL or cell.get('planner_family')!=GROK_FAMILY
                or cell.get('planner_transport')!=GROK_TRANSPORT or cell.get('reasoning_effort')!='high'
                or method not in ('B0','Ours') or (method=='Ours' and backend!='jev')
                or (method=='B0' and backend not in (None,'luna'))
                or cell.get('results_matrix_row_id')!=('grok47_b0' if method=='B0' else 'grok47_jev')
                or cell.get('trial_protocol_version')!=plan.get('version')
                or cell.get('service_tier','fast')!='fast'
                or cell.get('initial_condition') or cell.get('preheld') or cell.get('replaces_prior_run')):
            raise ValueError('Grok must use the common fresh high/Fast B0/JEV initial condition')
        actual.add((cell['instance'],cell['seed'],method))
    expected={(instance,seed,method) for instance in instances for seed in (0,1,2) for method in ('B0','Ours')}
    if actual!=expected or {c['id'] for c in cells if c.get('planner_model')==GROK_MODEL}!=set(selected):
        raise ValueError('Grok 92 instances x 3 seeds x B0/JEV scope changed')
    return config


def validate_extension(plan):
    gemini=_gemini_validate(plan)
    if gemini is not None:
        if 'require_full_native' in gemini and type(gemini['require_full_native']) is not bool:
            raise ValueError('Native qualification opt-in must be an explicit boolean')
        if gemini.get('native_qualification_receipt') is not None:_receipt_reference(gemini['native_qualification_receipt'])
    validate_grok_extension(plan)
    return gemini


def _valid_hashes(mapping):
    return isinstance(mapping,dict) and bool(mapping) and all(isinstance(k,str) and re.fullmatch('[0-9a-f]{64}',str(v)) is not None for k,v in mapping.items())


def verify_native_qualification(config,root):
    """Software/native closure and measured timing, never a promise of success or parity."""
    if config.get('require_full_native') is not True:return None
    receipt=_read_evidence(config.get('native_qualification_receipt'),root)
    checks=receipt.get('qualification_checks') or {}
    if (receipt.get('model_id')!=config['model_id'] or receipt.get('transport')!=config['transport']
            or receipt.get('reasoning')!='high'
            or (config.get('service_tier') and receipt.get('service_tier')!=config['service_tier'])
            or any(checks.get(k)!='PASS' for k in ('text','tool','image','resume','isolation','full_native','latency_measured'))
            or not _valid_hashes(receipt.get('source_sha256'))
            or not isinstance(receipt.get('speed_parity_proven',False),bool)):
        raise ValueError('Actual source-pinned native and measured-latency qualification required')
    runtime=config.get('cli_runtime') or {}
    pins={key:runtime[key] for key in ('sha256','binary_sha256','entry_sha256') if key in runtime}
    recorded={key:(receipt.get('cli_runtime') or {}).get(key) for key in pins}
    if not _valid_hashes(pins) or recorded!=pins:
        raise ValueError('Native qualification runtime differs from the admitted CLI runtime')
    cases=receipt.get('native_cases')
    if not isinstance(cases,list):raise ValueError('Separate actual native B0/Ours cases required')
    own=[case for case in cases if isinstance(case,dict) and case.get('model_id',config['model_id'])==config['model_id']]
    if {case.get('method') for case in own}!={'B0','Ours'}:
        raise ValueError('Both B0 and Ours native cases required')
    for case in own:
        if (case.get('native_simulation') is not True or case.get('terminal_valid') is not True
                or not case.get('run_id') or not case.get('evidence_path')
                or case.get('outcome') not in ('SAFE_SUCCESS','SAFE_INCOMPLETE','HARM','NOT_TRIGGERED','UNSAFE_COMMIT')
                or type(case.get('nonzero_primitives')) is not bool or type(case.get('physical_completion')) is not bool
                or case.get('source_sha256')!=receipt['source_sha256']
                or not _valid_hashes(case.get('runtime_sha256'))
                or not set(pins.values()).issubset(case['runtime_sha256'].values())
                or not isinstance(case.get('wall_s'),(int,float)) or isinstance(case.get('wall_s'),bool) or not 0<case['wall_s']<float('inf')):
            raise ValueError('Native case lacks valid terminal, outcome, source/runtime identity or raw timing')
    return dict(receipt,speed_parity_proven=receipt.get('speed_parity_proven',False))


def verify_connectivity_receipt(plan,root,model_id=None):
    # Legacy Gemini configurations retain their original five-check contract.
    gemini=_gemini_validate(plan)
    result=None
    if gemini is not None and model_id in (None,GEMINI_MODEL):
        result=_gemini_verify(plan,root)
        if gemini['enabled'] and gemini.get('require_full_native') is True:verify_native_qualification(gemini,root)
    grok=validate_grok_extension(plan)
    if grok is not None and model_id in (None,GROK_MODEL):
        if grok.get('connectivity_receipt') is not None:
            receipt=_read_evidence(grok['connectivity_receipt'],root)
            if grok['enabled'] and (receipt.get('model_id')!=GROK_MODEL or receipt.get('transport')!=GROK_TRANSPORT
                    or receipt.get('reasoning')!='high' or any((receipt.get('qualification_checks')or{}).get(k)!='PASS' for k in ('text','tool','image','resume','isolation'))):
                raise ValueError('Full Grok transport qualification required')
            result=receipt if model_id==GROK_MODEL else result
        if grok['enabled']:verify_native_qualification(grok,root)
    return result


def extension_model_ids(plan):return tuple(config['model_id'] for config in extension_configurations(plan))


def extension_pairs(plan):return [(model,method) for model in extension_model_ids(plan) for method in ('B0','Ours')]


def extension_cells(plan):
    validate_extension(plan)
    ids={key for config in extension_configurations(plan) for key in config['selected_cell_ids']}
    return [cell for cell in plan.get('cells',[]) if cell['id'] in ids]


def row_id(row):
    if row.get('planner_model',row.get('model'))!=GROK_MODEL:return _gemini_row_id(row)
    if row.get('method')=='B0':return 'grok47_b0'
    if row.get('method') in ('Ours','Ours-conf') and row.get('reviewer_backend')=='jev':return 'grok47_jev'
    raise ValueError('Grok is authorized only for B0 and Ours JEV')


def grok_stage_status(plan):
    config=validate_grok_extension(plan)
    if config is None:return None
    # Apply the original closed/isolated-gap criteria to both prior stages.
    # Unknown tails still block; this read-only summary creates no attempt.
    ids=set(config['existing_stage_cell_ids'])
    old=[c for c in plan['cells'] if c['id'] in ids]
    active=[a for a in plan.get('episodes',[]) if a.get('cell_id') in ids and a.get('status') in ('registered','running')]
    pending=[c for c in old if c.get('status') in PENDING or c.get('status')=='queued_blocked_validation']
    isolated=[];unresolved=[]
    for cell in old:
        if cell.get('status')=='completed' or cell in pending:continue
        attempts=[a for a in plan.get('episodes',[]) if a.get('cell_id')==cell['id']]
        latest=attempts[-1] if attempts else {};proof=latest.get('engineering_exclusion') or {}
        if (cell.get('status')=='failure_pending_classification' and latest.get('status') not in ('registered','running')
                and proof.get('no_automatic_retry') is True and proof.get('evidence_path') and proof.get('reason') and latest.get('ended_utc')):
            isolated.append(dict(cell_id=cell['id'],run_id=latest.get('run_id'),reason=proof['reason'],evidence_path=proof['evidence_path'],ended_utc=latest['ended_utc']))
        else:unresolved.append(cell['id'])
    ready=not active and not pending and not unresolved
    from research.v4.bridge.distributed_gpu_resources import concurrent_grok_ready
    concurrent=concurrent_grok_ready(plan)
    return dict(concurrent_remote_override=concurrent,prior_stage_closed=ready,grok_ready=(ready or concurrent)and config['enabled'],
        stage=('gemini_and_grok' if concurrent and config['enabled'] else
               ('grok' if config['enabled'] else 'queued_blocked_validation') if ready else 'previous_stages'),
        prior_planned=len(old),prior_completed=sum(c.get('status')=='completed' for c in old),
        prior_active=len(active),prior_pending=len(pending),isolated_prior_gaps=isolated,
        unresolved_prior_gap_ids=unresolved,prior_fully_completed=all(c.get('status')=='completed' for c in old))


def stage_status(plan):
    status=_gemini_status(plan)
    grok=grok_stage_status(plan)
    if status is not None and grok is not None:
        status=dict(status,grok_progress=grok)
        if grok['prior_stage_closed'] or grok.get('concurrent_remote_override'):status['stage']=grok['stage']
    return status


def stage_candidates(plan,cells):
    grok=validate_grok_extension(plan)
    if grok is None:return _gemini_candidates(plan,cells)
    status=grok_stage_status(plan)
    if status.get('concurrent_remote_override'):
        local=_gemini_candidates(plan,[c for c in cells if c.get('planner_model')!=GROK_MODEL])
        return local+[c for c in cells if c.get('planner_model')==GROK_MODEL]
    if status['prior_stage_closed']:
        ids=set(grok['selected_cell_ids']) if status['grok_ready'] else set()
        return [cell for cell in cells if cell['id'] in ids]
    return _gemini_candidates(plan,[c for c in cells if c.get('planner_model')!=GROK_MODEL])


def activate_successor_stage(plan,root):
    changed=_gemini_activate(plan,root)
    config=validate_grok_extension(plan)
    if config is None or not grok_stage_status(plan)['grok_ready']:return changed
    verify_connectivity_receipt(plan,root,GROK_MODEL)
    if config.get('activation'):return changed
    status=grok_stage_status(plan)
    for cell in extension_cells(plan):
        if cell.get('planner_model')==GROK_MODEL and cell.get('status')=='queued_blocked_validation':cell['status']='registered_pending'
    config['activation']=dict(version=GROK_VERSION,activated_utc=datetime.now(timezone.utc).isoformat(),
        authority=config['authority'],prior_stage_fully_completed=status['prior_fully_completed'],
        concurrent_remote_override=status.get('concurrent_remote_override',False),
        concurrent_resource_authority=(plan.get('execution',{}).get('remote_gpu_execution')or{}).get('authority'),
        isolated_prior_gaps=status['isolated_prior_gaps'],qualification_receipt=config['connectivity_receipt'],
        native_qualification_receipt=config['native_qualification_receipt'],actual_attempts_created=0)
    return True


def registration_metadata(plan,cell):
    if cell.get('planner_model')!=GROK_MODEL:return _gemini_metadata(plan,cell)
    config=validate_grok_extension(plan)
    if (config is None or cell['id'] not in config['selected_cell_ids'] or not grok_stage_status(plan)['grok_ready']
            or (config.get('activation')or{}).get('version')!=GROK_VERSION):
        raise ValueError('Grok cannot register before prior stages and full qualification close')
    return dict(planner_family=GROK_FAMILY,planner_transport=GROK_TRANSPORT,reasoning_effort='high',
        service_tier='fast',results_matrix_row_id=row_id(cell),planner_extension_version=GROK_VERSION,
        connectivity_receipt=config['connectivity_receipt'],native_qualification_receipt=config['native_qualification_receipt'],
        planner_stage_activation=config['activation'])
