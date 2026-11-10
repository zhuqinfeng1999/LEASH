"""Explicit four-slot resource mapping for the one existing formal queue.

Remote GPUs compute the unchanged SAM3/Pi0.5 services. Scene ownership,
rendering, gates and signed-in planners stay in their original local process.
"""
import os
from pathlib import Path
from hashlib import sha256
import json

VERSION = 'local5090-remote4090d-four-slots-20261030'
ROOT = Path(__file__).resolve().parents[3]
REMOTE_MODEL = 'grok-4.7-high-fast'
GEMINI_MODEL = 'gemini-4-argon-high'
GEMINI_ALLOCATION = 'gemini-only-four-gpu-20261030'
FOLLOWUP_MODEL = 'authorized-saved-review-followup'
FOLLOWUP_ALLOCATION = 'saved-review-four-gpu-followup-20261101'
FOLLOWUP_MODELS = ('gpt-6-astra', 'gpt-6.1-sol', 'gpt-6-luna', GEMINI_MODEL)


def configuration(plan, root=None):
    config = (plan.get('execution') or {}).get('remote_gpu_execution')
    if config is None:
        return None
    if (not isinstance(config, dict) or config.get('version') != VERSION
            or config.get('state') != 'QUALIFIED'
            or not isinstance(config.get('authority'), str) or not config['authority'].strip()
            or config.get('local_slots') != [0, 1] or config.get('remote_slots') != [2, 3]
            or config.get('remote_host') != os.environ.get('LEASH_WORKER_HOST')
            or config.get('ai_logins_remain_local') is not True):
        raise ValueError('Explicit qualified user-authorized remote four-slot resources required')
    model = config.get('remote_planner_model', REMOTE_MODEL)
    if model == REMOTE_MODEL:
        if config.get('parallel_grok') is not True:
            raise ValueError('Original Grok allocation requires explicit permission')
    elif model == GEMINI_MODEL:
        if (config.get('parallel_grok') is not False
                or config.get('allocation_version') != GEMINI_ALLOCATION
                or plan.get('grok_extension') is not None):
            raise ValueError('Gemini-only allocation must cancel the Grok extension')
    elif model == FOLLOWUP_MODEL:
        phase = plan.get('postqueue_review') or {}
        ids = phase.get('admitted_cell_ids')
        counts = phase.get('attempt_counts_at_admission')
        if (config.get('parallel_grok') is not False
                or config.get('allocation_version') != FOLLOWUP_ALLOCATION
                or plan.get('grok_extension') is not None
                or phase.get('version') != 'saved-review-followup-20261101'
                or not phase.get('authority')
                or phase.get('maximum_new_attempts_per_cell') != 1
                or not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
                or not all(isinstance(value, str) for value in ids)
                or not isinstance(counts, dict) or set(counts) != set(ids)
                or any(type(value) is not int or not 0 <= value < 3 for value in counts.values())):
            raise ValueError('Fixed user-authorized saved-review follow-up required')
    else:
        raise ValueError('Remote planner model is not explicitly authorized')
    from research.v4.bridge import remote_gpu_services
    validate_configuration = remote_gpu_services.validate_configuration
    services = config.get('service_configurations')
    if not isinstance(services, dict) or set(services) != {'2', '3'}:
        raise ValueError('Two distinct pinned remote service configurations required')
    for slot, device in [('2', 0), ('3', 1)]:
        if validate_configuration(services[slot])['gpu'] != device:
            raise ValueError('Remote execution slot crossed physical GPU identity')
    reference = config.get('qualification_receipt') or {}
    relative = reference.get('path')
    if (not isinstance(relative, str) or not relative.startswith('research/v4/results/')
            or Path(relative).is_absolute() or '..' in Path(relative).parts):
        raise ValueError('Saved worker qualification path required')
    path = Path(root or ROOT) / relative
    if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != reference.get('sha256'):
        raise ValueError('Saved remote worker qualification changed')
    value = json.loads(path.read_bytes())
    if (value.get('result') != 'PASS' or value.get('qualified_remote_gpus') != [0, 1]
            or value.get('exact_sources_models_and_precision') is not True
            or value.get('offline_rpc_fixture_checks_pass') is not True
            or value.get('owned_lease_and_disconnect_cleanup_pass') is not True
            or value.get('ai_credentials_copied') is not False):
        raise ValueError('Actual two-GPU inference and ownership/cleanup proof required')
    client_sha = sha256(Path(remote_gpu_services.__file__).read_bytes()).hexdigest()
    if (value.get('tested_service_configurations') != services
            or value.get('remote_client_source_sha256') != client_sha):
        raise ValueError('Current model/manifest/worker/client identity differs from actual qualification')
    cases = value.get('native_witnesses') or []
    if ({case.get('method') for case in cases} != {'B0', 'Ours'}
            or any(case.get('native_normal_closeout') is not True
                   or case.get('provider_terminal_success') is not True
                   or case.get('remote_services_used') is not True for case in cases)):
        raise ValueError('Actual remote-service B0/Ours normal native witnesses required')
    for case in cases:
        device = case.get('remote_gpu')
        if type(device) is not int or device not in (0, 1):
            raise ValueError('Native witness remote GPU identity is missing')
        expected = sha256(json.dumps(services[str(device + 2)], sort_keys=True).encode()).hexdigest()
        if case.get('service_configuration_sha256') != expected:
            raise ValueError('Native witness is not bound to this tested remote service identity')
    return config


def render_device(plan, slot):
    config = configuration(plan)
    if config is None:
        if slot not in (0, 1):
            raise ValueError('Unconfigured physical GPU slot')
        return slot
    if slot not in (0, 1, 2, 3):
        raise ValueError('Unknown execution slot')
    return slot % 2


def identity(plan, slot):
    render = render_device(plan, slot)
    return dict(execution_slot=slot, compute_host=os.environ.get('LEASH_WORKER_HOST') if slot >= 2 else 'local',
                compute_gpu=slot - 2 if slot >= 2 else slot,
                render_host='local', render_gpu=render,
                remote_ai_logins=False)


def pair_for_slot(plan, slot, output, episode):
    config = configuration(plan)
    if slot in (0, 1):
        return None
    if config is None or slot not in (2, 3):
        raise ValueError('Remote service slot not admitted')
    from research.v4.bridge.remote_gpu_services import RemoteServicePair
    return RemoteServicePair(config['service_configurations'][str(slot)], output, episode)


def concurrent_grok_ready(plan):
    config = configuration(plan)
    return config is not None and config.get('remote_planner_model', REMOTE_MODEL) == REMOTE_MODEL


def prefer_cells(plan, slot, candidates):
    """Follow the explicit current allocation without creating or retrying cells."""
    config = configuration(plan)
    if config is None:
        return candidates
    if config.get('remote_planner_model') == FOLLOWUP_MODEL:
        phase = plan['postqueue_review']
        selected = set(phase['admitted_cell_ids'])
        counts = {}
        for row in plan.get('episodes', []):
            key = (row['cell_id'], row.get('trial_protocol_version'))
            counts[key] = counts.get(key, 0) + 1
        from research.v4.bridge.engineering_autoretry import dispatch_allowed
        return [cell for cell in candidates if cell['id'] in selected
                and cell['planner_model'] in FOLLOWUP_MODELS
                and (counts.get((cell['id'], cell.get('trial_protocol_version')), 0)
                    < phase['attempt_counts_at_admission'][cell['id']] + 1
                    or dispatch_allowed(plan,cell))]
    if config.get('remote_planner_model') == GEMINI_MODEL:
        return [cell for cell in candidates if cell['planner_model'] == GEMINI_MODEL]
    if slot in (2, 3):
        return [cell for cell in candidates if cell['planner_model'] == REMOTE_MODEL]
    local = [cell for cell in candidates if cell['planner_model'] != REMOTE_MODEL]
    return local or candidates
