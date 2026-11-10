"""Shared provisional v4.2 action budget for B0 and Ours development.

Historical RPent LIBERO successes include a 35-call task and 2/82 above the
previous 24-call candidate. 48 reserves 13 calls beyond that observed maximum.
One archived success hit the old 50 planner-turn cap, so the provisional
planner ceiling is 80 turns. I1 #84 took 377 s over 18 live planner turns;
1200 s could preempt 80 turns before the call cap. The provisional planner
and episode wall ceilings therefore become 2400/3000 s, shared by B0/Ours.
They remain subject to graspable normal-task calibration before formal freeze.
The 3000 s episode wall limit is for a one-round task. Multi-round tasks
need an independent 2400 s allowance for each additional round; otherwise
a three-round S task could be cut off despite satisfying every round limit.
This is not a formal freeze: qualify on graspable SafeVLA normal tasks first.
Historical v4.0 runs used 15/40; v4.2 #69 used the earlier 24/72 candidate.
"""

PROTOCOL_REVISION = "v4.2-resource-candidate-48x144-80turn-2400x3000s"
PER_ROUND_PRIMITIVE_CALLS = 48
PER_EPISODE_PRIMITIVE_CALLS = 144
PLANNER_TURNS_PER_ROUND = 80
PLANNER_TIMEOUT_S = 2400
EPISODE_WALL_TIMEOUT_S = 3000


def use_installed_codex_cli():
    """Use the existing terminal CLI instead of the SDK's older bundled CLI."""
    import os
    from pathlib import Path
    import shutil
    selected = os.environ.get('CODEX_BIN') or shutil.which('codex')
    if not selected or not Path(selected).is_file():
        raise FileNotFoundError('The selected existing Codex CLI is unavailable')
    os.environ['CODEX_BIN'] = str(Path(selected).resolve())
    return os.environ['CODEX_BIN']


def active_live_limits():
    """Explicit resource profile; historical runs keep their original limits."""
    import os
    profile = os.environ.get('SAFEVLA_EXECUTION_PROFILE', 'legacy_30_90')
    if profile == 'legacy_30_90':
        return dict(primitive_calls_per_round=30, primitive_calls_per_episode=90,
                    episode_wall_seconds=1800, planner_turns_per_round=80)
    if profile == 'operation_development_20261004':
        return dict(primitive_calls_per_round=60, primitive_calls_per_episode=180,
                    episode_wall_seconds=2700, planner_turns_per_round=80)
    if profile == 'user_review_development_20261008':
        return dict(primitive_calls_per_round=60, primitive_calls_per_episode=180,
                    episode_wall_seconds=5400, planner_timeout_seconds=5400,
                    planner_turns_per_round=80)
    if profile == 'confidence_formal_20261008':
        # No episode wall ceiling. Individual model/Owner timeouts remain finite.
        return dict(primitive_calls_per_round=60, primitive_calls_per_episode=180,
                    episode_wall_seconds=float('inf'), planner_timeout_seconds=5400,
                    planner_turns_per_round=80)
    raise ValueError('Unknown explicit execution resource profile')


def active_policy_text():
    """Select the declared policy asset; old runs keep their original default."""
    import os
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    version = os.environ.get('SAFEVLA_POLICY_VERSION', 'v43_20260928')
    if version == 'v43_20260928':
        text = (root/'research/v4/docs/leash_v43_revised_20260928.txt').read_text()
        return text.split('Shared Household Safety Policy: DRAFT\n', 1)[1].split('这份政策', 1)[0].strip()
    if version == 'stage_scoped_dev1':
        return (root/'research/v4/configs/p1_stage_scoped_dev1_20261004.txt').read_text().strip()
    if version == 'stage_scoped_dev2':
        return (root/'research/v4/configs/p1_stage_scoped_dev2_20261004.txt').read_text().strip()
    if version == 'stage_scoped_dev3':
        return (root/'research/v4/configs/local_candidate_dev3_20261005.txt').read_text().strip()
    raise ValueError('Unknown declared policy version')


def remaining_formal_items(plan, used, *, primary_integrity_completed=False):
    """Pause the registered order at its first-stage integrity checkpoint."""
    remaining = [item for item in plan['order'] if item['ordinal'] not in used]
    checkpoint = plan.get('first_stage_integrity_checkpoint_after_ordinal')
    if checkpoint is not None and not primary_integrity_completed:
        remaining = [item for item in remaining if item['ordinal'] <= checkpoint]
    return remaining


def current_formal_cells(rows):
    """Exclude only explicitly superseded infrastructure attempts, never scores."""
    return [row for row in rows if row.get('formal_cell_current') is not False]


def formal_registration_items(plan, rows, authority):
    """Only named implementation repairs can repeat an already started cell."""
    by_id = {row['id']: row for row in rows}
    by_ordinal = {item['ordinal']: item for item in plan['order']}
    selected = []
    for repair in authority.get('infrastructure_startup_repairs', []):
        if repair.get('registered_replacement_id'):
            continue
        old = by_id[repair['failed_attempt_id']]
        if old['status'] != 'invalid' or old.get('valid_method_result') is True:
            raise ValueError('A named infrastructure repair requires an invalid attempt')
        if repair['runtime_revision'] != authority['runtime_revision']:
            raise ValueError('Infrastructure repair must use the declared repaired revision')
        selected.append((by_ordinal[old['preregistered_ordinal']], old['id']))
    remaining = remaining_formal_items(plan, {row['preregistered_ordinal'] for row in rows},
        primary_integrity_completed=bool(authority.get('primary_seed0_integrity_completed_utc')))
    return selected + [(item, None) for item in remaining]


def episode_wall_timeout_s(round_count: int) -> int:
    """Provisional wall ceiling for one to three sequential instruction rounds.

    The 600 s difference between first-round wall and planner limits reserves
    scene/service startup. Later rounds each retain their full planner limit.
    This is a shared B0/Ours capacity candidate, not a formal freeze.
    """
    if type(round_count) is not int or not 1 <= round_count <= 3:
        raise ValueError("round_count must be an integer from 1 to 3")
    return EPISODE_WALL_TIMEOUT_S + (round_count - 1) * PLANNER_TIMEOUT_S
