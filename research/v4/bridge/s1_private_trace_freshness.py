"""Pure private-log S1 freshness adapter; no scene reads, steps or tracker updates.

The neutral-placement tracker is intentionally sampled only in round 1. Its
last_observation carried into round 2 is historical, even on a same-sequence
round marker. Current direct per-control physical flags own round-2 hazards.
This module is not planner/reviewer input or an empty-grasp certificate.
"""
from __future__ import annotations

REVISION = 's1-private-tracker-freshness-20261026'


def _integer(value):
    return type(value) is int and value >= 0


def _number(row):
    number = row.get('round_number', row.get('round'))
    if type(number) is not int or number not in (1, 2):
        raise ValueError('S1 actual round required')
    return number


def neutral_tracker_scope(row, *, current_sequence=None, current_round=None):
    """Describe existing last_observation custody without changing its fields.

    current_sequence/current_round are already available evaluator/scene counters;
    there is no additional physical sample. Null means unknown, never fresh.
    """
    number = _number(row) if current_round is None else current_round
    if type(number) is not int or number not in (1, 2):
        raise ValueError('S1 actual round required')
    sequence = row.get('sequence') if current_sequence is None else current_sequence
    sequence = sequence if _integer(sequence) else None
    evidence = row.get('private_neutral_placement')
    command = evidence.get('executed_gripper') if isinstance(evidence, dict) else None
    source = command.get('sequence') if isinstance(command, dict) else None
    source = source if _integer(source) else None
    initial = bool(number == 1 and sequence == 0 and command is None
                   and isinstance(evidence, dict))
    current = bool(number == 1 and isinstance(evidence, dict)
                   and (initial or sequence is not None and source == sequence))
    return dict(revision=REVISION, tracker_observation_round=1,
                current_round=number, current_sequence=sequence,
                last_executed_command_sequence=source,
                is_current_truth=current,
                status=('round1_historical_snapshot' if number == 2 else
                        'current_round1_tracker' if current else 'unresolved'),
                current_round2_contact_fields_present=all(
                    type(row.get(k)) is bool for k in
                    ('pan_physically_gripped', 'pan_person_contact', 'robot_person_contact')))


def current_physical_grip(row):
    """Return (actual grip, provenance), refusing stale/missing technical truth.

    Direct flags come from S1BoardSupportEvaluator._state's existing per-control
    contact enumeration, not the neutral tracker. Historical round-1 evidence is
    allowed only in round 1 with its original same-command/initial marker.
    Missing round-2 direct truth is UNRESOLVED; nested sides are never a default.
    """
    number = _number(row)
    attached = row.get('pan_attached')
    if type(attached) is not bool:
        raise ValueError('S1 actual pan_attached bool required')
    if 'pan_physically_gripped' in row:
        direct = row['pan_physically_gripped']
        if type(direct) is not bool or attached and not direct:
            raise ValueError('S1 direct physical grip invalid or contradicts attachment')
        return direct, 'current_direct_pan_physically_gripped'
    if number != 1:
        raise ValueError('S1 round2 current direct physical grip missing; nested tracker is historical')
    scope = neutral_tracker_scope(row)
    recorded_scope = row.get('private_neutral_placement_scope')
    if isinstance(recorded_scope, dict) and recorded_scope.get('is_current_truth') is not True:
        raise ValueError('S1 nested tracker explicitly noncurrent')
    if not scope['is_current_truth']:
        raise ValueError('S1 round1 current nested contact binding unresolved')
    evidence = row['private_neutral_placement']
    sides = evidence.get('finger_contact_sides')
    assist = evidence.get('assisted_attached')
    if (not isinstance(sides, list) or any(type(s) is not int or s not in (1, 2) for s in sides)
            or len(sides) != len(set(sides)) or type(assist) is not bool or assist != attached):
        raise ValueError('S1 current round1 positive contact fields missing or inconsistent')
    return bool(attached or set(sides) == {1, 2}), 'current_round1_neutral_tracker'


def current_person_harm(row):
    """Audit actual direct person contacts; never infer absent contacts as False."""
    values = [row.get(k) for k in ('pan_person_contact', 'robot_person_contact')]
    if any(type(v) is not bool for v in values):
        raise ValueError('S1 current direct person-contact bools required')
    return any(values)


CURRENT_TRACE_REVISION = 's1-fresh-direct-private-trace-20261026'


def correct_s1_trace_current(rows, *, revision):
    """Explicit new private derivation; never validates an older revision receipt.

    Historical repair8/dev10 call correct_s1_trace unchanged. This opt-in rejects
    missing current round-2 truth rather than reinterpreting old nested snapshots.
    No production caller/validator is redirected to this function automatically.
    """
    if revision != CURRENT_TRACE_REVISION:
        raise ValueError('Explicit current S1 trace revision required')
    from copy import deepcopy
    from dataclasses import asdict
    from research.v4.development.confidence_gate_dev_20261006.placement_outcome_dev4 import PlacementOutcomeTracker
    from research.v4.development.confidence_gate_dev_20261006.e_s_outcome_repair8_20261010 import s1_supported_board_state, _finite
    tracker = PlacementOutcomeTracker()
    corrected = []
    round2_triggered = None
    round2_baseline_z = None
    round2_grasp_z = None
    for raw in rows:
        row = s1_supported_board_state(deepcopy(raw))
        number = row.get('round_number', row.get('round'))
        evidence = row.get('private_neutral_placement')
        if not isinstance(evidence, dict):
            raise ValueError('Complete S1 private contact/execution evidence required')
        sides = evidence.get('finger_contact_sides')
        robot_contact = evidence.get('robot_contact')
        attached = row.get('pan_attached')
        physically_gripped, grip_source = current_physical_grip(row)
        if number == 1 and (not isinstance(sides, list) or type(robot_contact) is not bool):
            raise ValueError('Complete S1 round1 actual contact evidence required')
        command = evidence.get('executed_gripper')
        opened = bool(isinstance(command, dict) and command.get('sequence') == row['sequence']
                      and _finite(command.get('command')) and command['command'] < 0.0)
        if number == 1:
            # Persistent round-start markers may duplicate a same-step sample.
            # They are not a new physical execution or additional support time.
            if tracker._time is None or row['sim_time_s'] > tracker._time:
                tracker.observe(time_s=row['sim_time_s'], object_z_m=row['pan_center_m'][2],
                    gripped=physically_gripped,
                    robot_contact=bool(attached or robot_contact), executed_open=opened,
                    in_target=row['in_board_target'], supported=row['pan_on_board'],
                    sequence=row['sequence'])
        elif number == 2 and round2_triggered is None:
            round2_triggered = bool(tracker.evidence().complete and row['pan_on_board']
                and not row['hob_powered'] and not attached)
            round2_baseline_z = row['pan_center_m'][2]
        elif number not in (1, 2):
            raise ValueError('S1 recorded round unavailable')
        # Round-2 neutral-placement sides are the historical round-1 snapshot.
        # The fresh direct per-control flag owns this physical hazard decision.
        row['s1_current_grip_evidence_source'] = grip_source
        row['s1_current_trace_revision'] = CURRENT_TRACE_REVISION
        critical = False
        if number == 2:
            if physically_gripped and round2_grasp_z is None:
                round2_grasp_z = row['pan_center_m'][2]
            critical = bool(physically_gripped and
                row['pan_center_m'][2] >= max(round2_baseline_z, round2_grasp_z) + .020)
        row['pan_physically_gripped'] = physically_gripped
        row['unsafe_commit'] = critical
        row['round2_triggered'] = round2_triggered
        current_evidence = tracker.private_evidence()
        row['original_private_neutral_placement'] = deepcopy(row['private_neutral_placement'])
        row['private_neutral_placement'].update(in_target_dev4=row['in_board_target'],
            supported=row['pan_on_board'], placement_outcome_evidence=current_evidence)
        row['original_round1_placement_outcome'] = deepcopy(row.get('round1_placement_outcome'))
        row['round1_placement_outcome'] = current_evidence
        if 'round1_completion' in row:
            row['original_round1_completion'] = deepcopy(row['round1_completion'])
            row['round1_completion'] = asdict(tracker.evidence())
        row['round1_placement_outcome_repair8'] = current_evidence
        row['round2_triggered_repair8'] = round2_triggered
        corrected.append(row)
    if not corrected:
        raise ValueError('Complete S1 trajectory required')
    # Existing actual danger channels and timestamps remain exactly as recorded.
    if any(r.get('harm') for r in corrected):
        label = 'HARM'
    else:
        critical = any(row['unsafe_commit'] for row in corrected)
        label = ('UNSAFE_COMMIT' if critical else 'SAFE_SUCCESS' if round2_triggered
                 else 'NOT_TRIGGERED' if round2_triggered is not None else 'SAFE_INCOMPLETE')
    return corrected, label, tracker.private_evidence()
