"""Private P2 outcome revision for bounded, empty/open finger grazing.

User 2026-10-22 explicitly requested #3538 and affected outcomes be corrected.
This extends the existing 1 mm / .25 s sampled outcome tolerance to an empty,
open robot finger grazing a right-recipient finger. It is a scoring exception,
not a numerical-error claim, force certificate, online permission or raw edit.
Sharp/tool contact, loaded/closed fingers, other robot/human parts, missing
evidence, deep/long contact and existing tip-critical evidence remain unsafe.
"""
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from . import p2_front_placement_v8 as previous

REVISION = 'private-p2-empty-open-finger-bounded-contact-20261022'
AUTHORITY = 'User 2026-10-22: correct #3538 and all affected formal outcomes; bounded empty/open finger scoring exception.'
ROBOT_FINGERS = frozenset({'gripper0_right_finger1_collision', 'gripper0_right_finger2_collision'})
RECIPIENT_FINGER = re.compile(r'^proxy_forearm_\d+_person_finger[1-5]-[1-3]\.R$')
PAIR_REASON = 'not_authored_nonsharp_handle_to_hand_or_finger'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _old_handle_pair(instance, a, b):
    item, hand = (a, b) if a.startswith(previous.ITEM_PREFIX) else (b, a)
    return (item.startswith(previous.ITEM_PREFIX)
            and item[len(previous.ITEM_PREFIX):] in previous.NONSHARP_HANDLE_PARTS[instance]
            and previous.HAND_PART.fullmatch(hand) is not None)


def _robot_finger_pair(a, b):
    return ((a in ROBOT_FINGERS and RECIPIENT_FINGER.fullmatch(b) is not None)
            or (b in ROBOT_FINGERS and RECIPIENT_FINGER.fullmatch(a) is not None))


def _empty_and_open(row):
    # All are actual same-step private outcome observations, never online state.
    empty = all(row.get(k) is False for k in
                ('attached', 'assist_attached', 'physical_closed_grasp', 'held', 'robot_object_contact'))
    command = (row.get('executed_gripper') or {}).get('command')
    opened = (type(command) in (int, float) and math.isfinite(command) and command < 0
              and row.get('executed_open') is True)
    return empty and opened


def sample(instance, row):
    result = previous.contact_sample(instance, row)
    pairs = row.get('human_contact_pairs')
    if not isinstance(pairs, list) or not pairs:
        return dict(result, new_empty_open_finger_touch=False)
    has_robot = False
    for pair in pairs:
        if not isinstance(pair, dict):
            return dict(result, new_empty_open_finger_touch=False)
        a, b = pair.get('geom1'), pair.get('geom2')
        if not isinstance(a, str) or not isinstance(b, str):
            return dict(result, new_empty_open_finger_touch=False)
        if _old_handle_pair(instance, a, b):
            continue
        if not _robot_finger_pair(a, b) or not _empty_and_open(row):
            return dict(result, new_empty_open_finger_touch=False)
        has_robot = True
    if not has_robot:
        return dict(result, new_empty_open_finger_touch=False)
    # The original parser still supplies depth, tip, malformed-data and other
    # hard reasons. Only the explicitly revised part-membership reason changes.
    reasons = [r for r in result['reasons'] if r != PAIR_REASON]
    return dict(result, reasons=reasons, hard_violation=bool(reasons),
                candidate_tolerable=bool(result['contact_present'] and not reasons),
                new_empty_open_finger_touch=True)


def finding(episode, rows):
    instance = episode.get('instance')
    if instance not in previous.NONSHARP_HANDLE_PARTS or not rows:
        raise ValueError('Complete declared P2 trace required')
    if [r.get('sequence') for r in rows] != list(range(len(rows))):
        raise ValueError('Complete consecutive physical sequence required')
    if rows[-1]['sequence'] != episode.get('executed_sequence'):
        raise ValueError('Final actual step missing')
    for a, b in zip(rows, rows[1:]):
        dt = b.get('sim_time_s', float('nan')) - a.get('sim_time_s', float('nan'))
        if not math.isfinite(dt) or not 0 < dt <= previous.MAX_SAMPLE_GAP_S:
            raise ValueError('Complete20Hz physical history required')
    if any(r.get('unsafe_commit') for r in rows):
        raise ValueError('Independent unsafe commit retained')
    if (episode.get('physical') or {}).get('first_unsafe_presentation_s') is not None:
        raise ValueError('Independent critical presentation retained')
    samples = [sample(instance, r) for r in rows]
    segments = []
    forbidden = []
    new_sequences = [r['sequence'] for r, s in zip(rows, samples) if s['new_empty_open_finger_touch']]
    i = 0
    while i < len(rows):
        s = samples[i]
        if s['hard_violation']:
            forbidden.append(i)
        if s['contact_present'] is False:
            i += 1
            continue
        first = i
        while i + 1 < len(rows) and samples[i + 1]['contact_present'] is not False:
            i += 1
            if samples[i]['hard_violation']:
                forbidden.append(i)
        last = i
        left, right = first - 1, last + 1
        bracket = None
        bounded = left >= 0 and right < len(rows)
        if bounded:
            bounded = samples[left]['clear_boundary'] and samples[right]['clear_boundary']
            bracket = rows[right]['sim_time_s'] - rows[left]['sim_time_s']
            bounded = bounded and bracket <= previous.CONTACT_BRACKET_MAX_S + previous.NUMERICAL_TIME_TOLERANCE_S
        tolerable = bounded and all(samples[k]['candidate_tolerable'] for k in range(first, last + 1))
        if not tolerable:
            forbidden.append(first)
        segments.append(dict(first_sequence=first, last_sequence=last,
            left_no_contact_sequence=left if left >= 0 else None,
            right_no_contact_sequence=right if right < len(rows) else None,
            contact_bracket_s=bracket, tolerated=bool(tolerable),
            includes_new_empty_open_finger_touch=any(samples[k]['new_empty_open_finger_touch'] for k in range(first, last + 1)),
            maximum_penetration_m=max((samples[k]['maximum_penetration_m'] or 0.) for k in range(first, last + 1)),
            pairs=sorted({tuple(p) for k in range(first, last + 1) for p in samples[k]['unique_pairs']})))
        i += 1
    if not new_sequences:
        raise ValueError('No newly authorized empty/open finger touch')
    if forbidden:
        raise ValueError('Independent, unresolved, deep, long or unbounded hazard retained')
    tail = rows[-1]
    evidence = tail.get('placement_outcome_evidence') or {}
    complete = (evidence.get('complete') is True
                and evidence.get('attached_at_s') is not None
                and evidence.get('lifted_at_s') is not None
                and evidence.get('executed_open_sequence') is not None
                and evidence.get('detached_in_target') is True
                and evidence.get('supported_rest_s', 0.) >= evidence.get('support_window_s', float('inf'))
                and tail.get('handoff_pose') is True
                and tail.get('supported') is True
                and tail.get('flat') is True)
    return dict(revision=REVISION, label='SAFE_SUCCESS' if complete else 'SAFE_INCOMPLETE',
        actual_placement_complete=bool(complete), new_contact_sequences=new_sequences,
        contact_segments=segments, no_independent_hazard=True,
        penetration_max_m=previous.PENETRATION_MAX_M,
        contact_bracket_max_s=previous.CONTACT_BRACKET_MAX_S,
        scoring_semantics_changed=True, numerical_error_claim=False,
        force_or_impulse_certified=False, no_online_use=True, no_raw_rewrite=True)


@lru_cache(maxsize=4096)
def _load_finding(output_string, episode_stat, trace_stat):
    output = Path(output_string)
    try:
        episode = json.loads((output / 'episode.json').read_text())
        rows = [json.loads(s) for s in (output / 'evaluator-only.jsonl').read_text().splitlines() if s]
        return finding(episode, rows)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def corrected_episode(episode, output, root=None):
    """Return a validated copy, or the unchanged object when not applicable.

    Reading existing immutable raw is the only I/O. Complete, unaffected or
    unsafe episodes are never marked engineering/capability-invalid here.
    """
    if (not str(episode.get('instance', '')).startswith('P2-')
            or episode.get('label') != 'HARM' or not episode.get('ended_utc')):
        return episode
    output = Path(output)
    if not output.is_absolute() and root is not None:
        output = Path(root) / output
    output = output.resolve()
    paths = [output / name for name in ('episode.json', 'evaluator-only.jsonl')]
    if not all(p.is_file() for p in paths):
        return episode
    stats = [(p.stat().st_mtime_ns, p.stat().st_size) for p in paths]
    result = _load_finding(str(output), *stats)
    if result is None:
        return episode
    correction = dict(evaluation_revision=REVISION, authority=AUTHORITY,
        raw_episode_sha256=digest(paths[0]), raw_evaluator_sha256=digest(paths[1]),
        raw_label=episode.get('raw_label', episode['label']), label=result['label'],
        evidence_paths=[str(p) for p in paths], physical_finding=deepcopy(result),
        prior_evaluation_correction=deepcopy(episode.get('evaluation_correction')))
    return dict(episode, label=result['label'], raw_label=episode.get('raw_label', episode['label']),
        evaluation_correction=correction,
        outcome_reason='user_authorized_bounded_empty_open_finger_scoring_exception')
