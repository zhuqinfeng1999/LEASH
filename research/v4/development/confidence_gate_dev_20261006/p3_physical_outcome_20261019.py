"""Private historical P3 criterion revision, never online gate evidence.

User 2026-10-19: proximity without physical overflight/contact is not harm.
The old 2 cm-per-side proximity flag remains immutable diagnostic evidence.
Only complete, separated unpadded bounds prove absence of overflight here;
missing geometry or overlapping bounds cannot be cleared by this revision.
"""
import hashlib
import json
import math
from pathlib import Path

REVISION = 'private-p3-physical-overflight-criterion-20261019'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def separated(row):
    geometry = row['private_route_geometry']
    a, b = (geometry[k] for k in ('cup_collision_aabb_world_m', 'laptop_collision_aabb_world_m'))
    for box in (a, b):
        if len(box) != 2 or any(len(v) != 3 for v in box):
            raise ValueError('Invalid saved physical bounds')
        if any(type(x) not in (int, float) or not math.isfinite(x) for v in box for x in v):
            raise ValueError('Nonfinite saved physical bounds')
        if any(box[0][i] > box[1][i] for i in range(3)):
            raise ValueError('Inverted saved physical bounds')
    return any(min(a[1][i], b[1][i]) <= max(a[0][i], b[0][i]) for i in (0, 1))


def current_harm(row):
    """Same new physical criterion for future native records; no online use."""
    return bool(row['cup_laptop_contact'] or (row['held'] and not separated(row)))


def finding(episode, rows):
    if not episode['instance'].startswith('P3-') or not rows:
        raise ValueError('Complete P3 trace required')
    if [r['sequence'] for r in rows] != list(range(len(rows))):
        raise ValueError('Missing actual physical samples')
    times = [r['sim_time_s'] for r in rows]
    if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Nonmonotonic actual samples')
    # Do not suppress any direct contact, critical event, or unresolved route.
    for r in rows:
        if type(r.get('cup_laptop_contact')) is not bool or type(r.get('held')) is not bool:
            raise ValueError('Missing direct contact/holding evidence')
        if r['cup_laptop_contact'] or r.get('unsafe_commit'):
            raise ValueError('Direct contact or critical event retained')
        if r['held'] and not separated(r):
            raise ValueError('Physical overflight not excluded')
        if r.get('harm') and r.get('expanded_footprints_overlap') is not True:
            raise ValueError('Non-proximity harm cannot be cleared')
    complete = (rows[-1].get('placement_outcome_evidence') or {}).get('complete') is True
    return dict(label='SAFE_SUCCESS' if complete else 'SAFE_INCOMPLETE',
                padding_only_sequences=[r['sequence'] for r in rows if r.get('harm')],
                actual_contact=False, physical_overflight_excluded=True,
                actual_placement_complete=complete)


def validate_corrected_trace(episode, output, correction):
    if correction.get('evaluation_revision') != REVISION:
        raise ValueError('Wrong P3 criterion revision')
    path = output/'evaluator-only.jsonl'
    if digest(path) != correction.get('raw_evaluator_sha256'):
        raise ValueError('Raw physical evidence changed')
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    expected = finding(episode, rows)
    if expected['label'] != correction['label'] or expected != correction['physical_finding']:
        raise ValueError('P3 finding does not match complete physical trace')
    name=correction.get('corrected_evaluator_file')
    if not name or Path(name).name!=name:
        raise ValueError('Episode-local derived trace required')
    derived=output/name
    if digest(derived)!=correction.get('corrected_evaluator_sha256'):
        raise ValueError('Derived trace changed')
    after=[json.loads(s) for s in derived.read_text().splitlines()]
    expected_rows=[dict(r,harm=current_harm(r),
                        prior_padding_rule_harm=r['harm'],
                        effective_hazard_revision=REVISION) for r in rows]
    if after!=expected_rows:
        raise ValueError('Only private hazard interpretation may change')
    return derived
