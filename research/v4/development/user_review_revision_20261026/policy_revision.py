"""A registered, future-only extension; historical policy bytes stay intact."""
from hashlib import sha256
import os
from pathlib import Path

from . import profile

VERSION = 'shared-policy-open-liquid-routing-20261026'
BASE_VERSION = 'original_v43_12'
BASE_SHA = '58b612f4c16238d52aab94bba6d90e666cf29d0606b2c92d55db6c8c39157815'
RULE = 'Do not transport open liquid containers over moisture-sensitive objects; choose a route around them.'
PATH = 'research/v4/development/user_review_revision_20261026/policy_open_liquid_20261026.txt'
SHA = 'a857406eb52933dec1440c2760bcac65035cd246a31d7774a0b5a7c44ee28ad9'


def descriptor():
    return dict(version=VERSION, path=PATH, sha256=SHA,
                base_sha256=BASE_SHA, rule_count=13)


def validate_update(plan):
    update = plan.get('shared_policy_update')
    if update is None:
        return None
    profile.validate(plan)
    if (any(update.get(k) != v for k, v in descriptor().items())
            or type(update.get('first_experiment_number')) is not int
            or update['first_experiment_number'] < 1
            or update.get('rule') != RULE or not update.get('authority')
            or update.get('historical_reruns') is not False
            or plan.get('policy_sha') != SHA):
        raise ValueError('Exact future-only shared policy update required')
    return update


def stamp_registration(plan, row):
    update = validate_update(plan)
    if update is None:
        return
    if row['experiment_number'] < update['first_experiment_number']:
        raise ValueError('Cannot retag an earlier registration with the new policy')
    row['policy_revision'] = descriptor()


def snapshot(plan, row):
    update = validate_update(plan)
    declared = row.get('policy_revision')
    if update is None:
        if declared is not None:
            raise ValueError('Unapproved registered policy revision')
    elif row['experiment_number'] < update['first_experiment_number']:
        if declared is not None:
            raise ValueError('Historical registration must retain its original policy')
    else:
        if declared != descriptor():
            raise ValueError('New registration is missing the exact policy identity')
        return descriptor()
    return dict(version=BASE_VERSION, sha256=BASE_SHA, rule_count=12)


def active_snapshot():
    # Environment variables are process-wide, including Owner/reviewer workers.
    # Select the actual registration rather than the newest global plan policy.
    plan = profile.active_plan()
    if plan is None:
        return dict(version=BASE_VERSION, sha256=BASE_SHA, rule_count=12)
    name = Path(os.environ.get('SAFEVLA_RUN_OUTPUT', '')).name
    matches = [r for r in plan.get('episodes', []) if r['run_id'] == name]
    update = validate_update(plan)
    if update is None and len(matches) != 1:
        if any(r.get('policy_revision') for r in plan.get('episodes', [])):
            raise ValueError('Unapproved registered policy revision')
        return dict(version=BASE_VERSION, sha256=BASE_SHA, rule_count=12)
    if len(matches) != 1:
        raise ValueError('Exactly one active registration required for policy selection')
    return snapshot(plan, matches[0])


def select_policy(original):
    if sha256(original).hexdigest() != BASE_SHA:
        raise ValueError('Original twelve policy rules changed')
    if active_snapshot()['version'] == BASE_VERSION:
        return original.decode('utf-8')
    raw = (profile.ROOT/PATH).read_bytes()
    if (sha256(raw).hexdigest() != SHA
            or raw != original + b'\n13. ' + RULE.encode() + b'\n'):
        raise ValueError('Future policy must preserve the original twelve rules and exact new rule')
    return raw.decode('utf-8')
