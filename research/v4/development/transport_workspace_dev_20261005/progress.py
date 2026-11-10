"""One development progress file; no robot decision or source-custody path."""
import datetime as dt
import fcntl
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = HERE / 'progress.json'


def update(callback):
    with (HERE / 'progress.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(PATH.read_text())
        callback(data)
        data['updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        temporary = PATH.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(PATH)
        return data


def continuous_development(data):
    return data.get('development_authorization', {}).get('mode') == 'continuous_development_no_admin_caps'


def available(data, *, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    if (data['stop_requested'] or data['formal_allowed'] is not False
            or data['source_custody'] is not False):
        return False
    if continuous_development(data):
        return True
    deadline = data.get('deadline_utc')
    return isinstance(deadline, str) and now < dt.datetime.fromisoformat(deadline)


def claim(kind, count=1):
    def change(data):
        if not available(data):
            raise RuntimeError('Development stopped or fixed deadline reached')
        if type(count) is not int or count < 1:
            raise ValueError('A claim is a positive integer')
        continuous = continuous_development(data)
        if not continuous and data['used'][kind] + count > data['limits'][kind]:
            raise RuntimeError('Development budget exhausted: ' + kind)
        package=(data.get('definition_repair_package')
                 if kind=='natural_episode_starts' and not continuous else None)
        if package is not None and package['natural_starts_used']+count>package['natural_starts_max']:
            raise RuntimeError('Definition repair natural-start limit exhausted')
        data['used'][kind] += count
        if package is not None:
            package['natural_starts_used']+=count
    return update(change)
