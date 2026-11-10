"""Copy only closed episode-owned inference logs; no action/model retries."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import re
import shlex
import subprocess


def archive_service_logs(output, *, runner=None):
    from research.v4.bridge.remote_gpu_services import ssh_command, _ssh_environment, WORKSPACE, USER, HOST
    output = Path(output)
    source = output / 'remote_gpu_worker.jsonl'
    target = output / 'remote_service_log_archive.json'
    if target.exists():
        return json.loads(target.read_bytes())
    rows = [json.loads(line) for line in source.read_bytes().splitlines() if line.strip()]
    closed = [row for row in rows if row.get('event') == 'owned_worker_closed']
    ready = [row['message'] for row in rows if row.get('event') == 'worker_message'
             and row.get('message', {}).get('op') == 'ready']
    results = []
    errors = []
    if len(closed) != 1:
        errors.append('Remote worker has no unique owned closeout')
    for value in (ready if len(closed) == 1 else []):
        try:
            service = value.get('service')
            lease = value.get('lease_id')
            if service not in ('vla', 'sam3') or not isinstance(lease, str) or not re.fullmatch('[0-9a-f]{32}', lease):
                raise ValueError('Invalid original remote service/lease identity')
            if lease != closed[0].get('lease_id') or value.get('gpu') != closed[0].get('remote_gpu'):
                raise ValueError('Remote log lease differs from the actual closed worker')
            expected = Path(WORKSPACE) / 'worker/runs' / output.name / lease / (service + '.log')
            if value.get('log_path') != str(expected):
                raise ValueError('Remote log escaped this exact episode lease')
            directory = output / 'remote_service_logs' / lease
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (service + '.log')
            if path.exists():
                raise ValueError('Refusing to overwrite pre-existing archived log')
            code = ('import pathlib,sys;p=pathlib.Path(sys.argv[1]);'
                    'assert p.is_file()and not p.is_symlink()and p.stat().st_size<=16777216;'
                    'sys.stdout.buffer.write(p.read_bytes())')
            command = [WORKSPACE + '/env/bin/python', '-c', code, str(expected)]
            partial = path.with_suffix('.partial')
            with partial.open('xb') as stream:
                result = (runner or subprocess.run)(ssh_command() + [USER + '@' + HOST, shlex.join(command)],
                    stdout=stream, stderr=subprocess.PIPE, env=_ssh_environment(), timeout=30, check=False)
            if result.returncode != 0:
                raise RuntimeError('Read-only remote service log copy failed (exit ' + str(result.returncode) + ')')
            partial.replace(path)
            results.append(dict(service=service, lease_id=lease, remote_path=str(expected),
                path=str(path.relative_to(output)), bytes=path.stat().st_size,
                sha256=sha256(path.read_bytes()).hexdigest(), original_bytes_preserved=True))
        except Exception as exc:
            errors.append(type(exc).__name__ + ': ' + str(exc))
    if {x['service'] for x in results} != {'vla', 'sam3'}:
        errors.append('Both original inference-service logs were not archived')
    value = dict(version='closed-original-remote-service-log-custody-20261030',
        utc=datetime.now(timezone.utc).isoformat(), complete=not errors, files=results,
        errors=errors, model_requests=0, robot_action_replay=False,
        credential_files_read_or_copied=False)
    target.write_text(json.dumps(value, indent=2) + '\n')
    return value
