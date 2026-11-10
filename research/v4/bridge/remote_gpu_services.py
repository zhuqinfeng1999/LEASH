"""Episode-owned GPU services over pinned SSH; no planner or queue on worker.

This transport carries only the original SAM3/Pi0.5 RPC. It never issues a
robot control, retries an RPC, copies an AI login, or selects an experiment.
"""
from pathlib import Path
from datetime import datetime, timezone
from queue import Queue, Empty
import json
import os
import re
import shlex
import socket
import subprocess
import threading
import time
import uuid

VERSION = 'episode-owned-ssh-gpu-services-20261030'
HOST = os.environ.get("LEASH_WORKER_HOST", "")
USER = os.environ.get("LEASH_WORKER_USER", "")
WORKSPACE = os.environ.get("LEASH_WORKER_ROOT", "")
KEY = os.environ.get("LEASH_WORKER_SSH_KEY", "")
KNOWN_HOSTS = os.environ.get("LEASH_WORKER_KNOWN_HOSTS", "")


def validate_configuration(config):
    if (not isinstance(config, dict) or config.get('version') != VERSION
            or config.get('host') != HOST or config.get('user') != USER
            or config.get('workspace') != WORKSPACE
            or type(config.get('gpu')) is not int or config['gpu'] not in (0, 1)):
        raise ValueError('Explicit authorized host and physical remote GPU required')
    if re.fullmatch(r'[0-9a-f]{64}', str(config.get('manifest_sha256', ''))) is None:
        raise ValueError('Pinned provisioned manifest SHA256 required')
    manifest = config.get('manifest_path')
    if (not isinstance(manifest, str) or not Path(manifest).is_relative_to(WORKSPACE)
            or '..' in Path(manifest).parts):
        raise ValueError('Remote manifest must be inside isolated worker workspace')
    if config.get('ai_credentials_forwarded') is not False:
        raise ValueError('Remote worker must receive no AI account credentials')
    for key in ('worker_sha256', 'launcher_sha256'):
        if re.fullmatch(r'[0-9a-f]{64}', str(config.get(key, ''))) is None:
            raise ValueError('Pinned tested worker and launch source required')
    for service in ('vla', 'sam3'):
        if re.fullmatch(r'[0-9a-f]{64}', str(config.get('checkpoint_sha256', {}).get(service, ''))) is None:
            raise ValueError('Exact original checkpoint digest required')
    return config


def ssh_command():
    return ['/usr/bin/ssh', '-F', '/dev/null', '-a', '-i', KEY, '-o', 'BatchMode=yes',
            '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'UserKnownHostsFile=' + KNOWN_HOSTS, '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']


def _ssh_environment():
    return {key: os.environ[key] for key in ('PATH', 'LANG', 'LC_ALL') if key in os.environ}


def _append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), **value)) + '\n')


def validate_ready(ready, config, service, lease_id):
    port = ready.get('port')
    sources = ready.get('source_sha256')
    if (ready.get('service') != service or ready.get('lease_id') != lease_id
            or ready.get('gpu') != config['gpu']
            or ready.get('manifest_sha256') != config['manifest_sha256']
            or ready.get('checkpoint_sha256') != config['checkpoint_sha256'][service]
            or type(port) is not int or not 0 < port < 65536
            or type(ready.get('pid')) is not int or ready['pid'] <= 0
            or not isinstance(sources, dict) or not sources
            or any(not isinstance(k, str) or re.fullmatch(r'[0-9a-f]{64}', str(v)) is None
                   for k, v in sources.items())):
        raise RuntimeError('Remote actual service/source/checkpoint/lease identity mismatch')
    return port


class RemoteServicePair:
    """One remote GPU lease, two serial original services, owned SSH tunnels."""

    def __init__(self, config, output, episode):
        self.config = dict(validate_configuration(config))
        if re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,199}', episode) is None:
            raise ValueError('Finite ordinary episode identifier required')
        self.output = Path(output)
        self.episode = episode
        self.process = None
        self.error = None
        self.messages = Queue()
        self.tunnels = []
        self.started = []
        self.closed = False
        self.lease_id = None
        self._stderr = None
        self._write_lock = threading.Lock()

    def _reader(self):
        try:
            while True:
                line = self.process.stdout.readline(262145)
                if not line:
                    if not self.closed:
                        raise RuntimeError('Remote GPU worker stream ended')
                    return
                if len(line) > 262144:
                    raise RuntimeError('Remote GPU worker protocol output exceeds bound')
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError('Remote worker message is not an object')
                _append(self.output / 'remote_gpu_worker.jsonl', dict(event='worker_message', message=item))
                if item.get('op') == 'error':
                    self.error = RuntimeError('Remote GPU worker reported failure: ' + str(item.get('error', 'unavailable'))[:1000])
                self.messages.put(item)
        except BaseException as exc:
            self.error = exc
            self.messages.put(dict(op='reader_failure'))

    def _receive(self, expected, request_id=None, timeout=300):
        deadline = time.monotonic() + timeout
        while True:
            if self.error is not None:
                raise RuntimeError('Remote GPU service delivery failed') from self.error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Remote GPU worker response timed out; no automatic retry')
            try:
                value = self.messages.get(timeout=min(remaining, .5))
            except Empty:
                if self.process.poll() is not None:
                    raise RuntimeError('Episode-owned remote worker exited')
                continue
            if value.get('op') != expected or (request_id is not None and value.get('request_id') != request_id):
                raise RuntimeError('Remote GPU worker response identity/order mismatch')
            return value

    def _send(self, value):
        with self._write_lock:
            self.process.stdin.write((json.dumps(value) + '\n').encode())
            self.process.stdin.flush()

    def _start_worker(self):
        if self.process is not None:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        c = self.config
        bootstrap = ('import hashlib,pathlib,runpy,sys;'
            'p=pathlib.Path(sys.argv[1]);q=pathlib.Path(sys.argv[3]);'
            'assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2],"Worker source changed";'
            'assert hashlib.sha256(q.read_bytes()).hexdigest()==sys.argv[4],"Launcher source changed";'
            'sys.argv=[str(p)]+sys.argv[5:];runpy.run_path(str(p),run_name="__main__")')
        command = [WORKSPACE + '/env/bin/python', '-u', '-c', bootstrap,
                   WORKSPACE + '/worker/service_worker.py', c['worker_sha256'],
                   WORKSPACE + '/worker/launch-service.sh', c['launcher_sha256'],
                   '--gpu', str(c['gpu']), '--episode', self.episode, '--manifest', c['manifest_path']]
        self._stderr = (self.output / 'remote_gpu_worker_stderr.log').open('ab')
        self.process = subprocess.Popen(ssh_command() + [USER + '@' + HOST, shlex.join(command)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                        env=_ssh_environment())
        threading.Thread(target=self._reader, daemon=True).start()
        value = self._receive('worker_ready')
        if (value.get('gpu') != c['gpu'] or value.get('manifest_sha256') != c['manifest_sha256']
                or not isinstance(value.get('lease_id'), str) or not value['lease_id']):
            raise RuntimeError('Remote worker identity or provisioned manifest changed')
        self.lease_id = value['lease_id']

    def start_service(self, service):
        if self.closed or service not in ('vla', 'sam3') or self.started != ([] if service == 'vla' else ['vla']):
            raise ValueError('Original VLA then SAM3 service order, once per episode, required')
        from rpent.utils.http_rpc import HttpRpcClient
        from rpent.utils.rpc import wait_for_ready
        from rpent.utils.daemon import ProcessDaemon, pick_free_port
        try:
            self._start_worker()
            request_id = uuid.uuid4().hex
            self._send(dict(op='start', service=service, request_id=request_id))
            ready = self._receive('ready', request_id)
            remote_port = validate_ready(ready, self.config, service, self.lease_id)
            local_port = pick_free_port()
            command = ssh_command() + ['-o', 'ExitOnForwardFailure=yes', '-N', '-L',
                f'127.0.0.1:{local_port}:127.0.0.1:{remote_port}', USER + '@' + HOST]
            tunnel = ProcessDaemon('remote_' + service + '_tunnel', command,
                log_path=str(self.output / (service + '-ssh-tunnel.log')), cwd=str(self.output), env=_ssh_environment())
            self.tunnels.append(tunnel)
            tunnel.start()
            url = f'http://127.0.0.1:{local_port}'
            client = HttpRpcClient(url)
            wait_for_ready(client, daemon=tunnel, timeout_s=30)
            if self.poll() is not None:
                raise RuntimeError('Remote worker exited while opening service tunnel')
            self.started.append(service)
            _append(self.output / 'remote_gpu_worker.jsonl', dict(event='service_tunnel_ready',
                service=service, host=HOST, remote_gpu=self.config['gpu'], remote_pid=ready['pid'],
                remote_port=remote_port, local_url=url, lease_id=self.lease_id))
            return client, self, url
        except BaseException:
            self.stop()
            raise

    def poll(self):
        if self.error is not None:
            return -1
        if self.process is not None and self.process.poll() is not None:
            return self.process.returncode
        for tunnel in self.tunnels:
            if tunnel.poll() is not None:
                return tunnel.poll()
        return None

    def stop(self, timeout=15):
        if self.closed:
            return
        self.closed = True
        process = self.process
        if process is not None:
            if process.poll() is None:
                try:
                    self._send(dict(op='stop', request_id=uuid.uuid4().hex))
                except (OSError, ValueError):
                    pass
                try:
                    process.stdin.close()
                except (OSError, ValueError):
                    pass
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
        for tunnel in reversed(self.tunnels):
            tunnel.stop(timeout=5)
        if self._stderr is not None:
            self._stderr.close()
        _append(self.output / 'remote_gpu_worker.jsonl', dict(event='owned_worker_closed',
            remote_gpu=self.config['gpu'], lease_id=self.lease_id, automatic_retry=False,
            robot_control_replay=False, remote_ai_credentials=False))
