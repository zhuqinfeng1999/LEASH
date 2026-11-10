"""Episode-owned RPent startup with child-bound, kernel-assigned HTTP ports.

No model import, global environment mutation, inference, or shared service.
The child must be invoked with --host 127.0.0.1 --port 0. Existing RPent
RpcFacade prints its actual endpoint only after model load and atomic bind.
"""
from pathlib import Path
import re
import time

_PREFIX = b'RPC server listening on '
_ENDPOINT = re.compile(rb'http://127\.0\.0\.1:([0-9]{1,5})')
_MAX_LINE_BYTES = 65536


def _announced_endpoint(line):
    """Only the child's exact, unprefixed, loopback announcement is authority."""
    line = line.rstrip(b'\r')
    if not line.startswith(_PREFIX):
        return None
    endpoint = line[len(_PREFIX):]
    match = _ENDPOINT.fullmatch(endpoint)
    if match is None or not 1 <= int(match.group(1)) <= 65535:
        raise ValueError('Invalid episode-owned local RPC endpoint announcement')
    return endpoint.decode('ascii')


def start_owned_auto_port(daemon, log_path, client_factory, wait_for_ready,
                          *, timeout_s=300., poll_interval_s=.05,
                          clock=time.monotonic, sleep=time.sleep):
    """Start once, accept only newly appended endpoint, health-check, own cleanup.

    daemon, environment, cwd, checkpoint, argv and outputs remain owned by the
    calling episode. The existing --parent-watch lifecycle remains unchanged.
    A single deadline includes loading, endpoint discovery and health readiness.
    """
    if timeout_s <= 0 or poll_interval_s <= 0:
        raise ValueError('Startup limits must be positive')
    path = Path(log_path)
    offset = path.stat().st_size if path.exists() else 0
    deadline = clock() + timeout_s
    pending = b''
    try:
        daemon.start()
        while True:
            rc = daemon.poll()
            if rc is not None:
                raise RuntimeError(f'{daemon.name} exited with code {rc} before ready endpoint')
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError('Episode-owned RPC endpoint was not announced before startup deadline')
            if path.exists():
                if path.stat().st_size < offset:
                    raise RuntimeError('Episode-owned service startup log was truncated')
                with path.open('rb') as stream:
                    stream.seek(offset)
                    chunk = stream.read(_MAX_LINE_BYTES)
                offset += len(chunk)
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    if len(line) > _MAX_LINE_BYTES:
                        raise ValueError('Service startup log line exceeds bounded reader')
                    endpoint = _announced_endpoint(line)
                    if endpoint is None:
                        continue
                    remaining = deadline - clock()
                    if remaining <= 0:
                        raise TimeoutError('RPC endpoint became available after startup deadline')
                    client = client_factory(endpoint)
                    wait_for_ready(client, daemon=daemon, timeout_s=remaining)
                    if daemon.poll() is not None:
                        raise RuntimeError('Episode-owned RPC service exited after health readiness')
                    return client, daemon, endpoint
                if len(pending) > _MAX_LINE_BYTES:
                    raise ValueError('Service startup log line exceeds bounded reader')
                if len(chunk) == _MAX_LINE_BYTES:
                    continue
            sleep(min(poll_interval_s, max(0., deadline-clock())))
    except BaseException:
        daemon.stop()
        raise
