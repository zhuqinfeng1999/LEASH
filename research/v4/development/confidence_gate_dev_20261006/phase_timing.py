"""Passive wall-clock spans. Nested spans are explicitly inclusive."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
import json
import time

_sink = ContextVar('formal_phase_sink', default=None)

@contextmanager
def phase(name, **metadata):
    sink = _sink.get()
    if sink is None:
        yield
        return
    start = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    error = None
    try:
        yield
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        row = dict(phase=name, started_utc=started,
                   ended_utc=datetime.now(timezone.utc).isoformat(),
                   wall_s=time.monotonic()-start, inclusive=True,
                   error_type=error, **metadata)
        with sink.open('a') as f: f.write(json.dumps(row, allow_nan=False)+'\n')

@contextmanager
def timing_scope(output):
    token = _sink.set(Path(output)/'phase_timings.jsonl')
    try: yield
    finally: _sink.reset(token)

def timed_toolkit(toolkit):
    original = toolkit.execute_tool
    def execute(name, arguments):
        with phase('tool', primitive=name):
            return original(name, arguments)
    toolkit.execute_tool = execute
    return toolkit
