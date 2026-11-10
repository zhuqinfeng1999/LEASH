"""Ordinary executed-call ranges for offline labels; no gate/robot changes.

Install on the actual episode's ConfidenceToolBoundary before its Owner proxy
is created. The row is the existing scripted_calls schema, also usable for
natural calls. There is one original execute_tool call and no extra capture.
"""
from contextlib import contextmanager

from .evidence import plain
from .runtime import BUDGETED_ACTIONS


@contextmanager
def primitive_call_log_scope(boundary, emit):
    if getattr(boundary, '_primitive_call_log_active', False):
        raise RuntimeError('This episode already has an ordinary primitive-call logger')
    previous = boundary.__dict__.get('execute_tool')
    had_instance_method = 'execute_tool' in boundary.__dict__
    original = boundary.execute_tool
    adapter, core = boundary.adapter, boundary.core
    boundary._primitive_call_log_active = True

    def execute(name, arguments):
        if name not in BUDGETED_ACTIONS:
            return original(name, arguments)
        before = adapter.dispatcher.sequence
        start = len(core.records)
        original_error = None
        returned_normally = False
        try:
            result = original(name, arguments)
            returned_normally = True
            return result
        except BaseException as exc:
            original_error = exc
            raise
        finally:
            # Only normal public call arguments and existing execution counters.
            row = dict(name=name, arguments=plain(arguments),
                sequence_before=before, sequence_after=adapter.dispatcher.sequence,
                gate_record_start=start, gate_record_end=len(core.records),
                returned_normally=returned_normally,
                error_type=type(original_error).__name__ if original_error is not None else None)
            try:
                emit(row)
            except Exception as exc:
                if original_error is None:
                    raise
                # A secondary log failure must not overwrite the original error.
                original_error.add_note(f'Primitive-call log also failed: {type(exc).__name__}: {exc}')

    boundary.execute_tool = execute
    try:
        yield boundary
    finally:
        if had_instance_method:
            boundary.execute_tool = previous
        else:
            del boundary.execute_tool
        del boundary._primitive_call_log_active
