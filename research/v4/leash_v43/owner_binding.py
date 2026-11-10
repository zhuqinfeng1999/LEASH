"""Opt-in cancellation hook on the existing scene-owner queue.

The original queue, execute_tool and run_planner stay unchanged. Ordinary
commit BLOCK cancels only its current primitive. Original owner.abort is a
terminal infrastructure stop and must also withhold a reviewer-pending command.
"""
from __future__ import annotations

from types import MethodType

from research.v4.bridge.thread_owner import OwnerThreadToolkit
from .dispatch import CommitDispatcher


def bind_owner_abort(owner, adapter, *, enabled=True):
    """Bind after placing V43ToolBoundary INSIDE the original owner toolkit.

    B0/B1 callers pass enabled=False and retain the identical original object
    and method. No default consumer or historical owner class is patched.
    """
    if type(enabled) is not bool:
        raise ValueError("explicit method cancellation binding required")
    if not enabled:
        return owner
    if (type(owner) is not OwnerThreadToolkit
            or getattr(getattr(owner.base, "adapter", None), "dispatcher", None) is not adapter.dispatcher
            or type(getattr(adapter, "dispatcher", None)) is not CommitDispatcher
            or getattr(owner.abort, "__func__", None) is not OwnerThreadToolkit.abort
            or owner._active or owner._aborted.is_set()):
        raise ValueError("fresh original owner / selected v4.3 adapter binding required")
    original_abort = owner.abort

    def abort(instance):
        # Record cancellation without the dispatch lock before draining callers.
        # A response arriving later must not consume the candidate or its tail.
        adapter.dispatcher.request_owner_abort("original simulator owner dispatcher stopped")
        return original_abort()

    owner.abort = MethodType(abort, owner)
    return owner
