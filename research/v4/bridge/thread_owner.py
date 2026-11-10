"""Execute RPent MCP tools on the thread that created the MuJoCo EGL scene.

CodexPlanner's in-process MCP server calls Toolkit.execute_tool from an
asyncio executor thread. EGL contexts are thread-affine, so the simulator must
stay on its original owner thread. This bridge runs the planner in a separate
thread and synchronously marshals its tool calls to the owner. No camera,
primitive, evaluator, or action semantics are changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import queue
import threading
from typing import Any, Callable

from research.v3.bridge.b0 import _tool_result


@dataclass
class _Request:
    name: str
    arguments: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


class OwnerThreadToolkit:
    """RPent Toolkit proxy whose commands always run on its creator thread."""

    def __init__(self, base: Any, *, request_timeout_s: float = 1800) -> None:
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        self.base = base
        self.owner_thread_id = threading.get_ident()
        self.request_timeout_s = float(request_timeout_s)
        self._requests: queue.Queue[_Request] = queue.Queue()
        self._aborted = threading.Event()
        self._active = False
        self.dispatch_fault: str | None = None
        self._capacity_inflight = False
        self._capacity_completed = 0
        self._capacity_last_tool = None
        self._capacity_execution_exception = False

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_tools_spec(self):
        return self.base.get_tools_spec()

    def _error_result(self, name: str):
        return _tool_result(name, {"error": "simulator owner dispatcher stopped"})

    def execute_tool(self, name: str, input_dict: dict[str, Any]):
        if self._aborted.is_set():
            return self._error_result(name)
        if threading.get_ident() == self.owner_thread_id:
            return self.base.execute_tool(name, input_dict)
        if not self._active:
            return self._error_result(name)
        request = _Request(name, dict(input_dict))
        self._requests.put(request)
        if not request.done.wait(self.request_timeout_s):
            self.dispatch_fault = "simulator owner dispatcher timed out"
            adapter = getattr(self.base, "adapter", None)
            if adapter is not None:
                adapter._latch("owner_dispatch", TimeoutError(self.dispatch_fault))
            self.abort()
            return _tool_result(name, {"error": self.dispatch_fault})
        if request.error is not None:
            raise request.error
        return request.result

    def abort(self) -> None:
        """Release queued callers without executing any more simulator tools."""
        self._aborted.set()
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                break
            request.result = self._error_result(request.name)
            request.done.set()

    def run_planner(self, callback: Callable[[], Any]):
        """Pump MCP tool calls on the scene owner while planner waits elsewhere."""
        if threading.get_ident() != self.owner_thread_id:
            raise RuntimeError("planner pump must run on the simulator owner thread")
        if self._active or self._aborted.is_set():
            raise RuntimeError("dispatcher is already active or stopped")
        self._active = True
        outcome: dict[str, Any] = {}

        def worker():
            try:
                from contextlib import nullcontext
                from research.v4.development.g2_candidate_evidence_dev_20261005.capacity_runtime import enabled,owner_scope
                with owner_scope(self) if enabled() else nullcontext():
                    outcome["result"] = callback()
            except BaseException as exc:
                outcome["error"] = exc

        planner = threading.Thread(target=worker, name="v4-b0-planner", daemon=True)
        planner.start()
        try:
            while planner.is_alive() or not self._requests.empty():
                try:
                    request = self._requests.get(timeout=.1)
                except queue.Empty:
                    continue
                if self._aborted.is_set():
                    request.result = self._error_result(request.name)
                else:
                    try:
                        self._capacity_inflight = True
                        request.result = self.base.execute_tool(
                            request.name, request.arguments)
                        self._capacity_completed += 1
                        self._capacity_last_tool = request.name
                    except BaseException as exc:
                        request.error = exc
                        self._capacity_execution_exception = True
                        from research.v4.leash_v43.qa import is_episode_wall_stop
                        episode_resource_stop = (
                            getattr(exc, 'episode_resource_stop', False) is True
                            and getattr(exc, 'resource', None) in ('wall_time', 'output_bytes'))
                        if episode_resource_stop:
                            # Release the MCP request with a normal tool result;
                            # the trusted outer runner receives the original
                            # typed quota boundary directly, never an ASGI crash.
                            request.error = None
                            request.result = _tool_result(request.name, dict(
                                error='episode resource budget exhausted',
                                resource=exc.resource, reason=str(exc)))
                        if episode_resource_stop or isinstance(exc, (KeyboardInterrupt, SystemExit)) or (
                            isinstance(exc, TimeoutError) and is_episode_wall_stop(
                                f'TimeoutError: {exc}', os.environ.get('SAFEVLA_LEASH_INSTANCE'))
                        ):
                            request.done.set()
                            raise
                self._capacity_inflight = False
                request.done.set()
            planner.join()
        except BaseException:
            self.abort()
            planner.join(timeout=5)
            raise
        finally:
            self._active = False
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("result")

    def cancel_active_and_wait(self):
        return self.base.cancel_active_and_wait()

    def close(self):
        if threading.get_ident() != self.owner_thread_id:
            raise RuntimeError("simulator toolkit must close on owner thread")
        self.abort()
        return self.base.close()
