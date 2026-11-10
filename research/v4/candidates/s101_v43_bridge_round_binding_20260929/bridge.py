"""Actual original S1 bridge: seal public history and consume typed recovery.

Only the existing history-append hook and boolean recovery certificate are
adapted. Persistent SDK, Deferred wrappers, order/finish/quotas and motion stay
in their original implementations. This slice is not a main/launch entry.
"""
from research.v4.leash_v43.run_manifest import source_custody_enabled
import ast
import copy
import inspect
import textwrap

from pathlib import Path
from research.v4.candidates.s1_user_revision_real_binding_20260928 import factory as parent
from research.v4.candidates.s101_v43_recovery_state_20260929.state import certified

RESOURCE_BOUNDARIES = frozenset({"round_call_quota", "episode_call_quota",
                               "planner_turn_quota", "planner_time_quota"})


def current_original_bridge():
    # Current 982 qualified source selection only; no Scene/main/preflight.
    module = parent.runner_module("S1-01")
    return module.optin_bridge.S1PersistentSession.original_constructor


def selected_bridge(consumer, *, original=None, seal_round=None):
    if source_custody_enabled():
        from .source import source_hashes
        source_hashes()
        qualified = current_original_bridge()
        original = qualified if original is None else original
    elif original is None:
        from research.v4.candidates.s1_multiround_driver_20260925.bridge import S1PersistentSession
        original = S1PersistentSession
        qualified = original
    else:
        qualified = original
    if source_custody_enabled() and original is not qualified:
        raise ValueError("exact current S1 bridge class required")
    # Own semantic recipes retain the source AST but their co_filename has
    # a compile suffix. Read the actual pinned source instead of pretending
    # inspect.getsource can recover a generated function from that filename.
    path = parent.ROOT / "research/v4/candidates/s1_multiround_driver_20260925/bridge.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "S1PersistentSession")
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_round")
    tree = ast.Module(body=[copy.deepcopy(method)], type_ignores=[])
    baseline = copy.deepcopy(tree) if source_custody_enabled() else None
    history_hooks = []
    certificates = []
    originals = {}
    class Bind(ast.NodeTransformer):
        def visit_Expr(self, node):
            node = self.generic_visit(node)
            if (isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == "self.history.append"):
                history_hooks.append(node)
                hook = ast.parse("_v43_seal(self, turn_input, result, seal)").body[0]
                return [node, hook]
            return node
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in {"no_motion", "ordinary"}:
                    originals[name] = copy.deepcopy(node)
                    certificates.append(name)
                    text = ("no_motion = _v43_recovery(self, recovery, decision.outcome)"
                            if name == "no_motion" else "ordinary = False")
                    return ast.parse(text).body[0]
            return self.generic_visit(node)
    tree = Bind().visit(tree)
    if len(history_hooks) != 1 or certificates != ["no_motion", "ordinary"]:
        raise RuntimeError("actual S1 history/recovery consumer seam changed")
    # Reversibility protects the original exact instruction/order, driver
    # finish, zero planner turns after ASK/REFUSE, quotas and recovery timing.
    if source_custody_enabled():
        reverted = copy.deepcopy(tree)
        class Undo(ast.NodeTransformer):
            def visit_Expr(self, node):
                if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "_v43_seal"):
                    return None
                return self.generic_visit(node)
            def visit_Assign(self, node):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name in originals:
                        return originals[name]
                return self.generic_visit(node)
        reverted = Undo().visit(reverted)
        if ast.dump(reverted, include_attributes=False) != ast.dump(baseline, include_attributes=False):
            raise RuntimeError("unexpected S1 bridge algorithm change")

    def seal(instance, turn_input, result, boundary):
        row = instance.history[-1]
        if row.index != turn_input.index or row.instruction != turn_input.instruction:
            raise RuntimeError("actual bridge's sealed row differs from trusted current round")
        resource_end = boundary if boundary in RESOURCE_BOUNDARIES else None
        if result.resource_end is not None and resource_end != result.resource_end:
            raise RuntimeError("actual session resource boundary differs from bridge seal")
        (seal_round or consumer.seal_round)(primitive_summaries=row.primitive_summaries,
                                          boundary=boundary, resource_end=resource_end)

    def recover(instance, record, task_outcome):
        return certified(record, task_outcome=task_outcome,
                         require_action_gate=instance.require_action_gate_for_recovery)

    namespace = {**original.run_round.__globals__, "_v43_seal": seal,
                 "_v43_recovery": recover}
    ast.fix_missing_locations(tree)
    exec(compile(tree, __file__ + "::original-bridge-two-seams", "exec"), namespace)
    selected = type("S101V43OriginalBridge", (original,),
                    {"run_round": namespace["run_round"], "__module__": __name__})
    selected.original_bridge = original
    selected.algorithm_inverse_ast_exact = source_custody_enabled()
    selected.actual_round_consumer = consumer
    return selected


class DriverObservationBoundary:
    """Permit the original driver's read-only input before task admission.

    Motion/finish continue through the same Consumer boundary. No capture,
    schema, observation content, planner turn or reset is introduced here.
    """
    def __init__(self, consumer):
        self.base = consumer.boundary
        self.consumer = consumer
        self.observation_base = (consumer.base if hasattr(consumer, "base")
                                 else consumer.boundary.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def execute_tool(self, name, arguments):
        if (not self.base.round_open and not self.observation_base.round_finished
                and name in self.base.observation_tools):
            return self.observation_base.execute_tool(name, arguments)
        return self.base.execute_tool(name, arguments)

    def get_tools_spec(self):
        return self.base.get_tools_spec()
