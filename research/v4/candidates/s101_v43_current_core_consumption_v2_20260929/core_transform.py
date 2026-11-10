"""Bind exact original core AST seams; leave B0's actual function untouched."""
from __future__ import annotations
from research.v4.leash_v43.run_manifest import source_custody_enabled

import ast
import copy
from types import FunctionType, ModuleType
from .runtime import CoreBindings

SOURCE = "research/v4/candidates/s1_multiround_driver_20260925/method_runner.py"


def selected_core(parent, *, policy, calibration, source_hashes, method="Ours", context_factory=None):
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    original = parent.optin_core.run_s1_method_core
    tree = ast.parse((root / SOURCE).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_s1_method_core")
    tree = ast.Module(body=[node], type_ignores=[])
    if source_custody_enabled():
        from research.v4.candidates.s101_v43_actual_main_scope_binding_v2_20260929 import main_transform as main
        source_hashes()
        original_node = copy.deepcopy(tree)
        compiled = compile(tree, str(root / SOURCE), "exec")
        original_code = next(c for c in compiled.co_consts if getattr(c, "co_name", None) == original.__name__)
        if main.old._code_identity(original_code) != main.old._code_identity(original.__code__):
            raise RuntimeError("actual current S101 core differs from original pinned AST")
    changes = []
    originals = {}

    def remember(old, new_text, label):
        replacement = ast.parse(new_text).body
        if source_custody_enabled():
            originals[label] = copy.deepcopy(old)
        for item in replacement:
            item._v43_seam = label
        changes.append(label)
        return replacement

    class Bind(ast.NodeTransformer):
        def visit_If(self, n):
            text = ast.unparse(n)
            if text.startswith("if method == 'Ours':\n    if reviewer is None"):
                return remember(n, "V43_CORE.require_interfaces(method, adapter, reviewer, workspace, locate_vla_target, recovery_on_owner, diagnostic_only)", "required_interfaces")
            if ast.unparse(n.test) == "method == 'Ours' and workspace is None":
                return remember(n, "pass", "own_provider")
            if ast.unparse(n.test) == "method == 'Ours'" and n.body and isinstance(n.body[0], ast.Assign) and ast.unparse(n.body[0].targets[0]) == "action":
                return remember(n, """consumer = V43_CORE.make_consumer(public_base, adapter, scene, sam3_client, reviewer, adapter.instruction, episode_id)
action = consumer.boundary
workspace = V43_CORE.consumer_workspace(consumer)
public_workspace = V43_CORE.consumer_provider(consumer)
method_base = V43_CORE.observe_boundary(consumer)
""", "actual_consumer")
            if ast.unparse(n.test) == "public_workspace is not None":
                return remember(n, "pass", "old_task_workspace_hook")
            if ast.unparse(n.test) == "action is not None":
                return remember(n, "pass", "old_action_task_hook")
            return self.generic_visit(n)

        def visit_FunctionDef(self, n):
            if n.name == "default_recover":
                return remember(n, """def default_recover():
    return consumer.recover_on_owner(scene=scene, initial_public_eef=initial_public_eef, initial_sequence=initial_sequence)
""", "actual_recovery")
            return self.generic_visit(n)

        def visit_Assign(self, n):
            target = ast.unparse(n.targets[0])
            value = ast.unparse(n.value)
            if target == "motion_reviewer":
                return remember(n, "motion_reviewer = None", "old_motion_reviewer")
            if target == "task_gate":
                return remember(n, "task_gate = None", "old_task_gate")
            if target == "fault_stop":
                return remember(n, ast.unparse(n) + "\npending_task = {'value': None}", "owner_task_state")
            if target == "callbacks":
                return remember(n, "callbacks = S1OwnerCallbacks(fault_stop, capture=lambda: V43_CORE.admit_on_owner(consumer, scene, pending_task), recover=default_recover, advance=advance_on_owner)", "actual_owner_capture")
            if target == "owner":
                return remember(n, ast.unparse(n) + "\nV43_CORE.bind_owner(owner, adapter)", "actual_owner_abort")
            if target == "task_adapter":
                return remember(n, "task_adapter = V43_CORE.task_adapter(consumer, pending_task, owner, callbacks)", "actual_task_adapter")
            if target == "action_system_failure":
                return remember(n, "action_system_failure = V43_CORE.action_fault(consumer)", "actual_action_fault")
            if target == "commit_system_failure":
                return remember(n, "commit_system_failure = V43_CORE.commit_fault(adapter)", "actual_commit_fault")
            return self.generic_visit(n)

        def visit_Call(self, n):
            n = self.generic_visit(n)
            if isinstance(n.func, ast.Name) and n.func.id == "run_s1_b0_candidate":
                label = "actual_round_helper_" + str(sum(x.startswith("actual_round_helper_") for x in changes) + 1)
                if source_custody_enabled():
                    originals[label] = copy.deepcopy(n)
                n.func = ast.parse("V43_CORE.run_rounds", mode="eval").body
                n.args.insert(0, ast.Name("consumer", ast.Load()))
                n._v43_expr = label
                changes.append(label)
            elif isinstance(n.func, ast.Name) and n.func.id == "CodexPersistentSession":
                label = "actual_persistent_policy"
                if source_custody_enabled():
                    originals[label] = copy.deepcopy(n)
                n.func = ast.parse("V43_CORE.session_constructor()", mode="eval").body
                n._v43_expr = label
                changes.append(label)
            return n

        def visit_Compare(self, n):
            n = self.generic_visit(n)
            if method != "Ours" and ast.unparse(n) == "method not in {'B0', 'Ours'}":
                return ast.parse("method != " + repr(method), mode="eval").body
            return n

        def visit_Dict(self, n):
            n = self.generic_visit(n)
            replacements = {
                "task_gate_logs": "V43_CORE.task_logs(consumer)",
                "action_gate_logs": "V43_CORE.action_logs(consumer)",
                "action_proposals": "V43_CORE.proposal_logs(consumer)",
                "public_perception": "V43_CORE.perception_logs(consumer)",
                "motion_name_bindings": "[]",
                "commit_gate_logs": "V43_CORE.commit_logs(adapter)",
                "recovery_gate_logs": "V43_CORE.recovery_report(consumer)",
            }
            for index, key in enumerate(n.keys):
                if isinstance(key, ast.Constant) and key.value in replacements:
                    label = "report_" + key.value
                    if source_custody_enabled():
                        originals[label] = copy.deepcopy(n.values[index])
                    n.values[index] = ast.parse(replacements[key.value], mode="eval").body
                    n.values[index]._v43_expr = label
                    changes.append(label)
            return n

    selected = Bind().visit(tree)
    expected = {"required_interfaces", "own_provider", "actual_consumer", "old_task_workspace_hook",
        "old_action_task_hook", "actual_recovery", "old_motion_reviewer", "old_task_gate",
        "owner_task_state", "actual_owner_capture", "actual_owner_abort", "actual_task_adapter",
        "actual_action_fault", "actual_commit_fault", "actual_round_helper_1", "actual_round_helper_2",
        "actual_persistent_policy", "report_task_gate_logs", "report_action_gate_logs",
        "report_action_proposals", "report_public_perception", "report_motion_name_bindings",
        "report_commit_gate_logs", "report_recovery_gate_logs"}
    if len(changes) != len(expected) or set(changes) != expected:
        raise RuntimeError("actual original core seams are missing, repeated or changed: " + repr(changes))

    class Undo(ast.NodeTransformer):
        def visit(self, n):
            if not isinstance(n, ast.AST):
                return super().visit(n)
            label = getattr(n, "_v43_seam", None)
            if label is not None:
                if label in restored:
                    return None
                restored.add(label)
                return copy.deepcopy(originals[label])
            label = getattr(n, "_v43_expr", None)
            if label is not None:
                return copy.deepcopy(originals[label])
            return super().visit(n)
    if source_custody_enabled():
        restored = set()
        inverse = Undo().visit(copy.deepcopy(selected))
        if ast.dump(inverse) != ast.dump(original_node):
            raise RuntimeError("inverse current core AST differs from original core algorithms")
    context = (context_factory or CoreBindings)(parent, policy=policy, calibration=calibration,
                                               source_hashes=source_hashes)
    value = ModuleType(parent.optin_core.__name__ + "_v43_current_core")
    value.__dict__.update(vars(parent.optin_core))
    value.V43_CORE = context
    value.ReviewerSession = __import__("research.v4.leash_v43.reviewer_transport", fromlist=["ReviewerSession"]).ReviewerSession
    ast.fix_missing_locations(selected)
    exec(compile(selected, __file__ + "::actual-current-core", "exec"), vars(value))
    # Original main calls this original-signature function. Its actual entry/end
    # source_hashes calls now consume the complete postcompile source guard.
    value.source_hashes = source_hashes
    if source_custody_enabled():
        value.selected_core_source = ast.unparse(selected)
        value.original_core_source = ast.unparse(original_node)
        value.source_proof = {"inverse_entire_core_exact": True, "actual_seams": changes}
    value.native_method_ready = value.launch_authorized = False
    return value
