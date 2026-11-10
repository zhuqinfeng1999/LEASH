from research.v4.leash_v43.run_manifest import source_custody_enabled
"""One actual original whole-main compiler, six Ours call-site replacements."""
import ast
from hashlib import sha256
import importlib
import json
from pathlib import Path
from types import FunctionType, ModuleType

from .source import HERE, ROOT, VERSION, admit, source_text

ORIGINAL = "research.v4.candidates.r3_v42_visibility_20260926.run_r3"
RUNNER_PATH = "research/v4/candidates/r3_v42_visibility_20260926/run_r3.py"
FACTORY = "research.v4.candidates.r301_current_u_front_ordinary_method_factory_source_prepare_20260929.binding"
CONSUMER = "research.v4.candidates.r301_v43_task_lease_consumer_prepare_20260929.runtime"


def one(nodes, predicate, label):
    rows = [node for node in nodes if predicate(node)]
    if len(rows) != 1:
        raise RuntimeError(f"original R301 {label} count differs: {len(rows)}")
    return rows[0]


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                              for target in node.targets)


def statements(text):
    return ast.parse(text).body


def rewrite(text):
    contract = json.loads((HERE / "contract.json").read_text()) if source_custody_enabled() else None
    if source_custody_enabled() and sha256(text.encode()).hexdigest() != contract["original_runner_sha256"]:
        raise RuntimeError("exact original own R301 runner source differs")
    tree = text if isinstance(text, ast.Module) else ast.parse(text)
    main = one(tree.body, lambda n: isinstance(n, ast.FunctionDef) and n.name == "main", "main")
    imports = one(tree.body, lambda n: isinstance(n, ast.ImportFrom) and n.module == "method_entry", "entry import")
    tree.body[tree.body.index(imports)] = statements(
        "INSTANCE = OWN_ENTRY.INSTANCE\n_base_toolkit = OWN_ENTRY._base_toolkit\nopen_native = OWN_ENTRY.open_native")[0:3]
    # Flatten the one import replacement. No selected legacy gate/door consumer remains.
    tree.body = [item for node in tree.body for item in (node if isinstance(node, list) else [node])]
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom)
                 and node.module in {"research.v4.candidates.leash_reviewer_transport_20260924.transport",
                                     "research.v4.candidates.leash_commit_gate_20260924.preflight_gates",
                                     "research.v4.accelerated.public_geometry"})]
    hashes = one(tree.body, lambda n: isinstance(n, ast.FunctionDef) and n.name == "hashes", "hashes")
    hashes.body = statements("return OWN_HASHES()")
    admission = one(main.body, lambda n: assigned(n, "source"), "source admission")
    main.body[main.body.index(admission) + 1:main.body.index(admission) + 1] = statements(
        "if a.method != SELECTED_METHOD:\n    raise ValueError('selected own R301 method differs')\n"
        "if not a.preflight_only:\n    V43.admit_run(a.method, V43_POLICY, V43_CALIBRATION)")
    initialize = one(main.body, lambda n: assigned(n, "entry"), "local initialization")
    main.body.insert(main.body.index(initialize) + 1, statements("consumer = None")[0])
    block = one(main.body, lambda n: isinstance(n, ast.Try), "runtime try")
    old_reviewer = one(block.body, lambda n: isinstance(n, ast.If) and any(
        assigned(x, "reviewer") for x in n.body), "legacy reviewer constructor")
    block.body.remove(old_reviewer)
    entry = one(block.body, lambda n: assigned(n, "entry"), "original open_native call")
    call = one(ast.walk(entry), lambda n: isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "open_native", "actual shared Scene constructor")
    method = one(call.keywords, lambda n: n.arg == "method", "entry construction mode")
    method.value = ast.Constant("B0")
    reviewer = one(call.keywords, lambda n: n.arg == "reviewer", "entry reviewer argument")
    reviewer.value = ast.Constant(None)
    clients = one(block.body, lambda n: isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Tuple) and any(isinstance(x, ast.Name) and x.id == "vla_client" for x in t.elts)
        for t in n.targets), "original external clients")
    block.body[block.body.index(clients) + 1:block.body.index(clients) + 1] = statements(
        "if a.method == 'Ours':\n"
        "    def v43_private_sink(_scene):\n"
        "        state = entry.evaluator.observe(scene)\n"
        "        on_step(state)\n"
        "    reviewer, entry.adapter = V43.make_reviewer_adapter(\n"
        "        scene, instruction=scene.instance.instruction, model='gpt-5.5',\n"
        "        episode=output.name, output=output, policy=V43_POLICY,\n"
        "        calibration=V43_CALIBRATION, on_step=v43_private_sink)")
    task = one(block.body, lambda n: isinstance(n, ast.If) and any(
        isinstance(x, ast.Assign) and isinstance(x.value, ast.Call)
        and isinstance(x.value.func, ast.Name) and x.value.func.id == "begin_ours_task_gate" for x in n.body),
        "legacy task/action branch")
    old_b0 = ast.dump(ast.Module(body=task.orelse, type_ignores=[]), include_attributes=False) if source_custody_enabled() else None
    task.body = statements(
        "stage = time.perf_counter()\n"
        "consumer = V43.Consumer(\n"
        "    base, entry.adapter, components=CONFIGURED, scene=scene, sam_client=sam_client,\n"
        "    reviewer=reviewer, policy=V43_POLICY, instruction=scene.instance.instruction,\n"
        "    episode=output.name, primitives=base.original._primitives, vla_chunk_controls=None)\n"
        "gated = consumer.boundary\n"
        "gated.begin_round(scene.instance.instruction, round_id=1)\n"
        "report['task_gate'] = consumer.task_report()\n"
        "report['stage_seconds']['task_reviewer'] = round(time.perf_counter() - stage, 3)\n"
        "if consumer.task_fault():\n    raise RuntimeError('actual own R301 task reviewer failed')\n"
        "if gated.task_approved and gated.round_open:\n"
        "    bridge = OwnerThreadToolkit(FaultStopToolkit(gated, entry.adapter), request_timeout_s=1800)\n"
        "    V43.bind_owner_abort(bridge, entry.adapter)\n"
        "else:\n"
        "    report['terminal'] = base.finish_result\n"
        "    report['tool_calls'] = list(base.calls)\n"
        "    report['primitive_calls'] = base.episode_calls")
    if source_custody_enabled() and ast.dump(ast.Module(body=task.orelse, type_ignores=[]), include_attributes=False) != old_b0:
        raise RuntimeError("original registry-free B0 Owner branch changed")
    proceed = one(block.body, lambda n: isinstance(n, ast.If) and ast.unparse(n.test) == "bridge is not None",
                  "original planner loop")
    old_run = one(proceed.body, lambda n: assigned(n, "run"), "planner call")
    ours_run = ast.parse(ast.unparse(old_run)).body[0]
    planner = one(ast.walk(ours_run), lambda n: isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == "run_single_round_b0", "trusted planner constructor")
    planner.func = ast.Attribute(ast.Name("V43", ast.Load()), "run_planner", ast.Load())
    planner.args.insert(0, ast.Name("run_single_round_b0", ast.Load()))
    planner.keywords.append(ast.keyword("policy", ast.Name("V43_POLICY", ast.Load())))
    proceed.body[proceed.body.index(old_run)] = ast.If(
        test=ast.parse("a.method == 'Ours'", mode="eval").body, body=[ours_run], orelse=[old_run])
    action = one(block.finalbody, lambda n: isinstance(n, ast.If) and ast.unparse(n.test) == "gated is not None",
                 "action report")
    action.test = ast.parse("consumer is not None", mode="eval").body
    action.body = statements("report['action_gate'] = consumer.action_report()\n"
                             "report['public_perception'] = consumer.provider.log")
    entry_final = one(block.finalbody, lambda n: isinstance(n, ast.If) and ast.unparse(n.test) == "entry is not None",
                      "private final evidence")
    commit = one(entry_final.body, lambda n: isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Subscript) and ast.unparse(t) == "report['commit_gate']" for t in n.targets), "commit report")
    entry_final.body[entry_final.body.index(commit)] = ast.If(
        test=ast.parse("consumer is not None", mode="eval").body,
        body=statements("report['commit_gate'] = V43.commit_report(entry.adapter)"), orelse=[commit])
    fault = one(block.finalbody, lambda n: assigned(n, "fault"), "final fault classification")
    fault.value = ast.parse(
        "bool(report['error'] or report.get('observation_fault') or report['source_changes_after'] or "
        "V43.structured_fault(report))", mode="eval").body
    # Hazard priority, evaluator.finish, reason-judge-pending, budgets, services,
    # complete ordinary B0 planner call/Owner/observations remain original AST.
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n", old_b0) if source_custody_enabled() else (ast.fix_missing_locations(tree), None)


def structured_fault(report):
    rows = (report.get("action_gate") or {}).get("logs", ())
    rows = [*rows, *(report.get("commit_gate") or {}).get("logs", ())]
    return any(isinstance(row, dict) and row.get("system_failure") is True for row in rows)


def current_qualification(*, require_native=False):
    admit()
    if require_native:
        raise RuntimeError("own R301 actual Consumer/public/Owner/REAL I/O/native prerequisites remain pending")
    return dict(source_prepared=True, source_qualified=False, runtime_qualified=False,
                native_method_ready=False, launch_authorized=False, default_adopted=False)


def selected(instance="R3-01", *, method="Ours", version=None, policy=None, calibration=None,
             root_carrier_approved=False, provenance_directory=None):
    if version is None:
        return importlib.import_module(ORIGINAL)
    if instance != "R3-01" or method not in {"B0", "Ours"} or version != VERSION:
        raise ValueError("exact own R301 B0/Ours/version required")
    if method == "B0" and (policy is not None or calibration is not None):
        raise ValueError("original B0 receives no safety policy or calibration")
    pins = admit()
    out = Path(provenance_directory).resolve()
    if out.parent != (HERE / "root_source_outputs").resolve() or out.exists():
        raise ValueError("fresh own main Source provenance direct child required")
    factory_module = importlib.import_module(FACTORY)
    factory_out = factory_module.HERE / "root_source_outputs" / out.name
    entry = factory_module.make_method_entry(version=factory_module.VERSION,
        root_carrier_approved=root_carrier_approved, provenance_directory=factory_out)
    original = importlib.import_module(ORIGINAL)
    own = importlib.import_module(CONSUMER)
    context = own.prepared_context()
    # Rebind only guard suppliers. The actual provider/parser/constructor code,
    # globals and closures are those qualified in the independent2186 packet.
    context["provider_context"]["provider_namespace"].update(admit=admit, source_text=source_text)
    context["provider_context"]["sam_namespace"]["source_text"] = source_text
    old_capture = context["provider_context"]["guarded_capture"]
    capture = FunctionType(old_capture.__code__, dict(old_capture.__globals__, admit=admit),
                           old_capture.__name__, old_capture.__defaults__, old_capture.__closure__)
    context["provider_context"]["provider_namespace"]["_remember_capture"] = capture
    context["provider_context"]["guarded_capture"] = capture
    v43 = context["namespace"]
    v43.source_hashes = admit
    v43.structured_fault = structured_fault
    text, b0_ast = rewrite(source_text(RUNNER_PATH))
    module = ModuleType(__name__ + "._selected_" + method)
    module.__file__ = str(ROOT / RUNNER_PATH) + "::R301-v43-current-U-main"
    module.__package__ = original.__package__
    module.__dict__.update(OWN_ENTRY=entry, V43=v43, CONFIGURED=context["components"],
        OWN_HASHES=admit, SELECTED_METHOD=method, V43_POLICY=policy, V43_CALIBRATION=calibration)
    exec(compile(text, module.__file__, "exec"), module.__dict__)
    # Preserve the actual original parser CodeType, but globals belong to this
    # selected module. No parse/main invocation occurs during selection.
    parser = original.parse_args
    module.parse_args = FunctionType(parser.__code__, module.__dict__, parser.__name__,
                                    parser.__defaults__, parser.__closure__)
    module.parse_args.__kwdefaults__ = parser.__kwdefaults__
    module.OWN_HASHES = admit  # Actual hashes()/main consumes AFTER compilation.
    module.PROTOCOL = VERSION
    module.original_runner = original
    module.original_run_source = source_text(RUNNER_PATH)
    module.original_entry = entry
    module.consumer_context = context
    module.selection_text = text
    module.selection_b0_ast = b0_ast
    module.source_admission = pins
    module.selection_metadata = dict(instance=instance, method=method, version=version,
        factory_version=factory_module.VERSION, same_scene_B0_Ours=True,
        policy_default=None, calibration_default=None, grasp_assumption="unverified",
        vla_chunk_controls=None, source_qualified=False, runtime_qualified=False,
        native_ready=False, launch_authorized=False, default_adopted=False)
    module.method_native_ready = module.native_method_ready = module.launch_authorized = False
    module.current_qualification = current_qualification
    # Same existing synchronous private sink scope, both methods. Only its
    # immutable-source admission supplier is rebound to this complete union.
    scope_module = importlib.import_module("research.v4.candidates.r3_current_real_private_scope_20260928.hooks")
    invoke = scope_module.invoke_main
    selected_invoke = FunctionType(invoke.__code__, dict(invoke.__globals__, admit=admit),
                                   invoke.__name__, invoke.__defaults__, invoke.__closure__)
    selected_invoke.__kwdefaults__ = invoke.__kwdefaults__
    actual_main = module.main

    def scoped_main():
        return selected_invoke(actual_main)

    module.original_main = actual_main
    module.main = scoped_main
    module.main.launch_authorized = module.main.native_method_ready = False
    module.original_private_invocation = invoke
    module.selected_private_invocation = selected_invoke
    out.mkdir(parents=True, exist_ok=False)
    (out / "original_runner.py.txt").write_text(source_text(RUNNER_PATH))
    (out / "selected_runner.py.txt").write_text(text)
    module.source_admission = admit()
    return module
