from research.v4.leash_v43.run_manifest import source_custody_enabled
"""Opt-in actual E101 Ours main consumers, over its original source generator."""
import ast
from hashlib import sha256
import importlib
import json
from pathlib import Path
import textwrap
from types import FunctionType, ModuleType

from .capture import capture_parent, parent

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INSTANCE = "E1-01"
VERSION = "E101-own-visibleblade-ordinary-RPent-v43-actual-main-optin-v1"
RUNTIME_PACKAGE = "research.v4.candidates.e101_v43_consumer_runtime_20260929"
AUDIT = ROOT / "research/v4/candidates/e101_v43_actual_consumer_seam_readonly_20260929"
ARCHIVE = ROOT / "research/v4/candidates/e101_p401_v43_root_readonly_20260929/input_archive_receipt.json"
FILES = ("__init__.py", "capture.py", "prepare_parent.py", "binding.py", "entry.py", "check_source.py",
         "contract.json", "HANDOFF.md", "SOURCE_COMMANDS.md", "parent_capture_report.json", "parent_capture.raw.log")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def contract():
    return json.loads((HERE / "contract.json").read_text())


def archived_readonly_sources():
    """Retain old evidence membership, with Root's explicit exact snapshots."""
    freeze = AUDIT / "source_freeze.json"
    pins = dict(json.loads(freeze.read_text())["sha256"])
    archive = json.loads(ARCHIVE.read_text())["snapshots"]
    for name in ("research/v4/results/status.json", "research/v4/configs/run_manifest.json"):
        value = archive[name]
        if value["sha256"] != pins[name] or digest(ROOT / value["archived_path"]) != pins[name]:
            raise RuntimeError("exact historical read-only input archive differs")
        del pins[name]
        pins[value["archived_path"]] = value["sha256"]
    for name, expected in pins.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("immutable read-only source/evidence differs: " + name)
    for path in (freeze, AUDIT / "evidence_freeze.json", AUDIT / "HANDOFF.md", ARCHIVE,
                 ROOT / "research/v4/results/e101_v43_actual_consumer_seam_readonly_root_review_20260929.json"):
        pins[str(path.relative_to(ROOT))] = digest(path)
    return pins


def sources():
    expected = contract()
    receipt = expected.get("runtime_source_receipt")
    if not receipt:
        raise RuntimeError("Root own runtime SOURCE qualification has not been received")
    path = ROOT / receipt["path"]
    if digest(path) != receipt["sha256"]:
        raise RuntimeError("Root own runtime SOURCE receipt differs")
    runtime = importlib.import_module(RUNTIME_PACKAGE + ".runtime")
    pins = dict(parent.source_hashes())
    for mapping in (archived_readonly_sources(), runtime.source_hashes()):
        for name, value in mapping.items():
            if name in pins and pins[name] != value:
                raise RuntimeError("own parent/current runtime source conflict: " + name)
            pins[name] = value
    for path in (parent.HERE / "source_freeze.json", path,
                 *[HERE / name for name in FILES],
                 *sorted((HERE / "original_capture").glob("*"))):
        if path.is_file():
            pins[str(path.relative_to(ROOT))] = digest(path)
        else:
            raise RuntimeError("complete own source member missing: " + str(path))
    return dict(sorted(pins.items()))


def validate_source_packet(actual, expected):
    if type(actual) is not dict or type(expected) is not dict or actual != expected:
        raise RuntimeError("complete E101 parent/current source membership or bytes differs")
    return actual


def source_hashes(expected=None):
    frozen = json.loads((HERE / "source_freeze.json").read_text())
    if frozen["version"] != VERSION:
        raise ValueError("exact E101 source version required")
    return validate_source_packet(sources(), frozen["sha256"] if expected is None else expected)


def statements(source):
    return ast.parse(textwrap.dedent(source)).body


def one(nodes, predicate, label):
    found = [node for node in nodes if predicate(node)]
    if len(found) != 1:
        raise RuntimeError(f"actual E101 {label} missing/duplicated: {len(found)}")
    return found[0]


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                              for t in node.targets)


def transformed_tree(source):
    if source_custody_enabled() and sha256(source.encode()).hexdigest() != contract()["parent_module_source_sha256"]:
        raise RuntimeError("exact original own generated E101 module differs")
    tree = source if isinstance(source, ast.Module) else ast.parse(source)
    main = one(tree.body, lambda n: isinstance(n, ast.FunctionDef) and n.name == "main", "main")
    admit = one(main.body, lambda n: assigned(n, "source"), "actual hashes consumer")
    index = main.body.index(admit) + 1
    main.body[index:index] = statements("""
        if a.method != SELECTED_METHOD:
            raise ValueError('selected own E101 method differs')
        if not a.preflight_only:
            if source_custody_enabled() and not SOURCE_NATIVE_LAUNCH_APPROVED:
                raise RuntimeError('E101 source preparation is not native or model launch approval')
            V43.admit_run(a.method, V43_POLICY, V43_CALIBRATION)
    """)
    initial = one(main.body, lambda n: assigned(n, "scene"), "resource initialization")
    main.body.insert(main.body.index(initial) + 1, statements("consumer = None")[0])
    block = one(main.body, lambda n: isinstance(n, ast.Try), "runtime try/finally")
    reviewer = one(block.body, lambda n: isinstance(n, ast.If) and any(assigned(x, "config") for x in n.body),
                   "Ours reviewer constructor")
    reviewer.body = statements("""
        reviewer, adapter = V43.make_reviewer_adapter(
            scene, instruction=INSTRUCTION, model=reviewer_model,
            episode=output.name, output=output, policy=V43_POLICY,
            calibration=V43_CALIBRATION, on_step=on_step)
    """)
    # Keep this original reviewer.orelse (B0 adapter construction) intact.
    task = one(block.body, lambda n: isinstance(n, ast.If) and any(assigned(x, "task_gate") for x in n.body),
               "Ours actual task/tool/Owner branch")
    task.body = statements("""
        stage = time.perf_counter()
        consumer = V43.Consumer(
            public_base, adapter, components=CONFIGURED, scene=scene,
            sam_client=sam_client, reviewer=reviewer, policy=V43_POLICY,
            instruction=INSTRUCTION, episode=output.name,
            primitives=base.original._primitives, vla_chunk_controls=None)
        consumer.boundary.begin_round(INSTRUCTION, round_id=1)
        report['task_gate'] = consumer.task_report()
        report['stage_seconds']['task_reviewer'] = round(time.perf_counter() - stage, 3)
        perception_log = consumer.provider.log
        if consumer.task_fault():
            raise RuntimeError('actual E101 task reviewer failed; fail-closed finish is not safety success')
        if consumer.boundary.task_approved and consumer.boundary.round_open:
            pick_toolkit = PlainRPentToolkit(
                FaultStopToolkit(consumer.boundary, adapter), public_packet=scene.capture_agent)
            bridge = OwnerThreadToolkit(pick_toolkit, request_timeout_s=1800)
            V43.bind_owner_abort(bridge, adapter)
            stage = time.perf_counter()
            run = bridge.run_planner(lambda: V43.run_planner(
                run_single_round_b0, policy=V43_POLICY,
                instruction=INSTRUCTION, toolkit=bridge,
                output_dir=output / 'rpent', timeout_s=1500, max_turns=80,
                model_id=a.planner_model, reasoning_effort='high'))
            report['stage_seconds']['planner'] = round(time.perf_counter() - stage, 3)
            report['terminal'] = run.terminal
            report['planner_error'] = run.planner_result.error
            report['planner_stats'] = run.planner_result.stats
            report['usage_tokens'] = usage_from_stats(run.planner_result.stats)
            report['tool_calls'] = run.calls
            report['primitive_calls'] = bridge.episode_calls
        else:
            report['terminal'] = base.finish_result
            report['tool_calls'] = list(base.calls)
            report['primitive_calls'] = base.episode_calls
    """)
    # task.orelse is actual original ordinary B0/Owner/planner branch.
    gated = one(block.finalbody, lambda n: isinstance(n, ast.If) and ast.unparse(n.test) == "gated is not None",
                "old action report")
    gated.test = ast.parse("consumer is not None", mode="eval").body
    gated.body = statements("""
        report['action_gate'] = consumer.action_report()
        report['public_capability_uncertainties'] = report['action_gate']['public_capability_uncertainties']
    """)
    journal = one(block.finalbody, lambda n: isinstance(n, ast.If)
                  and "hasattr(adapter, 'commit_gate')" in ast.unparse(n.test), "old commit journal")
    block.finalbody.insert(block.finalbody.index(journal), statements("""
        if consumer is not None:
            report['commit_gate'] = V43.commit_report(adapter)
    """)[0])
    # Old journal remains only for its actual legacy commit_gate-bearing adapter.
    return ast.fix_missing_locations(tree)


def selected(instance=INSTANCE, *, method="Ours", version=VERSION, policy=None, calibration=None):
    if instance != INSTANCE or method not in {"B0", "Ours"}:
        raise ValueError("exact own E101 B0/Ours selection required")
    if version is None:
        if policy is not None or calibration is not None:
            raise ValueError("original no-version selection has no v43 overrides")
        return parent.runner_module(INSTANCE)
    if version != VERSION or (method == "B0" and (policy is not None or calibration is not None)):
        raise ValueError("exact own version and method policy scope required")
    admitted = source_hashes()
    original, source = capture_parent()
    runtime = importlib.import_module(RUNTIME_PACKAGE + ".runtime")
    namespace = runtime.namespace()
    tree = transformed_tree(source)
    code = compile(tree, str(HERE / "selected_module.py.txt"), "exec")
    mains = [x for x in code.co_consts if isinstance(x, type(code)) and x.co_name == "main"]
    if len(mains) != 1:
        raise RuntimeError("actual whole transformed module emitted a nonunique main")
    module = ModuleType(__name__ + "._selected_" + method)
    module.__dict__.update(original.__dict__)
    module.__dict__.update(V43=namespace, V43_POLICY=policy, V43_CALIBRATION=calibration,
                           SELECTED_METHOD=method, SOURCE_NATIVE_LAUNCH_APPROVED=False)
    module.main = FunctionType(mains[0], module.__dict__, original.main.__name__,
                               original.main.__defaults__, original.main.__closure__)
    module.main.__kwdefaults__ = original.main.__kwdefaults__
    module.hashes = source_hashes  # The actual main source=hashes() consumes this after full compilation.
    module.PROTOCOL = VERSION
    module.selection_tree = tree
    module.original_tree = ast.parse(source)
    module.original_main = original.main
    module.original_module = original
    module.capture_receipt = original.capture_receipt
    module.source_admission = admitted
    module.native_method_ready = module.launch_authorized = False
    module.source_version = VERSION
    module.consumer_selection = dict(instance=instance, method=method, policy_default=None,
        calibration_default=None, grasp_assumption="unverified", vla_chunk_controls=None,
        native_ready=False, launch_authorized=False)
    if (module.main.__globals__ is not module.__dict__ or module.main.__globals__["hashes"] is not source_hashes
            or module.main.__globals__["V43"].Consumer is not runtime.Consumer
            or module.Scene is not module.CONFIGURED.Scene or module.Scene is not original.Scene
            or module.Evaluator is not original.Evaluator or module.args is not original.args
            or module.PlainRPentToolkit is not original.PlainRPentToolkit
            or module.OwnerThreadToolkit is not original.OwnerThreadToolkit
            or module.R2V42Adapter is not original.R2V42Adapter):
        raise RuntimeError("actual selected main/components/guard globals differs")
    source_hashes()
    return module
