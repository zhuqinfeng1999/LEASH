"""Exact own P102 call-site rewrite; old source and defaults remain intact."""
from __future__ import annotations

import ast
from research.v4.leash_v43.run_manifest import source_custody_enabled
from hashlib import sha256
import json
from pathlib import Path
import sys
import textwrap

from research.v4.candidates.p1_02_unknown_tilt_optin_20260927 import binding as parent
from research.v4.candidates.e1_plain_config_dispatch_20260927.factory import compiled
from . import runtime

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VERSION = "P102-own-v43-actual-consumer-optin-v1"
PARENT_TEXT = "21527a5e3c589bdb4112cefc987d2ff14ee9b6b78feca5574d814d17080385c1"
EXTRA = (
    "AGENTS.md", "research/v4/docs/leash_v43_revised_20260928.txt",
    "research/v4/leash_v43/__init__.py", "research/v4/leash_v43/adapter.py",
    "research/v4/leash_v43/registry.py", "research/v4/leash_v43/dispatch.py",
    "research/v4/leash_v43/tool_boundary.py", "research/v4/leash_v43/method_inputs.py",
    "research/v4/leash_v43/reviewer_transport.py", "research/v4/leash_v43/owner_binding.py",
    "research/v4/candidates/p1_02_unknown_tilt_optin_20260927/source_freeze.json",
)


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def sources():
    result = dict(parent.source_hashes())
    for name in EXTRA:
        result[name] = sha(ROOT / name)
    for name in ("binding.py", "runtime.py", "test_contract.py", "HANDOFF.md",
                 "planner_contract.json", "actual_generated_main.py.txt"):
        path = HERE / name
        result[str(path.relative_to(ROOT))] = sha(path)
    return dict(sorted(result.items()))


def source_hashes(expected=None):
    frozen = json.loads((HERE / "source_freeze.json").read_text())
    if frozen["schema"] != "p102-v43-actual-consumer-freeze-v1" or frozen["version"] != VERSION:
        raise RuntimeError("selected consumer freeze version differs")
    expected = frozen["sha256"] if expected is None else expected
    current = sources()
    if set(expected) != set(current):
        raise RuntimeError("selected consumer required source membership differs")
    if current != expected:
        raise RuntimeError("selected consumer source bytes changed")
    return current


def planner_pins():
    receipt = json.loads((HERE / "planner_contract.json").read_text())
    if sha(ROOT / receipt["source"]) != receipt["source_sha256"]:
        raise RuntimeError("selected actual SDK planner source changed")
    return receipt["functions"]


def statements(source):
    return ast.parse(textwrap.dedent(source)).body


def one(nodes, predicate, name):
    found = [n for n in nodes if predicate(n)]
    if len(found) != 1:
        raise RuntimeError(f"exact actual main {name} count differs: {len(found)}")
    return found[0]


def rewrite(source):
    if source_custody_enabled() and sha256(source.encode()).hexdigest() != PARENT_TEXT:
        raise RuntimeError("actual own generated P102 source changed")
    tree = ast.parse(source)
    main = one(tree.body, lambda n: isinstance(n, ast.FunctionDef) and n.name == "main", "main")
    source_guard = one(main.body, lambda n: isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "source" for t in n.targets), "source admission")
    at = main.body.index(source_guard) + 1
    main.body[at:at] = statements("""
        if a.method != SELECTED_METHOD:
            raise ValueError('selected consumer method differs')
        if not a.preflight_only:
            V43.admit_run(a.method, V43_POLICY, V43_CALIBRATION)
    """)
    init = one(main.body, lambda n: isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "scene" for t in n.targets), "runtime local initialization")
    main.body.insert(main.body.index(init) + 1, statements("consumer = None")[0])
    block = one(main.body, lambda n: isinstance(n, ast.Try), "runtime try")
    reviewer_branch = one(block.body, lambda n: isinstance(n, ast.If)
                          and any(isinstance(x, ast.Assign) and any(
                              isinstance(t, ast.Name) and t.id == "config" for t in x.targets)
                              for x in n.body), "reviewer constructor")
    reviewer_branch.body = statements("""
        reviewer, adapter = V43.make_reviewer_adapter(
            scene, instruction=INSTRUCTION, model=reviewer_model,
            episode=f'P1-02-seed{a.seed}-round1', output=output,
            policy=V43_POLICY, calibration=V43_CALIBRATION, on_step=on_step)
    """)
    task_branch = one(block.body, lambda n: isinstance(n, ast.If)
                      and any(isinstance(x, ast.Assign) and any(
                          isinstance(t, ast.Name) and t.id == "task_gate" for t in x.targets)
                          for x in n.body), "task/action consumer")
    old_proceed = one(task_branch.body, lambda n: isinstance(n, ast.If)
                      and ast.unparse(n.test) == "decision.result == 'PROCEED'", "task proceed")
    run = one(old_proceed.body, lambda n: isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "run" for t in n.targets), "original planner loop")
    run_call = one(ast.walk(run), lambda n: isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == "run_single_round_b0", "planner call")
    run_call.func = ast.Attribute(ast.Name("V43", ast.Load()), "run_planner", ast.Load())
    run_call.args.insert(0, ast.Name("run_single_round_b0", ast.Load()))
    run_call.keywords.append(ast.keyword("policy", ast.Name("V43_POLICY", ast.Load())))
    report_assignments = old_proceed.body[old_proceed.body.index(run) + 1:]
    report_assignments = [n for n in report_assignments if not (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript)
        and ast.unparse(t) in {"report['action_gate']", "report['perception']"} for t in n.targets))]
    task_branch.body = statements("""
        consumer = V43.Consumer(
            public_base, adapter, components=CONFIGURED, scene=scene, sam_client=sam_client,
            reviewer=reviewer, policy=V43_POLICY, instruction=INSTRUCTION,
            episode=f'P1-02-seed{a.seed}-round1', primitives=base.original._primitives,
            vla_chunk_controls=V43_VLA_CHUNK_CONTROLS)
        gated = consumer.boundary
        gated.begin_round(INSTRUCTION, round_id=1)
        report['task_gate'] = consumer.task_report()
        if consumer.task_fault():
            raise RuntimeError('actual typed task review failed')
        if gated.task_approved and gated.round_open:
            pick_toolkit = PlainRPentToolkit(
                FaultStopToolkit(gated, adapter), public_packet=scene.capture_agent)
            bridge = OwnerThreadToolkit(pick_toolkit, request_timeout_s=1800)
            V43.bind_owner_abort(bridge, adapter)
            stage = time.perf_counter()
        else:
            report['terminal'] = base.finish_result
            report['tool_calls'] = list(base.calls)
            report['primitive_calls'] = base.episode_calls
    """)
    new_proceed = one(task_branch.body, lambda n: isinstance(n, ast.If)
                      and ast.unparse(n.test) == "gated.task_approved and gated.round_open", "new proceed")
    new_proceed.body.extend([run, *report_assignments])
    final_gated = one(block.finalbody, lambda n: isinstance(n, ast.If)
                      and ast.unparse(n.test) == "gated is not None", "final action report")
    final_gated.test = ast.parse("consumer is not None", mode="eval").body
    final_gated.body = statements("""
        report['action_gate'] = consumer.action_report()
        report['perception'] = consumer.provider.log
    """)
    final_commit = one(block.finalbody, lambda n: isinstance(n, ast.If)
                       and "hasattr(adapter, 'commit_gate')" in ast.unparse(n.test), "final commit report")
    final_commit.test = ast.parse("adapter is not None and hasattr(adapter, 'dispatcher')", mode="eval").body
    final_commit.body = statements("report['commit_gate'] = V43.commit_report(adapter)")
    # Keep the original typed classifier and hazards-first decision. The new
    # raw dispatcher dictionaries retain real UNRESOLVED/system fault fields.
    capability = one(block.finalbody, lambda n: isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Subscript) and ast.unparse(t) ==
                             "report['public_capability_uncertainties']" for t in n.targets), "typed uncertainty report")
    capability.value = ast.parse(
        "[row for row in commit_rows if row.get('failure_kind') == FailureKind.PUBLIC_UNKNOWN_TILT.value] + "
        "((report.get('action_gate') or {}).get('public_capability_uncertainties', []))", mode="eval").body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def selected(instance="P1-02", *, method="Ours", version=VERSION, policy=None,
             calibration=None, vla_chunk_controls=None, admit=True):
    if instance != "P1-02" or method not in {"B0", "Ours"} or version != VERSION:
        raise ValueError("exact own P102 B0/Ours consumer only")
    if admit:
        source_hashes()
    old = parent.build_entry(instance)
    original_globals = {k: old.main.__globals__[k] for k in ("Scene", "Evaluator", "CONFIGURED")}
    name = __name__ + "._generated_" + method
    previous = sys.modules.get(name)
    text = rewrite(old.selection_text)
    try:
        module = compiled(text, name, old.__file__, {
            "CONFIGURED": old.CONFIGURED, "V43": runtime, "SELECTED_METHOD": method,
            "V43_POLICY": policy, "V43_CALIBRATION": calibration,
            "V43_VLA_CHUNK_CONTROLS": vla_chunk_controls})
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    module.hashes = source_hashes  # Install whole selected packet AFTER compile.
    module.PROTOCOL = VERSION
    module.native_method_ready = module.launch_authorized = False
    module.selection_text = text
    module.consumer_selection = {
        "instance": instance, "method": method, "version": version,
        "policy_sha256": None if policy is None else policy.sha256,
        "calibration_bound": calibration is not None,
        "grasp_assumption": "unverified" if method == "Ours" else None,
        "vla_chunk_controls": vla_chunk_controls,
        "native_ready": False, "launch_authorized": False,
    }
    for key, value in original_globals.items():
        if module.main.__globals__[key] is not value:
            raise RuntimeError("own Scene/scorer/config changed")
    if admit:
        module.source_admission = source_hashes()
    return module
