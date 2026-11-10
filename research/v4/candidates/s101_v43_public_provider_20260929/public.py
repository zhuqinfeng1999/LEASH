"""Narrow source binding over current S101 dual-view acquisition/proposals."""
from research.v4.leash_v43.run_manifest import source_custody_enabled
import ast
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from research.v4.candidates.s1_own_method_dispatch_20260927 import factory as own
from research.v4.candidates.r202_v43_public_fault_types_20260929 import public as typed
from research.v4.candidates.p102_v43_actual_consumer_20260928.runtime import _public_workspace
from research.v4.leash_v43.method_inputs import is_model_response_failure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def _function_node(path, name, binding):
    contract = json.loads((HERE / "contract.json").read_text()) if source_custody_enabled() else None
    text = (ROOT / path).read_text()
    if source_custody_enabled() and sha256(text.encode()).hexdigest() != contract["source_sha256"][path]:
        raise RuntimeError("Exact own S101 algorithm source changed")
    tree = ast.parse(text)
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise RuntimeError("Exact original S101 function missing or duplicated")
    node = found[0]
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            n.value = own.labels(n.value, binding)
    return node


def _names(provider):
    try:
        names = provider.forbidden_entities()
        if not isinstance(names, (tuple, list, set, frozenset)) or not all(
                isinstance(n, str) and n in provider.public_prompts for n in names):
            raise ValueError("Current validated S1 public entity names unavailable")
        return names
    except Exception as exc:
        if is_model_response_failure(exc):
            raise
        typed._fault(provider, exc, stage="current_public_entities")


class _Client:
    """Reuse the reviewed result typing under the original sam_public_box."""
    def __init__(self, provider, packet, camera, name, prompt):
        self.provider, self.packet, self.camera, self.name, self.prompt = (
            provider, packet, camera, name, prompt)

    def segment(self, image, *, text_prompt, min_score):
        if min_score != .2 or text_prompt != self.prompt:
            raise RuntimeError("Original S1 SAM arguments changed")
        return typed._segment(SimpleNamespace(sam=self.provider.sam3_client), self.packet,
                              self.camera, self.name, self.prompt, image)


def _miss(provider, packet, name, camera, exc):
    row = {**exc.row, "failure_kind": "structured_public_detection_miss", "fault": False}
    provider.log.append(row)
    return f"{camera}: structured Sam3Result no-detection"


def _unknown_geometry(proposal, hold):
    return replace(proposal, geometry_ambiguous=True) if hold == "unknown" else proposal


def selected_components():
    if source_custody_enabled():
        from .sources import source_hashes
        source_hashes()
    # Existing own semantic factory: definitions only, no Scene construction.
    current = own.components("S1-01")
    node = _function_node(
        "research/v4/candidates/s1_native_cli_v5_typed_place_20260925/dual_view_workspace.py",
        "__call__", current.binding)
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
             and ast.unparse(n) == "self.scene.capture_agent()"]
    evidence = [n for n in node.body if isinstance(n, ast.Assign)
                and ast.unparse(n) == "evidence = self.adapter.capture_commit_evidence()"]
    returns = [n for n in node.body if isinstance(n, ast.Return)]
    role = [n for n in node.body if isinstance(n, ast.If)
            and ast.unparse(n.test) == "self.task_adapter and self.task_adapter.task_decisions"]
    sam = [n for n in ast.walk(node) if isinstance(n, ast.Call)
           and isinstance(n.func, ast.Name) and n.func.id == "sam_public_box"]
    handlers = [n for n in ast.walk(node) if isinstance(n, ast.ExceptHandler)]
    if (len(calls), len(evidence), len(returns), len(role), len(sam), len(handlers)) != (1, 1, 1, 1, 1, 1):
        raise RuntimeError("Exact S101 public acquisition seam changed")
    if ast.unparse(returns[0].value) != "PublicWorkspace(evidence, eef, tuple(boxes))":
        raise RuntimeError("Exact S101 public evidence suffix changed")
    calls[0].func = ast.Name("_capture", ast.Load())
    calls[0].args = [ast.Name("self", ast.Load())]
    node.body[node.body.index(role[0])] = ast.parse("names.update(_names(self))").body[0]
    node.body.remove(evidence[0])
    returns[0].value = ast.parse("_public_workspace(self, packet, eef, tuple(boxes))", mode="eval").body
    client = next(k for k in sam[0].keywords if k.arg == "sam3_client")
    if ast.unparse(client.value) != "self.sam3_client":
        raise RuntimeError("Original S1 SAM client site changed")
    client.value = ast.parse("_Client(self, packet, camera_name, name, prompt)", mode="eval").body
    owner = next(n for n in ast.walk(node) if isinstance(n, ast.Try) and handlers[0] in n.handlers)
    owner.handlers = ast.parse("""try:
    pass
except DetectionMiss as exc:
    failures.append(_miss(self, packet, name, camera_name, exc))
except Exception as exc:
    _view_fault(self, packet, camera_name, name, prompt, exc)
""").body[0].handlers
    namespace = dict(current.dual.S1DualViewWorkspace.__call__.__globals__,
        _capture=typed._capture, _names=_names, _Client=_Client, _miss=_miss,
        _view_fault=typed._view_fault, DetectionMiss=typed.DetectionMiss,
        _public_workspace=_public_workspace)
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(tree, __file__ + "::own-S101-acquisition", "exec"), namespace)
    provider_class = type("S101V43PublicProvider", (current.dual.S1DualViewWorkspace,),
                          {"__call__": namespace["__call__"]})

    proposal = _function_node(
        "research/v4/candidates/s1_multiround_driver_20260925/action_boundary.py",
        "s1_public_proposal", current.binding)
    guard = [n for n in ast.walk(proposal) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "hold not in {'held', 'empty'}"]
    close = [n for n in ast.walk(proposal) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "not held and grip == 1"]
    outputs = [n for n in ast.walk(proposal) if isinstance(n, ast.Return)]
    if (len(guard), len(close), len(outputs)) != (1, 1, 6):
        raise RuntimeError("Exact S101 UNKNOWN proposal seam changed")
    guard[0].test = ast.parse("hold not in {'held', 'empty', 'unknown'}", mode="eval").body
    close[0].test = ast.parse("hold == 'empty' and grip == 1", mode="eval").body
    for n in outputs:
        n.value = ast.Call(ast.Name("_unknown_geometry", ast.Load()),
                           [n.value, ast.Name("hold", ast.Load())], [])
    proposal_namespace = dict(current.action.s1_public_proposal.__globals__,
                              _unknown_geometry=_unknown_geometry)
    tree = ast.fix_missing_locations(ast.Module(body=[proposal], type_ignores=[]))
    exec(compile(tree, __file__ + "::unknown-proposal", "exec"), proposal_namespace)
    return SimpleNamespace(PublicWorkspaceFactory=provider_class,
        s1_public_proposal=proposal_namespace["s1_public_proposal"],
        original=current, prompts=dict(current.grounding.PROMPTS),
        transform_counts={"capture":1,"evidence_assignment_removed":1,"return_suffix":1,
                          "current_roles":1,"sam_client":1,"per_view_catch":1,
                          "unknown_guard":1,"empty_only_close_guard":1,"proposal_returns":6})


def make_provider(*, scene, adapter, sam3_client, forbidden_entities=None, components=None):
    selected = selected_components() if components is None else components
    callback = (lambda: ()) if forbidden_entities is None else forbidden_entities
    if not callable(callback):
        raise TypeError("Current public entity callback required")
    value = selected.PublicWorkspaceFactory(scene=scene, adapter=adapter, sam3_client=sam3_client)
    value.forbidden_entities = callback
    value.public_prompts = selected.prompts
    return value


def s1_public_proposal(name, args, workspace, locate_vla_target, *, components=None):
    selected = selected_components() if components is None else components
    return selected.s1_public_proposal(name, args, workspace, locate_vla_target)
