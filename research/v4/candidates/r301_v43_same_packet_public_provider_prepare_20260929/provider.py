"""Own public acquisition suffix only; no Consumer/main/factory is selected here."""
import ast
from research.v4.leash_v43.run_manifest import source_custody_enabled
from pathlib import Path
from dataclasses import replace
from itertools import product
import json
from types import SimpleNamespace

import numpy as np

from research.v4.accelerated.public_geometry import sam_public_box
from research.v4.accelerated.p3_gated_toolkit import proposal_for_p3
from research.v4.candidates.e2_laundry_method_20260926.public_intent import infer_public_goal
from research.v4.candidates.leash_commit_gate_20260924.preflight_gates import PublicWorkspace
from research.v4.candidates.r3_current_real_front_consumer_20260928.signals import NoPublicMask
from research.v4.leash_v43.registry import Holding, HoldingEstimator

from .sam_source import validate_supplied_client
if source_custody_enabled():
    from .source import HERE, ROOT, admit, source_text
else:
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parents[3]
    def admit():
        return None
    def source_text(name):
        return (ROOT / name).read_text()

DONOR = "research/v4/candidates/r3_current_real_front_consumer_20260928/public.py"
FAMILY = "research/v4/candidates/r3_family_method_config_20260927/public.py"


class OriginalDecodedClient:
    def __init__(self, original):
        self.result_class = validate_supplied_client(original)
        self.original = original

    def segment(self, *args, **kwargs):
        result = self.original.segment(*args, **kwargs)
        if type(result) is not self.result_class or type(result.found) is not bool:
            raise TypeError("expected original decoded Sam3Result")
        if result.found is False:
            if result.mask is not None or result.mask_shape is not None:
                raise ValueError("decoded SAM miss has contradictory mask")
            raise NoPublicMask()
        if result.mask is None:
            raise ValueError("decoded SAM detection omitted mask")
        return result


def _remember_packet(provider, packet):
    provider.current_packet = packet
    return packet


def _public_workspace(provider, packet, eef, boxes):
    if provider.current_packet is not packet:
        raise RuntimeError("public acquisition identity changed")
    estimator = provider.adapter.estimator
    if type(estimator) is not HoldingEstimator or type(estimator.kind) is not Holding:
        raise TypeError("original public holding estimator required")
    # This is the original estimator's executed/proprio state. No observation,
    # common planner latch, object, assist, weld or private geometry is read.
    robot = SimpleNamespace(hold_estimate=estimator.kind.value.lower())
    return PublicWorkspace(SimpleNamespace(robot=robot), eef, boxes)


def prepared_functions():
    """Compile only the guarded acquisition and original public parser/locator."""
    admit()
    contract = json.loads((HERE / "contract.json").read_text()) if source_custody_enabled() else {
        "query_list_ast": "[('microwave', 'microwave'), ('microwave_door', 'open microwave door panel'), (b.source, b.source_prompt)]"}
    tree = ast.parse(source_text(DONOR))
    original = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "workspace"]
    if len(original) != 1 or (source_custody_enabled() and ast.dump(original[0], include_attributes=False) != contract["workspace_ast"]):
        raise RuntimeError("original two-view acquisition changed")
    function = original[0]
    captures = [n for n in ast.walk(function) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "entry.scene.capture_agent"]
    if len(captures) != 1:
        raise RuntimeError("original single capture seam changed")
    capture = captures[0]
    capture.func = ast.Name(id="_remember_capture", ctx=ast.Load())
    capture.args = [ast.Name(id="entry", ctx=ast.Load())]
    # _remember_capture performs exactly the original single capture below.
    lists = [n for n in ast.walk(function) if isinstance(n, ast.List)
             and ast.unparse(n) == contract["query_list_ast"]]
    if len(lists) != 1 or len(lists[0].elts) != 3:
        raise RuntimeError("original query list changed")
    del lists[0].elts[1]
    returns = [n for n in ast.walk(function) if isinstance(n, ast.Return)]
    if len(returns) != 1 or (source_custody_enabled() and ast.unparse(returns[0].value) != contract["workspace_return_ast"]):
        raise RuntimeError("original evidence suffix changed")
    old = returns[0].value
    returns[0].value = ast.Call(func=ast.Name(id="_public_workspace", ctx=ast.Load()),
                               args=[ast.Name(id="entry", ctx=ast.Load()),
                                     ast.Name(id="packet", ctx=ast.Load()), *old.args[1:]], keywords=[])
    # A reader exception propagates as a required public technical fault.
    # No private adapter latch is exposed to the decision-side provider.
    if not source_custody_enabled():
        class PublicFault(ast.NodeTransformer):
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "entry.adapter._latch":
                    return None
                return self.generic_visit(node)
        function = PublicFault().visit(function)
    namespace = dict(np=np, replace=replace, sam_public_box=sam_public_box,
                     DetectionClient=OriginalDecodedClient, NoPublicMask=NoPublicMask,
                     _remember_capture=_remember_capture, _public_workspace=_public_workspace,
                     __name__=__name__)
    selected = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(selected, str(ROOT / DONOR) + "::R301-v43-samepacket", "exec",
                 dont_inherit=True), namespace)
    family = ast.parse(source_text(FAMILY))
    functions = [n for n in family.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("proposal", "locate")]
    if len(functions) != 2:
        raise RuntimeError("original own public proposal/locator unavailable")
    namespace.update(SimpleNamespace=SimpleNamespace, product=product,
                     proposal_for_p3=proposal_for_p3, infer_public_goal=infer_public_goal)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT / FAMILY), "exec",
                 dont_inherit=True), namespace)
    # Extract the same public eight-corner bound, not authored payload sizes.
    proposal_node = next(n for n in functions if n.name == "proposal")
    held_if = next(n for n in proposal_node.body if isinstance(n, ast.If))
    sweep_node = ast.FunctionDef(name="observed_sweep", args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg="b"), ast.arg(arg="workspace")],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=[*held_if.body,
        ast.Return(value=ast.Name(id="sweep", ctx=ast.Load()))], decorator_list=[])
    sweep_tree = ast.fix_missing_locations(ast.Module(body=[sweep_node], type_ignores=[]))
    exec(compile(sweep_tree, str(ROOT / FAMILY) + "::observed-corner-bound", "exec",
                 dont_inherit=True), namespace)
    return SimpleNamespace(workspace=namespace["workspace"], proposal=namespace["proposal"],
                           locate=namespace["locate"], observed_sweep=namespace["observed_sweep"],
                           emitted_workspace_ast=ast.dump(function, include_attributes=False) if source_custody_enabled() else None)


def _remember_capture(provider):
    return _remember_packet(provider, provider.scene.capture_agent())


def public_roles():
    admit()
    if not source_custody_enabled():
        from research.v4.candidates.r3_family_method_config_20260927.config import binding
        return binding("R3-01")
    roles = json.loads((HERE / "contract.json").read_text())["public_roles"]
    return SimpleNamespace(**{**roles, "source_aliases": tuple(roles["source_aliases"])})


class SamePacketR301PublicProvider:
    """Future supplied real scene/adapter/client only. Not instantiated in preparation."""
    def __init__(self, scene, adapter, sam_client):
        self.prepared = prepared_functions()
        validate_supplied_client(sam_client)
        self.scene, self.adapter, self.sam_client = scene, adapter, sam_client
        self.roles = public_roles()
        self.log, self.current_packet = [], None

    def __call__(self):
        return self.prepared.workspace(self.roles, self, self.sam_client, self.log)

    @property
    def public_entity_names(self):
        return ('microwave', self.roles.source)

    def locate_vla_target(self, prompt, workspace):
        return self.prepared.locate(self.roles, prompt, workspace)

    def observed_sweep(self, workspace):
        return self.prepared.observed_sweep(self.roles, workspace)

    def proposal(self, name, args, workspace):
        return self.prepared.proposal(self.roles, name, args, workspace,
                                      self.locate_vla_target)
