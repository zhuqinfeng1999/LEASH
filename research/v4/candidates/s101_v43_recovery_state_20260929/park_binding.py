"""Reuse the original park AST with only public holding/record reads changed."""
from research.v4.leash_v43.run_manifest import source_custody_enabled
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

from research.v4.candidates.s1_multiround_driver_20260925 import park as original
from . import state

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VERSION = "S101-v43-public-recovery-record-original-park-source-slice-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    parent = ROOT / "research/v4/candidates/s1_user_revision_real_binding_20260928/source_freeze.json"
    pins = json.loads(parent.read_text())
    pins[str(parent.relative_to(ROOT))] = sha(parent)
    for path in (ROOT / "research/v4/leash_v43").glob("*.py"):
        name = str(path.relative_to(ROOT))
        if name in pins and pins[name] != sha(path):
            raise RuntimeError("parent/current source conflict: " + name)
        pins[name] = sha(path)
    for path in HERE.glob("*.py"):
        pins[str(path.relative_to(ROOT))] = sha(path)
    doc = ROOT / "research/v4/docs/leash_v43_revised_20260928.txt"
    pins[str(doc.relative_to(ROOT))] = sha(doc)
    return dict(sorted(pins.items()))


def admit(actual, expected):
    if actual != expected:
        raise RuntimeError("recovery slice missing/changed source")
    for name, digest in actual.items():
        path = Path(name) if name.startswith("/") else ROOT / name
        if sha(path) != digest:
            raise RuntimeError("recovery slice source drift: " + name)
    return actual


def source_hashes():
    if not source_custody_enabled():
        return {}
    return admit(sources(), json.loads((HERE / "source_freeze.json").read_text()))


def selected_park():
    source_hashes()
    function = original.park_to_public_start
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    baseline = copy.deepcopy(tree) if source_custody_enabled() else None
    targets = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and ast.unparse(n) == "scene.public_grasp_latched()"]
    if len(targets) != 3:
        raise RuntimeError("original park's three holding reads changed")
    replacements = {}
    for node in targets:
        replacements[id(node)] = ast.parse("_public_holding(controller) is HoldingKind.HELD", mode="eval").body
    final = [n for n in ast.walk(tree) if isinstance(n, ast.keyword)
             and n.arg == "final_public_grasp_latched"]
    if len(final) != 1 or ast.unparse(final[0].value) != "scene.public_grasp_latched()":
        raise RuntimeError("original final recovery record changed")
    # Final field retains the full enum; only the two guard predicates use
    # HELD/EMPTY. The callback refuses UNKNOWN, so it is never latch False.
    final_id = id(final[0].value)
    replacements[final_id] = ast.parse("_public_holding(controller)", mode="eval").body
    class Replace(ast.NodeTransformer):
        def visit_Call(self, node):
            return replacements[id(node)] if id(node) in replacements else self.generic_visit(node)
        def visit_keyword(self, node):
            node = self.generic_visit(node)
            if node.arg == "final_public_grasp_latched":
                node.arg = "final_holding"
            return node
    tree = Replace().visit(tree)
    ast.fix_missing_locations(tree)
    # Independent inverse verifies controls, delta/clip/tolerances/cap and
    # step sequencing are byte-for-byte the same AST as the original.
    if source_custody_enabled():
        reverted = copy.deepcopy(tree)
        class Undo(ast.NodeTransformer):
            def visit_Compare(self, node):
                if ast.unparse(node) == "_public_holding(controller) is HoldingKind.HELD":
                    return ast.parse("scene.public_grasp_latched()", mode="eval").body
                return self.generic_visit(node)
            def visit_Call(self, node):
                if ast.unparse(node) == "_public_holding(controller)":
                    return ast.parse("scene.public_grasp_latched()", mode="eval").body
                return self.generic_visit(node)
            def visit_keyword(self, node):
                node = self.generic_visit(node)
                if node.arg == "final_holding":
                    node.arg = "final_public_grasp_latched"
                return node
        reverted = Undo().visit(reverted)
        if ast.dump(reverted, include_attributes=False) != ast.dump(baseline, include_attributes=False):
            raise RuntimeError("unexpected recovery motion/control change")

    namespace = {**function.__globals__, "_public_holding": state.public_holding,
                 "HoldingKind": state.HoldingKind, "RecoveryRecord": state.park_record}
    exec(compile(tree, str(HERE / "selected_original_park.py.txt"), "exec"), namespace)
    selected = namespace[function.__name__]
    selected.original = function
    selected.inverse_motion_ast_exact = source_custody_enabled()
    return selected
