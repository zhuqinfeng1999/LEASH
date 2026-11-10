"""Own R202 public acquisition over existing Consumer and RPent execution."""
import ast
from dataclasses import replace
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import textwrap
from types import FunctionType, SimpleNamespace

import numpy as np

from research.v4.candidates.r2_02_natural_method_20260926 import public_boundary as own
from research.v4.candidates.p4_natural_method_20260926 import public_boundary as acquisition
from research.v4.candidates.p102_v43_actual_consumer_20260928 import runtime as common
from research.v4.candidates.v43_task_constraints_binding_20260928.runtime import TaskConstraints
from research.v4.candidates.v43_own_lease_lifecycle_20260928.runtime import OwnLeaseLifecycle
from research.v4.candidates.e2_cat_corrected_method_20260926.phase_adapter import LandingBoundary

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SWEEP_SOURCE = ROOT / "research/v4/candidates/e1_02_plain_rpent_entry_20260927/bindings.py"

admit_run = common.admit_run
make_reviewer_adapter = common.make_reviewer_adapter
bind_owner_abort = common.bind_owner_abort
commit_report = common.commit_report
run_planner = common.run_planner


def _cell(value):
    return (lambda: value).__closure__[0]


def public_reader_class():
    """Change only the existing acquisition's evidence suffix, not perception."""
    contract = json.loads((HERE / "contract.json").read_text())
    original = acquisition.PublicWorkspaceReader.__call__
    source = textwrap.dedent(inspect.getsource(original))
    if sha256(source.encode()).hexdigest() != contract["reader_source_sha256"]:
        raise RuntimeError("original R202 acquisition source changed")
    tree = ast.parse(source)
    returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
    if len(returns) != 1 or ast.unparse(returns[0].value) != contract["reader_return_ast"]:
        raise RuntimeError("exact original public evidence suffix changed")
    returns[0].value = ast.parse(
        "_public_workspace(self, packet, tuple(float(x) for x in packet['robot_state']['eef_pos']), tuple(boxes))",
        mode="eval").body
    namespace = dict(original.__globals__)
    namespace.update(_public_workspace=common._public_workspace,
                     PROMPTS={"iron": own.PUBLIC_NOUNS["item"], "box": own.PUBLIC_NOUNS["box"]})
    exec(compile(ast.fix_missing_locations(tree), original.__code__.co_filename + "::R202-v43-return", "exec"), namespace)
    base = type("OwnR202PublicAcquisition", (acquisition.PublicWorkspaceReader,),
                {"__call__": namespace["__call__"]})
    reader = type("OwnR202PublicWorkspaceReader", (base,), {})
    outer = own.PublicWorkspaceReader.__call__
    if outer.__code__.co_freevars != ("__class__",):
        raise RuntimeError("own ItemWorkspace super delegation changed")
    # The original outer normalization/cache code is unchanged. Its super now
    # reaches the exact source-bound acquisition above, rather than old evidence.
    reader.__call__ = FunctionType(outer.__code__, outer.__globals__, outer.__name__,
                                   outer.__defaults__, (_cell(reader),))
    reader.locate_vla_target = own.PublicWorkspaceReader.locate_vla_target
    reader.selected_acquisition_source = ast.unparse(tree)
    reader.original_outer_code = outer.__code__
    return reader


def observed_sweep_class():
    """Reuse the actual corner/EEF algorithm with the own public item name."""
    contract = json.loads((HERE / "contract.json").read_text())
    source = SWEEP_SOURCE.read_text()
    if sha256(source.encode()).hexdigest() != contract["sweep_file_sha256"]:
        raise RuntimeError("existing observed payload source changed")
    classes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
               and node.name == "ObservedPayloadSweep"]
    if len(classes) != 1:
        raise RuntimeError("observed payload class missing or duplicated")
    node = classes[0]
    labels = [value for value in ast.walk(node) if isinstance(value, ast.Constant)
              and value.value == "held cleaver full public box unavailable"]
    if len(labels) != 1:
        raise RuntimeError("observed payload diagnostic source changed")
    labels[0].value = "held item public box unavailable"
    namespace = {"__name__": __name__, "SOURCE": "item", "LandingBoundary": LandingBoundary,
                 "math": math, "np": np}
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(tree, str(SWEEP_SOURCE) + "::R202-public-item", "exec"), namespace)
    return namespace["ObservedPayloadSweep"]


def components():
    sweep = observed_sweep_class()
    return SimpleNamespace(binding=SimpleNamespace(instance="R2-02", source="item", target="box"),
                           bindings=SimpleNamespace(ObservedPayloadSweep=sweep),
                           PublicWorkspaceFactory=public_reader_class(),
                           public_nouns=dict(own.PUBLIC_NOUNS), aliases=dict(own.ALIASES))


def public_provider(selected, scene, adapter, sam_client):
    # No provider, Consumer, Scene or SAM object is constructed by selection.
    return selected.PublicWorkspaceFactory(scene, adapter, sam_client)


class OwnPublicConsumer(common.Consumer):
    def __init__(self, *args, components, **kwargs):
        original = common.Consumer.__init__
        namespace = dict(original.__globals__, public_provider=public_provider)
        selected = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
        selected.__kwdefaults__ = original.__kwdefaults__
        selected(self, *args, components=components, **kwargs)

    def action_report(self):
        report = super().action_report()
        # Existing target cache remains a previous public observation. A new
        # image/acquisition never certifies that cached box was recognized again.
        return {**report, "public_box_cache_protocol": {
            "source_reobserved": "item", "cached_previous_public_result": "box",
            "new_capture_is_not_fresh_cached_box_recognition": True,
            "no_extra_SAM_or_pixel_hash_invalidation": True}}


class R202Consumer(TaskConstraints, OwnLeaseLifecycle, OwnPublicConsumer):
    """Existing task/own-lease lifecycle, with exact own public components."""


def namespace():
    return SimpleNamespace(**{**vars(common), "Consumer": R202Consumer})
