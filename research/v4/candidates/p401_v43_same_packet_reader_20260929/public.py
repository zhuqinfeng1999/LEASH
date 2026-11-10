"""Thin typed-data and same-packet seam over the emitted own P4 reader.

The corner envelope, configuration-space algorithm, prompts, target parser,
static cache and successful two-view box union stay in the original factory.
This module neither installs a Consumer nor constructs a scene/client.
"""
import ast
from copy import deepcopy
from dataclasses import replace
from types import FunctionType, SimpleNamespace

import numpy as np

from research.v4.candidates.p102_v43_actual_consumer_20260928.runtime import _public_workspace


def function(original, namespace):
    selected = FunctionType(original.__code__, namespace, original.__name__,
                            original.__defaults__, original.__closure__)
    selected.__kwdefaults__ = original.__kwdefaults__
    return selected


def _segment(reader, packet, camera_name, name, prompt, image):
    metadata = {"frame_id": packet["frame_id"], "camera": camera_name,
                "name": name, "prompt": prompt}
    result = _selected_Client(reader.sam, metadata).segment(
        image, text_prompt=prompt, min_score=.2)
    # Only the original decoder's found=False with no contradictory mask is
    # a semantic miss. A malformed successful response remains technical.
    if (not isinstance(result.mask, np.ndarray) or result.mask.dtype != np.bool_
            or result.mask.shape != (480, 640) or result.mask_shape != (480, 640)
            or result.score is None or not np.isfinite(float(result.score))
            or float(result.score) < .2):
        raise ValueError("Successful original SAM result schema malformed")
    return result


def _view_fault(reader, packet, camera_name, name, prompt, exc):
    _selected_fault(reader, exc, stage="per_view_acquisition",
                    frame_id=packet["frame_id"], camera=camera_name,
                    name=name, prompt=prompt)


def reader_class(original, source, *, client, result_class, detection_miss,
                 capture, fault, guard):
    tree = ast.parse(source)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == "PublicWorkspaceReader")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef)
                  and n.name == "__call__")
    before = deepcopy(method)
    captures = [n for n in ast.walk(method) if isinstance(n, ast.Call)
                and ast.unparse(n) == "self.scene.capture_agent()"]
    segments = [n for n in ast.walk(method) if isinstance(n, ast.Call)
                and ast.unparse(n) == "self.sam.segment(buffer.getvalue(), text_prompt=prompt, min_score=0.2)"]
    returns = [n for n in ast.walk(method) if isinstance(n, ast.Return)]
    handlers = [n for n in ast.walk(method) if isinstance(n, ast.ExceptHandler)]
    expected_return = ("PublicWorkspace(self.adapter.capture_commit_evidence(), "
                       "tuple((float(x) for x in packet['robot_state']['eef_pos'])), tuple(boxes))")
    if ((len(captures), len(segments), len(returns), len(handlers)) != (1, 1, 1, 1)
            or ast.unparse(returns[0].value) != expected_return):
        raise RuntimeError("Exact own inline-EEF/capture/query/return/catch seam changed")
    # The original factory's observation validation unnecessarily requires
    # quaternion/gripper values for this xyz-only reader. Geometry retains its
    # separate original quaternion requirement; the reader validates only its
    # own required RGB-D/K/T/frame/xyz through the reviewed capture callback.
    removable = [(i, n) for i, n in enumerate(method.body)
                 if (isinstance(n, ast.ImportFrom) and n.module == "research.v4.accelerated.observation")
                 or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                     and ast.unparse(n.value.func) == "public_observation")]
    if len(removable) != 2:
        raise RuntimeError("Exact original public-observation prefix changed")
    original_capture, original_segment = deepcopy(captures[0]), deepcopy(segments[0])
    original_return, original_handlers = deepcopy(returns[0].value), deepcopy(handlers)
    handler_owner = next(n for n in ast.walk(method) if isinstance(n, ast.Try)
                         and handlers[0] in n.handlers)
    captures[0].func, captures[0].args = ast.Name("_capture", ast.Load()), [ast.Name("self", ast.Load())]
    segments[0].func = ast.Name("_segment", ast.Load())
    segments[0].args = [ast.Name(x, ast.Load()) for x in
                        ("self", "packet", "camera_name", "name", "prompt")] + segments[0].args
    segments[0].keywords = []
    returns[0].value = ast.parse(
        "_public_workspace(self, packet, tuple(float(x) for x in packet['robot_state']['eef_pos']), tuple(boxes))",
        mode="eval").body
    handler_owner.handlers = ast.parse("""try:
    pass
except DetectionMiss as exc:
    self.log.append({**exc.row, 'failure_kind': 'structured_public_detection_miss', 'fault': False})
except Exception as exc:
    _view_fault(self, packet, camera_name, name, prompt, exc)
""").body[0].handlers
    method.body = [n for i, n in enumerate(method.body) if i not in {i for i, _ in removable}]
    method.body.insert(0, ast.parse("_whole_source_guard()").body[0])
    selected_tree = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(selected_tree)
    typed_client = type("P401ActualOriginalTypedClient", (client,), {
        "__init__": client.__init__,
        "segment": function(client.segment, dict(client.segment.__globals__, Sam3Result=result_class))})
    segment = function(_segment, dict(_segment.__globals__, _selected_Client=typed_client))
    view_fault = function(_view_fault, dict(_view_fault.__globals__, _selected_fault=fault))
    namespace = dict(original.__call__.__globals__, _capture=capture, _segment=segment,
                     _view_fault=view_fault, DetectionMiss=detection_miss,
                     _public_workspace=_public_workspace, _whole_source_guard=guard)
    exec(compile(selected_tree, original.__call__.__code__.co_filename + "::P401-v43-typed", "exec"), namespace)
    reader = type("P401OwnSamePacketTypedReader", (original,), {"__call__": namespace["__call__"]})
    # An exact AST inverse, using captured original nodes rather than text
    # reparse or bytecode normalization. It covers the successful algorithm.
    inverse = deepcopy(method)
    inverse.body.pop(0)
    for i, node in removable:
        inverse.body.insert(i, deepcopy(node))
    transformed_capture = next(n for n in ast.walk(inverse) if isinstance(n, ast.Call)
                               and ast.unparse(n.func) == "_capture")
    transformed_capture.func, transformed_capture.args = original_capture.func, original_capture.args
    transformed_segment = next(n for n in ast.walk(inverse) if isinstance(n, ast.Call)
                               and ast.unparse(n.func) == "_segment")
    transformed_segment.func = original_segment.func
    transformed_segment.args, transformed_segment.keywords = original_segment.args, original_segment.keywords
    next(n for n in ast.walk(inverse) if isinstance(n, ast.Return)).value = original_return
    next(n for n in ast.walk(inverse) if isinstance(n, ast.Try)).handlers = original_handlers
    reader.inverse_ast_exact = ast.dump(inverse) == ast.dump(before)
    if not reader.inverse_ast_exact:
        raise RuntimeError("Original full P4 successful acquisition AST changed outside the thin seam")
    reader.selected_acquisition_source = ast.unparse(selected_tree)
    reader.original_acquisition_code = original.__call__.__code__
    reader.source_changes = {"capture": 1, "segment": 1, "typed_catch": 1,
                             "same_packet_inline_eef_return": 1, "required_only_prefix": 1, "whole_guard": 1}
    reader.original_geometry_unchanged = True
    return reader
