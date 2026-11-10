"""E101 own public callbacks; execution remains in the original RPent outlet.

No scene, consumer, acquisition or model is constructed by source selection.
The parent ordinary observation and common declared planner latch are unchanged.
"""
from research.v4.leash_v43.run_manifest import source_custody_enabled
import ast
from collections.abc import Mapping
from dataclasses import asdict, replace
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import textwrap
from types import FunctionType, SimpleNamespace

import numpy as np

from research.v4.accelerated.observation import _camera, PUBLIC_CAMERAS
from research.v4.accelerated.e1_gate_boundary import requires_descent_review
from research.v4.candidates.p102_v43_actual_consumer_20260928 import runtime as common
from research.v4.candidates.v43_task_constraints_binding_20260928.runtime import TaskConstraints
from research.v4.candidates.v43_own_lease_lifecycle_20260928.runtime import OwnLeaseLifecycle
from research.v4.candidates.r4_02_typed_public_masks_20260927.public import Client, DetectionMiss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VERSION = "E101-own-v43-task-lease-public-runtime-source-v1"
PARENTS = ("v43_task_constraints_binding_20260928", "e1_plain_config_dispatch_20260927",
           "r4_02_typed_public_masks_20260927")
admit_run = common.admit_run
make_reviewer_adapter = common.make_reviewer_adapter
bind_owner_abort = common.bind_owner_abort
run_planner = common.run_planner
commit_report = common.commit_report


def sources():
    required = {}
    for name in PARENTS:
        path = ROOT / "research/v4/candidates" / name / "source_freeze.json"
        parent = json.loads(path.read_text())["sha256"]
        for key, value in parent.items():
            if key in required and required[key] != value:
                raise RuntimeError("Complete E101 consumer parent source obligations conflict")
            required[key] = value
        required[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
    for name in ("__init__.py", "runtime.py", "contract.json", "HANDOFF.md", "check_source.py"):
        path = HERE / name
        required[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
    actual = {}
    for key, expected in required.items():
        path = Path(key) if key.startswith("/") else ROOT / key
        value = sha256(path.read_bytes()).hexdigest()
        if value != expected:
            raise RuntimeError("Immutable complete E101 consumer source changed: " + key)
        actual[key] = value
    return dict(sorted(actual.items()))


def source_hashes(expected=None):
    if not source_custody_enabled():
        return {}
    freeze = json.loads((HERE / "source_freeze.json").read_text())
    if freeze["version"] != VERSION:
        raise ValueError("Exact E101 public consumer source version required")
    actual = sources()
    if actual != (freeze["sha256"] if expected is None else expected):
        raise RuntimeError("Whole E101 public consumer source membership/bytes changed")
    return actual


class PublicReaderTechnicalFault(RuntimeError):
    def __init__(self, row):
        self.row = row
        super().__init__("Required E101 public reader input/query failed")


def _fault(reader, exc, **metadata):
    row = {**metadata, "fault": True, "error_type": type(exc).__name__, "error": str(exc)}
    reader.log.append(row)
    raise PublicReaderTechnicalFault(row) from exc


def _capture(reader):
    try:
        packet = reader.scene.capture_agent()
        if not isinstance(packet, Mapping):
            raise TypeError("Public capture must be a mapping")
        frame = packet["frame_id"]
        if type(frame) is not int or frame < 0 or set(packet["cameras"]) != set(PUBLIC_CAMERAS):
            raise ValueError("Exact public frame/two-camera input unavailable")
        for name in PUBLIC_CAMERAS:
            camera = packet["cameras"][name]
            _camera(name, camera, frame)
            if abs(float(np.linalg.det(np.asarray(camera["extrinsics"], float)[:3, :3])) - 1.) > 1e-4:
                raise ValueError("Public camera rotation malformed")
        eef = np.asarray(packet["robot_state"]["eef_pos"], float)
        if eef.shape != (3,) or not np.isfinite(eef).all():
            raise ValueError("Required public end-effector position unavailable")
        return packet
    except Exception as exc:
        _fault(reader, exc, boundary="existing_E101_capture")


class _TypedClient(Client):
    def segment(self, *args, **kwargs):
        result = super().segment(*args, **kwargs)
        if (not isinstance(result.mask, np.ndarray) or result.mask.dtype != np.bool_
                or result.mask.shape != (480, 640) or result.mask_shape != (480, 640)):
            raise ValueError("Successful SAM3 mask schema malformed")
        return result


def _box(reader, packet, view, noun):
    names = reader.original_names
    instance = reader.instance
    canonical = names.canonical(instance, noun)
    prompt = names.perception_queries(instance)[canonical]
    metadata = {"frame_id": packet["frame_id"], "camera": view, "name": canonical, "prompt": prompt}
    return names.public_box(instance, noun, camera=packet["cameras"][view],
                            sam3_client=_TypedClient(reader.sam_client, metadata))


def public_reader_class(components):
    """Keep the actual E1 union/parser/geometry and replace its evidence suffix."""
    if components.binding.instance != "E1-01":
        raise ValueError("Only exact own E101 source scope is selected")
    original = components.PublicWorkspaceFactory
    source = textwrap.dedent(inspect.getsource(original.__call__))
    contract = json.loads((HERE / "contract.json").read_text()) if source_custody_enabled() else None
    if source_custody_enabled() and sha256(source.encode()).hexdigest() != contract["reader_source_sha256"]:
        raise RuntimeError("Actual own E101 acquisition source changed")
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    captures = [n for n in calls if ast.unparse(n.func) == "self.scene.capture_agent"]
    queries = [n for n in calls if ast.unparse(n.func) == "names.public_box"]
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    if (len(captures), len(queries), len(returns), len(handlers)) != (1, 1, 1, 1):
        raise RuntimeError("Exact own capture/query/return/catch source counts changed")
    if source_custody_enabled() and ast.unparse(returns[0].value) != contract["reader_return_ast"]:
        raise RuntimeError("Exact own E101 public return changed")
    captures[0].func = ast.Name(id="_capture", ctx=ast.Load())
    captures[0].args = [ast.Name(id="self", ctx=ast.Load())]
    queries[0].func = ast.Name(id="_box", ctx=ast.Load())
    queries[0].args = [ast.Name(id=name, ctx=ast.Load()) for name in ("self", "packet", "view", "noun")]
    queries[0].keywords = []
    returns[0].value = ast.parse("_public_workspace(self, packet, eef, tuple(boxes))", mode="eval").body
    parent_try = next(n for n in ast.walk(tree) if isinstance(n, ast.Try) and handlers[0] in n.handlers)
    parent_try.handlers = ast.parse("""
try:
    pass
except DetectionMiss as exc:
    self.log.append(exc.row)
except Exception as exc:
    _fault(self, exc, frame_id=frame, camera=view, name=noun)
""").body[0].handlers
    namespace = dict(original.__call__.__globals__)
    namespace.update(_capture=_capture, _box=_box, _fault=_fault,
                     _public_workspace=common._public_workspace, DetectionMiss=DetectionMiss)
    exec(compile(ast.fix_missing_locations(tree), original.__call__.__code__.co_filename + "::E101-v43", "exec"), namespace)
    methods = {"__call__": namespace["__call__"],
        "original_names": original.__call__.__globals__["names"], "instance": components.binding.instance,
        "public_entity_names": tuple(components.bindings.PROMPTS)}
    if source_custody_enabled():
        methods.update(selected_acquisition_source=ast.unparse(tree), original_acquisition_code=original.__call__.__code__)
    return type("E101OwnTypedPublicWorkspace", (original,), methods)


def public_provider(components, scene, adapter, sam_client):
    return public_reader_class(components)(scene, adapter, sam_client)


class E101BaseConsumer(common.Consumer):
    def __init__(self, *args, components, **kwargs):
        if components.binding.instance != "E1-01":
            raise ValueError("Exact own E101 components required")
        super().__init__(*args, components=components, provider_factory=public_provider, **kwargs)

    def resolve(self, name, parameters, workspace):
        resolved = super().resolve(name, parameters, workspace)
        proposal = resolved.proposal
        kind = self.adapter.estimator.kind.value.lower()
        source = self.components.binding.source
        # The exact original own nearest-box rule remains for known EMPTY.
        if kind == "empty" and name in {"move_to", "move_pose", "set_gripper"}:
            box = workspace.public.box(source)
            point = proposal.motion_waypoints[-1] if proposal.motion_waypoints else workspace.public.eef_xyz
            if box is not None and point is not None:
                nearest = tuple(min(max(p, lo), hi) for p, lo, hi in zip(point, box.lower, box.upper))
                if math.dist(point, nearest) <= .055:
                    proposal = replace(proposal, operated_entity=source)
        path = proposal.motion_waypoints
        if kind != "empty" and len(path) >= 2 and requires_descent_review(
                held_public=True, primitive_name="move_to", current_eef_xyz=path[0], proposed_eef_xyz=path[-1]):
            proposal = replace(proposal, geometry_ambiguous=True,
                               parameters=(*proposal.parameters, ("phase", "held_descent_precritical")))
        if kind != "empty" and proposal.name == "vla_act":
            proposal = replace(proposal, geometry_ambiguous=True)
        # This final proposal is seen by OwnLeaseLifecycle.resolve, so phase
        # changes expire the existing binding before its next commit projection.
        self.proposals[-1]["proposal"] = asdict(proposal)
        return replace(resolved, proposal=proposal,
                       entities=tuple(x for x in (proposal.operated_entity,) if x),
                       geometry_unknown=proposal.geometry_ambiguous)


class Consumer(TaskConstraints, OwnLeaseLifecycle, E101BaseConsumer):
    """Current task constraints, own lease lifetime and original E101 phase."""


def namespace():
    source_hashes()
    return SimpleNamespace(**{**vars(common), "Consumer": Consumer})
