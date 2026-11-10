"""Own full-corner admission callbacks over the existing V43 outlet.

No executor or holding estimator is introduced. Namespace preparation binds
the original constructor to the qualified supplied-client provider. Physical
execution and current task/lease lifecycle remain in their reviewed parents.
"""
import ast
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research.v4.candidates.p102_v43_actual_consumer_20260928 import runtime as common
from research.v4.candidates.v43_task_constraints_binding_20260928.runtime import TaskConstraints
from research.v4.candidates.v43_own_lease_lifecycle_20260928.runtime import OwnLeaseLifecycle
from research.v4.candidates.p401_v43_actual_client_provider_20260929 import binding as provider_binding
from research.v4.candidates.p401_v43_same_packet_reader_20260929.public import function
from research.v4.candidates.r202_v43_public_fault_types_20260929.public import PublicReaderTechnicalFault

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
VERSION = "P401-own-task-lease-full-corner-Consumer-source-v1"
FILES = ("__init__.py", "runtime.py", "check_source.py", "contract.json", "HANDOFF.md")


def digest(path):
    return provider_binding.digest(path)


def sources():
    contract = json.loads((HERE / "contract.json").read_text())
    pins = {}
    for name, expected in contract["parent_manifests"].items():
        path = ROOT / name
        if digest(path) != expected:
            raise RuntimeError("Exact qualified own Consumer parent changed: " + name)
        value = json.loads(path.read_text())
        for key, member in value.get("sha256", value).items():
            if key in pins and pins[key] != member:
                raise RuntimeError("Complete own Consumer source obligations conflict: " + key)
            pins[key] = member
        pins[name] = expected
    pins.update(contract["qualified_receipts"])
    for name in FILES:
        path = HERE / name
        pins[str(path.relative_to(ROOT))] = digest(path)
    for name, expected in pins.items():
        path = Path(name) if name.startswith("/") else ROOT / name
        if digest(path) != expected:
            raise RuntimeError("Whole own P401 Consumer source changed: " + name)
    return dict(sorted(pins.items()))


_archival_source_hashes = function(provider_binding.source_hashes, dict(
    provider_binding.source_hashes.__globals__, HERE=HERE, VERSION=VERSION, sources=sources))


def source_hashes(*args, **kwargs):
    from research.v4.leash_v43.run_manifest import source_custody_enabled
    if not source_custody_enabled():
        return {}
    return _archival_source_hashes(*args, **kwargs)


def _provider(components, scene, adapter, sam_client):
    source_hashes()
    return components.own_provider.public_provider(scene, adapter, sam_client)


def geometry_callback(original_source, original_module, evidence_dir):
    """Select only the saved original mathematical prefix, never its executor."""
    tree = ast.parse(original_source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "P4ActionBoundary")
    execute = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "execute_tool")
    start = next(i for i, n in enumerate(execute.body) if ast.unparse(n) == "robot = packet['robot_state']")
    end = next(i for i, n in enumerate(execute.body) if ast.unparse(n) == "self.geometry_workspace = transformed")
    prefix = deepcopy(execute.body[start:end])
    provider_binding.parent.write(Path(evidence_dir) / "own_geometry_prefix_before_guard.json", {
        "original_method": provider_binding.parent.code_row(original_module.P4ActionBoundary.execute_tool),
        "source_sha256": provider_binding.parent.sha256(original_source.encode()).hexdigest(),
        "start": start, "end": end, "prefix_ast": ast.dump(ast.Module(body=prefix, type_ignores=[])),
        "configuration_box": provider_binding.parent.code_row(original_module.configuration_box)})
    if not prefix or not any(ast.unparse(n).startswith("if held and (not zero_motion):") for n in prefix):
        raise RuntimeError("Exact original full-corner configuration-space prefix changed")
    # This label records public envelope provenance, not a latch observation.
    labels = [n for n in ast.walk(ast.Module(body=prefix, type_ignores=[]))
              if isinstance(n, ast.Constant) and n.value == "latch_public_payload_envelope"]
    if len(labels) != 1:
        raise RuntimeError("Exact original public envelope record changed")
    labels[0].value = "executed_public_payload_envelope"
    selected = deepcopy(execute)
    selected.name = "own_configuration"
    selected.args = ast.parse("def own_configuration(self, name, arguments, public, packet): pass").body[0].args
    selected.body = ast.parse("lo = hi = None").body + prefix + ast.parse(
        "return SimpleNamespace(public=transformed, lower=lo, upper=hi)").body
    selected.decorator_list = []
    module_tree = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    namespace = dict(vars(original_module), SimpleNamespace=SimpleNamespace)
    exec(compile(module_tree, str(HERE / "runtime.py") + "::own-original-geometry", "exec"), namespace)
    callback = namespace["own_configuration"]
    inverse = deepcopy(prefix)
    next(n for n in ast.walk(ast.Module(body=inverse, type_ignores=[]))
         if isinstance(n, ast.Constant) and n.value == "executed_public_payload_envelope").value = "latch_public_payload_envelope"
    callback.original_prefix_ast_exact = ast.dump(ast.Module(body=inverse, type_ignores=[])) == ast.dump(
        ast.Module(body=execute.body[start:end], type_ignores=[]))
    if not callback.original_prefix_ast_exact:
        raise RuntimeError("Original own corner/rotation/.8/.035 geometry changed")
    callback.selected_source = ast.unparse(module_tree)
    return callback


def _initialize_geometry(consumer):
    if not hasattr(consumer, "payload_local_corners"):
        consumer.payload_local_corners = consumer.pending_payload = None
        consumer.last_unheld_span = None
        consumer.payload_events = []
        consumer._pending_sequence = None
        consumer._payload_generation = None


def _pose(packet, *, required):
    try:
        q = np.asarray(packet["robot_state"]["eef_quat"], float)
        if q.shape != (4,) or not np.isfinite(q).all() or not .9 < np.linalg.norm(q) < 1.1:
            raise ValueError("Required public EEF quaternion malformed")
        return True
    except (KeyError, TypeError, ValueError) as exc:
        if required:
            raise PublicReaderTechnicalFault({"stage": "own_payload_geometry", "fault": True,
                "error_type": type(exc).__name__, "error": str(exc)}) from exc
        return False  # Optional preclosure geometry supplies no holding claim.


def _executed_stationary_close(consumer):
    start = consumer._pending_sequence
    end = consumer.adapter.dispatcher.sequence
    if start is None or end <= start:
        return False
    rows = [row for row in consumer.adapter.dispatcher.logs
            if row.get("kind") == "dispatched" and start < row["executed_sequence"] <= end]
    if [row["executed_sequence"] for row in rows] != list(range(start + 1, end + 1)):
        return False
    # Only already-consumed complete control vectors can retain a preclosure
    # envelope. A denied candidate, render or planned command contributes none.
    return all(len(row["payload"]) == 7 and all(x == 0 for x in row["payload"][:6])
               and row["payload"][6] > 0 for row in rows)


def _configuration(consumer, name, parameters, workspace):
    _initialize_geometry(consumer)
    kind, generation = consumer._holding()
    zero_motion = name in {"release", "set_gripper", "hold"}
    public, packet = workspace.public, workspace.packet
    visual = getattr(consumer.adapter.estimator,'visual_record',None)
    if kind == 'HELD' and visual is not None:
        if zero_motion:
            return SimpleNamespace(public=public,lower=None,upper=None)
        if not visual['full_extent_supported'] or visual['relative_corners'] is None:
            return None
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_quat(np.asarray(packet['robot_state']['eef_quat'])).as_matrix()
        offsets = np.asarray(visual['relative_corners']) @ rotation.T
        uncertainty = visual['relative_uncertainty_m']
        lo = np.minimum(offsets.min(axis=0)-uncertainty,-.035)
        hi = np.maximum(offsets.max(axis=0)+uncertainty,.035)
        if name in {'rotate_wrist','rotate_pitch','move_pose','pi0_pick','pi0_doubled'}:
            radius = max(float(np.linalg.norm(offsets,axis=1).max())+uncertainty,.035)
            lo,hi = np.full(3,-radius),np.full(3,radius)
        return SimpleNamespace(public=public,lower=lo,upper=hi)
    if kind != "HELD":
        consumer.payload_local_corners = None
        consumer._payload_generation = None
    if kind == "UNKNOWN":
        consumer.pending_payload = None
        consumer._pending_sequence = None
        return None if not zero_motion else SimpleNamespace(public=public, lower=None, upper=None)
    if kind == "EMPTY":
        source = public.box(consumer.components.binding.source)
        if source is not None:
            consumer.last_unheld_span = np.asarray(source.upper) - np.asarray(source.lower)
        consumer.pending_payload = None
        consumer._pending_sequence = None
        if name == "set_gripper" and parameters.get("gripper", -1) > 0 and _pose(packet, required=False):
            consumer.components.own_configuration(consumer, name, parameters, public, packet)
            consumer._pending_sequence = consumer.adapter.dispatcher.sequence
        # Preserve the original empty-gripper .035 expansion; no object pose
        # is necessary for this envelope or an unrelated no-motion command.
        return SimpleNamespace(public=public, lower=np.full(3, -.035), upper=np.full(3, .035))
    if consumer._payload_generation != generation:
        consumer.payload_local_corners = None
        if not _executed_stationary_close(consumer):
            consumer.pending_payload = None
        consumer._payload_generation = generation
    if zero_motion:
        return SimpleNamespace(public=public, lower=None, upper=None)
    _pose(packet, required=True)
    return consumer.components.own_configuration(consumer, name, parameters, public, packet)


def _missing_payload(self, name):
    self.capabilities.append({"kind": "missing_own_public_full_corner_envelope",
                              "tool": name, "system_failure": False})
    return None


def resolve(self, name, parameters, workspace):
    source_hashes()
    self.public_workspace = workspace.public
    configuration = _configuration(self, name, parameters, workspace)
    workspace.own_configuration = configuration
    proposal = self.components.own_proposal(self, name, parameters, workspace.public,
                                            self.provider.locate_vla_target)
    proposal = replace(proposal, geometry_ambiguous=proposal.geometry_ambiguous
                       or self._holding()[0] == "UNKNOWN" or configuration is None)
    self.proposals.append({"name": name, "arguments": parameters, "proposal": asdict(proposal),
                           "holding_source": "executed_records_and_public_proprio"})
    return common.ResolvedPrimitive(workspace.version, proposal,
        entities=tuple(x for x in (proposal.operated_entity,) if x),
        geometry_unknown=proposal.geometry_ambiguous)


def geometry(self, resolved, workspace, lease):
    source_hashes()
    proposal = resolved.proposal
    if proposal.name in {"release", "set_gripper", "hold"} and not proposal.motion_waypoints:
        return common.GeometryDecision(True, "own no-motion call; full-vector commit outlet remains active")
    configuration = workspace.own_configuration
    if configuration is None:
        return common.GeometryDecision(False, "Current own public full-corner envelope is unavailable")
    path = proposal.motion_waypoints
    if proposal.name == "vla_act":
        path = (workspace.public.eef_xyz, proposal.vla_target_xyz) if proposal.vla_target_xyz else ()
    if not path or any(point is None for point in path):
        return common.GeometryDecision(False, "Allowed own public motion target is unavailable")
    for region in lease.get("no_go_regions", ()):
        box = workspace.public.box(region) if isinstance(region, str) else common.PublicBox(
            region["name"], tuple(region["lower_xyz"]), tuple(region["upper_xyz"]))
        if box is None:
            return common.GeometryDecision(False, "Reviewer-bound own public entity is not localized")
        expanded = self.components.configuration_box(box, configuration.lower, configuration.upper)
        if any(common._segment_hits_box(a, b, expanded) for a, b in zip(path, path[1:])):
            return common.GeometryDecision(False, "Own configuration-space path intersects a validated no-go region")
        if proposal.placement_xyz is not None and expanded.contains(proposal.placement_xyz):
            return common.GeometryDecision(False, "Own configuration-space placement intersects a validated no-go region")
    return common.GeometryDecision(True, "Original own eight-corner configuration-space geometry passed")


def namespace(*, evidence_dir):
    admission = source_hashes()
    # Preserve each original provider/selector CodeType through private globals;
    # the actual future acquisition and constructor both read this whole scope.
    reader_factory = function(provider_binding._reader_for_module, dict(
        provider_binding._reader_for_module.__globals__, source_hashes=source_hashes))
    supplied = function(provider_binding._supplied_reader, dict(
        provider_binding._supplied_reader.__globals__, _reader_for_module=reader_factory))
    selected_namespace = function(provider_binding.namespace, dict(
        provider_binding.namespace.__globals__, HERE=HERE, source_hashes=source_hashes,
        _supplied_reader=supplied))
    selected = selected_namespace(evidence_dir=evidence_dir)
    source = (Path(evidence_dir) / "original_public_module.py.txt").read_text()
    callback = geometry_callback(source, selected.original_module, evidence_dir)
    (Path(evidence_dir) / "selected_own_geometry.py.txt").write_text(callback.selected_source)
    components = SimpleNamespace(**vars(selected.original_components), own_provider=selected,
        own_configuration=callback, configuration_box=selected.configuration_box,
        own_proposal=selected.P4ActionBoundary._proposal)
    constructor = function(common.Consumer.__init__, dict(
        common.Consumer.__init__.__globals__, public_provider=_provider))
    base = type("P401OwnBaseConsumer", (common.Consumer,), {"__init__": constructor,
        "resolve": resolve, "geometry": geometry, "_missing_payload": _missing_payload})
    consumer = type("P401OwnTaskLeaseConsumer", (TaskConstraints, OwnLeaseLifecycle, base), {})
    source_hashes()
    return SimpleNamespace(**{**vars(common), "Consumer": consumer, "components": components,
        "provider_selection": selected, "provider_namespace": selected_namespace,
        "source_hashes": source_hashes, "source_admission": admission,
        "runtime_qualified": False, "native_ready": False, "launch_authorized": False,
        "policy": None, "calibration": None, "grasp_assumption": "unverified",
        "vla_chunk_controls": None})
