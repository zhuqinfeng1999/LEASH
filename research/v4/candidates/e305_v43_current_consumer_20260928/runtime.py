"""Own E305 public seams over existing qualified v4.3 coordination code.

No Scene, assist, object pose or private scorer is read in this module.
The independent pose measurement is not a runtime certificate.
"""
from dataclasses import replace
import ast
from types import FunctionType, SimpleNamespace

from research.v4.candidates.p102_v43_actual_consumer_20260928 import runtime as common
from research.v4.accelerated.e3_gate_boundary import requires_cup_approach_review

admit_run = common.admit_run
make_reviewer_adapter = common.make_reviewer_adapter
bind_owner_abort = common.bind_owner_abort
commit_report = common.commit_report
run_planner = common.run_planner


def public_acquisition(components):
    """Compile the actual OWN E305 acquisition body, without constructing it."""
    original = components.public_module.PublicWorkspaceFactory.__call__
    tree = ast.parse(components.public_module.selected_text)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "PublicWorkspaceFactory"]
    if len(classes) != 1:
        raise RuntimeError("own E305 public factory class changed")
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "__call__"]
    if len(methods) != 1:
        raise RuntimeError("own E305 acquisition method changed")
    function = methods[0]
    returns = [n for n in ast.walk(function) if isinstance(n, ast.Return)]
    if len(returns) != 1 or ast.unparse(returns[0].value) != (
            "PublicWorkspace(self.adapter.capture_commit_evidence(), eef, tuple(boxes))"):
        raise RuntimeError("own E305 public evidence return changed")
    returns[0].value = ast.parse("_public_workspace(self, packet, eef, tuple(boxes))", mode="eval").body
    namespace = dict(original.__globals__)
    namespace["_public_workspace"] = common._public_workspace
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 original.__code__.co_filename + "::E305-v43-public-return", "exec"), namespace)
    return namespace["__call__"]


def public_provider(components, scene, adapter, sam_client):
    """Use own current nouns and acquisition, never P102's workspace class."""
    acquire = public_acquisition(components)

    class OwnPublicWorkspace(components.PublicWorkspaceFactory):
        def __call__(self):
            try:
                return acquire(self)
            except Exception as exc:
                # Same own fault persistence, without falling through a legacy
                # adapter/latch evidence suffix or swallowing SDK/capture faults.
                self.public_faults.append({"kind": "public_acquisition_or_projection_fault",
                    "boundary": "own_current_public_workspace", "exception_type": type(exc).__name__})
                raise

    return OwnPublicWorkspace(scene, adapter, sam_client)


class Consumer(common.Consumer):
    """Reuse coordination/lease/projection, with own acquisition and E3 phase."""
    def __init__(self, *args, components, **kwargs):
        # The existing coordinator's only shape/name dependencies are explicit:
        # SOURCE='vessel' and the own compiled ObservedPayloadSweep. Its parser,
        # public projection, owned lease, bounds and geometry functions are reused.
        selected = SimpleNamespace(**vars(components), binding=SimpleNamespace(source="vessel"),
                                   bindings=components.public_module)
        original = common.Consumer.__init__
        namespace = dict(original.__globals__)
        namespace["public_provider"] = public_provider
        init = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
        init.__kwdefaults__ = original.__kwdefaults__
        init(self, *args, components=selected, **kwargs)
        self.own_components = components

    def resolve(self, name, parameters, workspace):
        resolved = super().resolve(name, parameters, workspace)
        proposal = resolved.proposal
        path = proposal.motion_waypoints
        if proposal.name == "vla_act" and proposal.vla_target_xyz is not None:
            path = (workspace.public.eef_xyz, proposal.vla_target_xyz)
        kind = self.adapter.estimator.kind.value.lower()
        if kind == "held" and len(path) >= 2:
            target = workspace.public.box("target")
            vessel = workspace.public.box("vessel")
            if target is not None and vessel is not None:
                sweep = self.own_components.ObservedPayloadSweep(workspace.public)
                center = tuple((a + b) / 2 for a, b in zip(target.lower, target.upper))
                radius = max((target.upper[0] - target.lower[0]) / 2,
                             (target.upper[1] - target.lower[1]) / 2)
                if requires_cup_approach_review(
                        held_public=True, primitive_name="vla_act" if proposal.name == "vla_act" else "move_to",
                        current_eef_xyz=path[0], proposed_eef_xyz=path[-1],
                        cup_center_xyz=center, payload_reach_m=sweep.total_radius_m,
                        cup_radius_m=radius):
                    proposal = replace(proposal, geometry_ambiguous=True,
                        parameters=(*proposal.parameters, ("phase", "held_vessel_target_approach_precritical")))
                    resolved = replace(resolved, proposal=proposal, geometry_unknown=True)
        # UNKNOWN remains semantic uncertainty and reaches gate2; no old
        # pre-gate 'public hold state unknown' throw or non-container bypass.
        return resolved
