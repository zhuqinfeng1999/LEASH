"""Thin bindings over existing public factory, E3 phase gate and typed outlet."""
import ast
import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from dataclasses import replace

from research.v4.accelerated.e3_gate_boundary import requires_cup_approach_review
from research.v4.accelerated.p3_gated_toolkit import proposal_for_p3
from research.v4.candidates.e1_02_plain_rpent_entry_20260927.factory import PlainRPentToolkit
from research.v4.candidates.e1_unknown_tilt_typed_20260927.outlet import (
    TypedContainerAdapter, commit_infrastructure_failure, FailureKind)
from research.v4.candidates.e1_02_plain_rpent_v2_20260927.signals import (
    MissingPublicEvidence as ParentMissing, initial_missing_has_other_fault)
from .config import CONFIG, INSTANCE
from . import public_names

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = __package__

def compiled(text, suffix, supplied):
    name = PACKAGE + '.' + suffix
    before = sys.modules.get(name)
    module = ModuleType(name)
    module.__file__ = str(Path(__file__)) + '[' + suffix + ']'
    module.__package__ = PACKAGE
    module.__dict__.update(supplied)
    sys.modules[name] = module
    try:
        exec(compile(text, module.__file__, 'exec'), module.__dict__)
    finally:
        if before is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = before
    module.selected_text = text
    module.selected_text_sha256 = hashlib.sha256(text.encode()).hexdigest()
    return module

def public_components():
    path = ROOT/'research/v4/candidates/e1_02_plain_rpent_entry_20260927/bindings.py'
    tree = ast.parse(path.read_text())
    # Keep the exact approved two-view factory and visible-cloud/EEF sweep.
    # No E1 descent rule is carried into E3.
    nodes = [n for n in tree.body if not (
        isinstance(n, ast.FunctionDef) and n.name == 'proposal')]
    tree.body = nodes
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'research.v4.candidates.e1_future_public_binding_20260927':
            node.module = PACKAGE
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'INSTANCE', 'SOURCE', 'TARGET'}:
                    node.value = ast.Constant({'INSTANCE': INSTANCE, 'SOURCE': 'vessel', 'TARGET': 'target'}[target.id])
        if isinstance(node, ast.Constant) and node.value == 'held cleaver full public box unavailable':
            node.value = 'held public vessel box unavailable'
    bound = compiled(ast.unparse(ast.fix_missing_locations(tree)), '_public', {})

    class CurrentPublicWorkspaceFactory(bound.PublicWorkspaceFactory):
        """Persist real acquisition/projection exceptions through gate denial.

        The existing factory consumes valid SAM misses internally. Exceptions
        which escape that exact factory are capture/calibration/SDK faults;
        no reason text, Scene ID, risk label or private geometry is classified.
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.public_faults = []
        def __call__(self):
            try:
                return super().__call__()
            except Exception as exc:
                self.public_faults.append({'kind': 'public_acquisition_or_projection_fault',
                    'boundary': 'own_current_public_workspace',
                    'exception_type': type(exc).__name__})
                raise

    def proposal(name, arguments, workspace, locate_vla_target):
        hold = workspace.evidence.robot.hold_estimate
        if hold not in {'held', 'empty'}:
            raise ValueError('public hold state unknown')
        held = hold == 'held'
        sweep = bound.ObservedPayloadSweep(workspace) if held else None
        result = proposal_for_p3(name, arguments, workspace, locate_vla_target,
                                 held_sweep=sweep)
        # Existing source AABB/EEF proximity logic, no material/grasp metadata.
        entity = 'vessel' if held else result.operated_entity
        if not held and name in {'move_to', 'move_pose', 'set_gripper'}:
            box = workspace.box('vessel')
            point = result.motion_waypoints[-1] if result.motion_waypoints else workspace.eef_xyz
            if box is not None:
                from math import dist
                nearest = tuple(min(max(p, lo), hi) for p, lo, hi in zip(point, box.lower, box.upper))
                if dist(point, nearest) <= .055:
                    entity = 'vessel'
        if not held and result.name == 'vla_act':
            entity = 'vessel' if name == 'pi0_pick' else entity
        result = replace(result, operated_entity=entity)
        if not held:
            return result
        path = result.motion_waypoints
        if result.name == 'vla_act' and result.vla_target_xyz is not None:
            path = (workspace.eef_xyz, result.vla_target_xyz)
        if len(path) >= 2:
            target = workspace.box('target')
            center = None if target is None else tuple((a+b)/2 for a,b in zip(target.lower,target.upper))
            radius = None if target is None else max((target.upper[0]-target.lower[0])/2,
                                                    (target.upper[1]-target.lower[1])/2)
            if requires_cup_approach_review(
                held_public=True, primitive_name='vla_act' if result.name == 'vla_act' else 'move_to',
                current_eef_xyz=path[0], proposed_eef_xyz=path[-1],
                cup_center_xyz=center, payload_reach_m=sweep.total_radius_m,
                cup_radius_m=radius):
                result = replace(result, geometry_ambiguous=True,
                    parameters=(*result.parameters, ('phase', 'held_vessel_target_approach_precritical')))
        elif result.name == 'vla_act':
            result = replace(result, geometry_ambiguous=True)
        return result

    path = ROOT/'research/v4/accelerated/e3_gated_toolkit.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (
        isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in {'proposal_for_e3', 'E3HeldSweep'})]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'research.v4.accelerated.e3_name_binding':
            node.module = PACKAGE + '.public_names'
    tool = compiled(ast.unparse(ast.fix_missing_locations(tree)), '_toolkit', {'proposal_for_e3': proposal})

    class OrdinaryE3Toolkit(tool.E3GatedToolkit):
        def operation_scope(self, *args, **kwargs):
            raise ValueError('No pick macro is registered in this common protocol')

    class MissingPublicEvidence(ParentMissing):
        def __init__(self, missing, perception):
            if not missing or any(x not in public_names.PROMPTS for x in missing):
                raise ValueError('invalid own E305 missing-entity cause')
            self.missing = tuple(sorted(set(missing)))
            self.perception = tuple(perception)
            Exception.__init__(self, 'Valid initial public capture had no mask for: ' + ', '.join(self.missing))

    return SimpleNamespace(PublicWorkspaceFactory=CurrentPublicWorkspaceFactory,
        ObservedPayloadSweep=bound.ObservedPayloadSweep, proposal=proposal,
        E3GatedToolkit=OrdinaryE3Toolkit, E3MotionReviewer=tool.E3MotionReviewer,
        PlainRPentToolkit=PlainRPentToolkit, TypedContainerAdapter=TypedContainerAdapter,
        MissingPublicEvidence=MissingPublicEvidence, public_module=bound, toolkit_module=tool)
