"""Reuse the production three gates and dispatcher with an explicit SI decoder.

This is a hardware *integration boundary*, not a commissioned controller.
Only injected transports can consume commands. No sockets, enable/reset,
simulation truth, VLA normalization, or assumed empty state are present here.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from types import MethodType, FunctionType

import numpy as np

from research.v4.development.confidence_gate_dev_20261006.core import Thresholds, candidate_hash
from .upstream_refusal import GateCore, review as upstream_review
from research.v4.development.confidence_gate_dev_20261006.evidence import EvidenceBuilder, plain
from research.v4.development.confidence_gate_dev_20261006.evidence import annotate as reference_annotate
from research.v4.development.confidence_gate_dev_20261006.runtime import ConfidenceAdapter, ConfidenceToolBoundary
from research.v4.leash_v43.dispatch import CommitDispatcher
from research.v4.leash_v43.registry import Command, GripperCalibration, HoldingEstimator, axis_rotation, rotation_matrix
from research.v3.bridge.b0 import _tool_result
from .hardware_registry import HardwareCommitRegistry, HardwareProprio
from research.v4.development.g3_event_count_repair_20261020.estimator import CompletedOpenHoldingEstimator
from research.v4.development.g3_transition_dev_20261019.scope import TransitionDispatcher, digest
from .thresholds import configured_thresholds

VERSION = 'dobot-official-rpent-port-v4-20261015'
CONTROL_SEMANTICS = {
    'version': VERSION,
    'control': '[base_dx_m,base_dy_m,base_dz_m,world_rotvec_x_rad,world_rotvec_y_rad,world_rotvec_z_rad,gripper_sign]',
    'gripper_sign': 'negative=OPEN,positive=CLOSE,zero=HOLD; driver must map to Jodell opening percent',
    'warning': 'explicit hardware SI commands; NOT Panda OSC or LIBERO-normalized VLA actions',
}


@dataclass(frozen=True)
class SIControlSemantics:
    def decode(self, payload, p):
        a = np.asarray(payload, dtype=float)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError('An explicit finite hardware SI control is required')
        if p.eef_xyz is None and np.any(a[:3]):
            raise ValueError('Translation needs current public base-frame EEF position')
        return Command(a.copy(), 'OPEN' if a[6] < 0 else 'CLOSE' if a[6] > 0 else 'HOLD',
            a[:3].copy(), a[3:6].copy(),
            None if p.eef_xyz is None else p.eef_xyz + a[:3],
            None if p.eef_rotation is None else axis_rotation(a[3:6]) @ p.eef_rotation)


def hardware_annotate(rgb, camera, robot, candidate):
    """Missing physical K/T means no overlay, not a fabricated camera transform."""
    if candidate.get("name") in ("move_to", "move_pose", "rotate_pitch", "rotate_wrist"):
        if "intrinsics" not in camera or "extrinsics" not in camera:
            return None, dict(drawn=False,
                reason="Same-frame qualified hardware K/T not supplied for this view; inspect raw public RGB and numeric command",
                execution_route=plain(candidate.get("execution_route")))
    return reference_annotate(rgb, camera, robot, candidate)


# Preserve the frozen capture implementation and its public-data projection;
# only hardware's optional calibrated annotation differs.
_capture_globals = dict(EvidenceBuilder.capture.__globals__, annotate=hardware_annotate)
_hardware_capture = FunctionType(EvidenceBuilder.capture.__code__, _capture_globals,
                                EvidenceBuilder.capture.__name__, EvidenceBuilder.capture.__defaults__)


class HardwareEvidence(EvidenceBuilder):
    def capture(self, packet, *args, **kwargs):
        from .public_sensors import reviewer_rgb_packet
        gate_packet = reviewer_rgb_packet(packet)
        result = _hardware_capture(self, gate_packet, *args, **kwargs)
        if getattr(self, 'shared_policy_text', None) is not None:
            result['shared_policy'] = self.shared_policy_text
            result['hardware_policy_provenance'] = self.policy_provenance
        result['reviewer_rgb_preprocessing'] = {
            name: camera['reviewer_rgb_preprocessing'] for name, camera in gate_packet['cameras'].items()}
        result['hardware_control_semantics'] = dict(CONTROL_SEMANTICS)
        if result['gate'] == 'G2':
            candidate = result['candidate']
            if candidate.get('name') == 'move_to' and not getattr(self, 'native_primitives', False):
                result['candidate_description'] = ('Move the declared EEF reference point to base XYZ '
                    + json.dumps(candidate['arguments']['xyz']) + ' metres; retain measured current orientation and gripper (HOLD). '
                    'Actual Dobot path qualification is separate; no simulator servo/OSC path or camera calibration is assumed.')
            else:
                result['candidate_description'] += ' Low-level gripper signs are interface symbols, not Panda motor units; see hardware_control_semantics.'
        return result


class HardwareAdapter:
    """Driver contract: read_state, capture, assert_stationary, send_si, close.

    send_si must acknowledge completion before returning; queued trajectories
    or partial combined arm/gripper sends do not satisfy this contract.
    An exception after send enters the reference consumption-unknown path.
    """
    _review = upstream_review

    def _project(self, **kwargs):
        request = ConfidenceAdapter._project(self, **kwargs)
        proposal = self.registry.pending_window
        if proposal is not None:
            scope = json.loads(json.dumps(proposal.descriptor))
            request['evidence']['event']['continuation']['reviewed_window'] = scope
            scope_hash = digest(scope)
            request['candidate']['transition_scope_sha256'] = scope_hash
            request['evidence']['candidate']['transition_scope_sha256'] = scope_hash
            request['evidence']['review_scope'] = dict(
                kind='pending_commit_with_explicit_finite_continuation',
                scored_field='candidate.control + event + event.continuation.reviewed_window',
                context_only_fields=['trusted_instruction', 'parent_primitive_context'],
                excluded_scope='all unlisted controls, later VLA chunks, new gripper phases and later primitives')
        return request

    def __init__(self, driver, *, output, backend, calibration, episode, method='Ours', review_timeout_s=180.):
        if method not in {'B0', 'Ours'}:
            raise ValueError('Only matched B0/Ours conditions are implemented')
        if not isinstance(calibration, GripperCalibration):
            raise ValueError('Explicit measured calibration or labelled mock fixture required')
        self.driver, self.episode, self.method = driver, episode, method
        self.round_id, self.instruction, self.history = 0, '', []
        self.core = GateCore(backend, configured_thresholds(getattr(driver, 'config', {})), include_comment=True, include_score=False)
        self.evidence_builder = HardwareEvidence(output)
        if method == 'Ours' and getattr(driver, 'config', {}).get('hardware_policy_revision'):
            from .hardware_policy import effective_policy
            from research.v4.development.confidence_gates_20261006.config import policy
            text, provenance = effective_policy(driver.config, policy())
            self.evidence_builder.shared_policy_text = text
            self.evidence_builder.policy_provenance = provenance
        self.estimator = CompletedOpenHoldingEstimator(calibration, capture_grasp_pose_on_close=True)
        self.registry = HardwareCommitRegistry(self.estimator, position_tolerance_m=.008,
            direction_tolerance_rad=.06, method='Ours', require_grasp_assumption=False,
            semantic_unknown_pour=True)
        self.semantics = SIControlSemantics()
        self.dispatcher = CommitDispatcher(registry=self.registry, decode=self.semantics.decode,
            read_proprio=self._proprio, send=self._consume, capture=self._capture,
            review=self._review, project=self._project, method='Ours', enabled=method=='Ours',
            review_timeout_s=review_timeout_s)
        self.dispatcher.run_chunk = MethodType(TransitionDispatcher.run_chunk, self.dispatcher)
        self.pending_primitive = self.pending_name = self.last_gate = None
        # No fake camera extrinsics or guaranteed OSC path annotation.
        self.candidate_preparer = lambda name, parameters: dict(route='dobot_si_single_outlet', path_known=False,
            reason='Dobot control port; simulator path overlay unavailable until calibrated hardware qualification')
        self.closed = False
        self._state_anchor = None

    def _state(self):
        state = self.driver.read_state(self.dispatcher.sequence)
        if state.get('executed_sequence') != self.dispatcher.sequence:
            raise ValueError('Driver executed sequence differs from dispatcher')
        return state

    def _proprio(self, sequence):
        if sequence != self.dispatcher.sequence:
            raise ValueError('Requested sequence is not the actual executed sequence')
        s = self._state()
        xyz = None if s.get('eef_pos') is None else np.asarray(s['eef_pos'], dtype=float)
        if xyz is not None and (xyz.shape != (3,) or not np.isfinite(xyz).all()):
            raise ValueError('Invalid measured EEF position')
        r = None if s.get('eef_rotation') is None else rotation_matrix(s['eef_rotation'])
        return HardwareProprio(s.get('gripper_aperture_m'), xyz, r, sequence,
            s.get('gripper_command_motion_complete'), s.get('gripper_motion_feedback_source'))

    def epoch(self):
        s = self._state()
        # Real feedback has encoder noise. A fixed anchor avoids reusing a
        # pre-review approval after drift while ignoring sub-resolution jitter.
        current = {k: plain(s.get(k)) for k in
            ('eef_pos', 'eef_rotation', 'gripper_aperture_m', 'fault', 'queued_commands')}
        changed = self._state_anchor is None
        if not changed:
            for key, tolerance in (('eef_pos', .00025), ('eef_rotation', .0005), ('gripper_aperture_m', .0005)):
                a, b = current[key], self._state_anchor[key]
                if (a is None) != (b is None) or (a is not None and np.max(np.abs(np.asarray(a)-np.asarray(b))) > tolerance):
                    changed = True
            changed = changed or any(current[k] != self._state_anchor[k] for k in ('fault', 'queued_commands'))
        if changed:
            self._state_anchor = current
        fingerprint = candidate_hash(self._state_anchor)
        return dict(episode=self.episode, round=self.round_id,
            executed_sequence=self.dispatcher.sequence, holding_generation=self.estimator.generation,
            hardware_state_fingerprint=fingerprint)

    def public_robot(self):
        p = self._proprio(self.dispatcher.sequence)
        return dict(eef_reference=self._state().get('eef_reference', 'grip_site'), eef_pos=plain(p.eef_xyz), eef_rotation=plain(p.eef_rotation),
            gripper_aperture_m=p.aperture_m, holding_estimate=self.estimator.kind.value,
            gripper_command_motion_complete=p.gripper_command_motion_complete,
            last_gripper_command=self.estimator.last_gripper)

    def _capture(self):
        self.driver.assert_stationary()
        packet = self.driver.capture(self.dispatcher.sequence)
        if packet.get('frame_id') != self.dispatcher.sequence:
            raise ValueError('Images must correspond to the current acknowledged state')
        return packet

    def evidence(self, gate, candidate, *, packet=None, event=None):
        packet = self._capture() if packet is None else packet
        return self.evidence_builder.capture(packet, self.public_robot(), self.instruction,
            gate, candidate, event=event, history=self.history)

    def _consume(self, payload):
        if self.closed:
            raise RuntimeError('Closed hardware adapter')
        self.driver.assert_stationary()
        result = self.driver.send_si(np.asarray(payload, dtype=float).copy())
        if not isinstance(result, dict) or result.get('acknowledged_complete') is not True:
            raise RuntimeError('No physical completion acknowledgement; consumption unknown')
        return result

    @contextmanager
    def primitive_scope(self, name, *, parameters=None, **options):
        with self.dispatcher.primitive(name=name, **options):
            yield self

    def step(self, action):
        # A blocking Dobot send is the known complete physical control.
        # Never label unexpanded normalized VLA actions as physical commands.
        return self.dispatcher.run_chunk([np.asarray(action, dtype=float).copy()])[0]

    def cancel(self, reason):
        self.dispatcher.request_owner_abort(reason)
        if hasattr(self.driver, 'cancel'):
            self.driver.cancel()

    def close(self):
        self.closed = True
        self.driver.close()


class PortTools:
    """Legacy integration fixture retained for its original regression tests.

    Production build_port uses NativePortTools and the official real-robot
    declarations. This helper also supplies the public RGB result serializer.
    """
    def __init__(self, adapter):
        self.adapter = adapter
        self.round_finished = False
        self.finish_result = None
        self.calls = []
        self.round_calls = 0
        self.memory_dir = Path(__file__).resolve().parents[2]/'upstream/RPent-reproduce-libero/resources/libero/memory'

    def begin_round(self):
        self.round_finished = False
        self.finish_result = None
        self.round_calls = 0

    def get_tools_spec(self):
        return [dict(name='move_to', description='Base-frame absolute EEF XYZ metres; keep current gripper unchanged.',
            input_schema=dict(type='object', properties={'xyz':{'type':'array','items':{'type':'number'},'minItems':3,'maxItems':3}}, required=['xyz'],additionalProperties=False)),
            dict(name='set_gripper', description='Stationary Jodell: negative=open, positive=close, zero=hold. No inferred grasp.',
                input_schema=dict(type='object',properties={'gripper':{'type':'number'}},required=['gripper'],additionalProperties=False)),
            dict(name='release', description='Stationary OPEN; reviewed at primitive and exact send boundaries.',input_schema=dict(type='object',properties={},additionalProperties=False)),
            dict(name='observe', description='Fresh public RGB and measured state; no motion.',input_schema=dict(type='object',properties={},additionalProperties=False)),
            dict(name='view_camera_meta',description='Current camera metadata and explicit candidate calibration status; no motion.',
                input_schema=dict(type='object',properties={},additionalProperties=False)),
            dict(name='back_project',description='Project a pixel in the last wrist RGBD to candidate base XYZ metres; calibration uncertainty remains explicit.',
                input_schema=dict(type='object',properties={'row':{'type':'integer','minimum':0},'col':{'type':'integer','minimum':0},
                    'plane_z_base_mm':{'type':'number'}},required=['row','col'],additionalProperties=False)),
            dict(name='read_text_file',description='Read unchanged frozen Global Memory only; numerical Panda recipes require their applicability conditions.',
                input_schema=dict(type='object',properties={'path':{'type':'string'}},required=['path'],additionalProperties=False)),
            dict(name='list_directory',description='List unchanged frozen Global Memory only.',
                input_schema=dict(type='object',properties={'path':{'type':'string'}},required=['path'],additionalProperties=False)),
            dict(name='finish',description='End current instruction round.',input_schema=dict(type='object',properties={'status':{'type':'string','enum':['done','refused','asked']},'message':{'type':'string'}},required=['status','message'],additionalProperties=False))]

    def observation_result(self, name, result):
        p = self.adapter._capture()
        images = [self.adapter.evidence_builder._save(p['cameras'][v]['rgb'],v) for v in ('agentview','robot0_eye_in_hand')]
        metadata = self.adapter.driver.planner_observation_metadata() if hasattr(self.adapter.driver, 'planner_observation_metadata') else {}
        return _tool_result(name, dict(result,robot_state=self.adapter.public_robot(),planner_geometry=metadata,image_paths=images,
            _image_cam_bytes=Path(images[0]).read_bytes(),_image_wrist_bytes=Path(images[1]).read_bytes()))

    def execute_tool(self, name, args):
        if name == 'finish':
            if args.get('status') not in {'done','refused','asked'} or not isinstance(args.get('message'),str):
                raise ValueError('Explicit actual terminal status/message required')
            self.round_finished = True
            self.finish_result = dict(args)
            return _tool_result(name, dict(_finish=True, **args))
        if name == 'observe':
            return self.observation_result(name,{})
        if name == 'view_camera_meta':
            return _tool_result(name,self.adapter.driver.planner_observation_metadata())
        if name == 'back_project':
            return _tool_result(name,self.adapter.driver.back_project(**args))
        if name in {'read_text_file','list_directory'}:
            path=Path(args['path'])
            path=(self.memory_dir/path if not path.is_absolute() else path).resolve()
            path.relative_to(self.memory_dir.resolve())
            if name=='list_directory':return _tool_result(name,dict(entries=sorted(p.name for p in path.iterdir())))
            if path.stat().st_size>256*1024:raise ValueError('Global Memory text exceeds the declared read bound')
            return _tool_result(name,dict(text=path.read_text(encoding='utf-8')))
        a = np.zeros(7)
        if name == 'move_to':
            if set(args) != {'xyz'}:
                raise ValueError('Move only XYZ; combined gripper/rotation control is not qualified')
            xyz = np.asarray(args['xyz'],dtype=float)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError('Finite base-frame XYZ metres required')
            a[:3] = xyz - self.adapter._proprio(self.adapter.dispatcher.sequence).eef_xyz
        elif name in {'set_gripper','release'}:
            value = -1. if name=='release' else args.get('gripper')
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('Finite gripper interface sign required')
            a[6] = value
        else:
            raise ValueError('Unsupported tool; no command sent')
        if self.round_calls >= 60 or len(self.calls) >= 180:
            return _tool_result(name,dict(error='Shared primitive budget reached; finish with the actual result.',candidate_dispatched=False))
        self.round_calls += 1
        self.calls.append(dict(name=name,arguments=plain(args),started_monotonic_s=time.monotonic()))
        if name == 'move_to' and hasattr(self.adapter.driver, 'preflight_translation'):
            preflight = self.adapter.driver.preflight_translation(a[:3].tolist())
            self.calls[-1]['hardware_preflight'] = plain(preflight)
            if preflight.get('accepted') is not True:
                if preflight.get('recovery_allowed') is not True:
                    raise RuntimeError(preflight.get('error', 'Physical preflight transport unavailable'))
                return _tool_result(name, dict(error=preflight['error'], hardware_preflight_rejected=True,
                    candidate_dispatched=False, consumption_unknown=False, recovery_allowed=True,
                    system_failure=False))
        return self.observation_result(name,self.adapter.step(a))

    def cancel_active_and_wait(self):
        self.adapter.cancel('Operator/owner cancelled the hardware task')

    def close(self):
        self.adapter.close()


class B0Boundary:
    """Same physical outlet and observations, with all LEASH gates disabled."""
    def __init__(self, base, adapter):self.base,self.adapter=base,adapter
    def __getattr__(self,name):return getattr(self.base,name)
    def begin_round(self, instruction='', *, round_id=None):
        self.adapter.instruction=instruction
        self.adapter.round_id=(self.adapter.round_id+1 if round_id is None else round_id)
        self.base.begin_round()
    def get_tools_spec(self):return self.base.get_tools_spec()
    def execute_tool(self,name,args):
        from research.v3.bridge.b0 import BUDGETED_ACTIONS
        from .official_primitives import OFFICIAL_ACTIONS
        if name not in BUDGETED_ACTIONS|OFFICIAL_ACTIONS:return self.base.execute_tool(name,args)
        start=self.adapter.dispatcher.sequence
        with self.adapter.primitive_scope(name,episode=self.adapter.episode,round_id=self.adapter.round_id,
                binding_version=VERSION,completion_controls=self.completion_controls(name,args)):
            result=self.base.execute_tool(name,args)
        self.adapter.history.append(dict(name=name,arguments=plain(args),executed_start=start,
            executed_end=self.adapter.dispatcher.sequence))
        return result
    def cancel_active_and_wait(self):return self.base.cancel_active_and_wait()
    def close(self):return self.base.close()


def build_port(driver, *, output, backend, calibration, episode, method='Ours', native_primitives=True, model=None):
    adapter = HardwareAdapter(driver, output=output, backend=backend, calibration=calibration,
        episode=episode, method=method)
    base = PortTools(adapter)
    controls = lambda *_: 1
    if native_primitives:
        from .native_primitives import NativePortTools
        base = NativePortTools(adapter, base, model=model)
        controls = base.completion_controls
        adapter.evidence_builder.native_primitives = True
    from .official_primitives import registered_boundary_class
    boundary_type=registered_boundary_class(ConfidenceToolBoundary) if native_primitives else ConfidenceToolBoundary
    boundary = (boundary_type(base, adapter, core=adapter.core, completion_controls=controls)
        if method=='Ours' else B0Boundary(base,adapter))
    if method == 'B0': boundary.completion_controls = controls
    return adapter, base, boundary
