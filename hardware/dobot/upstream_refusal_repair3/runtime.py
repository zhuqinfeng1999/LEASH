"""Model-only tool gates around the unchanged low-level commit dispatcher.

Scene access is confined to the public adapter. Neither GateCore nor the
evidence builder receives the adapter, toolkit, scene, or simulator.
"""
from contextlib import contextmanager
from copy import deepcopy
import time

import numpy as np

from research.v3.bridge.b0 import BUDGETED_ACTIONS, _tool_result
from research.v4.accelerated.r2_adapter import R2V42Adapter
from research.v4.leash_v43.adapter import OSCControlSemantics, V43PublicEnvAdapter
from research.v4.leash_v43.dispatch import CommitDispatcher, V43Intercepted
from research.v4.leash_v43.registry import CommitEventRegistry, HoldingEstimator
from .core import candidate_hash
from .evidence import plain

VERSION = 'confidence-production-boundary-dev3-semantic-g3-recovery-20261010'


class ConfidenceAdapter(V43PublicEnvAdapter):
    """Reuse the physical send/registry; replace the retired review projection."""
    def __init__(self, scene, *, instruction, core, evidence_builder, calibration,
                 episode, capture=None, semantics=None, on_sim_step=None,
                 review_timeout_s=1800., position_tolerance_m=.008,
                 orientation_tolerance_rad=.06):
        R2V42Adapter.__init__(self, scene, instruction=instruction, on_sim_step=on_sim_step)
        initial_sequence = scene.sequence
        if type(initial_sequence) is not int or initial_sequence < 0:
            raise ValueError('Scene requires its actual nonnegative executed sequence')
        self.core, self.evidence_builder = core, evidence_builder
        self.episode, self.round_id = episode, 0
        self.history = []
        self._capture = capture or scene.capture_agent
        self.semantics = semantics or OSCControlSemantics.from_scene(scene)
        self.estimator = HoldingEstimator(calibration, grasp_assumption='unverified',
                                          capture_grasp_pose_on_close=True)
        initial = getattr(scene, 'public_initial_occupancy', lambda: None)()
        if initial is not None:
            self.estimator.initialize_empty(initial, self._proprio(initial_sequence), episode=episode)
        self.registry = CommitEventRegistry(self.estimator,
            position_tolerance_m=position_tolerance_m,
            direction_tolerance_rad=orientation_tolerance_rad,
            method='Ours', require_grasp_assumption=False,
            semantic_unknown_pour=True)
        self.dispatcher = CommitDispatcher(registry=self.registry, decode=self.semantics.decode,
            read_proprio=self._proprio, send=self._consume, capture=self._capture,
            review=self._review, project=self._project, method='Ours', enabled=True,
            review_timeout_s=review_timeout_s)
        # Some accepted resets execute setup controls before the method starts.
        # Preserve their actual public frame ID without fabricating method
        # commands, holding evidence, or another episode reset.
        self.dispatcher.sequence = initial_sequence
        self.last_gate = None
        self.pending_primitive = None
        self.pending_name = None
        self.candidate_preparer = None

    def public_robot(self):
        p = self._proprio(self.dispatcher.sequence)
        raw = self.scene.raw
        robot = dict(eef_pos=None if p.eef_xyz is None else p.eef_xyz.tolist(),
            eef_rotation=None if p.eef_rotation is None else p.eef_rotation.tolist(),
            eef_quat=plain(raw.get('robot0_eef_quat')),
            gripper_qpos=plain(raw.get('robot0_gripper_qpos')),
            gripper_aperture_m=p.aperture_m, holding_estimate=self.estimator.kind.value,
            last_gripper_command=self.estimator.last_gripper)
        if self.estimator.initial_occupancy is not None:
            robot['initial_occupancy'] = deepcopy(self.estimator.initial_occupancy)
        return robot

    def epoch(self):
        return dict(episode=self.episode, round=self.round_id,
                    executed_sequence=self.dispatcher.sequence,
                    holding_generation=self.estimator.generation)

    def evidence(self, gate, candidate, *, packet=None, event=None):
        if packet is None:
            packet = self._capture()
        if packet.get('frame_id') != self.dispatcher.sequence:
            raise ValueError('Gate images and public executed sequence differ')
        return self.evidence_builder.capture(packet, self.public_robot(), self.instruction,
            gate, candidate, event=event, history=self.history)

    def _project(self, *, events, command, proprio, classification, evidence, context, lease):
        # Every component is public numeric data. No depth, object geometry,
        # lease conditions, or visual holding records are added.
        candidate = dict(name=self.pending_name,
            arguments=deepcopy(self.pending_primitive or {}),
            control=plain(command.payload), events=list(events))
        semantic_reference = {}
        if ('POUR_ONSET' in events and self.estimator.grasp_rotation is None):
            semantic_reference = dict(tilt_reference_available=False,
                trigger_basis='conservative possible pour; no executed grasp baseline',
                occupancy_confirmed=False)
        version = f'{self.episode}:{self.round_id}:{self.dispatcher.sequence}:{evidence["frame_id"]}'
        return dict(evidence=self.evidence('G3', candidate, packet=evidence,
                    event=dict(events=list(events), gripper=command.gripper,
                        translation_m=plain(command.translation),
                        rotation_world_axis_angle_rad=plain(command.rotation_delta),
                        current_tilt_deg=classification.current_tilt_deg,
                        candidate_tilt_deg=classification.candidate_tilt_deg,
                        maximum_path_tilt_deg=classification.maximum_path_tilt_deg,
                        release_requested=command.gripper == 'OPEN',
                        **semantic_reference,
                        continuation=dict(completion_controls=context.completion_controls,
                            position_tolerance_m=self.registry.position_tolerance_m,
                            direction_tolerance_rad=self.registry.direction_tolerance_rad,
                            scope='existing same-transition permit only; no later primitive approval'))),
                    candidate=candidate, epoch=self.epoch(),
                    events=tuple(events), evidence_version=version,
                    deadline_monotonic_ns=time.monotonic_ns()+int(self.dispatcher.review_timeout_s*1e9))

    def _review(self, request):
        row = self.core.score('G3', request['evidence'], candidate=request['candidate'],
                              epoch=request['epoch'])
        self.last_gate = row
        if row['system_failure']:
            # The existing dispatcher classifies ordinary RuntimeError as
            # infrastructure failure, withholding the exact control and tail.
            raise RuntimeError(self.core.feedback(row, event=request['events'][0]))
        if self.epoch() != request['epoch'] or candidate_hash(request['candidate']) != row['candidate_hash']:
            raise RuntimeError('Commit candidate or public state changed during review')
        reason = ('The reviewer declined to score; the pending control and tail were withheld.'
            if row.get('reviewer_refusal') else row.get('comment') or ('The valid safety score met the commit threshold.'
            if row['allowed'] else 'The valid safety score did not meet the commit threshold.'))
        return dict(decision=row['decision'], reason=reason,
                    events=request['events'], evidence_version=request['evidence_version'])


def pure_open(name, parameters):
    """Only the existing stationary release/set_gripper commands qualify."""
    if name == 'release':
        return True
    if name != 'set_gripper':
        return False
    value = parameters.get('gripper', -1.)
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and np.isfinite(value) and value < 0)


class ConfidenceToolBoundary:
    def __init__(self, base, adapter, *, core, completion_controls, max_denials=5):
        self.base, self.adapter, self.core = base, adapter, core
        self.completion_controls = completion_controls
        self.max_denials = max_denials
        self.denials = 0
        self.system_failure = None
        self.round_id = 0
        self.round_open = False
        self.logs = []
        self.counted_denials = set()
        self.last_local_denial = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def get_tools_spec(self):
        return self.base.get_tools_spec()

    def begin_round(self, instruction, *, round_id=None):
        if round_id is None:
            round_id = self.round_id + 1
        if (self.round_open or type(round_id) is not int or round_id <= self.round_id
                or not isinstance(instruction, str) or not instruction.strip()):
            raise ValueError('A new round requires completed prior round and increasing round ID')
        self.base.begin_round()
        self.round_id = round_id
        self.adapter.round_id = round_id
        self.adapter.instruction = instruction
        self.denials = 0
        self.counted_denials.clear()
        self.last_local_denial = None
        self.round_open = True
        self.core.begin_round()
        candidate = dict(instruction=instruction)
        try:
            evidence = self.adapter.evidence('G1', candidate)
            row = self.core.score('G1', evidence, candidate=candidate, epoch=self.adapter.epoch())
        except Exception as exc:
            self.system_failure = f'G1 public evidence unavailable: {type(exc).__name__}: {exc}'
            return dict(gate='G1', status='system_failure', system_failure=True,
                        allowed=False, decision='REFUSE', reason=self.system_failure)
        self.logs.append(row)
        if row['system_failure']:
            self.system_failure = self.core.feedback(row)
        # REFUSE never fabricates a finish call or the planner's final reason.
        return row

    def _deny(self, name, row, *, event=None):
        self.logs.append(row)
        if row.get('system_failure'):
            self.system_failure = self.core.feedback(row, event=event)
        else:
            key = (row['gate'], candidate_hash(row['epoch']), row['candidate_hash'])
            if key not in self.counted_denials:
                self.counted_denials.add(key)
                self.denials += 1
            self.last_local_denial = row
        feedback = self.core.feedback(row, event=event)
        remaining = max(0, self.max_denials - self.denials)
        feedback += f' Distinct local denials: {self.denials}/{self.max_denials}; remaining: {remaining}.'
        if self.denials >= self.max_denials:
            feedback += ' The existing denial limit was reached; call finish with the actual incomplete result.'
        return _tool_result(name, dict(error=feedback, gate=row['gate'],
            decision=row['decision'], system_failure=bool(row.get('system_failure')),
            candidate_dispatched=False, local_denials=self.denials,
            remaining_local_denials=remaining,
            recovery_allowed=not self.system_failure and remaining > 0))

    def execute_tool(self, name, input_dict):
        if name == 'finish':
            result = self.base.execute_tool(name, input_dict)
            if getattr(self.base, 'round_finished', False):
                self.round_open = False
            return result
        if not self.round_open:
            return _tool_result(name, dict(error='No open instruction round; no candidate was dispatched.',
                                           candidate_dispatched=False))
        if self.system_failure:
            return _tool_result(name, dict(error=self.system_failure, system_failure=True,
                                           candidate_dispatched=False))
        if not self.core.may_call(name):
            return _tool_result(name, dict(error=self.core.feedback(self.core.records[-1]),
                                           candidate_dispatched=False))
        if self.denials >= self.max_denials:
            return _tool_result(name, dict(error='Denial limit reached; only finish is available.',
                                           candidate_dispatched=False))
        if name not in BUDGETED_ACTIONS:
            return self.base.execute_tool(name, input_dict)
        # This single immutable copy is both judged and passed to the original
        # toolkit. No later lookup/reconstruction of the proposal is performed.
        parameters = deepcopy(input_dict)
        candidate = dict(name=name, arguments=parameters)
        try:
            controls = self.completion_controls(name, parameters)
            if self.adapter.candidate_preparer is not None:
                candidate['execution_route'] = self.adapter.candidate_preparer(name, parameters)
            elif name == 'move_to':
                candidate['execution_route'] = dict(route='original_rpent_servo',path_known=False,
                    reason='No prepared common execution route was supplied')
            epoch = self.adapter.epoch()
            bypass_g2 = self.core.action_route(holding=self.adapter.estimator.kind.value,
                open_only=pure_open(name, parameters)) == 'G3'
            row = None
            if not bypass_g2:
                evidence = self.adapter.evidence('G2', candidate)
                row = self.core.score('G2', evidence, candidate=candidate, epoch=epoch)
                if not row['allowed']:
                    return self._deny(name, row)
                if self.adapter.epoch() != epoch or candidate_hash(candidate) != row['candidate_hash']:
                    raise RuntimeError('Approved primitive candidate or state changed before dispatch')
            self.adapter.pending_primitive = parameters
            self.adapter.pending_name = name
            self.adapter.last_gate = None
            start = self.adapter.dispatcher.sequence
            gate_start = len(self.core.records)
            try:
                with self.adapter.primitive_scope(name, parameters=parameters,
                    completion_controls=controls, episode=self.adapter.episode,
                    round_id=self.round_id, binding_version=VERSION, lease=None):
                    result = self.base.execute_tool(name, parameters)
            except V43Intercepted as exc:
                outcome = exc.outcome
                # Original RPent often catches interception and returns a tool
                # error. This branch preserves the same result when it escapes.
                result = _tool_result(name, dict(error=outcome['reason'], **outcome))
            outcome = self.adapter.dispatcher.last_outcome
            if outcome is not None:
                last = self.adapter.last_gate
                if last is not None and last['gate'] == 'G3' and not last['allowed']:
                    result = self._deny(name, last, event=(outcome.get('events') or [None])[0])
                else:
                    # A prior score may have passed before a later command
                    # lacks technical state or fails revalidation. Preserve
                    # that actual stop instead of inventing a model BLOCK.
                    if (outcome.get('root_cause') == 'technical_unresolved'
                            and not outcome.get('system_failure')):
                        result = self._technical_stop(name, parameters, outcome)
                    else:
                        result = _tool_result(name, {**outcome, 'error': outcome['reason'], 'gate': 'G3'})
                    if outcome.get('system_failure'):
                        self.system_failure = outcome['reason']
            end = self.adapter.dispatcher.sequence
            if end > start:
                self.adapter.history.append(dict(name=name, arguments=plain(parameters),
                    executed_start=start, executed_end=end,
                    gate_outcomes=[{k:g[k] for k in ('gate','decision','status','comment')}
                        for g in (([row] if row is not None else []) + self.core.records[gate_start:])]))
            return result
        except Exception as exc:
            self.system_failure = f'Primitive admission failure: {type(exc).__name__}: {exc}'
            return _tool_result(name, dict(error=self.system_failure, system_failure=True,
                                           candidate_dispatched=False))
        finally:
            self.adapter.pending_primitive = None
            self.adapter.pending_name = None
            self.core.end_primitive()

    def cancel_active_and_wait(self):
        return self.base.cancel_active_and_wait()

    def _technical_stop(self, name, parameters, outcome):
        """Expose a real telemetry gap without inventing a model decision."""
        epoch = self.adapter.epoch()
        digest = candidate_hash(dict(name=name, arguments=plain(parameters),
            events=outcome.get('events', []), reason=outcome['reason']))
        row = dict(gate='G3', status='technical_unresolved', decision='UNRESOLVED',
            system_failure=False, candidate_hash=digest, epoch=epoch,
            reason=outcome['reason'], events=list(outcome.get('events', [])))
        key = ('G3', candidate_hash(epoch), digest)
        if key not in self.counted_denials:
            self.counted_denials.add(key)
            self.denials += 1
        self.last_local_denial = row
        self.logs.append(row)
        remaining = max(0, self.max_denials - self.denials)
        feedback = ('Commit gate: required current technical evidence is unavailable. '
            'This is not a model safety rejection or a rejection of the user goal. '
            'The pending control and remaining primitive tail were withheld. '
            'Missing evidence: '+outcome['reason']+'. '
            'Use permitted observations and replan a materially different supported candidate; '
            'do not repeat the same unresolved command, fabricate holding evidence, '
            'or bypass a gate. Preserve the actual state and remaining budgets. '
            f'Distinct local denials: {self.denials}/{self.max_denials}; remaining: {remaining}.')
        if remaining == 0:
            feedback += ' The existing denial limit was reached; finish with the actual incomplete result.'
        return _tool_result(name, {**outcome, 'error': feedback, 'gate': 'G3',
            'technical_state_missing': True, 'local_denials': self.denials,
            'remaining_local_denials': remaining,
            'recovery_allowed': remaining > 0 and not self.system_failure})

    def close(self):
        return self.base.close()
