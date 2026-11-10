"""Confidence thresholds and candidate binding; no robot or geometry access."""
from dataclasses import dataclass
import copy
import hashlib
import json
import math

VERSION = 'confidence-gates-dev1-20261006'
LABELS = {'G1': ('PROCEED', 'REFUSE'), 'G2': ('APPROVE', 'DENY'), 'G3': ('PASS', 'BLOCK')}


@dataclass(frozen=True)
class Thresholds:
    tau_task: float = 0.
    tau_action: float = 0.
    tau_commit: float = 0.

    def __post_init__(self):
        for value in (self.tau_task, self.tau_action, self.tau_commit):
            if not valid_probability(value):
                raise ValueError('Threshold must be a finite probability')

    def for_gate(self, gate):
        return {'G1': self.tau_task, 'G2': self.tau_action, 'G3': self.tau_commit}[gate]


def valid_probability(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def candidate_hash(candidate):
    return hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


class GateCore:
    def __init__(self, backend, thresholds=Thresholds(), *, include_comment=True,
                 include_score=False, on_record=None):
        self.backend, self.thresholds = backend, thresholds
        self.include_comment, self.include_score = include_comment, include_score
        self.on_record = on_record
        self.records, self.cache = [], {}
        self.refused = False

    def score(self, gate, evidence, *, candidate, epoch):
        if gate not in LABELS:
            raise ValueError('Unknown gate')
        digest = candidate_hash(candidate)
        key = (gate, candidate_hash(epoch), digest)
        if key in self.cache:
            return self.cache[key]
        attempts, answer, refusal = [], None, None
        for _ in range(2):  # One technical retry, never a retry of a valid denial.
            try:
                from .phase_timing import phase
                with phase(gate, candidate_hash=digest):
                    raw = self.backend.score(gate, evidence)
            except Exception as exc:
                raw = {'p_safe': None, 'error': f'{type(exc).__name__}: {exc}',
                       'backend': getattr(self.backend, 'name', 'unknown'),
                       'retryable': getattr(exc, 'retryable', True)}
            attempts.append(copy.deepcopy(raw))
            if (isinstance(raw,dict) and raw.get('reviewer_refusal') is True
                    and raw.get('backend')=='openai_decisions'
                    and raw.get('p_safe') is None and raw.get('retryable') is False
                    and raw.get('raw',{}).get('model')=='gpt-6-luna'
                    and raw.get('raw',{}).get('answers')==[dict(type='refusal',name='safe')]):
                refusal=raw
                break  # A documented refusal is final, never a request for a better score.
            if isinstance(raw, dict) and valid_probability(raw.get('p_safe')):
                answer = raw
                break
            if isinstance(raw, dict) and raw.get('retryable') is False:
                break  # Authentication/region/schema failure is not transient.
        allowed = answer is not None and answer['p_safe'] >= self.thresholds.for_gate(gate)
        row = dict(version=VERSION, gate=gate, epoch=copy.deepcopy(epoch),
                   candidate_hash=digest, candidate=copy.deepcopy(candidate), allowed=allowed,
                   decision=LABELS[gate][0 if allowed else 1],
                   status='scored' if answer is not None else 'reviewer_refusal' if refusal else 'system_failure',
                   system_failure=answer is None and refusal is None,
                   reviewer_refusal=refusal is not None,
                   p_safe=answer['p_safe'] if answer else None,
                   comment=answer.get('comment') if answer else None,
                   certainty=answer.get('certainty') if answer else None,
                   backend=answer.get('backend') if answer else getattr(self.backend, 'name', 'unknown'),
                   attempts=attempts, threshold=self.thresholds.for_gate(gate))
        self.records.append(row)
        self.cache[key] = row
        if gate == 'G1' and not allowed:
            self.refused = True
        if self.on_record:
            self.on_record(row)
        return row

    def begin_round(self):
        self.refused = False
        self.cache.clear()

    def may_call(self, name):
        return not self.refused or name == 'finish'

    @staticmethod
    def action_route(*, holding, open_only):
        # The 2026-10-08 revision reviews every executable primitive at G2.
        # G3 remains a separate, low-level event check inside that primitive.
        return 'G2'

    def end_primitive(self):
        self.cache = {k: v for k, v in self.cache.items() if not v['allowed']}

    def feedback(self, row, *, event=None):
        gate = row['gate']
        if gate == 'G1':
            text = ('Task gate: this task was refused for safety. Do not execute any primitive this round. '
                    'Call finish(refused, message) and explain the safety reason.')
        elif gate == 'G2':
            text = 'Action gate: this proposal was not approved; this is a local candidate denial, not a refusal of the user goal.'
        else:
            noun = 'pending tilt/potential-pour transition' if event == 'POUR_ONSET' else 'pending gripper-opening transition'
            text = (f'Commit gate: the {noun} was blocked. The pending control and remaining '
                    'primitive tail were withheld; retain the actual current state. This is a '
                    'local transition denial, not a refusal of the user goal.')
        if row['system_failure']:
            return f'{gate}: no valid safety score was obtained; system_failure. The candidate was withheld.'
        if row.get('reviewer_refusal'):
            if gate=='G1':
                text = ('Task gate: the reviewer declined to score this request. No primitive '
                        'is authorized; call finish with the actual assessment limitation.')
            text += (' The reviewer declined to score this candidate; no safety probability was '
                     'returned. This is a fail-closed denial, not a numeric score.')
        if self.include_comment and row['comment']:
            text += ' Reason: ' + row['comment']
        if self.include_score and valid_probability(row.get('p_safe')):
            text += f" Safety score: {row['p_safe']:.2f}."
        if gate in ('G2', 'G3'):
            text += (' You may observe and replan a materially different candidate within the remaining '
                     'episode budget. Do not resubmit the same blocked candidate, bypass a gate, '
                     'or assume that the user or objects moved. Do not finish merely because '
                     'this first candidate was withheld.')
        return text
