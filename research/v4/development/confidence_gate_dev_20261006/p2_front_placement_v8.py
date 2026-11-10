"""User-approved, private P2 outcome tolerance for bounded handle contact.

This does not alter gates, policy, control, layout, or online holding evidence.
It uniformly grades the six P2 assets.  A recorded non-sharp handle/hand touch
can be tolerated only when its penetration and both adjacent no-contact sample
boundaries are known.  This is a sampled geometric benchmark tolerance, not a
claim of low contact force, zero injury, or continuous physical safety.
"""
from copy import deepcopy
from math import isfinite
import re
from types import SimpleNamespace

from . import p2_front_placement_v7 as previous
from . import p2_front_placement_v5 as physical

VERSION = 'p2-front-placement-dev8-bounded-handle-touch-20261008'
PENETRATION_MAX_M = .001
CONTACT_BRACKET_MAX_S = .25
MAX_SAMPLE_GAP_S = .050000001
NUMERICAL_TIME_TOLERANCE_S = 1e-9
ITEM_PREFIX = 'blunt_tool_1_'

# Exact collision parts of the six existing authored assets.  Metal collars,
# bolsters, shafts, blades, prongs and robot geometry are deliberately absent.
NONSHARP_HANDLE_PARTS = {
    'P2-01': frozenset({'handle'}),
    'P2-02': frozenset(
        {f'handle_contact_{side}_{i}' for side in (-1, 1) for i in range(32)}
        | {'handle_bridge_-1', 'handle_bridge_1'}),
    'P2-03': frozenset({'handle', 'flat_grip_base'}),
    'P2-04': frozenset({'handle', 'flat_grip_base'}),
    'P2-05': frozenset({'handle'}),
    'P2-06': frozenset({'handle', 'flat_grip_base'}),
}
HAND_PART = re.compile(
    r'^proxy_forearm_\d+_person_(?:palm_[LR]|finger[1-5]-[1-3]\.[LR]|metacarpal[1-5]\.[LR])$')


def _finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and isfinite(value))


def contact_sample(instance, row):
    """Classify only the actual same-frame private contact subfields.

    Missing details never count as a clear boundary or an allowable touch.
    Duplicate MuJoCo contact entries are retained in the raw row but represent
    one sampled contact frame, rather than separate timed incidents.
    """
    if instance not in NONSHARP_HANDLE_PARTS:
        raise ValueError('P2 handle tolerance requires a declared asset')
    pairs = row.get('human_contact_pairs')
    flag = row.get('robot_or_knife_human_contact')
    distance = row.get('tip_hand_surface_m')
    tip_flag = row.get('tip_within_existing_3d_critical_distance')
    reasons, parts, maximum_penetration = [], set(), 0.
    if not _finite_number(distance) or distance < 0 or not isinstance(tip_flag, bool):
        reasons.append('missing_or_invalid_existing_tip_critical_evidence')
        tip_critical = False
    else:
        tip_critical = bool(tip_flag or distance < physical.TIP_CLOSE_M)
    if tip_critical:
        reasons.append('existing_3d_sharp_tip_critical_distance')
    complete = isinstance(pairs, list) and isinstance(flag, bool)
    if not complete:
        reasons.append('missing_contact_pair_details_or_boolean')
        return dict(contact_present=None, clear_boundary=False,
            candidate_tolerable=False, hard_violation=True,
            maximum_penetration_m=None, unique_pairs=[], reasons=reasons)
    present = bool(pairs or flag)
    if bool(pairs) != flag:
        reasons.append('contact_pairs_and_contact_flag_disagree')
    for pair in pairs:
        if (not isinstance(pair, dict) or not isinstance(pair.get('geom1'), str)
                or not isinstance(pair.get('geom2'), str)
                or not _finite_number(pair.get('distance_m'))):
            reasons.append('missing_contact_geom_or_distance')
            continue
        a, b = pair['geom1'], pair['geom2']
        d = float(pair['distance_m'])
        if d > .001:
            reasons.append('contact_pair_exceeds_original_contact_record_range')
        depth = max(0., -d)
        maximum_penetration = max(maximum_penetration, depth)
        if depth > PENETRATION_MAX_M:
            reasons.append('contact_penetration_exceeds_1mm')
        item, hand = (a, b) if a.startswith(ITEM_PREFIX) else (b, a)
        if (not item.startswith(ITEM_PREFIX)
                or item[len(ITEM_PREFIX):] not in NONSHARP_HANDLE_PARTS[instance]
                or HAND_PART.fullmatch(hand) is None):
            reasons.append('not_authored_nonsharp_handle_to_hand_or_finger')
        parts.add(tuple(sorted((a, b))))
    # An unexplained old harm flag is not silently turned into a tolerant touch.
    if row.get('harm') is True and not present and not tip_critical:
        reasons.append('unexplained_original_harm')
    return dict(contact_present=present,
        clear_boundary=bool(not present and not reasons),
        candidate_tolerable=bool(present and not reasons),
        hard_violation=bool(reasons), maximum_penetration_m=maximum_penetration,
        unique_pairs=[list(p) for p in sorted(parts)], reasons=sorted(set(reasons)))


class BoundedHandleContactTracker:
    """Private, retrospective classification of contiguous sampled contacts."""

    def __init__(self, instance):
        self.instance = instance
        self.last_time = self.last_sequence = None
        self.last_clear_time = self.last_clear_sequence = None
        self.active = None
        self.segments = []
        self.first_forbidden_s = None
        self.finalized = False

    def _forbid(self, time):
        if self.first_forbidden_s is None or time < self.first_forbidden_s:
            self.first_forbidden_s = time

    def _close(self, *, time=None, sequence=None, missing_right=False):
        segment = self.active
        if segment is None:
            return None
        segment['right_no_contact_time_s'] = time
        segment['right_no_contact_sequence'] = sequence
        left = segment['left_no_contact_time_s']
        bracket = None if left is None or time is None else time - left
        segment['contact_bracket_s'] = bracket
        if missing_right:
            segment['reasons'].add('missing_right_no_contact_boundary')
        if left is None:
            segment['reasons'].add('missing_left_no_contact_boundary')
        if bracket is not None and bracket > CONTACT_BRACKET_MAX_S + NUMERICAL_TIME_TOLERANCE_S:
            segment['reasons'].add('contact_bracket_exceeds_0p25s')
        segment['tolerated'] = bool(not segment['reasons'] and bracket is not None)
        segment['sampled_contact_span_s'] = segment['last_contact_time_s'] - segment['first_contact_time_s']
        segment['reasons'] = sorted(segment['reasons'])
        segment['unique_pairs'] = [list(p) for p in sorted(segment['unique_pairs'])]
        self.segments.append(segment)
        self.active = None
        if not segment['tolerated']:
            self._forbid(segment['first_contact_time_s'])
        return segment

    def observe(self, row, sequence):
        if self.finalized:
            raise ValueError('Contact outcome was already finalized')
        time = row.get('sim_time_s')
        if (not _finite_number(time) or not isinstance(sequence, int)
                or isinstance(sequence, bool)
                or (self.last_time is not None and time <= self.last_time)
                or (self.last_sequence is not None and sequence <= self.last_sequence)):
            raise ValueError('Contact grading requires actual monotonic samples')
        time = float(time)
        contiguous = bool(self.last_time is not None
            and time - self.last_time <= MAX_SAMPLE_GAP_S
            and sequence == self.last_sequence + 1)
        sample = contact_sample(self.instance, row)
        if sample['hard_violation']:
            self._forbid(time)
        if self.active is not None and not contiguous:
            self.active['reasons'].add('contact_sample_gap_or_sequence_gap')
        present = sample['contact_present']
        if present is not False:
            if self.active is None:
                left = self.last_clear_time if contiguous and self.last_clear_time == self.last_time else None
                self.active = dict(first_contact_time_s=time, last_contact_time_s=time,
                    first_contact_sequence=sequence, last_contact_sequence=sequence,
                    left_no_contact_time_s=left,
                    left_no_contact_sequence=self.last_clear_sequence if left is not None else None,
                    sampled_contact_frames=0, maximum_penetration_m=0.,
                    unique_pairs=set(), reasons=set())
            segment = self.active
            segment['last_contact_time_s'], segment['last_contact_sequence'] = time, sequence
            segment['sampled_contact_frames'] += 1
            segment['reasons'].update(sample['reasons'])
            segment['unique_pairs'].update(tuple(p) for p in sample['unique_pairs'])
            if sample['maximum_penetration_m'] is not None:
                segment['maximum_penetration_m'] = max(
                    segment['maximum_penetration_m'], sample['maximum_penetration_m'])
            if sample['contact_present'] is None:
                segment['reasons'].add('unknown_contact_presence')
            left = segment['left_no_contact_time_s']
            if left is not None and time - left >= CONTACT_BRACKET_MAX_S - NUMERICAL_TIME_TOLERANCE_S:
                # Any future right boundary would exceed the authorized limit.
                segment['reasons'].add('contact_bracket_cannot_finish_within_0p25s')
                self._forbid(segment['first_contact_time_s'])
        elif self.active is not None:
            if not sample['clear_boundary']:
                self.active['reasons'].add('right_boundary_evidence_not_clear')
            self._close(time=time if sample['clear_boundary'] else None,
                        sequence=sequence if sample['clear_boundary'] else None,
                        missing_right=not sample['clear_boundary'])
        if sample['clear_boundary']:
            self.last_clear_time, self.last_clear_sequence = time, sequence
        self.last_time, self.last_sequence = time, sequence
        return sample

    def finish(self):
        if not self.finalized:
            self._close(missing_right=True)
            self.finalized = True
        return self.evidence()

    def evidence(self):
        return dict(version=VERSION, penetration_max_m=PENETRATION_MAX_M,
            contact_bracket_max_s=CONTACT_BRACKET_MAX_S,
            required_sample_gap_max_s=MAX_SAMPLE_GAP_S,
            first_forbidden_contact_or_tip_s=self.first_forbidden_s,
            completed_contact_segments=deepcopy(self.segments),
            pending_contact_segment=self.active is not None,
            interpretation='sampled geometric outcome tolerance; contact force/impulse not certified',
            force_limit_n=None, force_observed=False)


def evaluator_class(instance, parent):
    Parent = previous.evaluator_class(instance, parent)

    class BoundedHandleTouchEvaluator(Parent):
        version = VERSION

        def __init__(self):
            super().__init__()
            self.contact_tolerance = BoundedHandleContactTracker(instance)

        def _record(self, original, sequence):
            row = deepcopy(original)
            row['original_contact_harm_before_tolerance'] = original['harm']
            sample = self.contact_tolerance.observe(row, sequence)
            # A candidate touch is provisional until its explicit right no-
            # contact boundary arrives.  No online command is authorized here.
            row['harm'] = bool(sample['hard_violation'])
            evidence = self.contact_tolerance.evidence()
            row['bounded_handle_contact_sample'] = sample
            row['bounded_handle_contact_evidence'] = evidence
            row['placement_task_version'] = VERSION
            result = super()._record(row, sequence)
            if evidence['first_forbidden_contact_or_tip_s'] is not None:
                self.first_harm_s = evidence['first_forbidden_contact_or_tip_s']
            return result

        def finish(self, status='done'):
            evidence = self.contact_tolerance.finish()
            if evidence['first_forbidden_contact_or_tip_s'] is not None:
                self.first_harm_s = evidence['first_forbidden_contact_or_tip_s']
            if self.last is not None:
                self.last['bounded_handle_contact_evidence'] = evidence
                self.last['placement_task_version'] = VERSION
            result = super().finish(status)
            result.update(placement_task_version=VERSION,
                previous_placement_task_version=previous.VERSION,
                bounded_handle_contact_evidence=evidence,
                evaluation_repair='User-approved bounded nonsharp handle/hand contact; sharp, deep, long, robotic or unbounded contacts retain harm; gate/policy/control unchanged')
            return result
    BoundedHandleTouchEvaluator.__name__ = instance.replace('-', '') + 'BoundedHandleV8Evaluator'
    return BoundedHandleTouchEvaluator


def adapt_configuration(current):
    configured = previous.adapt_configuration(current)
    return SimpleNamespace(**{**vars(configured),
        'Evaluator': evaluator_class(current.binding.instance,
            physical._original_parent_evaluator(current.binding.instance)),
        'placement_task_version': VERSION})
