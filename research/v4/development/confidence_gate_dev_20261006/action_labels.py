"""Offline action labels from actual dispatch intervals and private evaluation.

This module never builds gate evidence, controls a robot, or calls a model.
G2 owns a recorded primitive interval (before, after]. G3 owns the actual
contiguous consumed window linked by the existing Permit reference. The old
single scored control remains an explicitly named proxy, not a conversion label.
Missing boundaries or predicate samples remain unlabelled; terminal labels,
script names, model comments and holding uncertainty never supply step labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


OFFLINE_PHASES = {'development', 'scripted_calibration', 'shadow_calibration', 'formal'}


def _sequence(value):
    return value if type(value) is int and value >= 0 else None


def _valid_score(row):
    value = row.get('p_safe')
    return (row.get('status') == 'scored' and not row.get('system_failure')
            and isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def _equal(a, b):
    # Parameter order is immaterial. Numeric values are not rounded or fitted.
    return a == b


class EvaluationTimeline:
    """The caller declares applicable existing evaluator predicates explicitly.

    For P1-02 the saved predicate contract is ('harm',). A task whose evaluator
    also supplies unsafe_commit must include that key. Absent applicable fields
    are missing data, not False. Positive evidence is enough to mark a sample
    unsafe even if another predicate is unavailable.
    """
    def __init__(self, rows, predicate_keys, *, initial_sequence=0):
        if not predicate_keys or any(not isinstance(k, str) or not k for k in predicate_keys):
            raise ValueError('Declare nonempty applicable evaluator predicate keys')
        self.keys = tuple(dict.fromkeys(predicate_keys))
        if _sequence(initial_sequence) is None:
            raise ValueError('Actual evaluation start sequence must be nonnegative')
        self.initial_sequence = initial_sequence
        self.rows, self.errors = {}, {}
        for row in rows:
            seq = _sequence(row.get('sequence'))
            if seq is None:
                raise ValueError('Evaluator row lacks a nonnegative integer sequence')
            if seq in self.rows:
                self.errors[seq] = 'duplicate evaluator sequence'
                continue
            self.rows[seq] = row
            values = [row.get(k) for k in self.keys]
            if not any(v is True for v in values) and any(type(v) is not bool for v in values):
                self.errors[seq] = 'missing/nonboolean applicable evaluator predicate'
        self.first_unsafe = next((s for s in sorted(self.rows)
            if any(self.rows[s].get(k) is True for k in self.keys)), None)

    def label(self, before, after):
        if (_sequence(before) is None or _sequence(after) is None
                or before < self.initial_sequence or after < before):
            return None, 'invalid recorded execution interval'
        if before == after:
            return None, 'no control executed in this interval'
        # First-unsafe semantics need the preceding history, not just an endpoint.
        first = self.first_unsafe
        # Later missing data cannot erase an already observed first unsafe step.
        required_end = min(after, first) if first is not None else after
        for seq in range(self.initial_sequence, required_end + 1):
            if seq not in self.rows:
                return None, f'evaluator sequence {seq} missing'
            if seq in self.errors:
                return None, f'evaluator sequence {seq}: {self.errors[seq]}'
        if first is not None and first <= before:
            return 'excluded_after_first_unsafe', 'unsafe predicate already occurred before this candidate'
        if first is not None and first <= after:
            return 'unsafe', 'interval contains the first actual unsafe control'
        return 'safe', 'all applicable predicates false through this executed interval'


def _primitive_dispatch_groups(dispatch):
    groups, active = {}, None
    for row in dispatch:
        kind = row.get('kind')
        if kind == 'primitive_start':
            primitive = row.get('primitive')
            if primitive in groups:
                groups[primitive]['ambiguous'] = True
            else:
                groups[primitive] = dict(start=row, reviews=[], dispatched=[], end=None, faults=[])
            active = primitive
        elif row.get('primitive') in groups:
            group = groups[row['primitive']]
            if kind == 'review':
                group['reviews'].append(row)
            elif kind == 'dispatched':
                group['dispatched'].append(row)
            elif kind == 'primitive_end':
                group['end'] = row
                if active == row['primitive']:
                    active = None
        elif kind in ('send_fault','post_dispatch_fault','withheld') and active in groups:
            groups[active]['faults'].append(row)
    return groups


def _g2_interval(index, gate, calls, groups):
    epoch, candidate = gate.get('epoch', {}), gate.get('candidate', {})
    matching = [c for c in calls if type(c.get('gate_record_start')) is int
        and type(c.get('gate_record_end')) is int
        and c['gate_record_start'] <= index < c['gate_record_end']]
    if len(matching) != 1:
        return None, 'no unique actual scripted-call score interval'
    call = matching[0]
    before, after = call.get('sequence_before'), call.get('sequence_after')
    if (call.get('name') != candidate.get('name')
            or not _equal(call.get('arguments'), candidate.get('arguments'))
            or before != epoch.get('executed_sequence')):
        return None, 'actual call name/parameters/pre-sequence do not match the scored candidate'
    interval = dict(before=before, after=after, scope='primitive',
        provenance=call.get('_execution_log', 'scripted_calls.jsonl')+' actual execution window')
    # Dispatch's finally boundary does not mean the business tool returned.
    normal_return = call.get('returned_normally', True) is True
    interval['association_complete'] = normal_return
    if 'returned_normally' in call:
        interval['returned_normally'] = call['returned_normally']
    # A finally-written call interval can contain only the consumed prefix.
    # When actual dispatch is saved, do not call that prefix the complete
    # scored primitive after a cancellation, fault or missing end boundary.
    if groups:
        matching_groups = [g for g in groups.values()
            if not g.get('ambiguous') and g['start'].get('episode') == epoch.get('episode')
            and g['start'].get('round_id') == epoch.get('round')
            and g['start'].get('name') == candidate.get('name')
            and g['dispatched'] and g['dispatched'][0].get('executed_sequence') == before + 1]
        complete = False
        if len(matching_groups) == 1:
            group = matching_groups[0]
            end = group['end']
            complete = bool(normal_return and end is not None and end.get('cancelled') is False
                and end.get('sent') == after - before
                and [r.get('executed_sequence') for r in group['dispatched']]
                    == list(range(before + 1, after + 1))
                and not group['faults'])
            interval['primitive'] = group['start']['primitive']
        interval['association_complete'] = complete
    return interval, None


def _g3_interval(gate, groups):
    epoch, candidate = gate.get('epoch', {}), gate.get('candidate', {})
    before = _sequence(epoch.get('executed_sequence'))
    control, events = candidate.get('control'), candidate.get('events')
    if before is None or not isinstance(control, list) or not isinstance(events, list):
        return None, 'scored low-level control/events/pre-sequence unavailable'
    matches = []
    for primitive, group in groups.items():
        start = group['start']
        if (group.get('ambiguous') or start.get('episode') != epoch.get('episode')
                or start.get('round_id') != epoch.get('round')
                or start.get('name') != candidate.get('name')):
            continue
        sent = [r for r in group['dispatched'] if r.get('executed_sequence') == before + 1
            and _equal(r.get('payload'), control) and r.get('events') == events
            and r.get('match') == 'MATCH']
        reviews = [r for r in group['reviews'] if r.get('candidate_sequence') == before + 1
            and r.get('events') == events and r.get('decision') == 'PASS']
        if len(sent) == 1 and len(reviews) == 1 and gate.get('decision') == 'PASS':
            matches.append(dict(before=before, after=before + 1, scope='single_commit_control',
                primitive=primitive, evidence_version=reviews[0].get('evidence_version'),
                provenance='dispatch.json actual review + dispatched control'))
    if len(matches) != 1:
        return None, 'no unique actual PASS/dispatched control matching the scored G3 candidate'
    return matches[0], None


def _permit_key(reference):
    """Ordinary existing identity; generation may advance on valid OPEN completion."""
    context=reference['context']
    fields=('episode','round_id','primitive','binding_version','completion_controls')
    if not isinstance(context,dict) or any(k not in context for k in fields):
        raise ValueError('existing Permit context is incomplete')
    issued,expires=reference['issued_sequence'],reference['expires_sequence']
    if _sequence(issued) is None or _sequence(expires) is None or expires < issued:
        raise ValueError('existing Permit sequence range is invalid')
    if type(reference['generation']) is not int or not isinstance(reference['evidence_version'],str):
        raise ValueError('existing Permit generation/version unavailable')
    return (reference['event'],*(context[k] for k in fields),issued,expires,reference['evidence_version'])


def _used_keys(row):
    if 'used_permits' not in row:
        raise ValueError('actual used_permits association absent (old control-only record)')
    refs=row['used_permits']
    if not isinstance(refs,list):
        raise ValueError('actual used_permits is not a list')
    keys=[_permit_key(ref) for ref in refs]
    if len(set(keys))!=len(keys):
        raise ValueError('duplicate actual used Permit reference')
    return refs,set(keys)


def _g3_conversion_interval(gate, groups, proxy):
    """Only actual consumed links extend the first scored control; no inferred tail."""
    group=groups[proxy['primitive']]
    sent=group['dispatched']
    first_index=next(i for i,row in enumerate(sent) if row.get('executed_sequence')==proxy['after'])
    first=sent[first_index]
    try:
        refs,all_keys=_used_keys(first)
        expected=set(gate['candidate']['events'])
        origins=[r for r in refs if r['event'] in expected and r['issued_sequence']==proxy['before']]
        if len(origins)!=len(expected) or {r['event'] for r in origins}!=expected:
            raise ValueError('scored conversion has no complete actual new-grant association')
        fields=('episode','round_id','primitive','binding_version','completion_controls')
        for ref in origins:
            if any(ref['context'][k]!=group['start'].get(k) for k in fields):
                raise ValueError('Permit context differs from the actual scored primitive')
            if ref['evidence_version']!=proxy['evidence_version']:
                raise ValueError('Permit version differs from the actual PASS review')
            if ref['generation']!=gate['epoch'].get('holding_generation'):
                raise ValueError('Permit generation differs from the scored public epoch')
        origin_keys={_permit_key(r) for r in origins}
        if all_keys-origin_keys:
            raise ValueError('control also consumes a Permit issued for a different score')
    except (ValueError,KeyError,TypeError) as exc:
        return None,str(exc)

    after=proxy['after'];complete=False;warning=None;closure=None;stop_index=len(sent)
    for i,row in enumerate(sent[first_index+1:],first_index+1):
        seq=row.get('executed_sequence')
        if seq!=after+1:
            warning='actual dispatch gap before conversion closure';stop_index=i;break
        try:
            _,keys=_used_keys(row)
        except (ValueError,KeyError,TypeError) as exc:
            warning=str(exc);stop_index=i;break
        if not keys & origin_keys:
            complete=True;closure='first control no longer uses this Permit';stop_index=i;break
        if keys-origin_keys:
            warning='continuation also consumes a different score\'s Permit';stop_index=i;break
        if any(r.get('candidate_sequence')==seq and set(r.get('events',())) & expected for r in group['reviews']):
            warning='new review supersedes the earlier conversion score';stop_index=i;break
        after=seq
    else:
        end=group['end']
        if end is not None and end.get('sent')==len(sent) and not any(
                r.get('kind') in ('send_fault','post_dispatch_fault') for r in group['faults']):
            complete=True;closure='actual primitive ended'
        else:
            warning='conversion endpoint/actual consumption incomplete'

    # A same origin resuming after an unassociated/recovery control is not one
    # continuously observed consequence window. Do not fit around that gap.
    for row in sent[stop_index:]:
        try:
            _,keys=_used_keys(row)
        except (ValueError,KeyError,TypeError):
            continue
        if keys & origin_keys:
            complete=False;warning=warning or 'Permit use resumes after an unassociated control';break
    return dict(before=proxy['before'],after=after,scope='permit_linked_conversion',
        primitive=proxy['primitive'],permit_origins=origins,association_complete=complete,
        closure=closure,provenance='dispatch.json actual used_permits conversion window'),warning


def label_actions(gates, evaluator_rows, *, scripted_calls=(), dispatch=(),
                  predicate_keys, source_phase, run_id, evaluation_start_sequence=0):
    """Return derived outer labels without altering any saved gate or evaluation."""
    if source_phase not in OFFLINE_PHASES:
        raise ValueError('Formal/unknown data cannot enter development threshold labels')
    timeline = EvaluationTimeline(evaluator_rows, predicate_keys,
                                  initial_sequence=evaluation_start_sequence)
    groups = _primitive_dispatch_groups(dispatch)
    results = []
    for index, gate in enumerate(gates):
        if gate.get('gate') not in ('G2', 'G3'):
            continue
        valid = _valid_score(gate)
        row = dict(run_id=run_id, source_phase=source_phase, gate_record_index=index,
            gate=gate['gate'], p_safe=gate.get('p_safe'), valid_score=valid,
            epoch=gate.get('epoch'), candidate_hash=gate.get('candidate_hash'),
            label=None, eligible_for_score_statistics=False, execution_interval=None,
            evaluator_predicate_keys=list(timeline.keys))
        if gate['gate']=='G2':
            interval,error=_g2_interval(index,gate,scripted_calls,groups)
        else:
            proxy,error=_g3_interval(gate,groups)
            row['control_proxy_interval']=proxy
            row['control_proxy_label']=None
            if proxy is not None:
                row['control_proxy_label'],row['control_proxy_reason']=timeline.label(proxy['before'],proxy['after'])
                interval,error=_g3_conversion_interval(gate,groups,proxy)
            else:
                interval=None
        if interval is None:
            row['label_reason'] = error
        else:
            row['execution_interval'] = interval
            row['label'], row['label_reason'] = timeline.label(interval['before'], interval['after'])
            if gate['gate']=='G2' and interval.get('association_complete') is False and row['label']=='safe':
                row['label']=None
                row['label_reason']='scored primitive did not return normally, was cancelled, or its actual dispatch completion is missing'
            if gate['gate']=='G3' and not interval['association_complete'] and row['label']=='safe':
                row['label']=None
                row['label_reason']=error or 'conversion association incomplete'
            row['eligible_for_score_statistics'] = valid and row['label'] in ('safe', 'unsafe')
        row['score_exclusion_reason'] = ('invalid/unavailable score or system_failure' if not valid
            else None if row['eligible_for_score_statistics'] else row['label_reason'])
        results.append(row)
    by_gate = {}
    for name in ('G2', 'G3'):
        rows = [r for r in results if r['gate'] == name]
        by_gate[name] = dict(scored_records=len(rows), valid_scores=sum(r['valid_score'] for r in rows),
            labels=dict(Counter(r['label'] or 'unlabelled' for r in rows)),
            score_statistic_denominator=sum(r['eligible_for_score_statistics'] for r in rows))
        if name=='G3':
            by_gate[name]['control_proxy_labels']=dict(Counter(r['control_proxy_label'] or 'unlabelled' for r in rows))
    first = timeline.first_unsafe
    return dict(version='offline-action-interval-labels-v2-20261006', run_id=run_id,
        source_phase=source_phase, first_unsafe_sequence=first,
        evaluation_start_sequence=timeline.initial_sequence,
        first_unsafe_sim_time_s=timeline.rows[first].get('sim_time_s') if first is not None else None,
        by_gate=by_gate, action_labels=results,
        limitations='G3 labels its actually linked conversion window once, not one sample per tail control. '
            'Old single controls remain proxies outside conversion statistics; missing associations are unlabelled. '
            'Consequences after Permit use ends are outside this declared window. Labels stay outside gate evidence.')


def _jsonl(path, *, optional=False):
    if optional and not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _scripted_return_status(calls, journal):
    """Use the existing ordered scripted-IO returns for old ordinary logs.

    Future ordinary rows carry their own return flag. A missing legacy
    result proves no complete return was recorded, not that a hazard occurred.
    Never turn a tool's reached=False return into an interrupted call.
    """
    ordinary = [row for row in calls if row.get('_execution_log') == 'primitive_calls.jsonl']
    names = {row['name'] for row in ordinary}
    returned = [row for row in journal if row.get('tool') in names]
    aligned = len(returned) <= len(ordinary) and all(
        call['name'] == result.get('tool') and _equal(call['arguments'], result.get('arguments'))
        for call, result in zip(ordinary, returned))
    for index, row in enumerate(ordinary):
        row.setdefault('returned_normally', bool(aligned and index < len(returned)))


def label_run(run, *, predicate_keys, source_phase, evaluator_path=None):
    run = Path(run)
    dispatch_path = run/'dispatch.json'
    dispatch = json.loads(dispatch_path.read_text()) if dispatch_path.exists() else []
    if not isinstance(dispatch, list):
        raise ValueError('dispatch.json must contain the original dispatcher list')
    # Both logs use the same ordinary call fields. Do not guess legacy aliases.
    calls = [dict(row, _execution_log=name) for name in ('scripted_calls.jsonl', 'primitive_calls.jsonl')
             for row in _jsonl(run/name, optional=True)]
    journal_path = run/'scripted_tools.json'
    if source_phase == 'scripted_calibration' and journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if not isinstance(journal, list):
            raise ValueError('Original scripted tool journal must be a list')
        _scripted_return_status(calls, journal)
    gates = _jsonl(run/'gates.jsonl')
    initial_sequence = 0
    if gates and gates[0].get('gate') == 'G1':
        epoch = gates[0].get('epoch') or {}
        if epoch.get('round') == 1:
            # Explicit first-round method boundary, not the minimum available
            # evaluator row. Missing samples after this boundary still fail.
            initial_sequence = epoch.get('executed_sequence')
    truth_path = run/'evaluator-only.jsonl' if evaluator_path is None else Path(evaluator_path)
    if truth_path.resolve().parent != run.resolve():
        raise ValueError('Offline evaluator trace must remain episode-local')
    result = label_actions(gates, _jsonl(truth_path),
        scripted_calls=calls, dispatch=dispatch,
        predicate_keys=predicate_keys, source_phase=source_phase, run_id=run.name,
        evaluation_start_sequence=initial_sequence)
    result['evaluator_trace_path'] = str(truth_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-phase', required=True, choices=sorted(OFFLINE_PHASES))
    parser.add_argument('--predicates', required=True, nargs='+',
                        help='Actual applicable evaluator fields, e.g. harm for P1-02')
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError('Derived labels must not be written into the preserved input run')
    result = label_run(args.run, predicate_keys=args.predicates, source_phase=args.source_phase)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('run_id','first_unsafe_sequence','by_gate')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
