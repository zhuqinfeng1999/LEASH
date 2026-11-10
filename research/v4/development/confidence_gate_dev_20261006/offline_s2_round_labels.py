"""Offline-only S2 same-control administrative round rows for action labels.

Reuses original harm/unsafe_commit booleans and the existing action labeler.
No scorer, raw log, online input, task threshold or robot state is changed.
"""
from copy import deepcopy
import json
import math
from pathlib import Path


KEY = '_offline_original_s2_unsafe'
PREDICATES = ('harm','unsafe_commit')


def _number(value):
    return type(value) in (int,float) and math.isfinite(value) and value >= 0


def _unsafe(row):
    values=[row.get(k) for k in PREDICATES]
    if any(v is True for v in values):return True
    if all(type(v) is bool for v in values):return False
    return None


def _administrative(previous,current):
    return (type(previous.get('round')) is int and previous['round'] in (1,2)
        and current.get('round') == previous['round']+1
        and _number(previous.get('sim_time')) and previous['sim_time'] == current.get('sim_time')
        and previous.get('actual_controller_steps') == previous.get('sequence')
        and current.get('actual_controller_steps') == current.get('sequence'))


def adapt_s2_rows(rows):
    """Deduplicate only verified same-sequence/time administrative transitions.

    A new instruction can change the proximity commit at that same state.
    Such a positive is preserved in metadata/raw, but is not assigned to the
    preceding physical action; its action window remains unlabelled. Earlier
    genuinely recorded positive physical evidence cannot be erased.
    """
    derived,notes,indices=[],[],{}
    for index,original in enumerate(rows):
        if not isinstance(original,dict):raise ValueError('Actual S2 evaluator JSON object required')
        row=deepcopy(original)
        sequence=row.get('sequence')
        if type(sequence) is not int or sequence < 0:
            raise ValueError('Actual nonnegative controller sequence required')
        row[KEY]=_unsafe(row)
        # S2's original snapshot field is sim_time, not sim_time_s. This alias
        # is a derived display/label time only, never rewritten into raw files.
        if _number(row.get('sim_time')):
            if 'sim_time_s' not in row:
                row['sim_time_s']=float(row['sim_time'])
            elif row['sim_time_s'] != row['sim_time']:
                row[KEY]=None
                notes.append(dict(original_row=index,sequence=sequence,
                    reason='Conflicting existing timestamp fields; action association unresolved'))
        if sequence not in indices:
            indices[sequence]=len(derived);derived.append(row)
            continue
        location=indices[sequence];previous=derived[location]
        if _administrative(previous,row):
            unchanged = all(type(previous.get(k)) is bool and type(row.get(k)) is bool
                            and previous[k] == row[k] for k in PREDICATES)
            if unchanged and previous[KEY] is not None and row[KEY] is not None:
                derived[location]=row
                reason='Same actual control/time S2 round transition; identical original predicates, no extra control'
            elif previous[KEY] is True:
                # The known earlier physical unsafe control is still unsafe.
                reason='Administrative mismatch cannot erase already recorded physical unsafe evidence'
            else:
                row[KEY]=None;derived[location]=row
                reason='Predicate changed/missing at no-physics instruction boundary; not attributed to preceding action'
            notes.append(dict(original_row=index,sequence=sequence,reason=reason,
                previous_predicates={k:previous.get(k) for k in PREDICATES},
                administrative_predicates={k:row.get(k) for k in PREDICATES}))
        else:
            previous[KEY]=None
            notes.append(dict(original_row=index,sequence=sequence,
                reason='Duplicate sequence is not a verified administrative transition; action association unresolved'))
    return dict(rows=derived,predicate_key=KEY,row_notes=notes,
        provenance='CurrentS2Scorer -> S2ControlScorer.observe -> original evaluator.snapshot harm/unsafe_commit',
        raw_logs_unchanged=True,terminal_labels_not_used=True,
        limitations='Only original per-control bools. Changed round-start critical facts do not create a scored physical action. '
            'Original labeler still requires actual primitive/Permit windows and complete prior control history.')


def label_s2_run(run,*,source_phase):
    """Root can select this existing offline branch after an active run closes."""
    from .action_labels import _jsonl,label_actions
    run=Path(run)
    dispatch_path=run/'dispatch.json'
    dispatch=json.loads(dispatch_path.read_text()) if dispatch_path.exists() else []
    if not isinstance(dispatch,list):raise ValueError('Original dispatcher list required')
    calls=[dict(row,_execution_log=name)
        for name in ('scripted_calls.jsonl','primitive_calls.jsonl')
        for row in _jsonl(run/name,optional=True)]
    adapted=adapt_s2_rows(_jsonl(run/'evaluator-only.jsonl'))
    result=label_actions(_jsonl(run/'gates.jsonl'),adapted['rows'],
        predicate_keys=(KEY,),scripted_calls=calls,dispatch=dispatch,
        source_phase=source_phase,run_id=run.name)
    result['offline_predicate_adaptation']={k:v for k,v in adapted.items() if k!='rows'}
    return result
