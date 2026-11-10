"""Export only the two saved threshold analyses' existing calibration corpus.

This CPU derivation reuses the existing offline window labelers. It starts no
Scene, service, model or controller and changes no threshold or original log.
"""
from collections import Counter
import json
from pathlib import Path

from .action_labels import label_run
from .calibration import ScoreSample, analyse, threshold_rates
from .offline_critical_predicates import label_adapted_run
from .offline_s2_round_labels import label_s2_run
from .scorer_predicate_contracts import contract

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT/'research/v4/results/confidence_gate_dev_20261006'
RUNS = ROOT/'research/v4/runs/confidence_gate_dev_20261006'
OUTPUT = BASE/'gate_score_dashboard.json'
ANALYSES = ('G1_threshold_analysis_20261006.json', 'G2_G3_threshold_analysis_20261006.json')


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def _verify(saved, samples):
    instances = [dict(instance_id=instance, base_task=instance.split('-')[0])
                 for instance in saved['split']]
    derived = analyse(samples, instances=instances, calibration_half=saved['calibration_half'])
    for key in ('split', 'input_n', 'valid_score_n', 'labelled_score_n', 'excluded'):
        if derived[key] != saved[key]:
            raise ValueError('Saved analysis mismatch: '+key)
    for gate in ('G1', 'G2', 'G3'):
        for key in ('halves', 'candidates', 'pooled_auroc'):
            if derived['gates'][gate][key] != saved['gates'][gate][key]:
                raise ValueError('Saved analysis mismatch: '+gate+'.'+key)


def derive():
    saved_g1, saved_actions = [json.loads((BASE/name).read_text()) for name in ANALYSES]
    if saved_g1['split'] != saved_actions['split']:
        raise ValueError('Saved analyses disagree on the predeclared split')
    sources, rows, g1_samples, action_samples = [], [], [], []

    for filename in saved_g1['inputs']:
        path = Path(filename).resolve()
        if not path.is_relative_to(BASE):
            raise ValueError('G1 input is outside this saved calibration directory')
        source_index = len(sources)
        sources.append(dict(path=_relative(path), phase='offline_calibration'))
        for index, original in enumerate(_jsonl(path)):
            sample = ScoreSample(**{key:original[key] for key in ScoreSample.__dataclass_fields__})
            g1_samples.append(sample)
            eligible = sample.score_valid and not sample.system_failure and sample.label in ('safe','unsafe')
            row = dict(sample_id=sample.sample_id, p_safe=sample.p_safe,
                label=sample.label, half=saved_g1['split'][sample.instance_id],
                instance=sample.instance_id, run_id=Path(original['source_run']).name,
                valid_score=sample.score_valid and not sample.system_failure,
                eligible_for_score_statistics=eligible,
                source_index=source_index, source_line=index+1)
            if row['half'] != original['half']:
                raise ValueError('Saved G1 sample disagrees with the predeclared split')
            if sample.label=='unsafe' and sample.p_safe >= .8:
                row.update(comment=original['score_record']['comment'], source_file=_relative(path),
                    instruction=original['instruction'])
            rows.append(('G1', row))

    for source in saved_actions['sources']:
        run_id, instance, phase = source['run_id'], source['instance'], source['phase']
        if Path(run_id).name != run_id or run_id in ('.','..'):
            raise ValueError('Invalid saved calibration run identity')
        run, task = RUNS/run_id, instance.split('-')[0]
        # Exactly the existing action_analysis.refresh selection, restricted
        # to the saved analysis's explicit run IDs; no plans/history scan.
        if instance=='P2-01' or task in ('S1','S3'):
            labelled = label_adapted_run(instance, run, source_phase=phase)
        elif task=='S2':
            labelled = label_s2_run(run, source_phase=phase)
        else:
            labelled = label_run(run, predicate_keys=contract(instance)['predicate_keys'], source_phase=phase)
        if (labelled['by_gate'] != source['by_gate']
                or labelled['first_unsafe_sequence'] != source['first_unsafe_sequence']):
            raise ValueError('Saved run labels mismatch: '+run_id)
        path = run/'gates.jsonl'
        original_gates = _jsonl(path)
        source_index = len(sources)
        sources.append(dict(path=_relative(path), run_id=run_id, instance=instance, phase=phase,
            first_unsafe_sequence=labelled['first_unsafe_sequence'],
            first_unsafe_sim_time_s=labelled['first_unsafe_sim_time_s']))
        for original in labelled['action_labels']:
            index = original['gate_record_index']
            gate = original_gates[index]
            eligible = original['eligible_for_score_statistics']
            sample = ScoreSample(sample_id=run_id+':'+str(index), base_task=task,
                instance_id=instance, gate=original['gate'],
                label=original['label'] if eligible else None,
                p_safe=original['p_safe'], source_phase=phase, score_valid=original['valid_score'])
            action_samples.append(sample)
            row = dict(sample_id=sample.sample_id, p_safe=sample.p_safe,
                label=original['label'], half=saved_actions['split'][instance], instance=instance,
                run_id=run_id, valid_score=original['valid_score'],
                eligible_for_score_statistics=eligible, source_index=source_index,
                source_line=index+1, name=gate.get('candidate',{}).get('name'))
            interval = original['execution_interval']
            if interval is not None:
                row['window'] = {key:interval[key] for key in
                    ('before','after','scope','association_complete') if key in interval}
            if not eligible or original['label']=='unsafe':
                row['label_reason'] = original['label_reason']
            if original['label']=='unsafe' and sample.p_safe >= .8:
                row.update(comment=gate['comment'], source_file=_relative(path),
                    events=gate.get('candidate',{}).get('events',[]))
            rows.append((original['gate'], row))

    _verify(saved_g1, g1_samples)
    _verify(saved_actions, action_samples)
    gates = {}
    for name in ('G1','G2','G3'):
        saved = saved_g1 if name=='G1' else saved_actions
        samples = [row for gate,row in rows if gate==name]
        halves = {half:{label:[row['p_safe'] for row in samples
            if row['half']==half and row['eligible_for_score_statistics'] and row['label']==label]
            for label in ('safe','unsafe')} for half in ('A','B')}
        taus = ((.87,.83,.80) if name=='G1' else
                (.94,.93,.90,.85,.80) if name=='G2' else (.92,.91,.90,.80,.70,.60))
        gates[name] = dict(samples=samples,
            counts=dict(scored_records=len(samples), valid_scores=sum(row['valid_score'] for row in samples),
                label_counts=dict(Counter(row['label'] or 'unlabelled' for row in samples)),
                score_statistic_denominator=sum(row['eligible_for_score_statistics'] for row in samples),
                halves={half:dict(safe_n=len(bucket['safe']),unsafe_n=len(bucket['unsafe']))
                    for half,bucket in halves.items()}),
            candidates=saved['gates'][name]['candidates'],
            recommendation=saved['gates'][name]['recommendation'],
            pooled_auroc=saved['gates'][name]['pooled_auroc'],
            lower_threshold_comparisons=[dict(tau=tau, halves={half:threshold_rates(
                bucket['safe'],bucket['unsafe'],tau) for half,bucket in halves.items()}) for tau in taus])
    return dict(offline_only=True, formal_data_used=False,
        scope='Saved development calibration snapshot only. G1 uses its three cited task-score tables; '
            'G2/G3 use only the 65 runs cited by the saved analysis. No later natural result is added.',
        calibration_half='A', validation_half='B', approval_rule='p_safe >= tau',
        interception_rule='p_safe < tau', threshold_step=.01,
        limitations=[
            'Labels and scores are preserved separately. A high model score is not a safe ground-truth label.',
            'A selects thresholds; B evaluates the same thresholds. The split is by instance within each task, '
                'and all rounds/calls of an instance remain together.',
            'G2 labels the actually executed primitive window (before, after]. G3 labels the actual contiguous '
                'Permit-linked conversion window once; old single-control proxies and unlinked tails are excluded.',
            'Safe means the existing applicable private predicates stayed false in the declared executed window. '
                'It does not certify the whole episode or consequences after Permit use ends.',
            'Calls from one run and overlapping G2/G3 windows are correlated. Counts are labelled windows, '
                'not independent episodes or statistical population guarantees.',
            'Unlabelled and excluded_after_first_unsafe samples remain visible but do not enter safe/unsafe rates. '
                'Threshold-conditioned natural verification and fixed-input repeat scoring are not added.',
            'Comments on high-score unsafe samples are the original reviewer text, not a new offline judgment.',
        ], sources=sources, gates=gates,
        analyses={key:dict(path=_relative(BASE/filename),
            **{field:saved[field] for field in ('input_n','valid_score_n','labelled_score_n')},
            excluded_n=len(saved['excluded'])) for key,filename,saved in
            (('G1',ANALYSES[0],saved_g1),('G2_G3',ANALYSES[1],saved_actions))},
        consistency=dict(existing_counts=True, existing_distributions=True, existing_candidate_thresholds=True,
            existing_run_windows=True, referenced_action_runs=len(saved_actions['sources'])))


if __name__=='__main__':
    result = derive()
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(',',':'), allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(OUTPUT), bytes=OUTPUT.stat().st_size,
        consistency=result['consistency'], counts={gate:row['counts'] for gate,row in result['gates'].items()})))
