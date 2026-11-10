"""Refresh the existing offline threshold analysis from closed registered runs.

Only actual labelled primitive/conversion windows enter the denominator.
Fixed-input repeated scores, old control proxies and formal results do not.
"""
import argparse
import json
from pathlib import Path

from .action_labels import label_run
from .calibration import ScoreSample, analyse
from .scorer_predicate_contracts import contract
from .offline_critical_predicates import label_adapted_run

ROOT = Path(__file__).resolve().parents[4]
PREDICATES = {'P1': ('harm',), 'P3': ('harm',), 'P4': ('hot_shirt_contact',),
              **{f'I{i}': ('harm','unsafe_commit') for i in range(1,5)}}


def refresh(plans, output):
    roster = json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances']
    instances = [dict(instance_id=r['id'],base_task=r['base_task']) for r in roster]
    samples, sources, skipped, seen = [], [], [], set()
    for plan in plans:
        for case in json.loads(plan.read_text())['episodes']:
            run_id = case['run_id']
            if run_id in seen:
                raise ValueError('Duplicate registered calibration run: '+run_id)
            seen.add(run_id)
            if case['method'] != 'Ours-conf':
                continue
            if case.get('calibration_excluded_reason'):
                skipped.append(dict(run_id=run_id,reason=case['calibration_excluded_reason']))
                continue
            run = ROOT/'research/v4/runs/confidence_gate_dev_20261006'/run_id
            if not (run/'episode.json').exists():
                skipped.append(dict(run_id=run_id,reason='not closed'))
                continue
            episode = json.loads((run/'episode.json').read_text())
            if episode['formal'] or episode.get('included_in_formal'):
                raise ValueError('Formal data cannot select a threshold')
            task = case['instance'].split('-')[0]
            specification = contract(case['instance'])
            if not specification['confidence_selector_available']:
                skipped.append(dict(run_id=run_id,reason='current own scorer is not selected'))
                continue
            phase = ('scripted_calibration' if episode.get('source_phase')=='scripted_calibration'
                     else 'shadow_calibration')
            if case['instance']=='P2-01' or task in ('S1','S3'):
                labels = label_adapted_run(case['instance'],run,source_phase=phase)
            elif task=='S2':
                from .offline_s2_round_labels import label_s2_run
                labels = label_s2_run(run,source_phase=phase)
            elif specification['predicate_keys'] is not None:
                labels = label_run(run,predicate_keys=specification['predicate_keys'],source_phase=phase)
            else:
                skipped.append(dict(run_id=run_id,reason='complete offline predicate contract unavailable'))
                continue
            sources.append(dict(run_id=run_id,instance=case['instance'],phase=phase,
                first_unsafe_sequence=labels['first_unsafe_sequence'],by_gate=labels['by_gate']))
            for row in labels['action_labels']:
                samples.append(ScoreSample(sample_id=run_id+':'+str(row['gate_record_index']),
                    base_task=task,instance_id=case['instance'],gate=row['gate'],
                    label=row['label'] if row['eligible_for_score_statistics'] else None,
                    p_safe=row['p_safe'],source_phase=phase,score_valid=row['valid_score']))
    report = analyse(samples,instances=instances)
    report.update(sources=sources,skipped=skipped,
        scope='Partial registered development corpus; repeated inputs and G3 control proxies excluded.')
    for gate in ('G2','G3'):
        report['gates'][gate]['recommendation']['status']='development_recommendation_authorized'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plans',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=refresh(args.plans,args.output)
    print(json.dumps({g:dict(auroc=report['gates'][g]['pooled_auroc'],
        recommendation=report['gates'][g]['recommendation']) for g in ('G2','G3')}))
