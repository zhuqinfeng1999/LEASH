"""Associate saved calibration samples with their exact public scoring inputs.

Read only the dashboard's 65 action runs and three G1 scoring-table families.
No model, Scene, image read/copy/render, original log or threshold is changed.
"""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT/'research/v4/results/confidence_gate_dev_20261006'
DASHBOARD = BASE/'gate_score_dashboard.json'
OUTPUT = BASE/'gate_score_input_examples.json'


def _relative(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT/path
    return str(path.resolve().relative_to(ROOT))


def _plain(value):
    """Keep small public metadata; omit inline binary/code payloads."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in ('base64','source_code','python_source','html_source','script_source','fullbody'):
                continue
            if key.endswith('_path') and isinstance(item, str):
                result[key] = _relative(item)
            else:
                result[key] = _plain(item)
        return result
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, str) and 'data:image/' in value:
        return None
    return value


def enrich():
    dashboard = json.loads(DASHBOARD.read_text())
    cache, examples, image_refs, raw_refs = {}, {}, [], []

    def read(path):
        path = Path(path)
        if path not in cache:
            cache[path] = [] if not path.is_file() else [(line, json.loads(text))
                for line,text in enumerate(path.read_text().splitlines(),1) if text.strip()]
        return cache[path]

    def image(row, *, name, role, missing, index, round_index=None):
        selected = row.get('annotated_rgb_path',row['rgb_path']) if role=='current' else row['rgb_path']
        relative = _relative(selected)
        exists = (ROOT/relative).is_file()
        result = dict(name=name, path=relative, frame_id=row.get('frame_id'),
            view=row.get('view'), role=role, image_index=index, exists=exists)
        image_refs.append(relative)
        if not exists:
            missing.append('Submitted image is not saved: '+relative)
        if round_index is not None:
            result['round_index'] = round_index
        if row.get('path_overlay') is not None:
            result['annotation'] = _plain(row['path_overlay'])
        if selected != row['rgb_path']:
            raw = _relative(row['rgb_path'])
            result.update(raw_path=raw, raw_exists=(ROOT/raw).is_file())
            raw_refs.append(raw)
            if not result['raw_exists']:
                missing.append('Original RGB companion is not saved: '+raw)
        return result

    for gate, data in dashboard['gates'].items():
        for sample in data['samples']:
            missing = []
            source = dashboard['sources'][sample['source_index']]
            score_path = ROOT/source['path']
            original_rows = [row for line,row in read(score_path) if line==sample['source_line']]
            entry = dict(sample_id=sample['sample_id'], gate=gate, instance=sample['instance'],
                label=sample['label'], half=sample['half'], run_id=sample['run_id'],
                p_safe=sample['p_safe'], eligible_for_score_statistics=sample['eligible_for_score_statistics'],
                gate_score_source=dict(path=_relative(score_path),line=sample['source_line']),
                input_images=[], candidate=None, instruction=None, robot_state=None,
                recent_executed_actions=[], history=[], history_submitted=False,
                model_id='unavailable', reasoning='unavailable', missing=missing)
            if len(original_rows)!=1:
                missing.append('Original score table record unavailable')
                entry['association_status'] = 'missing'
                examples[sample['sample_id']] = entry
                continue
            original = original_rows[0]
            score = original['score_record'] if gate=='G1' else original
            entry.update(comment=score.get('comment'), candidate=_plain(score.get('candidate')))
            if score.get('gate')!=gate or score.get('p_safe')!=sample['p_safe']:
                missing.append('Dashboard sample does not match its original gate score')
            backend = score_path.parent/'luna_backend' if gate=='G1' else score_path.parent
            input_path, reviewer_path = backend/'gate_inputs.jsonl', backend/'reviewer_scores.jsonl'
            matched = []
            for line, record in read(input_path):
                evidence = record.get('evidence',{})
                if record.get('gate')!=gate or evidence.get('candidate')!=score.get('candidate'):
                    continue
                if gate=='G1':
                    token = sample['sample_id'].replace(':','_')
                    exact_sample = any(token in Path(im.get('rgb_path','')).parts
                        for im in evidence.get('current_images',[]))
                    same_frame = evidence.get('frame_id')==score.get('epoch',{}).get('frame_id')
                    same_instruction = evidence.get('trusted_instruction')==original['instruction']
                    if exact_sample and same_frame and same_instruction:
                        matched.append((line,evidence))
                elif evidence.get('frame_id')==score.get('epoch',{}).get('executed_sequence'):
                    matched.append((line,evidence))
            entry['input_source'] = dict(path=_relative(input_path), lines=[line for line,_ in matched],
                association=('sample-specific saved image directory + exact candidate/frame/instruction'
                    if gate=='G1' else 'exact gate + candidate + executed public frame'))
            if len(matched)!=1:
                missing.append('No unique saved gate input: '+str(len(matched))+' matches')
            else:
                line, evidence = matched[0]
                entry['input_source']['line'] = line
                entry.update(instruction=evidence.get('trusted_instruction'),
                    candidate=_plain(evidence.get('candidate')), robot_state=_plain(evidence.get('robot_state')),
                    frame_id=evidence.get('frame_id'),
                    recent_executed_actions=_plain(evidence.get('recent_executed_actions',[])),
                    history_submitted='round_history' in evidence)
                if 'candidate_description' in evidence:
                    entry['candidate_description'] = evidence['candidate_description']
                if 'event' in evidence:
                    entry['event'] = _plain(evidence['event'])
                # Exactly backend.wire_evidence order and path selection. The
                # source RGB companion is never labelled an additional SDK input.
                images = entry['input_images']
                for current in evidence.get('current_images',[]):
                    images.append(image(current,name=current['view'],role='current',
                        missing=missing,index=len(images)))
                first = evidence.get('first_workspace_image')
                if first is None:
                    missing.append('Saved first workspace input unavailable')
                else:
                    images.append(image(first,name='first_workspace',role='first',
                        missing=missing,index=len(images)))
                history = []
                for previous in evidence.get('round_history',[]):
                    record = _plain({key:value for key,value in previous.items() if key!='start_keyframes'})
                    frames = []
                    for frame in previous.get('start_keyframes',[]):
                        position = len(images)
                        images.append(image(frame,name='previous_round_start',role='previous_round',
                            round_index=previous.get('round_index'),missing=missing,index=position))
                        frames.append(_plain({key:value for key,value in frame.items()
                            if not key.endswith('_path')})|dict(image_index=position))
                    record['start_keyframes'] = frames
                    history.append(record)
                entry['history'] = history

            score_records = []
            for attempt in score.get('attempts',[]):
                if attempt.get('p_safe') is None:
                    continue
                matches = [(line,row) for line,row in read(reviewer_path)
                    if row.get('event')=='score' and row.get('gate')==gate
                    and row.get('thread_id')==attempt.get('thread_id')
                    and row.get('turn_id')==attempt.get('turn_id')
                    and row.get('p_safe')==attempt['p_safe']
                    and row.get('comment')==attempt.get('comment')]
                if len(matches)!=1:
                    missing.append('No unique reviewer score for recorded attempt: '+str(len(matches))+' matches')
                    continue
                line, record = matches[0]
                score_records.append(dict(line=line, **{key:record[key] for key in
                    ('thread_id','turn_id','call_index','model_id','reasoning_effort','logged_utc') if key in record},
                    raw_p_safe=record.get('raw',{}).get('p_safe')))
            if not score_records:
                missing.append('Actual reviewer response record unavailable')
            entry['score_source'] = dict(path=_relative(reviewer_path), records=score_records,
                association='exact gate + recorded thread_id/turn_id + p_safe + comment')
            model_ids = {record.get('model_id') or 'unavailable' for record in score_records}
            efforts = {record.get('reasoning_effort') or 'unavailable' for record in score_records}
            entry['model_id'] = next(iter(model_ids)) if len(model_ids)==1 else 'unavailable'
            entry['reasoning'] = next(iter(efforts)) if len(efforts)==1 else 'unavailable'
            entry['association_status'] = 'missing' if missing else 'complete'
            examples[sample['sample_id']] = entry

    overlaps = []
    for gate, identifier in (('G1','G1:R2-02:round1'),
        ('G2','P305_naive_confidence_script_seed0_20261007:18'),
        ('G2','S301_naive_confidence_script_seed0_20261007:13'),
        ('G3','P401_naive_confidence_script_seed0_20261006:20')):
        unsafe = examples[identifier]
        overlaps.append(dict(gate=gate,p_safe=unsafe['p_safe'],unsafe_sample_id=identifier,
            safe_sample_ids=[sample['sample_id'] for sample in dashboard['gates'][gate]['samples']
                if sample['label']=='safe' and sample['eligible_for_score_statistics']
                and sample['p_safe']==unsafe['p_safe']]))

    def locations(paths):
        return dict(references=len(paths), unique_paths=len(set(paths)),
            by_directory={prefix:dict(references=sum(path.startswith(prefix) for path in paths),
                unique_paths=sum(path.startswith(prefix) for path in set(paths)))
                for prefix in ('research/v4/runs/','research/v4/results/')},
            missing_unique_paths=sum(not (ROOT/path).is_file() for path in set(paths)))

    counts = dict(samples=len(examples), by_gate={gate:dict(Counter(
        example['association_status'] for example in examples.values() if example['gate']==gate))
        for gate in ('G1','G2','G3')},
        complete=sum(example['association_status']=='complete' for example in examples.values()),
        missing=sum(example['association_status']=='missing' for example in examples.values()),
        sdk_images=locations(image_refs), original_rgb_companions=locations(raw_refs),
        annotated_sdk_references=sum('raw_path' in image for example in examples.values()
            for image in example['input_images']),
        previous_round_image_references=sum(image['role']=='previous_round'
            for example in examples.values() for image in example['input_images']))
    result_prefixes = sorted({str(Path(*Path(path).parts[:5]))+'/'
        for path in set(image_refs+raw_refs) if path.startswith('research/v4/results/')})
    return dict(offline_only=True, models=0, physical_controls=0,
        scope='Exact saved scoring inputs for the existing dashboard: 65 cited action runs and three G1 '
            'task-score-table families only. No other run, frame replacement or re-render is used.',
        dashboard_source=_relative(DASHBOARD),
        association_method='Original gate record -> unique gate_inputs candidate/public-frame record '
            '(G1 also uses its sample-specific saved image directory and instruction); original score attempt '
            '-> unique reviewer score by thread_id, turn_id, gate, p_safe and comment.',
        image_submission_order='current_images in saved order, selecting annotated_rgb_path when present; '
            'first_workspace_image; each round_history start_keyframe in saved order. Duplicate submissions remain.',
        image_note='raw_path is an unannotated companion, not an extra SDK image. History absent from the '
            'original scoring input remains history_submitted=false; no previous frame is substituted.',
        results_image_directory_prefixes=result_prefixes,
        counts=counts, overlap_examples=overlaps, samples_by_id=examples)


if __name__=='__main__':
    result = enrich()
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(',',':'), allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(OUTPUT), bytes=OUTPUT.stat().st_size,
        counts=result['counts'],results_image_directory_prefixes=result['results_image_directory_prefixes'])))
