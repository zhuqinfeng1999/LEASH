"""Offline complete-attempt G3 development accounting, without model calls."""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / 'research/v4/results/g3_transition_dev_20261019'
HISTORICAL = {
    'P2-01': 'V4D2652_P201_OursJev_seed0_gpt_6_astra_recording_repair2',
    'P3-05': 'V4D2647_P305_OursJev_seed0_gpt_6_astra',
    'S3-01': 'V4D2651_S301_OursJev_seed0_gpt_6_astra',
}


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def offline_review():
    folder = HERE / 'fixed_negative_review'
    result = load(folder / 'score_result.json')
    if result is None:
        return load(folder / 'approval_block_receipt.json')
    return dict(result,
        user_authorization=load(folder / 'user_authorization_receipt.json'),
        historical_approval_rejection=load(folder / 'approval_block_receipt.json'),
        actual_sdk_wire_assertions=lines(folder / 'actual_wire_assertions.jsonl'),
        accounting='One additional offline review, separate from natural episodes; no robot dispatch.')


def completion_stage(item):
    if item.get('complete'):
        return 'target_release_and_supported_rest'
    if item.get('detached_in_target'):
        return 'released_in_target_without_complete_rest'
    if item.get('detached_at_s') is not None:
        return 'detached_outside_target_or_incomplete'
    if item.get('lifted_at_s') is not None:
        return 'actually_lifted'
    if item.get('attached_at_s') is not None:
        return 'actually_grasped'
    return 'no_verified_grasp'


def metrics(row, historical=False):
    out = Path(row['output'])
    ep = load(out / 'episode.json', {})
    manifest = load(out / 'run_manifest.json', {})
    gates = lines(out / 'gates.jsonl')
    reviews = lines(out / 'reviewer_scores.jsonl')
    http_submitted = [r for r in reviews if r.get('event') == 'score_http_submitted']
    sdk_submitted = [r for r in reviews if r.get('event') == 'score_turn_started']
    submitted = http_submitted + sdk_submitted
    responses = [r for r in reviews if r.get('event') == 'score']
    requests = {g: sum(r.get('gate') == g for r in submitted) for g in ('G1', 'G2', 'G3')}
    waits = {g: sum(r.get('latency_s', 0) for r in responses if r.get('gate') == g) for g in ('G1', 'G2', 'G3')}
    denials = Counter(g['gate'] for g in gates if not g.get('allowed') and not g.get('system_failure'))
    dispatch = load(out / 'dispatch.json', [])
    physical = ep.get('physical') or load(out / 'final-result.json', {})
    stages = {}
    for key in ('completion_evidence', 'plate_completion', 'ketchup_completion'):
        if isinstance(physical.get(key), dict):
            stages[key] = dict(stage=completion_stage(physical[key]), evidence=physical[key])
    if not stages:
        stages['physical'] = dict(stage='not_saved', evidence=None)
    terminal = ep.get('terminal') or {}
    denied = next((g for g in gates if not g.get('allowed')), None)
    stop = ep.get('error') or ep.get('system_failure') or ep.get('budget_exhausted')
    if stop is None and terminal.get('status') == 'refused':
        stop = 'planner_refused: ' + str(terminal.get('message'))
    if stop is None:
        stop = physical.get('reason') or ep.get('termination_kind')
    videos = [dict(view=p.stem, path=str(p.relative_to(ROOT))) for p in sorted((out / 'videos').glob('*.mp4'))]
    controls = [r for r in dispatch if r.get('kind') == 'dispatched']
    transition_windows = load(out / 'transition_windows.json', [])
    reuse_sequences = {(r['primitive'], r['sequence']) for r in transition_windows
                       if r.get('kind') == 'reviewed_window_reuse'}
    primitive_records = lines(out / 'primitive_calls.jsonl')
    g2_associations = []
    for call in primitive_records:
        associated = gates[call['gate_record_start']:call['gate_record_end']]
        selected = [g for g in associated if g['gate'] == 'G2']
        g2_associations.append(len(selected) == 1
            and selected[0]['epoch']['executed_sequence'] == call['sequence_before']
            and selected[0]['candidate']['name'] == call['name'])
    def identifier(record):
        return record.get('request_local_id') or record.get('turn_id')
    # Raw outcomes remain unchanged. A budget-limited model attempt can be a
    # valid result; absent manifest/technical failure cannot become one.
    valid = manifest.get('valid') is True and not ep.get('system_failure') and not ep.get('error')
    return dict(run_id=row['run_id'], instance=row['instance'], historical_reference=historical,
        status=row.get('status'), valid=valid, label=ep.get('label'), raw_physical_label=physical.get('label'),
        terminal=terminal.get('status'), terminal_message=terminal.get('message'),
        process_wall_s=None if historical else row.get('process_wall_s'), episode_wall_s=ep.get('wall_s'),
        G1_calls=requests['G1'], G2_calls=requests['G2'], G3_calls=requests['G3'],
        reviewer_backend='jev' if historical else 'luna',
        actual_scoring_submissions=len(submitted), completed_scores=len(responses),
        actual_HTTP_submissions=len(http_submitted) if http_submitted else None,
        actual_SDK_turn_submissions=len(sdk_submitted) if sdk_submitted else None,
        unfinished_score_response_count=len({identifier(r) for r in submitted}
            - {identifier(r) for r in responses}),
        G1_wait_s=waits['G1'], G2_wait_s=waits['G2'], G3_wait_s=waits['G3'],
        total_observed_review_wait_s=sum(waits.values()), gate_denials=dict(denials),
        gate_record_counts=dict(Counter(g['gate'] for g in gates)),
        primitive_calls=len(primitive_records), executed_controls=len(controls),
        G2_primitive_association_checks=len(g2_associations),
        G2_primitive_association_passed=all(g2_associations),
        reused_permit_control_count=sum(bool(r.get('used_permits')) and r.get('match') == 'NO_MATCH' for r in controls),
        transition_window_counts=dict(Counter(r.get('kind') for r in transition_windows)),
        reviewed_window_reused_controls=len(reuse_sequences),
        reviewed_window_sources=dict(Counter(r['window']['source'] for r in transition_windows
            if r.get('kind') == 'reviewed_window_granted')),
        phase_wall_s_inclusive={phase:sum(r.get('wall_s', 0) for r in lines(out / 'phase_timings.jsonl')
            if r.get('phase') == phase) for phase in sorted({r.get('phase')
                for r in lines(out / 'phase_timings.jsonl')})},
        G3_requests_per_100_controls=100 * requests['G3'] / len(controls) if controls else None,
        physical_stages=stages, stop_cause=stop,
        first_denial=dict(gate=denied['gate'], comment=denied.get('comment'),
            sequence=(denied.get('epoch') or {}).get('executed_sequence'), p_safe=denied.get('p_safe')) if denied else None,
        output_bytes=ep.get('output_bytes') or sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        estimated_reviewer_cost_usd=sum(r['estimated_cost_usd'] for r in responses if isinstance(r.get('estimated_cost_usd'), (int, float))) if historical else None,
        videos=videos, output=str(out.relative_to(ROOT)), episode_path=str((out / 'episode.json').relative_to(ROOT)))


def summarize():
    plan = load(HERE / 'plan.json')
    rows = [metrics(r) for r in plan['episodes'] if r['status'] not in ('registered', 'starting')]
    valid_rows = [r for r in rows if r['valid']]
    excluded_rows = [r for r in rows if not r['valid']]
    history = [metrics(dict(run_id=rid, instance=task,
        output=str(ROOT / 'research/v4/runs/confidence_gate_dev_20261006' / rid), status='historical'), True)
        for task, rid in HISTORICAL.items()]
    comparison = []
    for row in valid_rows:
        previous = next(h for h in history if h['instance'] == row['instance'])
        comparison.append(dict(instance=row['instance'], historical_run=previous['run_id'], new_run=row['run_id'],
            old_G3_calls=previous['G3_calls'], new_G3_calls=row['G3_calls'],
            call_reduction_percent=100 * (1 - row['G3_calls'] / previous['G3_calls']) if previous['G3_calls'] else None,
            old_label=previous['label'], new_label=row['label'],
            old_episode_wall_s=previous['episode_wall_s'], new_episode_wall_s=row['episode_wall_s'],
            causal_matched_pair=False))
    data = dict(version=plan['version'], updated_utc=datetime.now(timezone.utc).isoformat(),
        planned=3, ended=len({r['instance'] for r in valid_rows}),
        attempts_registered=len(plan['episodes']), attempts_ended=len(rows),
        valid_natural_attempts=len(valid_rows), excluded_engineering_attempts=len(excluded_rows),
        active=[dict(run_id=r['run_id'], pid=r.get('pid')) for r in plan['episodes'] if r['status'] == 'starting' and r.get('pid')],
        formal_paused=True, historical_reference_only=True, rows=rows, historical=history, comparison=comparison,
        conditional_grasp_accounting=load(HERE / 'capability_review.json'),
        additional_offline_review=offline_review(),
        prepared_boundary_cpu=load(HERE / 'prepared_boundary_cpu_receipt.json'),
        window_provenance_audit=load(HERE / 'window_provenance_audit.json'),
        time_scope='JEV HTTP waits and Luna reviewer-call waits are inclusive observed durations already inside process/episode wall. '
            'No pure network/remote queue/inference split is available. Historical full process wall was not recorded. '
            'Different reviewer backends prevent attribution of latency differences to the G3 scope change.',
        conclusion_limit='Different natural planner choices mean before/after call counts and durations are descriptive; '
            'failure, early refusal and incomplete tasks are never successful completion speedup.')
    (HERE / 'summary.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    flat = []
    for r in history + rows:
        entry = {k: r.get(k) for k in ('run_id', 'instance', 'historical_reference', 'status', 'valid', 'label',
            'raw_physical_label', 'terminal', 'process_wall_s', 'episode_wall_s', 'G1_calls', 'G2_calls', 'G3_calls',
            'G1_wait_s', 'G2_wait_s', 'G3_wait_s', 'actual_scoring_submissions', 'completed_scores',
            'actual_HTTP_submissions', 'actual_SDK_turn_submissions', 'unfinished_score_response_count',
            'primitive_calls', 'executed_controls', 'G3_requests_per_100_controls',
            'output_bytes', 'estimated_reviewer_cost_usd', 'output')}
        entry['stop_cause'] = json.dumps(r['stop_cause'], ensure_ascii=False)
        entry['physical_stages'] = json.dumps(r['physical_stages'], ensure_ascii=False)
        flat.append(entry)
    with (HERE / 'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    text = ['# G3 bounded-transition development', '',
        f"{data['ended']}/3 fixed new seed0 natural checks ended; {len(rows)} total starts, including {len(excluded_rows)} excluded engineering startup. Astra/high + Luna/high, tau=.5/.5/.5. No paid backend tests. Formal remains paused.", '',
        '| Task | Version / reviewer | Outcome | G1/G2/G3 requests | G3 observed wait s | Episode wall s | Full process wall s |',
        '|---|---|---|---|---:|---:|---:|']
    for r in history + valid_rows:
        text.append('| ' + ' | '.join(map(str, [r['instance'], ('historical / JEV' if r['historical_reference'] else 'new / Luna high'),
            r['label'], f"{r['G1_calls']}/{r['G2_calls']}/{r['G3_calls']}", round(r['G3_wait_s'], 3),
            round(r['episode_wall_s'], 3) if r['episode_wall_s'] is not None else 'unavailable',
            round(r['process_wall_s'], 3) if r['process_wall_s'] is not None else 'unavailable'])) + ' |')
    text += ['', data['conclusion_limit'], '', data['time_scope'], '', '## Actual physical stages and stopping causes', '']
    for r in rows:
        text += [f"- {r['run_id']}: " + (r['label'] if r['valid'] else 'EXCLUDED_ENGINEERING_STARTUP (raw label retained: '+str(r['label'])+')') + '; stages ' + ', '.join(k + '=' + v['stage'] for k, v in r['physical_stages'].items())
                 + '; stop: ' + str(r['stop_cause']) + f"; output {r['output_bytes']} bytes."]
        text += [f"  - [{v['view']}](/" + str(ROOT / v['path']).lstrip('/') + ')' for v in r['videos']]
    text += ['', '## Observed finite-window reuse', '',
        '| Task | Granted windows | Reused controls (not event count) | Source types | G2 / primitive associations |',
        '|---|---:|---:|---|---|']
    for r in valid_rows:
        text.append(f"| {r['instance']} | {r['transition_window_counts'].get('reviewed_window_granted',0)} | {r['reviewed_window_reused_controls']} | {r['reviewed_window_sources']} | {r['G2_primitive_association_checks']} / {r['G2_primitive_association_passed']} |")
    text += ['', 'P2-01 does not exercise the expanded continuation. P3-05 does not establish a grasp. '
        'S3-01 physically completes both transfers and exercises generated chunks plus prepared motion. '
        'None of these static checks qualifies unobserved dynamic intrusions.', '',
        '## Additional boundary validation', '']
    extra = data['additional_offline_review'] or {}
    if extra.get('status') == 'scored':
        answer = extra['answer']
        text += [f"One fixed historical P2-06 public packet was reviewed by Luna/high after explicit user authorization: "
            f"p_safe={answer['p_safe']}, {extra['decision']} at tau={extra['threshold']}; "
            f"{answer['latency_s']:.3f}s, {extra['model_requests_attempted']} request, 0 robot controls. "
            'Actual SDK serialized text, reviewed-window hash and all three RGB byte hashes were verified. '
            'The prior automatic rejection remains historical evidence, not a pending permission request.', '',
            '> ' + answer['comment'], '',
            'This is an UNKNOWN possible-release/person-exposure case, not confirmed HELD, a proven '
            'counterfactual HARM, broad unsafe-recall validation, or a natural method result.', '']
    elif extra:
        text += ['The extra fixed old P2-06 review has not produced a real model result; see its retained receipt.', '']
    cpu = data['prepared_boundary_cpu'] or {}
    if cpu:
        text += [f"Prepared-motion boundary CPU integration: {cpu['tests']} tests, exit={cpu['exit_code']}, "
            f"{cpu['elapsed_s']:.3f}s. Real common numeric executor and original CommitDispatcher, with fake "
            'public-state/send/reviewer callbacks: valid finite continuation executes; initial BLOCK sends zero; '
            'out-of-scope BLOCK and late PASS withhold the next candidate and remaining tail. '
            'An initial fixture-only exact-start mismatch is retained separately; runtime equality checks were not relaxed.', '']
    provenance = data['window_provenance_audit'] or {}
    if provenance:
        text += [f"Saved-log provenance audit: {provenance['granted_windows']} windows and "
            f"{provenance['distinct_reused_controls']} distinct reused controls; "
            f"all_checks_passed={provenance['all_checks_passed']}. Each grant links to the actual PASS/input packet; "
            'each reused event links to that grant and dispatched payload within its context/sequence/phase. '
            'This log join does not independently reconstruct every intermediate public pose.', '']
    text += ['No raw historical outcome, score, image or failure is overwritten. '
        'New results remain development only; production and formal experiments remain paused.']
    (HERE / 'report.md').write_text('\n'.join(text) + '\n')
    status_path = ROOT / 'research/v4/results/status.json'
    status = load(status_path, {})
    q = status.setdefault('research_questions', {})
    info = q.setdefault('g3_transition_development', {})
    info.update(summary=data, planned=3, ended=data['ended'], attempts_ended=len(rows), active=data['active'], updated_utc=data['updated_utc'])
    item = next((i for i in q.get('items', []) if i.get('id') == 'g3-transition-development'), None)
    if item is not None:
        base_links = [link for link in item.get('links', []) if not link.get('label', '').startswith('新回放 ')]
        item['links'] = base_links + [dict(label='新回放 ' + r['instance'] + ' ' + v['view'], path=v['path'])
            for r in valid_rows for v in r['videos']]
        item['state'] = f"{data['ended']}/3 已结束；{len(excluded_rows)} 条独立工程启动失败保留；" + ('实际进程运行中' if data['active'] else '当前无活动机器人')
    temporary = status_path.with_suffix('.g3report.tmp')
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(status_path)
    return data


if __name__ == '__main__':
    summarize()
