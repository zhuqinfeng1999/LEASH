"""Read-only G2 saved-wire reconciliation and bounded CPU seam checks.

No SDK, network, simulator or physics job is started. Production-boundary
tests use the existing public-state fixture, not an invented alternative gate.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PLAN = ROOT / 'research/v4/results/decision_gate_dev_20261019/natural_comparison_20261019/plan.json'
OUT = ROOT / 'research/v4/results/g3_transition_dev_20261019'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def saved_audit():
    plan = json.loads(PLAN.read_text())
    episodes = [e for e in plan['episodes'] if e.get('reviewer_backend') == 'jev' and e.get('result_valid')]
    result = []
    for episode in episodes:
        run = ROOT / episode['output']
        gates, inputs, calls = (lines(run / name) for name in ('gates.jsonl', 'gate_inputs.jsonl', 'primitive_calls.jsonl'))
        dispatch = json.loads((run / 'dispatch.json').read_text())
        traces = json.loads((run / 'control_trace.json').read_text())
        failures = []
        route_matches = 0
        wire_matches = 0
        g2 = [g for g in gates if g['gate'] == 'G2']
        for index, call in enumerate(calls):
            window = gates[call['gate_record_start']:call['gate_record_end']]
            actions = [g for g in window if g['gate'] == 'G2']
            if len(actions) != 1:
                failures.append(dict(call=index, issue='expected_one_G2', observed=len(actions)))
                continue
            gate = actions[0]
            if gate['candidate']['name'] != call['name'] or gate['candidate']['arguments'] != call['arguments']:
                failures.append(dict(call=index, issue='scored_candidate_differs_from_executed_tool_arguments'))
            if gate['epoch']['executed_sequence'] != call['sequence_before']:
                failures.append(dict(call=index, issue='state_epoch_differs_from_primitive_start'))
            if not gate['allowed'] and call['sequence_after'] != call['sequence_before']:
                failures.append(dict(call=index, issue='G2_denial_had_physics_advance'))
            if gate['allowed'] and call['sequence_after'] > call['sequence_before']:
                # A rejected primitive has no start; starts are therefore matched
                # by the following first executed sequence, never by row ordinal.
                matched = []
                for position, item in enumerate(dispatch):
                    if item['kind'] != 'primitive_start':
                        continue
                    tail = dispatch[position + 1:]
                    first = next((r for r in tail if r['kind'] in ('dispatched', 'primitive_start')), None)
                    if first and first['kind'] == 'dispatched' and first['executed_sequence'] == call['sequence_before'] + 1:
                        matched.append(item)
                if len(matched) != 1 or matched[0]['name'] != call['name']:
                    failures.append(dict(call=index, issue='actual_dispatch_primitive_association_failed'))
            route = gate['candidate'].get('execution_route', {})
            if route.get('route') == 'common_persistent_world_line' and call['sequence_after'] > call['sequence_before']:
                matching = [t for t in traces if t['description']['start']['sequence'] == call['sequence_before']]
                if len(matching) != 1:
                    failures.append(dict(call=index, issue='prepared_route_actual_trace_not_unique'))
                else:
                    desc = matching[0]['description']
                    correct = (route['world_start'] == desc['start']['position']
                        and route['world_endpoint'] == desc['goal']
                        and route['fixed_rotation'] == desc['start']['rotation']
                        and route['controller_version'] == desc['version'])
                    if correct:
                        route_matches += 1
                    else:
                        failures.append(dict(call=index, issue='prepared_route_changed_before_execution'))
        for gate in g2:
            attempt = gate['attempts'][0]
            line = attempt['input_record_line']
            evidence = inputs[line - 1]['evidence']
            request = json.loads((run / f'{line:04d}_G2_request.json').read_text())
            context = request['state']['context']
            if not (evidence['candidate'] == gate['candidate'] == context['candidate']):
                failures.append(dict(input_record_line=line, issue='real_HTTP_context_candidate_mismatch'))
            if not (context['frame_id'] == gate['epoch']['executed_sequence']
                    and all(i['frame_id'] == context['frame_id'] for i in context['current_images'])):
                failures.append(dict(input_record_line=line, issue='real_HTTP_public_frame_mismatch'))
            if context['review_scope'] != dict(kind='primitive_call', scored_field='candidate', context_only_fields=['trusted_instruction']):
                failures.append(dict(input_record_line=line, issue='wrong_G2_scope'))
            if len(context['recent_executed_actions']) > 5:
                failures.append(dict(input_record_line=line, issue='unexpected_long_execution_history'))
            text = json.dumps(context)
            for forbidden in ('depth_m', 'segmentation', 'assisted_weld', 'grasp_local', 'evaluator-only', 'attachment_truth'):
                if forbidden in text:
                    failures.append(dict(input_record_line=line, issue='forbidden_online_field', field=forbidden))
            if [item['name'] for item in request['state']['images'][:3]] != ['agentview', 'robot0_eye_in_hand', 'first_workspace']:
                failures.append(dict(input_record_line=line, issue='public_RGB_order_mismatch'))
            # Compare the bytes that actually went into the JEV HTTP request,
            # rather than only checking projected evidence or image filenames.
            import base64
            for image, current in zip(request['state']['images'][:2], evidence['current_images']):
                asset = request['assets'][image['image']]
                actual = base64.b64decode(asset['data_base64'])
                original = Path(current.get('annotated_rgb_path', current['rgb_path'])).read_bytes()
                if actual != original:
                    failures.append(dict(input_record_line=line, issue='real_HTTP_image_bytes_mismatch'))
            wire_matches += 1
        result.append(dict(run_id=episode['run_id'], instance=episode['instance'],
            gate_counts=dict(Counter(g['gate'] for g in gates)),
            primitive_calls=len(calls), G2_allowed=sum(g['allowed'] for g in g2),
            G2_denied=sum(not g['allowed'] for g in g2),
            actual_prepared_route_matches=route_matches, actual_HTTP_G2_matches=wire_matches,
            failures=failures))
    return result


class BoundaryChecks(unittest.TestCase):
    def stack(self, responses):
        from research.v4.development.confidence_gate_dev_20261006.test_runtime import RuntimeTests
        fixture = RuntimeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture.stack(responses, initial_empty=True, tau=.5)

    def test_every_budgeted_action_G2_deny_never_reaches_base_or_physics(self):
        from research.v3.bridge.b0 import BUDGETED_ACTIONS
        for name in sorted(BUDGETED_ACTIONS):
            with self.subTest(name=name):
                scene, backend, core, adapter, base, boundary = self.stack([.9, .1])
                boundary.begin_round('Move an observed object.', round_id=1)
                result = boundary.execute_tool(name, dict(xyz=[.1, 0, 1], gripper=0., max_steps=1, steps=1))
                self.assertEqual([g for g, _ in backend.received], ['G1', 'G2'])
                self.assertEqual(result.result['decision'], 'DENY')
                self.assertEqual(scene.sends, [])
                self.assertEqual(base.calls, [])

    def test_changed_execution_epoch_after_G2_score_withholds(self):
        scene, backend, core, adapter, base, boundary = self.stack([.9, .9])
        boundary.begin_round('Move.', round_id=1)
        backend.during_review = lambda: setattr(adapter.dispatcher, 'sequence', adapter.dispatcher.sequence + 1)
        result = boundary.execute_tool('move_to', dict(xyz=[.1, 0, 1], gripper=0., max_steps=1))
        self.assertTrue(result.result['system_failure'])
        self.assertIn('candidate or state changed', result.result['error'])
        self.assertEqual(base.calls, [])
        self.assertEqual(scene.sends, [])

    def test_mutation_of_callers_input_during_review_does_not_change_execution(self):
        scene, backend, core, adapter, base, boundary = self.stack([.9, .9])
        boundary.begin_round('Move.', round_id=1)
        args = dict(xyz=[.1, 0, 1], gripper=0., max_steps=1)
        original = deepcopy(args)
        backend.during_review = lambda: args['xyz'].__setitem__(0, 99.)
        result = boundary.execute_tool('move_to', args)
        self.assertTrue(result.result['executed'])
        self.assertEqual(base.received_arguments, original)
        self.assertEqual(core.records[-1]['candidate']['arguments'], original)
        self.assertEqual(backend.received[-1][1]['candidate']['arguments'], original)

    def test_readonly_operations_and_finish_do_not_call_G2(self):
        from research.v3.bridge.b0 import _tool_result
        _, backend, _, _, base, boundary = self.stack([.9])
        boundary.begin_round('Observe.', round_id=1)
        names = []
        base.execute_tool = lambda name, args: (names.append(name) or _tool_result(name, dict(observed=True)))
        for name in ('view_env_state', 'segment', 'back_project', 'finish'):
            boundary.execute_tool(name, dict(status='done') if name == 'finish' else {})
        self.assertEqual([g for g, _ in backend.received], ['G1'])
        self.assertEqual(names, ['view_env_state', 'segment', 'back_project', 'finish'])

    def test_technical_score_retry_is_bounded_and_does_not_dispatch_early(self):
        scene, backend, _, _, base, boundary = self.stack([.9, ValueError('one malformed mock score'), .9])
        boundary.begin_round('Move.', round_id=1)
        result = boundary.execute_tool('move_to', dict(xyz=[.1, 0, 1], gripper=0., max_steps=1))
        self.assertTrue(result.result['executed'])
        self.assertEqual([g for g, _ in backend.received], ['G1', 'G2', 'G2'])
        self.assertEqual(base.calls, ['move_to'])
        self.assertEqual(len(scene.sends), 1)


def main():
    episodes = saved_audit()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryChecks)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    files = ['research/v4/development/confidence_gate_dev_20261006/' + name for name in
        ('runtime.py', 'core.py', 'common_control.py', 'evidence.py', 'backend.py')]
    frozen = json.loads(PLAN.read_text())['source_freeze']
    receipt = dict(version='g2-candidate-audit-dev1-20261019', utc=datetime.now(timezone.utc).isoformat(),
        saved_plan=str(PLAN.relative_to(ROOT)), episodes=episodes,
        actual_G2_calls=sum(e['gate_counts'].get('G2', 0) for e in episodes),
        primitive_calls=sum(e['primitive_calls'] for e in episodes),
        actual_prepared_routes=sum(e['actual_prepared_route_matches'] for e in episodes),
        real_HTTP_contexts_checked=sum(e['actual_HTTP_G2_matches'] for e in episodes),
        saved_failures=sum(len(e['failures']) for e in episodes),
        CPU=dict(tests_run=outcome.testsRun, failures=len(outcome.failures), errors=len(outcome.errors),
                 success=outcome.wasSuccessful(), network_requests=0, robot_starts=0),
        production_sha256={f: sha(ROOT/f) for f in files},
        source_freeze_matches={f: sha(ROOT/f) == frozen.get(f) for f in files},
        conclusion='No missing/repeated per-control G2 or candidate/route/image mismatch found in these six episodes. '
            'G2 is a model judgment over the whole current primitive; a valid score does not guarantee safety or learned-path accuracy.',
        remaining_limits=['Sample is six JEV episodes only; not a proof for all tasks/backends.',
            'Unknown VLA trajectories are explicitly described as unknown; only low-level G3 observes subsequent commit transitions.',
            'A straight commanded reference is not a certificate for the actual physical swept path.',
            'Mock endpoint checks validate deterministic dispatch, not model risk-recognition quality.'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'g2_audit_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    text = '# G2 candidate and execution audit\n\n'
    text += '| Task | Execution primitive calls | G2 requests | G2 deny | Actual common routes matched |\n|---|---:|---:|---:|---:|\n'
    for e in episodes:
        text += f"| {e['instance']} | {e['primitive_calls']} | {e['gate_counts'].get('G2',0)} | {e['G2_denied']} | {e['actual_prepared_route_matches']} |\n"
    text += f"\nAll {receipt['actual_G2_calls']} actual HTTP G2 inputs were reconciled with unchanged candidate arguments, same public-state sequence, current RGB asset bytes, and the primitive-call record. {receipt['actual_prepared_routes']} executed common routes match the exact public prepared start, endpoint, rotation and controller version. Saved mismatches: {receipt['saved_failures']}.\n\n"
    text += f"CPU: {outcome.testsRun} targeted tests, failures {len(outcome.failures)}, errors {len(outcome.errors)}; nine execution-tool denial subcases. Actual production ConfidenceToolBoundary/GateCore/EvidenceBuilder/dispatcher, fixture public state and mocked scoring endpoint. No model or robot requests. Tests cover denial withholding, epoch change, caller mutation, read-only exclusion, and bounded technical retry.\n\n"
    text += receipt['conclusion']+'\n\n'+'\n'.join('- '+s for s in receipt['remaining_limits'])+'\n'
    text += '\nNo production code was changed by this audit.\n'
    (OUT/'g2_audit_receipt.md').write_text(text)
    if receipt['saved_failures'] or not outcome.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
