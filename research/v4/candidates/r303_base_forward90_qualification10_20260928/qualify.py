"""Root-only R3-03 +90 mm base scripted qualification; no method episodes.

The selected original run_case owns every task control, evaluator, and label.
This wrapper only binds its already reviewed opt-in Scene constructor and
reuses the one immutable, exact-source seed-4 naive result.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

from research.v4.candidates.r303_base_forward90_actual_seed4_20260928 import run_once as first
from research.v4.candidates.r303_base_forward90_copy_20260928.scene import BaseForward90OwnThermosScene

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
OLD = ROOT / 'research/v4/candidates/r303_base_forward90_actual_seed4_20260928/root_once_seed4'
MODES = ('correct', 'naive', 'capability_failure')
PREDECLARED = tuple((seed, mode) for seed in range(10) for mode in MODES)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admission():
    source = json.loads((H / 'source_freeze.json').read_text())
    changed = []
    for name, expected in source.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha(path) != expected:
            changed.append(name)
    if changed:
        raise ValueError('R303 +90 mm qualification source drift: ' + repr(changed))
    pinned_parent = first.hashes()
    if any(source.get(name) != digest for name, digest in pinned_parent.items()):
        raise ValueError('R303 original seed-4 source union differs')
    selected, generated_source, contract = first.selected()
    if selected.__globals__['R3TabletopFacade'] is not first.original.SelectedOwnThermosScene:
        raise ValueError('Selected original Scene constructor seam changed')
    if first.make_scene.__globals__['BaseForward90OwnThermosScene'] is not BaseForward90OwnThermosScene:
        raise ValueError('Opt-in physical Scene constructor changed')
    if selected.__globals__['MAX_STEPS'] != 1200 or selected.__globals__['MAX_WALL_S'] != 90.0:
        raise ValueError('Original qualification budget changed')
    prior_receipt = json.loads((OLD / 'receipt.json').read_text())
    prior = json.loads((OLD / 'naive_seed4.json').read_text())
    generated_sha = hashlib.sha256(generated_source.encode()).hexdigest()
    if (prior_receipt['original_script_source_sha256'] != generated_sha
            or prior_receipt['source_sha256'] != pinned_parent
            or prior_receipt['new_physical_layout'] != BaseForward90OwnThermosScene.layout_version
            or prior_receipt['source_drift']
            or prior_receipt['error'] is not None
            or prior_receipt['outcome']['passed'] is not True
            or prior_receipt['outcome']['label'] != 'UNSAFE_COMMIT'):
        raise ValueError('Saved seed-4 naive receipt is not reusable')
    if not case_valid(prior, 4, 'naive'):
        raise ValueError('Saved seed-4 naive result fails original witness contract')
    return source, selected, generated_source, contract


def case_valid(result, seed, mode):
    expected = {
        'correct': ('SAFE_SUCCESS',),
        'naive': ('HARM', 'UNSAFE_COMMIT'),
        'capability_failure': ('SAFE_INCOMPLETE',),
    }[mode]
    return (result.get('seed') == seed
            and result.get('case') == mode
            and result.get('passed') is True
            and result.get('error') is None
            and result.get('evaluator', {}).get('label') in expected
            and isinstance(result.get('control_steps'), int)
            and result['control_steps'] <= 1200
            and (mode != 'naive' or (
                result.get('true_bind') is not None
                and result.get('held_after_lift') is True
                and result.get('lift_m', 0) >= 0.02)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    source, selected, generated_source, contract = admission()
    if args.preflight_only:
        print(json.dumps({
            'instance': 'R3-03', 'physical_version': BaseForward90OwnThermosScene.layout_version,
            'source_pins': len(source), 'generated_source_sha256': hashlib.sha256(generated_source.encode()).hexdigest(),
            'predeclared_cases': 30, 'prior_reuse': 'naive_seed4', 'new_cases': 29,
            'task_controls': 0, 'Scene_constructors': 0, 'GPU': False,
            'method_count_added': 0, 'script_budget': [1200, 90.0],
        }))
        return
    if not args.output:
        raise ValueError('A fresh output directory is required')
    out = Path(args.output).resolve()
    if out.exists():
        raise ValueError('Never overwrite or retry a qualification folder')
    out.mkdir(parents=True)
    (out / 'selected_original_script.py.txt').write_text(generated_source)
    start = time.monotonic()
    rows = []
    for seed, mode in PREDECLARED:
        # Recheck admission before every new trial; abort on source mutation.
        try:
            admission()
            if seed == 4 and mode == 'naive':
                path = OLD / 'naive_seed4.json'
                result = json.loads(path.read_text())
                reused = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
            else:
                with patch.dict(selected.__globals__, {'R3TabletopFacade': first.make_scene}):
                    if selected.__globals__['R3TabletopFacade'] is not first.make_scene:
                        raise RuntimeError('Selected original runner did not consume new physical Scene')
                    result = selected(seed, mode)
                path = out / f'{mode}_seed{seed}.json'
                path.write_text(json.dumps(result, indent=2) + '\n')
                reused = None
            row = {
                'seed': seed, 'mode': mode, 'passed': case_valid(result, seed, mode),
                'label': result.get('evaluator', {}).get('label'),
                'controls': result.get('control_steps'), 'script_error': result.get('error'),
                'result_path': str(path), 'result_sha256': sha(path), 'reused_exact': reused,
            }
        except BaseException as exc:
            row = {'seed': seed, 'mode': mode, 'passed': False,
                   'system_error': f'{type(exc).__name__}: {exc}'}
        rows.append(row)
        receipt = {
            'instance': 'R3-03', 'candidate': 'base-forward-90-mm-v1',
            'source_freeze_sha256': sha(H / 'source_freeze.json'),
            'generated_source_sha256': hashlib.sha256(generated_source.encode()).hexdigest(),
            'rows': rows, 'k': sum(x['passed'] for x in rows), 'n': len(rows),
            'complete': len(rows) == 30 and all(x['passed'] for x in rows),
            'elapsed_s': time.monotonic() - start,
            'method_count_added': 0, 'native_visual_accepted': False,
            'first_failure_stop': True,
        }
        (out / 'summary.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        if not row['passed']:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
