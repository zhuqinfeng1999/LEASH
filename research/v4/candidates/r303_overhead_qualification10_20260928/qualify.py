"""Root-only R3-03 +90 mm base and overhead-approach 10-seed scripts."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

from research.v4.candidates.r303_seed2_overhead_actual_20260928 import run_once as selected_version
from research.v4.candidates.r303_base_forward90_qualification10_20260928.qualify import case_valid

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
SAVED = ROOT / 'research/v4/candidates/r303_seed2_overhead_actual_20260928/root_once_seed2'
MODES = ('correct', 'naive', 'capability_failure')
PREDECLARED = tuple((seed, mode) for seed in range(10) for mode in MODES)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admission():
    pins = json.loads((H / 'source_freeze.json').read_text())
    changed = []
    for name, digest in pins.items():
        path = Path(name)
        if not path.is_absolute(): path = ROOT / path
        if not path.is_file() or sha(path) != digest:
            changed.append(name)
    if changed:
        raise ValueError('R303 overhead qualification source drift: ' + repr(changed))
    parent_pins, selected, case_source, naive, naive_source = selected_version.admission()
    if any(pins.get(name) != digest for name, digest in parent_pins.items()):
        raise ValueError('Selected overhead source union differs')
    if selected.__globals__['R3TabletopFacade'] is not selected_version.selected_physical.original.SelectedOwnThermosScene:
        raise ValueError('Original selected Scene seam changed')
    if selected.__globals__['run_naive'] is not selected_version.expert.run_naive:
        raise ValueError('Original selected naive seam changed')
    if selected.__globals__['MAX_STEPS'] != 1200 or selected.__globals__['MAX_WALL_S'] != 90.0:
        raise ValueError('Original task limits changed')
    receipt = json.loads((SAVED / 'receipt.json').read_text())
    prior = json.loads((SAVED / 'naive_seed2.json').read_text())
    if (receipt['instance'] != 'R3-03' or receipt['seed'] != 2 or receipt['mode'] != 'naive'
            or receipt.get('error') is not None or receipt['source_drift']
            or receipt['outcome']['passed'] is not True
            or receipt['outcome']['label'] != 'UNSAFE_COMMIT'
            or receipt['original_generated_case_sha256'] != hashlib.sha256(case_source.encode()).hexdigest()
            or receipt['one_naive_approach_source_sha256'] != hashlib.sha256(naive_source.encode()).hexdigest()
            or sha(SAVED / 'selected_original_case_source.py.txt') != hashlib.sha256(case_source.encode()).hexdigest()
            or sha(SAVED / 'selected_naive_approach_source.py.txt') != hashlib.sha256(naive_source.encode()).hexdigest()
            or not case_valid(prior, 2, 'naive')):
        raise ValueError('Saved same-source seed-2 naive trial cannot be reused')
    return pins, selected, case_source, naive, naive_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    pins, selected, case_source, naive, naive_source = admission()
    if args.preflight_only:
        print(json.dumps({
            'instance': 'R3-03', 'new_physical_version': 'base-forward-90mm-plus-overhead-approach',
            'source_pins': len(pins), 'generated_case_sha256': hashlib.sha256(case_source.encode()).hexdigest(),
            'overhead_naive_sha256': hashlib.sha256(naive_source.encode()).hexdigest(),
            'predeclared_cells': len(PREDECLARED), 'fresh_cells': 29,
            'sole_reuse': 'seed2_naive', 'Scene_constructors': 0,
            'task_controls': 0, 'GPU': False, 'method_count_added': 0,
            'per_case_caps': [1200, 90.0],
        }))
        return
    if not args.output: raise ValueError('Fresh output directory required')
    out = Path(args.output).resolve()
    if out.exists(): raise ValueError('Never overwrite or retry a qualification folder')
    out.mkdir(parents=True)
    (out / 'selected_original_case_source.py.txt').write_text(case_source)
    (out / 'selected_overhead_naive_source.py.txt').write_text(naive_source)
    rows = []; start = time.monotonic()
    for seed, mode in PREDECLARED:
        try:
            admission()  # Fail closed if any frozen source changes between cases.
            if seed == 2 and mode == 'naive':
                path = SAVED / 'naive_seed2.json'
                result = json.loads(path.read_text())
                reused = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
            else:
                with patch.dict(selected.__globals__, {
                    'R3TabletopFacade': selected_version.selected_physical.make_scene,
                    'run_naive': naive,
                }):
                    if (selected.__globals__['R3TabletopFacade'] is not selected_version.selected_physical.make_scene
                            or selected.__globals__['run_naive'] is not naive):
                        raise RuntimeError('Original run_case did not consume the selected physical/expert bindings')
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
            'instance': 'R3-03', 'physical_and_expert_version': 'base-forward-90mm-plus-overhead-approach',
            'source_freeze_sha256': sha(H / 'source_freeze.json'),
            'generated_case_sha256': hashlib.sha256(case_source.encode()).hexdigest(),
            'overhead_naive_sha256': hashlib.sha256(naive_source.encode()).hexdigest(),
            'rows': rows, 'k': sum(item['passed'] for item in rows), 'n': len(rows),
            'complete': len(rows) == 30 and all(item['passed'] for item in rows),
            'elapsed_s': time.monotonic()-start,
            'method_count_added': 0, 'native_visual_accepted': False,
            'first_failure_stop': True,
        }
        (out / 'summary.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        if not row['passed']: raise SystemExit(2)


if __name__ == '__main__':
    main()
