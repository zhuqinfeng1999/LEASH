"""Reuse exact mirrored route; only final source IK position stop is8mm."""
import hashlib
import json
from pathlib import Path

from research.v4.candidates.r302_condim6_top_jaw_mirror_saved_copy_20260928 import check as mirror

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
CHANGES = (
    ('        def observed_original_ik(p, R):\n',
     '        def observed_original_ik(p, R, final_source=False):\n'),
    ('                answer = _ik(sim, np.asarray(p), R)\n',
     "                answer = _ik(sim, np.asarray(p), R, **({'position_tol': .008} if final_source else {}))\n"),
    ('            ans, evaluations = observed_original_ik(p, R)\n',
     "            ans, evaluations = observed_original_ik(p, R, final_source=(stage == 'source_closure_open'))\n"),
)


def admit():
    f = json.loads((H / 'source_freeze_before.json').read_text())
    drift = [p for p, s in f['source_sha256'].items() if not (ROOT / p).is_file() or mirror.parent.sha(ROOT / p) != s]
    if drift:
        raise RuntimeError('Exact final-source8mm admission failed: ' + repr(drift))
    return len(f['source_sha256'])


def selected():
    _, _, original = mirror.selected()
    changed = original
    for before, after in CHANGES:
        if changed.count(before) != 1 or after in changed:
            raise RuntimeError('Exact mirrored final-source seam/version mismatch')
        changed = changed.replace(before, after)
    restored = changed
    for before, after in reversed(CHANGES):
        restored = restored.replace(after, before)
    if restored != original:
        raise RuntimeError('Only final source position stop may change')
    ns = dict(mirror.parent.main.__globals__, H=H, admit=admit)
    exec(compile(changed, str(H / 'selected_parent_main.py.txt'), 'exec'), ns)
    if ns['main'].__globals__['_ik'] is not mirror.parent._ik or ns['main'].__globals__['SAVED'] != mirror.parent.SAVED:
        raise RuntimeError('Actual original IK/saved80 binding identity gap')
    return ns['main'], original, changed


def main():
    admit()
    if (H / 'binding_receipt.json').exists():
        raise RuntimeError('Preserve one final-source8mm attempt; no repeat')
    selected_main, original, changed = selected()
    (H / 'selected_parent_main.py.txt').write_text(changed)
    receipt = dict(kind='actual selected mirrored route, source_closure_open8mm only',
        mirrored_function_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        selected_function_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        inverse_source_changes_exact=True, original_ik_function_identity=True,
        only_explicit_position_stop_call_stage='source_closure_open', source_position_tol_m=.008,
        all_other_calls_original_defaults=True, angle_tol_default_unchanged=True,
        same_saved80=str(mirror.parent.SAVED), original_region_m=[.015, .015, .020],
        Scene_constructors=0, controls_GPU_services_models=0, launch_authorized=False)
    (H / 'binding_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    selected_main()


if __name__ == '__main__':
    main()
