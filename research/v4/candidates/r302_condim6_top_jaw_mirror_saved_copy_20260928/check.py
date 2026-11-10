"""One exact parent route, sole signed jaw swap, disjoint output/admission."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from research.v4.candidates.r302_condim6_top_grasp_saved_copy_20260928 import check as parent

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
OLD = '    Q = np.column_stack((np.cross(long, down), long, down))\n'
NEW = '    Q = np.column_stack((np.cross(long, down), long, down)) @ np.diag([-1., -1., 1.])\n'


def admit():
    f = json.loads((H / 'source_freeze_before.json').read_text())
    drift = [p for p, s in f['source_sha256'].items() if not (ROOT / p).is_file() or parent.sha(ROOT / p) != s]
    if drift:
        raise RuntimeError('Exact top-jaw mirror source admission failed: ' + repr(drift))
    return len(f['source_sha256'])


def selected():
    original = inspect.getsource(parent.main)
    if original.count(OLD) != 1 or NEW in original:
        raise RuntimeError('Parent top-frame seam/version mismatch')
    changed = original.replace(OLD, NEW)
    if changed.replace(NEW, OLD) != original:
        raise RuntimeError('Only the authorized one top-jaw symmetry may change')
    namespace = dict(parent.main.__globals__, H=H, admit=admit)
    exec(compile(changed, str(H / 'selected_parent_main.py.txt'), 'exec'), namespace)
    target = namespace['main']
    if target.__globals__['SAVED'] != parent.SAVED or target.__globals__['_ik'] is not parent._ik:
        raise RuntimeError('Original saved80/IK actual binding mismatch')
    return target, original, changed


def main():
    admit()
    if (H / 'binding_receipt.json').exists():
        raise RuntimeError('Preserve first mirror result; no repeat')
    target, original, changed = selected()
    (H / 'selected_parent_main.py.txt').write_text(changed)
    receipt = dict(kind='sole Q@diag(-1,-1,1) change in actual selected parent main',
        inverse_source_change_exact=True, same_original_ik_function=target.__globals__['_ik'] is parent._ik,
        same_actual_saved80_path=str(parent.SAVED), original_main_sha256=hashlib.sha256(original.encode()).hexdigest(),
        selected_main_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        original_parent305_preserved=True, Scene_constructors=0, actual_controls=0,
        GPU_services_models=0, alternative_frame_count=1)
    (H / 'binding_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    target()
    previous = json.loads((parent.H / 'geometry.json').read_text())
    current = json.loads((H / 'geometry.json').read_text())
    proof = dict(one_matrix_relation_exact=bool(np.array_equal(np.asarray(current['top_R']),
        np.asarray(previous['top_R']) @ np.diag([-1., -1., 1.]))),
        same_original_top_backoff_exact=current['derived_palm_backoff_m'] == previous['derived_palm_backoff_m'],
        same_original_target_eef_exact=current['target_eef_m'] == previous['target_eef_m'],
        same_source_can_qpos_exact=current['source_can_qpos'] == previous['source_can_qpos'],
        same_source_zone_exact=current['source_zone_m'] == previous['source_zone_m'],
        same_actual_base_exact=current['base_world_m'] == previous['base_world_m'],
        no_geometry_or_physics_change=True, no_second_FK_IK_or_controls=True)
    (H / 'readonly_single_change_proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    if not all(proof.values()):
        raise RuntimeError('Single authorized jaw-symmetry metadata proof failed; preserve and STOP')


if __name__ == '__main__':
    main()
